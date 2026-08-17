# cgp_optim.py — Composite Gaussian Process (CGP) hyperparameter optimization
#
# Fits the CGP composite kernel (Ba & Joseph, 2012) via:
#   1. LHD screening: 500+ Latin Hypercube candidates → top-K for optimization
#   2. Constrained parameterization: ell_l ≤ ell_g via softplus(delta) offset
#   3. IRLS inner loop: 4-iteration residual-based estimation of v(x) params
#   4. Box constraints: projected gradient L-BFGS (clamping in closure)
#
# Entry point: optimize_cgp(). Kernel primitives come from estimation.py.

import logging
import math
import torch

from .estimation import safe_cholesky, cgp_cov, nll_concentrated

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Latin Hypercube Design candidate generation
# ---------------------------------------------------------------------------

def _lhd_candidates(n_params, n_cand=500, bounds=None, device=None):
    """
    Generate Latin Hypercube Design candidates in [0, 1]^n_params,
    then scale to bounds.

    Parameters
    ----------
    n_params : int
        Number of parameters.
    n_cand : int
        Number of LHD candidates to generate.
    bounds : list of (lo, hi) tuples, length n_params
        Box constraints for each parameter.
    device : torch.device

    Returns
    -------
    candidates : (n_cand, n_params) tensor
    """
    if device is None:
        device = torch.device('cpu')

    # Stratified sampling: divide [0,1] into n_cand strata per dimension
    candidates = torch.zeros(n_cand, n_params, device=device, dtype=torch.float32)
    for j in range(n_params):
        perm = torch.randperm(n_cand, device=device)
        # Uniform within each stratum
        u = (perm.float() + torch.rand(n_cand, device=device)) / n_cand
        candidates[:, j] = u

    # Scale to bounds
    if bounds is not None:
        for j, (lo, hi) in enumerate(bounds):
            candidates[:, j] = lo + (hi - lo) * candidates[:, j]

    return candidates


# ---------------------------------------------------------------------------
# 2. Constrained parameterization (R-style: alpha = theta + kappa)
# ---------------------------------------------------------------------------

def _get_param_bounds(d_input, log_ell_min, log_ell_max=6.0):
    """
    Return box-constraint bounds for the R-style CGP parameterization.

    Parameter order: [log_ell(d), log_delta(d), log_sigma_f, log_sigma_n, logit_lambda]
    Total: 2*d + 3

    log_delta is the raw parameter for the local-global lengthscale offset:
        log_ell_l = log_ell - softplus(log_delta)
    This ensures ell_l ≤ ell_g structurally.
    """
    bounds = []
    # log_ell: global lengthscales
    for _ in range(d_input):
        bounds.append((log_ell_min, log_ell_max))
    # log_delta: softplus(log_delta) = gap between log_ell and log_ell_l
    # softplus(log_delta) ∈ [~0, ~5] → ell_l / ell_g ∈ [exp(-5), 1] ≈ [0.007, 1]
    for _ in range(d_input):
        bounds.append((-3.0, 5.0))
    # log_sigma_f
    bounds.append((-10.0, 6.0))
    # log_sigma_n
    bounds.append((-6.0, 6.0))
    # logit_lambda
    bounds.append((-5.0, 5.0))

    return bounds


def _raw_to_theta(raw, d_input, device=None):
    """
    Convert raw parameter vector to theta dict compatible with make_kernel.

    Raw order: [log_ell(d), log_delta(d), log_sigma_f, log_sigma_n, logit_lambda]

    The constraint ell_l ≤ ell_g is enforced via:
        log_ell_l = log_ell - softplus(log_delta)
    where softplus(x) = log(1 + exp(x)) ≥ 0.
    """
    if device is None:
        device = raw.device

    log_ell = raw[:d_input]
    log_delta = raw[d_input:2*d_input]
    log_sigma_f = raw[2*d_input]
    log_sigma_n = raw[2*d_input + 1]
    logit_lambda = raw[2*d_input + 2]

    # Constrained local lengthscale: ell_l ≤ ell_g
    softplus_delta = torch.nn.functional.softplus(log_delta)
    log_ell_l = log_ell - softplus_delta

    theta = {
        'log_ell': log_ell,
        'log_ell_l': log_ell_l,
        'log_sigma_f': log_sigma_f.unsqueeze(0) if log_sigma_f.dim() == 0 else log_sigma_f,
        'log_sigma_n': log_sigma_n.unsqueeze(0) if log_sigma_n.dim() == 0 else log_sigma_n,
        'logit_lambda': logit_lambda.unsqueeze(0) if logit_lambda.dim() == 0 else logit_lambda,
    }
    return theta


def _theta_to_raw(theta, d_input):
    """
    Convert theta dict back to raw parameter vector.
    Inverse of _raw_to_theta.
    """
    log_ell = theta['log_ell'].detach()
    log_ell_l = theta['log_ell_l'].detach()

    # Invert: softplus(log_delta) = log_ell - log_ell_l
    gap = (log_ell - log_ell_l).clamp(min=1e-6)
    # Inverse softplus: log_delta = log(exp(gap) - 1)
    log_delta = torch.log(torch.expm1(gap).clamp(min=1e-6))

    log_sf = theta['log_sigma_f'].detach()
    log_sn = theta['log_sigma_n'].detach()
    logit_lam = theta['logit_lambda'].detach()

    # Flatten scalars
    if log_sf.dim() > 0:
        log_sf = log_sf.squeeze()
    if log_sn.dim() > 0:
        log_sn = log_sn.squeeze()
    if logit_lam.dim() > 0:
        logit_lam = logit_lam.squeeze()

    raw = torch.cat([log_ell, log_delta, log_sf.unsqueeze(0),
                     log_sn.unsqueeze(0), logit_lam.unsqueeze(0)])
    return raw


# ---------------------------------------------------------------------------
# 3. IRLS estimation of v(x) = exp(v_a + v_b^T x) from GP residuals
# ---------------------------------------------------------------------------

def _irls_estimate_v(tau, y, L, n_iter=4):
    """
    Estimate spatially-varying variance v(x) = exp(v_a + v_b^T x)
    from GP residuals using Iteratively Reweighted Least Squares.

    Following R's CGP package: uses local bandwidth kernel for
    smoothed variance estimation.

    Parameters
    ----------
    tau : (n, d) input locations
    y : (n, r) PCA-reduced responses
    L : (n, n) Cholesky factor of K_theta
    n_iter : int
        Number of IRLS iterations (R uses 4)

    Returns
    -------
    v_a : scalar tensor
    v_b : (d,) tensor
    """
    n, r = y.shape
    d = tau.shape[1]
    device = tau.device

    # 1. Compute GP residuals: e = y - K @ alpha = y - K @ K^{-1} y
    #    But for the IRLS we want LOO residuals or cross-validation residuals.
    #    Simpler approach following R: use the prediction residuals
    #    e_i = y_i - mu_{-i}(x_i), approximated by LOO.
    alpha = torch.cholesky_solve(y, L)  # (n, r)
    I = torch.eye(n, device=device, dtype=tau.dtype)
    Linv = torch.linalg.solve_triangular(L, I, upper=False)
    Kinv_diag = torch.sum(Linv**2, dim=1)  # (n,)
    # LOO residuals: e_i = alpha_i / K^{-1}_{ii}
    loo_resid = alpha / Kinv_diag.unsqueeze(1)  # (n, r)

    # 2. Per-point variance estimate: ||e_i||^2 / r
    e_var = (loo_resid ** 2).mean(dim=1).clamp(min=1e-12)  # (n,)

    # 3. Bandwidth kernel for local smoothing
    with torch.no_grad():
        dists = torch.cdist(tau, tau)
        bw = float(dists.median()) * 0.5  # half-median bandwidth
        G_bw = torch.exp(-dists ** 2 / (2 * bw ** 2))  # (n, n) Gaussian kernel
        G_bw_sum = G_bw.sum(dim=1, keepdim=True).clamp(min=1e-8)  # (n, 1)

    # 4. IRLS iterations
    # Design matrix: [1, x_1, ..., x_d]
    X_design = torch.cat([torch.ones(n, 1, device=device, dtype=tau.dtype), tau], dim=1)
    current_resid = loo_resid  # will be re-weighted each iteration

    for _ in range(n_iter):
        # Smoothed variance estimate via bandwidth kernel
        # Sig[i] = sum_j G_bw[i,j] * e^2[j] / sum_j G_bw[i,j]
        smoothed_var = (G_bw @ (current_resid ** 2)) / G_bw_sum  # (n, r)
        smoothed_var_mean = smoothed_var.mean(dim=1).clamp(min=1e-12)  # (n,)
        log_e_var = torch.log(smoothed_var_mean)

        # Ordinary least squares: log(e_var) = v_a + v_b^T x
        XtX = X_design.T @ X_design  # (d+1, d+1)
        Xty = X_design.T @ log_e_var  # (d+1,)
        try:
            beta = torch.linalg.solve(XtX + 1e-6 * torch.eye(d + 1, device=device), Xty)
        except RuntimeError:
            # Fallback: stationary
            beta = torch.zeros(d + 1, device=device, dtype=tau.dtype)

        # Update residuals with current v(x) for next IRLS iteration
        # Re-weight by v(x)^{-1/2} so next iteration sees standardized residuals
        v_x = torch.exp(X_design @ beta)  # (n,)
        current_resid = loo_resid / v_x.unsqueeze(1).sqrt().clamp(min=1e-6)

    v_a = beta[0].detach()
    v_b = beta[1:].detach()

    return v_a, v_b


# ---------------------------------------------------------------------------
# 4. LHD screening
# ---------------------------------------------------------------------------

def _screen_lhd(candidates, tau, y, make_kernel_fn, v_a, v_b, top_k=5):
    """
    Evaluate concentrated NLL at each LHD candidate (no gradient),
    return top-K best candidates.

    Parameters
    ----------
    candidates : (n_cand, n_params) raw parameter vectors
    tau : (n, d) inputs
    y : (n, r) PCA-reduced responses
    make_kernel_fn : callable(theta_dict) -> K matrix
    v_a : scalar tensor, current v(x) intercept
    v_b : (d,) tensor, current v(x) slopes
    top_k : int

    Returns
    -------
    top_indices : (top_k,) LongTensor
    top_nlls : (top_k,) float tensor
    """
    n_cand = candidates.shape[0]
    d_input = tau.shape[1]
    nlls = torch.full((n_cand,), float('inf'), device=tau.device)

    with torch.no_grad():
        for i in range(n_cand):
            try:
                theta = _raw_to_theta(candidates[i], d_input, device=tau.device)
                # Add v_a, v_b to theta
                theta['v_a'] = v_a
                theta['v_b'] = v_b
                loss, _ = nll_concentrated(theta, tau, y, make_kernel_fn)
                nlls[i] = loss
            except (RuntimeError, ValueError):
                nlls[i] = float('inf')

    # Select top-K (lowest NLL)
    k_actual = min(top_k, int((nlls < float('inf')).sum()))
    if k_actual == 0:
        # All failed — return first few
        logger.warning("LHD screening: all candidates failed, using first %d", top_k)
        return torch.arange(top_k, device=tau.device), nlls[:top_k]

    top_nlls, top_indices = torch.topk(nlls, k_actual, largest=False)
    return top_indices, top_nlls


# ---------------------------------------------------------------------------
# 5. Projected-gradient L-BFGS (box constraints via clamping)
# ---------------------------------------------------------------------------

def _lbfgs_optimize_with_bounds(raw_init, tau, y, make_kernel_fn, v_a, v_b,
                                bounds, d_input, max_iter=150,
                                max_grad_norm=5.0, priors=None):
    """
    L-BFGS optimization with box constraints (via clamping in closure).

    Parameters
    ----------
    raw_init : (n_params,) initial raw parameter vector
    bounds : list of (lo, hi) tuples
    Returns : (best_raw, best_nll, info)
    """
    device = tau.device
    lo = torch.tensor([b[0] for b in bounds], device=device, dtype=torch.float32)
    hi = torch.tensor([b[1] for b in bounds], device=device, dtype=torch.float32)

    raw = raw_init.clone().detach().requires_grad_(True)
    opt = torch.optim.LBFGS([raw], lr=0.5, max_iter=20, line_search_fn='strong_wolfe')

    best_loss = float('inf')
    best_raw = raw_init.clone()
    last_loss = None
    stall = 0
    _failed = [False]

    def closure():
        opt.zero_grad(set_to_none=True)
        # Project into bounds (soft box constraint)
        with torch.no_grad():
            raw.data.clamp_(min=lo, max=hi)
        try:
            theta = _raw_to_theta(raw, d_input, device=device)
            theta['v_a'] = v_a.detach()
            theta['v_b'] = v_b.detach()
            loss, _ = nll_concentrated(theta, tau, y, make_kernel_fn, priors=priors)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([raw], max_grad_norm)
            _failed[0] = False
            return loss
        except (RuntimeError, ValueError):
            _failed[0] = True
            return torch.tensor(1e10, device=device, dtype=torch.float32, requires_grad=False)

    for it in range(max_iter):
        try:
            loss = opt.step(closure)
        except RuntimeError:
            break
        if _failed[0]:
            break
        val = float(loss.detach())
        if val < best_loss:
            best_loss = val
            best_raw = raw.detach().clone()
        if last_loss is None or last_loss - val > 1e-6:
            last_loss = val
            stall = 0
        else:
            stall += 1
            if stall >= 5:
                break

    # Final clamp
    best_raw = best_raw.clamp(min=lo, max=hi)
    return best_raw, best_loss


# ---------------------------------------------------------------------------
# 6. Main R-style optimizer
# ---------------------------------------------------------------------------

def optimize_cgp(tau, y, make_kernel, log_ell_min,
                 n_cand=500, n_top=5, max_iter=150,
                 n_alternating=3, prev_theta=None):
    """
    R-style CGP optimization with LHD screening + alternating optimization.

    Flow:
    1. Initialize v_a=0, v_b=0 (stationary start)
    2. LHD screening: generate candidates, evaluate NLL, pick top n_top
    3. For each top candidate:
       a. Alternating optimization (n_alternating rounds):
          i.  Fix v_a/v_b → L-BFGS kernel params with box constraints
          ii. Fix kernel params → IRLS update v_a/v_b (4 iterations)
       b. Record final NLL
    4. Return global best

    Parameters
    ----------
    tau : (n, d) normalized input locations
    y : (n, r) PCA-reduced responses
    make_kernel : callable(log_ell, log_sigma_f, log_sigma_n, **extra) -> K
    log_ell_min : float, lower bound for log lengthscales
    n_cand : int, number of LHD screening candidates
    n_top : int, number of top candidates to refine
    max_iter : int, max L-BFGS iterations per refinement
    n_alternating : int, rounds of alternating optimization
    prev_theta : dict, optional, previous MAP estimate for warm-starting

    Returns
    -------
    best_theta : dict, best hyperparameters (same format as optimize_hypers_robust)
    info : dict, optimization info
    """
    d_input = tau.shape[1]
    n_params = 2 * d_input + 3  # log_ell(d) + log_delta(d) + log_sf + log_sn + logit_lambda
    device = tau.device
    bounds = _get_param_bounds(d_input, log_ell_min)

    # Light prior: only v_b shrinkage (std=1.0, much looser than original 0.3)
    # No prior on kernel params — the box constraints handle regularization
    # We do keep a very mild prior on v_b to avoid wild spatial variance patterns
    priors = None  # Priors applied separately in IRLS; kernel params bounded by box

    # Initialize v(x) = constant (stationary start)
    v_a = torch.tensor(0.0, device=device, dtype=torch.float32)
    v_b = torch.zeros(d_input, device=device, dtype=torch.float32)

    # --- Step 1: Generate LHD candidates ---
    logger.info(f"R-style CGP: generating {n_cand} LHD candidates in {n_params}D space")
    candidates = _lhd_candidates(n_params, n_cand, bounds, device=device)

    # Add prev_theta and perturbations as extra candidates
    if prev_theta is not None:
        prev_raw = _theta_to_raw(prev_theta, d_input).to(device)
        extras = [prev_raw.unsqueeze(0)]
        for _ in range(5):
            perturb = prev_raw + 0.3 * torch.randn_like(prev_raw)
            extras.append(perturb.unsqueeze(0))
        extras = torch.cat(extras, dim=0)
        candidates = torch.cat([candidates, extras], dim=0)
        logger.info(f"CGP: added 6 warm-start candidates from prev_theta")

    # --- Step 2: Screen candidates ---
    top_indices, top_nlls = _screen_lhd(candidates, tau, y, make_kernel, v_a, v_b, top_k=n_top)
    logger.info(f"CGP: top {len(top_indices)} NLLs from screening: "
                f"{[f'{v:.2f}' for v in top_nlls.tolist()]}")

    # --- Step 3: Refine each top candidate ---
    global_best_nll = float('inf')
    global_best_theta = None

    for rank, idx in enumerate(top_indices):
        raw_init = candidates[idx].clone()

        # Alternating optimization
        current_v_a = v_a.clone()
        current_v_b = v_b.clone()

        for alt_round in range(n_alternating):
            # (i) Fix v_a/v_b → optimize kernel params with L-BFGS + box constraints
            best_raw, nll_val = _lbfgs_optimize_with_bounds(
                raw_init, tau, y, make_kernel, current_v_a, current_v_b,
                bounds, d_input, max_iter=max_iter
            )
            raw_init = best_raw  # warm-start next alternation

            # (ii) Fix kernel params → IRLS update v_a/v_b
            theta_current = _raw_to_theta(best_raw, d_input, device=device)
            theta_current['v_a'] = current_v_a
            theta_current['v_b'] = current_v_b
            # Build K from current kernel params (no v_a/v_b effect on K — they only
            # affect the CGP composite kernel via the sqrt(v(x)v(x')) term)
            try:
                log_ell = theta_current['log_ell'].detach()
                log_sf = theta_current['log_sigma_f'].detach()
                log_sn = theta_current['log_sigma_n'].detach()
                extra = {k: v.detach() for k, v in theta_current.items()
                         if k not in ('log_ell', 'log_sigma_f', 'log_sigma_n')}
                K = make_kernel(log_ell, log_sf, log_sn, **extra)
                L, _ = safe_cholesky(K)
                new_v_a, new_v_b = _irls_estimate_v(tau, y, L, n_iter=4)

                # Mild shrinkage on v_b (std=1.0)
                v_b_penalty = 0.5 * (new_v_b ** 2).sum()
                if float(v_b_penalty) > 10.0:
                    # v_b too large, scale back
                    scale = math.sqrt(10.0 / float(v_b_penalty))
                    new_v_b = new_v_b * scale
                    logger.debug(f"  IRLS: scaled v_b by {scale:.3f}")

                current_v_a = new_v_a
                current_v_b = new_v_b
            except (RuntimeError, ValueError) as e:
                logger.debug(f"  IRLS failed at alt_round {alt_round}: {e}")
                # Keep current v_a/v_b

        # Final NLL evaluation
        try:
            theta_final = _raw_to_theta(best_raw, d_input, device=device)
            theta_final['v_a'] = current_v_a
            theta_final['v_b'] = current_v_b
            with torch.no_grad():
                final_nll, info = nll_concentrated(theta_final, tau, y, make_kernel)
            final_nll_val = float(final_nll)
        except (RuntimeError, ValueError):
            final_nll_val = float('inf')
            info = {'nll': float('inf'), 'jitter': 0.0}

        logger.info(f"  Candidate {rank}: NLL={final_nll_val:.4f}, "
                    f"v_a={float(current_v_a):.4f}, "
                    f"v_b={current_v_b.tolist()}")

        if final_nll_val < global_best_nll:
            global_best_nll = final_nll_val
            global_best_theta = {
                'log_ell': theta_final['log_ell'].detach().clone(),
                'log_ell_l': theta_final['log_ell_l'].detach().clone(),
                'log_sigma_f': theta_final['log_sigma_f'].detach().clone(),
                'log_sigma_n': theta_final['log_sigma_n'].detach().clone(),
                'logit_lambda': theta_final['logit_lambda'].detach().clone(),
                'v_a': current_v_a.detach().clone(),
                'v_b': current_v_b.detach().clone(),
            }
            best_info = info

    if global_best_theta is None:
        raise RuntimeError("R-style CGP optimization: all candidates failed")

    logger.info(f"R-style CGP: best NLL={global_best_nll:.4f}")
    return global_best_theta, best_info
