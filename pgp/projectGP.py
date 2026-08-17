# projectGP.py — Projected GP on Grassmannian with:
#   - Separable kernel K_Theta (x) K_Y
#   - Empirical Bayes anisotropic K_Y estimation (Method C)
#   - Matheron's rule sampling in N-dim weight space (Method A)
#   - Batched geodesic for vectorized predict_phi
#   - MPS / CPU support (no CUDA dependency)
#   - No hardcoded hyperparameters

import logging
import math
import torch
from .estimation import (rbf_cov, matern52_cov, poly_rbf_cov, gibbs_cov, cgp_cov,
                         optimize_hypers_robust, safe_cholesky,
                         KERNEL_TYPES)
from .cgp_optim import optimize_cgp
from .manifold import log_map, exp_map, geodesic, geodesic_batch, karcher_mean

logger = logging.getLogger(__name__)


class projectGP:
    """
    Gaussian process regression on the Grassmannian via projection.
    - Separable kernel: K_Theta (input) x K_Y (output)
    - ARD lengthscales: alpha = 1/ell^2 per input dim
    - K_Y estimated via empirical Bayes (low-rank) for anisotropic sampling
    """

    def __init__(self, pod_modes: int, device: str = "cpu", kernel: str = "rbf",
                 reference_pt_method: str = "stacked_svd", cgp_pca_rank: int = 5):
        self.pod_modes = int(pod_modes)
        assert self.pod_modes > 0
        assert kernel in KERNEL_TYPES, f"kernel must be one of {KERNEL_TYPES}, got '{kernel}'"
        self.cgp_pca_rank = cgp_pca_rank
        assert reference_pt_method in ("stacked_svd", "karcher"), \
            f"reference_pt_method must be 'stacked_svd' or 'karcher', got '{reference_pt_method}'"
        self.kernel_type = kernel
        self.reference_pt_method = reference_pt_method
        # MPS lacks linalg ops (svd, qr, cholesky_solve) in PyTorch <=2.7.
        # For Apple Silicon: use CPU — it already leverages Accelerate/vecLib
        # for multi-threaded BLAS, which is the fastest option.
        if device == "mps":
            logger.info("MPS requested but linalg ops unsupported; using CPU with Accelerate BLAS.")
            device = "cpu"
        self.device = torch.device(device)
        self.dtype = torch.float32

        # Hyperparameters
        self.ls = torch.tensor(1.0, dtype=self.dtype, device=self.device)
        self.sigma = torch.tensor(1.0, dtype=self.dtype, device=self.device)
        self.sigma_n = None

        # Training data and caches
        self.tau = None        # (k, d)
        self.imgs = None       # (k, n, t)
        self.mean = None       # (k, n, 1)
        self.phi = None        # (k, n, p)
        self.reference_pt = None  # (n, p)
        self.z = None          # (k, n, p)
        self._f_blocks = None  # list of p tensors, each (n-1, n)
        self.y = None          # (k, ny), ny = p*(n-1)
        self.L = None          # Cholesky lower triangular
        self.Linv_y = None     # K^{-1} Y via Cholesky

        # Empirical Bayes K_Y decomposition (Method C)
        self._ky_V = None      # (N, q) principal directions of K_Y
        self._ky_lam = None    # (N,) eigenvalues (variance per direction)

        # Eigendecomposition of K for per-PC variance (breaks separability)
        self._K_eigvals = None  # (k,) eigenvalues of K = K_signal + σ²_n I
        self._K_eigvecs = None  # (k, k) eigenvectors

    def fit(self, tau, imgs, fixed_reference_pt=None, pilot_imgs=None):
        """
        Build projection pipeline: SVD -> log_map -> F -> y.

        Parameters
        ----------
        tau  : (k, d) parameter values
        imgs : (k, n, t) snapshot matrices
        fixed_reference_pt : (n, p) tensor, optional
            If provided, use this fixed reference point instead of computing
            from the data.  This keeps the tangent space consistent across
            sequential design steps.
        pilot_imgs : (M, n, t) tensor, optional
            Additional snapshots (e.g. test data) used ONLY for computing the
            reference point (stacked SVD / Karcher mean).  Not used for GP
            training.  Provides better coverage of the parameter domain.
        """
        tau = torch.as_tensor(tau, dtype=self.dtype, device=self.device)
        imgs = torch.as_tensor(imgs, dtype=self.dtype, device=self.device)

        if tau.dim() == 1:
            tau = tau.unsqueeze(1)
        assert tau.dim() == 2
        assert imgs.dim() == 3
        assert tau.shape[0] == imgs.shape[0]

        self.tau = tau
        self.imgs = imgs

        # Time-centered snapshots
        self.mean = imgs.mean(dim=2, keepdim=True)
        std_imgs = imgs - self.mean

        k, n = imgs.shape[0], imgs.shape[1]
        p = self.pod_modes

        # Per-sample POD subspace
        self.phi = torch.empty((k, n, p), device=self.device, dtype=self.dtype)
        for i in range(k):
            U, _, _ = torch.linalg.svd(std_imgs[i], full_matrices=False)
            self.phi[i] = U[:, :p]

        # Reference point: fixed (from initial design) or computed from data
        if fixed_reference_pt is not None:
            self.reference_pt = torch.as_tensor(
                fixed_reference_pt, dtype=self.dtype, device=self.device)
            logger.info("Using fixed reference point (from initial design)")
        else:
            # Build stacked matrix from training (+ optional pilot) snapshots
            all_std = [std_imgs[i] for i in range(k)]
            if pilot_imgs is not None:
                pilot_t = torch.as_tensor(pilot_imgs, dtype=self.dtype, device=self.device)
                pilot_centered = pilot_t - pilot_t.mean(dim=2, keepdim=True)
                all_std.extend([pilot_centered[j] for j in range(pilot_centered.shape[0])])
                logger.info(f"Reference point: using {k} training + {pilot_centered.shape[0]} pilot snapshots")
            stacked = torch.cat(all_std, dim=1)
            U, _, _ = torch.linalg.svd(stacked, full_matrices=False)
            stacked_svd_ref = U[:, :p]

            if self.reference_pt_method == "karcher":
                # For Karcher mean, also include pilot subspaces
                if pilot_imgs is not None:
                    pilot_phi = torch.empty((pilot_centered.shape[0], n, p),
                                            device=self.device, dtype=self.dtype)
                    for j in range(pilot_centered.shape[0]):
                        Uj, _, _ = torch.linalg.svd(pilot_centered[j], full_matrices=False)
                        pilot_phi[j] = Uj[:, :p]
                    all_phi = torch.cat([self.phi, pilot_phi], dim=0)
                else:
                    all_phi = self.phi
                self.reference_pt, n_iters, final_norm = karcher_mean(
                    all_phi, init=stacked_svd_ref, max_iter=200, tol=1e-8)
                self.karcher_iters = n_iters
                self.karcher_final_norm = float(final_norm)
                if n_iters >= 200:
                    logger.warning("Karcher mean did not converge, falling back to stacked SVD")
                    self.reference_pt = stacked_svd_ref
                else:
                    logger.info(f"Using Karcher mean reference point (converged in {n_iters} iters)")
            else:
                self.reference_pt = stacked_svd_ref

        # Log map to tangent space
        self.z = log_map(self.reference_pt, self.phi)

        # Construct F blocks (avoid dense block-diagonal to save memory)
        self._f_blocks = []
        for i in range(p):
            phi_tilde = self.reference_pt[:, i]
            torch.manual_seed(i)
            rand = torch.randn(n, n - 1, device=self.device, dtype=self.dtype)
            M = torch.cat([phi_tilde.unsqueeze(1), rand], dim=1)
            Q, _ = torch.linalg.qr(M, mode='reduced')
            f_block = Q[:, 1:].T  # (n-1, n)
            self._f_blocks.append(f_block)

        # Map Z -> y (Euclidean space) using block-wise multiply
        self.y = torch.empty((k, p * (n - 1)), device=self.device, dtype=self.dtype)
        for i in range(k):
            for j in range(p):
                z_col = self.z[i, :, j]  # (n,)
                self.y[i, j*(n-1):(j+1)*(n-1)] = self._f_blocks[j] @ z_col

        # Reset K_Y cache (will be computed after set_params)
        self._ky_V = None
        self._ky_lam = None

        return self

    def estimate_hypers(self, add_loo_weight: float = 0.05, n_restarts: int = 5,
                         prev_theta=None, warmstart_shrink: float = 0.5,
                         weighted: bool = False):
        """
        Estimate hyperparameters (MAP + LOO + multi-restart LBFGS).

        Warm-start (Eriksson et al. 2019, TuRBO): If prev_theta is provided,
        the prior is centered on the previous MAP estimate with tighter std.
        This prevents wild jumps between modes when N is small, stabilizing
        the acquisition function across iterations.

        Parameters
        ----------
        prev_theta : dict, optional
            Previous MAP estimate in log-space: {'log_ell', 'log_sigma_f', 'log_sigma_n'}.
            If provided, enables warm-starting.
        warmstart_shrink : float
            Factor to shrink prior std when warm-starting. 0.5 = halve the std.

        Returns
        -------
        dict : {'alpha', 'sigma', 'sigma_n', 'jitter', '_raw_theta'}
            _raw_theta contains the log-space parameters for warm-starting next iteration.
        """
        if self.tau is None or self.y is None:
            raise RuntimeError("Call fit() before estimate_hypers().")

        tau = self.tau
        y = self.y
        kt = self.kernel_type

        # PCA reduction for CGP: avoid D-column averaging killing nonstationarity
        if kt == "cgp":
            with torch.no_grad():
                U_pca, S_pca, Vt_pca = torch.linalg.svd(y, full_matrices=False)
                r = min(self.cgp_pca_rank, S_pca.shape[0])
                y_for_opt = (U_pca[:, :r] * S_pca[:r]).detach().clone()  # (n, r) PCA scores
                explained = float((S_pca[:r] ** 2).sum() / (S_pca ** 2).sum())
                logger.info(f"CGP PCA reduction: {y.shape[1]} -> {r} dims, "
                            f"explained variance ratio = {explained:.4f}")
        else:
            y_for_opt = y

        # Adaptive lengthscale lower bound: median nearest-neighbor distance / 5
        import math
        with torch.no_grad():
            dists = torch.cdist(tau, tau)
            dists.fill_diagonal_(float('inf'))
            nn_median = float(dists.min(dim=1).values.median())
            log_ell_min = max(math.log(max(nn_median / 5.0, 1e-10)), -4.0)
        logger.info(f"Adaptive log_ell_min={log_ell_min:.3f} (ell_min={math.exp(log_ell_min):.4f})")

        # Weighted GP: compute per-point noise weights from tangent norms
        if weighted:
            with torch.no_grad():
                tangent_norms = torch.linalg.norm(y, dim=1)  # (k,)
                median_norm = torch.median(tangent_norms)
                # Inverse-quadratic weighting: small norm → w≈1, large norm → w↓
                tn_weights = 1.0 / (1.0 + (tangent_norms / (median_norm + 1e-8)) ** 2)
                tn_weights = tn_weights / tn_weights.mean()  # normalize to mean 1
            logger.info(f"Weighted GP: tn_weights range [{float(tn_weights.min()):.3f}, {float(tn_weights.max()):.3f}]")
        else:
            tn_weights = None

        def make_kernel(log_ell, log_sigma_f, log_sigma_n, **extra):
            LOG_ELL_MIN, LOG_ELL_MAX = log_ell_min, 6.0
            LOG_SIG_MIN, LOG_SIG_MAX = -10.0, 6.0
            LOG_SN_MIN = -6.0

            log_ell     = torch.clamp(log_ell,     LOG_ELL_MIN, LOG_ELL_MAX)
            log_sigma_f = torch.clamp(log_sigma_f, LOG_SIG_MIN, LOG_SIG_MAX)
            log_sigma_n = torch.clamp(log_sigma_n, LOG_SN_MIN,  LOG_SIG_MAX)

            ell = torch.exp(log_ell)
            sf = torch.exp(log_sigma_f)
            sn = torch.exp(log_sigma_n)
            I_n = torch.eye(len(self.tau), dtype=self.tau.dtype, device=self.tau.device)

            if kt == "rbf":
                Ksig = rbf_cov(self.tau, None, ls=1.0 / (ell**2), sigma=sf, jitter=1e-6)
            elif kt == "matern52":
                Ksig = matern52_cov(self.tau, None, ls=1.0 / (ell**2), sigma=sf, jitter=1e-6)
            elif kt == "poly_rbf":
                log_sp = extra.get('log_sigma_p', torch.tensor(0.0))
                log_s0 = extra.get('log_sigma_0', torch.tensor(0.0))
                logit_w = extra.get('logit_w', torch.tensor(-2.0))
                sp = torch.exp(torch.clamp(log_sp, -6.0, 6.0))
                s0 = torch.exp(torch.clamp(log_s0, -6.0, 6.0))
                w = torch.sigmoid(logit_w)
                Ksig = poly_rbf_cov(self.tau, None, ls=1.0 / (ell**2), sigma=sf,
                                    sigma_p=sp, sigma_0=s0, mix_w=w, jitter=1e-6)
            elif kt == "gibbs":
                a = extra.get('gibbs_a', torch.zeros(ell.shape, dtype=ell.dtype, device=ell.device))
                B = extra.get('gibbs_B', torch.zeros(ell.numel(), self.tau.shape[1],
                              dtype=ell.dtype, device=ell.device))
                # a acts as offset to log_ell (base lengthscale)
                ls_params = {'a': log_ell + a, 'B': B}
                Ksig = gibbs_cov(self.tau, None, ls_params=ls_params, sigma=sf, jitter=1e-6)
            elif kt == "cgp":
                ell_g = ell  # global lengthscales (from log_ell)
                log_ell_l = extra.get('log_ell_l', log_ell)
                ell_l = torch.exp(torch.clamp(log_ell_l, LOG_ELL_MIN, LOG_ELL_MAX))
                lam_val = torch.sigmoid(extra.get('logit_lambda', torch.tensor(0.0)))
                va = extra.get('v_a', torch.tensor(0.0))
                vb = extra.get('v_b', torch.zeros(self.tau.shape[1],
                               dtype=ell.dtype, device=ell.device))
                Ksig = cgp_cov(self.tau, None, ell_g=ell_g, ell_l=ell_l,
                               sigma=sf, lam=lam_val, v_a=va, v_b=vb, jitter=1e-6)
            else:
                raise ValueError(f"Unknown kernel type: {kt}")

            if tn_weights is not None:
                # Heteroscedastic noise: high tangent-norm points get more noise
                return Ksig + (sn**2) * torch.diag(1.0 / tn_weights)
            return Ksig + (sn**2) * I_n

        best_theta, info = optimize_hypers_robust(
            tau, y_for_opt, make_kernel=make_kernel,
            n_restarts=n_restarts, add_loo_weight=add_loo_weight, device=self.device,
            prev_theta=prev_theta, warmstart_shrink=warmstart_shrink,
            kernel_type=kt
        )

        alpha = 1.0 / (torch.exp(best_theta['log_ell'])**2)
        sigma = torch.exp(best_theta['log_sigma_f'])
        sigma_n = torch.exp(best_theta['log_sigma_n'])
        results = {
            'alpha': alpha, 'sigma': sigma,
            'sigma_n': sigma_n, 'jitter': info.get('jitter', None),
            'kernel_type': kt,
            '_raw_theta': {k: v.detach().clone() for k, v in best_theta.items()},
        }
        logger.info(f"Estimated hypers [{kt}]: alpha={alpha}, sigma={sigma}, sigma_n={sigma_n}")
        if kt == "poly_rbf":
            w = torch.sigmoid(best_theta.get('logit_w', torch.tensor(-2.0)))
            logger.info(f"  poly_rbf mix_w={float(w):.4f}")
        elif kt == "gibbs":
            if 'gibbs_B' in best_theta:
                logger.info(f"  gibbs B={best_theta['gibbs_B'].detach()}")
        elif kt == "cgp":
            results['ell_g'] = torch.exp(best_theta['log_ell'])
            results['ell_l'] = torch.exp(best_theta['log_ell_l'])
            results['lam'] = torch.sigmoid(best_theta['logit_lambda'])
            results['v_a'] = best_theta['v_a']
            results['v_b'] = best_theta['v_b']
            logger.info(f"  CGP: lambda={float(results['lam']):.4f}, "
                        f"v_a={float(results['v_a']):.4f}, "
                        f"v_b={results['v_b'].detach().tolist()}")
        return results

    def estimate_hypers_cgp(self, prev_theta=None, weighted=False):
        """
        CGP composite-kernel hyperparameter optimization.

        Uses LHD screening + constrained parameterization + IRLS alternating
        optimization (Ba & Joseph, 2012). Only valid for kernel_type == "cgp".

        Returns same format as estimate_hypers() for drop-in compatibility.
        """
        if self.tau is None or self.y is None:
            raise RuntimeError("Call fit() before estimate_hypers_cgp().")
        if self.kernel_type != "cgp":
            raise ValueError("estimate_hypers_cgp() only supports CGP kernel")

        tau = self.tau
        y = self.y
        kt = self.kernel_type

        # PCA reduction (same as estimate_hypers)
        with torch.no_grad():
            U_pca, S_pca, Vt_pca = torch.linalg.svd(y, full_matrices=False)
            r = min(self.cgp_pca_rank, S_pca.shape[0])
            y_for_opt = (U_pca[:, :r] * S_pca[:r]).detach().clone()
            explained = float((S_pca[:r] ** 2).sum() / (S_pca ** 2).sum())
            logger.info(f"CGP R-style PCA reduction: {y.shape[1]} -> {r} dims, "
                        f"explained variance ratio = {explained:.4f}")

        # Adaptive lengthscale lower bound (same as estimate_hypers)
        import math
        with torch.no_grad():
            dists = torch.cdist(tau, tau)
            dists.fill_diagonal_(float('inf'))
            nn_median = float(dists.min(dim=1).values.median())
            log_ell_min = max(math.log(max(nn_median / 5.0, 1e-10)), -4.0)
        logger.info(f"Adaptive log_ell_min={log_ell_min:.3f} (ell_min={math.exp(log_ell_min):.4f})")

        # Build make_kernel (same closure as estimate_hypers)
        tn_weights = None
        if weighted:
            with torch.no_grad():
                tangent_norms = torch.linalg.norm(y, dim=1)
                median_norm = torch.median(tangent_norms)
                tn_weights = 1.0 / (1.0 + (tangent_norms / (median_norm + 1e-8)) ** 2)
                tn_weights = tn_weights / tn_weights.mean()
            logger.info(f"Weighted GP: tn_weights range [{float(tn_weights.min()):.3f}, {float(tn_weights.max()):.3f}]")

        def make_kernel(log_ell, log_sigma_f, log_sigma_n, **extra):
            LOG_ELL_MIN, LOG_ELL_MAX = log_ell_min, 6.0
            LOG_SIG_MIN, LOG_SIG_MAX = -10.0, 6.0
            LOG_SN_MIN = -6.0

            log_ell     = torch.clamp(log_ell,     LOG_ELL_MIN, LOG_ELL_MAX)
            log_sigma_f = torch.clamp(log_sigma_f, LOG_SIG_MIN, LOG_SIG_MAX)
            log_sigma_n = torch.clamp(log_sigma_n, LOG_SN_MIN,  LOG_SIG_MAX)

            ell = torch.exp(log_ell)
            sf = torch.exp(log_sigma_f)
            sn = torch.exp(log_sigma_n)
            I_n = torch.eye(len(self.tau), dtype=self.tau.dtype, device=self.tau.device)

            ell_g = ell
            log_ell_l = extra.get('log_ell_l', log_ell)
            ell_l = torch.exp(torch.clamp(log_ell_l, LOG_ELL_MIN, LOG_ELL_MAX))
            lam_val = torch.sigmoid(extra.get('logit_lambda', torch.tensor(0.0)))
            va = extra.get('v_a', torch.tensor(0.0))
            vb = extra.get('v_b', torch.zeros(self.tau.shape[1],
                           dtype=ell.dtype, device=ell.device))
            Ksig = cgp_cov(self.tau, None, ell_g=ell_g, ell_l=ell_l,
                           sigma=sf, lam=lam_val, v_a=va, v_b=vb, jitter=1e-6)

            if tn_weights is not None:
                return Ksig + (sn**2) * torch.diag(1.0 / tn_weights)
            return Ksig + (sn**2) * I_n

        # Convert prev_theta if provided
        prev_raw_theta = None
        if prev_theta is not None:
            prev_raw_theta = {k: v.detach().clone().to(device=self.device)
                              for k, v in prev_theta.items()
                              if k in ('log_ell', 'log_ell_l', 'log_sigma_f',
                                       'log_sigma_n', 'logit_lambda', 'v_a', 'v_b')}

        best_theta, info = optimize_cgp(
            tau, y_for_opt, make_kernel=make_kernel,
            log_ell_min=log_ell_min,
            prev_theta=prev_raw_theta,
        )

        # Build results dict (same format as estimate_hypers)
        alpha = 1.0 / (torch.exp(best_theta['log_ell'])**2)
        sigma = torch.exp(best_theta['log_sigma_f'])
        sigma_n = torch.exp(best_theta['log_sigma_n'])
        results = {
            'alpha': alpha, 'sigma': sigma,
            'sigma_n': sigma_n, 'jitter': info.get('jitter', None),
            'kernel_type': kt,
            '_raw_theta': {k: v.detach().clone() for k, v in best_theta.items()},
            'ell_g': torch.exp(best_theta['log_ell']),
            'ell_l': torch.exp(best_theta['log_ell_l']),
            'lam': torch.sigmoid(best_theta['logit_lambda']),
            'v_a': best_theta['v_a'],
            'v_b': best_theta['v_b'],
        }
        logger.info(f"Estimated hypers [cgp R-style]: alpha={alpha}, sigma={sigma}, sigma_n={sigma_n}")
        logger.info(f"  CGP: lambda={float(results['lam']):.4f}, "
                    f"v_a={float(results['v_a']):.4f}, "
                    f"v_b={results['v_b'].detach().tolist()}")
        return results

    def set_params(self, ls=1.0, sigma=1.0, sigma_n=None, jitter: float = 1e-6,
                   extra_kernel_params=None):
        """
        Set hyperparameters and compute Cholesky decomposition + K_Y.

        Parameters
        ----------
        extra_kernel_params : dict, optional
            Additional kernel parameters for non-stationary kernels.
            For poly_rbf: {'sigma_p', 'sigma_0', 'mix_w'}
            For gibbs: {'ls_params': {'a': ..., 'B': ...}}
        """
        if self.tau is None or self.y is None:
            raise RuntimeError("Call fit() before set_params().")

        self.ls = torch.as_tensor(ls, dtype=self.dtype, device=self.device)
        self.sigma = torch.as_tensor(sigma, dtype=self.dtype, device=self.device)
        if sigma_n is not None:
            self.sigma_n = torch.as_tensor(sigma_n, dtype=self.dtype, device=self.device)
        self._extra_kernel_params = extra_kernel_params or {}

        k = self.tau.shape[0]
        R = self._build_kernel_matrix(self.tau, None, jitter=jitter)
        if self.sigma_n is not None:
            R = R + (self.sigma_n**2) * torch.eye(k, device=self.device, dtype=R.dtype)

        self.L, _ = safe_cholesky(R)
        self.Linv_y = torch.cholesky_solve(self.y, self.L)  # (k, ny)

        # Eigendecompose K for per-PC variance computation
        with torch.no_grad():
            self._K_eigvals, self._K_eigvecs = torch.linalg.eigh(R)

        # Compute empirical Bayes K_Y decomposition (Method C)
        self._compute_ky_decomposition()

        return {'L_shape': self.L.shape, 'Linv_y_shape': self.Linv_y.shape}

    def _build_kernel_matrix(self, tau_left, tau_right=None, jitter=1e-6):
        """Build kernel matrix using the configured kernel type."""
        if self.kernel_type == "rbf":
            return rbf_cov(tau_left, tau_right, ls=self.ls, sigma=self.sigma, jitter=jitter)
        elif self.kernel_type == "matern52":
            return matern52_cov(tau_left, tau_right, ls=self.ls, sigma=self.sigma, jitter=jitter)
        elif self.kernel_type == "poly_rbf":
            ekp = self._extra_kernel_params
            return poly_rbf_cov(
                tau_left, tau_right, ls=self.ls, sigma=self.sigma,
                sigma_p=ekp.get('sigma_p', 1.0),
                sigma_0=ekp.get('sigma_0', 1.0),
                mix_w=ekp.get('mix_w', 0.1),
                jitter=jitter
            )
        elif self.kernel_type == "gibbs":
            ekp = self._extra_kernel_params
            return gibbs_cov(
                tau_left, tau_right,
                ls_params=ekp.get('ls_params', None),
                sigma=self.sigma, jitter=jitter
            )
        elif self.kernel_type == "cgp":
            ekp = self._extra_kernel_params
            d = tau_left.shape[-1] if tau_left.ndim > 1 else 1
            return cgp_cov(
                tau_left, tau_right,
                ell_g=ekp.get('ell_g', torch.ones(d, dtype=self.dtype, device=self.device)),
                ell_l=ekp.get('ell_l', torch.ones(d, dtype=self.dtype, device=self.device)),
                sigma=self.sigma,
                lam=ekp.get('lam', 0.5),
                v_a=ekp.get('v_a', 0.0),
                v_b=ekp.get('v_b', None),
                jitter=jitter
            )
        else:
            return rbf_cov(tau_left, tau_right, ls=self.ls, sigma=self.sigma, jitter=jitter)

    def _compute_ky_decomposition(self):
        """
        Empirical Bayes estimation of output covariance K_Y.

        For separable GP with K = K_Theta (x) K_Y, the MLE of K_Y is:
            K_Y_hat = (1/N) * Y^T K_Theta^{-1} Y

        Since rank(K_Y_hat) <= N << q, we store its low-rank SVD.
        This gives anisotropic variance directions for sampling.
        """
        # W = L^{-1} Y, where K = L L^T
        # Then K_Y_hat = (1/N) W^T W
        W = torch.linalg.solve_triangular(self.L, self.y, upper=False)  # (N, q)
        N = W.shape[0]

        # SVD of W: W = U_w S_w V_w^T
        # => K_Y_hat = (1/N) V_w S_w^2 V_w^T
        _, S_w, Vt_w = torch.linalg.svd(W, full_matrices=False)  # S_w: (N,), Vt_w: (N, q)

        self._ky_V = Vt_w       # (N, q): principal directions
        self._ky_lam = S_w**2 / N   # (N,): variance per direction

    @torch.no_grad()
    def _compute_per_pc_variance(self, tau_pred):
        """
        Per-K̂_Y-direction predictive variance (breaks separability).

        For direction j of K̂_Y with eigenvalue λ_j, the effective GP has:
            K_j = λ_j K_signal + σ²_n I

        Its predictive variance at x* is:
            σ²_j(x*) = λ_j k(x*,x*) - λ_j² k_*^T K_j^{-1} k_*

        Different λ_j give different SNR → different spatial variance patterns.
        This is computed efficiently via eigendecomposition of K (done once in set_params).

        Returns
        -------
        var_per_dir : (N, n_dir) per-direction predictive variances
        """
        tau_pred = torch.as_tensor(tau_pred, dtype=self.dtype, device=self.device)
        if tau_pred.dim() == 1:
            tau_pred = tau_pred.unsqueeze(0)

        # Cross-covariance (signal only, no noise)
        r = self._build_kernel_matrix(tau_pred, self.tau, jitter=0.0)  # (N, k)

        # Project onto K eigenbasis: a_i(x*) = (U^T k_*)_i
        a = r @ self._K_eigvecs  # (N, k)

        sn2 = float(self.sigma_n ** 2) if self.sigma_n is not None else 0.0
        # Signal eigenvalues: K = K_signal + σ²_n I → d_sig = eigvals(K) - σ²_n
        d_sig = (self._K_eigvals - sn2).clamp(min=0)  # (k,)

        # k_signal(x*,x*) = σ²_f for stationary kernels
        k_xx_signal = float(self.sigma ** 2)

        # Use top pod_modes directions (dominant K̂_Y eigenvalues)
        n_dir = min(int(self._ky_lam.shape[0]), self.pod_modes)
        N = tau_pred.shape[0]
        var_per_dir = torch.empty(N, n_dir, dtype=self.dtype, device=self.device)

        for j in range(n_dir):
            lam_j = float(self._ky_lam[j])
            # K_j eigenvalues: λ_j d_sig_i + σ²_n
            mu_j = lam_j * d_sig + sn2  # (k,)
            mu_j = mu_j.clamp(min=1e-12)
            # σ²_j(x*) = λ_j k_xx - λ_j² Σ_i a_i² / μ_j_i  (latent, no noise)
            quad_j = torch.sum(a ** 2 / mu_j.unsqueeze(0), dim=1)  # (N,)
            var_per_dir[:, j] = lam_j * k_xx_signal - lam_j ** 2 * quad_j

        return var_per_dir.clamp(min=1e-12)

    @torch.no_grad()
    def predict(self, tau_pred):
        """
        GP prediction.
        Returns (mean, var):
          - mean: (N, ny) posterior mean
          - var:  (N,) scalar posterior variance (from input kernel)
        """
        if self.L is None or self.Linv_y is None:
            raise RuntimeError("Call set_params() before predict().")

        tau_pred = torch.as_tensor(tau_pred, dtype=self.dtype, device=self.device)
        if tau_pred.dim() == 1 and tau_pred.shape[0] == self.tau.shape[-1]:
            tau_pred = tau_pred.unsqueeze(0)
        elif tau_pred.dim() == 1:
            raise ValueError(f"Feature dims mismatch: {tau_pred.shape[0]} vs {self.tau.shape[-1]}")

        r = self._build_kernel_matrix(tau_pred, self.tau, jitter=0.0)  # (N, k)
        mean = r @ self.Linv_y  # (N, ny)

        Kinv_rT = torch.cholesky_solve(r.T, self.L)  # (k, N)
        quad = torch.sum(r * Kinv_rT.T, dim=1)       # (N,)
        k_xx = (self.sigma**2).expand_as(quad)
        var = torch.clamp(k_xx - quad, min=1e-12)
        return mean, var

    @torch.no_grad()
    def bijective_map(self, y_pred):
        """
        Map y in Euclidean space back to Grassmannian via F^T -> reshape -> exp_map.
        y_pred: (ny,) or (N, ny)
        Returns: (n, p) or (N, n, p)

        Uses block-wise multiply with stored F blocks to avoid materializing
        the full (p*(n-1), p*n) dense matrix — saves ~p× memory.
        """
        y = torch.as_tensor(y_pred, dtype=self.dtype, device=self.device)
        squeezed = y.dim() == 1
        if squeezed:
            y = y.unsqueeze(0)
        N = y.shape[0]

        n, p = self.mean.shape[1], self.pod_modes
        Z = torch.empty(N, n, p, dtype=self.dtype, device=self.device)
        nm1 = n - 1
        for j in range(p):
            Z[:, :, j] = y[:, j*nm1:(j+1)*nm1] @ self._f_blocks[j]
        phi_pred = exp_map(self.reference_pt, Z)

        return phi_pred.squeeze(0) if squeezed else phi_pred

    @torch.no_grad()
    def predict_phi(self, tau_pred, n_samples: int = 100, method: str = "empirical_bayes",
                    generator=None):
        """
        Predict subspace and manifold variance via MC sampling.

        Parameters
        ----------
        tau_pred  : (N, d) test parameters
        n_samples : number of MC samples
        method    : sampling method
            "empirical_bayes" — anisotropic K_Y sampling (Method C, recommended)
            "matheron"        — Matheron's rule in N-dim weight space (Method A)
            "isotropic"       — original isotropic sampling (baseline)
        generator : torch.Generator, optional
            If provided, used for MC noise sampling (isolates MC randomness).

        Returns
        -------
        Phi_mean : (N, n, p) predicted subspaces
        var_phi  : (N,) manifold variance
        """
        mu_y, var_y = self.predict(tau_pred)   # (N, ny), (N,)
        N, ny = mu_y.shape
        Phi_mean = self.bijective_map(mu_y)    # (N, n, p)
        if Phi_mean.ndim == 2:
            Phi_mean = Phi_mean.unsqueeze(0)
        std = torch.sqrt(var_y)                # (N,)

        if method == "empirical_bayes":
            var_phi = self._sample_empirical_bayes(mu_y, std, Phi_mean, n_samples,
                                                   tau_pred=tau_pred,
                                                   generator=generator)
        elif method == "matheron":
            var_phi = self._sample_matheron(tau_pred, Phi_mean, n_samples,
                                            generator=generator)
        else:
            var_phi = self._sample_isotropic(mu_y, std, Phi_mean, n_samples,
                                             generator=generator)

        return Phi_mean, var_phi

    # -- Batch size for vectorized MC sampling (candidates per batch) --------
    MC_BATCH_SIZE = 16

    def _batched_mc_manifold_var(self, mu_y, std, Phi_mean, n_samples,
                                 V_proj=None, sqrt_lam=None, per_pc_std=None,
                                 generator=None):
        """
        Batched MC estimation of manifold variance on Gr(p,n).

        Processes candidates in batches of MC_BATCH_SIZE to balance throughput
        vs. memory.  All heavy ops (randn, matmul, exp_map, geodesic_batch)
        are executed on (B*M, ...) tensors instead of one-by-one.

        Parameters
        ----------
        mu_y      : (N, q)  posterior means
        std       : (N,)    posterior std-devs (used when per_pc_std is None)
        Phi_mean  : (N, n, p)  posterior mean subspaces on Grassmannian
        n_samples : int     MC samples per candidate
        V_proj    : (r, q)  if given, project noise through V (anisotropic)
        sqrt_lam  : (r,)    sqrt of eigenvalues (used with V_proj)

        Returns
        -------
        var_phi : (N,) manifold variance estimates
        """
        N = mu_y.shape[0]
        n_dim, p_dim = Phi_mean.shape[1], Phi_mean.shape[2]
        per_pc = per_pc_std is not None and V_proj is not None
        aniso = V_proj is not None and sqrt_lam is not None and not per_pc
        if per_pc:
            noise_dim = per_pc_std.shape[1]
        elif aniso:
            noise_dim = sqrt_lam.shape[0]
        else:
            noise_dim = mu_y.shape[1]
        B = self.MC_BATCH_SIZE

        parts = []
        for start in range(0, N, B):
            end = min(start + B, N)
            Bc = end - start                                   # actual batch size

            # --- sample noise --------------------------------------------------
            eps = torch.randn(Bc, n_samples, noise_dim,
                              dtype=self.dtype, device=self.device,
                              generator=generator)

            if per_pc:
                # Per-direction std: each K̂_Y direction has its own σ_j(x*)
                # per_pc_std: (N, n_dir), V_proj: (n_dir, q)
                scale = per_pc_std[start:end].unsqueeze(1)      # (Bc, 1, n_dir)
                delta_y = (eps * scale) @ V_proj                # (Bc, M, q)
            elif aniso:
                # scale = std_i * sqrt(lambda_j)  →  (Bc, 1, r)
                scale = (std[start:end].view(Bc, 1, 1)
                         * sqrt_lam.view(1, 1, -1))            # (Bc, 1, r)
                delta_y = (eps * scale) @ V_proj                # (Bc, M, q)
            else:
                # isotropic: delta_y = std_i * eps
                delta_y = eps * std[start:end].view(Bc, 1, 1)  # (Bc, M, q)
            del eps

            # --- posterior samples ---------------------------------------------
            Y_flat = (mu_y[start:end].unsqueeze(1) + delta_y)  # (Bc, M, q)
            del delta_y
            Y_flat = Y_flat.reshape(Bc * n_samples, -1)        # (Bc*M, q)

            # --- map to Grassmannian -------------------------------------------
            Phi_s = self.bijective_map(Y_flat)                  # (Bc*M, n, p)
            del Y_flat

            # --- geodesic distances --------------------------------------------
            Phi_ref = (Phi_mean[start:end]
                       .unsqueeze(1)
                       .expand(Bc, n_samples, n_dim, p_dim)
                       .reshape(Bc * n_samples, n_dim, p_dim))
            d2 = geodesic_batch(Phi_ref, Phi_s)                 # (Bc*M,)
            del Phi_ref, Phi_s

            parts.append(d2.view(Bc, n_samples).mean(dim=1))   # (Bc,)
            del d2

        return torch.cat(parts)                                 # (N,)

    def _sample_empirical_bayes(self, mu_y, std, Phi_mean, n_samples,
                                tau_pred=None, generator=None):
        """
        Method C: Anisotropic sampling using empirical Bayes K_Y.

        Two modes:
        1. Separable (default): sigma^2(x*) * K̂_Y — shared scalar variance.
        2. Per-PC (when eigendecomposition available): per-direction σ²_j(x*)
           from K_j = λ_j K_signal + σ²_n I. Breaks separability so that
           different K̂_Y directions have different spatial variance patterns.

        Per-PC mode is activated when tau_pred is provided and
        eigendecomposition was computed in set_params().
        """
        # Per-PC variance: different σ²_j(x*) per K̂_Y direction
        if tau_pred is not None and self._K_eigvals is not None:
            tau_t = torch.as_tensor(tau_pred, dtype=self.dtype, device=self.device)
            if tau_t.dim() == 1:
                tau_t = tau_t.unsqueeze(0)
            var_per_dir = self._compute_per_pc_variance(tau_t)  # (N, n_dir)
            pc_std = torch.sqrt(var_per_dir)                     # (N, n_dir)
            n_dir = var_per_dir.shape[1]
            V_proj = self._ky_V[:n_dir]  # (n_dir, q)
            logger.debug(f"Per-PC variance: {n_dir} directions, "
                         f"var range [{float(var_per_dir.min()):.4e}, {float(var_per_dir.max()):.4e}]")
            return self._batched_mc_manifold_var(
                mu_y, std, Phi_mean, n_samples,
                V_proj=V_proj, per_pc_std=pc_std,
                generator=generator,
            )

        # Fallback: separable model (shared scalar variance)
        sqrt_lam = torch.sqrt(self._ky_lam.clamp(min=1e-20))
        return self._batched_mc_manifold_var(
            mu_y, std, Phi_mean, n_samples,
            V_proj=self._ky_V, sqrt_lam=sqrt_lam,
            generator=generator,
        )

    def _sample_matheron(self, tau_pred, Phi_mean, n_samples, generator=None):
        """
        Method A: Matheron's rule — sample in N-dim weight space.

        For separable GP with K = K_Theta (x) K_Y, a posterior sample is:
            y*(theta*) = mu(theta*) + delta(theta*)

        where the perturbation uses the posterior covariance decomposition:
            Cov(theta*) = sigma^2(theta*) * K_Y_hat

        We sample using the empirical K_Y directions (same as Method C)
        but derive sigma^2(theta*) from the cross-covariance structure.
        """
        tau_pred = torch.as_tensor(tau_pred, dtype=self.dtype, device=self.device)
        if tau_pred.dim() == 1:
            tau_pred = tau_pred.unsqueeze(0)

        # Use configured kernel (not hardcoded rbf_cov)
        k_star = self._build_kernel_matrix(tau_pred, self.tau, jitter=0.0)
        Linv_kstar = torch.linalg.solve_triangular(self.L, k_star.T, upper=False)

        mu_y = k_star @ self.Linv_y  # (N_test, ny)

        # Vectorised variance: var_i = sigma^2 - ||L^{-1} k_*_i||^2
        var_y = (self.sigma ** 2) - (Linv_kstar ** 2).sum(dim=0)  # (N_test,)
        var_y = var_y.clamp(min=1e-12)
        std = torch.sqrt(var_y)

        # Use K_Y decomposition if available, fall back to isotropic
        has_ky = self._ky_V is not None and self._ky_lam is not None
        if has_ky:
            sqrt_lam = torch.sqrt(self._ky_lam.clamp(min=1e-20))
            return self._batched_mc_manifold_var(
                mu_y, std, Phi_mean, n_samples,
                V_proj=self._ky_V, sqrt_lam=sqrt_lam,
                generator=generator,
            )
        else:
            return self._batched_mc_manifold_var(
                mu_y, std, Phi_mean, n_samples,
                generator=generator,
            )

    def check_ball_constraint(self, warn: bool = True):
        """
        Check whether tangent vectors lie within the geodesic ball of radius π/2.

        The log map on Gr(p,n) is only a bijection within B(Φ_ref, π/2).
        If ||Z_i|| >= π/2 for some training point i, the bijective map
        assumption is violated, potentially causing prediction artifacts.

        Returns
        -------
        norms : (k,) tangent vector norms
        violations : list of indices where norm >= π/2
        """
        if self.z is None:
            raise RuntimeError("Call fit() before check_ball_constraint().")

        norms = []
        k = self.z.shape[0]
        for i in range(k):
            # Tangent vector norm = ||Z_i|| = ||arctan(S_i)||  (from SVD of Z)
            U, S, Vh = torch.linalg.svd(self.z[i], full_matrices=False)
            norm = torch.linalg.norm(S)  # S already contains arctan(principal angles)
            norms.append(float(norm))

        norms_arr = torch.tensor(norms)
        violations = [i for i, n in enumerate(norms) if n >= math.pi / 2]

        if warn and violations:
            logger.warning(
                f"Ball constraint violated at {len(violations)}/{k} points. "
                f"Max norm = {max(norms):.4f} >= π/2 = {math.pi/2:.4f}. "
                f"Tangent space approximation may be inaccurate."
            )

        return norms_arr, violations

    @torch.no_grad()
    def predict_geodesic_to_true(self, tau_pred, phi_true):
        """
        Compute geodesic distance from predicted subspace to true subspace.

        This is the primary evaluation metric from the pGP paper:
            d_G(Φ_pred, Φ_true) = || arccos(svd(Φ_pred^T Φ_true)) ||

        Parameters
        ----------
        tau_pred : (N, d) test parameters
        phi_true : (N, n, p) true subspaces at test points

        Returns
        -------
        geodesic_dists : (N,) geodesic distances
        """
        mu_y, _ = self.predict(tau_pred)
        Phi_pred = self.bijective_map(mu_y)  # (N, n, p)
        if Phi_pred.ndim == 2:
            Phi_pred = Phi_pred.unsqueeze(0)

        phi_true = torch.as_tensor(phi_true, dtype=self.dtype, device=self.device)
        if phi_true.ndim == 2:
            phi_true = phi_true.unsqueeze(0)

        return geodesic_batch(Phi_pred, phi_true)

    def _sample_isotropic(self, mu_y, std, Phi_mean, n_samples, generator=None):
        """
        Original isotropic sampling (baseline): eps ~ N(0, I_q).
        Vectorized version using batched geodesic.
        """
        return self._batched_mc_manifold_var(
            mu_y, std, Phi_mean, n_samples,
            generator=generator,
        )
