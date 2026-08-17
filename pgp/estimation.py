# estimation.py — kernel library + robust hyperparam estimation (float32)
#
# Kernels available:
#   - "rbf"          : stationary RBF with ARD (default)
#   - "matern52"     : Matérn 5/2 with ARD (stationary, finite differentiability)
#   - "poly_rbf"     : RBF + polynomial (non-stationary, position-dependent variance)
#   - "gibbs"        : Gibbs-Paciorek (non-stationary, input-dependent lengthscales)
#
import logging
import math
import torch

logger = logging.getLogger(__name__)


def _ensure_2d(X: torch.Tensor) -> torch.Tensor:
    return X.unsqueeze(1) if X.dim() == 1 else X


def rbf_cov(tau_left, tau_right=None, ls=1.0, sigma=1.0, jitter: float = 1e-6):
    """
    Unified RBF covariance with optional ARD.
    lengthscale = alpha (scalar or (d,)), alpha_d = 1 / ell_d^2
    sigma       = signal std (not variance)
    - If tau_right is None: returns K(X,X) with jitter
    - Else: returns K(X,Y) without jitter
    """
    X = _ensure_2d(torch.as_tensor(tau_left))
    X = X.to(dtype=torch.float32, device=X.device)
    alpha = torch.as_tensor(ls, device=X.device, dtype=X.dtype)
    sig = torch.as_tensor(sigma, dtype=X.dtype, device=X.device)

    def _dist2(A, B):
        A = torch.as_tensor(A)
        B = torch.as_tensor(B, dtype=A.dtype, device=A.device)
        if A.shape[-1] != B.shape[-1]:
            raise ValueError(f"Feature dims mismatch: {A.shape[-1]} vs {B.shape[-1]}")
        d = A.shape[-1]
        if alpha.numel() == 1:
            d2 = torch.cdist(A, B).pow(2) * alpha
        elif alpha.numel() == d:
            sa = torch.sqrt(alpha).view(1, -1)
            d2 = torch.cdist(A * sa, B * sa).pow(2)
        else:
            raise ValueError(f"alpha must be scalar or length {d}, got {alpha.numel()}")
        return d2.clamp_(0.0, 80.0)

    if tau_right is None:
        k = X.shape[0]
        if k == 1:
            raise ValueError("Failed to build model with 1 training data point")
        dist2 = _dist2(X, X)
        K = (sig**2) * torch.exp(-0.5 * dist2)
        K = 0.5 * (K + K.T)
        diag_mean = torch.mean(torch.clamp(torch.diagonal(K).abs(), min=1e-12))
        rel_jitter = float(jitter) * float(torch.clamp(diag_mean, min=1.0))
        K = K + rel_jitter * torch.eye(k, device=X.device, dtype=X.dtype)
        return K
    else:
        Y = _ensure_2d(torch.as_tensor(tau_right)).to(dtype=X.dtype, device=X.device)
        dist2 = _dist2(X, Y)
        return (sig**2) * torch.exp(-0.5 * dist2)


def matern52_cov(tau_left, tau_right=None, ls=1.0, sigma=1.0, jitter: float = 1e-6):
    """
    Matérn 5/2 kernel with optional ARD.

    k(r) = sigma^2 * (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r)

    where r = sqrt(sum_d alpha_d * (x_id - x_jd)^2), alpha_d = 1/ell_d^2.

    Matérn 5/2 is twice differentiable (C^2), making it suitable for
    physical processes that are smooth but not infinitely differentiable.
    """
    X = _ensure_2d(torch.as_tensor(tau_left)).to(dtype=torch.float32)
    alpha = torch.as_tensor(ls, device=X.device, dtype=X.dtype)
    sig = torch.as_tensor(sigma, dtype=X.dtype, device=X.device)

    def _scaled_dist(A, B):
        """Compute scaled Euclidean distance sqrt(sum_d alpha_d * (a_d - b_d)^2)."""
        A = torch.as_tensor(A, dtype=X.dtype, device=X.device)
        B = torch.as_tensor(B, dtype=A.dtype, device=A.device)
        d = A.shape[-1]
        if alpha.numel() == 1:
            sa = torch.sqrt(alpha).view(1, -1).expand(1, d)
        elif alpha.numel() == d:
            sa = torch.sqrt(alpha).view(1, -1)
        else:
            raise ValueError(f"alpha must be scalar or length {d}, got {alpha.numel()}")
        return torch.cdist(A * sa, B * sa).clamp(min=0.0)

    is_sym = tau_right is None
    if is_sym:
        Y = X
    else:
        Y = _ensure_2d(torch.as_tensor(tau_right)).to(dtype=X.dtype, device=X.device)

    r = _scaled_dist(X, Y)  # (n, m)
    sqrt5_r = math.sqrt(5.0) * r
    K = (sig ** 2) * (1.0 + sqrt5_r + 5.0 / 3.0 * r ** 2) * torch.exp(-sqrt5_r)

    if is_sym:
        n = X.shape[0]
        if n == 1:
            raise ValueError("Failed to build model with 1 training data point")
        K = 0.5 * (K + K.T)
        diag_mean = torch.mean(torch.clamp(torch.diagonal(K).abs(), min=1e-12))
        rel_jitter = float(jitter) * float(torch.clamp(diag_mean, min=1.0))
        K = K + rel_jitter * torch.eye(n, device=X.device, dtype=X.dtype)

    return K


def poly_rbf_cov(tau_left, tau_right=None, ls=1.0, sigma=1.0,
                  sigma_p=1.0, sigma_0=1.0, mix_w=0.1,
                  degree=2, jitter: float = 1e-6):
    """
    Non-stationary kernel: weighted sum of RBF and polynomial.

        k(x_i, x_j) = (1 - w) * k_RBF(x_i, x_j) + w * k_poly(x_i, x_j)

    The polynomial component k_poly = sigma_p^2 * (sigma_0^2 + x_i^T x_j)^degree
    is non-stationary: it depends on absolute positions, not just differences.
    This captures large-scale trends and position-dependent variance.

    Parameters
    ----------
    sigma_p : polynomial signal amplitude
    sigma_0 : polynomial bias/offset
    mix_w   : mixing weight in [0, 1] (0 = pure RBF, 1 = pure polynomial)
    degree  : polynomial degree (1 = linear, 2 = quadratic)
    """
    X = _ensure_2d(torch.as_tensor(tau_left)).to(dtype=torch.float32)
    sig_p = torch.as_tensor(sigma_p, dtype=X.dtype, device=X.device)
    sig_0 = torch.as_tensor(sigma_0, dtype=X.dtype, device=X.device)
    w = torch.as_tensor(mix_w, dtype=X.dtype, device=X.device).clamp(0.0, 1.0)

    is_sym = tau_right is None
    if is_sym:
        Y = X
    else:
        Y = _ensure_2d(torch.as_tensor(tau_right)).to(dtype=X.dtype, device=X.device)

    # RBF component (reuse existing function for cross-covariance)
    K_rbf = rbf_cov(tau_left, tau_right, ls=ls, sigma=sigma, jitter=0.0)

    # Polynomial component: sigma_p^2 * (sigma_0^2 + X @ Y^T)^degree
    dot = X @ Y.T  # (n, m)
    K_poly = (sig_p ** 2) * (sig_0 ** 2 + dot).pow(degree)

    K = (1.0 - w) * K_rbf + w * K_poly

    if is_sym:
        K = 0.5 * (K + K.T)
        diag_mean = torch.mean(torch.clamp(torch.diagonal(K).abs(), min=1e-12))
        rel_jitter = float(jitter) * float(torch.clamp(diag_mean, min=1.0))
        K = K + rel_jitter * torch.eye(X.shape[0], device=X.device, dtype=X.dtype)

    return K


def gibbs_cov(tau_left, tau_right=None, ls_params=None, sigma=1.0,
              jitter: float = 1e-6):
    """
    Gibbs-Paciorek non-stationary kernel with input-dependent lengthscales.

    Lengthscale function (log-linear):
        log ell_d(x) = a_d + b_d1 * x_1 + b_d2 * x_2

    Kernel (Paciorek & Schervish, 2004):
        k(x_i, x_j) = sigma^2 * |Sigma_i|^{1/4} |Sigma_j|^{1/4} |bar{Sigma}|^{-1/2}
                       * exp(-Q_ij)
    where:
        Sigma(x) = diag(ell_1(x)^2, ell_2(x)^2)
        bar{Sigma} = (Sigma_i + Sigma_j) / 2
        Q_ij = (x_i - x_j)^T bar{Sigma}^{-1} (x_i - x_j)

    Parameters
    ----------
    ls_params : dict with keys 'a' (d,), 'B' (d, d_input)
        a[k] = intercept for log ell_k
        B[k, j] = slope of x_j for log ell_k
    """
    X = _ensure_2d(torch.as_tensor(tau_left)).to(dtype=torch.float32)
    sig = torch.as_tensor(sigma, dtype=X.dtype, device=X.device)
    n, d = X.shape

    is_sym = tau_right is None
    if is_sym:
        Y = X
    else:
        Y = _ensure_2d(torch.as_tensor(tau_right)).to(dtype=X.dtype, device=X.device)
    m = Y.shape[0]

    # Default: stationary (a=0, B=0 => constant lengthscale = 1)
    if ls_params is None:
        ls_params = {
            'a': torch.zeros(d, dtype=X.dtype, device=X.device),
            'B': torch.zeros(d, d, dtype=X.dtype, device=X.device),
        }

    a = ls_params['a']  # (d,)
    B = ls_params['B']  # (d, d_input)

    # Compute local lengthscales: log_ell(x) = a + B @ x  =>  ell(x) = exp(a + B @ x)
    log_ell_X = a.unsqueeze(0) + X @ B.T  # (n, d)
    log_ell_Y = a.unsqueeze(0) + Y @ B.T  # (m, d)
    ell2_X = torch.exp(2.0 * log_ell_X)   # (n, d)  ell^2 per dim
    ell2_Y = torch.exp(2.0 * log_ell_Y)   # (m, d)

    # Determinant prefactors: |Sigma|^{1/4} = prod(ell_d^{1/2})
    # log|Sigma_i|^{1/4} = 0.5 * sum(log_ell_d(x_i))
    log_det_quarter_X = 0.5 * log_ell_X.sum(dim=1)  # (n,)
    log_det_quarter_Y = 0.5 * log_ell_Y.sum(dim=1)  # (m,)

    # For each pair (i, j): bar_Sigma = diag((ell2_X[i] + ell2_Y[j]) / 2)
    # |bar_Sigma|^{-1/2} = prod_d (2 / (ell2_X[i,d] + ell2_Y[j,d]))^{1/2}
    # log|bar_Sigma|^{-1/2} = 0.5 * sum_d log(2 / (ell2_X[i,d] + ell2_Y[j,d]))
    ell2_X_exp = ell2_X.unsqueeze(1)  # (n, 1, d)
    ell2_Y_exp = ell2_Y.unsqueeze(0)  # (1, m, d)
    avg_ell2 = 0.5 * (ell2_X_exp + ell2_Y_exp)  # (n, m, d)

    log_det_neg_half_avg = 0.5 * torch.log(1.0 / avg_ell2).sum(dim=2)  # (n, m)

    # Full prefactor: |Si|^{1/4} |Sj|^{1/4} |bar_S|^{-1/2}
    log_prefactor = (log_det_quarter_X.unsqueeze(1) + log_det_quarter_Y.unsqueeze(0)
                     + log_det_neg_half_avg)

    # Quadratic form: Q_ij = sum_d (x_id - x_jd)^2 / avg_ell2_ijd
    diff = X.unsqueeze(1) - Y.unsqueeze(0)  # (n, m, d)
    Q = (diff ** 2 / avg_ell2).sum(dim=2)   # (n, m)

    K = (sig ** 2) * torch.exp(log_prefactor - 0.5 * Q)

    if is_sym:
        K = 0.5 * (K + K.T)
        diag_mean = torch.mean(torch.clamp(torch.diagonal(K).abs(), min=1e-12))
        rel_jitter = float(jitter) * float(torch.clamp(diag_mean, min=1.0))
        K = K + rel_jitter * torch.eye(n, device=X.device, dtype=X.dtype)

    return K


def _matern52_corr(X, Y, ell):
    """
    Matérn 5/2 correlation matrix (sigma=1, no jitter).
    ell: (d,) lengthscales.
    Returns (n, m) correlation matrix.
    """
    X = _ensure_2d(torch.as_tensor(X, dtype=torch.float32))
    if Y is None:
        Y = X
    else:
        Y = _ensure_2d(torch.as_tensor(Y, dtype=X.dtype, device=X.device))
    ell = torch.as_tensor(ell, dtype=X.dtype, device=X.device).clamp(min=1e-8)
    r = torch.cdist(X / ell.view(1, -1), Y / ell.view(1, -1)).clamp(0.0, 20.0)
    sqrt5_r = math.sqrt(5.0) * r
    R = (1.0 + sqrt5_r + 5.0 / 3.0 * r ** 2) * torch.exp(-sqrt5_r)
    return R


def cgp_cov(tau_left, tau_right=None, ell_g=None, ell_l=None, sigma=1.0,
            lam=0.5, v_a=0.0, v_b=None, jitter: float = 1e-6):
    """
    Composite Gaussian Process kernel (Ba & Joseph, 2012).

    k(x,x') = sigma^2 * [lam * R_g(x,x'; ell_g)
              + (1-lam) * sqrt(v(x)*v(x')) * R_l(x,x'; ell_l)]

    R_g, R_l: Matérn 5/2 correlation with different lengthscales.
    v(x) = exp(v_a + v_b^T x): log-linear spatially-varying variance.
    lam: mixing weight (1 = pure global/stationary, 0 = pure local/nonstationary).

    Parameters
    ----------
    tau_left : (n, d) input locations
    tau_right : (m, d) or None (symmetric case)
    ell_g : (d,) global lengthscales
    ell_l : (d,) local lengthscales
    sigma : signal std
    lam : mixing weight in [0, 1]
    v_a : scalar intercept for log v(x)
    v_b : (d,) slopes for log v(x)
    """
    X = _ensure_2d(torch.as_tensor(tau_left, dtype=torch.float32))
    n, d = X.shape
    sig = torch.as_tensor(sigma, dtype=X.dtype, device=X.device)
    lam = torch.as_tensor(lam, dtype=X.dtype, device=X.device).clamp(1e-6, 1 - 1e-6)
    v_a = torch.as_tensor(v_a, dtype=X.dtype, device=X.device)

    if ell_g is None:
        ell_g = torch.ones(d, dtype=X.dtype, device=X.device)
    else:
        ell_g = torch.as_tensor(ell_g, dtype=X.dtype, device=X.device)
    if ell_l is None:
        ell_l = torch.ones(d, dtype=X.dtype, device=X.device)
    else:
        ell_l = torch.as_tensor(ell_l, dtype=X.dtype, device=X.device)
    if v_b is None:
        v_b = torch.zeros(d, dtype=X.dtype, device=X.device)
    else:
        v_b = torch.as_tensor(v_b, dtype=X.dtype, device=X.device)

    is_sym = tau_right is None
    if is_sym:
        Y = X
    else:
        Y = _ensure_2d(torch.as_tensor(tau_right, dtype=X.dtype, device=X.device))

    # Global correlation
    R_g = _matern52_corr(X, Y, ell_g)

    # Local correlation
    R_l = _matern52_corr(X, Y, ell_l)

    # Spatially-varying variance: v(x) = exp(v_a + v_b^T x)
    log_v_X = v_a + X @ v_b  # (n,)
    log_v_Y = v_a + Y @ v_b  # (m,)
    # sqrt(v(x_i) * v(x_j)) = exp(0.5 * (log_v_i + log_v_j))
    sqrt_vv = torch.exp(0.5 * (log_v_X.unsqueeze(1) + log_v_Y.unsqueeze(0)))  # (n, m)

    # Composite kernel
    K = (sig ** 2) * (lam * R_g + (1 - lam) * sqrt_vv * R_l)

    if is_sym:
        K = 0.5 * (K + K.T)
        diag_mean = torch.mean(torch.clamp(torch.diagonal(K).abs(), min=1e-12))
        rel_jitter = float(jitter) * float(torch.clamp(diag_mean, min=1.0))
        K = K + rel_jitter * torch.eye(n, device=X.device, dtype=X.dtype)

    return K


def _rbf_corr_matrix(X, Y, theta, jitter=0.0):
    """
    RBF correlation matrix: G[i,j] = exp(-sum_d theta_d (x_id - x_jd)^2).
    theta: (d,) inverse squared lengthscales.
    Returns (n, m) or (n, n) matrix.
    """
    X = _ensure_2d(X)
    if Y is None:
        Y = X
    else:
        Y = _ensure_2d(Y)
    sqrt_th = torch.sqrt(theta.clamp(min=1e-12)).view(1, -1)
    dist2 = torch.cdist(X * sqrt_th, Y * sqrt_th).pow(2).clamp(0.0, 80.0)
    G = torch.exp(-dist2)
    if Y is X and jitter > 0:
        G = 0.5 * (G + G.T)
        G = G + jitter * torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
    return G


# Available kernel types
KERNEL_TYPES = ("rbf", "matern52", "poly_rbf", "gibbs", "cgp")


LOG2PI = math.log(2 * math.pi)


def safe_cholesky(K: torch.Tensor, jitter: float = 1e-6, max_tries: int = 8):
    """
    Robust Cholesky with adaptive jitter and SPD fallback.
    Returns (L, used_jitter).
    """
    K = 0.5 * (K + K.T)
    I = torch.eye(K.shape[0], dtype=K.dtype, device=K.device)
    diag = torch.diagonal(K)
    diag_scale = torch.mean(torch.clamp(diag.abs(), min=1e-6))
    j = max(float(jitter) * float(diag_scale), 1e-6)

    for _ in range(max_tries):
        K_try = K + j * I
        if not torch.isfinite(K_try).all():
            j *= 10.0
            continue
        try:
            L = torch.linalg.cholesky(K_try)
            return L, j
        except RuntimeError:
            j *= 10.0

    # Fallback: nearest SPD projection via eigendecomposition
    with torch.no_grad():
        K = 0.5 * (K + K.T)
        try:
            w, V = torch.linalg.eigh(K)
            eps = max(1e-6 * float(diag_scale), -float(torch.min(w)) + 1e-6)
            w_clamped = torch.clamp(w, min=eps)
            K_spd = (V * w_clamped) @ V.T
            K_spd = 0.5 * (K_spd + K_spd.T)
        except RuntimeError:
            # eigh/svd both failed — heavy diagonal regularization
            eps = 0.01 * float(diag_scale)
            K_spd = K + eps * I

    L = torch.linalg.cholesky(K_spd)
    return L, eps


@torch.no_grad()
def loo_stats_from_chol(L: torch.Tensor, y: torch.Tensor):
    """LOO statistics via K^{-1} from Cholesky."""
    n = L.shape[0]
    alpha = torch.cholesky_solve(y, L)
    I = torch.eye(n, dtype=L.dtype, device=L.device)
    Linv = torch.linalg.solve_triangular(L, I, upper=False)
    Kinv_diag = torch.sum(Linv**2, dim=1, keepdim=True)
    mu_loo = y - alpha / Kinv_diag
    sigma2_loo = 1.0 / Kinv_diag
    return mu_loo, sigma2_loo


def nll_map_robust(theta, tau, y, make_kernel, priors=None, add_loo_weight: float = 0.0):
    """
    Negative log-likelihood with MAP prior + optional LOO regularization.

    Normalized per-output-dimension: datafit divided by m to prevent
    extreme gradients when m >> 1 (e.g., m = 12495). logdet and LOO
    log-sigma terms are NOT divided by m (they only sum over n, not m).
    """
    log_ell = theta['log_ell']
    log_sf = theta['log_sigma_f']
    log_sn = theta['log_sigma_n']
    # Pass any extra kernel params (for poly_rbf or gibbs)
    extra = {k: v for k, v in theta.items() if k not in ('log_ell', 'log_sigma_f', 'log_sigma_n')}
    K = make_kernel(log_ell, log_sf, log_sn, **extra)
    L, used_jitter = safe_cholesky(K)
    alpha = torch.cholesky_solve(y, L)
    m = y.shape[1] if y.ndim == 2 else 1
    n = y.shape[0]
    # Per-output-dim normalization: datafit sums over n*m, so divide by m.
    # logdet = sum(log(diag(L))) corresponds to (1/2)*log|K| after the m factor
    # cancels (the full NLL has m*log|K|/2, divided by m = log|K|/2).
    datafit = 0.5 * torch.sum(y * alpha) / m
    logdet = torch.sum(torch.log(torch.diagonal(L)))
    const = 0.5 * n * LOG2PI
    nll = datafit + logdet + const
    prior_pen = 0.0
    if priors is not None:
        for name, (mu, std) in priors.items():
            v = theta[name]
            prior_pen = prior_pen + 0.5 * torch.sum(((v - mu) / std)**2)
    loss = nll + prior_pen
    if add_loo_weight and add_loo_weight > 0:
        with torch.no_grad():
            mu_loo, sigma2_loo = loo_stats_from_chol(L, y)
            # LOO NLL: log(sigma2) sums over n (not m), datafit sums over n*m
            loo_term = 0.5 * torch.sum(torch.log(sigma2_loo)) \
                     + 0.5 * torch.sum((y - mu_loo)**2 / sigma2_loo) / m \
                     + 0.5 * n * LOG2PI
        loss = loss + add_loo_weight * loo_term
    info = dict(nll=float(nll.detach()), jitter=float(used_jitter))
    return loss, info


def nll_concentrated(theta, tau, y, make_kernel, priors=None, add_loo_weight: float = 0.0):
    """
    Concentrated (profile) NLL for matrix-variate normal.

    Profiles out K_Y analytically:  K̂_Y = (1/n) Y^T K_θ^{-1} Y.
    NLL ∝ (r/2) log|K_θ| + (n/2) log|Y^T K_θ^{-1} Y|

    The determinant term log|S| (where S = Y^T K_θ^{-1} Y) is much more
    sensitive to K_θ structure changes than the trace tr(S)/m used in
    nll_map_robust, because it captures ALL eigenvalue ratios of S
    rather than averaging them.

    Requires r ≤ n (PCA-reduced outputs satisfy this naturally).

    Reference: Gu & Berger (2016), "Multi-output separable GP".
    """
    log_ell = theta['log_ell']
    log_sf = theta['log_sigma_f']
    log_sn = theta['log_sigma_n']
    extra = {k: v for k, v in theta.items() if k not in ('log_ell', 'log_sigma_f', 'log_sigma_n')}

    K = make_kernel(log_ell, log_sf, log_sn, **extra)
    L, used_jitter = safe_cholesky(K)

    n = y.shape[0]
    r = y.shape[1] if y.ndim == 2 else 1

    # W = L^{-1} Y  →  S = W^T W = Y^T K_θ^{-1} Y   (r × r)
    W = torch.linalg.solve_triangular(L, y, upper=False)
    S = W.T @ W
    L_S, _ = safe_cholesky(S, jitter=1e-8)

    # Concentrated NLL:  (r/2) log|K_θ| + (n/2) log|S|
    logdet_K = torch.sum(torch.log(torch.diagonal(L)))    # = (1/2) log|K_θ|
    logdet_S = torch.sum(torch.log(torch.diagonal(L_S)))  # = (1/2) log|S|
    nll = r * logdet_K + n * logdet_S

    # Prior penalty (same as nll_map_robust)
    prior_pen = 0.0
    if priors is not None:
        for name, (mu, std) in priors.items():
            v = theta[name]
            prior_pen = prior_pen + 0.5 * torch.sum(((v - mu) / std) ** 2)

    loss = nll + prior_pen

    # Optional LOO regularization (unchanged from nll_map_robust)
    if add_loo_weight and add_loo_weight > 0:
        m = r
        with torch.no_grad():
            mu_loo, sigma2_loo = loo_stats_from_chol(L, y)
            loo_term = 0.5 * torch.sum(torch.log(sigma2_loo)) \
                     + 0.5 * torch.sum((y - mu_loo) ** 2 / sigma2_loo) / m \
                     + 0.5 * n * LOG2PI
        loss = loss + add_loo_weight * loo_term

    info = dict(nll=float(nll.detach()), jitter=float(used_jitter))
    return loss, info


def _default_inits_fp32(tau, y):
    X = tau if tau.ndim == 2 else tau.unsqueeze(0)
    n, d = X.shape
    ell0 = []
    for j in range(d):
        xj = X[:, j]
        diffs = torch.cdist(xj[:, None], xj[:, None], p=2).view(-1)
        diffs = diffs[diffs > 0]
        if diffs.numel() == 0:
            ell0.append(torch.tensor(1.0, dtype=X.dtype, device=X.device))
        else:
            ell0.append(torch.median(diffs) / math.sqrt(2))
    ell0 = torch.stack(ell0)
    ell0 = torch.clamp(ell0, min=1e-3)
    if y.ndim == 1:
        ystd = torch.std(y)
    else:
        ystd = torch.std(y, dim=0).mean()
    sf0 = torch.clamp(ystd, min=1e-3)
    sn0 = 1e-4 * sf0
    init = {'log_ell': torch.log(ell0),
            'log_sigma_f': torch.log(sf0),
            'log_sigma_n': torch.log(sn0)}
    prior_mu = {k: v.clone().detach() for k, v in init.items()}
    return init, prior_mu


def optimize_hypers_robust(
    tau, y, make_kernel, n_restarts: int = 5, add_loo_weight: float = 0.05,
    prior_std_log_ell: float = 1.5, prior_std_log_sf: float = 1.0, prior_std_log_sn: float = 1.0,
    max_iter: int = 200, max_grad_norm: float = 5.0, device=None,
    prev_theta=None, warmstart_shrink: float = 0.5,
    kernel_type: str = "rbf"
):
    """
    float32-friendly MAP+LOO+LBFGS multi-restart.

    Warm-start support (Eriksson et al., 2019, TuRBO):
      If prev_theta is provided, the prior is centered on the previous MAP estimate
      with tighter standard deviations (scaled by warmstart_shrink). This prevents
      wild jumps between modes when N is small. One restart always starts from
      the previous MAP for continuity.

    kernel_type : str
      "rbf" (default), "poly_rbf", or "gibbs"

    Returns: (best_theta_dict, info)
    """
    tau = tau.to(dtype=torch.float32, device=device if device is not None else tau.device)
    y = y.to(dtype=torch.float32, device=tau.device)
    d_input = tau.shape[1] if tau.ndim == 2 else 1
    best = None

    base_init, prior_mu = _default_inits_fp32(tau, y)

    # Add kernel-specific parameters
    if kernel_type == "poly_rbf":
        base_init['log_sigma_p'] = torch.log(torch.clamp(torch.std(y, dim=0).mean(), min=1e-3)).to(tau.device)
        base_init['log_sigma_0'] = torch.tensor(0.0, dtype=torch.float32, device=tau.device)
        base_init['logit_w'] = torch.tensor(-2.0, dtype=torch.float32, device=tau.device)  # sigmoid(-2) ~ 0.12
        prior_mu['log_sigma_p'] = base_init['log_sigma_p'].clone()
        prior_mu['log_sigma_0'] = base_init['log_sigma_0'].clone()
        prior_mu['logit_w'] = base_init['logit_w'].clone()
    elif kernel_type == "gibbs":
        d_ell = base_init['log_ell'].numel()
        base_init['gibbs_a'] = torch.zeros(d_ell, dtype=torch.float32, device=tau.device)
        base_init['gibbs_B'] = torch.zeros(d_ell, d_input, dtype=torch.float32, device=tau.device)
        prior_mu['gibbs_a'] = torch.zeros_like(base_init['gibbs_a'])
        prior_mu['gibbs_B'] = torch.zeros_like(base_init['gibbs_B'])
    elif kernel_type == "cgp":
        # CGP has separate local lengthscales, mixing weight, and v(x) params
        base_init['log_ell_l'] = base_init['log_ell'].clone()  # local ℓ initialized same as global
        base_init['logit_lambda'] = torch.tensor(0.0, dtype=torch.float32, device=tau.device)  # sigmoid(0)=0.5
        base_init['v_a'] = torch.tensor(0.0, dtype=torch.float32, device=tau.device)
        base_init['v_b'] = torch.zeros(d_input, dtype=torch.float32, device=tau.device)
        prior_mu['log_ell_l'] = base_init['log_ell_l'].clone()
        prior_mu['logit_lambda'] = base_init['logit_lambda'].clone()
        prior_mu['v_a'] = base_init['v_a'].clone()
        prior_mu['v_b'] = base_init['v_b'].clone()

    # Warm-start: center prior on previous MAP with tighter std
    if prev_theta is not None:
        for k in base_init.keys():
            if k in prev_theta:
                prior_mu[k] = prev_theta[k].detach().clone().to(dtype=torch.float32, device=tau.device)
        prior_std_log_ell = prior_std_log_ell * warmstart_shrink
        prior_std_log_sf = prior_std_log_sf * warmstart_shrink
        prior_std_log_sn = prior_std_log_sn * warmstart_shrink
        logger.info(f"Warm-start: prior centered on prev MAP, std shrunk by {warmstart_shrink}")

    def _leaf(x: torch.Tensor) -> torch.Tensor:
        return x.detach().clone().requires_grad_(True)

    # Build base keys list
    base_keys = ['log_ell', 'log_sigma_f', 'log_sigma_n']
    if kernel_type == "poly_rbf":
        base_keys += ['log_sigma_p', 'log_sigma_0', 'logit_w']
    elif kernel_type == "gibbs":
        base_keys += ['gibbs_a', 'gibbs_B']
    elif kernel_type == "cgp":
        base_keys += ['log_ell_l', 'logit_lambda', 'v_a', 'v_b']

    def new_theta(shift: bool = False, from_prev: bool = False):
        if from_prev and prev_theta is not None:
            th = {k: _leaf(prev_theta[k].detach().clone().to(dtype=torch.float32, device=tau.device))
                  for k in base_keys if k in prev_theta}
            # Fill missing keys from base_init
            for k in base_keys:
                if k not in th:
                    th[k] = _leaf(base_init[k])
        else:
            th = {k: _leaf(base_init[k]) for k in base_keys}
        if shift:
            th['log_ell']     = _leaf(th['log_ell']     + 0.5 * torch.randn_like(th['log_ell']))
            th['log_sigma_f'] = _leaf(th['log_sigma_f'] + 0.2 * torch.randn_like(th['log_sigma_f']))
            th['log_sigma_n'] = _leaf(th['log_sigma_n'] + 0.5 * torch.randn_like(th['log_sigma_n']))
            if kernel_type == "poly_rbf":
                th['logit_w'] = _leaf(th['logit_w'] + 0.3 * torch.randn_like(th['logit_w']))
            elif kernel_type == "gibbs":
                th['gibbs_B'] = _leaf(th['gibbs_B'] + 0.1 * torch.randn_like(th['gibbs_B']))
            elif kernel_type == "cgp":
                th['log_ell_l'] = _leaf(th['log_ell_l'] + 0.8 * torch.randn_like(th['log_ell_l']))
                th['logit_lambda'] = _leaf(th['logit_lambda'] + 0.5 * torch.randn_like(th['logit_lambda']))
                th['v_a'] = _leaf(th['v_a'] + 0.3 * torch.randn_like(th['v_a']))
                th['v_b'] = _leaf(th['v_b'] + 0.3 * torch.randn_like(th['v_b']))
        return th

    # Build priors (shrinkage toward stationary for non-stationary params)
    priors = {
        'log_ell':     (prior_mu['log_ell'],   torch.full_like(prior_mu['log_ell'], prior_std_log_ell)),
        'log_sigma_f': (prior_mu['log_sigma_f'], torch.tensor(prior_std_log_sf,  dtype=torch.float32, device=tau.device)),
        'log_sigma_n': (prior_mu['log_sigma_n'], torch.tensor(prior_std_log_sn,  dtype=torch.float32, device=tau.device)),
    }
    if kernel_type == "poly_rbf":
        priors['log_sigma_p'] = (prior_mu['log_sigma_p'], torch.tensor(1.0, dtype=torch.float32, device=tau.device))
        priors['log_sigma_0'] = (prior_mu['log_sigma_0'], torch.tensor(1.0, dtype=torch.float32, device=tau.device))
        priors['logit_w'] = (prior_mu['logit_w'], torch.tensor(1.0, dtype=torch.float32, device=tau.device))
    elif kernel_type == "gibbs":
        # Shrinkage prior: a ~ N(0, 0.5), B ~ N(0, 0.3) — encourages stationarity
        priors['gibbs_a'] = (prior_mu['gibbs_a'], torch.full_like(prior_mu['gibbs_a'], 0.5))
        priors['gibbs_B'] = (prior_mu['gibbs_B'], torch.full_like(prior_mu['gibbs_B'], 0.3))
    elif kernel_type == "cgp":
        # CGP priors: shrinkage toward stationarity
        priors['log_ell_l'] = (prior_mu['log_ell_l'], torch.full_like(prior_mu['log_ell_l'], prior_std_log_ell))
        priors['logit_lambda'] = (prior_mu['logit_lambda'], torch.tensor(1.5, dtype=torch.float32, device=tau.device))
        priors['v_a'] = (prior_mu['v_a'], torch.tensor(0.5, dtype=torch.float32, device=tau.device))
        priors['v_b'] = (prior_mu['v_b'], torch.full_like(prior_mu['v_b'], 0.3))  # strong shrinkage toward stationary

    for r in range(n_restarts):
        # First restart: start from previous MAP (warm-start)
        # Others: data-driven init with random perturbation
        # For CGP: second restart always uses fresh data-driven init (escape mode-lock)
        if r == 0 and prev_theta is not None:
            theta = new_theta(shift=False, from_prev=True)
        elif r == 1 and prev_theta is not None and kernel_type == "cgp":
            # CGP escape hatch: ignore previous MAP entirely
            theta = new_theta(shift=False, from_prev=False)
        elif r == 0:
            theta = new_theta(shift=False)
        else:
            theta = new_theta(shift=True)

        params = [theta[k] for k in base_keys]
        opt = torch.optim.LBFGS(params, lr=0.5, max_iter=20, line_search_fn='strong_wolfe')
        last_loss = None
        stall = 0

        _closure_failed = [False]

        # Select NLL function: concentrated likelihood for CGP, standard MAP for others
        _nll_fn = nll_concentrated if kernel_type == "cgp" else nll_map_robust

        def closure():
            opt.zero_grad(set_to_none=True)
            try:
                loss, _ = _nll_fn(theta, tau, y, make_kernel, priors=priors, add_loo_weight=add_loo_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                _closure_failed[0] = False
                return loss
            except (RuntimeError, ValueError):
                # Kernel produced ill-conditioned matrix; return large loss
                _closure_failed[0] = True
                return torch.tensor(1e10, dtype=torch.float32, device=tau.device, requires_grad=False)

        for it in range(max_iter):
            try:
                loss = opt.step(closure)
            except RuntimeError:
                break
            if _closure_failed[0]:
                break
            val = float(loss.detach())
            if last_loss is None or last_loss - val > 1e-6:
                last_loss = val
                stall = 0
            else:
                stall += 1
                if stall >= 5:
                    break

        with torch.no_grad():
            final_loss, info = _nll_fn(theta, tau, y, make_kernel, priors=priors, add_loo_weight=add_loo_weight)
            pack = (float(final_loss), {k: v.detach().clone() for k, v in theta.items()}, info)
            if best is None or pack[0] < best[0]:
                best = pack

    best_loss, best_theta, best_info = best
    return best_theta, best_info
