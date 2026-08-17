# kl_expansion.py — Karhunen-Loève expansion for 2D exponential covariance.
#
# Provides KL eigenpairs for log-permeability parameterization in the Darcy flow
# benchmark. Uses product structure: 2D KL = tensor product of 1D KL modes.
#
# Covariance: C(x,y) = exp(-|x₁-y₁|/ℓ - |x₂-y₂|/ℓ)  (separable exponential)

import numpy as np
from scipy.optimize import brentq


def _solve_kl_1d_eigenvalues(ell, n_terms, L=1.0):
    """Solve transcendental equations for 1D exponential covariance KL on [0, L].

    For C(x,y) = exp(-|x-y|/ℓ), the eigenfunctions are sines and cosines
    satisfying transcendental equations (see e.g. Ghanem & Spanos, 1991):
      - Even modes:  ℓ·ω·tan(ω·L/2) = 1     → φ(x) = cos(ω·(x - L/2))
      - Odd modes:   ℓ·ω·cot(ω·L/2) = -1    → φ(x) = sin(ω·(x - L/2))
    with eigenvalue λ = 2ℓ / (1 + (ω·ℓ)²).

    Returns
    -------
    omegas : (n_terms,) — angular frequencies, sorted by decreasing λ
    lambdas : (n_terms,) — eigenvalues
    parities : (n_terms,) — 0 for even (cos), 1 for odd (sin)
    """
    half_L = L / 2.0
    # We need enough roots; search more than needed
    n_search = n_terms + 5

    omegas_even = []
    omegas_odd = []

    # Even modes: ℓ·ω·tan(ω·L/2) = 1
    for k in range(n_search):
        lo = k * np.pi / half_L + 1e-12
        hi = (k + 0.5) * np.pi / half_L - 1e-12
        if lo >= hi:
            continue
        f = lambda w: ell * w * np.tan(w * half_L) - 1.0
        try:
            if f(lo) * f(hi) < 0:
                w = brentq(f, lo, hi)
                omegas_even.append(w)
        except ValueError:
            pass

    # Odd modes: -ℓ·ω·cot(ω·L/2) = 1, i.e. ℓ·ω / tan(ω·L/2) = -1
    for k in range(n_search):
        lo = (k + 0.5) * np.pi / half_L + 1e-12
        hi = (k + 1) * np.pi / half_L - 1e-12
        if lo >= hi:
            continue
        f = lambda w: -ell * w / np.tan(w * half_L) - 1.0
        try:
            if f(lo) * f(hi) < 0:
                w = brentq(f, lo, hi)
                omegas_odd.append(w)
        except ValueError:
            pass

    # Combine and compute eigenvalues
    all_entries = []
    for w in omegas_even:
        lam = 2.0 * ell / (1.0 + (w * ell) ** 2)
        all_entries.append((lam, w, 0))  # parity=0 → even (cos)
    for w in omegas_odd:
        lam = 2.0 * ell / (1.0 + (w * ell) ** 2)
        all_entries.append((lam, w, 1))  # parity=1 → odd (sin)

    # Sort by decreasing eigenvalue
    all_entries.sort(key=lambda x: -x[0])
    all_entries = all_entries[:n_terms]

    omegas = np.array([e[1] for e in all_entries])
    lambdas = np.array([e[0] for e in all_entries])
    parities = np.array([e[2] for e in all_entries])
    return omegas, lambdas, parities


def _eval_kl_1d_modes(omegas, parities, x, L=1.0):
    """Evaluate normalized 1D KL eigenfunctions at points x.

    Parameters
    ----------
    omegas : (k,) angular frequencies
    parities : (k,) 0=cos, 1=sin
    x : (n,) grid points in [0, L]
    L : domain length

    Returns
    -------
    phi : (n, k) — normalized eigenfunctions
    """
    half_L = L / 2.0
    x_shift = x - half_L  # center at L/2
    k = len(omegas)
    n = len(x)
    phi = np.zeros((n, k))

    for j in range(k):
        w = omegas[j]
        if parities[j] == 0:
            phi[:, j] = np.cos(w * x_shift)
        else:
            phi[:, j] = np.sin(w * x_shift)
        # L2-normalize
        norm = np.sqrt(np.sum(phi[:, j] ** 2) * (L / n))
        if norm > 1e-14:
            phi[:, j] /= norm

    return phi


def build_kl_2d(ell, n_terms, nx, ny, L=1.0):
    """Build 2D KL expansion using tensor product of 1D modes.

    For separable exponential covariance C(x,y) = exp(-‖x-y‖₁/ℓ),
    2D eigenpairs = products of 1D eigenpairs.

    Parameters
    ----------
    ell : float — correlation length
    n_terms : int — number of 2D KL terms to keep
    nx, ny : int — grid resolution per dimension
    L : float — domain [0, L]²

    Returns
    -------
    kl_basis : dict with keys:
        'lambdas' : (n_terms,) eigenvalues (sorted decreasing)
        'modes'   : (nx*ny, n_terms) eigenfunctions on flattened grid
        'x_grid'  : (nx,) x coordinates
        'y_grid'  : (ny,) y coordinates
    """
    # 1D KL solve — get enough terms for tensor product
    n_1d = int(np.ceil(np.sqrt(n_terms))) + 3
    omegas_x, lam_x, par_x = _solve_kl_1d_eigenvalues(ell, n_1d, L)
    omegas_y, lam_y, par_y = _solve_kl_1d_eigenvalues(ell, n_1d, L)

    # Grid (cell centers)
    x_grid = np.linspace(0.5 / nx, L - 0.5 / nx, nx)
    y_grid = np.linspace(0.5 / ny, L - 0.5 / ny, ny)

    # Evaluate 1D modes
    phi_x = _eval_kl_1d_modes(omegas_x, par_x, x_grid, L)  # (nx, n_1d)
    phi_y = _eval_kl_1d_modes(omegas_y, par_y, y_grid, L)  # (ny, n_1d)

    # Build 2D product pairs, sort by λ_i · λ_j
    pairs = []
    for i in range(len(lam_x)):
        for j in range(len(lam_y)):
            pairs.append((lam_x[i] * lam_y[j], i, j))
    pairs.sort(key=lambda t: -t[0])
    pairs = pairs[:n_terms]

    lambdas_2d = np.array([p[0] for p in pairs])
    modes_2d = np.zeros((nx * ny, n_terms))

    for idx, (_, i, j) in enumerate(pairs):
        # Outer product: φ_i(x) ⊗ φ_j(y), flattened in row-major (x varies slowest)
        mode_2d = np.outer(phi_y[:, j], phi_x[:, i]).ravel()  # (ny, nx) → (ny*nx,)
        # Normalize
        norm = np.sqrt(np.sum(mode_2d ** 2) * (L / nx) * (L / ny))
        if norm > 1e-14:
            mode_2d /= norm
        modes_2d[:, idx] = mode_2d

    return {
        'lambdas': lambdas_2d,
        'modes': modes_2d,
        'x_grid': x_grid,
        'y_grid': y_grid,
    }


def sample_log_kappa(xi, kl_basis, mu0=0.0):
    """Evaluate log-permeability field: log κ(x; ξ) = μ₀ + Σᵢ ξᵢ √λᵢ φᵢ(x).

    Parameters
    ----------
    xi : (d,) — KL coefficients
    kl_basis : dict from build_kl_2d
    mu0 : float — mean of log κ

    Returns
    -------
    log_kappa : (nx*ny,) — log-permeability on flattened grid
    """
    xi = np.asarray(xi, dtype=np.float64).ravel()
    d = len(xi)
    lambdas = kl_basis['lambdas'][:d]
    modes = kl_basis['modes'][:, :d]
    return mu0 + modes @ (xi * np.sqrt(lambdas))
