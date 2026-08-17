# manifold.py — Grassmannian geometry: log/exp maps, geodesic/chordal distances
import logging
import torch

logger = logging.getLogger(__name__)


def log_map(reference_pt: torch.Tensor, Phi: torch.Tensor) -> torch.Tensor:
    """
    Logarithmic map from Grassmannian to tangent (horizontal) space.
    reference_pt: (n, p)
    Phi: (n, p) or (M, n, p)
    Returns: tangent vectors Z of same batch shape as Phi.
    """
    was_2d = Phi.ndim == 2
    Phi = Phi.unsqueeze(0) if was_2d else Phi
    reference_pt = reference_pt.unsqueeze(0) if reference_pt.ndim == 2 else reference_pt

    M, n, p = Phi.shape
    reference_pt_rep = reference_pt.expand(M, n, p)
    P = torch.matmul(reference_pt_rep.transpose(1, 2), Phi)    # (M, p, p)
    # Conditioning check: regularize P before inversion
    P_cond = P + 1e-10 * torch.eye(p, device=P.device, dtype=P.dtype).unsqueeze(0)
    P_inv = torch.linalg.inv(P_cond)                            # (M, p, p)
    Z = torch.matmul(Phi, P_inv) - reference_pt_rep             # (M, n, p)
    U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
    S_at = torch.diag_embed(torch.atan(S))                      # (M, r, r)
    Zs = U @ S_at @ Vh                                          # (M, n, p)
    return Zs.squeeze(0) if was_2d else Zs


def exp_map(reference_pt: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """
    Exponential map from tangent space back to Grassmannian.
    reference_pt: (n, p)
    Z: (n, p) or (M, n, p)
    Returns: subspace points on Gr(p,n).
    """
    Zs = Z.unsqueeze(0) if Z.ndim == 2 else Z
    reference_pt = reference_pt.unsqueeze(0) if reference_pt.ndim == 2 else reference_pt

    M, n, p = Zs.shape
    U, S, Vh = torch.linalg.svd(Zs, full_matrices=False)
    cosS = torch.diag_embed(torch.cos(S))
    sinS = torch.diag_embed(torch.sin(S))
    reference_pt_rep = reference_pt.expand(M, n, p)

    term1 = torch.matmul(reference_pt_rep, torch.matmul(Vh.transpose(1, 2), cosS))
    term2 = torch.matmul(U, sinS)
    Phi = term1 + term2

    return Phi.squeeze(0) if Z.ndim == 2 else Phi


def geodesic(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Geodesic distance on Gr(p,n).
    X, Y: both (n, p)
    Returns: scalar distance.
    """
    s = torch.linalg.svdvals(X.T @ Y)
    s = s.clamp(min=0.0, max=1.0 - 1e-12)
    theta = torch.acos(s)
    return torch.linalg.norm(theta)


def geodesic_batch(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Batched geodesic distance on Gr(p,n).
    X: (M, n, p) or (n, p) — broadcasts to match Y
    Y: (M, n, p)
    Returns: (M,) distances.
    """
    if X.ndim == 2:
        X = X.unsqueeze(0).expand_as(Y)
    prod = torch.bmm(X.transpose(1, 2), Y)          # (M, p, p)
    S = torch.linalg.svdvals(prod)                    # (M, p)
    S = S.clamp(0.0, 1.0 - 1e-12)
    theta = torch.acos(S)                             # (M, p)
    return torch.linalg.norm(theta, dim=1)            # (M,)


def chordal(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Chordal distance on Gr(p,n).
    X, Y: (n, p)
    Returns: scalar distance.
    """
    Prod = X.T @ Y
    fro2 = (Prod * Prod).sum()
    p = X.shape[1]
    dc2 = (p - fro2).clamp_min(0.0)
    return torch.sqrt(dc2)


def manifold_var(mean: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Manifold variance = mean of geodesic distances from Karcher mean.
    mean: (n, p), phi: (M, n, p)
    Returns: scalar (mean geodesic distance).
    """
    dists = geodesic_batch(mean, phi)
    return torch.mean(dists)


def manifold_range(phi: torch.Tensor) -> float:
    """
    Maximum pairwise geodesic distance.
    phi: (M, n, p)
    Returns: scalar.
    """
    M, n, p = phi.shape
    phi_inner_prod = phi.transpose(1, 2).unsqueeze(1) @ phi.unsqueeze(0)
    phi_inner_prod = phi_inner_prod + 1e-8 * torch.eye(p, dtype=phi.dtype, device=phi.device).view(1, 1, p, p)
    S = torch.linalg.svdvals(phi_inner_prod.reshape(M * M, p, p)).reshape(M, M, p)
    theta = torch.acos(S.clamp(-1.0, 1.0))
    distance = torch.linalg.norm(theta, dim=-1)
    mask = torch.ones(M, M, dtype=torch.bool, device=phi.device).triu(1)
    distance = distance.masked_fill(~mask, float("-inf"))
    max_val, _ = torch.max(distance.view(-1), dim=0)
    return float(max_val)


def karcher_mean(phi: torch.Tensor, max_iter: int = 200, tol: float = 1e-8,
                 init: torch.Tensor = None, alpha: float = 0.5) -> tuple:
    """
    Compute the Karcher (Fréchet) mean on the Grassmannian Gr(p,n).

    The Karcher mean minimizes the sum of squared geodesic distances:
        mu* = argmin_{mu in Gr(p,n)} sum_i d_G(mu, Phi_i)^2

    Algorithm: iterative tangent-space averaging with exp map steps.
        1. Project all Phi_i to tangent space at current mean
        2. Average tangent vectors
        3. Step along the average direction via exp map (with damping)
        4. Re-orthogonalize to stay on Gr(p,n)

    Parameters
    ----------
    phi : (M, n, p) — training subspaces on Gr(p,n)
    max_iter : int — maximum iterations (default 200)
    tol : float — convergence tolerance on ||Z_mean|| (default 1e-8)
    init : (n, p), optional — initialization. If None, use phi[0].
    alpha : float — step-size damping factor in (0, 1]. Default 0.5.
        Smaller values prevent overshooting when data is spread on Gr(p,n).

    Returns
    -------
    mean : (n, p) — the Karcher mean subspace
    n_iters : int — number of iterations used
    final_norm : float — final ||Z_mean|| norm (lower = better convergence)
    """
    M, n, p = phi.shape
    if init is not None:
        mean = init.clone()
    else:
        mean = phi[0].clone()

    # Ensure initial point is on the Stiefel manifold
    Q, _ = torch.linalg.qr(mean)
    mean = Q[:, :p]

    prev_norm = float('inf')
    diverge_count = 0

    for it in range(max_iter):
        # Map all points to tangent space at current mean
        Z = log_map(mean, phi)          # (M, n, p)
        Z_mean = Z.mean(dim=0)          # (n, p)
        norm = float(torch.linalg.norm(Z_mean))

        if norm < tol:
            logger.info(f"Karcher mean converged in {it+1} iterations (norm={norm:.2e})")
            return mean, it + 1, norm

        # Divergence guard: if norm increases consecutively, break
        if norm > prev_norm:
            diverge_count += 1
            if diverge_count >= 3:
                logger.warning(f"Karcher mean: norm increasing for 3 iterations, stopping at iter {it+1}")
                return mean, it + 1, norm
        else:
            diverge_count = 0
        prev_norm = norm

        # Step along the mean tangent direction (with damping)
        mean = exp_map(mean, alpha * Z_mean)    # (n, p)

        # Re-orthogonalize via QR to prevent drift off Stiefel manifold
        Q, _ = torch.linalg.qr(mean)
        mean = Q[:, :p]

    logger.warning(f"Karcher mean did not converge in {max_iter} iterations (norm={norm:.2e})")
    return mean, max_iter, norm
