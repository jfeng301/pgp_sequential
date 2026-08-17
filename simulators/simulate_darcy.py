#!/usr/bin/env python3
"""Transient diffusion solver with heterogeneous conductivity (Darcy flow benchmark).

Solves:  ∂u/∂t = ∇·(κ(x)∇u)  on Ω = [0,1]², t ∈ [0, T_final]

BCs:  Left u=1 (Dirichlet), Right u=0 (Dirichlet),
      Top/Bottom ∂u/∂n=0 (Neumann)
IC:   u(x,0) = 1 - x₁

Discretization: 5-point finite difference with harmonic averaging at
cell interfaces, implicit Euler time stepping.

Usage:
    python simulate_darcy.py --xi 0.5 -0.3 0.1 0.8 -0.2 --output out.csv
    python simulate_darcy.py --xi-file params.txt --output out.csv
"""

import argparse
import sys
import os
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def build_diffusion_matrix(kappa_field, nx, ny, dx, dy):
    """Build sparse diffusion operator with harmonic averaging at interfaces.

    Parameters
    ----------
    kappa_field : (ny, nx) conductivity on cell centers
    nx, ny : grid dimensions
    dx, dy : cell spacings

    Returns
    -------
    A : (N, N) sparse CSC matrix, where N = nx*ny
        Implements -∇·(κ∇u) for interior + Neumann BCs.
        Dirichlet BCs handled separately.
    """
    N = nx * ny

    def idx(i, j):
        """Map (row=j, col=i) to flat index. j=0..ny-1 (y), i=0..nx-1 (x)."""
        return j * nx + i

    rows, cols, vals = [], [], []

    for j in range(ny):
        for i in range(nx):
            k = idx(i, j)
            diag_val = 0.0

            # East (i+1, j)
            if i < nx - 1:
                # Harmonic average
                kh = 2.0 * kappa_field[j, i] * kappa_field[j, i + 1] / (
                    kappa_field[j, i] + kappa_field[j, i + 1]
                )
                coeff = kh / dx**2
                rows.append(k)
                cols.append(idx(i + 1, j))
                vals.append(-coeff)
                diag_val += coeff

            # West (i-1, j)
            if i > 0:
                kh = 2.0 * kappa_field[j, i] * kappa_field[j, i - 1] / (
                    kappa_field[j, i] + kappa_field[j, i - 1]
                )
                coeff = kh / dx**2
                rows.append(k)
                cols.append(idx(i - 1, j))
                vals.append(-coeff)
                diag_val += coeff

            # North (i, j+1)
            if j < ny - 1:
                kh = 2.0 * kappa_field[j, i] * kappa_field[j + 1, i] / (
                    kappa_field[j, i] + kappa_field[j + 1, i]
                )
                coeff = kh / dy**2
                rows.append(k)
                cols.append(idx(i, j + 1))
                vals.append(-coeff)
                diag_val += coeff

            # South (i, j-1)
            if j > 0:
                kh = 2.0 * kappa_field[j, i] * kappa_field[j - 1, i] / (
                    kappa_field[j, i] + kappa_field[j - 1, i]
                )
                coeff = kh / dy**2
                rows.append(k)
                cols.append(idx(i, j - 1))
                vals.append(-coeff)
                diag_val += coeff

            # Neumann BCs: no-flux at top/bottom → no ghost node needed,
            # just skip the missing neighbor (already done by the if-checks).
            # Similarly at left/right boundaries, but those get Dirichlet below.

            rows.append(k)
            cols.append(k)
            vals.append(diag_val)

    A = sparse.csc_matrix((vals, (rows, cols)), shape=(N, N))
    return A


def solve_darcy_transient(xi, kl_basis, nx=50, ny=50, T_final=2.0, dt=0.02, mu0=0.0):
    """Solve transient diffusion for given KL coefficients.

    Parameters
    ----------
    xi : (d,) KL coefficients
    kl_basis : dict from build_kl_2d
    nx, ny : spatial grid
    T_final : final time
    dt : time step
    mu0 : mean log-permeability

    Returns
    -------
    U : (nx*ny, n_steps) solution snapshots (excludes t=0)
    """
    from core.kl_expansion import sample_log_kappa

    xi = np.asarray(xi, dtype=np.float64).ravel()
    log_kappa = sample_log_kappa(xi, kl_basis, mu0)
    kappa = np.exp(log_kappa).reshape(ny, nx)

    L = 1.0
    dx = L / nx
    dy = L / ny
    n_steps = int(round(T_final / dt))
    N = nx * ny

    # Build diffusion matrix
    A_diff = build_diffusion_matrix(kappa, nx, ny, dx, dy)

    # Implicit Euler: (I + dt*A) u^{n+1} = u^n + dt*b
    # where b incorporates Dirichlet BCs
    M = sparse.eye(N, format='csc') + dt * A_diff

    # Dirichlet BCs: Left (i=0) → u=1, Right (i=nx-1) → u=0
    left_nodes = [j * nx for j in range(ny)]          # i=0
    right_nodes = [j * nx + (nx - 1) for j in range(ny)]  # i=nx-1
    bc_nodes = left_nodes + right_nodes

    # Use lil_matrix for efficient row modification, then convert back
    M = M.tolil()
    for k in bc_nodes:
        M[k, :] = 0
        M[k, k] = 1.0
    M = M.tocsc()
    lu = splu(M)

    # Initial condition: u(x,0) = 1 - x₁
    x_grid = kl_basis['x_grid']  # (nx,) cell centers
    u = np.tile(1.0 - x_grid, ny)  # repeat for each row

    # RHS vector for Dirichlet BCs
    U_out = np.zeros((N, n_steps))

    for step in range(n_steps):
        rhs = u.copy()
        # Enforce Dirichlet in RHS
        for k in left_nodes:
            rhs[k] = 1.0
        for k in right_nodes:
            rhs[k] = 0.0

        u = lu.solve(rhs)
        U_out[:, step] = u

    return U_out


def main():
    parser = argparse.ArgumentParser(description="Darcy transient diffusion solver")
    parser.add_argument("--xi", type=float, nargs='+', help="KL coefficients")
    parser.add_argument("--xi-file", type=str, help="File with KL coefficients (one per line)")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--nx", type=int, default=50)
    parser.add_argument("--ny", type=int, default=50)
    parser.add_argument("--T-final", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--ell", type=float, default=0.3, help="KL correlation length")
    parser.add_argument("--n-kl", type=int, default=5, help="Number of KL terms")
    parser.add_argument("--mu0", type=float, default=0.0)
    args = parser.parse_args()

    if args.xi is not None:
        xi = np.array(args.xi)
    elif args.xi_file is not None:
        xi = np.loadtxt(args.xi_file)
    else:
        parser.error("Provide --xi or --xi-file")

    from core.kl_expansion import build_kl_2d
    kl_basis = build_kl_2d(args.ell, args.n_kl, args.nx, args.ny)

    U = solve_darcy_transient(xi, kl_basis, args.nx, args.ny, args.T_final, args.dt, args.mu0)
    np.savetxt(args.output, U, delimiter=",")
    print(f"Saved {U.shape} to {args.output}")


if __name__ == "__main__":
    main()
