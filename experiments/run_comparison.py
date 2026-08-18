#!/usr/bin/env python3
"""
Run all methods (Proposed pGP, MaxPro, Greedy CG, Random) and generate comparison plots.

Usage:

    # Run all methods in parallel (default 3 workers):
    python3 experiments/run_comparison.py

    # Customize:
    python3 experiments/run_comparison.py --workers 5 --steps 30

    # Plot from existing logs without re-running:
    python3 experiments/run_comparison.py --plot-only

    # Interrupt with Ctrl+C at any time — partial results will be plotted.

GP Model Consistency (controlled variable):
    All methods use core/projectGP.py for GP fitting and manifold prediction.
    The R bridge (methods/hrk_r_bridge.py) is ONLY used in proposed_hrk for
    the variance component of its acquisition function — it fits heteroskedastic
    rational kriging (HRK) on K_Y principal component scores to estimate variance.
    This is the intended comparison: pGP manifold variance vs HRK variance,
    within the same projected GP manifold framework.
"""
import os, sys, csv, math, time, json, logging, pathlib, subprocess
import signal, argparse, threading
import numpy as np

# Setup paths
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# Also add project root (parent of revision/) for cgp_ktheta package
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from pgp.projectGP import projectGP
from pgp.data import data_normalizer as _data_normalizer_factory
from pgp.manifold import geodesic_batch, manifold_range, geodesic
from experiments.acquisition import AcquisitionConfig, build_acquisition, _normalize_to_unit

log = logging.getLogger("experiment")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

# Global stop event for graceful interruption
_STOP_EVENT = None  # Set to threading.Event() in main()

# ====================== CONFIG ======================
EXP_DIR    = pathlib.Path(__file__).resolve().parent
PRECOMP    = EXP_DIR / "precomp.rds"
SIM_R      = ROOT / "simulators" / "simulate_advecdiff.R"
SIM_PRECOMP_R = ROOT / "simulators" / "simulate_precompute.R"
RSCRIPT    = os.environ.get("RSCRIPT_BIN") or "Rscript"

# Greedy CG R scripts
R_ADJ      = ROOT / "simulators" / "fom_rom_adjoint.R"

PROBLEM    = "advecdiff"  # "advecdiff" or "darcy"
DX_BOUNDS  = (1e-4, 1e-2)
DY_BOUNDS  = (1e-4, 1e-2)
LOG_SPACE  = False
N_INIT     = 5
N_STEPS    = 50
N_CAND     = 1024
POD_MODES  = 5
MC_SAMPLES = 100
SEED       = 0        # neutral default; set via --seed or config for real runs
MC_SEED    = 0        # MC sampling seed (separate from SEED for stability tests)
# FIX: T=50 now, no downsampling needed
T_KEEP     = None

# Darcy-specific globals (set by config_darcy.yaml)
KL_BASIS   = None     # Built once in main(), used by simulate_darcy
XI_BOUNDS  = None     # Tuple of d identical (lo, hi) bounds

SMOOTH_H   = 0.05
RESTART_INTERVAL = 10   # Cold restart every N steps (0 = never)
WEIGHTED_GP = False      # Weighted GP: downweight high tangent-norm points
DARCY_T_FINAL = 2.0      # Darcy: simulation end time
DARCY_DT = 0.02          # Darcy: simulation time step

# Kernel type: "rbf", "matern52" (recommended), "poly_rbf", "gibbs"
KERNEL     = "matern52"

# Greedy CG parameters
GREEDY_RESTARTS = 8
GREEDY_OPT_STEPS = 30

# Test grid for projection RMSE evaluation
N_TEST_GRID = 8        # 8x8 regular grid = 64 points (includes boundaries)
N_TEST_RAND = 36       # additional random points for robust evaluation
# ====================================================


def get_bounds():
    """Return domain bounds as tuple of (lo, hi) per dimension."""
    if PROBLEM == "darcy":
        return XI_BOUNDS
    return (DX_BOUNDS, DY_BOUNDS)


def get_param_dim():
    """Return number of input dimensions."""
    if PROBLEM == "darcy":
        return len(XI_BOUNDS)
    return 2


def ensure_precompute():
    """Run R precomputation if precomp.rds doesn't exist (advecdiff only)."""
    if PROBLEM == "darcy":
        return  # Darcy uses Python solver, no R precompute
    if PRECOMP.exists():
        log.info(f"Found precompute: {PRECOMP}")
        return
    log.info("Running R precomputation (T=50)...")
    cmd = [RSCRIPT, str(SIM_PRECOMP_R), str(PRECOMP)]
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if cp.returncode != 0:
        log.error(f"Precompute failed: {cp.stderr}")
        raise RuntimeError("R precompute failed")
    log.info(f"Saved: {PRECOMP}")


def _param_filename(params):
    """Generate a filename from parameter vector."""
    if PROBLEM == "darcy":
        parts = "_".join(f"xi{i}={params[i]:.6g}" for i in range(len(params)))
        return f"{parts}.csv"
    return f"Dx={params[0]:.6g}_Dy={params[1]:.6g}.csv"


def simulate(params, snap_dir):
    """Simulate FOM for a parameter point. Returns path to CSV output.

    For advecdiff: params = [Dx, Dy], calls R script.
    For darcy: params = [xi1, ..., xi5], calls Python solver.
    """
    params = np.asarray(params).ravel()
    out = snap_dir / _param_filename(params)
    if out.exists():
        return out

    if PROBLEM == "darcy":
        from simulators.simulate_darcy import solve_darcy_transient
        U = solve_darcy_transient(params, KL_BASIS, T_final=DARCY_T_FINAL, dt=DARCY_DT)
        np.savetxt(str(out), U, delimiter=",")
    else:
        Dx, Dy = float(params[0]), float(params[1])
        cmd = [RSCRIPT, str(SIM_R), str(PRECOMP), f"{Dx:.12g}", f"{Dy:.12g}", str(out)]
        if T_KEEP is not None:
            cmd.append(str(int(T_KEEP)))
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if cp.returncode != 0:
            log.error(f"R sim failed: {cp.stderr}")
            raise RuntimeError(f"Simulation failed for ({Dx}, {Dy})")
    return out


def sobol_candidates(n, bounds, logspace, seed_offset=7):
    """Generate Sobol quasi-random candidates (arbitrary dimension)."""
    d = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    try:
        from scipy.stats.qmc import Sobol
        eng = Sobol(d=d, scramble=True, seed=SEED + seed_offset)
        U = eng.random(n)
        if logspace:
            log_lo = np.log10(np.maximum(lo, 1e-30))
            log_hi = np.log10(np.maximum(hi, 1e-30))
            return 10 ** (log_lo + U * (log_hi - log_lo))
        return lo + U * (hi - lo)
    except ImportError:
        rng = np.random.default_rng(SEED + seed_offset)
        if logspace:
            log_lo = np.log10(np.maximum(lo, 1e-30))
            log_hi = np.log10(np.maximum(hi, 1e-30))
            return 10 ** rng.uniform(log_lo, log_hi, (n, d))
        return rng.uniform(lo, hi, (n, d))


def lhs_init(n, bounds, logspace, seed):
    """Latin Hypercube Sampling initialization (arbitrary dimension)."""
    rng = np.random.default_rng(seed)
    d = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    def lhs_1d(lo_val, hi_val, m):
        grid = (np.arange(m) + rng.random(m)) / m
        rng.shuffle(grid)
        return lo_val + (hi_val - lo_val) * grid

    cols = []
    for j in range(d):
        if logspace:
            vals = 10 ** lhs_1d(np.log10(max(lo[j], 1e-30)), np.log10(max(hi[j], 1e-30)), n)
        else:
            vals = lhs_1d(lo[j], hi[j], n)
        cols.append(vals)
    return np.column_stack(cols)


def gaussian_smoother(X, v, h):
    D2 = ((X[:, None, :] - X[None, :, :])**2).sum(axis=2)
    K = np.exp(-0.5 * D2 / max(h * h, 1e-24))
    denom = K.sum(axis=1, keepdims=True) + 1e-12
    return (K @ v.reshape(-1, 1) / denom).ravel()


def _extract_extra_kernel_params(theta, kernel):
    """Extract extra kernel parameters (gibbs, poly_rbf) from theta dict."""
    extra = {}
    raw = theta.get('_raw_theta', {})
    if not raw:
        return extra
    if kernel == "gibbs" and 'gibbs_a' in raw and 'gibbs_B' in raw:
        extra['ls_params'] = {'a': raw['gibbs_a'], 'B': raw['gibbs_B']}
    elif kernel == "poly_rbf":
        if 'log_sigma_p' in raw:
            extra['sigma_p'] = torch.exp(raw['log_sigma_p'])
        if 'log_sigma_0' in raw:
            extra['sigma_0'] = torch.exp(raw['log_sigma_0'])
        if 'logit_w' in raw:
            extra['mix_w'] = torch.sigmoid(raw['logit_w'])
    elif kernel == "cgp":
        if 'log_ell_l' in raw:
            extra['ell_l'] = torch.exp(raw['log_ell_l'])
        if 'logit_lambda' in raw:
            extra['lam'] = torch.sigmoid(raw['logit_lambda'])
        if 'v_a' in raw:
            extra['v_a'] = raw['v_a']
        if 'v_b' in raw:
            extra['v_b'] = raw['v_b']
        extra['ell_g'] = torch.exp(raw.get('log_ell', torch.tensor(0.0)))
    return extra


def build_gp(tau, imgs, prev_raw_theta=None, kernel=None, ref_method=None,
             reuse_theta=None, fixed_reference_pt=None, pilot_imgs=None,
             step=None, restart_interval=10, weighted=False,
             opt_style=None):
    """Build pGP model with warm-start support.

    Uses LogStandardizer when LOG_SPACE=True, ZScoreNormalizer when LOG_SPACE=False.

    Parameters
    ----------
    kernel : str, optional
        Kernel type: "rbf", "matern52", "poly_rbf", or "gibbs". Defaults to module-level KERNEL.
    ref_method : str, optional
        Reference point method: "stacked_svd" (default) or "karcher".
    reuse_theta : dict, optional
        If given, skip hyperparameter estimation and use these parameters directly.
        Must contain 'alpha', 'sigma', 'sigma_n' keys.
        This is ~10-30× faster than estimating hyperparameters.
    fixed_reference_pt : tensor, optional
        If given, use this fixed Grassmannian reference point instead of
        computing from data.  Keeps tangent space consistent across steps.
    pilot_imgs : array-like, optional
        Additional snapshots used only for computing the reference point
        (not for GP training).  Provides better domain coverage.
    step : int, optional
        Current sequential design step (1-based). Used for periodic cold restart.
    restart_interval : int
        Every restart_interval steps, do a cold restart (no warm-start, more restarts).
    """
    if kernel is None:
        kernel = KERNEL
    norm_kind = "logz" if LOG_SPACE else "zscore"
    scaler = _data_normalizer_factory(kind=norm_kind).fit(tau)
    tau_std = scaler.transform(tau)
    if kernel == "cgp":
        from pgp.projectGP import projectGP as projectGP_cgp
        gp = projectGP_cgp(pod_modes=POD_MODES, device='cpu', kernel=kernel,
                            reference_pt_method=ref_method or "stacked_svd",
                            cgp_pca_rank=10)
    else:
        gp = projectGP(pod_modes=POD_MODES, device='cpu', kernel=kernel,
                        reference_pt_method=ref_method or "stacked_svd")
    gp.fit(torch.as_tensor(tau_std, dtype=torch.float32),
           torch.as_tensor(imgs, dtype=torch.float32),
           fixed_reference_pt=fixed_reference_pt,
           pilot_imgs=pilot_imgs)
    if reuse_theta is not None:
        # Fast path: reuse hyperparameters from a previous GP build
        extra = _extract_extra_kernel_params(reuse_theta, kernel)
        gp.set_params(ls=reuse_theta['alpha'], sigma=reuse_theta['sigma'],
                      sigma_n=reuse_theta['sigma_n'], extra_kernel_params=extra)
        theta = reuse_theta
    elif opt_style == "rstyle" and kernel == "cgp":
        # R-style CGP optimization: LHD screening + IRLS + box constraints
        log.info(f"[CGP R-style] Using R-style optimization (LHD + IRLS)")
        theta = gp.estimate_hypers_cgp(
            prev_theta=prev_raw_theta, weighted=weighted)
        extra = _extract_extra_kernel_params(theta, kernel)
        gp.set_params(ls=theta['alpha'], sigma=theta['sigma'],
                      sigma_n=theta['sigma_n'], extra_kernel_params=extra)
    else:
        # CGP has 10 params → multi-modal likelihood, needs more aggressive
        # cold restarts and relaxed warm-start to avoid mode-locking
        is_cgp = (kernel == "cgp")
        cgp_restart_interval = max(restart_interval // 2, 3)  # e.g. 5 instead of 10
        eff_interval = cgp_restart_interval if is_cgp else restart_interval
        cold = (step is not None and eff_interval > 0
                and step % eff_interval == 0)
        if cold:
            log.info(f"[Step {step}] COLD RESTART: n_restarts=20, no warm-start")
            n_r, ws, prt = 20, 1.0, None
        elif is_cgp:
            # CGP warm-start: relax prior (0.8 vs 0.5) + more restarts (8 vs 5)
            n_r, ws, prt = 8, 0.8, prev_raw_theta
        else:
            n_r, ws, prt = 5, 0.5, prev_raw_theta
        theta = gp.estimate_hypers(n_restarts=n_r, add_loo_weight=0.3,
                                    prev_theta=prt, warmstart_shrink=ws,
                                    weighted=weighted)
        extra = _extract_extra_kernel_params(theta, kernel)
        gp.set_params(ls=theta['alpha'], sigma=theta['sigma'],
                      sigma_n=theta['sigma_n'], extra_kernel_params=extra)
    return gp, scaler, theta


def make_mc_generator(step):
    """Create a torch.Generator seeded for MC sampling at a given step.

    Uses MC_SEED + step so MC randomness is isolated from other randomness
    (initial design, test grid, candidates all use SEED instead).
    """
    gen = torch.Generator(device='cpu')
    gen.manual_seed(MC_SEED + step)
    return gen


def compute_manifold_var(gp, scaler, cands, generator=None):
    cands_std = scaler.transform(cands)
    tau_t = torch.as_tensor(cands_std, dtype=torch.float32)
    _, var_phi = gp.predict_phi(tau_t, n_samples=MC_SAMPLES, method='empirical_bayes',
                                generator=generator)
    return var_phi.detach().numpy()


def compute_euclidean_var(gp, scaler, cands, generator=None):
    """Euclidean GP variance baseline (ALM).

    Since Sigma(theta) = k_var(theta) * K_Y and K_Y is shared across all
    candidates, ranking by tr(Sigma) or max(diag(Sigma)) is equivalent to
    ranking by the scalar GP posterior variance k_var(theta).
    No MC sampling needed — much faster than manifold variance.
    """
    cands_std = scaler.transform(cands)
    tau_t = torch.as_tensor(cands_std, dtype=torch.float32)
    _, var = gp.predict(tau_t)          # scalar posterior variance per point
    return var.detach().numpy()


def compute_diversity_bonus(gp, scaler, cands, batch_size=128):
    """Compute geodesic diversity bonus for each candidate.

    For each candidate x, predicts the subspace Phi_pred(x) via GP mean,
    then computes the minimum geodesic distance to all training subspaces:
        diversity(x) = min_i  d_G(Phi_pred(x), Phi_train_i)

    Candidates whose predicted subspace is far from all training subspaces
    get a higher bonus, directly optimizing for manifold range.

    Returns normalized diversity scores in [0, 1].
    """
    cands_std = scaler.transform(cands)
    n_cand = len(cands_std)
    phi_train = gp.phi  # (k, n, p)
    k = phi_train.shape[0]

    min_dists = np.zeros(n_cand)
    for start in range(0, n_cand, batch_size):
        end = min(start + batch_size, n_cand)
        tau_batch = torch.as_tensor(cands_std[start:end], dtype=torch.float32)
        mu_y, _ = gp.predict(tau_batch)
        phi_pred = gp.bijective_map(mu_y)  # (batch, n, p)
        if phi_pred.ndim == 2:
            phi_pred = phi_pred.unsqueeze(0)
        B = phi_pred.shape[0]

        # Compute min geodesic to all training subspaces
        batch_min = np.full(B, np.inf)
        for j in range(k):
            # geodesic_batch expects both args to have batch dim or X to be (n,p)
            # phi_train[j] is (n, p) -> expand to (B, n, p) for batch comparison
            phi_j = phi_train[j].unsqueeze(0).expand(B, -1, -1)
            d = geodesic_batch(phi_pred, phi_j)  # (B,)
            d_np = d.detach().numpy()
            batch_min = np.minimum(batch_min, d_np)
        min_dists[start:end] = batch_min

    # Normalize to [0, 1]
    dmax = min_dists.max()
    if dmax > 1e-12:
        return min_dists / dmax
    return np.ones(n_cand)


def _param_col_names():
    """Return CSV column names for parameters."""
    if PROBLEM == "darcy":
        return [f"xi{i}" for i in range(get_param_dim())]
    return ["Dx", "Dy"]


def load_training(meta_csv):
    rows = []
    with open(meta_csv) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    snaps, tau = [], []
    pcols = _param_col_names()
    for r in rows:
        s = np.loadtxt(r["snapshot_csv"], delimiter=",")
        if s.ndim == 1: s = s.reshape(-1, 1)
        snaps.append(s)
        tau.append([float(r[c]) for c in pcols])
    t_max = max(s.shape[1] for s in snaps)
    snaps = [s if s.shape[1] == t_max else np.pad(s, ((0, 0), (0, t_max - s.shape[1])), mode="edge") for s in snaps]
    return np.array(tau), np.stack(snaps)


def append_meta(meta_csv, params, snap_path):
    """Append a row to the training metadata CSV.

    params: array-like of parameter values (2D for advecdiff, 5D for darcy).
    """
    params = np.asarray(params).ravel()
    pcols = _param_col_names()
    fieldnames = pcols + ["snapshot_csv"]
    is_new = not meta_csv.exists()
    with open(meta_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new: w.writeheader()
        row = {c: float(params[i]) for i, c in enumerate(pcols)}
        row["snapshot_csv"] = str(snap_path)
        w.writerow(row)


def maxpro_criterion(x, existing, eps=1e-6):
    if existing.size == 0: return 0.0
    diffs = np.abs(existing - x.reshape(1, -1)) + eps
    return float(np.sum(1.0 / np.prod(diffs, axis=1)**2))


def _extract_hyper_info(theta, gp):
    """Extract loggable hyperparameter summary from theta dict and GP state."""
    raw = theta.get('_raw_theta', {})
    alpha = theta.get('alpha')
    info = {}
    try:
        if alpha is not None:
            if hasattr(alpha, 'tolist'):
                info['lengthscales'] = (1.0 / torch.sqrt(alpha)).tolist()
            else:
                info['lengthscales'] = [float(1.0 / alpha**0.5)]
        info['sigma_f'] = float(theta.get('sigma', float('nan')))
        info['sigma_n'] = float(theta.get('sigma_n', float('nan')))
        if 'gibbs_a' in raw:
            info['gibbs_a'] = raw['gibbs_a'].detach().tolist()
        if 'gibbs_B' in raw:
            info['gibbs_B'] = raw['gibbs_B'].detach().tolist()
        if hasattr(gp, '_ky_lam') and gp._ky_lam is not None:
            lam = gp._ky_lam.detach()
            info['ky_eigenvalues'] = lam.tolist()
            info['ky_top5'] = lam[:5].tolist()  # kept for backward compat
            info['ky_energy_ratio'] = float(lam[0] / lam.sum()) if lam.sum() > 0 else float('nan')
    except Exception as e:
        log.warning(f"_extract_hyper_info failed: {e}")
    return info


def _extract_karcher_ball_info(gp):
    """Extract Karcher convergence and ball constraint info from fitted GP."""
    info = {}
    try:
        info['karcher_iters'] = getattr(gp, 'karcher_iters', -1)
        info['karcher_final_norm'] = getattr(gp, 'karcher_final_norm', float('nan'))
        norms, violations = gp.check_ball_constraint(warn=False)
        info['ball_violations'] = len(violations)
        info['max_tangent_norm'] = float(norms.max()) if len(norms) > 0 else float('nan')
        info['tangent_norms'] = norms.tolist()
    except Exception as e:
        log.warning(f"_extract_karcher_ball_info failed: {e}")
        info.setdefault('karcher_iters', -1)
        info.setdefault('ball_violations', 0)
        info.setdefault('max_tangent_norm', float('nan'))
    return info


def evaluate_mmv(tau, imgs, cands_raw, prev_raw_theta=None, generator=None):
    """Evaluate max manifold variance for comparison metrics."""
    mmv = float('nan')
    raw_theta = None
    try:
        gp, scaler, theta = build_gp(tau, imgs, prev_raw_theta=prev_raw_theta)
        raw_theta = theta.get('_raw_theta', None)
        v = compute_manifold_var(gp, scaler, cands_raw, generator=generator)
        base = _normalize_to_unit(cands_raw, get_bounds(), log_space=LOG_SPACE)
        v_s = gaussian_smoother(base, v, SMOOTH_H)
        mmv = float(np.max(v_s))
    except Exception as e:
        log.warning(f"MMV eval failed: {e}")
    return mmv, raw_theta


def _simulate_batch(tau_pts, snap_dir):
    """Simulate FOM for a batch of parameter points and return stacked snapshots.

    Parameters
    ----------
    tau_pts : (N, d) parameter points
    snap_dir : Path to snapshot directory

    Returns
    -------
    imgs : (N, N_spatial, T) stacked FOM snapshots
    """
    snap_dir.mkdir(parents=True, exist_ok=True)
    snapshots = []
    for pt in tau_pts:
        csv_path = simulate(pt, snap_dir)
        s = np.loadtxt(str(csv_path), delimiter=",")
        if s.ndim == 1:
            s = s.reshape(-1, 1)
        snapshots.append(s)
    t_max = max(s.shape[1] for s in snapshots)
    snapshots = [
        s if s.shape[1] == t_max else np.pad(s, ((0, 0), (0, t_max - s.shape[1])), mode="edge")
        for s in snapshots
    ]
    return np.stack(snapshots)


def generate_sobol_test_grid(snap_dir, init_tau, init_imgs):
    """Generate test grid for high-dimensional problems (d >= 3) using Sobol sequence.

    For d=5 with N_TEST_GRID=3: 3^5=243 grid + N_TEST_RAND random = 300 total.
    Uses Sobol quasi-random sequence for space-filling coverage.
    """
    snap_dir.mkdir(parents=True, exist_ok=True)
    d = get_param_dim()
    n_test = N_TEST_GRID ** d + N_TEST_RAND
    log.info(f"Generating Sobol test grid: {n_test} points ({d}D)")
    grid = sobol_candidates(n_test, get_bounds(), LOG_SPACE, seed_offset=9999)
    test_imgs = _simulate_batch(grid, snap_dir)
    log.info(f"Test grid ready: shape {test_imgs.shape}")
    return grid, test_imgs


def generate_adaptive_test_grid(snap_dir, init_tau, init_imgs):
    """Generate test points concentrated where GP prediction varies most.

    For Darcy (d>=3), falls back to Sobol (adaptive gradient estimation
    is unreliable in high dimensions with sparse data).

    Fits an initial GP on the seed points and estimates the gradient of
    the predicted tangent-vector norm ||z*|| across the parameter domain.
    Test points are sampled with density proportional to the gradient
    magnitude, so regions where the function changes rapidly get more
    test coverage.

    For d=2: uses a structured 30×30 fine grid + np.gradient.
    For d>=3: uses Sobol candidates + KNN-based gradient estimation.

    Anchor points guarantee baseline coverage:
        d=2: 4 corners + center = 5 anchors
        d>=3: center + 2d axis extremes = 1+2d anchors

    Parameters
    ----------
    snap_dir : Path for test FOM snapshots
    init_tau : (n_init, d) initial design parameters
    init_imgs : (n_init, N_spatial, T) initial FOM snapshots
    """
    # Darcy 5D: use Sobol test grid (adaptive gradient estimation not beneficial)
    if PROBLEM == "darcy":
        return generate_sobol_test_grid(snap_dir, init_tau, init_imgs)

    snap_dir.mkdir(parents=True, exist_ok=True)
    d = get_param_dim()
    bounds = get_bounds()
    if d == 2:
        n_test = N_TEST_GRID * N_TEST_GRID + N_TEST_RAND
    else:
        n_test = N_TEST_GRID ** d + N_TEST_RAND

    # --- Step 1: Fit initial GP on seed points ---
    log.info(f"Adaptive test grid ({d}D): fitting initial GP for gradient estimation...")
    gp, scaler, _ = build_gp(init_tau, init_imgs)

    # --- Step 2: Generate fine candidate grid ---
    if d == 2:
        n_fine = 30
        if LOG_SPACE:
            dx_vals = np.logspace(np.log10(DX_BOUNDS[0]), np.log10(DX_BOUNDS[1]), n_fine)
            dy_vals = np.logspace(np.log10(DY_BOUNDS[0]), np.log10(DY_BOUNDS[1]), n_fine)
        else:
            dx_vals = np.linspace(DX_BOUNDS[0], DX_BOUNDS[1], n_fine)
            dy_vals = np.linspace(DY_BOUNDS[0], DY_BOUNDS[1], n_fine)
        fine_grid = np.array([[dx, dy] for dx in dx_vals for dy in dy_vals])  # (900, 2)
    else:
        n_fine = min(2048, max(512, n_test * 4))
        fine_grid = sobol_candidates(n_fine, bounds, LOG_SPACE, seed_offset=8888)
        log.info(f"  Sobol fine grid: {n_fine} candidates in {d}D")

    # --- Step 3: Predict tangent vector norms on fine grid ---
    fine_std = scaler.transform(fine_grid)
    fine_t = torch.as_tensor(fine_std, dtype=torch.float32)
    mu_y, _ = gp.predict(fine_t)

    # ||z*|| measures how different the predicted subspace is from the reference
    z_norm = torch.linalg.norm(mu_y, dim=1).detach().numpy()

    # --- Step 4: Compute gradient magnitude ---
    if d == 2:
        # Structured grid: use np.gradient for finite differences
        z_grid = z_norm.reshape(n_fine, n_fine)
        grad_x, grad_y = np.gradient(z_grid)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2).ravel()
    else:
        # High-d: KNN-based gradient estimation
        from scipy.spatial import cKDTree
        k = min(2 * d, 15)  # k=10 for d=5
        tree = cKDTree(fine_grid)
        dists, idxs = tree.query(fine_grid, k=k + 1)  # +1 for self
        grad_mag = np.zeros(len(fine_grid))
        for i in range(len(fine_grid)):
            neighbor_dists = dists[i, 1:]  # exclude self
            neighbor_z = z_norm[idxs[i, 1:]]
            fd = np.abs(z_norm[i] - neighbor_z) / np.maximum(neighbor_dists, 1e-12)
            grad_mag[i] = np.max(fd)
        log.info(f"  KNN gradient: k={k}, grad_mag range [{grad_mag.min():.4f}, {grad_mag.max():.4f}]")

    # --- Step 5: Build sampling density ---
    eps = 0.1 * (grad_mag.max() if grad_mag.max() > 0 else 1.0)
    density = grad_mag + eps
    density /= density.sum()

    # --- Step 6: Anchor points for guaranteed coverage ---
    if d == 2:
        if LOG_SPACE:
            center_dx = np.sqrt(DX_BOUNDS[0] * DX_BOUNDS[1])
            center_dy = np.sqrt(DY_BOUNDS[0] * DY_BOUNDS[1])
        else:
            center_dx = (DX_BOUNDS[0] + DX_BOUNDS[1]) / 2
            center_dy = (DY_BOUNDS[0] + DY_BOUNDS[1]) / 2
        anchor_pts = np.array([
            [DX_BOUNDS[0], DY_BOUNDS[0]],
            [DX_BOUNDS[0], DY_BOUNDS[1]],
            [DX_BOUNDS[1], DY_BOUNDS[0]],
            [DX_BOUNDS[1], DY_BOUNDS[1]],
            [center_dx, center_dy],
        ])
    else:
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        if LOG_SPACE:
            center = np.sqrt(lo * hi)
        else:
            center = (lo + hi) / 2.0
        anchor_pts = [center.copy()]
        for dim in range(d):
            pt_lo = center.copy(); pt_lo[dim] = lo[dim]; anchor_pts.append(pt_lo)
            pt_hi = center.copy(); pt_hi[dim] = hi[dim]; anchor_pts.append(pt_hi)
        anchor_pts = np.array(anchor_pts)  # (1 + 2*d, d)
    n_anchor = len(anchor_pts)

    # --- Step 7: Adaptive sampling proportional to gradient magnitude ---
    rng = np.random.default_rng(SEED + 777)
    n_adaptive = max(n_test - n_anchor, 0)
    if n_adaptive > 0 and len(fine_grid) > n_adaptive:
        indices = rng.choice(len(fine_grid), size=n_adaptive, replace=False, p=density)
        adaptive_pts = fine_grid[indices]
    elif n_adaptive > 0:
        adaptive_pts = fine_grid.copy()
    else:
        adaptive_pts = np.empty((0, d))

    # --- Step 8: Combine and deduplicate ---
    all_pts = np.vstack([anchor_pts, adaptive_pts])
    min_rel_dist = 0.02  # 2% of domain range
    if d == 2:
        if LOG_SPACE:
            dx_range = np.log10(DX_BOUNDS[1]) - np.log10(DX_BOUNDS[0])
            dy_range = np.log10(DY_BOUNDS[1]) - np.log10(DY_BOUNDS[0])
        else:
            dx_range = DX_BOUNDS[1] - DX_BOUNDS[0]
            dy_range = DY_BOUNDS[1] - DY_BOUNDS[0]
        keep_mask = np.ones(len(all_pts), dtype=bool)
        for i in range(n_anchor, len(all_pts)):
            if LOG_SPACE:
                rel_dists = np.maximum(
                    np.abs(np.log10(all_pts[:n_anchor, 0]) - np.log10(all_pts[i, 0])) / dx_range,
                    np.abs(np.log10(all_pts[:n_anchor, 1]) - np.log10(all_pts[i, 1])) / dy_range
                )
            else:
                rel_dists = np.maximum(
                    np.abs(all_pts[:n_anchor, 0] - all_pts[i, 0]) / dx_range,
                    np.abs(all_pts[:n_anchor, 1] - all_pts[i, 1]) / dy_range
                )
            if rel_dists.min() < min_rel_dist:
                keep_mask[i] = False
        grid = all_pts[keep_mask]
    else:
        ranges = np.array([b[1] - b[0] for b in bounds])
        if LOG_SPACE:
            ranges = np.array([np.log10(b[1]) - np.log10(b[0]) for b in bounds])
        keep_mask = np.ones(len(all_pts), dtype=bool)
        for i in range(n_anchor, len(all_pts)):
            if LOG_SPACE:
                rel_dists = np.max(
                    np.abs(np.log10(all_pts[:n_anchor]) - np.log10(all_pts[i])) / ranges,
                    axis=1)
            else:
                rel_dists = np.max(
                    np.abs(all_pts[:n_anchor] - all_pts[i]) / ranges,
                    axis=1)
            if rel_dists.min() < min_rel_dist:
                keep_mask[i] = False
        grid = all_pts[keep_mask]

    log.info(f"Adaptive test grid: {len(grid)} points "
             f"({n_anchor} anchors + {len(grid) - n_anchor} gradient-adaptive)")

    # --- Step 9: Simulate FOM for each test point ---
    test_imgs = _simulate_batch(grid, snap_dir)
    log.info(f"Test grid ready: shape {test_imgs.shape}")
    return grid, test_imgs


def compute_projection_rmse(train_imgs, test_imgs, n_modes=None):
    """[DEPRECATED — only used by run_greedy_cg]
    Projection RMSE: Build POD basis from training snapshots, project test FOM snapshots,
    measure relative reconstruction error.  For GP-based methods, use evaluate_on_test_grid().

    Returns: mean relative RMSE across all test points and time steps.
    """
    if n_modes is None:
        n_modes = POD_MODES
    # Build POD basis from training snapshots
    k = train_imgs.shape[0]
    centered = []
    for i in range(k):
        s = train_imgs[i]
        s_centered = s - s.mean(axis=1, keepdims=True)
        centered.append(s_centered)
    S = np.concatenate(centered, axis=1)  # (N_spatial, k*T)
    U, _, _ = np.linalg.svd(S, full_matrices=False)
    Phi = U[:, :n_modes]  # (N_spatial, n_modes)

    # Projection error on test snapshots
    # Center test snapshots (matching training preprocessing) before projection
    n_test = test_imgs.shape[0]
    rel_errors = []
    for i in range(n_test):
        Y = test_imgs[i]  # (N_spatial, T)
        Y_c = Y - Y.mean(axis=1, keepdims=True)  # center like training
        Y_proj = Phi @ (Phi.T @ Y_c)  # (N_spatial, T)
        numer = np.linalg.norm(Y_c - Y_proj, 'fro')
        denom = np.linalg.norm(Y_c, 'fro')
        if denom > 1e-12:
            rel_errors.append(numer / denom)
    return float(np.mean(rel_errors)) if rel_errors else float('nan')


def build_pod_basis(train_imgs, n_modes=None):
    """Build POD basis from training snapshots, return Phi and singular values."""
    if n_modes is None:
        n_modes = POD_MODES
    k = train_imgs.shape[0]
    centered = []
    for i in range(k):
        s = train_imgs[i]
        s_centered = s - s.mean(axis=1, keepdims=True)
        centered.append(s_centered)
    S = np.concatenate(centered, axis=1)
    U, sig, _ = np.linalg.svd(S, full_matrices=False)
    return U[:, :n_modes], sig[:n_modes]



def compute_manifold_range_metric(train_imgs, n_modes=None):
    """[DEPRECATED — only used by run_greedy_cg]
    Manifold range: max pairwise geodesic distance among training POD subspaces.
    Each training snapshot is mapped to its own rank-p subspace, then pairwise distances
    are computed on the Grassmannian.
    """
    if n_modes is None:
        n_modes = POD_MODES
    k = train_imgs.shape[0]
    if k < 2:
        return float('nan')

    # Build per-snapshot subspaces
    phis = []
    for i in range(k):
        Y = train_imgs[i]  # (N_spatial, T)
        Y_c = Y - Y.mean(axis=1, keepdims=True)
        U, _, _ = np.linalg.svd(Y_c, full_matrices=False)
        Q, _ = np.linalg.qr(U[:, :n_modes])
        phis.append(Q)
    phi_t = torch.as_tensor(np.stack(phis), dtype=torch.float32)  # (k, N_spatial, n_modes)
    return manifold_range(phi_t)



def _precompute_test_phi_true(test_imgs):
    """Pre-compute true subspaces for all test points (called once)."""
    n_test = test_imgs.shape[0]
    true_phis = []
    for i in range(n_test):
        Y = test_imgs[i]
        Y_c = Y - Y.mean(axis=1, keepdims=True)
        U, _, _ = np.linalg.svd(Y_c, full_matrices=False)
        true_phis.append(U[:, :POD_MODES])
    return torch.as_tensor(np.stack(true_phis), dtype=torch.float32)


def evaluate_on_test_grid(gp, scaler, test_tau, test_imgs, phi_true,
                           prev_phi_pred=None):
    """Evaluate all metrics on the test grid using one GP predict call.

    Parameters
    ----------
    gp : projectGP — fitted GP model
    scaler : normalizer for parameter space
    test_tau : (N_test, d) test parameters
    test_imgs : (N_test, n, t) true test snapshots
    phi_true : (N_test, n, p) true subspaces (pre-computed)
    prev_phi_pred : (N_test, n, p) tensor, optional
        GP-predicted subspaces from the previous step (for pod_angle)

    Returns
    -------
    metrics : dict with 'proj_rmse', 'geo_pred', 'pod_angle'
    Phi_pred : (N_test, n, p) tensor — current predicted subspaces (save for next step)
    """
    try:
        test_tau_std = scaler.transform(test_tau)
        test_tau_t = torch.as_tensor(test_tau_std, dtype=torch.float32)
        mu_y, _ = gp.predict(test_tau_t)
        Phi_pred = gp.bijective_map(mu_y)  # (N_test, n, p)
        if Phi_pred.ndim == 2:
            Phi_pred = Phi_pred.unsqueeze(0)

        n_test = test_imgs.shape[0]

        # --- proj_rmse: reconstruction error using GP-predicted basis ---
        rel_errors = []
        for i in range(n_test):
            phi_pred_i = Phi_pred[i].detach().numpy()  # (n, p)
            Y_c = test_imgs[i] - test_imgs[i].mean(axis=1, keepdims=True)
            Y_recon = phi_pred_i @ (phi_pred_i.T @ Y_c)
            numer = np.linalg.norm(Y_c - Y_recon, 'fro')
            denom = np.linalg.norm(Y_c, 'fro')
            if denom > 1e-12:
                rel_errors.append(numer / denom)
        proj_rmse = float(np.mean(rel_errors)) if rel_errors else float('nan')

        # --- qoi_error: domain-averaged state at final time ---
        qoi_errors = []
        for i in range(n_test):
            phi_pred_i = Phi_pred[i].detach().numpy()  # (n, p)
            Y = test_imgs[i]  # (n, t)
            Y_c = Y - Y.mean(axis=1, keepdims=True)
            Y_recon = Y.mean(axis=1, keepdims=True) + phi_pred_i @ (phi_pred_i.T @ Y_c)
            # QoI = domain-averaged state at final time
            qoi_fom = float(np.mean(Y[:, -1]))
            qoi_rom = float(np.mean(Y_recon[:, -1]))
            denom = abs(qoi_fom) if abs(qoi_fom) > 1e-12 else 1e-12
            qoi_errors.append(abs(qoi_rom - qoi_fom) / denom)
        qoi_error = float(np.mean(qoi_errors)) if qoi_errors else float('nan')

        # --- geo_pred: geodesic distance between predicted and true subspaces ---
        from pgp.manifold import geodesic_batch
        phi_true_t = torch.as_tensor(phi_true, dtype=torch.float32)
        geo_dists = geodesic_batch(Phi_pred, phi_true_t)
        geo_pred = float(geo_dists.mean())

        # --- pod_angle: angle change in GP predictions between consecutive steps ---
        pod_angle = float('nan')
        if prev_phi_pred is not None:
            try:
                angles = []
                for i in range(n_test):
                    M = prev_phi_pred[i].T @ Phi_pred[i]
                    svals = torch.linalg.svdvals(M)
                    svals = torch.clamp(svals, -1.0, 1.0)
                    angle_i = torch.max(torch.acos(svals))
                    angles.append(float(torch.rad2deg(angle_i)))
                pod_angle = float(np.mean(angles)) if angles else float('nan')
            except Exception as e:
                log.warning(f"pod_angle computation failed: {e}")

        return {'proj_rmse': proj_rmse, 'geo_pred': geo_pred,
                'pod_angle': pod_angle, 'qoi_error': qoi_error}, Phi_pred

    except Exception as e:
        log.warning(f"evaluate_on_test_grid failed: {e}")
        return {'proj_rmse': float('nan'), 'geo_pred': float('nan'),
                'pod_angle': float('nan'), 'qoi_error': float('nan')}, prev_phi_pred


def evaluate_on_test_grid_with_uq(gp, scaler, test_tau, test_imgs, phi_true,
                                    prev_phi_pred=None):
    """Evaluate metrics + per-test-point UQ data for calibration analysis.

    Wraps evaluate_on_test_grid() and adds per-test-point:
      - manifold_var: MC manifold variance at each test point
      - euclidean_var: scalar GP posterior variance at each test point
      - geo_error: geodesic distance to true subspace per point
      - proj_error: projection RMSE per point

    Returns
    -------
    metrics : dict (same as evaluate_on_test_grid)
    Phi_pred : tensor
    uq_data : dict with lists of per-test-point values
    """
    metrics, Phi_pred = evaluate_on_test_grid(
        gp, scaler, test_tau, test_imgs, phi_true, prev_phi_pred)

    uq_data = {'manifold_var': [], 'euclidean_var': [],
                'geo_error': [], 'proj_error': []}
    try:
        # Per-test-point manifold variance
        mvar = compute_manifold_var(gp, scaler, test_tau)
        uq_data['manifold_var'] = mvar.tolist()

        # Per-test-point euclidean variance
        evar = compute_euclidean_var(gp, scaler, test_tau)
        uq_data['euclidean_var'] = evar.tolist()

        # Per-test-point geodesic error
        if Phi_pred is not None:
            phi_true_t = torch.as_tensor(phi_true, dtype=torch.float32)
            geo_dists = geodesic_batch(Phi_pred, phi_true_t)
            uq_data['geo_error'] = geo_dists.detach().numpy().tolist()

        # Per-test-point projection RMSE
        n_test = test_imgs.shape[0]
        for i in range(n_test):
            if Phi_pred is not None:
                phi_pred_i = Phi_pred[i].detach().numpy()
                Y_c = test_imgs[i] - test_imgs[i].mean(axis=1, keepdims=True)
                Y_recon = phi_pred_i @ (phi_pred_i.T @ Y_c)
                numer = np.linalg.norm(Y_c - Y_recon, 'fro')
                denom = np.linalg.norm(Y_c, 'fro')
                uq_data['proj_error'].append(float(numer / denom) if denom > 1e-12 else float('nan'))
            else:
                uq_data['proj_error'].append(float('nan'))
    except Exception as e:
        log.warning(f"UQ data computation failed: {e}")

    return metrics, Phi_pred, uq_data


def check_ball_violations(train_tau, train_imgs, kernel=None):
    """Check how many training tangent vectors violate the pi/2 ball constraint."""
    try:
        gp, scaler, _ = build_gp(train_tau, train_imgs, kernel=kernel)
        norms, violations = gp.check_ball_constraint(warn=True)
        return float(norms.max()), len(violations), len(norms)
    except Exception:
        return float('nan'), 0, 0


# ===================== HELPERS =====================

def _seed_init_points(init_seeds, snap_dir, meta_csv, shared_init_dir=None):
    """Copy or simulate initial seed points and write to meta CSV."""
    import shutil
    for s in init_seeds:
        fname = _param_filename(s)
        if shared_init_dir is not None:
            src = shared_init_dir / fname
            dst = snap_dir / fname
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
            csv_path = dst
        else:
            csv_path = simulate(s, snap_dir)
        append_meta(meta_csv, s, csv_path)


def select_batch(v_s, cands, bounds, acq_config, batch_size):
    """Greedy batch selection with distance-based diversity penalty.

    Selects batch_size points: pick argmax, penalize neighbors, repeat.
    When batch_size=1, equivalent to single argmax.
    """
    if batch_size <= 1:
        idx = int(np.argmax(v_s))
        return [idx]
    scores = v_s.copy()
    base = _normalize_to_unit(cands, bounds, log_space=acq_config.log_space)
    h = max(acq_config.smooth_h * 2.0, 0.40)
    selected = []
    for b in range(batch_size):
        idx = int(np.argmax(scores))
        selected.append(idx)
        if b < batch_size - 1:
            d2 = ((base - base[idx:idx+1]) ** 2).sum(axis=1)
            scores *= (1.0 - np.exp(-d2 / (2 * h * h)))
    return selected


def select_batch_maxpro(cands_raw, tau_existing, logspace, batch_size):
    """Greedy batch MaxPro: pick best, add to existing, repeat."""
    existing = np.log10(tau_existing) if logspace else tau_existing.copy()
    cands_work = np.log10(cands_raw) if logspace else cands_raw.copy()
    selected = []
    for _ in range(batch_size):
        vals = np.array([maxpro_criterion(c, existing) for c in cands_work])
        idx = int(np.argmin(vals))
        selected.append(idx)
        existing = np.vstack([existing, cands_work[idx:idx+1]])
    return selected


# ===================== FOUR METHODS =====================

def run_proposed(init_seeds, snap_dir, out_dir, test_tau=None, test_imgs=None,
                 kernel=None, ref_method=None, acq_config=None, stop_event=None,
                 n_steps=None, n_cand=None, shared_init_dir=None, batch_size=1,
                 opt_style=None):
    """Proposed method: multi-objective manifold-aware acquisition.

    Parameters
    ----------
    kernel : str, optional
        Kernel type ("rbf", "matern52", "poly_rbf", "gibbs"). Defaults to KERNEL.
    ref_method : str, optional
        Reference point method ("stacked_svd", "karcher"). Defaults to "stacked_svd".
    acq_config : AcquisitionConfig, optional
        Acquisition function configuration. If None, uses original multiplicative acquisition.
    stop_event : multiprocessing.Event, optional
        If set, the method will stop and return partial results.
    n_steps : int, optional
        Number of sequential steps. Defaults to N_STEPS.
    n_cand : int, optional
        Number of candidates per step. Defaults to N_CAND.
    """
    if kernel is None:
        kernel = KERNEL
    if acq_config is None:
        acq_config = AcquisitionConfig.original()
    if n_steps is None:
        n_steps = N_STEPS
    if n_cand is None:
        n_cand = N_CAND

    log.info(f"===== Running PROPOSED method (kernel={kernel}, ref={ref_method or 'default'}, "
             f"acq={acq_config.mode}) =====")
    out_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(exist_ok=True)
    meta_csv = out_dir / "train_meta.csv"
    iter_log = []

    # Seed — use shared init snapshots if available, else simulate independently
    _seed_init_points(init_seeds, snap_dir, meta_csv, shared_init_dir)

    # Pre-compute true subspaces for test grid (done once)
    phi_true = _precompute_test_phi_true(test_imgs) if test_imgs is not None else None

    fixed_ref = None       # Will be set from the first GP build
    prev_phi_pred = None   # For pod_angle (GP predictions at test grid)
    prev_raw_theta = None

    for step in range(1, n_steps + 1):
        if stop_event is not None and stop_event.is_set():
            log.info(f"  Proposed: stopped early at step {step}")
            break

        tau, imgs = load_training(meta_csv)
        # On the first build, pass test_imgs as pilot for better reference coverage
        pilot = test_imgs if (fixed_ref is None and test_imgs is not None) else None
        gp, scaler, theta = build_gp(tau, imgs, prev_raw_theta=prev_raw_theta,
                                      kernel=kernel, ref_method=ref_method,
                                      fixed_reference_pt=fixed_ref,
                                      pilot_imgs=pilot,
                                      step=step, restart_interval=RESTART_INTERVAL,
                                      weighted=WEIGHTED_GP,
                                      opt_style=opt_style)
        prev_raw_theta = theta.get('_raw_theta', None)
        hyper_info = _extract_hyper_info(theta, gp)

        # Karcher/ball constraint diagnostics
        karcher_info = _extract_karcher_ball_info(gp)

        # Fix reference point from the first build (initial design)
        if fixed_ref is None:
            fixed_ref = gp.reference_pt.clone()

        # --- ALL metrics from N-point GP (before simulate) ---
        # MMV: on candidate grid
        mmv = float('nan')
        bounds = get_bounds()
        cands = sobol_candidates(n_cand, bounds, LOG_SPACE, seed_offset=7 + step)
        mc_gen = make_mc_generator(step)
        try:
            eval_cands = cands[::4] if len(cands) > 256 else cands
            v_mmv = compute_manifold_var(gp, scaler, eval_cands, generator=mc_gen)
            base_mmv = _normalize_to_unit(eval_cands, bounds, log_space=LOG_SPACE)
            v_mmv_s = gaussian_smoother(base_mmv, v_mmv, SMOOTH_H)
            mmv = float(np.max(v_mmv_s))
        except Exception as e:
            log.warning("MMV eval failed: {}".format(e))

        # Test-grid metrics: proj_rmse, geo_pred, pod_angle, qoi_error + UQ data
        metrics = {'proj_rmse': float('nan'), 'geo_pred': float('nan'),
                   'pod_angle': float('nan'), 'qoi_error': float('nan')}
        uq_data = None
        if test_tau is not None and test_imgs is not None and phi_true is not None:
            metrics, prev_phi_pred, uq_data = evaluate_on_test_grid_with_uq(
                gp, scaler, test_tau, test_imgs, phi_true, prev_phi_pred)

        # Modular acquisition
        from functools import partial
        mc_gen_acq = make_mc_generator(step)
        v_s = build_acquisition(cands, gp, scaler, tau, bounds, acq_config,
                                manifold_var_fn=partial(compute_manifold_var,
                                                        generator=mc_gen_acq),
                                diversity_fn=compute_diversity_bonus)

        batch_indices = select_batch(v_s, cands, bounds, acq_config, batch_size)
        x_batch = cands[batch_indices]
        x_new = x_batch[0]  # first selected point (for diagnostics)

        # Diagnostic plots at regular intervals and final step
        if step % DIAG_INTERVAL == 0 or step == n_steps:
            diag_dir = out_dir.parent / "diagnostics"
            mname = out_dir.name
            try:
                if get_param_dim() == 2:
                    plot_gp_surface(gp, scaler, tau, diag_dir, step, mname,
                                    test_tau=test_tau)
                    plot_variance_surface(gp, scaler, tau, diag_dir, step, mname,
                                          selected_pt=x_new)
                    plot_acquisition_surface(cands, v_s, tau, diag_dir, step, mname,
                                             selected_pt=x_new)
                else:
                    plot_scatter_nd(tau, N_INIT, diag_dir, step, mname,
                                   n_steps_total=n_steps)
                    plot_variance_nd(gp, scaler, tau, diag_dir, step, mname,
                                    selected_pt=x_new)
            except Exception as e:
                log.warning("Diagnostic plot failed at step {}: {}".format(step, e))

        # Simulate batch and append (NO post-simulate metric computation)
        for x_pt in x_batch:
            csv_path = simulate(x_pt, snap_dir)
            append_meta(meta_csv, x_pt, csv_path)

        entry = {"iter": step,
                 "picked_params": [x.tolist() for x in x_batch],
                 "batch_size": len(x_batch),
                 "mmv": mmv, "proj_rmse": metrics['proj_rmse'],
                 "geo_pred": metrics['geo_pred'],
                 "pod_angle": metrics['pod_angle'],
                 "qoi_error": metrics.get('qoi_error', float('nan')),
                 "hypers": hyper_info,
                 **karcher_info}
        if uq_data is not None:
            entry["uq_calibration"] = uq_data
        iter_log.append(entry)
        log.info("  Proposed step {}: {} pts, mmv={:.6f}, "
                 "proj_rmse={:.6f}, pod_angle={:.2f}°, geo_pred={:.4f}, "
                 "qoi_err={:.6f}".format(
                     step, len(x_batch), mmv, metrics['proj_rmse'],
                     metrics['pod_angle'], metrics['geo_pred'],
                     metrics.get('qoi_error', float('nan'))))

        # Save partial log after each iteration for interruptibility
        log_path = out_dir.parent / "{}_log.json".format(out_dir.name)
        with open(log_path, "w") as f:
            json.dump(iter_log, f, indent=2)

    return iter_log


def run_maxpro(init_seeds, snap_dir, out_dir, test_tau=None, test_imgs=None,
               stop_event=None, n_steps=None, n_cand=None, shared_init_dir=None,
               eval_interval=1, batch_size=1):
    """MaxPro baseline: space-filling design."""
    if n_steps is None:
        n_steps = N_STEPS
    if n_cand is None:
        n_cand = N_CAND
    log.info("===== Running MAXPRO baseline =====")
    out_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(exist_ok=True)
    meta_csv = out_dir / "train_meta.csv"
    iter_log = []

    _seed_init_points(init_seeds, snap_dir, meta_csv, shared_init_dir)

    # Pre-compute true subspaces for test grid (done once)
    phi_true = _precompute_test_phi_true(test_imgs) if test_imgs is not None else None

    fixed_ref = None
    prev_phi_pred_mp = None
    prev_raw_theta = None
    for step in range(1, n_steps + 1):
        if stop_event is not None and stop_event.is_set():
            log.info(f"  MaxPro: stopped early at step {step}")
            break
        tau, imgs = load_training(meta_csv)
        bounds = get_bounds()

        # GP-based metrics (eval_interval controls frequency)
        do_full_eval = (eval_interval <= 1 or step % eval_interval == 0 or step == n_steps)
        mmv = float('nan')
        metrics = {'proj_rmse': float('nan'), 'geo_pred': float('nan'), 'pod_angle': float('nan')}
        hyper_info_mp = {}
        karcher_info_mp = {}
        gp_eval = None
        if do_full_eval:
            try:
                pilot = test_imgs if (fixed_ref is None and test_imgs is not None) else None
                gp_eval, scaler_eval, theta_eval = build_gp(
                    tau, imgs, prev_raw_theta=prev_raw_theta,
                    kernel="matern52",
                    fixed_reference_pt=fixed_ref,
                    pilot_imgs=pilot,
                    step=step, restart_interval=RESTART_INTERVAL,
                    weighted=WEIGHTED_GP)
                prev_raw_theta = theta_eval.get('_raw_theta', None)
                hyper_info_mp = _extract_hyper_info(theta_eval, gp_eval)
                karcher_info_mp = _extract_karcher_ball_info(gp_eval)
                if fixed_ref is None:
                    fixed_ref = gp_eval.reference_pt.clone()

                # MMV on candidate grid
                cands_raw = sobol_candidates(n_cand, bounds, LOG_SPACE, seed_offset=7 + step)
                eval_cands = cands_raw[::4] if len(cands_raw) > 256 else cands_raw
                mc_gen = make_mc_generator(step)
                v = compute_manifold_var(gp_eval, scaler_eval, eval_cands, generator=mc_gen)
                base_eval = _normalize_to_unit(eval_cands, bounds, log_space=LOG_SPACE)
                v_s = gaussian_smoother(base_eval, v, SMOOTH_H)
                mmv = float(np.max(v_s))

                # Test-grid metrics
                if test_tau is not None and test_imgs is not None and phi_true is not None:
                    metrics, prev_phi_pred_mp = evaluate_on_test_grid(
                        gp_eval, scaler_eval, test_tau, test_imgs, phi_true, prev_phi_pred_mp)
            except Exception as e:
                log.warning(f"Metric eval failed: {e}")

        # MaxPro batch selection in log space
        cands_raw = sobol_candidates(n_cand, bounds, LOG_SPACE, seed_offset=7 + step)
        batch_indices = select_batch_maxpro(cands_raw, tau, LOG_SPACE, batch_size)
        x_batch = cands_raw[batch_indices]

        # Diagnostic plots
        if (step % DIAG_INTERVAL == 0 or step == n_steps) and gp_eval is not None:
            diag_dir = out_dir.parent / "diagnostics"
            mname = out_dir.name
            try:
                if get_param_dim() == 2:
                    plot_gp_surface(gp_eval, scaler_eval, tau, diag_dir, step, mname,
                                    test_tau=test_tau)
                    plot_variance_surface(gp_eval, scaler_eval, tau, diag_dir, step, mname)
                else:
                    plot_scatter_nd(tau, N_INIT, diag_dir, step, mname,
                                   n_steps_total=n_steps)
                    plot_variance_nd(gp_eval, scaler_eval, tau, diag_dir, step, mname)
            except Exception as e:
                log.warning(f"MaxPro diagnostic plot failed at step {step}: {e}")

        # Simulate batch and append
        for x_pt in x_batch:
            csv_path = simulate(x_pt, snap_dir)
            append_meta(meta_csv, x_pt, csv_path)

        iter_log.append({"iter": step,
                         "picked_params": [x.tolist() for x in x_batch],
                         "batch_size": len(x_batch),
                         "mmv": mmv, "proj_rmse": metrics['proj_rmse'],
                         "geo_pred": metrics['geo_pred'],
                         "pod_angle": metrics['pod_angle'],
                         "qoi_error": metrics.get('qoi_error', float('nan')),
                         "hypers": hyper_info_mp if do_full_eval else {},
                         **(karcher_info_mp if do_full_eval else {})})
        log.info(f"  MaxPro step {step}: {len(x_batch)} pts, mmv={mmv:.6f}, "
                 f"proj_rmse={metrics['proj_rmse']:.6f}, "
                 f"pod_angle={metrics['pod_angle']:.2f}°, geo_pred={metrics['geo_pred']:.4f}")

        log_path = out_dir.parent / f"{out_dir.name}_log.json"
        with open(log_path, "w") as f:
            json.dump(iter_log, f, indent=2)

    return iter_log


def run_greedy_replay(init_seeds_unused, snap_dir, out_dir, test_tau=None, test_imgs=None,
                      stop_event=None, n_steps=None, n_cand=None, shared_init_dir=None,
                      eval_interval=1, batch_size=1, greedy_log_path=None, **kwargs):
    """Replay old greedy CG results: use pre-selected points, evaluate with current pGP metrics.

    Reads a greedy_log.json from a previous run.  Uses its initial points and
    chosen points (first n_steps) as the sequential design.  At each step,
    simulates any missing snapshots and builds a matern52 pGP to evaluate
    metrics (mmv, proj_rmse, geo_pred, pod_angle).

    Parameters
    ----------
    greedy_log_path : str or Path
        Path to the old greedy_log.json.
    init_seeds_unused : ignored (init points come from greedy_log's train_meta.csv)
    """
    if n_steps is None:
        n_steps = N_STEPS
    if n_cand is None:
        n_cand = N_CAND

    if greedy_log_path is None:
        raise ValueError("greedy_log_path is required for greedy_replay")
    greedy_log_path = pathlib.Path(greedy_log_path)

    # Load old greedy log
    with open(greedy_log_path) as f:
        old_data = json.load(f)
    old_logs = old_data['logs']
    if len(old_logs) < n_steps:
        log.warning(f"Old greedy log has only {len(old_logs)} steps, requested {n_steps}")
        n_steps = len(old_logs)

    # Load old initial points from train_meta.csv alongside greedy_log.json
    old_meta_csv = greedy_log_path.parent / "train_meta.csv"
    if not old_meta_csv.exists():
        # Also check parent directory (old code structure: work/greedy_log.json, ../train_meta.csv)
        old_meta_csv = greedy_log_path.parent.parent / "train_meta.csv"
    if not old_meta_csv.exists():
        raise FileNotFoundError(f"Need train_meta.csv near {greedy_log_path} for initial points")
    init_points = []
    with open(old_meta_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            init_points.append(np.array([float(row['Dx']), float(row['Dy'])]))

    log.info(f"===== Running GREEDY REPLAY from {greedy_log_path} =====")
    log.info(f"  {len(init_points)} initial points + {n_steps} greedy steps")
    out_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(exist_ok=True)
    meta_csv = out_dir / "train_meta.csv"
    iter_log = []

    # Write initial points to meta_csv and simulate snapshots
    if not meta_csv.exists():
        with open(meta_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Dx", "Dy", "snapshot_csv"])
        for pt in init_points:
            csv_path = simulate(pt, snap_dir)
            append_meta(meta_csv, pt, csv_path)
        log.info(f"  Simulated {len(init_points)} initial snapshots")

    # Pre-compute true subspaces
    phi_true = _precompute_test_phi_true(test_imgs) if test_imgs is not None else None

    fixed_ref = None
    prev_phi_pred = None
    prev_raw_theta = None

    for step in range(1, n_steps + 1):
        if stop_event is not None and stop_event.is_set():
            log.info(f"  Greedy replay: stopped early at step {step}")
            break

        tau, imgs = load_training(meta_csv)
        bounds = get_bounds()

        # Evaluate metrics via matern52 pGP
        do_full_eval = (eval_interval <= 1 or step % eval_interval == 0 or step == n_steps)
        mmv = float('nan')
        metrics = {'proj_rmse': float('nan'), 'geo_pred': float('nan'), 'pod_angle': float('nan')}
        hyper_info = {}
        karcher_info = {}
        if do_full_eval:
            try:
                pilot = test_imgs if (fixed_ref is None and test_imgs is not None) else None
                gp, scaler, theta = build_gp(
                    tau, imgs, prev_raw_theta=prev_raw_theta,
                    kernel="matern52",
                    fixed_reference_pt=fixed_ref,
                    pilot_imgs=pilot,
                    step=step, restart_interval=RESTART_INTERVAL,
                    weighted=WEIGHTED_GP)
                prev_raw_theta = theta.get('_raw_theta', None)
                hyper_info = _extract_hyper_info(theta, gp)
                karcher_info = _extract_karcher_ball_info(gp)
                if fixed_ref is None:
                    fixed_ref = gp.reference_pt.clone()

                # MMV
                cands_raw = sobol_candidates(n_cand, bounds, LOG_SPACE, seed_offset=7 + step)
                eval_cands = cands_raw[::4] if len(cands_raw) > 256 else cands_raw
                mc_gen = make_mc_generator(step)
                v = compute_manifold_var(gp, scaler, eval_cands, generator=mc_gen)
                base_eval = _normalize_to_unit(eval_cands, bounds, log_space=LOG_SPACE)
                v_s = gaussian_smoother(base_eval, v, SMOOTH_H)
                mmv = float(np.max(v_s))

                # Test-grid metrics
                if test_tau is not None and test_imgs is not None and phi_true is not None:
                    metrics, prev_phi_pred = evaluate_on_test_grid(
                        gp, scaler, test_tau, test_imgs, phi_true, prev_phi_pred)
            except Exception as e:
                log.warning(f"Metric eval failed at step {step}: {e}")

        # Add the pre-determined greedy point
        chosen = old_logs[step - 1]['chosen'][0]  # [Dx, Dy]
        x_new = np.array(chosen)
        csv_path = simulate(x_new, snap_dir)
        append_meta(meta_csv, x_new, csv_path)

        iter_log.append({"iter": step,
                         "picked_params": [x_new.tolist()],
                         "batch_size": 1,
                         "mmv": mmv, "proj_rmse": metrics['proj_rmse'],
                         "geo_pred": metrics['geo_pred'],
                         "pod_angle": metrics['pod_angle'],
                         "hypers": hyper_info,
                         **karcher_info})
        log.info(f"  Greedy replay step {step}: pt={x_new}, mmv={mmv:.6f}, "
                 f"proj_rmse={metrics['proj_rmse']:.6f}, "
                 f"pod_angle={metrics['pod_angle']:.2f}°, geo_pred={metrics['geo_pred']:.4f}")

        log_path = out_dir.parent / f"{out_dir.name}_log.json"
        with open(log_path, "w") as f:
            json.dump(iter_log, f, indent=2)

    return iter_log


def plot_gp_surface(gp, scaler, tau_train, diag_dir, step, method_name,
                    test_tau=None, n_fine=40):
    """Plot the GP mean prediction surface (tangent vector norm).

    Shows how the fitted GP "sees" the parameter space: regions with high
    ||z*|| correspond to solutions that differ significantly from the reference.

    Parameters
    ----------
    gp : fitted projectGP instance
    scaler : data normalizer
    tau_train : (k, 2) training parameters (original scale)
    diag_dir : Path for saving plots
    step : current iteration number
    method_name : method identifier for filename
    test_tau : (N_test, 2) test point locations (optional, plotted as red x)
    n_fine : resolution of the evaluation grid
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diag_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate GP mean on fine grid
    if LOG_SPACE:
        dx_vals = np.logspace(np.log10(DX_BOUNDS[0]), np.log10(DX_BOUNDS[1]), n_fine)
        dy_vals = np.logspace(np.log10(DY_BOUNDS[0]), np.log10(DY_BOUNDS[1]), n_fine)
    else:
        dx_vals = np.linspace(DX_BOUNDS[0], DX_BOUNDS[1], n_fine)
        dy_vals = np.linspace(DY_BOUNDS[0], DY_BOUNDS[1], n_fine)
    fine_grid = np.array([[dx, dy] for dx in dx_vals for dy in dy_vals])

    fine_std = scaler.transform(fine_grid)
    fine_t = torch.as_tensor(fine_std, dtype=torch.float32)
    mu_y, _ = gp.predict(fine_t)
    z_norm = torch.linalg.norm(mu_y, dim=1).detach().numpy()
    Z = z_norm.reshape(n_fine, n_fine)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    DX, DY = np.meshgrid(dx_vals, dy_vals, indexing='ij')
    cf = ax.contourf(DX, DY, Z, levels=30, cmap='viridis')
    plt.colorbar(cf, ax=ax, label=r'$\|z^*\|$ (tangent vector norm)')

    # Training points
    ax.scatter(tau_train[:, 0], tau_train[:, 1], c='black', s=40, zorder=5,
               edgecolors='white', linewidths=0.8, label='Training')
    # Test points
    if test_tau is not None:
        ax.scatter(test_tau[:, 0], test_tau[:, 1], c='red', marker='x', s=30,
                   zorder=4, alpha=0.6, label='Test')

    ax.set_xlabel('Dx', fontsize=12)
    ax.set_ylabel('Dy', fontsize=12)
    ax.set_title('GP Mean Surface: {} (step {})'.format(method_name, step), fontsize=13)
    if LOG_SPACE:
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(DX_BOUNDS)
    ax.set_ylim(DY_BOUNDS)
    ax.legend(fontsize=9, loc='upper left')

    path = diag_dir / '{}_gp_surface_step{:02d}.png'.format(method_name, step)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("  Saved: {}".format(path))


def plot_variance_surface(gp, scaler, tau_train, diag_dir, step, method_name,
                          selected_pt=None, n_fine=20):
    """Plot the manifold variance surface from pGP posterior.

    Shows where the GP is uncertain about the manifold structure. This is
    the primary acquisition signal — regions with high variance are where
    new points should be added.

    Uses a coarser grid (20x20=400 points) because predict_phi() requires
    MC sampling which is more expensive than predict().

    Parameters
    ----------
    gp : fitted projectGP instance
    scaler : data normalizer
    tau_train : (k, 2) training parameters
    diag_dir : Path for saving plots
    step : current iteration number
    method_name : method identifier
    selected_pt : (2,) the point selected in this step (plotted as star)
    n_fine : resolution (default 20 for speed)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diag_dir.mkdir(parents=True, exist_ok=True)

    if LOG_SPACE:
        dx_vals = np.logspace(np.log10(DX_BOUNDS[0]), np.log10(DX_BOUNDS[1]), n_fine)
        dy_vals = np.logspace(np.log10(DY_BOUNDS[0]), np.log10(DY_BOUNDS[1]), n_fine)
    else:
        dx_vals = np.linspace(DX_BOUNDS[0], DX_BOUNDS[1], n_fine)
        dy_vals = np.linspace(DY_BOUNDS[0], DY_BOUNDS[1], n_fine)
    fine_grid = np.array([[dx, dy] for dx in dx_vals for dy in dy_vals])

    fine_std = scaler.transform(fine_grid)
    fine_t = torch.as_tensor(fine_std, dtype=torch.float32)
    _, var_phi = gp.predict_phi(fine_t, n_samples=50, method='empirical_bayes')
    var_vals = var_phi.detach().numpy()
    V = var_vals.reshape(n_fine, n_fine)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    DX, DY = np.meshgrid(dx_vals, dy_vals, indexing='ij')
    cf = ax.contourf(DX, DY, V, levels=30, cmap='hot_r')
    plt.colorbar(cf, ax=ax, label='Manifold variance')

    # Training points
    ax.scatter(tau_train[:, 0], tau_train[:, 1], c='black', s=40, zorder=5,
               edgecolors='white', linewidths=0.8, label='Training')
    # Selected point
    if selected_pt is not None:
        ax.scatter([selected_pt[0]], [selected_pt[1]], c='lime', marker='*',
                   s=200, zorder=6, edgecolors='black', linewidths=1.0,
                   label='Selected')

    ax.set_xlabel('Dx', fontsize=12)
    ax.set_ylabel('Dy', fontsize=12)
    ax.set_title('Manifold Variance: {} (step {})'.format(method_name, step), fontsize=13)
    if LOG_SPACE:
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(DX_BOUNDS)
    ax.set_ylim(DY_BOUNDS)
    ax.legend(fontsize=9, loc='upper left')

    path = diag_dir / '{}_variance_surface_step{:02d}.png'.format(method_name, step)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("  Saved: {}".format(path))


def plot_acquisition_surface(cands, scores, tau_train, diag_dir, step, method_name,
                             selected_pt=None):
    """Plot the acquisition function scores on candidate points.

    Scatter plot of acquisition scores at candidate locations, showing
    how variance, diversity, repulsion, and interior combine to form the
    final acquisition score.

    Parameters
    ----------
    cands : (N_cand, 2) candidate points
    scores : (N_cand,) acquisition scores
    tau_train : (k, 2) training parameters
    diag_dir : Path for saving plots
    step : current iteration number
    method_name : method identifier
    selected_pt : (2,) the point selected in this step
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diag_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sc = ax.scatter(cands[:, 0], cands[:, 1], c=scores, cmap='plasma',
                    s=8, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='Acquisition score')

    # Training points
    ax.scatter(tau_train[:, 0], tau_train[:, 1], c='black', s=50, zorder=5,
               edgecolors='white', linewidths=0.8, label='Training')
    # Selected point
    if selected_pt is not None:
        ax.scatter([selected_pt[0]], [selected_pt[1]], c='lime', marker='*',
                   s=250, zorder=6, edgecolors='black', linewidths=1.0,
                   label='Selected')

    ax.set_xlabel('Dx', fontsize=12)
    ax.set_ylabel('Dy', fontsize=12)
    ax.set_title('Acquisition Score: {} (step {})'.format(method_name, step), fontsize=13)
    if LOG_SPACE:
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(DX_BOUNDS)
    ax.set_ylim(DY_BOUNDS)
    ax.legend(fontsize=9, loc='upper left')

    path = diag_dir / '{}_acquisition_step{:02d}.png'.format(method_name, step)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("  Saved: {}".format(path))


def plot_scatter_nd(tau_train, n_init, diag_dir, step, method_name,
                    n_steps_total=None):
    """Pairwise scatter matrix of design points in parameter space.

    Works for any dimensionality:
    - 2D: single scatter plot
    - 3-5D: pairwise scatter matrix with C(d,2) subplots
    - >5D: only first 5 dimensions

    Init points shown in gray, sequential points colored by iteration step.
    Colorbar range and axis limits are fixed across all steps.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diag_dir.mkdir(parents=True, exist_ok=True)

    d = tau_train.shape[1]
    d_show = min(d, 5)
    bounds = get_bounds()
    n_total = len(tau_train)
    n_seq = n_total - n_init

    # Compute iteration indices for sequential points
    batch_sz = max(1, getattr(sys.modules[__name__], 'BATCH_SIZE', 1))
    seq_iters = []
    for i in range(n_seq):
        seq_iters.append(i // batch_sz + 1)

    # Determine colorbar range (fixed across steps)
    vmax = n_steps_total if n_steps_total else (max(seq_iters) if seq_iters else 1)

    if d_show == 2:
        fig, ax = plt.subplots(figsize=(7, 6))
        _scatter_pair(ax, tau_train, n_init, seq_iters, 0, 1, bounds, vmax)
        fig.colorbar(ax.collections[-1] if n_seq > 0 else ax.collections[0],
                     ax=ax, label='Iteration')
    else:
        from itertools import combinations
        pairs = list(combinations(range(d_show), 2))
        n_pairs = len(pairs)
        ncols = min(n_pairs, 5)
        nrows = (n_pairs + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3 * nrows))
        if n_pairs == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        for idx, (i, j) in enumerate(pairs):
            if idx < len(axes_flat):
                _scatter_pair(axes_flat[idx], tau_train, n_init, seq_iters,
                              i, j, bounds, vmax)
        # Hide unused axes
        for idx in range(n_pairs, len(axes_flat)):
            axes_flat[idx].set_visible(False)
        # Shared colorbar
        if n_seq > 0:
            fig.colorbar(axes_flat[0].collections[-1], ax=axes_flat[:n_pairs],
                         label='Iteration', shrink=0.8)

    fig.suptitle(f'Design Points: {method_name} (step {step}, n={n_total})',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = diag_dir / f'{method_name}_scatter_step{step:02d}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved: {path}")


def _scatter_pair(ax, tau, n_init, seq_iters, dim_i, dim_j, bounds, vmax):
    """Helper: scatter one pair of dimensions."""
    d = tau.shape[1]
    # Dimension labels
    if PROBLEM == "darcy":
        labels = [f'ξ_{k+1}' for k in range(d)]
    else:
        labels = ['Dx', 'Dy'] + [f'x_{k+1}' for k in range(2, d)]

    # Init points (gray)
    ax.scatter(tau[:n_init, dim_i], tau[:n_init, dim_j],
               c='gray', s=20, alpha=0.5, edgecolors='none', label='Init')
    # Sequential points (colored by iteration)
    if len(tau) > n_init:
        sc = ax.scatter(tau[n_init:, dim_i], tau[n_init:, dim_j],
                        c=seq_iters, cmap='viridis', s=30, vmin=0, vmax=vmax,
                        edgecolors='k', linewidths=0.3)
    ax.set_xlabel(labels[dim_i], fontsize=9)
    ax.set_ylabel(labels[dim_j], fontsize=9)
    # Fixed axis limits
    if PROBLEM == "darcy":
        ax.set_xlim(bounds[dim_i])
        ax.set_ylim(bounds[dim_j])
    else:
        ax.set_xlim(bounds[dim_i])
        ax.set_ylim(bounds[dim_j])
        if LOG_SPACE:
            ax.set_xscale("log")
            ax.set_yscale("log")
    ax.tick_params(labelsize=7)


def plot_variance_nd(gp, scaler, tau_train, diag_dir, step, method_name,
                     selected_pt=None, n_fine=30):
    """1D marginal variance profiles for high-dimensional problems.

    For each dimension i: fix other dims at median of training data,
    sweep dim i across its bounds, compute manifold variance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diag_dir.mkdir(parents=True, exist_ok=True)

    d = tau_train.shape[1]
    bounds = get_bounds()
    medians = np.median(tau_train, axis=0)

    # Dimension labels
    if PROBLEM == "darcy":
        labels = [f'ξ_{i+1}' for i in range(d)]
    else:
        labels = ['Dx', 'Dy'] + [f'x_{i+1}' for i in range(2, d)]

    ncols = min(d, 5)
    nrows = (d + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    if d == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    all_var_max = 0.0

    for dim in range(min(d, 5)):
        ax = axes_flat[dim]
        lo, hi = bounds[dim]
        sweep = np.linspace(lo, hi, n_fine)

        # Build grid: all dims at median, sweep dim varies
        grid = np.tile(medians, (n_fine, 1))
        grid[:, dim] = sweep

        # Compute variance
        grid_std = scaler.transform(grid)
        grid_t = torch.as_tensor(grid_std, dtype=torch.float32)
        _, var_phi = gp.predict_phi(grid_t, n_samples=50,
                                     method='empirical_bayes')
        var_vals = var_phi.detach().numpy()
        all_var_max = max(all_var_max, var_vals.max())

        ax.plot(sweep, var_vals, '-', color='steelblue', linewidth=1.5)
        ax.fill_between(sweep, 0, var_vals, alpha=0.15, color='steelblue')

        # Rug plot of training points along this dimension
        ax.plot(tau_train[:, dim], np.zeros(len(tau_train)) - all_var_max * 0.02,
                '|', color='black', markersize=5, alpha=0.5)

        # Mark selected point
        if selected_pt is not None:
            ax.axvline(selected_pt[dim], color='lime', linewidth=2,
                       alpha=0.7, label='Selected')

        ax.set_xlabel(labels[dim], fontsize=10)
        ax.set_ylabel('Manifold var', fontsize=9)
        ax.set_xlim(lo, hi)
        ax.grid(True, alpha=0.2)

    # Unify y-axis across all subplots
    for dim in range(min(d, 5)):
        axes_flat[dim].set_ylim(-all_var_max * 0.05, all_var_max * 1.1)

    # Hide unused axes
    for idx in range(min(d, 5), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f'Variance Profile: {method_name} (step {step}, '
                 f'others @ median)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = diag_dir / f'{method_name}_variance_nd_step{step:02d}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved: {path}")


# How often to save diagnostic plots (every DIAG_INTERVAL steps + last step)
DIAG_INTERVAL = 5


# ===================== PLOTTING =====================

def _run_auto_diagnostics(results_dir, methods, display_names, logs_dict, cfg):
    """Generate lightweight diagnostic plots after all methods finish.

    Runs hyperparameter evolution and Karcher/ball diagnostics for each method.
    Failures are logged but never crash the experiment.
    """
    try:
        from experiments.diagnostics import (
            plot_hyperparameter_evolution,
            plot_karcher_ball_diagnostic,
        )
    except ImportError as e:
        log.warning(f"Cannot import diagnostics module: {e}")
        return

    diag_dir = results_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    for method_key in methods:
        display = display_names.get(method_key, method_key)
        if display not in logs_dict or not logs_dict[display]:
            continue

        log.info(f"  Auto-diagnostics for {method_key}...")

        try:
            plot_hyperparameter_evolution(results_dir, method_key, cfg, diag_dir)
            log.info(f"    hyperparameter_evolution.png saved")
        except Exception as e:
            log.warning(f"    Hyperparameter evolution plot failed: {e}")

        try:
            plot_karcher_ball_diagnostic(results_dir, method_key, cfg, diag_dir)
            log.info(f"    karcher_ball_diagnostic.png saved")
        except Exception as e:
            log.warning(f"    Karcher/ball diagnostic failed: {e}")


def _extract_dxdy(data):
    """Extract dx, dy lists from log data, handling both old/new picked_params formats."""
    if "picked_params" in data[0]:
        pp = data[0]["picked_params"]
        if isinstance(pp, list) and len(pp) > 0 and isinstance(pp[0], (list, tuple)):
            # New batch format: [[dx, dy], ...]
            dx_vals = [d["picked_params"][0][0] for d in data]
            dy_vals = [d["picked_params"][0][1] for d in data]
        else:
            # Old flat format: [dx, dy]
            dx_vals = [d["picked_params"][0] for d in data]
            dy_vals = [d["picked_params"][1] for d in data]
    else:
        dx_vals = [d["picked_Dx"] for d in data]
        dy_vals = [d["picked_Dy"] for d in data]
    return dx_vals, dy_vals


def plot_comparison(logs_dict, out_dir):
    """Generate comparison plots for all methods with 3 metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_styles = {
        "Proposed (pGP)": {"color": "#1f77b4", "marker": "o"},
        "Proposed (Gibbs)": {"color": "#9467bd", "marker": "P"},
        "Proposed (HRK)":  {"color": "#8c564b", "marker": "v"},
        "MaxPro":         {"color": "#ff7f0e", "marker": "s"},
        "Greedy CG":      {"color": "#d62728", "marker": "D"},
        "Random":         {"color": "#2ca02c", "marker": "^"},
    }

    # Filter out empty logs
    active = {k: v for k, v in logs_dict.items() if v}

    # ===== MAIN FIGURE: 4 metrics (2 rows x 2 cols) =====
    metrics = [
        ("mmv",             "Max Manifold Variance",        "lower is better"),
        ("proj_rmse",       "Projection RMSE",              "lower is better"),
        ("geo_pred",        "Geodesic Prediction Error",    "lower is better"),
        ("pod_angle",       "POD Angle Change (degrees)",   "lower = more stable"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()
    for ax, (key, ylabel, note) in zip(axes_flat, metrics):
        for name, data in active.items():
            style = method_styles.get(name, {"color": "gray", "marker": "x"})
            iters = [d["iter"] for d in data]
            vals = [d.get(key, float('nan')) for d in data]
            ax.plot(iters, vals, marker=style["marker"], markersize=4, label=name,
                    color=style["color"], linewidth=1.5, alpha=0.9)
        ax.set_xlabel("Iteration (# added points)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{ylabel}\n({note})", fontsize=12)
        ax.legend(frameon=True, fontsize=8)
        ax.grid(True, alpha=0.3)

    n_active = len(active)
    plt.suptitle(f"Sequential Design: {n_active}-Method Comparison  (N_init={N_INIT}, {N_STEPS} steps, d={get_param_dim()}, POD={POD_MODES})",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    path = out_dir / "metrics_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved: {path}")

    # ===== Parameter scatter for each method =====
    n_methods = len(active)
    if n_methods == 0:
        log.warning("No data to make scatter plots.")
        return
    if get_param_dim() != 2:
        log.info("Scatter plots skipped (>2D problem).")
        return
    fig, axes = plt.subplots(1, n_methods, figsize=(5.5 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]
    for ax_i, (name, data) in zip(axes, active.items()):
        style = method_styles.get(name, {"color": "gray", "marker": "x"})
        dx_vals, dy_vals = _extract_dxdy(data)
        iters = [d["iter"] for d in data]
        sc = ax_i.scatter(dx_vals, dy_vals, c=iters, cmap='viridis', s=50,
                          edgecolors='k', linewidths=0.5)
        if LOG_SPACE:
            ax_i.set_xscale("log"); ax_i.set_yscale("log")
        ax_i.set_xlabel("Dx", fontsize=12); ax_i.set_ylabel("Dy", fontsize=12)
        ax_i.set_title(f"{name}", fontsize=12)
        ax_i.set_xlim(DX_BOUNDS); ax_i.set_ylim(DY_BOUNDS)
        plt.colorbar(sc, ax=ax_i, label="Iteration")

    plt.suptitle(f"Selected Points per Method (N_init={N_INIT}, {N_STEPS} steps)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    path = out_dir / "scatter_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved: {path}")

    # ===== All methods scatter overlaid =====
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, data in active.items():
        style = method_styles.get(name, {"color": "gray", "marker": "x"})
        dx_vals, dy_vals = _extract_dxdy(data)
        ax.scatter(dx_vals, dy_vals, marker=style["marker"], s=50, label=name,
                   color=style["color"], edgecolors='k', linewidths=0.5, alpha=0.8)
    if LOG_SPACE:
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Dx", fontsize=12); ax.set_ylabel("Dy", fontsize=12)
    scale_label = "log scale" if LOG_SPACE else "linear scale"
    ax.set_title(f"All Methods: Selected Points Overlay\n(N_init={N_INIT}, {N_STEPS} steps, {scale_label})", fontsize=13)
    ax.set_xlim(DX_BOUNDS); ax.set_ylim(DY_BOUNDS)
    ax.legend(frameon=True, fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / "overlay_scatter.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved: {path}")

    # ===== Summary table =====
    log.info("\n" + "=" * 90)
    log.info("FINAL METRICS SUMMARY (last iteration)")
    log.info(f"{'Method':<20} {'MMV':>10} {'Proj RMSE':>12} {'POD angle':>10} {'Geo Pred':>10}")
    log.info("-" * 66)
    for name, data in active.items():
        if data:
            last = data[-1]
            mmv = last.get("mmv", last.get("acq_max_smooth", float('nan')))
            pr = last.get("proj_rmse", float('nan'))
            pa = last.get("pod_angle", float('nan'))
            gp = last.get("geo_pred", float('nan'))
            log.info(f"{name:<20} {mmv:>10.6f} {pr:>12.6f} {pa:>9.2f}° {gp:>10.4f}")
    log.info("=" * 66)


# ===================== HELPERS =====================

def _run_method(method_name, method_func, kwargs):
    """Run a single method in the main process."""
    try:
        result_log = method_func(stop_event=_STOP_EVENT, **kwargs)
        return method_name, result_log
    except Exception as e:
        log.error(f"Method {method_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return method_name, []


def load_logs_from_disk(results_dir):
    """Load all available *_log.json files from results directory."""
    name_map = {
        "proposed_matern52_log.json": "Proposed (Matérn)",
        "proposed_cgp_ktheta_r_log.json": "Proposed (CGP)",
        "maxpro_log.json": "MaxPro",
        "greedy_replay_log.json": "MC Greedy",
    }
    logs = {}
    for fname, display_name in name_map.items():
        fpath = results_dir / fname
        if fpath.exists():
            try:
                with open(fpath) as f:
                    data = json.load(f)
                if data:
                    logs[display_name] = data
                    log.info(f"  Loaded {fname}: {len(data)} iterations")
            except (json.JSONDecodeError, IOError) as e:
                log.warning(f"  Failed to load {fname}: {e}")
    return logs


def load_config(config_path=None):
    """Load configuration from YAML file, falling back to defaults.

    Returns a dict with all configuration values. Missing keys use module-level defaults.
    """
    import yaml

    defaults = {
        "problem": "advecdiff",
        "domain": {"dx_bounds": list(DX_BOUNDS), "dy_bounds": list(DY_BOUNDS)},
        "normalizer": "log",
        "design": {"n_init": N_INIT, "n_steps": N_STEPS, "n_cand": N_CAND, "seed": SEED},
        "gp": {"kernel": KERNEL, "pod_modes": POD_MODES, "mc_samples": MC_SAMPLES,
               "reference": "karcher"},
        "test_grid": {"n_grid": N_TEST_GRID, "n_rand": N_TEST_RAND},
        "acquisition": {"preset": "original"},
        "methods": ["proposed_matern52", "proposed_cgp_ktheta_r", "maxpro", "greedy_replay"],
        "execution": {"workers": 1},
    }

    if config_path is None:
        # Auto-detect: look for config.yaml in project root
        default_path = ROOT / "config.yaml"
        if default_path.exists():
            config_path = default_path

    if config_path is not None:
        config_path = pathlib.Path(config_path)
        if not config_path.exists():
            log.warning(f"Config file not found: {config_path}, using defaults")
            return defaults
        log.info(f"Loading config from: {config_path}")
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f) or {}
        # Deep merge: user values override defaults
        for section in defaults:
            if section in user_cfg:
                if isinstance(defaults[section], dict) and isinstance(user_cfg[section], dict):
                    defaults[section].update(user_cfg[section])
                else:
                    defaults[section] = user_cfg[section]
    return defaults


def parse_args():
    p = argparse.ArgumentParser(description="Run sequential design comparison experiments.")
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config file (default: auto-detect config.yaml in project root)")
    p.add_argument("--workers", type=int, default=None,
                   help="Max parallel methods")
    p.add_argument("--steps", type=int, default=None,
                   help="Number of sequential steps")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Number of points to add per step (default: 1)")
    p.add_argument("--plot-only", action="store_true",
                   help="Plot from existing logs without re-running")
    p.add_argument("--methods", nargs="+", default=None,
                   help="Which methods to run. "
                        "Choices: proposed_matern52, proposed_cgp_ktheta_r, maxpro, greedy_replay")
    p.add_argument("--acq", type=str, default=None,
                   help="Acquisition preset: original, rank_additive, variance_only, space_filling")
    p.add_argument("--normalizer", type=str, choices=["log", "zscore"], default=None,
                   help="Input normalization: log (log10+zscore) or zscore (linear)")
    p.add_argument("--n-test-grid", type=int, default=None,
                   help="Test grid size per axis (e.g. 4 -> 4x4=16 points)")
    p.add_argument("--n-test-rand", type=int, default=None,
                   help="Number of random test points")
    p.add_argument("--n-cand", type=int, default=None,
                   help="Number of Sobol candidates per step")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed (controls initial design, test grid, candidates)")
    p.add_argument("--mc-seed", type=int, default=None,
                   help="Separate seed for MC sampling only. If not set, uses --seed.")
    p.add_argument("--pod-modes", type=int, default=None,
                   help="POD rank (number of modes)")
    p.add_argument("--mc-samples", type=int, default=None,
                   help="MC samples for manifold variance estimation")
    p.add_argument("--eval-interval", type=int, default=1,
                   help="GP evaluation interval for maxpro/random (build GP + all metrics every N steps). "
                        "Default=1 (every step). Proposed always builds GP.")
    p.add_argument("--results-dir", type=str, default=None,
                   help="Output directory for results (default: experiments/results)")
    p.add_argument("--setup-only", action="store_true",
                   help="Compute shared resources (init_seeds, test_grid) and exit. "
                        "Saves to results-dir/shared/ for later --load-shared runs.")
    p.add_argument("--load-shared", type=str, default=None,
                   help="Load shared resources (init_seeds, test_grid, test_imgs) from "
                        "this directory instead of recomputing. Enables modular runs.")
    p.add_argument("--greedy-log-path", type=str, default=None,
                   help="Path to old greedy_log.json for greedy_replay method. "
                        "The parent directory must also contain train_meta.csv with initial points.")
    return p.parse_args()


# ===================== MAIN =====================

def main():
    global _STOP_EVENT, LOG_SPACE, N_TEST_GRID, N_TEST_RAND, N_CAND
    global DX_BOUNDS, DY_BOUNDS, N_INIT, N_STEPS, SEED, MC_SEED, KERNEL, POD_MODES, MC_SAMPLES
    global PROBLEM, KL_BASIS, XI_BOUNDS, RESTART_INTERVAL, WEIGHTED_GP
    global DARCY_T_FINAL, DARCY_DT

    args = parse_args()

    # Load config file (auto-detects config.yaml if --config not given)
    cfg = load_config(args.config)

    # Apply config values (CLI args override config file)
    PROBLEM = cfg.get("problem", "advecdiff")
    if PROBLEM == "darcy":
        # Darcy-specific config
        dom = cfg["domain"]
        xi_lo, xi_hi = dom["xi_bounds"]
        n_kl = dom["n_kl"]
        XI_BOUNDS = tuple((xi_lo, xi_hi) for _ in range(n_kl))
        from pgp.kl_expansion import build_kl_2d
        KL_BASIS = build_kl_2d(dom["correlation_length"], n_kl, dom["nx"], dom["ny"])
        DARCY_T_FINAL = dom.get("T_final", 2.0)
        DARCY_DT = dom.get("dt", 0.02)
        log.info(f"Darcy flow: {n_kl}D KL, xi∈[{xi_lo}, {xi_hi}], "
                 f"T_final={DARCY_T_FINAL}, dt={DARCY_DT}, "
                 f"eigenvalues={KL_BASIS['lambdas']}")
    else:
        DX_BOUNDS = tuple(cfg["domain"]["dx_bounds"])
        DY_BOUNDS = tuple(cfg["domain"]["dy_bounds"])

    N_INIT = cfg["design"]["n_init"]
    N_STEPS = cfg["design"]["n_steps"]
    N_CAND = cfg["design"]["n_cand"]
    SEED = cfg["design"]["seed"]
    KERNEL = cfg["gp"]["kernel"]
    POD_MODES = cfg["gp"]["pod_modes"]
    MC_SAMPLES = cfg["gp"]["mc_samples"]
    RESTART_INTERVAL = cfg["gp"].get("restart_interval", 10)
    WEIGHTED_GP = cfg["gp"].get("weighted", False)
    N_TEST_GRID = cfg["test_grid"]["n_grid"]
    N_TEST_RAND = cfg["test_grid"]["n_rand"]

    # CLI overrides (highest priority)
    normalizer = args.normalizer or cfg.get("normalizer", "log")
    if normalizer == "zscore":
        LOG_SPACE = False
    else:
        LOG_SPACE = True

    if args.steps is not None:
        N_STEPS = args.steps
    if args.n_test_grid is not None:
        N_TEST_GRID = args.n_test_grid
    if args.n_test_rand is not None:
        N_TEST_RAND = args.n_test_rand
    if args.n_cand is not None:
        N_CAND = args.n_cand
    if args.seed is not None:
        SEED = args.seed
    if args.pod_modes is not None:
        POD_MODES = args.pod_modes
    if args.mc_samples is not None:
        MC_SAMPLES = args.mc_samples
    # MC_SEED: separate seed for MC sampling (defaults to SEED)
    MC_SEED = args.mc_seed if args.mc_seed is not None else SEED
    eval_interval = args.eval_interval

    n_steps = N_STEPS
    batch_size = cfg["design"].get("batch_size", 1)
    if args.batch_size is not None:
        batch_size = args.batch_size
    log.info(f"Normalization: {'ZScore (linear)' if not LOG_SPACE else 'LogStandardizer (log10 + z-score)'}")
    log.info(f"Seeds: experiment={SEED}, mc_sampling={MC_SEED}")
    log.info(f"POD modes: {POD_MODES}")
    log.info(f"Test grid: {N_TEST_GRID}^{get_param_dim()}={N_TEST_GRID**get_param_dim()} grid + {N_TEST_RAND} random, "
             f"candidates/step: {N_CAND}, steps: {n_steps}, batch_size: {batch_size}")

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if args.results_dir:
        results_dir = pathlib.Path(args.results_dir)
    else:
        results_dir = EXP_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # --plot-only mode: just load existing logs and plot
    if args.plot_only:
        log.info("Plot-only mode: loading existing logs...")
        logs_dict = load_logs_from_disk(results_dir)
        if not logs_dict:
            log.error("No logs found. Run experiments first.")
            return
        plot_comparison(logs_dict, results_dir)
        log.info("Plots generated from existing logs.")
        return

    # Mutual exclusion check
    if args.setup_only and args.load_shared:
        log.error("Cannot use --setup-only and --load-shared together.")
        return

    ensure_precompute()
    if PROBLEM == "advecdiff":
        assert SIM_R.exists(), f"simulate_advecdiff.R not found at {SIM_R}"

    if args.load_shared:
        # ===== Load pre-computed shared resources =====
        shared_dir = pathlib.Path(args.load_shared)
        log.info(f"Loading shared resources from: {shared_dir}")
        init_seeds = np.load(shared_dir / "init_seeds.npy")
        test_grid  = np.load(shared_dir / "test_grid.npy")
        test_imgs  = np.load(shared_dir / "test_imgs.npy")
        log.info(f"  init_seeds: {init_seeds.shape}, test_grid: {test_grid.shape}, "
                 f"test_imgs: {test_imgs.shape}")
        # Validate config consistency
        meta_path = shared_dir / "config_snapshot.json"
        if meta_path.exists():
            with open(meta_path) as f:
                saved_meta = json.load(f)
            if saved_meta.get("seed") != SEED:
                log.warning(f"Seed mismatch: saved={saved_meta['seed']}, current={SEED}")
            if saved_meta.get("pod_modes") != POD_MODES:
                log.warning(f"POD modes mismatch: saved={saved_meta['pod_modes']}, current={POD_MODES}")
            if saved_meta.get("log_space") != LOG_SPACE:
                log.warning(f"LOG_SPACE mismatch: saved={saved_meta['log_space']}, current={LOG_SPACE}")
    else:
        # ===== Compute shared resources from scratch =====
        # Shared initial seeds (computed first — needed for adaptive test grid)
        init_seeds = lhs_init(N_INIT, get_bounds(), LOG_SPACE, SEED)
        log.info(f"Shared initial seeds:\n{init_seeds}")

        # Simulate FOM for initial seeds (needed to fit GP for adaptive test grid)
        init_snap_dir = results_dir / "init_seeds" / "snapshots"
        init_imgs = _simulate_batch(init_seeds, init_snap_dir)
        log.info(f"Initial seed snapshots: shape {init_imgs.shape}")

        # Generate adaptive test grid (GP-gradient-informed, shared by all methods)
        test_snap_dir = results_dir / "test_grid" / "snapshots"
        test_grid, test_imgs = generate_adaptive_test_grid(test_snap_dir, init_seeds, init_imgs)
        log.info(f"Test grid: {test_grid.shape[0]} points, snapshots: {test_imgs.shape}")

        # Save shared resources for modular runs
        shared_dir = results_dir / "shared"
        shared_dir.mkdir(parents=True, exist_ok=True)
        np.save(shared_dir / "init_seeds.npy", init_seeds)
        np.save(shared_dir / "test_grid.npy", test_grid)
        np.save(shared_dir / "test_imgs.npy", test_imgs)
        config_snapshot = {
            "seed": SEED, "pod_modes": POD_MODES, "log_space": LOG_SPACE,
            "n_init": N_INIT, "n_steps": N_STEPS, "n_cand": N_CAND,
            "kernel": KERNEL, "mc_samples": MC_SAMPLES,
            "problem": PROBLEM,
            "bounds": [list(b) for b in get_bounds()],
            "n_test_grid": N_TEST_GRID, "n_test_rand": N_TEST_RAND,
        }
        with open(shared_dir / "config_snapshot.json", "w") as f:
            json.dump(config_snapshot, f, indent=2)
        log.info(f"Saved shared resources to: {shared_dir}")

    # --setup-only mode: exit after saving shared resources
    if args.setup_only:
        log.info("--setup-only mode: shared resources saved. Exiting.")
        return

    # Acquisition config: preset from CLI > config file > default
    acq_preset_name = args.acq or cfg["acquisition"].get("preset", "original")
    acq_presets = {
        "original": AcquisitionConfig.original,
        "rank_additive": AcquisitionConfig.rank_additive,
        "variance_only": AcquisitionConfig.variance_only,
        "space_filling": AcquisitionConfig.space_filling,
        "variance_focused": AcquisitionConfig.variance_focused,
        "pure_mmv": AcquisitionConfig.pure_mmv,
    }
    if acq_preset_name == "custom":
        # Build from config file's acquisition section
        acq_cfg = cfg["acquisition"]
        acq_config = AcquisitionConfig(
            mode=acq_cfg.get("mode", "rank_additive"),
            smooth_h=acq_cfg.get("smooth_h", 0.05),
            variance=acq_cfg.get("variance", {"enabled": True, "weight": 0.4}),
            diversity=acq_cfg.get("diversity", {"enabled": True, "weight": 0.3, "beta": 0.5}),
            repulsion=acq_cfg.get("repulsion", {"enabled": True, "weight": 0.2, "scale": 2.0}),
            interior=acq_cfg.get("interior", {"enabled": True, "weight": 0.1, "margin": 0.08}),
        )
    else:
        acq_config = acq_presets.get(acq_preset_name, AcquisitionConfig.original)()
    # Sync acquisition log_space with normalizer choice
    acq_config.log_space = LOG_SPACE
    log.info(f"Acquisition config: {acq_preset_name} ({acq_config.mode}, log_space={LOG_SPACE})")

    # Setup stop event for graceful interruption
    _STOP_EVENT = threading.Event()

    def sigint_handler(sig, frame):
        log.warning("\nSIGINT received — stopping gracefully after current iteration...")
        _STOP_EVENT.set()
    signal.signal(signal.SIGINT, sigint_handler)

    # Pre-simulate init snapshots ONCE and share across all methods
    shared_init_dir = results_dir / "shared" / "init_snapshots"
    shared_init_dir.mkdir(parents=True, exist_ok=True)
    # If loading from existing shared dir, copy init snapshots instead of re-simulating
    if args.load_shared:
        import shutil
        src_init_dir = pathlib.Path(args.load_shared) / "init_snapshots"
        if src_init_dir.exists():
            for f in src_init_dir.iterdir():
                dst = shared_init_dir / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
            log.info(f"Copied init snapshots from {src_init_dir}")
    for s in init_seeds:
        out_path = shared_init_dir / _param_filename(s)
        if not out_path.exists():
            simulate(s, shared_init_dir)
    log.info(f"Shared init snapshots in: {shared_init_dir}")

    # Define all available methods
    ref_method = cfg["gp"].get("reference", "karcher")
    common_kw = dict(test_tau=test_grid, test_imgs=test_imgs, n_steps=n_steps,
                     n_cand=N_CAND, shared_init_dir=shared_init_dir,
                     batch_size=batch_size)
    all_methods = {
        # Proposed pGP-MMV with a stationary Matérn-5/2 kernel
        "proposed_matern52": (run_proposed, dict(
            init_seeds=init_seeds,
            snap_dir=results_dir / "proposed_matern52" / "snapshots",
            out_dir=results_dir / "proposed_matern52",
            kernel="matern52", ref_method=ref_method, acq_config=acq_config,
            **common_kw
        )),
        # Proposed pGP-MMV with the nonstationary CGP kernel (Ba & Joseph, 2012)
        "proposed_cgp_ktheta_r": (run_proposed, dict(
            init_seeds=init_seeds,
            snap_dir=results_dir / "proposed_cgp_ktheta_r" / "snapshots",
            out_dir=results_dir / "proposed_cgp_ktheta_r",
            kernel="cgp", opt_style="rstyle",
            ref_method=ref_method, acq_config=acq_config,
            **common_kw
        )),
        # Baseline: space-filling MaxPro design
        "maxpro": (run_maxpro, dict(
            init_seeds=init_seeds,
            snap_dir=results_dir / "maxpro" / "snapshots",
            out_dir=results_dir / "maxpro",
            eval_interval=eval_interval, **common_kw
        )),
        # Baseline: model-constrained greedy (MC Greedy)
        "greedy_replay": (run_greedy_replay, dict(
            init_seeds_unused=init_seeds,
            snap_dir=results_dir / "greedy_replay" / "snapshots",
            out_dir=results_dir / "greedy_replay",
            greedy_log_path=args.greedy_log_path if hasattr(args, 'greedy_log_path') else None,
            eval_interval=eval_interval, **common_kw
        )),
    }

    # Filter methods: CLI > config > all
    method_list = args.methods or cfg.get("methods")
    if method_list:
        methods = {k: v for k, v in all_methods.items() if k in method_list}
    else:
        methods = all_methods

    log.info(f"Running {len(methods)} methods: {list(methods.keys())}")
    t0 = time.time()

    # Parallel execution
    logs_dict = {}
    display_names = {
        "proposed_matern52": "Proposed (Matérn)",
        "proposed_cgp_ktheta_r": "Proposed (CGP)",
        "maxpro": "MaxPro",
        "greedy_replay": "MC Greedy",
    }

    # Run methods sequentially (fork/spawn unsafe with rpy2 + PyTorch on macOS)
    for method_key, (func, kw) in methods.items():
        if _STOP_EVENT.is_set():
            log.info(f"Skipping {method_key} (interrupted)")
            break
        log.info(f"--- Starting: {method_key} ---")
        name, result_log = _run_method(method_key, func, kw)
        display = display_names.get(name, name)
        logs_dict[display] = result_log
        log.info(f"  {display}: completed ({len(result_log)} iterations)")

    total_time = time.time() - t0
    log.info(f"Total experiment time: {total_time:.1f}s")

    # Save final logs
    for method_key in methods:
        display = display_names.get(method_key, method_key)
        if display in logs_dict:
            with open(results_dir / f"{method_key}_log.json", "w") as f:
                json.dump(logs_dict[display], f, indent=2)

    # Auto-generate diagnostic plots (hyperparameters, Karcher/ball)
    _run_auto_diagnostics(results_dir, methods, display_names, logs_dict, cfg)

    # Load any other method logs from disk (e.g. when proposed and maxpro
    # run in separate screen sessions, merge the other's saved JSON).
    for method_key, display in display_names.items():
        if display not in logs_dict:
            log_path = results_dir / f"{method_key}_log.json"
            if log_path.exists():
                with open(log_path) as f:
                    data = json.load(f)
                if data:
                    logs_dict[display] = data
                    log.info(f"  Loaded {display} log from disk ({len(data)} iters)")

    # Generate comparison plots
    plot_comparison(logs_dict, results_dir)

    if _STOP_EVENT.is_set():
        log.info("Experiment interrupted — partial results plotted.")
    else:
        log.info("Experiment complete!")
    log.info(f"Results in: {results_dir}")


if __name__ == "__main__":
    main()
