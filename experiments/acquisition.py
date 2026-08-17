"""
Modular acquisition function system for pGP sequential design.

Provides pluggable acquisition components that can be freely composed:
  - variance:  manifold variance from GP posterior (uncertainty reduction)
  - diversity: min geodesic distance from predicted subspace to training subspaces
  - repulsion: distance-based penalty to avoid clustering (HRK-inspired)
  - interior:  soft boundary dampening

Composition modes:
  - "multiplicative": original — product of all components
  - "additive":       normalize each to [0,1], weighted sum
  - "rank_additive":  rank-transform each to [0,1], weighted sum (most robust)

Usage:
    from experiments.acquisition import AcquisitionConfig, build_acquisition

    # Use a preset
    config = AcquisitionConfig.rank_additive()

    # Or customize
    config = AcquisitionConfig(
        mode="rank_additive",
        smooth_h=0.05,
        variance={"enabled": True, "weight": 0.4},
        diversity={"enabled": True, "weight": 0.3, "beta": 0.5},
        repulsion={"enabled": True, "weight": 0.2, "scale": 2.0},
        interior={"enabled": True, "weight": 0.1, "margin": 0.08},
    )

    # In the sequential design loop:
    scores = build_acquisition(cands, gp, scaler, tau_existing, bounds, config,
                               compute_manifold_var_fn, compute_diversity_bonus_fn)
    idx = int(np.argmax(scores))
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_COMPONENTS = {
    "variance":  {"enabled": True, "weight": 1.0},
    "diversity": {"enabled": True, "weight": 1.0, "beta": 0.5},
    "repulsion": {"enabled": True, "weight": 1.0, "scale": 1.0},
    "interior":  {"enabled": True, "weight": 1.0, "margin": 0.03},
}


@dataclass
class AcquisitionConfig:
    """Configuration for acquisition function composition.

    Parameters
    ----------
    mode : str
        How to combine components:
        - "multiplicative" (original): product of all components.
        - "additive": min-max normalize each to [0,1], then weighted sum.
        - "rank_additive": rank-percentile transform each, then weighted sum.
    smooth_h : float
        Gaussian smoothing bandwidth in [0,1]-normalized domain.
        h=0.05 means 5% of domain width. Applied after normalizing
        coordinates to [0,1] using domain bounds.
    log_space : bool
        Whether candidates / existing points are in log-space (deprecated, kept for compat).
    variance : dict
        Config for variance component. Keys: enabled, weight.
    diversity : dict
        Config for diversity component. Keys: enabled, weight, beta.
    repulsion : dict
        Config for repulsion component. Keys: enabled, weight, scale.
    interior : dict
        Config for interior component. Keys: enabled, weight, margin.
    """
    mode: str = "multiplicative"
    smooth_h: float = 0.05
    log_space: bool = False
    variance: dict = field(default_factory=lambda: {"enabled": True, "weight": 1.0})
    diversity: dict = field(default_factory=lambda: {"enabled": True, "weight": 1.0, "beta": 0.5})
    repulsion: dict = field(default_factory=lambda: {"enabled": True, "weight": 1.0, "scale": 1.0})
    interior: dict = field(default_factory=lambda: {"enabled": True, "weight": 1.0, "margin": 0.03})


    # --- Presets ---

    @classmethod
    def original(cls) -> "AcquisitionConfig":
        """Original multiplicative acquisition (backward compatible)."""
        return cls(
            mode="multiplicative",
            smooth_h=0.05,
            variance={"enabled": True, "weight": 1.0},
            diversity={"enabled": True, "weight": 1.0, "beta": 0.5},
            repulsion={"enabled": True, "weight": 1.0, "scale": 1.0},
            interior={"enabled": True, "weight": 1.0, "margin": 0.03},
        )

    @classmethod
    def rank_additive(cls) -> "AcquisitionConfig":
        """Rank-additive with stronger boundary avoidance."""
        return cls(
            mode="rank_additive",
            smooth_h=0.05,
            variance={"enabled": True, "weight": 0.4},
            diversity={"enabled": True, "weight": 0.3, "beta": 0.5},
            repulsion={"enabled": True, "weight": 0.2, "scale": 2.0},
            interior={"enabled": True, "weight": 0.1, "margin": 0.08},
        )

    @classmethod
    def variance_only(cls) -> "AcquisitionConfig":
        """Pure variance acquisition (ablation baseline)."""
        return cls(
            mode="multiplicative",
            smooth_h=0.05,
            variance={"enabled": True, "weight": 1.0},
            diversity={"enabled": False, "weight": 0.0},
            repulsion={"enabled": False, "weight": 0.0},
            interior={"enabled": True, "weight": 1.0, "margin": 0.05},
        )

    @classmethod
    def pure_mmv(cls) -> "AcquisitionConfig":
        """Pure manifold variance — zero regularization.

        Uses ONLY the smoothed manifold variance from GP posterior MC sampling.
        All regularization (diversity, repulsion, interior) disabled.
        Smoothing (smooth_h=0.05) is kept to reduce MC noise.
        """
        return cls(
            mode="multiplicative",
            smooth_h=0.05,
            variance={"enabled": True, "weight": 1.0},
            diversity={"enabled": False, "weight": 0.0},
            repulsion={"enabled": False, "weight": 0.0},
            interior={"enabled": False, "weight": 0.0},
        )

    @classmethod
    def variance_focused(cls) -> "AcquisitionConfig":
        """Variance-focused: heavy variance + repulsion, no diversity.

        Improves proj_RMSE by focusing on uncertainty reduction + parameter-space
        coverage (repulsion) rather than manifold geodesic distance (diversity).
        """
        return cls(
            mode="rank_additive",
            smooth_h=0.05,
            variance={"enabled": True, "weight": 0.6},
            diversity={"enabled": False, "weight": 0.0},
            repulsion={"enabled": True, "weight": 0.3, "scale": 2.0},
            interior={"enabled": True, "weight": 0.1, "margin": 0.08},
        )

    @classmethod
    def space_filling(cls) -> "AcquisitionConfig":
        """Heavy repulsion + interior, minimal variance."""
        return cls(
            mode="rank_additive",
            smooth_h=0.05,
            variance={"enabled": True, "weight": 0.2},
            diversity={"enabled": True, "weight": 0.2, "beta": 0.5},
            repulsion={"enabled": True, "weight": 0.4, "scale": 3.0},
            interior={"enabled": True, "weight": 0.2, "margin": 0.10},
        )

    def to_dict(self) -> dict:
        """Serialize to dict (for JSON logging)."""
        return {
            "mode": self.mode,
            "smooth_h": self.smooth_h,
            "log_space": self.log_space,
            "variance": dict(self.variance),
            "diversity": dict(self.diversity),
            "repulsion": dict(self.repulsion),
            "interior": dict(self.interior),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AcquisitionConfig":
        """Deserialize from dict."""
        return cls(**d)


# ---------------------------------------------------------------------------
# Component functions
# ---------------------------------------------------------------------------

def gaussian_smoother(X: np.ndarray, v: np.ndarray, h: float) -> np.ndarray:
    """Nadaraya-Watson Gaussian kernel smoother.

    X should be in [0,1]-normalized coordinates so that h is interpretable
    as a fraction of the domain width (e.g. h=0.05 → 5% bandwidth).
    """
    D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    K = np.exp(-0.5 * D2 / max(h * h, 1e-24))
    denom = K.sum(axis=1, keepdims=True) + 1e-12
    return (K @ v.reshape(-1, 1) / denom).ravel()


def _normalize_to_unit(pts: np.ndarray, bounds: tuple,
                       log_space: bool = False) -> np.ndarray:
    """Normalize points to [0,1] using domain bounds.

    This ensures smoothing bandwidth h is domain-independent:
    h=0.05 always means 5% of the domain width regardless of scale.

    When log_space=True, normalizes in log10 space so that equal
    ratios map to equal distances (consistent with LogStandardizer).
    """
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    if log_space:
        log_pts = np.log10(np.maximum(pts, 1e-30))
        log_lo = np.log10(np.maximum(lo, 1e-30))
        log_hi = np.log10(np.maximum(hi, 1e-30))
        return (log_pts - log_lo) / (log_hi - log_lo + 1e-30)
    return (pts - lo) / (hi - lo + 1e-30)


def compute_variance_score(cands: np.ndarray, gp, scaler, bounds: tuple,
                           config: AcquisitionConfig,
                           manifold_var_fn: Callable) -> np.ndarray:
    """Manifold variance from pGP posterior, smoothed."""
    v = manifold_var_fn(gp, scaler, cands)
    base = _normalize_to_unit(cands, bounds, log_space=config.log_space)
    return gaussian_smoother(base, v, config.smooth_h)


def compute_diversity_score(cands: np.ndarray, gp, scaler, bounds: tuple,
                            config: AcquisitionConfig,
                            diversity_fn: Callable) -> np.ndarray:
    """Min geodesic distance from predicted subspace to training subspaces, smoothed."""
    diversity = diversity_fn(gp, scaler, cands)
    base = _normalize_to_unit(cands, bounds, log_space=config.log_space)
    return gaussian_smoother(base, diversity, config.smooth_h)


def compute_repulsion_score(cands: np.ndarray, tau_existing: np.ndarray,
                            bounds: tuple,
                            config: AcquisitionConfig) -> np.ndarray:
    """Distance-based repulsion from existing training points."""
    base = _normalize_to_unit(cands, bounds, log_space=config.log_space)
    seen = _normalize_to_unit(tau_existing, bounds, log_space=config.log_space)
    dist2 = ((base[:, None, :] - seen[None, :, :]) ** 2).sum(axis=2).min(axis=1)
    h = config.smooth_h * config.repulsion.get("scale", 1.0)
    return 1.0 - np.exp(-dist2 / (2 * h * h))


def compute_interior_score(cands: np.ndarray, bounds: tuple,
                           config: AcquisitionConfig) -> np.ndarray:
    """Soft boundary dampening: penalizes points near domain edges.

    Operates in [0,1]-normalized space (log space when config.log_space=True).
    Returns values in [floor, 1.0] — edge points are down-weighted but never
    fully excluded.

    Config keys (via config.interior):
        margin : float  — fraction of domain width defining the penalty zone (default 0.05)
        floor  : float  — minimum weight at the boundary (default 0.1)
    """
    margin = config.interior.get("margin", 0.05)
    floor = config.interior.get("floor", 0.1)
    frac = _normalize_to_unit(cands, bounds, log_space=config.log_space)
    bdist = np.minimum(frac, 1.0 - frac).min(axis=1)  # in [0, 0.5]
    raw = np.clip(bdist / margin, 0.0, 1.0)
    return floor + (1.0 - floor) * raw


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def rank_transform(v: np.ndarray) -> np.ndarray:
    """Rank-percentile transform to [0, 1]."""
    n = len(v)
    if n <= 1:
        return np.ones(n)
    return np.argsort(np.argsort(v)).astype(float) / (n - 1)


def minmax_normalize(v: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    vmin, vmax = v.min(), v.max()
    rng = vmax - vmin
    if rng < 1e-12:
        return np.ones_like(v)
    return (v - vmin) / rng


def compose_multiplicative(components: dict, config: AcquisitionConfig) -> np.ndarray:
    """Original multiplicative composition."""
    n = len(next(iter(components.values())))
    acq = np.ones(n)
    for name, values in components.items():
        if name == "variance":
            acq *= values
        elif name == "diversity":
            beta = config.diversity.get("beta", 0.5)
            acq *= (values + 0.01) ** beta
        else:
            acq *= values
    return acq


def compose_additive(components: dict, config: AcquisitionConfig) -> np.ndarray:
    """Additive composition: min-max normalize each, then weighted sum."""
    n = len(next(iter(components.values())))
    total = np.zeros(n)
    weight_map = {
        "variance": config.variance.get("weight", 1.0),
        "diversity": config.diversity.get("weight", 1.0),
        "repulsion": config.repulsion.get("weight", 1.0),
        "interior": config.interior.get("weight", 1.0),
    }
    for name, values in components.items():
        w = weight_map.get(name, 1.0)
        total += w * minmax_normalize(values)
    return total


def compose_rank_additive(components: dict, config: AcquisitionConfig) -> np.ndarray:
    """Rank-additive: rank-transform each, then weighted sum (most robust)."""
    n = len(next(iter(components.values())))
    total = np.zeros(n)
    weight_map = {
        "variance": config.variance.get("weight", 1.0),
        "diversity": config.diversity.get("weight", 1.0),
        "repulsion": config.repulsion.get("weight", 1.0),
        "interior": config.interior.get("weight", 1.0),
    }
    for name, values in components.items():
        w = weight_map.get(name, 1.0)
        total += w * rank_transform(values)
    return total


_COMPOSE_FNS = {
    "multiplicative": compose_multiplicative,
    "additive": compose_additive,
    "rank_additive": compose_rank_additive,
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_acquisition(
    cands: np.ndarray,
    gp,
    scaler,
    tau_existing: np.ndarray,
    bounds: tuple,
    config: AcquisitionConfig,
    manifold_var_fn: Callable,
    diversity_fn: Optional[Callable] = None,
) -> np.ndarray:
    """
    Build the composite acquisition function.

    Parameters
    ----------
    cands : (N_cand, d) candidate points (original scale, not log)
    gp : projectGP instance with fitted hyperparameters
    scaler : data normalizer (LogStandardizer or ZScoreNormalizer)
    tau_existing : (k, d) existing training parameters (original scale)
    bounds : tuple of (lo, hi) per dimension, e.g. ((1e-4, 1e-2), (1e-4, 1e-2))
    config : AcquisitionConfig
    manifold_var_fn : callable(gp, scaler, cands) -> np.ndarray of shape (N_cand,)
    diversity_fn : callable(gp, scaler, cands) -> np.ndarray of shape (N_cand,)
        If None and diversity is enabled, it will be skipped with a warning.

    Returns
    -------
    scores : (N_cand,) acquisition scores (higher = better)
    """
    components = {}

    if config.variance.get("enabled", True):
        components["variance"] = compute_variance_score(cands, gp, scaler, bounds, config, manifold_var_fn)

    if config.diversity.get("enabled", True):
        if diversity_fn is not None:
            components["diversity"] = compute_diversity_score(cands, gp, scaler, bounds, config, diversity_fn)

    if config.repulsion.get("enabled", True):
        components["repulsion"] = compute_repulsion_score(cands, tau_existing, bounds, config)

    if config.interior.get("enabled", True):
        components["interior"] = compute_interior_score(cands, bounds, config)

    if not components:
        return np.ones(len(cands))

    compose_fn = _COMPOSE_FNS.get(config.mode, compose_multiplicative)
    scores = compose_fn(components, config)
    return scores
