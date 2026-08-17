"""
Regenerate the paper metric figures from the anonymized per-run logs in data/.

Reads data/<problem>/experiment*/<method>_log.json, aggregates across experiments
(mean +/- std), and writes fig4_metrics_2d.png and fig6_metrics_5d.png.

The snapshot figures (fig2, fig5) and the design-scatter (fig3) are provided as
static PNGs — they depend on full snapshot / design data not shipped here.

No seed information is used or produced: experiments are anonymized directories.
"""
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

STYLE = {  # method file-stem -> (label, color, marker)
    "cgp":       ("Proposed (CGP)",    "#1f77b4", "o"),
    "matern":    ("Proposed (Matérn)", "#2ca02c", "v"),
    "maxpro":    ("MaxPro",            "#d62728", "s"),
    "mc_greedy": ("MC Greedy",         "#8c564b", "D"),
}
METRICS = [
    ("mmv",       "Max Manifold Variance (normalized)", True),   # True -> normalize
    ("proj_rmse", "Projection RMSE (test)",             False),
    ("geo_pred",  "Geodesic Prediction Error",          False),
    ("pod_angle", "POD Angle Change (degrees)",         False),
]


def load_runs(problem, method):
    """Return list of per-experiment record-lists for one method."""
    runs = []
    for d in sorted(glob.glob(str(DATA / problem / "experiment*"))):
        f = Path(d) / f"{method}_log.json"
        if f.exists():
            runs.append(json.load(open(f)))
    return runs


def curve(runs, key, normalize):
    """Aggregate mean/std across experiments for one metric."""
    arrs = [[r.get(key, np.nan) for r in run] for run in runs]
    L = min(len(a) for a in arrs)
    A = np.array([a[:L] for a in arrs], dtype=float)
    mean, std = np.nanmean(A, axis=0), np.nanstd(A, axis=0)
    if normalize:
        mean, std = mean / mean[0], std / mean[0]
    return np.arange(1, L + 1), mean, std


def make_metrics_figure(problem, methods, title, out):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (key, ylab, norm) in zip(axes.ravel(), METRICS):
        for m in methods:
            runs = load_runs(problem, m)
            if not runs:
                continue
            label, color, marker = STYLE[m]
            x, mean, std = curve(runs, key, norm)
            ax.plot(x, mean, marker=marker, ms=4, lw=1.8, color=color, label=label)
            if len(runs) > 1:                       # no band for single-run methods
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_xlabel("Iteration (# added points)")
        ax.set_ylabel(ylab)
        ax.set_title(ylab)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(HERE / out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    n2d = len(glob.glob(str(DATA / "advecdiff_2d" / "experiment*")))
    n5d = len(glob.glob(str(DATA / "darcy_5d" / "experiment*")))
    make_metrics_figure(
        "advecdiff_2d", ["cgp", "matern", "maxpro", "mc_greedy"],
        f"2D advection-diffusion ({n2d} experiments, N_init=5, 30 steps, POD=5, MC=100)",
        "fig4_metrics_2d.png")
    make_metrics_figure(
        "darcy_5d", ["matern", "cgp", "maxpro"],
        f"5D Darcy flow ({n5d} experiments, N_init=100, 50 steps, POD=3, MC=500)",
        "fig6_metrics_5d.png")
