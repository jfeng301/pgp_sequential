# Figure data

Per-run iteration logs behind the paper figures. Each `experimentN/` is one
independent repeated run (an anonymized random seed — the seed integers are not
recorded). Aggregating across the `experiment*` folders reproduces the mean ± std
curves in the paper.

```
advecdiff_2d/   5 experiments   (2D advection-diffusion)
darcy_5d/       4 experiments   (5D Darcy flow)
```

Each `experimentN/` contains one `*_log.json` per method:

| file                | method                                             |
|---------------------|----------------------------------------------------|
| `cgp_log.json`      | Proposed pGP-MMV, nonstationary CGP kernel         |
| `matern_log.json`   | Proposed pGP-MMV, stationary Matérn-5/2 kernel     |
| `maxpro_log.json`   | MaxPro space-filling baseline                       |
| `mc_greedy_log.json`| Model-constrained greedy (2D only; single run)     |

Each log is a JSON list with one record per sequential step:

| key         | meaning                                             |
|-------------|-----------------------------------------------------|
| `iter`      | sequential step index                               |
| `picked_params` | parameter value(s) added at this step (design point coordinates) |
| `mmv`       | max manifold variance (acquisition objective)       |
| `proj_rmse` | projection RMSE on the held-out test set            |
| `geo_pred`  | mean geodesic prediction error on the test set      |
| `pod_angle` | max principal-angle change of the POD basis (deg)   |
| `hypers`    | fitted GP hyperparameters at this step              |

The logs contain **no seed information** — only metrics, hyperparameters, and
design-point coordinates.

Regenerate the metric figures with:

```bash
python ../plot_figures.py
```

The snapshot figures (`fig2_*`, `fig5_*`) and the design-scatter (`fig3_*`) are
provided as static PNGs; they depend on full snapshot / design data not shipped here.
