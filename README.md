# Extracting neural scaling laws in three ways

A small, self-contained tutorial that reproduces the **three compute-optimal
scaling-law extraction methods** from the Chinchilla paper ([Hoffmann et al.,
2022](https://arxiv.org/abs/2203.15556)) on a synthetic problem you can run on a laptop: MLP
**students of increasing size** trained on a **Gaussian teacher–student** task.

A scaling law predicts how the achievable loss falls with compute[FLOPs] and,
crucially, **how to split that compute** between model size `N` (parameters) and
data `D` (examples). We count MLP compute cost as **`C = (6N − 2·d·w)·D`** (`d` = input
dim, `w` = first hidden width). This repo trains a grid over `(N, D)` and recovers the
compute-optimal frontier `L*(C)` and allocation `(N*(C), D*(C))` in three ways.

| # | method | idea | what it gives |
|---|--------|------|---------------|
| 1 | **Training-curve envelope** | lower envelope of the loss-vs-compute curves | assumption-free `L*(C)`, `N*`, `D*` |
| 2 | **IsoFLOP profiles** | at fixed `C`, sweep `N`; the loss-minimising `N` is `N*(C)` | `N*`, `D*` from parabola minima |
| 3 | **Parametric fit** | fit `L(N,D) = E + A/Nᵅ + B/Dᵝ`, frontier in closed form | full surface + floor `E` |

## The synthetic problem

- **Teacher** — a *fixed, randomly initialised* MLP that **defines the target
  function** (never trained). Inputs are Gaussian `x ~ N(0, I)`.
- **Student** — the MLP we **train**, of growing width (size `N`), to imitate the
  teacher.
- **Targets** — `y = teacher(x) + σ·ε`, label noise `ε ~ N(0,1)`. Data is drawn
  **fresh every step** (single-pass), so `D` = examples seen.

Under the Loss parametric form `L ≈ E + A/Nᵅ + B/Dᵝ`:

- `A/Nᵅ` — a small student cannot represent the teacher (capacity error),
- `B/Dᵝ` — too few examples to pin the function down (data error),
- `E` — label noise no model can predict.

Because the task is synthetic we **know the floor exactly**: `E = σ²`. That lets us
grade the fitted floor from Approach 3 against the truth.

## Quickstart

Install the package:

```bash
pip install -e ".[notebook]"                         # editable install + jupyter
jupyter notebook notebooks/01_scaling_laws.ipynb     # 3 approaches + HP transfer + depth
jupyter notebook notebooks/02_double_descent.ipynb   # double descent (repeated-data regime)
```

Or with [pixi](https://pixi.sh):

```bash
pixi install          # build the env from pyproject.toml (editable install + deps)
pixi run notebooks    # execute both notebooks
```

<details>
<summary><b>Optional — regenerate the data from scratch</b> (only to reproduce the sweeps)</summary>

```bash
# 1) Calibrate the per-cell learning rate (the muP / transfer study, ~10 min CPU).
#    Writes results/hp_study_cosine.json + results/figures/hp_study_cosine.png
python scripts/run_hp_study.py

# 2) Train the (N, D) grid with each cell at its own tuned LR (~45 min CPU; the
#    default grid runs out to 17M examples). Incremental: re-running after
#    extending the data/width range only trains the new cells.
#    Writes results/sweep_cosine.csv, results/sweep_meta.json, results/teacher.pt
python scripts/run_sweep.py                 # or: --preset quick   for a fast smoke test
```
</details>

The split is deliberate: **sweeps run in scripts, visualisation lives in the
notebook.** Re-rendering the notebook is fast and deterministic because it just
reloads the cached CSVs.

## Scaling Laws

![overview](results/figures/01_overview.png)

*Loss vs compute, one curve per model size — small models win at low compute,
large models take over and approach the known floor `E = σ²`.*

| Approach 1 — envelope | Approach 2 — IsoFLOP | Approach 3 — parametric |
|:---:|:---:|:---:|
| ![a1](results/figures/02_approach1_envelope.png) | ![a2](results/figures/03_approach2_isoflop.png) | ![a3](results/figures/04_approach3_parametric.png) |

All three recover near-`√C` scaling (`N*` and `D*` grow together) and broadly
agree on the allocation; Approach 3 additionally recovers the known floor `E`.

### Double descent

The tutorial above is single-pass, so it never overfits and the loss is monotone.
A second notebook reproduces all three classic double descents
([Nakkiran et al., 2019](https://arxiv.org/abs/1912.02292)) on the same teacher–student setup:

- **Sample-wise** (`--mode samples`, vary `D`): test loss peaks near the
  interpolation threshold `D ~ N`, then second-descends.
- **Model-wise** (`--mode width`, vary `W` at fixed `D=10k`): classic U, a sharp peak
  **exactly at `N ~ D`**, then a second descent.
- **Epoch-wise** (`--mode epochs`, fixed `W,D` overparam, train long): test goes
  down (fit signal) → up (fit noise) → down (second descent), while train MSE falls
  monotonically to interpolation.

| sample-wise | model-wise | epoch-wise |
|:---:|:---:|:---:|
| ![](results/figures/double_descent_samples.png) | ![](results/figures/double_descent_width.png) | ![](results/figures/double_descent_epochs.png) |


![phase diagram](results/figures/double_descent_phase.png)


## References

Hoffmann, Borgeaud, Mensch et al., *Training Compute-Optimal Large Language Models*
(2022), [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)

Nakkiran, Kaplun, Bansal, Yang, Barak, Sutskever, *Deep Double Descent: Where Bigger
Models and More Data Hurt* (2019), [arXiv:1912.02292](https://arxiv.org/abs/1912.02292)