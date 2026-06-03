#!/usr/bin/env python
"""Depth study: scale N by adding layers, and compare against the width-only sweep.

Trains student families at depths 1 and 6 (vs the default 2 hidden layers) on the
*same* teacher and compares the scaling law at fixed depth and combined (N grown via
width and depth). Each family mirrors the L=2 sweep's coverage: every width L=2 has,
on exactly the data budgets that L=2 width saw (non-rectangular -- L=2's densified
small widths only went to 2^21).

The optimal LR shifts with depth, so each family
gets its own calibration. Probes (run_cell LR sweeps at w=16/64/128, T=2048) found,
relative to L=2's eta_ref=0.00535 at (w=64, T=2048):
    L=1: eta_ref ~ 0.0057  (~1.2x L=2 -- the depth->LR effect saturates when shallow)
    L=6: eta_ref ~ 0.0017  (~0.3x L=2 -- steep; plain MLPs lack residual rescaling)
Width and T exponents are depth-independent, so only the reference shifts.

Usage:
    python scripts/run_depth_compare.py --dry-run        # preview cells + cost, no training
    python scripts/run_depth_compare.py --depths 1       # train only L=1 (cheap), then plot
    python scripts/run_depth_compare.py                  # train L=1 and L=6, then plot
    python scripts/run_depth_compare.py --plot-only      # just (re)make the figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

from scaling_laws.data import TeacherStudentRegression  # noqa: E402
from scaling_laws.sweep import run_grid, transfer_lr, aggregate  # noqa: E402
from scaling_laws.flops import mlp_param_count, compute  # noqa: E402
from scaling_laws import approaches as ap               # noqa: E402
from scaling_laws import plotting as pl                 # noqa: E402

PROBLEM = dict(input_dim=32, teacher_width=256, teacher_depth=2, teacher_act="gelu",
               noise_std=0.1, val_size=16384, seed=0)
MAIN_DEPTH = 2                   # the depth the notebook uses; only it gets standalone figures
SEEDS = (0, 1)
_EXP = dict(w_ref=64, T_ref=2048, c_w=-0.374, c_T=-0.617)   # shared width/T exponents

# One family per depth. L=2 reuses the main sweep; the others train their own grids.
# eta_ref per depth from LR probes (eta* ~ l^-1.04 between L=2 and L=6); L=4 is
# interpolated (~0.0026) and refined by its own probe before the run.
DEPTHS = {
    1: dict(csv="sweep_cosine_L1.csv", color="seagreen",   lr=dict(eta_ref=0.0057, **_EXP)),
    2: dict(csv="sweep_cosine.csv",    color="steelblue",  lr=None),
    4: dict(csv="sweep_cosine_L4.csv", color="darkorange", lr=dict(eta_ref=0.0035, **_EXP)),
    6: dict(csv="sweep_cosine_L6.csv", color="crimson",    lr=dict(eta_ref=0.0017, **_EXP)),
}
# Effective CPU throughput for big-cell-dominated grids (measured: the 2^22..2^24
# extension ran at ~4.8e10 FLOP/s). Small cells are overhead-bound, so L=1 is rougher.
FLOPS_PER_S = 4.8e10


def l2_coverage(out: Path) -> dict[int, list[int]]:
    """Per-width sorted data budgets the L=2 *grid* trained on (excl. iso-FLOP cells)."""
    df = pd.read_csv(out / DEPTHS[2]["csv"])
    if "iso_C" in df:
        df = df[df["iso_C"].fillna(0) == 0]      # mirror the regular grid, not iso cells
    cov = {}
    for w in sorted(int(x) for x in df.width.unique()):
        cov[w] = sorted(int(d) for d in df[df.width == w].D.unique())
    return cov


def new_cells(out: Path, csv: str, n_hidden: int, cov: dict[int, list[int]]):
    """(width, D) cells in `cov` not already cached for this depth + their est. seconds."""
    done = set()
    p = out / csv
    if p.exists():
        c = pd.read_csv(p)
        done = {(int(w), int(d)) for w, d in zip(c.width, c.D)}
    cells, secs = [], 0.0
    for w, Ds in cov.items():
        for D in Ds:
            if (w, D) in done:
                continue
            cells.append((w, D))
            secs += compute(mlp_param_count(PROBLEM["input_dim"], w, n_hidden), D,
                            PROBLEM["input_dim"], w) / FLOPS_PER_S
    return cells, secs * len(SEEDS)


def main():
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--depths", nargs="+", type=int, default=[1, 6], choices=[1, 4, 6])
    ap_.add_argument("--max-width", type=int, default=None,
                     help="cap the widths to mirror (e.g. 256 to skip the costly width-512 cells)")
    ap_.add_argument("--dry-run", action="store_true", help="preview cells + cost, no training")
    ap_.add_argument("--plot-only", action="store_true", help="skip training, just plot")
    ap_.add_argument("--outdir", default="results")
    args = ap_.parse_args()

    out = Path(args.outdir); figdir = out / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    pl.set_style()
    cov = l2_coverage(out)
    if args.max_width is not None:
        cov = {w: Ds for w, Ds in cov.items() if w <= args.max_width}
    print("Mirroring L=2 coverage:")
    for w, Ds in cov.items():
        print(f"  w={w:<4} {len(Ds)} budgets  (D up to 2^{max(Ds).bit_length()-1} = {max(Ds):,})")

    if args.dry_run:
        print("\n--- dry run (no training) ---")
        for L in args.depths:
            cells, secs = new_cells(out, DEPTHS[L]["csv"], L, cov)
            biggest = max(cells, key=lambda c: mlp_param_count(32, c[0], L) * c[1]) if cells else None
            print(f"L={L}: {len(cells)} new cells, est. ~{secs/60:.0f} min "
                  f"({secs/3600:.1f} h)" + (f"; dominant cell w={biggest[0]}, D={biggest[1]:,}" if biggest else ""))
        print("\nLaunch with:  python scripts/run_depth_compare.py --depths " +
              " ".join(map(str, args.depths)))
        return

    prob = TeacherStudentRegression(**PROBLEM)
    E = prob.irreducible_loss
    if not args.plot_only:
        for L in args.depths:
            for w, Ds in tqdm(cov.items(), desc=f"train L={L}", unit="width"):
                run_grid(prob, [w], Ds, n_hidden=L, lr_rule=transfer_lr(**DEPTHS[L]["lr"]),
                         seeds=SEEDS, cache_path=str(out / DEPTHS[L]["csv"]), verbose=False)

    # ---- load each family ----
    aggs, envs = {}, {}
    for L, d in DEPTHS.items():
        if not (out / d["csv"]).exists():
            continue
        a = aggregate(pd.read_csv(out / d["csv"]))
        a = a[a.n_hidden == L].reset_index(drop=True)
        if len(a):
            aggs[L], envs[L] = a, ap.training_curve_envelope(a)
    envP = ap.training_curve_envelope(pd.concat(aggs.values()).reset_index(drop=True))
    print("a_N per depth:", {L: round(e.a_N, 3) for L, e in envs.items()},
          "| combined", round(envP.a_N, 3))

    # ---- comparison figure: loss vs compute per depth, with the envelope frontier ----
    fig, ax = plt.subplots(figsize=(7, 5))
    for L in sorted(aggs):
        c, a, env = DEPTHS[L]["color"], aggs[L], envs[L]
        for _, g in a.groupby("N"):
            g = g.sort_values("C"); ax.plot(g.C, g.val_loss, "-", color=c, alpha=0.18, lw=0.7)
        ax.plot(env.frontier.C, env.frontier.val_loss, "o-", color=c, ms=4,
                label=f"L={L}  (a={env.a_N:.2f})")
    ax.axhline(E, ls=":", c="k", lw=1)
    ax.set(xscale="log", yscale="log", xlabel=r"$C=(6N-2dw)D$", ylabel="loss (MSE)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir / "depth_comparison.png", dpi=110, bbox_inches="tight")
    fig.savefig(figdir / "depth_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- full per-depth analysis, only for the depth the notebook uses ----
    if MAIN_DEPTH in aggs:
        a = aggs[MAIN_DEPTH]
        env = ap.training_curve_envelope(a)
        iso = ap.isoflop_profiles(a, n_targets=8)
        par = ap.parametric_fit(a)
        pl.save_figure(pl.plot_all_runs(a, irreducible=E), f"depth_L{MAIN_DEPTH}_overview", str(figdir))
        pl.save_figure(pl.plot_envelope(a, env, irreducible=E), f"depth_L{MAIN_DEPTH}_envelope", str(figdir))
        pl.save_figure(pl.plot_isoflop(iso), f"depth_L{MAIN_DEPTH}_isoflop", str(figdir))
        pl.save_figure(pl.plot_parametric(a, par, irreducible=E), f"depth_L{MAIN_DEPTH}_parametric", str(figdir))
        pl.save_figure(pl.plot_comparison(env, iso, par, (a.C.min(), a.C.max())),
                       f"depth_L{MAIN_DEPTH}_comparison", str(figdir))
        plt.close("all")
        print(f"per-depth figures (L={MAIN_DEPTH} only): "
              f"a_N env={env.a_N:.3f} iso={iso.a_N:.3f} par={par.a_N:.3f}")


if __name__ == "__main__":
    main()
