#!/usr/bin/env python
"""Hyperparameter-transfer study: how does the optimal LR move with width and steps?

A fair scaling-law grid needs every cell trained near its own optimal learning rate;
otherwise mis-tuning biases the fitted exponents -- the issue muP addresses by
reparametrisation. Here we *measure* the transfer law instead:

  * eta* vs width at a fixed step budget T  -> exponent c_w  (the muP shift)
  * eta* vs steps T at a fixed width        -> exponent c_T  (only T >= T_min, since
                                               short runs are warmup-dominated)

For each (width, T) we sweep LR, fit a parabola to loss-vs-log(LR), and take the
vertex as eta*. The fitted law eta*(w,T) = eta_ref (w/w_ref)^c_w (T/T_ref)^c_T is what
scripts/run_sweep.py uses to set the per-cell LR. Results -> results/hp_study_cosine.json + .png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")

from scaling_laws.data import TeacherStudentRegression  # noqa: E402
from scaling_laws.sweep import run_cell                 # noqa: E402
from scaling_laws import plotting as pl                 # noqa: E402

PROBLEM = dict(input_dim=32, teacher_width=256, teacher_depth=2,
               teacher_act="gelu", noise_std=0.1, val_size=16384, seed=0)


def save_figure(law, outdir: Path):
    pl.set_style()
    fig = pl.plot_hp_study(law)
    fig.savefig(outdir / "figures/hp_study_cosine.png", bbox_inches="tight")


def eta_star(prob, width, n_data, lrs, seeds):
    """Loss-minimising LR at one (width, D) cell: parabola vertex in log10(LR)."""
    loss = [np.mean([run_cell(prob, width, n_data, batch_size=256, lr=float(lr),
                              seed=s)["val_loss"] for s in seeds])
            for lr in lrs]
    x = np.log10(lrs)
    p2, p1, _ = np.polyfit(x, loss, 2)
    xs = np.clip(-p1 / (2 * p2), x.min(), x.max()) if p2 > 0 else x[int(np.argmin(loss))]
    return float(10 ** xs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--w_ref", type=int, default=64)
    ap.add_argument("--T_ref", type=int, default=2048)      # ref step budget (b=256)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--plot-only", action="store_true", help="replot from the cached json")
    args = ap.parse_args()
    out = Path(args.outdir)

    if args.plot_only:
        law = json.loads((out / "hp_study_cosine.json").read_text())
        save_figure(law, out)
        print("replotted results/figures/hp_study_cosine.png")
        return

    seeds = tuple(range(args.seeds))
    b = 256

    prob = TeacherStudentRegression(**PROBLEM)
    lrs = np.geomspace(3e-4, 1e-1, 9)
    widths = [4, 8, 16, 32, 64, 128, 256, 512]
    Tsteps = [256, 512, 1024, 2048, 4096, 8192]

    eW = [eta_star(prob, w, args.T_ref * b, lrs, seeds)
          for w in tqdm(widths, desc="eta* vs width", unit="w")]
    eT = [eta_star(prob, args.w_ref, T * b, lrs, seeds)
          for T in tqdm(Tsteps, desc="eta* vs T", unit="T")]

    c_w, b_w = np.polyfit(np.log(widths), np.log(eW), 1)
    c_T, _ = np.polyfit(np.log(Tsteps), np.log(eT), 1)
    eta_ref = float(np.exp(b_w) * args.w_ref ** c_w)        # law value at (w_ref, T_ref)

    law = dict(schedule="cosine", eta_ref=round(eta_ref, 5), w_ref=args.w_ref,
               T_ref=args.T_ref, c_w=round(float(c_w), 3), c_T=round(float(c_T), 3),
               widths=widths, eta_width=eW, Tsteps=Tsteps, eta_T=eT)
    out.mkdir(parents=True, exist_ok=True)
    (out / "hp_study_cosine.json").write_text(json.dumps(law, indent=2))
    print(f"  eta*(w,T) = {eta_ref:.4f} * (w/{args.w_ref})^{c_w:.3f} "
          f"* (T/{args.T_ref})^{c_T:.3f}")
    save_figure(law, out)
    print("  saved results/hp_study_cosine.json + figure")


if __name__ == "__main__":
    main()
