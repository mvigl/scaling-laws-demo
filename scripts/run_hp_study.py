#!/usr/bin/env python
"""Driver for the HP-transfer study: calibrate eta*(w, T) and cache it.

A thin wrapper over ``scaling_laws.hp.fit_transfer_law``, which runs the inner LR sweeps
(``scaling_laws.hp.tune_lr_cell``) and fits

    eta*(w, T) = eta_ref * (w/w_ref)^c_w * (T/T_ref)^c_T.

Writes results/hp_study_cosine.json + figure; scripts/run_sweep.py reads the json to set
the per-cell LR. The two notebooks 01_* (one cell) and 02_* (the law) walk through the
same helpers interactively. Run from the repo root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from scaling_laws.data import TeacherStudentRegression  # noqa: E402
from scaling_laws.hp import fit_transfer_law            # noqa: E402
from scaling_laws import plotting as pl                 # noqa: E402

PROBLEM = dict(input_dim=32, teacher_width=256, teacher_depth=2,
               teacher_act="gelu", noise_std=0.1, val_size=16384, seed=0)
WIDTHS = [4, 8, 16, 32, 64, 128, 256, 512]
TSTEPS = [256, 512, 1024, 2048, 4096, 8192]


def save_figure(law, outdir: Path):
    pl.set_style()
    pl.plot_hp_study(law).savefig(outdir / "figures/hp_study_cosine.png", bbox_inches="tight")


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
        save_figure(json.loads((out / "hp_study_cosine.json").read_text()), out)
        print("replotted results/figures/hp_study_cosine.png")
        return

    prob = TeacherStudentRegression(**PROBLEM)
    law = fit_transfer_law(prob, widths=WIDTHS, Tsteps=TSTEPS, w_ref=args.w_ref,
                           T_ref=args.T_ref, seeds=tuple(range(args.seeds)), progress=True)
    law["schedule"] = "cosine"
    out.mkdir(parents=True, exist_ok=True)
    (out / "hp_study_cosine.json").write_text(json.dumps(law, indent=2))
    print(f"  eta*(w,T) = {law['eta_ref']} * (w/{law['w_ref']})^{law['c_w']} "
          f"* (T/{law['T_ref']})^{law['c_T']}")
    save_figure(law, out)
    print("  saved results/hp_study_cosine.json + figure")


if __name__ == "__main__":
    main()
