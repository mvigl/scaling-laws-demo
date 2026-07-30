#!/usr/bin/env python
"""Calibrate the LR transfer law eta*(w, T) for the flow-matching problem.

The flow-matching analogue of run_hp_study.py: same measurement (tune each cell's LR
by a parabola sweep, scaling_laws.hp.tune_lr_cell), same law

    eta*(w, T) = eta_ref * (w/w_ref)^c_w * (T/T_ref)^c_T,

on the Gaussian -> two-moons velocity-field regression instead of the teacher-student
task. The LR grid starts at 1e-3 (lower rates only underfit at these step budgets).
Writes results/flow_hp_study_cosine.json + figure; run_flow_sweep.py reads the json.
Run from the repo root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")

from scaling_laws.flow import (FlowMatchingProblem, MultiMoonsFlow,  # noqa: E402
                               HierarchicalMoonsFlow)
from scaling_laws.hp import fit_transfer_law            # noqa: E402
from scaling_laws import plotting as pl                 # noqa: E402

LRS = np.geomspace(1e-3, 1e-1, 9)
# harder grids train wider models and longer horizons, so their laws are calibrated
# on extended ranges. mode -> (problem factory, stem, tag, widths, tsteps)
MODES = {
    "2d":   (FlowMatchingProblem, "flow_hp_study_cosine", "flow_matching",
             [8, 16, 32, 64, 128, 256], [256, 512, 1024, 2048, 4096, 8192]),
    "8d":   (MultiMoonsFlow, "flow8_hp_study_cosine", "multi_moons_8d",
             [8, 16, 32, 64, 128, 256], [256, 512, 1024, 2048, 4096, 8192]),
    "hier": (HierarchicalMoonsFlow, "flowh_hp_study_cosine", "hierarchical_moons_8d",
             [16, 32, 64, 128, 256, 384], [256, 512, 1024, 2048, 4096, 8192, 16384]),
    "32d":  (lambda: HierarchicalMoonsFlow(n_pairs=16), "flow32_hp_study_cosine",
             "hierarchical_moons_32d",
             [32, 64, 128, 256, 512], [512, 1024, 2048, 4096, 8192, 16384, 32768]),
}


def save_figure(law, outdir: Path, stem: str):
    pl.set_style()
    pl.plot_hp_study(law).savefig(outdir / f"figures/{stem}.png", bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--w_ref", type=int, default=64)
    ap.add_argument("--T_ref", type=int, default=2048)      # ref step budget (b=256)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--hard", action="store_true",
                    help="the 8D entangled multi-moons problem (writes flow8_* files)")
    ap.add_argument("--hier", action="store_true",
                    help="the hierarchical two-scale moons problem (writes flowh_* files)")
    ap.add_argument("--dim32", action="store_true",
                    help="16 hierarchical pairs in R^32, the hardest problem (flow32_* files)")
    ap.add_argument("--plot-only", action="store_true", help="replot from the cached json")
    args = ap.parse_args()
    out = Path(args.outdir)
    mode = ("32d" if args.dim32 else
            "hier" if args.hier else ("8d" if args.hard else "2d"))
    cls, stem, tag, widths, tsteps = MODES[mode]

    if args.plot_only:
        save_figure(json.loads((out / f"{stem}.json").read_text()), out, stem)
        print(f"replotted results/figures/{stem}.png")
        return

    prob = cls()
    law = fit_transfer_law(prob, widths=widths, Tsteps=tsteps, w_ref=args.w_ref,
                           T_ref=args.T_ref, lrs=LRS,
                           seeds=tuple(range(args.seeds)), progress=True)
    law["schedule"] = "cosine"
    law["problem"] = tag
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.json").write_text(json.dumps(law, indent=2))
    print(f"  eta*(w,T) = {law['eta_ref']} * (w/{law['w_ref']})^{law['c_w']} "
          f"* (T/{law['T_ref']})^{law['c_T']}")
    save_figure(law, out, stem)
    print(f"  saved results/{stem}.json + figure")


if __name__ == "__main__":
    main()
