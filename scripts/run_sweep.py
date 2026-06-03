#!/usr/bin/env python
"""Run the (N, D) scaling sweep and cache it to CSV.

This is the *training* entry point. It trains a grid of MLP students of growing
size N on growing single-pass data budgets D, each at its own tuned (cosine) LR, and
writes the table plus a meta sidecar describing the problem. The notebook then loads
the CSV and does all the analysis and plotting -- it never trains.

Examples
--------
    python scripts/run_sweep.py                 # full grid
    python scripts/run_sweep.py --preset quick   # small/fast grid for a smoke test
    python scripts/run_sweep.py --seeds 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scaling_laws.data import TeacherStudentRegression  # noqa: E402
from scaling_laws.sweep import run_grid, transfer_lr     # noqa: E402

# --------------------------------------------------------------------------- #
# Problem + grid configuration. The ground truth is a fixed, frozen teacher
# network (weights saved to results/teacher.pt). A student >= teacher width hits
# the irreducible floor E = sigma^2; smaller students stay capacity-limited, so
# the loss falls smoothly with model size N.
# --------------------------------------------------------------------------- #
PROBLEM = dict(input_dim=32, teacher_width=256, teacher_depth=2, teacher_act="gelu",
               noise_std=0.1, val_size=16384, seed=0)

PRESETS = {
    "full": dict(widths=[8, 16, 32, 64, 128, 512],
                 log2_D=list(range(10, 25)),            # 1024 .. 16,777,216
                 batch_size=256, n_hidden=2),
    "quick": dict(widths=[16, 64, 128],
                  log2_D=[10, 12, 14, 16, 18],          # 1024 .. 262,144
                  batch_size=256, n_hidden=2),
}

# Per-cell tuned LR law eta*(w, T), calibrated by the cosine HP-transfer study for
# this regression task (scripts/run_hp_study.py -> results/hp_study_cosine.json).
# Anchored at (w_ref, T_ref). Filled in from that study's output.
COSINE_LR = dict(eta_ref=0.00535, w_ref=64, T_ref=2048, c_w=-0.374, c_T=-0.617)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS), default="full")
    ap.add_argument("--seeds", type=int, default=2, help="number of seeds (0..n-1)")
    ap.add_argument("--widths", nargs="+", type=int, default=None,
                    help="override the preset widths (e.g. to densify: 6 12 24 48 96)")
    ap.add_argument("--max-log2D", type=int, default=None,
                    help="cap the data budget at 2**this (e.g. 21 for D<=2.1M)")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    args = ap.parse_args()

    cfg = PRESETS[args.preset]
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(range(args.seeds))
    widths = args.widths if args.widths else cfg["widths"]
    log2_D = [k for k in cfg["log2_D"] if args.max_log2D is None or k <= args.max_log2D]
    Ds = [2 ** k for k in log2_D]
    lr_rule = transfer_lr(**COSINE_LR)

    prob = TeacherStudentRegression(**PROBLEM, save_path=str(outdir / "teacher.pt"))
    print(f"Teacher: width={PROBLEM['teacher_width']} depth={PROBLEM['teacher_depth']} "
          f"| noise sigma={PROBLEM['noise_std']}  ->  KNOWN floor E = sigma^2 = "
          f"{prob.irreducible_loss:g}")
    print(f"Per-cell tuned LR: eta*={COSINE_LR['eta_ref']}*(w/{COSINE_LR['w_ref']})"
          f"^{COSINE_LR['c_w']}*(T/{COSINE_LR['T_ref']})^{COSINE_LR['c_T']}")
    print(f"Grid: {len(widths)} widths x {len(Ds)} data sizes x {len(seeds)} seeds")

    t0 = time.time()
    cache = outdir / "sweep_cosine.csv"
    df = run_grid(prob, widths, Ds, n_hidden=cfg["n_hidden"],
                  batch_size=cfg["batch_size"], lr_rule=lr_rule, seeds=seeds,
                  cache_path=str(cache), force=args.force, verbose=True)
    print(f"  ({time.time() - t0:.0f}s)  ->  {cache}")

    # Write meta reflecting what's actually in the (possibly incrementally grown) CSV.
    meta = dict(problem=PROBLEM, irreducible_loss=prob.irreducible_loss,
                widths=sorted(int(w) for w in df["width"].unique()),
                D=sorted(int(d) for d in df["D"].unique()), seeds=list(seeds),
                batch_size=cfg["batch_size"], n_hidden=cfg["n_hidden"],
                cosine_lr_law=COSINE_LR, preset=args.preset)
    (outdir / "sweep_meta.json").write_text(json.dumps(meta, indent=2))

    print("\nDone. Visualise with notebooks/01_scaling_laws.ipynb")


if __name__ == "__main__":
    main()
