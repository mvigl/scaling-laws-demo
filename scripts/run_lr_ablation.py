#!/usr/bin/env python
"""What re-tuning the learning rate buys, along D and along N (depth L=2).

Both panels compare a single *fixed* LR against the fully per-cell-tuned baseline
eta*(w, T) (the transfer law run_sweep.py uses). The fixed LR is the one optimal at
the small end of the swept axis, then held constant:

  * left  -- fixed architecture (w=64), grow D. eta* falls with the step budget, so an
             LR tuned at the smallest D is too hot once D is large.
  * right -- fixed D, grow width N. eta* falls with width (the muP shift), so an LR
             tuned at the smallest width is too hot once the model is wide.

The gap between the curves is the loss left on the table by not re-tuning -- the bias
that hyperparameter transfer removes. Writes results/lr_ablation.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tqdm import tqdm                                     # noqa: E402

from scaling_laws.data import TeacherStudentRegression   # noqa: E402
from scaling_laws.sweep import run_cell, transfer_lr      # noqa: E402

PROBLEM = dict(input_dim=32, teacher_width=256, teacher_depth=2, teacher_act="gelu",
               noise_std=0.1, val_size=16384, seed=0)
LAW = dict(eta_ref=0.00535, w_ref=64, T_ref=2048, c_w=-0.374, c_T=-0.617)   # cosine law

L = 2                                   # depth held fixed throughout
B = 256                                 # batch size held fixed (see notebook caveat)
SEEDS = (0, 1)                          # the gap is read off bigger models, not more seeds
W0 = 64                                 # fixed architecture for the D panel
# D panel starts at 2^16 (=256 steps): below that T<T_min and the LR law is flat
# (warmup-dominated), so there would be nothing to show. From here eta* falls with D.
D_LEFT = [2 ** k for k in range(16, 23)]                # 65536 .. ~4.2M examples
W_RIGHT = [8, 16, 32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 3072]   # widths for the N panel
D0_RIGHT = 2 ** 18                                      # fixed data budget for the N panel


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    OUT = out / "lr_ablation.csv"

    prob = TeacherStudentRegression(**PROBLEM)
    tuned = transfer_lr(**LAW)
    rows, done = [], set()
    if OUT.exists():                                    # resume: skip cells already trained
        rows = pd.read_csv(OUT).to_dict("records")
        done = {(r["panel"], int(r["x"]), r["tuned"], int(r["seed"])) for r in rows}

    def cell(panel, w, D, label, lr):
        for s in SEEDS:
            if (panel, w if panel == "N" else D, label, s) in done:
                continue
            r = run_cell(prob, w, D, n_hidden=L, batch_size=B, lr=lr, seed=s)
            rows.append(dict(panel=panel, x=(w if panel == "N" else D), width=w, D=D,
                             lr=round(lr, 5), tuned=label, seed=s, val_loss=r["val_loss"]))
        pd.DataFrame(rows).to_csv(OUT, index=False)      # checkpoint after each cell

    # left: scale D at fixed width; fixed LR = eta* at the smallest D
    lr_fixed = tuned(W0, prob.n_steps(D_LEFT[0], B))
    for D in tqdm(D_LEFT, desc="scale D", unit="D"):
        eta = tuned(W0, prob.n_steps(D, B))
        cell("D", W0, D, "tuned", eta); cell("D", W0, D, "fixed", lr_fixed)

    # right: scale width at fixed D; fixed LR = eta* at the smallest width
    T0 = prob.n_steps(D0_RIGHT, B)
    lr_fixed = tuned(W_RIGHT[0], T0)
    for w in tqdm(W_RIGHT, desc="scale N", unit="width"):
        eta = tuned(w, T0)
        cell("N", w, D0_RIGHT, "tuned", eta); cell("N", w, D0_RIGHT, "fixed", lr_fixed)

    print(f"done: {len(rows)} runs -> {OUT}")


if __name__ == "__main__":
    main()
