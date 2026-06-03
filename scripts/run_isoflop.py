#!/usr/bin/env python
"""Dedicated IsoFLOP runs: actually train models at a shared compute C = 6ND.

Elsewhere Approach 2 reconstructs iso-compute slices by interpolating the rectangular
(N, D) grid. This trains, for a set of compute budgets C and each depth family
(L = 1/2/4/6), *real* models at (N, D = C/6N) -- genuine runs at the same compute.
Each is tagged with iso_C = C and appended to that depth's CSV; aggregate() carries
iso_C and Approach 2 then uses these runs directly. They are ordinary (N, D) cells,
so they also enrich the default overview / envelope / parametric plots.

Resumable: skips cells already in the CSV, and regenerates the depth figures at the end.
Writes the per-depth CSVs and figures under --outdir (default ./results); run from the
repo root.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from scaling_laws.data import TeacherStudentRegression   # noqa: E402
from scaling_laws.sweep import run_cell, transfer_lr      # noqa: E402
from scaling_laws.flops import mlp_param_count, flops_per_token  # noqa: E402

PROBLEM = dict(input_dim=32, teacher_width=256, teacher_depth=2, teacher_act="gelu",
               noise_std=0.1, val_size=16384, seed=0)
EXP = dict(w_ref=64, T_ref=2048, c_w=-0.374, c_T=-0.617)     # shared LR exponents
# depth -> (csv, eta_ref) ; L=2 uses the main sweep + its calibration
DEPTHS = {1: ("sweep_cosine_L1.csv", 0.0057), 2: ("sweep_cosine.csv", 0.00535),
          4: ("sweep_cosine_L4.csv", 0.0035), 6: ("sweep_cosine_L6.csv", 0.0017)}
WIDTHS = [8, 12, 16, 24, 32, 48, 64, 96, 128, 512]          # widths for the iso-FLOP grid
# D floor = 8192 -> >=32 optimizer steps at batch 256: below that a high-N model at a
# low budget barely trains and its loss sits near init, bowing the iso-parabola (the
# analysis side guards against any leftovers via the well-trim in approaches._fit_slice).
DLO, DHI = 8192, 2 ** 20                                     # keep data-gen bounded
SEEDS = (0,)
N_BUDGETS = 5


def pick_budgets(kvals):
    """Geometric compute budgets where >=4 widths land at D = C/k inside [DLO, DHI].

    k = 6N - 2*d*w is each width's FLOPs/example, so a true iso-compute run uses
    D = C/k (not C/6N)."""
    cand = np.geomspace(min(kvals) * DLO, max(kvals) * DHI, 400)
    good = [C for C in cand if sum(DLO <= C / k <= DHI for k in kvals) >= 4]
    return list(np.geomspace(min(good), max(good), N_BUDGETS)) if good else []


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    prob = TeacherStudentRegression(**PROBLEM)
    for L, (csvname, eta) in DEPTHS.items():
        csv = out / csvname
        rule = transfer_lr(eta_ref=eta, **EXP)
        kvals = [flops_per_token(mlp_param_count(32, w, L), 32, w) for w in WIDTHS]
        budgets = pick_budgets(kvals)
        done = set()
        if csv.exists():
            ex = pd.read_csv(csv)
            if "iso_C" in ex:
                iso = ex[ex["iso_C"] > 0]
                done = {(int(w), int(d)) for w, d in zip(iso["width"], iso["D"])}
        cells = []
        for C in budgets:
            for w in WIDTHS:
                N = mlp_param_count(32, w, L)
                D = int(round(C / flops_per_token(N, 32, w)))   # true iso-compute: D = C/k
                if DLO <= D <= DHI and (w, D) not in done:
                    cells.append((C, w, N, D))
        recs = []
        bar = tqdm(cells, desc=f"iso L={L}", unit="cell")
        for C, w, N, D in bar:
            for s in SEEDS:
                r = run_cell(prob, w, D, n_hidden=L, batch_size=256,
                             lr=rule(w, max(1, D // 256)), iso_C=float(C), seed=s)
                recs.append(r)
            bar.set_postfix(C=f"{C:.1e}", w=w, D=D, val=f"{r['val_loss']:.3f}")
        if recs:
            new = pd.DataFrame(recs)
            if csv.exists():
                old = pd.read_csv(csv)
                if "iso_C" not in old:
                    old["iso_C"] = 0.0
                new = pd.concat([old, new], ignore_index=True)
            new.to_csv(csv, index=False)
            print(f"L={L}: +{len(recs)} iso-FLOP rows -> {csvname}", flush=True)

    # regenerate the depth figures so Approach 2 picks up the new dedicated runs
    subprocess.run([sys.executable, "scripts/run_depth_compare.py", "--plot-only",
                    "--outdir", str(out)])
    print("done", flush=True)


if __name__ == "__main__":
    main()
