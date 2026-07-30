#!/usr/bin/env python
"""Run the (N, D) scaling sweep for the flow-matching problem and cache it to CSV.

The flow-matching analogue of run_sweep.py: a grid of MLP velocity fields of growing
width, each trained single-pass on fresh (x_t, t) -> v pairs at its own tuned LR. The
per-cell LR law is read from results/flow_hp_study_cosine.json (calibrate it first
with run_flow_hp_study.py). Writes results/flow_sweep_cosine.csv + meta sidecar; the
notebook loads the CSV and extracts the scaling law three ways. Run from the repo root.

Examples
--------
    python scripts/run_flow_hp_study.py         # once: calibrate the LR law
    python scripts/run_flow_sweep.py            # full grid
    python scripts/run_flow_sweep.py --seeds 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scaling_laws.flow import (FlowMatchingProblem, MultiMoonsFlow,  # noqa: E402
                               HierarchicalMoonsFlow)
from scaling_laws.sweep import run_grid, transfer_lr    # noqa: E402

WIDTHS = [4, 8, 16, 32, 64, 128, 256, 384]
WIDTHS_HARD = [8, 16, 32, 64, 128, 256, 384, 512]
WIDTHS_HIER = [16, 32, 64, 128, 256, 384, 512, 768]
WIDTHS_32 = [32, 64, 128, 256, 512, 768, 1024, 1536]
LOG2_D = list(range(10, 22))            # 1024 .. 2,097,152 examples
LOG2_D_HARD = list(range(10, 23))       # the harder problems get more octaves
LOG2_D_HIER = list(range(10, 24))
LOG2_D_32 = list(range(11, 25))
BATCH = 256
# mode -> (problem class, kwargs, file stem, widths, data budgets, notebook)
MODES = {
    "2d":   (FlowMatchingProblem, dict(moons_noise=0.05, val_size=65536, seed=0),
             "flow_sweep", "flow_hp_study_cosine.json", WIDTHS, LOG2_D,
             "05_flow_matching"),
    "8d":   (MultiMoonsFlow, dict(n_pairs=4, moons_noise=0.05, val_size=65536, seed=0),
             "flow8_sweep", "flow8_hp_study_cosine.json", WIDTHS_HARD, LOG2_D_HARD,
             "06_flow_matching_hard"),
    "hier": (HierarchicalMoonsFlow, dict(n_pairs=4, val_size=65536, seed=0),
             "flowh_sweep", "flowh_hp_study_cosine.json", WIDTHS_HIER, LOG2_D_HIER,
             "07_flow_matching_hierarchical"),
    "32d":  (HierarchicalMoonsFlow, dict(n_pairs=16, val_size=65536, seed=0),
             "flow32_sweep", "flow32_hp_study_cosine.json", WIDTHS_32, LOG2_D_32,
             "08_flow_matching_32d"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=2, help="number of seeds (0..n-1)")
    ap.add_argument("--max-log2D", type=int, default=None,
                    help="cap the data budget at 2**this")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--hard", action="store_true",
                    help="the 8D entangled multi-moons problem (flow8_* files)")
    ap.add_argument("--hier", action="store_true",
                    help="the hierarchical two-scale moons problem (flowh_* files)")
    ap.add_argument("--dim32", action="store_true",
                    help="16 hierarchical pairs in R^32, the hardest problem (flow32_* files)")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    mode = ("32d" if args.dim32 else
            "hier" if args.hier else ("8d" if args.hard else "2d"))
    cls, kwargs, stem, law_file, widths, log2d, nb = MODES[mode]
    law = json.loads((outdir / law_file).read_text())
    lr_rule = transfer_lr(eta_ref=law["eta_ref"], w_ref=law["w_ref"], T_ref=law["T_ref"],
                          c_w=law["c_w"], c_T=law["c_T"], lr_max=0.1)
    seeds = tuple(range(args.seeds))
    Ds = [2 ** k for k in log2d if args.max_log2D is None or k <= args.max_log2D]

    prob = cls(**kwargs)
    print(f"Flow matching [{mode}]: {type(prob).__name__}, dim={prob.dim}")
    print(f"Per-cell tuned LR: eta*={law['eta_ref']}*(w/{law['w_ref']})^{law['c_w']}"
          f"*(T/{law['T_ref']})^{law['c_T']}")
    print(f"Grid: {len(widths)} widths x {len(Ds)} data sizes x {len(seeds)} seeds")

    t0 = time.time()
    cache = outdir / f"{stem}_cosine.csv"
    df = run_grid(prob, widths, Ds, n_hidden=2, batch_size=BATCH, lr_rule=lr_rule,
                  seeds=seeds, cache_path=str(cache), force=args.force, verbose=True)
    print(f"  ({time.time() - t0:.0f}s)  ->  {cache}")

    meta = dict(problem=dict(kind=law.get("problem", mode), **kwargs),
                widths=sorted(int(w) for w in df["width"].unique()),
                D=sorted(int(d) for d in df["D"].unique()), seeds=list(seeds),
                batch_size=BATCH, n_hidden=2, cosine_lr_law=law)
    (outdir / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone. Visualise with notebooks/{nb}.ipynb")


if __name__ == "__main__":
    main()
