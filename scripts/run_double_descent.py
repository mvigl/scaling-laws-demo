#!/usr/bin/env python
"""Three flavours of double descent in the repeated-data / overfitting regime.

The main tutorial is single-pass (data never repeats), so it never overfits and the
loss is monotone. Here we do the opposite (Nakkiran et al. 2019): fix a dataset, repeat it
to interpolation, with extra label noise (sigma=0.4), and watch generalisation.

  --mode samples : vary dataset size D at fixed width   -> peak at D ~ N      (sample-wise)
  --mode width   : vary width W at fixed dataset D       -> peak at N ~ D      (model-wise)
  --mode epochs  : fix W and D (overparam), train long   -> test goes down-up-down (epoch-wise)

Test error is the *excess risk* (MSE to the clean teacher on held-out inputs); train MSE
is logged too (for width it shows the interpolation transition; for epochs it is half the
story). Writes results/double_descent_<mode>.{csv,png}.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

from scaling_laws.data import TeacherStudentRegression  # noqa: E402
from scaling_laws.models import make_mlp                # noqa: E402
from scaling_laws import plotting as pl                 # noqa: E402

SAMPLES_D = [4000, 8000, 16000, 24000, 32000, 48000, 64000, 80000, 96000,
             128000, 192000, 256000, 384000, 512000]
WIDTH_GRID = [4, 8, 16, 24, 32, 48, 64, 80, 96, 128, 192, 256, 384, 512]
EPOCH_D = [5000, 10000]                 # heavily overparam (D/N ~ 0.07, 0.13)
# 2D phase diagram (Nakkiran-style): sweep model size x dataset size, 1 seed.
GRID_W = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
GRID_D = [3125, 4419, 6250, 8839, 12500, 17678, 25000, 35355, 50000]


def n_params(width):
    return sum(p.numel() for p in make_mlp(32, width, 2, 1).parameters())


def fixed_dataset(prob, D, sigma, seed):
    g = torch.Generator().manual_seed(1000 + seed)
    X = torch.randn(D, prob.input_dim, generator=g)
    y = prob.clean_target(X) + sigma * torch.randn(D, 1, generator=g)
    return X, y


def train_final(prob, width, D, sigma, seed, max_steps, batch, lr, eps=1e-3):
    """Train to interpolation; return (final excess, early-stopped excess, train MSE, steps).

    The early-stopped excess is the *minimum* test risk seen over training (an oracle
    early stop, evaluated once per epoch). Comparing it to the final risk shows how much
    of the double-descent peak is an artefact of training all the way to interpolation --
    early stopping is the simplest regulariser that removes it.
    """
    X, y = fixed_dataset(prob, D, sigma, seed)
    torch.manual_seed(seed)
    m = make_mlp(prob.input_dim, width, 2, 1); opt = torch.optim.Adam(m.parameters(), lr)
    steps, train_mse, best = 0, 9.9, float("inf")
    while steps < max_steps:
        perm = torch.randperm(D)
        for i in range(0, D, batch):
            idx = perm[i:i + batch]
            opt.zero_grad(); F.mse_loss(m(X[idx]), y[idx]).backward(); opt.step(); steps += 1
            if steps >= max_steps:
                break
        with torch.no_grad():
            train_mse = F.mse_loss(m(X), y).item()
            best = min(best, F.mse_loss(m(prob.x_val), prob.y_val_clean).item())
        if train_mse < eps:
            break
    with torch.no_grad():
        excess = F.mse_loss(m(prob.x_val), prob.y_val_clean).item()
    return excess, best, train_mse, steps


def train_trajectory(prob, width, D, sigma, seed, max_steps, batch, lr, n_logs=240):
    """Train long (no early stop); log (step, train, excess) at LOG-spaced steps.

    Log-spaced (dense early) because the initial descent -- the model fitting the
    teacher signal -- happens in the first few hundred steps; uniform logging misses it.
    """
    X, y = fixed_dataset(prob, D, sigma, seed)
    torch.manual_seed(seed)
    m = make_mlp(prob.input_dim, width, 2, 1); opt = torch.optim.Adam(m.parameters(), lr)
    targets = np.unique(np.geomspace(1, max_steps, n_logs).astype(int))
    traj, steps, li = [], 0, 0
    while steps < max_steps:
        perm = torch.randperm(D)
        for i in range(0, D, batch):
            idx = perm[i:i + batch]
            opt.zero_grad(); F.mse_loss(m(X[idx]), y[idx]).backward(); opt.step(); steps += 1
            if li < len(targets) and steps >= targets[li]:
                with torch.no_grad():
                    traj.append(dict(step=steps,
                                     train=F.mse_loss(m(X), y).item(),
                                     excess=F.mse_loss(m(prob.x_val), prob.y_val_clean).item()))
                while li < len(targets) and targets[li] <= steps:
                    li += 1
            if steps >= max_steps:
                break
    return traj


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["samples", "width", "epochs", "grid"], default="samples")
    ap.add_argument("--width", type=int, default=256)        # samples / epochs
    ap.add_argument("--fixed-D", type=int, default=10000)    # width mode
    ap.add_argument("--sigma", type=float, default=0.4)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    out = Path(args.outdir); figdir = out / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    pl.set_style()
    prob = TeacherStudentRegression(input_dim=32, teacher_width=256, teacher_depth=2,
                                    noise_std=args.sigma, seed=0)
    t0 = time.time()

    if args.mode in ("samples", "width"):
        rows = []
        xs = SAMPLES_D if args.mode == "samples" else WIDTH_GRID
        combos = [(s, x) for s in range(args.seeds) for x in xs]
        for seed, x in tqdm(combos, desc=args.mode, unit="cell"):
            W = args.width if args.mode == "samples" else x
            D = x if args.mode == "samples" else args.fixed_D
            ex, es, tr, st = train_final(prob, W, D, args.sigma, seed, args.max_steps,
                                         args.batch, args.lr)
            rows.append(dict(W=W, D=D, N=n_params(W), seed=seed, excess=ex,
                             excess_es=es, train=tr, steps=st))
        df = pd.DataFrame(rows); df.to_csv(out / f"double_descent_{args.mode}.csv", index=False)
        axis = "D" if args.mode == "samples" else "N"
        thresh = n_params(args.width) if args.mode == "samples" else args.fixed_D
        fig = pl.plot_dd_final(df, axis, thresh, "D = N" if axis == "D" else "N = D",
                               args.sigma, train=("overlay" if axis == "N" else "none"))
        pl.save_figure(fig, f"double_descent_{args.mode}", str(figdir))
        print(f"saved double_descent_{args.mode}.png")

    elif args.mode == "grid":  # 2D phase diagram over (model size, dataset size), 1 seed
        csv = out / "double_descent_grid.csv"
        rows, done = [], set()
        if csv.exists():
            prev = pd.read_csv(csv); rows = prev.to_dict("records")
            done = {(int(w), int(d)) for w, d in zip(prev["W"], prev["D"])}
        todo = [(W, D) for W in GRID_W for D in GRID_D if (W, D) not in done]
        bar = tqdm(todo, desc="grid", unit="cell")
        for W, D in bar:
            ex, es, tr, st = train_final(prob, W, D, args.sigma, 0, args.max_steps,
                                         args.batch, args.lr)
            rows.append(dict(W=W, D=D, N=n_params(W), excess=ex, excess_es=es,
                             train=tr, steps=st))
            pd.DataFrame(rows).to_csv(csv, index=False)        # checkpoint each cell
            bar.set_postfix(W=W, D=D, test=f"{ex:.3f}", train=f"{tr:.3f}")
        fig = pl.plot_dd_phase(pd.read_csv(csv), args.sigma)
        pl.save_figure(fig, "double_descent_phase", str(figdir))
        print("saved double_descent_phase.png")

    else:  # epochs mode (incremental: skip (D, seed) trajectories already cached)
        csv = out / "double_descent_epochs.csv"
        rows, done = [], set()
        if csv.exists():
            prev = pd.read_csv(csv); rows = prev.to_dict("records")
            done = {(int(d), int(s)) for d, s in zip(prev["D"], prev["seed"])}
        todo = [(D, s) for D in EPOCH_D for s in range(args.seeds) if (D, s) not in done]
        for D, seed in tqdm(todo, desc="epochs", unit="run"):
            for r in train_trajectory(prob, args.width, D, args.sigma, seed,
                                      args.max_steps, args.batch, args.lr):
                rows.append(dict(D=D, seed=seed, **r))
        df = pd.DataFrame(rows); df.to_csv(csv, index=False)
        fig = pl.plot_dd_epochs(df, args.width, n_params(args.width), args.sigma)
        pl.save_figure(fig, "double_descent_epochs", str(figdir))
        print("saved double_descent_epochs.png")

    print(f"elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
