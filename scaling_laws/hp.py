"""The two inner hyperparameter loops that calibrate the learning rate.

The scaling sweep needs every (N, D) cell trained near its own optimal LR, otherwise
mis-tuning biases the fitted exponents. Rather than tune every cell by brute force, we
measure how the optimum *moves* and predict from it:

* L1 -- :func:`tune_lr_cell`: at one (width, n_data) cell, sweep the LR, fit a parabola
  to loss-vs-log10(LR), and return the vertex eta*(w, T).
* L2 -- :func:`fit_transfer_law`: run L1 across a (width, steps) grid and fit
  eta*(w, T) = eta_ref * (w/w_ref)^c_w * (T/T_ref)^c_T by regression in log space.

:func:`sweep.transfer_lr` turns the fitted law into the per-cell rule the grid uses;
``scripts/run_hp_study.py`` is a thin driver over these. See ``notebooks/01_*`` (L1) and
``notebooks/02_*`` (L2).
"""
from __future__ import annotations

import numpy as np
from tqdm import tqdm

from .sweep import run_cell


def tune_lr_cell(problem, width, n_data, lrs, *, n_hidden=2, batch_size=256,
                 seeds=(0, 1), progress=False):
    """L1: the loss-minimising LR at one (width, n_data) cell.

    Trains the cell at each LR in ``lrs`` (averaged over ``seeds``), fits a parabola to
    loss vs log10(LR), and returns ``(eta_star, losses)`` -- the parabola vertex and the
    per-LR mean losses (so callers can plot the U-curve). Falls back to the grid argmin
    if the parabola is not convex. ``progress=True`` shows a bar over the LR sweep.
    """
    lrs = np.asarray(lrs, dtype=float)
    bar = tqdm(lrs, desc=f"tune w={width}", unit="lr") if progress else lrs
    losses = np.array([
        np.mean([run_cell(problem, width, n_data, n_hidden=n_hidden,
                          batch_size=batch_size, lr=float(lr), seed=s)["val_loss"]
                 for s in seeds])
        for lr in bar])
    x = np.log10(lrs)
    a, b, _ = np.polyfit(x, losses, 2)
    x_star = np.clip(-b / (2 * a), x.min(), x.max()) if a > 0 else x[int(np.argmin(losses))]
    return float(10 ** x_star), losses


def fit_law_from_points(widths, eta_width, Tsteps, eta_T, w_ref, T_ref):
    """The L2 regression: fit the exponents from already-measured eta* points (no training).

    eta*(w, T) = eta_ref * (w/w_ref)^c_w * (T/T_ref)^c_T, with c_w, c_T from straight-line
    fits of log(eta*) vs log(w) and log(T). Returned as a plain dict (JSON-friendly).
    """
    c_w, b_w = np.polyfit(np.log(widths), np.log(eta_width), 1)
    c_T, _ = np.polyfit(np.log(Tsteps), np.log(eta_T), 1)
    eta_ref = float(np.exp(b_w) * w_ref ** c_w)        # law value at (w_ref, T_ref)
    return dict(eta_ref=round(eta_ref, 5), w_ref=w_ref, T_ref=T_ref,
                c_w=round(float(c_w), 3), c_T=round(float(c_T), 3),
                widths=list(widths), eta_width=list(eta_width),
                Tsteps=list(Tsteps), eta_T=list(eta_T))


def fit_transfer_law(problem, *, widths, Tsteps, w_ref, T_ref, batch_size=256,
                     lrs=None, seeds=(0, 1), progress=False):
    """L2: measure eta* across the (width, T) grid via :func:`tune_lr_cell`, then fit.

    Sweeps eta* vs width (at T_ref) and vs steps T (at w_ref); a cell with step budget T
    uses n_data = T * batch_size. Returns the fitted-law dict from
    :func:`fit_law_from_points`.
    """
    lrs = np.geomspace(3e-4, 1e-1, 9) if lrs is None else np.asarray(lrs, dtype=float)
    wbar = tqdm(widths, desc="eta* vs width", unit="w") if progress else widths
    eta_width = [tune_lr_cell(problem, w, T_ref * batch_size, lrs,
                              batch_size=batch_size, seeds=seeds)[0] for w in wbar]
    tbar = tqdm(Tsteps, desc="eta* vs T", unit="T") if progress else Tsteps
    eta_T = [tune_lr_cell(problem, w_ref, T * batch_size, lrs,
                          batch_size=batch_size, seeds=seeds)[0] for T in tbar]
    return fit_law_from_points(widths, eta_width, Tsteps, eta_T, w_ref, T_ref)
