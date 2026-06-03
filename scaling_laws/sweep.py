"""Run the (N, D) grid that feeds all three scaling-law approaches.

Each *cell* is one single-pass training run of a model of size N on D fresh
examples, annealed with its own cosine schedule. We record the final validation
loss. From the resulting table of (N, D, loss) every approach is derived:

* Approach 1 reads the lower envelope of the per-N loss-vs-compute curves.
* Approach 2 reads the loss-minimising N along iso-compute (C = 6ND) slices.
* Approach 3 fits L(N, D) = E + A/N^a + B/D^b to the whole table.
"""
from __future__ import annotations

import logging
import os
import warnings

import pandas as pd
import torch
import lightning as L
from tqdm import tqdm

from .data import TeacherStudentRegression
from .flops import compute
from .models import LitMLP

# Keep the sweep quiet -- hundreds of tiny runs.
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*GPU available but not used.*")


def transfer_lr(eta_ref: float, w_ref: float, T_ref: float, c_w: float, c_T: float,
                T_min: int = 256, lr_max: float = 0.05, lr_min: float = 3e-4):
    """A per-cell learning-rate rule eta*(width, steps) from the HP-transfer study.

    eta* = eta_ref * (width/w_ref)**c_w * (T_eff/T_ref)**c_T, with T_eff = max(T, T_min)
    so the (steep) T-law is not extrapolated into the warmup-dominated few-step regime,
    and the result clamped to [lr_min, lr_max] for stability. This is a miniature
    analogue of muP-style hyperparameter transfer: every grid
    cell is launched at its own optimum instead of one shared LR.
    """
    def rule(width: int, steps: int) -> float:
        T_eff = max(steps, T_min)
        lr = eta_ref * (width / w_ref) ** c_w * (T_eff / T_ref) ** c_T
        return float(min(lr_max, max(lr_min, lr)))
    return rule


def run_cell(problem: TeacherStudentRegression, width: int, n_data: int, *,
             n_hidden: int = 2, batch_size: int = 256, lr: float = 3e-3,
             weight_decay: float = 0.0, iso_C: float = 0.0, seed: int = 0) -> dict:
    """Train one (N, D) cell single-pass and return its record."""
    bs = min(batch_size, n_data)
    total_steps = problem.n_steps(n_data, bs)

    L.seed_everything(seed)
    model = LitMLP(problem.input_dim, width, n_hidden, output_dim=1, lr=lr,
                   weight_decay=weight_decay, total_steps=total_steps)
    n_params = model.n_params

    init_loss = problem.evaluate(model)
    train_loader = problem.train_loader(n_data, bs, seed=seed * 100003 + width)

    trainer = L.Trainer(
        accelerator="cpu", devices=1, max_steps=total_steps, max_epochs=1,
        enable_progress_bar=False, enable_checkpointing=False, logger=False,
        enable_model_summary=False, num_sanity_val_steps=0, limit_val_batches=0)
    trainer.fit(model, train_loader)

    val_loss = problem.evaluate(model)
    return dict(width=width, n_hidden=n_hidden, N=n_params, D=n_data,
                C=compute(n_params, n_data, problem.input_dim, width),
                batch_size=bs, lr=lr, steps=total_steps,
                iso_C=iso_C, seed=seed, init_loss=init_loss, val_loss=val_loss)


def run_grid(problem: TeacherStudentRegression, widths, n_data_list, *,
             n_hidden: int = 2, batch_size: int = 256, lr: float = 3e-3,
             lr_rule=None, seeds=(0, 1), cache_path: str | None = None,
             force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Run the grid (incrementally cached to CSV) and return the raw records.

    If ``lr_rule`` (a callable ``(width, steps) -> lr``) is given, every cell is
    trained at its own tuned learning rate; otherwise the fixed ``lr`` is used.

    Caching is *incremental*: cells already present in ``cache_path`` (matched on
    seed/width/D) are kept and skipped, so extending ``widths`` or ``n_data_list``
    only trains the genuinely new cells. ``force=True`` recomputes.
    """
    records, done = [], set()
    if cache_path and os.path.exists(cache_path) and not force:
        cached = pd.read_csv(cache_path)
        records = cached.to_dict("records")
        done = {(int(s), int(w), int(d))
                for s, w, d in zip(cached["seed"], cached["width"], cached["D"])}
        if verbose:
            print(f"Loaded {len(records)} cached cells from {cache_path}")

    todo = [(s, w, d) for s in seeds for w in widths for d in n_data_list
            if (int(s), int(w), int(d)) not in done]
    bar = tqdm(todo, desc="grid", unit="cell", disable=not verbose)
    for seed, width, n_data in bar:
        bs = min(batch_size, n_data)
        lr_cell = lr_rule(width, problem.n_steps(n_data, bs)) if lr_rule else lr
        rec = run_cell(problem, width, n_data, n_hidden=n_hidden,
                       batch_size=batch_size, lr=lr_cell, seed=seed)
        records.append(rec)
        bar.set_postfix(w=width, D=n_data, loss=f"{rec['val_loss']:.3f}")

    df = pd.DataFrame.from_records(records).sort_values(
        ["seed", "width", "D"]).reset_index(drop=True)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        df.to_csv(cache_path, index=False)
        if verbose:
            print(f"Saved {len(df)} cells to {cache_path}")
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Average the per-seed cells into one (N, D) record with a loss std.

    ``iso_C`` (the compute budget of a dedicated IsoFLOP run; 0 for ordinary grid
    cells) is carried through so Approach 2 can use those runs directly.
    """
    keys = ["width", "n_hidden", "N", "D"]
    if "iso_C" in df.columns:
        df = df.copy(); df["iso_C"] = df["iso_C"].fillna(0.0)
        keys.append("iso_C")
    grp = df.groupby(keys, as_index=False)
    out = grp.agg(C=("C", "first"), batch_size=("batch_size", "first"),
                  steps=("steps", "first"),
                  val_loss=("val_loss", "mean"), val_loss_std=("val_loss", "std"),
                  init_loss=("init_loss", "mean"))
    out["val_loss_std"] = out["val_loss_std"].fillna(0.0)
    return out.sort_values(["N", "D"]).reset_index(drop=True)
