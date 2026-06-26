"""Live training for the tutorial notebooks: transparent loops that print a progress
bar and build a loss curve in real time, so the descent of cheap cells is visible.

All three helpers share ``sweep.run_cell``'s recipe (``make_mlp`` + AdamW + cosine-warmup
schedule, single pass) -- the only difference is that the loop is spelled out so it can
be watched.

* :func:`train_cell_live`  -- one cell, one live curve (notebook 00).
* :func:`train_cells_live` -- a list of (width, D) cells, curves overlaid (notebook 00).
* :func:`sweep_lr_live`    -- one cell at many LRs, curves coloured by LR (notebook 01).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .models import make_mlp, cosine_lr_lambda


def train_cell_live(problem, width, n_data, lr, *, n_hidden=2, batch_size=256, seed=0,
                    weight_decay=0.0, plot=True, progress=True):
    """Train one cell, live; return ``(val_loss, steps, train_losses)``.

    Same recipe as :func:`sweep.run_cell`, written as an explicit single-pass loop. With
    ``plot=True`` a loss curve updates in place as training proceeds; ``progress=True``
    prints a per-step tqdm bar. Set both ``False`` to train quietly (used by the overlay
    helpers below).
    """
    bs = min(batch_size, n_data)
    total_steps = problem.n_steps(n_data, bs)
    torch.manual_seed(seed)
    model = make_mlp(problem.input_dim, width, n_hidden, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_lr_lambda(total_steps))
    loader = problem.train_loader(n_data, bs, seed=seed * 100003 + width)

    steps, losses = [], []
    fig = ax = handle = None
    if plot:
        import matplotlib.pyplot as plt
        from IPython.display import display
        fig, ax = plt.subplots(figsize=(6.2, 4))
        handle = display(fig, display_id=True)
    every = max(1, total_steps // 40)

    def redraw():
        ax.clear()
        ax.plot(steps, losses, color="steelblue", lw=1.2)
        ax.axhline(problem.irreducible_loss, ls=":", c="k", lw=1,
                   label=rf"floor $E={problem.irreducible_loss:g}$")
        ax.set(xlabel="optimiser step", ylabel="training loss (MSE)", yscale="log")
        ax.legend(frameon=False); fig.tight_layout(); handle.update(fig)

    step = 0
    loader_it = (tqdm(loader, total=total_steps, desc=f"w={width}, D={n_data:,}", unit="step")
                 if progress else loader)
    for x, y in loader_it:
        opt.zero_grad()
        loss = F.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        sched.step()
        step += 1
        steps.append(step); losses.append(loss.item())
        if plot and step % every == 0:
            redraw()
        if step >= total_steps:
            break

    val_loss = problem.evaluate(model)
    if plot:
        redraw()
        import matplotlib.pyplot as plt
        plt.close(fig)
    return val_loss, steps, losses


def train_cells_live(problem, cells, lr=0.005, *, n_hidden=2, batch_size=256, seed=0):
    """Train a list of cells, overlaying their training curves (one per cell, with a
    legend), and return run records for tabulation. Each cell is ``(width, n_data)`` or
    ``(width, n_data, lr)`` -- the optional third entry overrides the default ``lr``. Used
    by the cheap multi-cell grid in notebook 00."""
    import matplotlib.pyplot as plt
    from IPython.display import display
    from .flops import mlp_param_count, compute

    colors = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    ax.axhline(problem.irreducible_loss, ls=":", c="k", lw=1,
               label=rf"floor $E={problem.irreducible_loss:g}$")
    ax.set(xscale="log", yscale="log", xlabel="optimiser step", ylabel="training loss (MSE)")
    handle = display(fig, display_id=True)

    show_lr = len({(c[2] if len(c) > 2 else lr) for c in cells}) > 1   # only label eta if it varies
    records = []
    for i, cell in enumerate(tqdm(cells, desc="cells", unit="cell")):
        w, d = cell[0], cell[1]
        cell_lr = cell[2] if len(cell) > 2 else lr
        val, steps, losses = train_cell_live(problem, w, d, cell_lr, n_hidden=n_hidden,
                                             batch_size=batch_size, seed=seed,
                                             plot=False, progress=False)
        N = mlp_param_count(problem.input_dim, w, n_hidden)
        records.append(dict(width=w, N=N, D=d, lr=cell_lr, steps=len(steps),
                            C=compute(N, d, problem.input_dim, w), val_loss=val))
        lbl = fr"w={w}, D={d:,}" + (fr", $\eta$={cell_lr:g}" if show_lr else "")
        ax.plot(steps, losses, color=colors(i % 10), lw=1.4, label=lbl)
        ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
        handle.update(fig)
    plt.close(fig)
    return records


def sweep_lr_live(problem, width, n_data, lrs, *, n_hidden=2, batch_size=256, seed=0):
    """L1 sweep, live: train the cell at each LR, building up the per-LR training curves
    coloured by learning rate. Returns ``(eta_star, final_losses)``.

    You watch the hottest rates go noisy and the coldest underfit, with the optimum
    threading between. ``eta_star`` is the parabola vertex of final loss vs log10(LR).
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from IPython.display import display

    lrs = np.asarray(lrs, dtype=float)
    cmap, norm = cm.viridis, mcolors.LogNorm(lrs.min(), lrs.max())
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.axhline(problem.irreducible_loss, ls=":", c="k", lw=1,
               label=rf"floor $E={problem.irreducible_loss:g}$")
    ax.set(yscale="log", xlabel="optimiser step", ylabel="training loss (MSE)")
    ax.legend(frameon=False)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r"learning rate $\eta$")
    handle = display(fig, display_id=True)

    final = []
    for lr in tqdm(lrs, desc=f"tune w={width}", unit="lr"):
        val, steps, losses = train_cell_live(problem, width, n_data, float(lr),
                                             n_hidden=n_hidden, batch_size=batch_size,
                                             seed=seed, plot=False, progress=False)
        final.append(val)
        ax.plot(steps, losses, color=cmap(norm(lr)), lw=1, alpha=0.9)
        handle.update(fig)
    plt.close(fig)

    final = np.asarray(final)
    x = np.log10(lrs)
    a, b, _ = np.polyfit(x, final, 2)
    x_star = np.clip(-b / (2 * a), x.min(), x.max()) if a > 0 else x[int(np.argmin(final))]
    return float(10 ** x_star), final


def measure_law_live(problem, *, widths, Tsteps, w_ref, T_ref, batch_size=256, lrs=None, seed=0):
    """L2, live: measure eta* across a few widths and step budgets (each via tune_lr_cell),
    building up the eta*-vs-width and eta*-vs-T points, then fit and overlay the two power
    laws. Returns the fitted-law dict.

    A cheap, illustrative version of ``scripts/run_hp_study.py`` -- the committed law in
    ``results/hp_study_cosine.json`` uses a denser grid and two seeds.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from IPython.display import display
    from .hp import tune_lr_cell, fit_law_from_points

    lrs = np.geomspace(3e-4, 1e-1, 7) if lrs is None else np.asarray(lrs, dtype=float)
    lx = np.log10(lrs)
    widths, Tsteps = list(widths), list(Tsteps)
    fig, ((axWc, axTc), (axW, axT)) = plt.subplots(2, 2, figsize=(11, 8))
    for ax in (axWc, axTc):                                   # top: the LR sweeps
        ax.set(xscale="log", yscale="log", xlabel=r"learning rate $\eta$", ylabel="validation loss")
    axW.set(xscale="log", yscale="log", xlabel=r"width $w$ (at $T=T_{\rm ref}$)",
            ylabel=r"optimal LR $\eta^\star$")
    axT.set(xscale="log", yscale="log", xlabel=r"steps $T=D/b$ (at $w=w_{\rm ref}$)",
            ylabel=r"optimal LR $\eta^\star$")
    handle = display(fig, display_id=True)
    wcol = cm.viridis(np.linspace(0.15, 0.9, len(widths)))
    tcol = cm.plasma(np.linspace(0.1, 0.8, len(Tsteps)))

    def starred(ax, losses, eta, color, label):
        a, b, c = np.polyfit(lx, losses, 2)
        ax.plot(lrs, losses, "o-", color=color, ms=4, lw=1, label=label)
        ax.plot(eta, np.polyval([a, b, c], np.log10(eta)), "*", color=color, ms=15,
                mec="k", zorder=5)                            # the chosen eta*
        ax.legend(frameon=False, fontsize=8)

    eW = []
    for j, w in enumerate(tqdm(widths, desc="eta* vs width", unit="w")):
        eta, losses = tune_lr_cell(problem, w, T_ref * batch_size, lrs, batch_size=batch_size, seeds=(seed,))
        eW.append(eta)
        starred(axWc, losses, eta, wcol[j], f"w={w}")
        axW.plot(widths[:len(eW)], eW, "o", color="navy", ms=8); handle.update(fig)
    eT = []
    for j, T in enumerate(tqdm(Tsteps, desc="eta* vs T", unit="T")):
        eta, losses = tune_lr_cell(problem, w_ref, T * batch_size, lrs, batch_size=batch_size, seeds=(seed,))
        eT.append(eta)
        starred(axTc, losses, eta, tcol[j], f"T={T}")
        axT.plot(Tsteps[:len(eT)], eT, "o", color="seagreen", ms=8); handle.update(fig)

    law = fit_law_from_points(widths, eW, Tsteps, eT, w_ref, T_ref)
    ww = np.geomspace(min(widths), max(widths), 50)
    axW.plot(ww, law["eta_ref"] * (ww / w_ref) ** law["c_w"], "--", color="navy",
             label=rf"$\eta^\star\!\propto w^{{{law['c_w']}}}$")
    tt = np.geomspace(min(Tsteps), max(Tsteps), 50)
    axT.plot(tt, law["eta_ref"] * (tt / T_ref) ** law["c_T"], "--", color="seagreen",
             label=rf"$\eta^\star\!\propto T^{{{law['c_T']}}}$")
    axW.legend(frameon=False); axT.legend(frameon=False)
    fig.tight_layout(); handle.update(fig); plt.close(fig)
    return law
