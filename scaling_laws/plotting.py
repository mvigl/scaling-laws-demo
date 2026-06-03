"""Plotting helpers, one per approach, plus an overview and a comparison."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import mplhep as hep
import numpy as np
import pandas as pd

from .approaches import EnvelopeResult, IsoFlopResult, ParametricResult


def set_style():
    """The mplhep ATLAS style (boxed axes, inward major+minor ticks, Helvetica-like
    font), lightly adapted for the repo's multi-panel log-log tutorial figures."""
    hep.style.use("ATLAS")
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 120, "savefig.bbox": "tight",
        "axes.titlesize": 14, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 11, "legend.frameon": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.alpha": 0.3,
        "grid.linewidth": 0.6, "lines.markersize": 6,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def save_figure(fig, name: str, outdir: str = "results/figures"):
    """Save a figure as both PDF (for the repo) and PNG (for quick preview)."""
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=120, bbox_inches="tight")
    return fig


def _N_colormap(agg):
    norm = mcolors.LogNorm(vmin=agg["N"].min(), vmax=agg["N"].max())
    return cm.viridis, norm


def plot_all_runs(agg: pd.DataFrame, irreducible: float | None = None):
    """One line per model size N, vs data D (left) and vs compute C (right).

    The data view shows each model's single-pass training curve (small models plateau
    at their capacity, big ones keep descending); the compute view is the same data
    re-indexed by C=6ND, whose lower envelope is the compute-optimal frontier.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap, norm = _N_colormap(agg)
    for N, g in agg.groupby("N"):
        c = cmap(norm(N))
        gd = g.sort_values("D")
        axes[0].plot(gd["D"], gd["val_loss"], "-o", ms=3, color=c, alpha=0.9)
        gc = g.sort_values("C")
        axes[1].plot(gc["C"], gc["val_loss"], "-o", ms=3, color=c, alpha=0.9)
    for ax in axes:
        if irreducible is not None:
            ax.axhline(irreducible, ls=":", c="k", lw=1, label=f"irreducible  E={irreducible:g}")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylabel("validation loss")
    axes[0].set_xlabel("data  $D$  (examples seen)")
    axes[1].set_xlabel(r"compute  $C=(6N-2dw)D$  (FLOPs)")
    if irreducible is not None:
        axes[0].legend(frameon=False)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=axes, label="N (params)")
    return fig


def plot_envelope(agg: pd.DataFrame, env: EnvelopeResult, irreducible: float | None = None):
    """Approach 1: the lower envelope and the implied N*(C), D*(C)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    cmap, norm = _N_colormap(agg)
    ax = axes[0]
    for N, g in agg.groupby("N"):
        g = g.sort_values("C")
        ax.plot(g["C"], g["val_loss"], "-", color=cmap(norm(N)), alpha=0.4, lw=1)
    ax.plot(env.frontier["C"], env.frontier["val_loss"], "o-", c="crimson",
            label=r"compute-optimal frontier $L^\star(C)$")
    if irreducible is not None:
        ax.axhline(irreducible, ls=":", c="k", lw=1, label=f"known floor $E$={irreducible:g}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$C=(6N-2dw)D$"); ax.set_ylabel("loss (MSE)"); ax.legend(frameon=False)

    C = env.frontier["C"].values
    axes[1].plot(C, env.frontier["N"], "o", c="navy")
    axes[1].plot(C, env.N_star(C), "--", c="navy",
                 label=rf"$N^\star\!\propto C^{{{env.a_N:.3f}}}$")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("C"); axes[1].set_ylabel(r"$N^\star$"); axes[1].legend(frameon=False)

    axes[2].plot(C, env.frontier["D"], "o", c="seagreen")
    axes[2].plot(C, env.D_star(C), "--", c="seagreen",
                 label=rf"$D^\star\!\propto C^{{{env.a_D:.3f}}}$")
    axes[2].set_xscale("log"); axes[2].set_yscale("log")
    axes[2].set_xlabel("C"); axes[2].set_ylabel(r"$D^\star$"); axes[2].legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_isoflop(iso: IsoFlopResult):
    """Approach 2: iso-compute parabolas and the recovered allocation."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ax = axes[0]
    colors = cm.plasma(np.linspace(0, 0.9, len(iso.profiles)))
    for (C, prof), col in zip(sorted(iso.profiles.items()), colors):
        ax.plot(prof["N"], prof["val_loss"], "o", color=col, ms=4)
        xs = np.log10(prof["N"].values)
        xg = np.linspace(xs.min(), xs.max(), 100)
        p2, p1, p0 = np.polyfit(xs, prof["val_loss"].values, 2)
        ax.plot(10 ** xg, p0 + p1 * xg + p2 * xg ** 2, "-", color=col, lw=1,
                label=f"C={C:.1e}")
        row = iso.minima[iso.minima["C"] == C]
        if len(row):
            ax.plot(row["N_star"], row["loss_star"], "*", color=col, ms=14,
                    markeredgecolor="k")
    ax.set_xscale("log")
    ax.set_xlabel("N (params)"); ax.set_ylabel("loss")
    ax.legend(frameon=False, fontsize=8)

    C = iso.minima["C"].values
    axes[1].plot(C, iso.minima["N_star"], "*", c="navy", ms=12)
    axes[1].plot(C, iso.N_star(C), "--", c="navy",
                 label=rf"$N^\star\!\propto C^{{{iso.a_N:.3f}}}$")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("C"); axes[1].set_ylabel(r"$N^\star$"); axes[1].legend(frameon=False)

    axes[2].plot(C, iso.minima["D_star"], "*", c="seagreen", ms=12)
    axes[2].plot(C, iso.D_star(C), "--", c="seagreen",
                 label=rf"$D^\star\!\propto C^{{{iso.a_D:.3f}}}$")
    axes[2].set_xscale("log"); axes[2].set_yscale("log")
    axes[2].set_xlabel("C"); axes[2].set_ylabel(r"$D^\star$"); axes[2].legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_parametric(agg: pd.DataFrame, par: ParametricResult, irreducible: float | None = None):
    """Approach 3: fitted surface in the (N, D) plane + compute-optimal locus."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    Ng = np.geomspace(agg["N"].min(), agg["N"].max(), 120)
    Dg = np.geomspace(agg["D"].min(), agg["D"].max(), 120)
    NN, DD = np.meshgrid(Ng, Dg)
    Z = par.loss(NN, DD)
    ax = axes[0]
    pc = ax.pcolormesh(Ng, Dg, Z, shading="auto", cmap="magma_r")
    ax.scatter(agg["N"], agg["D"], s=18, facecolors="none", edgecolors="w", lw=0.8)
    Cl = np.geomspace(agg["C"].min(), agg["C"].max(), 50)
    ax.plot(par.N_star(Cl), par.D_star(Cl), c="cyan", lw=2, label="compute-optimal locus")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(Ng.min(), Ng.max()); ax.set_ylim(Dg.min(), Dg.max())   # no margin around the mesh
    ax.set_xlabel("N (params)"); ax.set_ylabel("D (examples)")
    ax.legend(frameon=False); plt.colorbar(pc, ax=ax, label="predicted loss")

    pred = par.loss(agg["N"].values, agg["D"].values)
    ax2 = axes[1]
    ax2.scatter(agg["val_loss"], pred, s=18, c="slateblue")
    lim = [agg["val_loss"].min() * 0.95, agg["val_loss"].max() * 1.05]
    ax2.plot(lim, lim, "k--", lw=1)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("measured loss"); ax2.set_ylabel("fitted loss")
    if irreducible is not None:
        txt = f"fitted  E = {par.E:.4f}\nknown   E = {irreducible:.4f}"
        ax2.text(0.05, 0.95, txt, transform=ax2.transAxes, va="top", ha="left",
                 family="monospace", fontsize=10,
                 bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    fig.tight_layout()
    return fig


def plot_comparison(env: EnvelopeResult, iso: IsoFlopResult, par: ParametricResult,
                    C_range):
    """Overlay N*(C) from all three approaches and tabulate the exponents."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    C = np.geomspace(*C_range, 100)
    ax = axes[0]
    ax.plot(C, env.N_star(C), c="crimson", label=f"approach 1  (a={env.a_N:.3f})")
    ax.plot(C, iso.N_star(C), c="navy", ls="--", label=f"approach 2  (a={iso.a_N:.3f})")
    ax.plot(C, par.N_star(C), c="green", ls=":", lw=2,
            label=f"approach 3  (a={par.a_N:.3f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$C=(6N-2dw)D$"); ax.set_ylabel(r"$N^\star(C)$")
    ax.legend(frameon=False)

    ax2 = axes[1]; ax2.axis("off")
    rows = [["", "a_N  (N*~C^a)", "a_D  (D*~C^a)"],
            ["Approach 1 (envelope)", f"{env.a_N:.3f}", f"{env.a_D:.3f}"],
            ["Approach 2 (IsoFLOP)", f"{iso.a_N:.3f}", f"{iso.a_D:.3f}"],
            ["Approach 3 (parametric)", f"{par.a_N:.3f}", f"{par.a_D:.3f}"]]
    tbl = ax2.table(cellText=rows, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2)
    fig.tight_layout()
    return fig


def plot_lr_ablation(df: pd.DataFrame, irreducible: float, n_hidden: int = 2):
    """The cost of *not* re-tuning the LR, along D (left) and along N (right).

    `df` has columns panel ('D'/'N'), x, width, tuned ('tuned'/'fixed'), val_loss.
    We plot the reducible (excess) loss L - E, averaged over seeds; the gap between
    the per-cell-tuned curve and the single held-fixed LR is the tuning tax.
    """
    from .flops import mlp_param_count
    g = df.groupby(["panel", "x", "width", "tuned"]).val_loss.mean().reset_index()
    g["excess"] = g["val_loss"] - irreducible
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    styles = {"tuned": dict(color="seagreen", marker="o", ls="-", label="per-cell tuned $\\eta^\\star$"),
              "fixed": dict(color="crimson", marker="s", ls="--", label="one fixed $\\eta$")}

    def panel(ax, key, xvals, xlabel):
        for t in ("tuned", "fixed"):
            d = g[(g.panel == key) & (g.tuned == t)].sort_values("x")
            ax.plot(xvals(d), d["excess"], **styles[t])
        ax.set(xscale="log", yscale="log", xlabel=xlabel,
               ylabel=r"excess loss  $L - E$"); ax.legend()

    panel(axes[0], "D", lambda d: d["x"], "data $D$ (examples)")
    panel(axes[1], "N", lambda d: [mlp_param_count(32, int(w), n_hidden) for w in d["width"]],
          "N (params)")
    fig.tight_layout()
    return fig


def plot_hp_study(law: dict):
    """HP-transfer study: optimal LR vs width (μP shift) and vs step budget (horizon)."""
    w = np.array(law["widths"], float); eW = np.array(law["eta_width"], float)
    T = np.array(law["Tsteps"], float); eT = np.array(law["eta_T"], float)
    c_w, b_w = np.polyfit(np.log(w), np.log(eW), 1)
    c_T, b_T = np.polyfit(np.log(T), np.log(eT), 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.7))
    ax[0].plot(w, eW, "o", c="darkgreen")
    ax[0].plot(w, np.exp(b_w) * w ** c_w, "--", c="darkgreen",
               label=rf"$\eta^\star\propto w^{{{c_w:.2f}}}$")
    ax[0].set(xscale="log", yscale="log", xlabel="width", ylabel=r"optimal $\eta^\star$")
    ax[0].legend()
    ax[1].plot(T, eT, "o", c="navy")
    ax[1].plot(T, np.exp(b_T) * T ** c_T, "--", c="navy",
               label=rf"$\eta^\star\propto T^{{{c_T:.2f}}}$")
    ax[1].set(xscale="log", yscale="log", xlabel="steps T", ylabel=r"optimal $\eta^\star$")
    ax[1].legend()
    fig.tight_layout()
    return fig


def plot_flop_validation(df: pd.DataFrame):
    """Closed-form training FLOPs/example vs a real FlopCounterMode measurement.

    `df` has columns N, measured, formula (=6N-2dw), naive (=6N). Left: the three
    overlaid; right: their ratio to the measurement (the first-layer term is what
    pulls 6N down to the truth, most at small N)."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(df.N, df.measured, "o", color="crimson", ms=8, zorder=3,
               label="measured (FlopCounterMode)")
    ax[0].plot(df.N, df.formula, "-", color="seagreen", label=r"closed form  $6N-2dw$")
    ax[0].plot(df.N, df.naive, "--", color="0.55", label=r"naive  $6N$")
    ax[0].set(xscale="log", yscale="log", xlabel="N (params)",
              ylabel="training FLOPs / example")
    ax[0].legend()
    ax[1].axhline(1, ls=":", c="k", lw=1)
    ax[1].plot(df.N, df.measured / df.formula, "o-", color="seagreen",
               label=r"measured / $(6N-2dw)$")
    ax[1].plot(df.N, df.measured / df.naive, "s--", color="0.55", label=r"measured / $6N$")
    ax[1].set(xscale="log", xlabel="N (params)", ylabel="measured / closed form")
    ax[1].legend()
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Double descent (repeated-data regime)
# --------------------------------------------------------------------------- #
def plot_dd_final(df: pd.DataFrame, axis: str, threshold: float, threshold_label: str,
                  sigma: float, train: str = "panel"):
    """Sample-/model-wise double descent vs (D or N). Test loss is the MSE against the
    *noisy* labels (excess risk + σ²), so it shares the σ² floor with the train MSE.
    ``train`` places the train MSE: 'panel' = a separate right panel, 'overlay' = on the
    test panel (shared log axis), 'none' = hidden. The interpolation threshold N~D is
    marked; if ``excess_es`` is present the oracle early-stopped test loss is overlaid."""
    s2 = sigma ** 2
    aggs = dict(excess=("excess", "mean"), train=("train", "mean"))
    if "excess_es" in df:
        aggs["excess_es"] = ("excess_es", "mean")
    g = df.groupby(axis).agg(**aggs).reset_index().sort_values(axis)
    xlabel = "dataset size $D$" if axis == "D" else "model size $N$ (params)"
    if train == "panel":
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8)); a0, panels = ax[0], list(ax)
    else:
        fig, a0 = plt.subplots(figsize=(7.5, 5)); panels = [a0]

    a0.plot(g[axis], g.excess + s2, "-o", color="crimson", label="final")
    if "excess_es" in g:
        a0.plot(g[axis], g.excess_es + s2, "-o", color="seagreen", label="early-stopped")
    if train == "overlay":
        a0.plot(g[axis], g.train, "--o", color="steelblue", ms=4, label="train MSE")
    a0.axhline(s2, ls=":", c="0.5", lw=1)
    a0.text(g[axis].min(), s2, r" $\sigma^2$", va="bottom", ha="left", fontsize=9, color="0.4")
    a0.set(xscale="log", yscale="log", xlabel=xlabel,
           ylabel=("loss" if train == "overlay" else "test loss"))

    if train == "panel":
        ax[1].plot(g[axis], g.train, "-o", color="steelblue")
        ax[1].axhline(s2, ls=":", c="0.5", lw=1)
        ax[1].text(g[axis].min(), s2, r" $\sigma^2$", va="bottom", ha="left", fontsize=9, color="0.4")
        ax[1].set(xscale="log", yscale="log", xlabel=xlabel, ylabel="train MSE")

    for a in panels:
        a.axvline(threshold, ls="--", c="0.55", lw=1.2)
        a.text(threshold, 0.97, f" {threshold_label}", transform=a.get_xaxis_transform(),
               va="top", ha="left", fontsize=9, color="0.4")
        a.legend()
    fig.tight_layout()
    return fig


def plot_dd_epochs(df: pd.DataFrame, width: int, n_params: int, sigma: float = 0.4):
    """Epoch-wise double descent: test loss (solid) and train MSE (dashed) vs step, on
    **one** shared log-loss axis. Test loss is against the *noisy* labels (excess + σ²),
    so it shares the σ² floor with train.

    A star marks each curve's test minimum -- where an oracle early stop would land,
    before the test risk climbs again (fitting the noise) and second-descends.
    """
    s2 = sigma ** 2
    fig, ax = plt.subplots(figsize=(8, 5))
    Ds = sorted(df["D"].unique())
    colors = cm.viridis(np.linspace(0.2, 0.65, len(Ds)))
    for D, c in zip(Ds, colors):
        g = df[df.D == D].groupby("step").agg(excess=("excess", "mean"),
                                              train=("train", "mean")).reset_index()
        ax.plot(g.step, g.excess + s2, "-", color=c, label=f"D={D:,} ({D/n_params:.2f} N)")
        ax.plot(g.step, g.train, "--", color=c, alpha=0.6)
        im = g.excess.idxmin()
        ax.plot(g.step[im], g.excess[im] + s2, "*", color=c, ms=16, markeredgecolor="k", zorder=5)
    ax.axhline(s2, ls=":", c="0.5", lw=1)
    ax.text(1, s2, r" $\sigma^2$", va="bottom", ha="left", fontsize=9, color="0.4")
    ax.plot([], [], "k--", alpha=0.6, label="train MSE")
    ax.plot([], [], "k*", ms=12, label="early-stop min")
    ax.set(xscale="log", yscale="log", xlabel="training step", ylabel="loss")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    return fig


def plot_dd_phase(df: pd.DataFrame, sigma: float = 0.4, n_hidden: int = 2, input_dim: int = 32):
    """2D double-descent phase diagram (repeated-data regime): final test loss (left) and
    train MSE (right) over the model-size x dataset-size grid. Test loss is against the
    *noisy* labels (excess + σ²). The bright test ridge tracks the interpolation threshold
    N ~ D (dashed), where train error collapses to zero -- the Nakkiran et al. picture."""
    from .flops import mlp_param_count
    Ws = sorted(df["W"].unique()); Ds = sorted(df["D"].unique())
    test = df.pivot(index="D", columns="W", values="excess").reindex(index=Ds, columns=Ws).values + sigma ** 2
    train = df.pivot(index="D", columns="W", values="train").reindex(index=Ds, columns=Ws).values
    Ns = np.array([mlp_param_count(input_dim, w, n_hidden) for w in Ws], float)
    ridge = [float(np.interp(np.log(D), np.log(Ns), np.arange(len(Ws)))) for D in Ds]  # N~D index

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, M, clabel in [(axes[0], test, "test loss"),
                          (axes[1], train, "train MSE")]:
        pos = M[np.isfinite(M) & (M > 0)]
        norm = mcolors.LogNorm(vmin=max(pos.min(), 1e-4), vmax=M[np.isfinite(M)].max())
        im = ax.imshow(M, origin="lower", aspect="auto", cmap="magma", norm=norm)
        ax.plot(ridge, range(len(Ds)), "w--", lw=1.6, label=r"$N \approx D$")
        ax.set_xticks(range(len(Ws))); ax.set_xticklabels(Ws, rotation=45, fontsize=8)
        ax.set_yticks(range(len(Ds))); ax.set_yticklabels([f"{d:,}" for d in Ds], fontsize=8)
        ax.set_xlabel("model size (width)"); ax.set_ylabel("num. train samples $D$")
        ax.set_xlim(-0.5, len(Ws) - 0.5); ax.set_ylim(-0.5, len(Ds) - 0.5)  # ridge can't pad it
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(False); fig.colorbar(im, ax=ax, label=clabel)
    fig.tight_layout()
    return fig
