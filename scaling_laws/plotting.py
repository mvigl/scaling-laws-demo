"""Plotting helpers, one per approach, plus an overview and a comparison."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from .approaches import EnvelopeResult, IsoFlopResult, ParametricResult


def set_style():
    """Paper style: clean boxed axes with inward ticks, no grid, no titles, large
    axis labels, framed legends with qualitative entries only (no fitted numbers or
    functional forms), and a plasma colormap for model size."""
    plt.rcdefaults()
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 12, "axes.labelsize": 16, "axes.titlesize": 16,
        "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
        "axes.linewidth": 1.1,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.size": 5.5, "ytick.major.size": 5.5,
        "xtick.minor.size": 3.0, "ytick.minor.size": 3.0,
        "legend.fontsize": 10.5, "legend.frameon": True,
        "legend.edgecolor": "0.2", "legend.framealpha": 1.0,
        "legend.fancybox": False,
        "axes.grid": False,
        "lines.linewidth": 1.6, "lines.markersize": 6,
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
    return cm.plasma, norm


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
        axes[0].plot(gd["D"], gd["val_loss"], "-", lw=1.3, color=c, alpha=0.9)
        axes[0].plot(gd["D"], gd["val_loss"], "o", ms=4.5, color=c,
                     markeredgecolor="0.15", markeredgewidth=0.5)
        gc = g.sort_values("C")
        axes[1].plot(gc["C"], gc["val_loss"], "-", lw=1.3, color=c, alpha=0.9)
        axes[1].plot(gc["C"], gc["val_loss"], "o", ms=4.5, color=c,
                     markeredgecolor="0.15", markeredgewidth=0.5)
    for ax in axes:
        if irreducible is not None:
            ax.axhline(irreducible, ls="--", c="0.25", lw=1.3,
                       label=r"irreducible floor $E$")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylabel("validation loss")
    axes[0].set_xlabel("data  $D$  (examples seen)")
    axes[1].set_xlabel(r"compute  $C=(6N-2dw)D$  (FLOPs)")
    if irreducible is not None:
        axes[0].legend()
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
    ax.plot(env.frontier["C"], env.frontier["val_loss"], "o", c="crimson", ms=5,
            markeredgecolor="0.15", markeredgewidth=0.5, label=r"$L^\star(C)$  (envelope)")
    ax.plot(env.frontier["C"], env.frontier["val_loss"], "--", c="red", lw=1.8)
    if irreducible is not None:
        ax.axhline(irreducible, ls="--", c="0.25", lw=1.3, label=r"irreducible floor $E$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"compute  $C$  (FLOPs)"); ax.set_ylabel("loss"); ax.legend()

    C = env.frontier["C"].values
    axes[1].plot(C, env.frontier["N"], "o", c="navy")
    axes[1].plot(C, env.N_star(C), "--", c="red", lw=1.8, label="power-law fit")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel(r"compute  $C$  (FLOPs)"); axes[1].set_ylabel(r"$N^\star$"); axes[1].legend()

    axes[2].plot(C, env.frontier["D"], "o", c="seagreen")
    axes[2].plot(C, env.D_star(C), "--", c="red", lw=1.8, label="power-law fit")
    axes[2].set_xscale("log"); axes[2].set_yscale("log")
    axes[2].set_xlabel(r"compute  $C$  (FLOPs)"); axes[2].set_ylabel(r"$D^\star$"); axes[2].legend()
    fig.tight_layout()
    return fig


def plot_isoflop(iso: IsoFlopResult):
    """Approach 2: iso-compute parabolas and the recovered allocation."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ax = axes[0]
    Cs = sorted(iso.profiles)
    cnorm = mcolors.LogNorm(vmin=min(Cs), vmax=max(Cs))
    for C in Cs:
        prof = iso.profiles[C]
        col = cm.plasma(cnorm(C) * 0.9)
        ax.plot(prof["N"], prof["val_loss"], "o", color=col, ms=4.5,
                markeredgecolor="0.15", markeredgewidth=0.4)
        xs = np.log10(prof["N"].values)
        xg = np.linspace(xs.min(), xs.max(), 100)
        p2, p1, p0 = np.polyfit(xs, prof["val_loss"].values, 2)
        ax.plot(10 ** xg, p0 + p1 * xg + p2 * xg ** 2, "-", color=col, lw=1.2)
        row = iso.minima[iso.minima["C"] == C]
        if len(row):
            ax.plot(row["N_star"], row["loss_star"], "*", color=col, ms=15,
                    markeredgecolor="k")
    ax.plot([], [], "k*", ms=11, markeredgecolor="k", label=r"minimum  $N^\star(C)$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$N$  (params)"); ax.set_ylabel("loss")
    ax.legend()
    sm = cm.ScalarMappable(norm=cnorm, cmap=cm.plasma); sm.set_array([])
    fig.colorbar(sm, ax=ax, label=r"$C$  (FLOPs)")

    C = iso.minima["C"].values
    axes[1].plot(C, iso.minima["N_star"], "*", c="navy", ms=12)
    axes[1].plot(C, iso.N_star(C), "--", c="red", lw=1.8, label="power-law fit")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel(r"compute  $C$  (FLOPs)"); axes[1].set_ylabel(r"$N^\star$"); axes[1].legend()

    axes[2].plot(C, iso.minima["D_star"], "*", c="seagreen", ms=12)
    axes[2].plot(C, iso.D_star(C), "--", c="red", lw=1.8, label="power-law fit")
    axes[2].set_xscale("log"); axes[2].set_yscale("log")
    axes[2].set_xlabel(r"compute  $C$  (FLOPs)"); axes[2].set_ylabel(r"$D^\star$"); axes[2].legend()
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
    ax.plot(C, env.N_star(C), c="crimson", lw=1.8, label="envelope")
    ax.plot(C, iso.N_star(C), c="navy", ls="--", lw=1.8, label="iso-FLOP")
    ax.plot(C, par.N_star(C), c="seagreen", ls=":", lw=2.2, label="parametric")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"compute  $C$  (FLOPs)"); ax.set_ylabel(r"$N^\star(C)$")
    ax.legend()

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
    ax[0].plot(w, eW, "o", c="darkgreen", ms=6.5, markeredgecolor="0.15", markeredgewidth=0.5)
    ax[0].plot(w, np.exp(b_w) * w ** c_w, "--", c="red", lw=1.8, label="power-law fit")
    ax[0].set(xscale="log", yscale="log", xlabel=r"width  $w$", ylabel=r"optimal LR  $\eta^\star$")
    ax[0].legend()
    ax[1].plot(T, eT, "o", c="navy", ms=6.5, markeredgecolor="0.15", markeredgewidth=0.5)
    ax[1].plot(T, np.exp(b_T) * T ** c_T, "--", c="red", lw=1.8, label="power-law fit")
    ax[1].set(xscale="log", yscale="log", xlabel=r"steps  $T$", ylabel=r"optimal LR  $\eta^\star$")
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
    ax[0].plot(df.N, df.formula, "-", color="seagreen", label="closed form")
    ax[0].plot(df.N, df.naive, "--", color="0.55", label="naive")
    ax[0].set(xscale="log", yscale="log", xlabel="N (params)",
              ylabel="training FLOPs / example")
    ax[0].legend()
    ax[1].axhline(1, ls=":", c="k", lw=1)
    ax[1].plot(df.N, df.measured / df.formula, "o-", color="seagreen",
               label="measured / closed form")
    ax[1].plot(df.N, df.measured / df.naive, "s--", color="0.55", label="measured / naive")
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
        ax.plot(g.step, g.excess + s2, "-", color=c, label=f"$D$ = {D:,}")
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


def plot_frontier_regimes(env: EnvelopeResult, par: ParametricResult,
                          irreducible: float | None = None):
    """Anatomy of the compute-optimal frontier L*(C): three regimes, and why a single
    fit to L(C) is unreliable.

    On the lower-envelope L*(C) we overlay four fits, each in a fixed compute window
    (shaded), with dashed continuations past the last data point:
      * fast descent (orange, C<1e9) -- an E=0 power law. Here L >> E, so it reads the
        *true* (steep) reducible exponent.
      * scaling (blue, 1e9-1e13)     -- an E=0 power law whose *apparent* exponent is
        already shallower; the floor is starting to bite.
      * saturation tail (red, C>=5e12) -- a free-E fit E + A*C^g; only this near-floor
        window lets the free E land on the true irreducible loss.
      * Approach 3 (green) -- the parametric L(N,D) projected to the frontier. It tracks
        all three regimes here, but that is the easy case (clean task, known floor).

    Extrapolated (dashed), the E=0 laws shoot through the floor and the windowed
    exponents disagree ~3x -- the point of the figure.
    """
    from scipy.optimize import curve_fit
    fr = env.frontier.sort_values("C")
    C = fr["C"].values.astype(float); L = fr["val_loss"].values.astype(float)
    Cmax = float(C.max()); top = Cmax * 15
    geo = np.geomspace
    f1 = lambda x, E, A, g: E + A * np.power(x, g)

    tail = C >= 5e12                                    # free-E fit on the saturated tail
    (E1, A1, g1), _ = curve_fit(f1, C[tail] / 5e12, L[tail],
        p0=[L[tail].min(), L[tail][0] - L[tail].min(), -0.1],
        bounds=([0, 0, -2], [1, 1e3, 0]), maxfev=20000)
    f1v = lambda c: f1(c / 5e12, E1, A1, g1)

    def powerlaw(lo, hi):                               # E=0 power law fit in [lo, hi]
        m = (C >= lo) & (C <= hi)
        g, b = np.polyfit(np.log10(C[m]), np.log10(L[m]), 1)
        return (lambda c: 10 ** b * np.power(c, g)), g
    f2, g2 = powerlaw(1e9, 1e13)                        # scaling window
    f3, g3 = powerlaw(1e6, 1e9)                         # fast-descent window

    fig, ax = plt.subplots(figsize=(7.6, 5))
    ax.axvspan(1e6, 1e9, color="tab:orange", alpha=0.07)
    ax.axvspan(1e9, 1e13, color="tab:blue", alpha=0.07)
    ax.axvspan(5e12, Cmax, color="red", alpha=0.06)
    ax.plot(C, L, "o", color="0.3", ms=6.5, zorder=7,
            markeredgecolor="0.15", markeredgewidth=0.6, label=r"envelope  $L^\star(C)$")
    if irreducible is not None:
        ax.axhline(irreducible, ls="--", c="0.25", lw=1.3, label=r"irreducible floor $E$")
    ax.axvline(Cmax, color="0.6", lw=1, ls=(0, (1, 2)))

    ax.plot(geo(1e6, 1e9, 200), f3(geo(1e6, 1e9, 200)), "-", color="tab:orange", lw=2,
            label="power-law fit, fast descent")
    ax.plot(geo(1e9, top, 200), f3(geo(1e9, top, 200)), "--", color="tab:orange", lw=2)
    ax.plot(geo(1e9, 1e13, 200), f2(geo(1e9, 1e13, 200)), "-", color="tab:blue", lw=2,
            label="power-law fit, scaling regime")
    ax.plot(geo(1e6, 1e9, 200), f2(geo(1e6, 1e9, 200)), "--", color="tab:blue", lw=2)
    ax.plot(geo(1e13, top, 200), f2(geo(1e13, top, 200)), "--", color="tab:blue", lw=2)
    ax.plot(geo(5e12, Cmax, 200), f1v(geo(5e12, Cmax, 200)), "-", color="red", lw=2, zorder=5,
            label="power law + floor, saturation")
    ax.plot(geo(Cmax, top, 200), f1v(geo(Cmax, top, 200)), "--", color="red", lw=2, zorder=5)
    cp = geo(C.min(), top, 300)
    ax.plot(cp, par.L_star(cp), "--", color="tab:green", lw=1.8, alpha=0.55,
            label=r"$L(N,D)$ parametric fit (unreliable)")

    ax.set(xscale="log", yscale="log", xlabel=r"compute  $C$  (FLOPs)",
           ylabel="validation loss", xlim=(1e6, top)); ax.set_ylim(7e-3, 1.1)
    ax.legend(fontsize=9.5, loc="upper right")
    fig.tight_layout()
    return fig


def plot_lr_tuning(lrs, losses, eta_star):
    """L1 figure: validation loss vs learning rate at one cell, with the parabola fit
    (in log10(LR)) and its vertex eta*."""
    lrs = np.asarray(lrs, dtype=float)
    x = np.log10(lrs)
    a, b, c = np.polyfit(x, losses, 2)
    xg = np.linspace(x.min(), x.max(), 100)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(10 ** xg, a * xg ** 2 + b * xg + c, "-", color="0.6", lw=1.5, label="parabola fit")
    ax.plot(lrs, losses, "o", color="steelblue", ms=7,
            markeredgecolor="0.15", markeredgewidth=0.5, label="measured")
    ax.axvline(eta_star, ls="--", color="crimson", lw=1.5,
               label=r"optimum  $\eta^\star$")
    ax.set(xscale="log", xlabel=r"learning rate  $\eta$", ylabel="validation loss")
    ax.legend()
    fig.tight_layout()
    return fig
