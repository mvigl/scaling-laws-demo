"""The three ways to extract the compute-optimal frontier from a (N, D) grid.

All three estimate, at each compute budget C = 6ND, the loss-minimising
allocation (N*(C), D*(C)) and the frontier loss L*(C). They differ in cost and
in what they are sensitive to:

* approach 1 (``training_curve_envelope``): assumption-free, reads the lower
  envelope of the loss-vs-compute curves.
* approach 2 (``isoflop_profiles``): fits a parabola in log N along iso-compute
  slices and reads its minimum.
* approach 3 (``parametric_fit``): fits the full additive surface
  L = E + A/N^a + B/D^b and gets the frontier in closed form.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit y = k * x**m in log-log space. Returns (exponent m, prefactor k)."""
    lx, ly = np.log(x), np.log(y)
    m, b = np.polyfit(lx, ly, 1)
    return float(m), float(np.exp(b))


# --------------------------------------------------------------------------- #
# Approach 1: training-curve / fixed-N lower envelope
# --------------------------------------------------------------------------- #
@dataclass
class EnvelopeResult:
    frontier: pd.DataFrame              # the compute-optimal cells (C, loss, N, D)
    a_N: float                          # N*(C) ~ C**a_N
    a_D: float                          # D*(C) ~ C**a_D  (a_N + a_D should be ~1)
    k_N: float
    k_D: float

    def N_star(self, C):
        return self.k_N * np.power(C, self.a_N)

    def D_star(self, C):
        return self.k_D * np.power(C, self.a_D)


def training_curve_envelope(agg: pd.DataFrame, tol: float = 1e-9) -> EnvelopeResult:
    """Lower-envelope (Pareto) frontier of the loss-vs-compute point cloud.

    A cell sits on the frontier if no cheaper cell reached an equal-or-lower
    loss. Sweeping C upward, we keep every cell that sets a new record-low loss;
    these are exactly the compute-optimal (N, D) allocations.
    """
    d = agg.sort_values("C").reset_index(drop=True)
    keep, running = [], np.inf
    for _, row in d.iterrows():
        if row["val_loss"] < running - tol:
            keep.append(row)
            running = row["val_loss"]
    frontier = pd.DataFrame(keep).reset_index(drop=True)

    a_N, k_N = fit_power_law(frontier["C"].values, frontier["N"].values)
    a_D, k_D = fit_power_law(frontier["C"].values, frontier["D"].values)
    return EnvelopeResult(frontier, a_N, a_D, k_N, k_D)


# --------------------------------------------------------------------------- #
# Approach 2: IsoFLOP profiles
# --------------------------------------------------------------------------- #
@dataclass
class IsoFlopResult:
    profiles: dict                      # C -> DataFrame(N, D, loss) used in the fit
    minima: pd.DataFrame                # one (C, N*, D*, loss*) per slice
    a_N: float
    a_D: float
    k_N: float
    k_D: float

    def N_star(self, C):
        return self.k_N * np.power(C, self.a_N)

    def D_star(self, C):
        return self.k_D * np.power(C, self.a_D)


def _interp_loss_vs_N_at_C(agg: pd.DataFrame, C: float, min_points: int = 4):
    """For one compute budget C, interpolate each width's loss at D = C / k.

    k = C/D = 6N - 2*d*w is that width's FLOPs-per-example (constant within a width
    group), so the iso-compute data budget is D = C/k, not C/(6N)."""
    rows = []
    for N, g in agg.groupby("N"):
        g = g.sort_values("D")
        k = float((g["C"] / g["D"]).iloc[0])      # FLOPs/token for this width
        D_needed = C / k
        if D_needed < g["D"].min() or D_needed > g["D"].max():
            continue  # this width cannot reach C inside the swept D range
        # interpolate in log-loss vs log-D: loss ~ E + B/D^b is convex in log D,
        # so a linear interp of log-loss is far less biased than of raw loss.
        log_loss = np.interp(np.log(D_needed), np.log(g["D"].values),
                             np.log(g["val_loss"].values))
        rows.append((N, D_needed, float(np.exp(log_loss)), k))
    if len(rows) < min_points:
        return None
    return pd.DataFrame(rows, columns=["N", "D", "val_loss", "k"]).sort_values("N")


def _fit_slice(prof: pd.DataFrame, trim: float = 6.0, min_keep: int = 4):
    """Parabola in log10(N) over an iso-compute slice -> (kept_prof, N*, loss*).

    Before fitting, trim the data-starved (high N, tiny D -> under-trained) and
    capacity-starved (low N) tails that bow the parabola: keep only the well around
    the minimum (loss <= trim x slice-min). Returns None if not a clean interior min.
    The compute coordinates (C, D*) are assigned by the caller via the slice's k(N).
    """
    prof = prof.sort_values("N")
    y = prof["val_loss"].values
    keep = y <= y.min() * trim
    if keep.sum() >= min_keep:
        prof = prof.iloc[np.flatnonzero(keep)]
    x = np.log10(prof["N"].values)
    y = prof["val_loss"].values
    if len(x) < min_keep:
        return None
    p2, p1, p0 = np.polyfit(x, y, 2)
    if p2 <= 0:                                # not convex -> no interior optimum
        return None
    x_star = -p1 / (2 * p2)
    if not (x.min() <= x_star <= x.max()):     # vertex outside measured N = extrapolation
        return None
    return prof, float(10 ** x_star), float(p0 + p1 * x_star + p2 * x_star ** 2)


def _k_at(prof: pd.DataFrame, N_star: float) -> float:
    """Interpolate the slice's FLOPs/token k = C/D at the (interpolated) N*."""
    xk = np.log(prof["N"].values)
    return float(np.exp(np.interp(np.log(N_star), xk, np.log(prof["k"].values))))


def _isoflop_from_runs(agg: pd.DataFrame, min_points: int) -> IsoFlopResult:
    """Approach 2 from *dedicated* iso-FLOP runs (rows tagged with their budget iso_C).

    The runner trains each cell at D = C/k(N), k = 6N-2dw (run_isoflop.py), so a slice
    is a genuine iso-compute set at its tagged budget C."""
    profiles, minima = {}, []
    for C, prof in agg[agg["iso_C"] > 0].groupby("iso_C"):
        prof = prof.sort_values("N").copy()
        if len(prof) < min_points:
            continue
        prof["k"] = prof["C"] / prof["D"]              # 6N - 2dw per width
        fit = _fit_slice(prof)
        if fit is None:
            continue
        kept, N_star, loss_star = fit
        profiles[float(C)] = kept[["N", "D", "val_loss"]]
        minima.append((float(C), N_star, float(C) / _k_at(kept, N_star), loss_star))
    minima = pd.DataFrame(minima, columns=["C", "N_star", "D_star", "loss_star"])
    a_N, k_N = fit_power_law(minima["C"].values, minima["N_star"].values)
    a_D, k_D = fit_power_law(minima["C"].values, minima["D_star"].values)
    return IsoFlopResult(profiles, minima, a_N, a_D, k_N, k_D)


def isoflop_profiles(agg: pd.DataFrame, compute_targets=None, n_targets: int = 6,
                     min_points: int = 4) -> IsoFlopResult:
    """IsoFLOP profiles. If dedicated iso-FLOP runs are present (iso_C > 0) use them
    directly; otherwise reconstruct each iso-compute slice by interpolating the grid."""
    if "iso_C" in agg.columns and (agg["iso_C"] > 0).any():
        return _isoflop_from_runs(agg, min_points)
    if compute_targets is None:
        # A compute C is usable by a width if D = C/k (k = C/D = 6N-2dw for that
        # width) falls inside the swept D range. Keep only the C's that at least
        # ``min_points`` widths can bracket, then spread targets over that range.
        kpw = agg.groupby("N").apply(lambda g: (g["C"] / g["D"]).iloc[0])
        ks = kpw.values
        D_lo, D_hi = agg["D"].min(), agg["D"].max()

        def bracket_count(C):
            D = C / ks
            return int(np.sum((D >= D_lo) & (D <= D_hi)))

        cand = np.geomspace(agg["C"].min(), agg["C"].max(), 400)
        usable = cand[[bracket_count(c) >= min_points for c in cand]]
        if len(usable) == 0:
            raise ValueError("No compute budget is bracketed by >= min_points widths; "
                             "widen the D grid or lower min_points.")
        compute_targets = np.geomspace(usable.min(), usable.max(), n_targets)

    profiles, minima = {}, []
    for C in compute_targets:
        prof = _interp_loss_vs_N_at_C(agg, C, min_points)
        if prof is None:
            continue
        # Trim the data-/capacity-starved tails, then read the interior parabola min.
        fit = _fit_slice(prof)
        if fit is None:
            continue
        kept, N_star, loss_star = fit
        D_star = float(C) / _k_at(kept, N_star)       # iso-compute data at N*
        profiles[float(C)] = kept[["N", "D", "val_loss"]]
        minima.append((float(C), N_star, D_star, loss_star))

    minima = pd.DataFrame(minima, columns=["C", "N_star", "D_star", "loss_star"])
    a_N, k_N = fit_power_law(minima["C"].values, minima["N_star"].values)
    a_D, k_D = fit_power_law(minima["C"].values, minima["D_star"].values)
    return IsoFlopResult(profiles, minima, a_N, a_D, k_N, k_D)


# --------------------------------------------------------------------------- #
# Approach 3: parametric (additive "Chinchilla") surface fit
# --------------------------------------------------------------------------- #
@dataclass
class ParametricResult:
    E: float
    A: float
    B: float
    alpha: float
    beta: float
    a_N: float = field(init=False)      # N*(C) ~ C**a_N, a_N = beta/(alpha+beta)
    a_D: float = field(init=False)
    # optional cost model C = k(N) * D; if None, fall back to the 6ND closed form.
    _cost = None                        # (k_of_N callable, (N_min, N_max))

    def __post_init__(self):
        s = self.alpha + self.beta
        self.a_N = self.beta / s
        self.a_D = self.alpha / s

    def loss(self, N, D):
        return self.E + self.A * np.power(N, -self.alpha) + self.B * np.power(D, -self.beta)

    def N_star(self, C):
        # The exponent a_N = beta/(alpha+beta) is convention-free; only the frontier's
        # placement depends on C = k(N)*D. With a cost model, minimise L(N, C/k(N))
        # numerically over the measured N range; else use the C=6ND closed form.
        if self._cost is None:
            coef = (self.alpha * self.A) / (self.beta * self.B)
            return np.power(coef, 1.0 / (self.alpha + self.beta)) * np.power(C / 6.0, self.a_N)
        k_of_N, (N_lo, N_hi) = self._cost
        grid = np.geomspace(N_lo, N_hi, 600)
        kg = k_of_N(grid)
        scalar = np.ndim(C) == 0
        out = [grid[np.argmin(self.loss(grid, float(c) / kg))] for c in np.atleast_1d(C)]
        return float(out[0]) if scalar else np.array(out)

    def D_star(self, C):
        Ns = self.N_star(C)
        if self._cost is None:
            return C / (6.0 * Ns)
        return np.asarray(C, float) / self._cost[0](Ns)

    def L_star(self, C):
        return self.loss(self.N_star(C), self.D_star(C))


def _cost_interp(agg: pd.DataFrame):
    """Build k(N) = C/D (FLOPs per example) as a log-log interpolator over the grid."""
    kpw = agg.groupby("N").apply(lambda g: (g["C"] / g["D"]).iloc[0])
    lnN, lnk = np.log(kpw.index.values.astype(float)), np.log(kpw.values.astype(float))

    def k_of_N(x):
        return np.exp(np.interp(np.log(np.asarray(x, float)), lnN, lnk))

    return k_of_N, (float(kpw.index.min()), float(kpw.index.max()))


def parametric_fit(agg: pd.DataFrame, huber_delta: float = 1e-3) -> ParametricResult:
    """Fit L = E + A N^-alpha + B D^-beta by Huber loss on log-loss residuals.

    Parameters are carried in log space (Chinchilla LSE trick) so E, A, B stay
    positive: theta = [logE, logA, alpha, logB, beta]. The frontier projection uses
    the per-width cost k(N)=C/D so it shares the same compute axis as approaches 1-2.
    """
    N = agg["N"].values.astype(float)
    D = agg["D"].values.astype(float)
    L = agg["val_loss"].values.astype(float)
    lnN, lnD, lnL = np.log(N), np.log(D), np.log(L)

    def predict(theta):
        logE, logA, alpha, logB, beta = theta
        terms = np.logaddexp.reduce(np.stack([
            np.full_like(lnN, logE),
            logA - alpha * lnN,
            logB - beta * lnD,
        ]), axis=0)
        return terms  # = log(E + A N^-a + B D^-b)

    def resid(theta):
        return predict(theta) - lnL

    L_min = float(L.min())
    theta0 = np.array([np.log(0.8 * L_min), np.log(L.max()), 0.3,
                       np.log(L.max()), 0.3])
    sol = least_squares(resid, theta0, loss="huber", f_scale=huber_delta,
                        max_nfev=20000,
                        bounds=([-20, -20, 0.01, -20, 0.01],
                                [5, 20, 3.0, 20, 3.0]))
    logE, logA, alpha, logB, beta = sol.x
    res = ParametricResult(E=float(np.exp(logE)), A=float(np.exp(logA)),
                           B=float(np.exp(logB)), alpha=float(alpha),
                           beta=float(beta))
    res._cost = _cost_interp(agg)
    return res
