"""Conditional flow matching: Gaussian sources transported to moon-shaped targets.

The generative analogue of the teacher-student regression. A velocity field
v_theta(x, t): R^dim x [0,1] -> R^dim is trained to transport the source
p0 = N(0, I_dim) to a target p1 along the linear interpolation path

    x_t = (1 - t) x0 + t x1,      x0 ~ p0,  x1 ~ p1,  t ~ U[0, 1],

by regressing the conditional velocity v = x1 - x0 (Lipman et al. 2023, "flow
matching"; the linear path is the rectified-flow choice). The training objective is
plain MSE, so the whole scaling pipeline applies unchanged:

    L(theta) = E || v_theta(x_t, t) - (x1 - x0) ||^2 / dim  =  E_floor + excess,

where the irreducible floor E_floor = E[tr Var(x1 - x0 | x_t, t)] / dim is the
variance of the conditional target around the marginal velocity field (many
(x0, x1) pairs pass through the same (x_t, t); the optimal net predicts their
mean). The role of the "teacher" is played by the data-generating process itself,
and in contrast to the synthetic regression the floor is NOT known in closed form:
it must be estimated (:meth:`FlowMatchingProblem.estimate_floor`) or fitted.

Two problems are provided:

* :class:`FlowMatchingProblem`: 2D, N(0, I_2) -> two moons (notebook 05).
* :class:`MultiMoonsFlow`: 2k dims, k independent moon-pairs stacked and mixed by a
  fixed random rotation, so the joint target is entangled across all coordinates
  (notebook 06). Generated samples are un-rotated for plane-by-plane inspection.
* :class:`HierarchicalMoonsFlow`: entangled planes of *two-scale* moons: every coarse
  point carries a miniature moon-shaped offset, so fine structure ~15x smaller than
  the arcs must also be resolved (notebook 07).

Data is streamed fresh every batch (single pass), so D = examples seen, exactly as
in the regression tutorial; the model input is (x_t, t) concatenated
(input_dim = dim + 1) and the output is the dim-dimensional velocity.
"""
from __future__ import annotations

import math

import torch
from torch.utils.data import IterableDataset, DataLoader

# Fixed affine that centres the raw sklearn-style moons (x in [-1, 2], y in [-1, 1.5])
# and scales them to roughly unit std, so source and target overlap.
_MOON_CENTER = (0.5, 0.25)
_MOON_SCALE = 1.5


def two_moons(n: int, noise: float = 0.05, generator: torch.Generator | None = None):
    """Sample n points from the two-moons distribution (centred, scaled, noisy)."""
    n_up = n // 2
    theta = torch.rand(n, generator=generator) * math.pi
    x = torch.empty(n, 2)
    x[:n_up, 0] = torch.cos(theta[:n_up])
    x[:n_up, 1] = torch.sin(theta[:n_up])
    x[n_up:, 0] = 1.0 - torch.cos(theta[n_up:])
    x[n_up:, 1] = 0.5 - torch.sin(theta[n_up:])
    x += noise * torch.randn(n, 2, generator=generator)
    x[:, 0] -= _MOON_CENTER[0]
    x[:, 1] -= _MOON_CENTER[1]
    return x * _MOON_SCALE


class StreamingFlowDataset(IterableDataset):
    """Yields fresh CFM batches totalling ``n_data`` examples, single pass."""

    def __init__(self, problem: "FlowMatchingProblem", n_data: int,
                 batch_size: int, seed: int):
        self.problem = problem
        self.n_data = n_data
        self.batch_size = batch_size
        self.seed = seed

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        remaining = self.n_data
        while remaining > 0:
            bs = min(self.batch_size, remaining)
            remaining -= bs
            yield self.problem.cfm_batch(bs, g)


class FlowMatchingProblem:
    """Gaussian -> two-moons flow matching with the tutorial's problem interface.

    Mirrors ``TeacherStudentRegression`` (train_loader / evaluate / n_steps /
    input_dim), so ``sweep.run_cell``, ``hp.tune_lr_cell`` and the live helpers work
    on it unchanged. ``evaluate`` returns the CFM loss on a big fixed validation set
    of (t, x0, x1) triplets: floor + excess, with the floor *unknown* a priori.
    Subclasses change the target by overriding ``dim`` and ``sample_target``.
    """

    dim = 2

    def __init__(self, moons_noise: float = 0.05, val_size: int = 65536,
                 seed: int = 0, floor: float | None = None):
        self.input_dim = self.dim + 1     # (x_t, t)
        self.output_dim = self.dim        # velocity
        self.moons_noise = moons_noise
        self.seed = seed
        self.irreducible_loss = floor     # settable; estimate with estimate_floor()

        gv = torch.Generator().manual_seed(seed + 2)
        self.x_val, self.v_val = self.cfm_batch(val_size, gv)

    def sample_target(self, n: int, generator: torch.Generator | None = None):
        return two_moons(n, self.moons_noise, generator)

    def cfm_batch(self, n: int, generator: torch.Generator | None = None):
        """One conditional-flow-matching batch: (x_t, t) features, target v = x1 - x0."""
        x0 = torch.randn(n, self.dim, generator=generator)
        x1 = self.sample_target(n, generator)
        t = torch.rand(n, 1, generator=generator)
        xt = (1.0 - t) * x0 + t * x1
        return torch.cat([xt, t], dim=1), x1 - x0

    @torch.no_grad()
    def evaluate(self, model) -> float:
        """Validation CFM loss (mean squared error over the velocity components)."""
        was_training = model.training
        model.eval()
        loss = torch.nn.functional.mse_loss(model(self.x_val), self.v_val).item()
        if was_training:
            model.train()
        return loss

    def train_loader(self, n_data: int, batch_size: int, seed: int) -> DataLoader:
        ds = StreamingFlowDataset(self, n_data, batch_size, seed)
        return DataLoader(ds, batch_size=None, num_workers=0)

    def n_steps(self, n_data: int, batch_size: int) -> int:
        return max(1, math.ceil(n_data / batch_size))

    @torch.no_grad()
    def estimate_floor(self, n_points: int = 4096, pool: int = 1 << 17,
                       t_max: float = 0.9, seed: int = 123):
        """Importance-sampling estimate of the floor E[tr Var(x1 - x0 | x_t, t)] / dim.

        For a point (t, x_t), the pairs passing through it satisfy
        x0 = (x_t - t x1) / (1 - t), so conditioning on (x_t, t) is an integral over
        x1 weighted by p0((x_t - t x1)/(1 - t)); with a large target pool this is a
        self-normalised importance-sampling average, and v = (x1 - x_t)/(1 - t) is a
        deterministic function of x1. The estimator degenerates as t -> 1 (the pool
        carries too few targets within ~(1-t) of x_t), and in higher dimension the
        weights concentrate faster, so the average is restricted to t <= t_max and
        reported with the mean effective sample size (compare against the converged
        loss of a large reference model).

        Returns ``(floor_estimate, mean_effective_sample_size)``.
        """
        g = torch.Generator().manual_seed(seed)
        targets = self.sample_target(pool, g)                           # (P, dim)
        x0 = torch.randn(n_points, self.dim, generator=g)
        x1 = self.sample_target(n_points, g)
        t = torch.rand(n_points, 1, generator=g) * t_max
        xt = (1.0 - t) * x0 + t * x1

        chunk = max(8, int(2 ** 24 / (pool * self.dim)))                # memory cap
        tot, ess_tot = 0.0, 0.0
        for i in range(0, n_points, chunk):
            tc, xc = t[i:i + chunk], xt[i:i + chunk]                    # (c,1), (c,dim)
            x0_imp = (xc[:, None, :] - tc[:, None, :] * targets[None, :, :]) / (1.0 - tc[:, None, :])
            logw = -0.5 * (x0_imp ** 2).sum(-1)                         # (c, P)
            w = torch.softmax(logw, dim=1)
            v = (targets[None, :, :] - xc[:, None, :]) / (1.0 - tc[:, None, :])
            m = (w[..., None] * v).sum(1, keepdim=True)                 # (c, 1, dim)
            var = (w * ((v - m) ** 2).sum(-1)).sum(1)                   # (c,)
            tot += var.sum().item()
            ess_tot += (1.0 / (w ** 2).sum(1)).sum().item()
        # tr Var sums the components; the MSE loss averages over them
        return tot / n_points / self.dim, ess_tot / n_points

    @torch.no_grad()
    def sample(self, model, n: int = 2048, steps: int = 100, seed: int = 0,
               return_traj: bool = False):
        """Generate samples by integrating dx/dt = v_theta(x, t) from t=0 to 1
        (midpoint rule), starting from x ~ N(0, I). Optionally return the whole
        trajectory (steps+1, n, dim)."""
        was_training = model.training
        model.eval()
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(n, self.dim, generator=g)
        dt = 1.0 / steps
        traj = [x.clone()] if return_traj else None
        for k in range(steps):
            t0 = torch.full((n, 1), k * dt)
            v0 = model(torch.cat([x, t0], dim=1))
            xm = x + 0.5 * dt * v0
            vm = model(torch.cat([xm, t0 + 0.5 * dt], dim=1))
            x = x + dt * vm
            if return_traj:
                traj.append(x.clone())
        if was_training:
            model.train()
        return (x, torch.stack(traj)) if return_traj else x


class MultiMoonsFlow(FlowMatchingProblem):
    """A harder target: k independent moon-pairs stacked into 2k dimensions and mixed
    by a fixed random rotation.

    The rotation entangles every coordinate with every plane, so the velocity field
    cannot decompose over 2D subspaces; capacity demands grow accordingly and the
    scaling regime extends to larger models than the 2D problem. For inspection,
    apply :attr:`rotation`'s transpose (``unrotate``) to generated samples and plot
    them plane by plane.
    """

    def __init__(self, n_pairs: int = 4, moons_noise: float = 0.05,
                 val_size: int = 65536, seed: int = 0, floor: float | None = None):
        self.n_pairs = n_pairs
        self.dim = 2 * n_pairs            # instance attribute overrides the class default
        # a fixed, seeded random rotation of R^{2k}
        g = torch.Generator().manual_seed(seed + 41)
        q, r = torch.linalg.qr(torch.randn(2 * n_pairs, 2 * n_pairs, generator=g))
        self.rotation = q * torch.sign(torch.diagonal(r))
        super().__init__(moons_noise=moons_noise, val_size=val_size, seed=seed,
                         floor=floor)

    def sample_plane(self, n: int, generator: torch.Generator | None = None):
        return two_moons(n, self.moons_noise, generator)

    def sample_target(self, n: int, generator: torch.Generator | None = None):
        planes = [self.sample_plane(n, generator) for _ in range(self.n_pairs)]
        return torch.cat(planes, dim=1) @ self.rotation.T

    def unrotate(self, x: torch.Tensor) -> torch.Tensor:
        """Map samples back to the axis-aligned moon planes (columns 2j, 2j+1)."""
        return x @ self.rotation


class HierarchicalMoonsFlow(MultiMoonsFlow):
    """The hardest target: entangled planes of *two-scale* moons.

    Each plane is a sharp two-moons convolved with a miniature two-moons: every
    coarse point carries a small moon-shaped offset, so each arc's cross-section
    resolves, at scale ``sub_scale``, into two fine arcs. The field must model the
    coarse geometry *and* the substructure (features ~15x smaller), on top of the
    rotation entangling all planes. Small models produce thick fuzzy moons; the fine
    arcs only appear at large N and D, so the scaling regime stretches much further
    than :class:`MultiMoonsFlow`. Inspect with :meth:`unrotate` and zoomed panels.
    """

    def __init__(self, n_pairs: int = 4, moons_noise: float = 0.02,
                 sub_scale: float = 0.18, sub_noise: float = 0.05,
                 val_size: int = 65536, seed: int = 0, floor: float | None = None):
        self.sub_scale = sub_scale
        self.sub_noise = sub_noise
        super().__init__(n_pairs=n_pairs, moons_noise=moons_noise,
                         val_size=val_size, seed=seed, floor=floor)

    def sample_plane(self, n: int, generator: torch.Generator | None = None):
        return (two_moons(n, self.moons_noise, generator)
                + self.sub_scale * two_moons(n, self.sub_noise, generator))
