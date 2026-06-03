"""Synthetic Gaussian teacher-student *regression* problem.

Inputs are standard Gaussian, x ~ N(0, I_d). The target is produced by a *fixed,
frozen* network -- the "teacher" -- plus i.i.d. label noise:

    y = f_teacher(x) + sigma * eps,    eps ~ N(0, 1).

The teacher is never trained; it only defines the ground-truth function we want to
learn (a fixed, reproducible, seeded network whose weights are saved to disk). The
student is the MLP we train, of growing size N.

Two knobs make a clean scaling law appear:
* the teacher is larger than at a fraction of the trained students, so smaller students have real
  approximation error -> the A/N^alpha term;
* data is drawn fresh every step (single pass) -> the B/D^beta term.
The label noise sets the KNOWN irreducible floor E = sigma^2 (the Bayes-optimal
predictor is f_teacher, leaving expected MSE = sigma^2). Targets are normalised to
unit variance so the loss scale is interpretable (init ~ 1 + sigma^2, floor ~ sigma^2).
"""
from __future__ import annotations

import math
import os

import torch
from torch.utils.data import IterableDataset, DataLoader

from .models import make_mlp


class StreamingTeacherDataset(IterableDataset):
    """Yields fresh (x, y) *batches* totalling ``n_data`` examples, single pass."""

    def __init__(self, problem: "TeacherStudentRegression", n_data: int,
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
            x = torch.randn(bs, self.problem.input_dim, generator=g)
            y = self.problem.clean_target(x) + self.problem.noise_std * torch.randn(
                bs, 1, generator=g)
            yield x, y


class TeacherStudentRegression:
    """A fixed Gaussian teacher (frozen network) and a held-out validation set."""

    def __init__(self, input_dim: int = 32, teacher_width: int = 256,
                 teacher_depth: int = 2, teacher_act: str = "gelu",
                 noise_std: float = 0.1, val_size: int = 16384, seed: int = 0,
                 normalize: bool = True, save_path: str | None = None):
        self.input_dim = input_dim
        self.noise_std = noise_std

        torch.manual_seed(seed)
        self.teacher = make_mlp(input_dim, teacher_width, teacher_depth, 1, teacher_act)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()
        if save_path:                                   # persist for reproducibility
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(self.teacher.state_dict(), save_path)

        # Normalise teacher output to zero mean / unit variance over a probe sample.
        g = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            out = self.teacher(torch.randn(65536, input_dim, generator=g))
        self.out_mean = out.mean().item() if normalize else 0.0
        self.out_scale = (1.0 / out.std().item()) if normalize else 1.0

        # Fixed validation inputs and their *clean* targets (noise-free), so the
        # measured loss = approximation error + sigma^2 is low-variance.
        gv = torch.Generator().manual_seed(seed + 2)
        self.x_val = torch.randn(val_size, input_dim, generator=gv)
        self.y_val_clean = self.clean_target(self.x_val)

    @torch.no_grad()
    def clean_target(self, x: torch.Tensor) -> torch.Tensor:
        return (self.teacher(x) - self.out_mean) * self.out_scale

    @property
    def irreducible_loss(self) -> float:
        """Known floor: best achievable MSE against noisy targets is E = sigma^2."""
        return self.noise_std ** 2

    @torch.no_grad()
    def evaluate(self, model) -> float:
        """Validation MSE: approximation error against the clean teacher, + sigma^2.

        Equals the expected test MSE against fresh noisy targets, but lower variance.
        """
        was_training = model.training
        model.eval()
        approx = torch.nn.functional.mse_loss(model(self.x_val), self.y_val_clean).item()
        if was_training:
            model.train()
        return approx + self.irreducible_loss

    def train_loader(self, n_data: int, batch_size: int, seed: int) -> DataLoader:
        ds = StreamingTeacherDataset(self, n_data, batch_size, seed)
        return DataLoader(ds, batch_size=None, num_workers=0)

    def n_steps(self, n_data: int, batch_size: int) -> int:
        return max(1, math.ceil(n_data / batch_size))
