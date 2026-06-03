"""The student model: a plain MLP wrapped as a ``LightningModule``.

The training setup is standard practice at a miniature scale: AdamW, a linear warmup over 8% of
steps from eta/100 -> eta, then cosine annealing eta -> eta/10. Every grid cell is a
*single-pass* run annealed to its own data budget D -- the compute-bottlenecked regime.
The loss is MSE (see data.py for why regression, not classification).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

_ACTS = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU}


def make_mlp(input_dim: int, width: int, n_hidden: int, output_dim: int = 1,
             act: str = "gelu") -> nn.Sequential:
    """Build an MLP: input_dim -> (width, act) x n_hidden -> output_dim."""
    if n_hidden < 1:
        raise ValueError("n_hidden must be >= 1")
    Act = _ACTS[act]
    layers: list[nn.Module] = [nn.Linear(input_dim, width), Act()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(width, width), Act()]
    layers += [nn.Linear(width, output_dim)]
    return nn.Sequential(*layers)


def cosine_lr_lambda(total_steps: int, warmup_frac: float = 0.08,
                     start_factor: float = 0.01, end_factor: float = 0.1):
    """LR multiplier: linear warmup eta/100 -> eta over ``warmup_frac`` of the steps,
    then cosine annealing eta -> eta/10. The per-cell cosine schedule used throughout."""
    warmup_steps = max(1, int(warmup_frac * total_steps))

    def fn(step: int) -> float:
        if step < warmup_steps:
            return start_factor + (1.0 - start_factor) * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return end_factor + (1.0 - end_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


class LitMLP(L.LightningModule):
    """An MLP student trained single-pass with MSE under the cosine schedule.

    Validation is computed externally by the problem (MSE against the known clean
    teacher), so there is no ``validation_step`` here.
    """

    def __init__(self, input_dim: int, width: int, n_hidden: int, output_dim: int = 1,
                 lr: float = 1e-3, weight_decay: float = 0.0, total_steps: int = 1000,
                 act: str = "gelu"):
        super().__init__()
        self.save_hyperparameters()
        self.net = make_mlp(input_dim, width, n_hidden, output_dim, act)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.mse_loss(self(x), y)
        self.log("train_loss", loss, on_step=True, on_epoch=False, batch_size=x.shape[0])
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, cosine_lr_lambda(self.hparams.total_steps))
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
