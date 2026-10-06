"""Parameter counting and the training-compute relation.

We follow the standard 6ND compute convention but keep the first-layer correction that
the rule of thumb drops. Per parameter and per example, a training step costs

    2 FLOPs  forward                (one multiply-add),
  + 2 FLOPs  backward, weight grad  (dL/dW),
  + 2 FLOPs  backward, input grad   (dL/dx, to keep back-propagating),
  ----------
    6 FLOPs  -> the familiar C = 6 N D.

The first layer does not need its input gradient, since nothing is back-propagated
into the raw data. This saves 2 FLOPs per example for every weight of the first layer
(the d_in x width matrix, input_dim * width weights):

    C = (6 N - 2 * input_dim * width) * D .

For small, narrow networks the first weight matrix is a large fraction of N, so the
correction is sizable (~25% at width 8, d=32) and shifts the low-N points of every fit;
it becomes negligible as the hidden width grows. For these MLPs D is the number of
training examples seen (single pass), the analogue of tokens in LLM scaling.
"""
from __future__ import annotations


def mlp_param_count(input_dim: int, width: int, n_hidden: int, output_dim: int = 1) -> int:
    """Exact parameter count of an MLP with ``n_hidden`` hidden layers of ``width``.

    Architecture: Linear(input_dim, width) -> [Linear(width, width)] * (n_hidden-1)
    -> Linear(width, output_dim), every Linear carrying a bias.
    """
    if n_hidden < 1:
        raise ValueError("n_hidden must be >= 1")
    first = input_dim * width + width
    middle = (n_hidden - 1) * (width * width + width)
    last = width * output_dim + output_dim
    return first + middle + last


def flops_per_token(n_params: int, input_dim: int, width: int) -> float:
    """Training FLOPs per example: forward (2N) + backward (4N) minus the first
    layer's input-gradient, which is never computed -> 6N - 2 * input_dim * width.

    (input_dim * width is the first weight matrix; its bias carries no input
    gradient either, but biases are already sub-leading and we ignore them.)
    """
    return 6.0 * float(n_params) - 2.0 * float(input_dim) * float(width)


def compute(n_params: int, n_data: int, input_dim: int, width: int) -> float:
    """Training compute in FLOPs, C = (6N - 2*d*w) * D (the 6ND convention + first-layer term)."""
    return flops_per_token(n_params, input_dim, width) * float(n_data)
