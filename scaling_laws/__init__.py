"""A miniature, self-contained demo of the three compute-optimal scaling-law
extraction methods (the Chinchilla recipe), applied to MLPs of
increasing size on a synthetic Gaussian teacher-student task.
"""
from . import flops, models, data, sweep, approaches, plotting  # noqa: F401

__all__ = ["flops", "models", "data", "sweep", "approaches", "plotting"]
