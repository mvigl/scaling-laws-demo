"""A miniature, self-contained demo of the three compute-optimal scaling-law
extraction methods (the Chinchilla recipe), applied to MLPs of
increasing size on a synthetic Gaussian teacher-student task.
"""
import warnings

# Suppress a lightning pytree deprecation warning here -- before any submodule imports
# lightning -- so it is silenced for every entry point (scripts and notebooks alike).
warnings.filterwarnings("ignore", ".*LeafSpec.*")

from . import flops, models, data, sweep, hp, live, approaches, plotting  # noqa: F401,E402

__all__ = ["flops", "models", "data", "sweep", "hp", "live", "approaches", "plotting"]
