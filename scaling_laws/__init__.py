"""Three ways to extract a compute-optimal scaling law (Hoffmann et al., 2022), applied
to MLPs of increasing size on a synthetic Gaussian teacher-student task.

Companion code to "How to scale your HEP ML models: A recipe for robust architecture
comparisons at scale" (Vigl et al., 2026); see the README for the citation.
"""
import warnings

# Silence a Lightning pytree deprecation warning before any submodule imports Lightning,
# so that it is suppressed in the scripts and the notebooks alike.
warnings.filterwarnings("ignore", ".*LeafSpec.*")

from . import flops, models, data, sweep, hp, live, approaches, plotting  # noqa: F401,E402

__all__ = ["flops", "models", "data", "sweep", "hp", "live", "approaches", "plotting"]
