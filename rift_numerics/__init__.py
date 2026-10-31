"""Public deterministic trajectory integration, containing no true world laws.

An acceleration callback is supplied by the caller. This package contains only
the published numerical method and input validation. Formal predictions use
the fixed-step ``rollout`` entry point; alternate steps are explicitly named
organizer convergence diagnostics.
"""
from .rollout import DT, NUMERICS_VERSION, IMPLEMENTATION_VERSION
from .rollout import rollout, state_rollout, diagnostic_rollout, sample_indices

__all__ = ["DT", "NUMERICS_VERSION", "IMPLEMENTATION_VERSION", "rollout", "state_rollout",
           "diagnostic_rollout", "sample_indices"]
