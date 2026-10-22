"""
Training Rollout Re-export Module (Layer 2 — Training).

Re-exports the unified `rollout_latent` engine from `spwm.models.rollout`.
"""

from spwm.models.rollout import RolloutResult, TelemetryContainer, rollout_latent

__all__ = [
    "rollout_latent",
    "RolloutResult",
    "TelemetryContainer",
]
