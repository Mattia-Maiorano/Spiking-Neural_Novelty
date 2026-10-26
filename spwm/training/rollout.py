"""
Training Rollout Re-export Module (Layer 2 — Training).

Re-exports the unified `rollout_latent` engine from `spwm.models.rollout`.
"""

from spwm.models.rollout import (
    RolloutResult,
    TelemetryContainer,
    compute_rollout_schedule,
    ground_initial_state,
    rollout_latent,
    step_grounded,
)

__all__ = [
    "compute_rollout_schedule",
    "ground_initial_state",
    "rollout_latent",
    "step_grounded",
    "RolloutResult",
    "TelemetryContainer",
]
