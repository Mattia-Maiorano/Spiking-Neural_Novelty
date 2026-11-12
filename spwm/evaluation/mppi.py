"""Backward-compatible barrel re-exporting MPPI controller and scorer symbols."""

from __future__ import annotations

from spwm.evaluation.mppi_controller import (
    LatentMPPIController,
    MPPIPlanResult,
    NeuralCostMPPIController,
)
from spwm.evaluation.mppi_scorer import (
    MPPIStructuralScorer,
    compute_structural_score,
    resolve_effective_vel_threshold,
)
from spwm.models.kinematics import classify_elevation, compute_rot_error

__all__ = [
    "LatentMPPIController",
    "MPPIPlanResult",
    "MPPIStructuralScorer",
    "NeuralCostMPPIController",
    "classify_elevation",
    "compute_rot_error",
    "compute_structural_score",
    "resolve_effective_vel_threshold",
]
