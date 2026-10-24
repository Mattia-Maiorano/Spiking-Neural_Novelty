"""Compatibility facade for `spwm.learning.phases.predictor_phase` -> `spwm.training.phases`."""

from spwm.training.phases import (
    M1PhaseStrategy,
    PredictorPhaseHandler,
    RepresentationPhaseStrategy,
    build_phase_strategy,
)

__all__ = [
    "RepresentationPhaseStrategy",
    "M1PhaseStrategy",
    "PredictorPhaseHandler",
    "build_phase_strategy",
]
