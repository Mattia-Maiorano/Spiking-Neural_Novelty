"""Compatibility facade for `spwm.learning.phases` -> `spwm.training.phases`."""

from spwm.training.phases import (
    M1PhaseStrategy,
    PhaseHandler,
    PhaseStrategy,
    PredictorPhaseHandler,
    RepresentationPhaseStrategy,
    build_phase_strategy,
)

__all__ = [
    "PhaseStrategy",
    "PhaseHandler",
    "RepresentationPhaseStrategy",
    "M1PhaseStrategy",
    "PredictorPhaseHandler",
    "build_phase_strategy",
]
