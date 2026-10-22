"""Compatibility facade for `spwm.learning.phases.predictor_phase` -> `spwm.training.phases`."""

from spwm.training.phases import (
    M1PhaseStrategy,
    M2PhaseStrategy,
    M3PhaseStrategy,
    PredictorPhaseHandler,
    build_phase_strategy,
)

__all__ = [
    "M1PhaseStrategy",
    "M2PhaseStrategy",
    "M3PhaseStrategy",
    "PredictorPhaseHandler",
    "build_phase_strategy",
]
