"""Compatibility facade for `spwm.learning.phases` -> `spwm.training.phases`."""

from spwm.training.phases import (
    M1PhaseStrategy,
    M2PhaseStrategy,
    M3PhaseStrategy,
    PhaseHandler,
    PhaseStrategy,
    PredictorPhaseHandler,
    build_phase_strategy,
)

__all__ = [
    "PhaseStrategy",
    "PhaseHandler",
    "M1PhaseStrategy",
    "M2PhaseStrategy",
    "M3PhaseStrategy",
    "PredictorPhaseHandler",
    "build_phase_strategy",
]
