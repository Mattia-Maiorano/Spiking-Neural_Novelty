"""Compatibility facade for `spwm.core.neurons` -> `spwm.models.neurons`."""

from spwm.models.neurons import (
    ALIFCell,
    ALIFState,
    AtanSurrogate,
    FastSigmoidSurrogate,
    SigmoidSurrogate,
    SurrogateSpike,
    get_surrogate,
    surrogate_derivative,
    surrogate_spike,
)

__all__ = [
    "ALIFCell",
    "ALIFState",
    "AtanSurrogate",
    "FastSigmoidSurrogate",
    "SigmoidSurrogate",
    "SurrogateSpike",
    "get_surrogate",
    "surrogate_derivative",
    "surrogate_spike",
]
