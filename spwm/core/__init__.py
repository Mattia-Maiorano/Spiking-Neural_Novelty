"""Compatibility facade for `spwm.core` -> `spwm.models`."""

from spwm.models import (
    ALIFCell,
    ALIFState,
    DynamicsOutput,
    DynamicsState,
    EventEncoder,
    LatentPredictor,
    MultiTimescaleMemory,
    MultiTimescaleState,
    PhysicalDecoder,
    PredictorOutput,
    SpatialSoftmax,
    SpikingLatentDynamics,
    SurrogateSpike,
    surrogate_spike,
)

__all__ = [
    "ALIFCell",
    "ALIFState",
    "SurrogateSpike",
    "surrogate_spike",
    "MultiTimescaleMemory",
    "MultiTimescaleState",
    "EventEncoder",
    "SpatialSoftmax",
    "SpikingLatentDynamics",
    "DynamicsState",
    "DynamicsOutput",
    "LatentPredictor",
    "PhysicalDecoder",
    "PredictorOutput",
]
