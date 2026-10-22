"""Compatibility facade for `spwm.core` -> `spwm.models`."""

from spwm.models import (
    ALIFCell,
    ALIFState,
    CascadedTemporalRelay,
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
from spwm.models.predictors import (
    compute_convergence_funnel_target,
    extrapolate_latent,
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
    "CascadedTemporalRelay",
    "PhysicalDecoder",
    "PredictorOutput",
    "extrapolate_latent",
    "compute_convergence_funnel_target",
]
