"""
SPWM Neural Model Components (Layer 1 — Models).

Pure neural network modules and forward/step state transitions.
Depends exclusively on Layer 0 (`spwm.contracts`).
"""

from spwm.contracts.states import (
    ALIFState,
    DynamicsOutput,
    DynamicsState,
    MultiTimescaleState,
    PredictorOutput,
    RolloutResult,
    SPWMSequenceOutput,
    SPWMState,
    SPWMStepOutput,
    TelemetryContainer,
)
from spwm.models.dynamics import SpikingLatentDynamics
from spwm.models.encoder import EventEncoder, SpatialSoftmax
from spwm.models.memory import MultiTimescaleMemory
from spwm.models.neurons import ALIFCell, SurrogateSpike, surrogate_spike
from spwm.models.predictors import (
    CascadedTemporalRelay,
    LatentPredictor,
    PhysicalDecoder,
)
from spwm.models.rollout import rollout_latent
from spwm.models.world_model import SPWM

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
    "SPWM",
    "SPWMState",
    "SPWMStepOutput",
    "SPWMSequenceOutput",
    "rollout_latent",
    "RolloutResult",
    "TelemetryContainer",
]
