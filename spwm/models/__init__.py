"""
SPWM neural models: neurons, memory, encoders, dynamics, and predictors.
"""

from spwm.models.surrogate import (
    get_surrogate,
    surrogate_derivative,
    FastSigmoidSurrogate,
    AtanSurrogate,
    SigmoidSurrogate,
    SurrogateSpike,
)
from spwm.models.neurons import LIFCell, NeuronState, ALIFCell, ALIFState
from spwm.models.memory import MultiTimescaleMemory, MultiTimescaleState, compute_tier_dims
from spwm.models.encoder import EventEncoder
from spwm.models.latent_dynamics import SpikingLatentDynamics, DynamicsState, DynamicsOutput
from spwm.models.predictor import LatentPredictor, PhysicalDecoder, PredictorOutput
from spwm.models.world_model import SPWM, SPWMState, SPWMStepOutput, SPWMSequenceOutput

__all__ = [
    "get_surrogate",
    "surrogate_derivative",
    "FastSigmoidSurrogate",
    "AtanSurrogate",
    "SigmoidSurrogate",
    "SurrogateSpike",
    "LIFCell",
    "NeuronState",
    "ALIFCell",
    "ALIFState",
    "MultiTimescaleMemory",
    "MultiTimescaleState",
    "compute_tier_dims",
    "EventEncoder",
    "SpikingLatentDynamics",
    "DynamicsState",
    "DynamicsOutput",
    "LatentPredictor",
    "PhysicalDecoder",
    "PredictorOutput",
    "SPWM",
    "SPWMState",
    "SPWMStepOutput",
    "SPWMSequenceOutput",
]
