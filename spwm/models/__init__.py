"""
SPWM neural models: neurons, memory, encoders, dynamics, and predictors.
"""

from spwm.models.surrogate import get_surrogate, FastSigmoidSurrogate, AtanSurrogate, SigmoidSurrogate
from spwm.models.neurons import LIFCell, NeuronState
from spwm.models.memory import MultiTimescaleMemory, MultiTimescaleState
from spwm.models.encoder import EventEncoder
from spwm.models.latent_dynamics import SpikingLatentDynamics, DynamicsState, DynamicsOutput
from spwm.models.predictor import LatentPredictor, PhysicalDecoder, PredictorOutput
from spwm.models.world_model import SPWM, SPWMState, SPWMStepOutput, SPWMSequenceOutput

__all__ = [
    "get_surrogate",
    "FastSigmoidSurrogate",
    "AtanSurrogate",
    "SigmoidSurrogate",
    "LIFCell",
    "NeuronState",
    "MultiTimescaleMemory",
    "MultiTimescaleState",
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
