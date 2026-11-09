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
from spwm.models.ann_ablation import (
    AblationRolloutResult,
    ANNWorldModel,
    ContinuousRateDynamics,
    ContinuousRateMemory,
    ContinuousRateState,
)
from spwm.models.dynamics import SpikingLatentDynamics
from spwm.models.encoder import EventEncoder, SpatialSoftmax
from spwm.models.kinematics import (
    CANONICAL_ANCHORS,
    CANONICAL_JOINT_AXES,
    DifferentiableForwardKinematics,
    END_EFFECTOR_TIP_OFFSET,
)
from spwm.models.memory import MultiTimescaleMemory
from spwm.models.neurons import ALIFCell, SurrogateSpike, surrogate_spike
from spwm.models.predictors import (
    LatentCritic,
    LatentPredictor,
    PhysicalDecoder,
)
from spwm.models.rollout import (
    compute_rollout_schedule,
    ground_initial_state,
    rollout_latent,
    step_grounded,
)
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
    "LatentCritic",
    "LatentPredictor",
    "PhysicalDecoder",
    "PredictorOutput",
    "DifferentiableForwardKinematics",
    "CANONICAL_ANCHORS",
    "CANONICAL_JOINT_AXES",
    "END_EFFECTOR_TIP_OFFSET",
    "SPWM",
    "ANNWorldModel",
    "ContinuousRateMemory",
    "ContinuousRateDynamics",
    "ContinuousRateState",
    "AblationRolloutResult",
    "SPWMState",
    "SPWMStepOutput",
    "SPWMSequenceOutput",
    "rollout_latent",
    "RolloutResult",
    "TelemetryContainer",
]
