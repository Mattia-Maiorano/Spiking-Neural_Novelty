"""
Layer 0 (`spwm.contracts`): Immutable Schemas, State Containers, Tensor Contracts, and Protocols.
Zero upward dependencies on models, training, or evaluation.
"""

from spwm.contracts.config import (
    ProjectConfig,
    EnvironmentConfig,
    DataConfig,
    ModelConfig,
    LatentNeuronConfig,
    MemoryConfig,
    TrainingConfig,
    LossConfig,
    EvaluationConfig,
    ExperimentConfig,
)
from spwm.contracts.tensors import (
    assert_tensor_contract,
    assert_phase_space_contract,
    assert_euclidean_coordinates,
)
from spwm.contracts.states import (
    ALIFState,
    MultiTimescaleState,
    DynamicsState,
    DynamicsOutput,
    PredictorOutput,
    SPWMState,
    SPWMStepOutput,
    SPWMSequenceOutput,
    TelemetryContainer,
    RolloutResult,
)
from spwm.contracts.protocols import (
    EncoderProtocol,
    DynamicsProtocol,
    PredictorModuleProtocol,
    RelayPredictorProtocol,
    PhysicalDecoderProtocol,
    WorldModelProtocol,
    PhaseStrategy,
    PhaseHandler,
)

__all__ = [
    "ProjectConfig",
    "EnvironmentConfig",
    "DataConfig",
    "ModelConfig",
    "LatentNeuronConfig",
    "MemoryConfig",
    "TrainingConfig",
    "LossConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "assert_tensor_contract",
    "assert_phase_space_contract",
    "assert_euclidean_coordinates",
    "ALIFState",
    "MultiTimescaleState",
    "DynamicsState",
    "DynamicsOutput",
    "PredictorOutput",
    "SPWMState",
    "SPWMStepOutput",
    "SPWMSequenceOutput",
    "TelemetryContainer",
    "RolloutResult",
    "EncoderProtocol",
    "DynamicsProtocol",
    "PredictorModuleProtocol",
    "RelayPredictorProtocol",
    "PhysicalDecoderProtocol",
    "WorldModelProtocol",
    "PhaseStrategy",
    "PhaseHandler",
]
