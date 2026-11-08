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
from spwm.contracts.version import (
    VersionSolver,
    GLOBAL_VERSION,
    VERSION_SOLVER,
    GLOBAL_VERSION_SOLVER,
)
from spwm.contracts.tensors import (
    C_EVENT,
    J,
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
    PredictorProtocol,
    PhysicalDecoderProtocol,
    WorldModelProtocol,
    PhaseStrategy,
    PhaseHandler,
    Tier2EvaluatorProtocol,
)

__all__ = [
    "C_EVENT",
    "J",
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
    "PredictorProtocol",
    "PhysicalDecoderProtocol",
    "WorldModelProtocol",
    "PhaseStrategy",
    "PhaseHandler",
    "Tier2EvaluatorProtocol",
    "VersionSolver",
    "GLOBAL_VERSION",
    "VERSION_SOLVER",
    "GLOBAL_VERSION_SOLVER",
]
