"""
Re-export facade for spwm.contracts.config (Single Source of Truth).
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
    _strict_dict_to_dataclass,
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
    "_strict_dict_to_dataclass",
]
