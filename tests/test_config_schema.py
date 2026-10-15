"""
Unit tests for typed ExperimentConfig schema.
Verifies strict validation, error-raising on misaligned parameters, and absence of silent fallbacks.
"""

import pytest
from spwm.config.schema import (
    ExperimentConfig,
    ModelConfig,
    MemoryConfig,
    TrainingConfig,
)


def test_experiment_config_instantiates_cleanly():
    cfg = ExperimentConfig()
    assert cfg.project.name == "spwm_v0_1_0"
    assert cfg.model.latent_dim == 128
    assert cfg.model.q_dim == 32
    assert cfg.model.p_dim == 96
    assert cfg.model.num_keypoints == 16
    assert cfg.memory.num_timescales == 2
    assert cfg.memory.timescale_dims == (128, 128)
    assert cfg.training.mode in ("train_predictor", "train_corrector")


def test_unknown_key_raises_value_error():
    """Every non-aligned parameter HAS to surface; unknown keys must raise immediately."""
    cfg = ExperimentConfig()
    d = cfg.to_dict()
    d["model"]["unrecognized_hyperparameter"] = 42

    with pytest.raises(ValueError, match="Unknown configuration parameter"):
        ExperimentConfig.from_dict(d)


def test_latent_dim_mismatch_raises_value_error():
    with pytest.raises(ValueError, match="latent_dim"):
        ModelConfig(latent_dim=128, q_dim=32, p_dim=64, num_keypoints=16)


def test_keypoints_mismatch_raises_value_error():
    with pytest.raises(ValueError, match="num_keypoints"):
        ModelConfig(latent_dim=128, q_dim=32, p_dim=96, num_keypoints=8)


def test_spatial_coupling_non_multiple_raises_value_error():
    with pytest.raises(ValueError, match="integer multiple"):
        ModelConfig(latent_dim=100, q_dim=30, p_dim=70, num_keypoints=15)


def test_timescale_dims_length_mismatch_raises():
    with pytest.raises(ValueError, match="timescale_dims length"):
        MemoryConfig(num_timescales=3, timescale_dims=(64, 64), betas=(0.9, 0.985))


def test_invalid_training_mode_raises():
    with pytest.raises(ValueError, match="Invalid training mode"):
        TrainingConfig(mode="bogus_mode")
