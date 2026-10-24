"""Unit tests for typed ExperimentConfig schema.
Verifies strict validation, error-raising on misaligned parameters, and absence of silent fallbacks.
"""

import pytest
from spwm.config.schema import (
    ExperimentConfig,
    MemoryConfig,
    ModelConfig,
    TrainingConfig,
)


def test_experiment_config_instantiates_cleanly():
    cfg = ExperimentConfig()
    assert cfg.project.name in ("spwm_v0_2_0", "spwm_v0_2_1", "spwm_v0_2_2", "spwm_v0_3_0", "spwm_v0_3_1")
    assert cfg.model.latent_dim == 128
    assert cfg.model.q_dim == 32
    assert cfg.model.p_dim == 96
    assert cfg.model.num_keypoints == 16
    assert cfg.memory.num_timescales == 2
    assert cfg.memory.timescale_dims == (128, 128)
    assert cfg.training.mode == "train_m1"
    assert cfg.training.phase_horizon in (5, 10, 50)
    assert cfg.training.early_stopping_patience == 20
    assert cfg.loss.lambda_encoder == 1.0

    assert cfg.loss.lambda_decoder in (1.0, 2.0)
    assert cfg.loss.lambda_position in (1.0, 1.5)
    assert cfg.loss.lambda_velocity == 1.0

    # Pruned attributes must not exist
    assert not hasattr(cfg.training, "max_rollout_horizon")
    assert not hasattr(cfg.training, "m2_checkpoint")
    assert not hasattr(cfg.training, "m2_num_phases")
    assert not hasattr(cfg.training, "m1_horizon")
    assert not hasattr(cfg.training, "m2_horizon")
    assert not hasattr(cfg.training, "m3_horizon")
    assert not hasattr(cfg.model, "encoder_type")
    assert not hasattr(cfg.model, "symplectic_m1_all_the_way")
    assert not hasattr(cfg.loss, "multi_step_horizon")


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


def test_invalid_and_pruned_training_modes_raise():
    with pytest.raises(ValueError, match="Invalid training mode"):
        TrainingConfig(mode="bogus_mode")
    # Verify train_representation and train_m2 are valid modes
    cfg_rep = TrainingConfig(mode="train_representation")
    assert cfg_rep.mode == "train_representation"
    cfg_m2 = TrainingConfig(mode="train_m2")
    assert cfg_m2.mode == "train_m2"
    # Legacy modalities remain pruned and raise ValueError
    for pruned_mode in [
        "train_predictor",
        "train_joint",
        "train_corrector",
        "train_m3",
        "train_cascade",
    ]:
        with pytest.raises(ValueError, match="Invalid training mode"):
            TrainingConfig(mode=pruned_mode)
