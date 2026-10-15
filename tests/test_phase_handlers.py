"""
Unit tests for Specialized Phase Handlers and BaseTrainer Engine.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from spwm.config.schema import (
    ExperimentConfig,
    ProjectConfig,
    EnvironmentConfig,
    DataConfig,
    ModelConfig,
    LatentNeuronConfig,
    MemoryConfig,
    CorrectorConfig,
    TrainingConfig,
    LossConfig,
    EvaluationConfig,
)
from spwm.data.datasets import EventWorldDataset, collate_event_batches
from spwm.learning.engine import BaseTrainer
from spwm.learning.losses import SPWMLoss
from spwm.learning.phases import PredictorPhaseHandler, CorrectorPhaseHandler
from spwm.models.world_model import SPWM


def make_test_cfg(mode: str) -> ExperimentConfig:
    return ExperimentConfig(
        project=ProjectConfig(name="test_proj"),
        environment=EnvironmentConfig(num_objects=1),
        data=DataConfig(sequence_length=15, batch_size=2, total_trajectories=4),
        model=ModelConfig(latent_dim=32, q_dim=8, p_dim=24, num_keypoints=4),
        neuron=LatentNeuronConfig(),
        memory=MemoryConfig(timescale_dims=(16, 16)),
        corrector=CorrectorConfig(enabled=True, dim=16),
        training=TrainingConfig(mode=mode, epochs=2, corrector_horizon=5, corrector_burn_in_steps=0),
        loss=LossConfig(),
        evaluation=EvaluationConfig(rollout_horizons=(1, 5)),
    )


def test_predictor_phase_handler_isolation():
    cfg = make_test_cfg("train_predictor")
    model = SPWM.from_config(cfg)
    loss_fn = SPWMLoss.from_config(cfg.loss, q_dim=8, p_dim=24)
    handler = PredictorPhaseHandler(model=model, cfg=cfg, loss_fn=loss_fn, device=torch.device("cpu"))

    # Corrector must be frozen
    assert model.corrector is not None
    for p in model.corrector.parameters():
        assert not p.requires_grad

    # Predictor and encoder must be active
    assert any(p.requires_grad for p in model.predictor.parameters())
    assert any(p.requires_grad for p in model.encoder.parameters())


def test_corrector_phase_handler_isolation():
    cfg = make_test_cfg("train_corrector")
    model = SPWM.from_config(cfg)
    loss_fn = SPWMLoss.from_config(cfg.loss, q_dim=8, p_dim=24)
    handler = CorrectorPhaseHandler(model=model, cfg=cfg, loss_fn=loss_fn, device=torch.device("cpu"))

    # Corrector must be trainable
    assert all(p.requires_grad for p in model.corrector.parameters())

    # All other modules must be strictly frozen
    for name, param in model.named_parameters():
        if "corrector" not in name:
            assert not param.requires_grad, f"{name} should be frozen"


def test_base_trainer_epoch_loop_and_checkpoint(tmp_path):
    cfg = make_test_cfg("train_predictor")
    model = SPWM.from_config(cfg)

    dataset = EventWorldDataset(
        trajectory_indices=list(range(4)),
        sequence_length=15,
        cache_data=True,
    )
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_event_batches)

    trainer = BaseTrainer(
        model=model,
        cfg=cfg,
        train_loader=loader,
        val_loader=loader,
        save_dir=tmp_path,
    )

    history = trainer.fit(epochs=2)
    assert len(history) == 2
    assert (tmp_path / "checkpoint.pt").is_file()
    assert (tmp_path / "training_log.json").is_file()
    assert (tmp_path / "training_log.csv").is_file()
