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
    TrainingConfig,
    LossConfig,
    EvaluationConfig,
)
from spwm.data.datasets import EventWorldDataset, collate_event_batches
from spwm.learning.engine import BaseTrainer
from spwm.learning.losses import SPWMLoss
from spwm.learning.phases import PredictorPhaseHandler
from spwm.models.world_model import SPWM


def make_test_cfg(mode: str, phase_horizon: int = 5) -> ExperimentConfig:
    return ExperimentConfig(
        project=ProjectConfig(name="test_proj"),
        environment=EnvironmentConfig(num_objects=1),
        data=DataConfig(sequence_length=25, batch_size=2, total_trajectories=4),
        model=ModelConfig(latent_dim=32, q_dim=8, p_dim=24, num_keypoints=4),
        neuron=LatentNeuronConfig(),
        memory=MemoryConfig(timescale_dims=(16, 16)),
        training=TrainingConfig(
            mode=mode,
            epochs=2,
            phase_horizon=phase_horizon,
            max_rollout_horizon=20,
        ),
        loss=LossConfig(),
        evaluation=EvaluationConfig(rollout_horizons=(1, 5)),
    )


def test_m1_phase_handler_isolation():
    cfg = make_test_cfg("train_m1", phase_horizon=5)
    model = SPWM.from_config(cfg)
    loss_fn = SPWMLoss.from_config(cfg.loss)
    handler = PredictorPhaseHandler(model=model, cfg=cfg, loss_fn=loss_fn, device=torch.device("cpu"))

    # M1 active, M2 and M3 frozen
    assert any(p.requires_grad for p in model.m1.parameters())
    assert all(not p.requires_grad for p in model.m2.parameters())
    assert all(not p.requires_grad for p in model.m3.parameters())
    assert any(p.requires_grad for p in model.encoder.parameters())
    assert any(p.requires_grad for p in model.dynamics.parameters())
    assert any(p.requires_grad for p in model.sensory_predictor.parameters())
    assert any(p.requires_grad for p in model.physical_decoder.parameters())


def test_m2_phase_handler_isolation():
    cfg = make_test_cfg("train_m2", phase_horizon=5)
    model = SPWM.from_config(cfg)
    loss_fn = SPWMLoss.from_config(cfg.loss)
    handler = PredictorPhaseHandler(model=model, cfg=cfg, loss_fn=loss_fn, device=torch.device("cpu"))

    # M2 active, M1 and M3 frozen
    assert any(p.requires_grad for p in model.m2.parameters())
    assert all(not p.requires_grad for p in model.m1.parameters())
    assert all(not p.requires_grad for p in model.m3.parameters())
    assert all(not p.requires_grad for p in model.encoder.parameters())


def test_m3_phase_handler_joint_setup():
    cfg = make_test_cfg("train_m3", phase_horizon=5)
    model = SPWM.from_config(cfg)
    loss_fn = SPWMLoss.from_config(cfg.loss)
    handler = PredictorPhaseHandler(model=model, cfg=cfg, loss_fn=loss_fn, device=torch.device("cpu"))

    # BOTH M2 and M3 active, M1 frozen
    assert any(p.requires_grad for p in model.m2.parameters())
    assert any(p.requires_grad for p in model.m3.parameters())
    assert all(not p.requires_grad for p in model.m1.parameters())
    assert all(not p.requires_grad for p in model.encoder.parameters())


def test_base_trainer_epoch_loop_and_checkpoint(tmp_path):
    cfg = make_test_cfg("train_m1", phase_horizon=5)
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
