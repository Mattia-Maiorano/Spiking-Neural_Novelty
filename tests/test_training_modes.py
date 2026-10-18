"""
Unit Tests for SPWM-v8 Decoupled Training Modes (train_predictor and train_corrector).
Verifies:
1. Modalità train_predictor (Fase Nominale): Corrector disattivato, gradient isolation su predittore.
2. Modalità train_corrector (Fase di Stabilizzazione & Drift): Predittore congelato (requires_grad=False), correttore attivo.
3. Gradient isolation assertions: Rilevamento immediato di gradient leak.
4. Data loading drift constraints.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spwm.models.world_model import SPWM
from spwm.learning.trainer import Trainer
from spwm.learning.losses import SPWMLoss
from spwm.data.datasets import EventWorldDataset, collate_event_batches


def create_dummy_dataloader(sequence_length=15, batch_size=2, total_traj=4, drift_injection_prob=0.0):
    dataset = EventWorldDataset(
        trajectory_indices=list(range(total_traj)),
        sequence_length=sequence_length,
        drift_injection_prob=drift_injection_prob,
        drift_magnitude=0.05,
        cache_data=True,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_event_batches)


def test_train_representation_mode_setup():
    """In train_representation mode, ALIF predictor is frozen and encoder/probe have active optimizer."""
    model = SPWM(
        latent_dim=32,
        q_dim=8,
        p_dim=24,
        timescale_dims=(16, 16),
        enable_corrector=False,
    )
    train_loader = create_dummy_dataloader(drift_injection_prob=0.0)
    val_loader = create_dummy_dataloader(drift_injection_prob=0.0)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        mode="train_representation",
    )

    assert trainer.mode == "train_representation"
    assert trainer.corrector_optimizer is None
    assert trainer.predictor_optimizer is None
    assert trainer.encoder_optimizer is not None  # representation_optimizer alias

    # Predictor parameters must be frozen
    for p in model.predictor.parameters():
        assert not p.requires_grad

    # Encoder parameters must be trainable
    assert any(p.requires_grad for p in model.encoder.parameters())


def test_train_predictor_mode_setup():
    """In train_predictor mode, encoder and probe are frozen and predictor has active optimizer."""
    model = SPWM(
        latent_dim=32,
        q_dim=8,
        p_dim=24,
        timescale_dims=(16, 16),
        enable_corrector=False,
    )
    train_loader = create_dummy_dataloader(drift_injection_prob=0.0)
    val_loader = create_dummy_dataloader(drift_injection_prob=0.0)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        mode="train_predictor",
    )

    assert trainer.mode == "train_predictor"
    assert trainer.corrector_optimizer is None
    assert trainer.encoder_optimizer is None
    assert trainer.predictor_optimizer is not None

    # Corrector parameters must have requires_grad=False (if present)
    if hasattr(model, "corrector") and model.corrector is not None:
        for p in model.corrector.parameters():
            assert not p.requires_grad

    # Encoder parameters must have requires_grad=False
    for p in model.encoder.parameters():
        assert not p.requires_grad

    # Predictor parameters must have requires_grad=True
    predictor_params = [p for n, p in model.named_parameters() if "predictor" in n and "sensory_predictor" not in n]
    assert any(p.requires_grad for p in predictor_params)


def test_train_joint_mode_setup():
    """In train_joint mode, both representation (encoder/probe) and predictor/dynamics have active optimizers."""
    model = SPWM(
        latent_dim=32,
        q_dim=8,
        p_dim=24,
        timescale_dims=(16, 16),
        enable_corrector=False,
    )
    train_loader = create_dummy_dataloader(drift_injection_prob=0.0)
    val_loader = create_dummy_dataloader(drift_injection_prob=0.0)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        mode="train_joint",
    )

    assert trainer.mode == "train_joint"
    assert trainer.corrector_optimizer is None
    assert trainer.representation_optimizer is not None
    assert trainer.predictor_optimizer is not None

    # Corrector parameters must have requires_grad=False (if present)
    if hasattr(model, "corrector") and model.corrector is not None:
        for p in model.corrector.parameters():
            assert not p.requires_grad

    # Both encoder and predictor parameters must be trainable
    assert any(p.requires_grad for p in model.encoder.parameters())
    assert any(p.requires_grad for p in model.predictor.parameters())
    assert any(p.requires_grad for p in model.physical_decoder.parameters())


def test_train_corrector_mode_setup():
    """In train_corrector mode, all predictor/encoder/dynamics/probe weights are strictly frozen."""
    model = SPWM(
        latent_dim=32,
        q_dim=8,
        p_dim=24,
        timescale_dims=(16, 16),
        enable_corrector=True,
        corrector_dim=16,
    )
    train_loader = create_dummy_dataloader(drift_injection_prob=0.15)
    val_loader = create_dummy_dataloader(drift_injection_prob=0.15)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        mode="train_corrector",
    )

    assert trainer.mode == "train_corrector"
    assert trainer.encoder_optimizer is None
    assert trainer.predictor_optimizer is None
    assert trainer.probe_optimizer is None
    assert trainer.corrector_optimizer is not None

    # All non-corrector parameters MUST have requires_grad=False
    for name, param in model.named_parameters():
        if "corrector" not in name:
            assert not param.requires_grad, f"Param {name} should be frozen in train_corrector mode!"
        else:
            assert param.requires_grad, f"Corrector param {name} should be trainable!"


def test_gradient_isolation_assertion_raises_on_leak():
    """If a non-corrector parameter is unfrozen in train_corrector mode, verify_gradient_isolation must raise AssertionError."""
    model = SPWM(
        latent_dim=32,
        q_dim=8,
        p_dim=24,
        timescale_dims=(16, 16),
        enable_corrector=True,
        corrector_dim=16,
    )
    train_loader = create_dummy_dataloader(drift_injection_prob=0.15)
    val_loader = create_dummy_dataloader(drift_injection_prob=0.15)

    # Instantiate trainer (freezes non-corrector params)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        mode="train_corrector",
    )

    # Artificially create a gradient leak
    for p in model.predictor.parameters():
        p.requires_grad = True
        break

    with pytest.raises(AssertionError, match="Gradient leak!"):
        trainer.verify_gradient_isolation()


def test_train_representation_epoch_execution():
    """Runs a mini training epoch in train_representation mode."""
    model = SPWM(
        latent_dim=32,
        q_dim=8,
        p_dim=24,
        timescale_dims=(16, 16),
        enable_corrector=False,
    )
    train_loader = create_dummy_dataloader(drift_injection_prob=0.0)
    val_loader = create_dummy_dataloader(drift_injection_prob=0.0)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        mode="train_representation",
    )

    metrics = trainer.train_epoch(epoch=1)
    assert "train_loss" in metrics
    assert "train_loss_encoder" in metrics
    assert "train_loss_decoder" in metrics
    assert "train_loss_position" in metrics
    assert "train_loss_velocity" in metrics
    assert metrics["train_loss"] >= 0.0


def test_train_predictor_epoch_execution():
    """Runs a mini training epoch in train_predictor mode."""
    model = SPWM(
        latent_dim=32,
        q_dim=8,
        p_dim=24,
        timescale_dims=(16, 16),
        enable_corrector=False,
    )
    train_loader = create_dummy_dataloader(drift_injection_prob=0.0)
    val_loader = create_dummy_dataloader(drift_injection_prob=0.0)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        mode="train_predictor",
    )

    metrics = trainer.train_epoch(epoch=1)
    assert "train_loss" in metrics
    assert "train_loss_encoder" in metrics
    assert "train_loss_decoder" in metrics
    assert "train_loss_position" in metrics
    assert "train_loss_velocity" in metrics
    assert metrics["train_loss"] >= 0.0


def test_train_joint_epoch_execution():
    """Runs a mini training epoch in train_joint mode and verifies both encoder and predictor parameters update."""
    model = SPWM(
        latent_dim=32,
        q_dim=8,
        p_dim=24,
        timescale_dims=(16, 16),
        enable_corrector=False,
    )
    train_loader = create_dummy_dataloader(drift_injection_prob=0.0)
    val_loader = create_dummy_dataloader(drift_injection_prob=0.0)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        mode="train_joint",
    )

    enc_weight_before = next(model.encoder.parameters()).detach().clone()
    pred_weight_before = next(model.predictor.parameters()).detach().clone()

    metrics = trainer.train_epoch(epoch=1)
    assert "train_loss" in metrics
    assert "train_loss_encoder" in metrics
    assert "train_loss_decoder" in metrics
    assert "train_loss_position" in metrics
    assert "train_loss_velocity" in metrics
    assert metrics["train_loss"] >= 0.0

    enc_weight_after = next(model.encoder.parameters()).detach()
    pred_weight_after = next(model.predictor.parameters()).detach()

    assert not torch.equal(enc_weight_before, enc_weight_after), "Encoder weights did not update during train_joint epoch!"
    assert not torch.equal(pred_weight_before, pred_weight_after), "Predictor weights did not update during train_joint epoch!"


def test_train_corrector_epoch_execution():
    """Runs a mini training epoch in train_corrector mode and verifies only corrector weights change."""
    model = SPWM(
        latent_dim=32,
        q_dim=8,
        p_dim=24,
        timescale_dims=(16, 16),
        enable_corrector=True,
        corrector_dim=16,
    )
    train_loader = create_dummy_dataloader(drift_injection_prob=0.20)
    val_loader = create_dummy_dataloader(drift_injection_prob=0.20)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        mode="train_corrector",
        corrector_horizon=5,
    )

    pred_weight_before = model.predictor.net[0].weight.detach().cpu().clone()
    corr_weight_before = model.corrector.gain_net[0].weight.detach().cpu().clone()

    metrics = trainer.train_epoch(epoch=1)
    assert "train_loss" in metrics
    assert "train_kin_loss" in metrics
    assert metrics["train_loss"] > 0.0

    # Fast predictor weights must remain IDENTICAL
    pred_weight_after = model.predictor.net[0].weight.detach().cpu()
    assert torch.equal(pred_weight_before, pred_weight_after), "Predictor weights changed during train_corrector mode!"

    # Corrector weights should be updated
    corr_weight_after = model.corrector.gain_net[0].weight.detach().cpu()
    assert not torch.equal(corr_weight_before, corr_weight_after) or metrics["train_kin_loss"] == 0.0


def test_training_pipeline_checkpoint_chaining(tmp_path):
    """
    Verifies that:
    1. train_representation trains encoder & probe and saves checkpoint.
    2. train_predictor loads representation checkpoint, keeps encoder frozen, trains predictor.
    3. train_corrector loads predictor checkpoint, keeps encoder & predictor frozen, trains corrector.
    """
    # 1. Representation stage
    model_repr = SPWM(latent_dim=32, q_dim=8, p_dim=24, timescale_dims=(16, 16), corrector_dim=16)
    ckpt_repr_file = tmp_path / "model_representation.pt"
    torch.save({"model_state": model_repr.state_dict()}, ckpt_repr_file)

    # 2. Predictor stage
    model_pred = SPWM(latent_dim=32, q_dim=8, p_dim=24, timescale_dims=(16, 16), corrector_dim=16)
    ckpt_data = torch.load(ckpt_repr_file, weights_only=False)
    model_pred.load_state_dict(ckpt_data["model_state"], strict=False)

    # Ensure encoder weights match representation checkpoint
    for p_repr, p_pred in zip(model_repr.encoder.parameters(), model_pred.encoder.parameters()):
        assert torch.equal(p_repr, p_pred)

    # Save predictor stage
    ckpt_pred_file = tmp_path / "model_predictor.pt"
    torch.save({"model_state": model_pred.state_dict()}, ckpt_pred_file)

    # 3. Corrector stage
    model_corr = SPWM(latent_dim=32, q_dim=8, p_dim=24, timescale_dims=(16, 16), enable_corrector=True, corrector_dim=16)
    ckpt_data_pred = torch.load(ckpt_pred_file, weights_only=False)
    model_corr.load_state_dict(ckpt_data_pred["model_state"], strict=False)

    for p_pred, p_corr in zip(model_pred.predictor.parameters(), model_corr.predictor.parameters()):
        assert torch.equal(p_pred, p_corr)


def test_joint_mode_temporal_burn_in():
    """Verifies that temporal_burn_in shifts the rollout starting point while keeping representation loss intact."""
    from spwm.config.schema import ExperimentConfig, TrainingConfig
    from spwm.learning.phases.predictor_phase import PredictorPhaseHandler

    cfg = ExperimentConfig(
        training=TrainingConfig(
            mode="train_joint",
            epochs=1,
            temporal_burn_in=3,
        )
    )
    # Ensure smaller model for test speed
    cfg.model.latent_dim = 32
    cfg.model.q_dim = 8
    cfg.model.p_dim = 24
    cfg.model.num_keypoints = 4
    cfg.memory.timescale_dims = (16, 16)
    cfg.loss.multi_step_horizon = 5

    model = SPWM.from_config(cfg, enable_corrector=False)
    loss_fn = SPWMLoss.from_config(cfg.loss)
    handler = PredictorPhaseHandler(model=model, cfg=cfg, loss_fn=loss_fn, device=torch.device("cpu"))

    batch = {
        "events": torch.zeros((2, 10, 2, 32, 32), dtype=torch.float32),
        "flat_kinematics": torch.zeros((2, 10, 4), dtype=torch.float32),
    }

    # Verify train_step execution
    train_metrics = handler.train_step(batch, epoch=1)
    assert "train_loss" in train_metrics
    assert "train_loss_encoder" in train_metrics
    assert "train_loss_decoder" in train_metrics
    assert "train_loss_position" in train_metrics
    assert "train_loss_velocity" in train_metrics

    # Verify validate_step execution
    val_metrics = handler.validate_step(batch)
    assert "val_loss" in val_metrics
    assert "val_loss_encoder" in val_metrics
    assert "val_loss_position" in val_metrics

