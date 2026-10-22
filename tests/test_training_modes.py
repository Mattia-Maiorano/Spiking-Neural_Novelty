"""
Unit Tests for Cascaded Temporal Relay Training Modes: train_m1, train_m2, train_m3.
Verifies:
1. Modalità train_m1: M1 strictly confined to 1 phase on clean manifold; M2/M3 frozen.
2. Modalità train_m2: M2 strictly confined to 1 or 2 cyclic phases with Convergence Funnel; M1/M3 frozen.
3. Modalità train_m3: M3 max rollout training where BOTH M2 and M3 are trained; M1 frozen.
4. Gradient isolation assertions: Immediate detection of gradient leaks.
5. Dynamic phase horizon and multi-phase training execution.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spwm.models.world_model import SPWM
from spwm.learning.trainer import Trainer
from spwm.learning.losses import SPWMLoss
from spwm.data.datasets import EventWorldDataset, collate_event_batches


def create_dummy_dataloader(sequence_length=35, batch_size=2, total_traj=4):
    dataset = EventWorldDataset(
        trajectory_indices=list(range(total_traj)),
        sequence_length=sequence_length,
        drift_injection_prob=0.0,
        drift_magnitude=0.0,
        cache_data=True,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_event_batches)


def test_train_m1_mode_setup():
    """In train_m1 mode, M1 is active and confined to 1 phase. M2 and M3 are frozen."""
    model = SPWM(
        latent_dim=32,
        q_dim=8,
        p_dim=24,
        timescale_dims=(16, 16),
        phase_horizon=8,
    )
    train_loader = create_dummy_dataloader(sequence_length=30)
    val_loader = create_dummy_dataloader(sequence_length=30)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        mode="train_m1",
        phase_horizon=8,
    )

    assert trainer.mode == "train_m1"
    assert trainer.predictor_optimizer is not None

    # M1 parameters must be trainable
    assert any(p.requires_grad for p in model.m1.parameters())

    # M2 and M3 must be frozen
    assert all(not p.requires_grad for p in model.m2.parameters())
    assert all(not p.requires_grad for p in model.m3.parameters())

    # Encoder, decoder, dynamics, and sensory_predictor are trained intertwined with M1
    assert any(p.requires_grad for p in model.encoder.parameters())
    assert any(p.requires_grad for p in model.dynamics.parameters())
    assert any(p.requires_grad for p in model.sensory_predictor.parameters())
    assert any(p.requires_grad for p in model.physical_decoder.parameters())


def test_train_m2_mode_setup():
    """In train_m2 mode, M2 is active and confined to 1 recovery phase (H -> 2H). M1 and M3 are frozen."""
    model = SPWM(
        latent_dim=32,
        q_dim=8,
        p_dim=24,
        timescale_dims=(16, 16),
        phase_horizon=8,
    )
    train_loader = create_dummy_dataloader(sequence_length=35)
    val_loader = create_dummy_dataloader(sequence_length=35)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        mode="train_m2",
        phase_horizon=8,
    )

    assert trainer.mode == "train_m2"
    assert trainer.predictor_optimizer is not None

    # M2 parameters must be trainable
    assert any(p.requires_grad for p in model.m2.parameters())

    # M1 and M3 must be frozen
    assert all(not p.requires_grad for p in model.m1.parameters())
    assert all(not p.requires_grad for p in model.m3.parameters())


def test_train_m3_mode_setup():
    """In train_m3 mode, BOTH M2 and M3 are trained over max rollout horizon. M1 is frozen."""
    model = SPWM(
        latent_dim=32,
        q_dim=8,
        p_dim=24,
        timescale_dims=(16, 16),
        phase_horizon=8,
    )
    train_loader = create_dummy_dataloader(sequence_length=35)
    val_loader = create_dummy_dataloader(sequence_length=35)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        mode="train_m3",
        phase_horizon=8,
        max_rollout_horizon=25,
    )

    assert trainer.mode == "train_m3"
    assert trainer.predictor_optimizer is not None

    # BOTH M2 and M3 parameters must be trainable
    assert any(p.requires_grad for p in model.m2.parameters())
    assert any(p.requires_grad for p in model.m3.parameters())

    # M1 must remain strictly frozen
    assert all(not p.requires_grad for p in model.m1.parameters())


def test_gradient_isolation_assertions():
    """verify_gradient_isolation verifies only authorized models require grad."""
    model = SPWM(latent_dim=32, q_dim=8, p_dim=24, timescale_dims=(16, 16))
    train_loader = create_dummy_dataloader()
    val_loader = create_dummy_dataloader()

    # 1. train_m1 isolation
    trainer_m1 = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, mode="train_m1")
    trainer_m1.verify_gradient_isolation()

    # Artificially create a leak into M2
    for p in model.m2.parameters():
        p.requires_grad = True
        break
    with pytest.raises(AssertionError, match="Gradient leak!"):
        trainer_m1.verify_gradient_isolation()

    # Reset
    model = SPWM(latent_dim=32, q_dim=8, p_dim=24, timescale_dims=(16, 16))

    # 2. train_m2 isolation
    trainer_m2 = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, mode="train_m2")
    trainer_m2.verify_gradient_isolation()

    # Artificially create a leak into M1
    for p in model.m1.parameters():
        p.requires_grad = True
        break
    with pytest.raises(AssertionError, match="Gradient leak!"):
        trainer_m2.verify_gradient_isolation()

    # Reset
    model = SPWM(latent_dim=32, q_dim=8, p_dim=24, timescale_dims=(16, 16))

    # 3. train_m3 isolation: M2 and M3 allowed, M1 must be frozen
    trainer_m3 = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, mode="train_m3")
    trainer_m3.verify_gradient_isolation()

    # Artificially create a leak into M1
    for p in model.m1.parameters():
        p.requires_grad = True
        break
    with pytest.raises(AssertionError, match="Gradient leak!"):
        trainer_m3.verify_gradient_isolation()


def test_train_m1_epoch_execution():
    """Runs a mini training epoch in train_m1 mode."""
    model = SPWM(latent_dim=32, q_dim=8, p_dim=24, timescale_dims=(16, 16), phase_horizon=5)
    train_loader = create_dummy_dataloader(sequence_length=15)
    val_loader = create_dummy_dataloader(sequence_length=15)

    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, mode="train_m1", phase_horizon=5)
    metrics = trainer.train_epoch(epoch=1)
    assert "train_loss" in metrics
    assert metrics["train_loss"] >= 0.0


def test_train_m2_epoch_execution_single_phase():
    """Runs a mini training epoch in train_m2 mode strictly over H -> 2H after frozen M1."""
    model = SPWM(latent_dim=32, q_dim=8, p_dim=24, timescale_dims=(16, 16), phase_horizon=5)
    train_loader = create_dummy_dataloader(sequence_length=15)
    val_loader = create_dummy_dataloader(sequence_length=15)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        mode="train_m2",
        phase_horizon=5,
    )
    metrics = trainer.train_epoch(epoch=1)
    assert "train_loss" in metrics


def test_train_m3_epoch_execution_max_rollout():
    """Runs a mini training epoch in train_m3 mode verifying joint M2 and M3 updates."""
    model = SPWM(latent_dim=32, q_dim=8, p_dim=24, timescale_dims=(16, 16), phase_horizon=5)
    train_loader = create_dummy_dataloader(sequence_length=25)
    val_loader = create_dummy_dataloader(sequence_length=25)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        mode="train_m3",
        phase_horizon=5,
        max_rollout_horizon=20,
    )

    m2_w_before = next(model.m2.parameters()).detach().clone()
    m3_w_before = next(model.m3.parameters()).detach().clone()
    m1_w_before = next(model.m1.parameters()).detach().clone()

    metrics = trainer.train_epoch(epoch=1)
    assert "train_loss" in metrics

    m2_w_after = next(model.m2.parameters()).detach()
    m3_w_after = next(model.m3.parameters()).detach()
    m1_w_after = next(model.m1.parameters()).detach()

    # Both M2 and M3 weights must update
    assert not torch.equal(m2_w_before, m2_w_after), "M2 weights did not update during train_m3!"
    assert not torch.equal(m3_w_before, m3_w_after), "M3 weights did not update during train_m3!"

    # M1 weights must remain identical (frozen)
    assert torch.equal(m1_w_before, m1_w_after), "M1 weights changed during train_m3!"


def test_train_cascade_and_auto_enforced_model_pt(tmp_path):
    """Verifies train_cascade runs M1 -> M2 -> M3 using best model.pt and early stopping patience."""
    from spwm.training.engine import enforce_phase_prerequisites, resolve_version_checkpoint

    model = SPWM(latent_dim=32, q_dim=8, p_dim=24, timescale_dims=(16, 16), phase_horizon=4)
    loader = create_dummy_dataloader(sequence_length=20)

    # Missing prerequisite model.pt must raise FileNotFoundError
    with pytest.raises(FileNotFoundError, match="model.pt"):
        resolve_version_checkpoint("spwm_v_test", "train_m1", results_root=tmp_path)

    cascade_dir = tmp_path / "spwm_v_test_train_cascade"
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        mode="train_cascade",
        phase_horizon=4,
        max_rollout_horizon=12,
        early_stopping_patience=2,
        save_dir=cascade_dir,
        results_root=tmp_path,
    )
    trainer.cfg = trainer.cfg.with_overrides()  # keep project name
    history = trainer.fit(epochs=2, patience=1)
    assert len(history) >= 3
    phases_seen = {rec["phase"] for rec in history}
    assert phases_seen == {"train_m1", "train_m2", "train_m3"}

    # Verify model.pt was created for each phase
    for pm in ("train_m1", "train_m2", "train_m3"):
        p_pt = resolve_version_checkpoint(trainer.cfg.project.name, pm, results_root=tmp_path)
        assert p_pt.name == "model.pt"
        assert p_pt.is_file()
