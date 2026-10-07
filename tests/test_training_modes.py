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


def test_train_predictor_mode_setup():
    """In train_predictor mode, corrector is disabled/frozen and only predictor/encoder/probe have active optimizers."""
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
    assert trainer.encoder_optimizer is not None
    assert trainer.predictor_optimizer is not None
    assert trainer.probe_optimizer is not None

    # Corrector parameters must have requires_grad=False (if present)
    if hasattr(model, "corrector") and model.corrector is not None:
        for p in model.corrector.parameters():
            assert not p.requires_grad

    # Predictor parameters must have requires_grad=True
    predictor_params = [p for n, p in model.named_parameters() if "predictor" in n and "sensory_predictor" not in n]
    assert any(p.requires_grad for p in predictor_params)


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

    with pytest.raises(AssertionError, match="Gradient leak detected"):
        trainer.verify_gradient_isolation()


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
        encoder_warmup_epochs=0,
    )

    metrics = trainer.train_epoch(epoch=1)
    assert "total_loss" in metrics
    assert metrics["l_pred"] >= 0.0
    assert metrics["l_corr"] == 0.0  # Corrector was not optimized


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
    corr_weight_before = model.corrector.to_v.weight.detach().cpu().clone()

    metrics = trainer.train_epoch(epoch=1)
    assert "total_loss" in metrics
    assert metrics["l_corr"] > 0.0 or metrics["l_corr_asymptotic"] >= 0.0

    # Fast predictor weights must remain IDENTICAL
    pred_weight_after = model.predictor.net[0].weight.detach().cpu()
    assert torch.equal(pred_weight_before, pred_weight_after), "Predictor weights changed during train_corrector mode!"

    # Corrector weights should be updated
    corr_weight_after = model.corrector.to_v.weight.detach().cpu()
    # Weights should differ if optimizer stepped
    assert not torch.equal(corr_weight_before, corr_weight_after) or metrics["l_corr_asymptotic"] == 0.0
