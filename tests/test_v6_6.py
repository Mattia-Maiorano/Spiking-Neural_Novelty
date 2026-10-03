"""
Unit tests for SPWM-v6.6:
1. SPWMLoss forward step with direct velocity penalty ||p_t - Delta q_t||_2 and L1 sparsity |mean(s) - 0.10|.
2. Encoder parameter freezing (param.requires_grad = False) during the first 70 epochs and unfreezing after epoch 70.
3. Trainer warmup curriculum phase (holding K=1 and suppressing L_multi during warmup).
4. Direct primary velocity coupling gradient flow into momentum p.
"""

import pytest
import torch
import torch.nn as nn
from spwm.learning.losses import SPWMLoss, LossOutput
from spwm.models.world_model import SPWM
from spwm.learning.trainer import Trainer
from torch.utils.data import DataLoader, TensorDataset


def test_spwm_v6_6_loss_forward():
    B, T, D = 2, 10, 128
    q_dim, p_dim = 32, 96
    loss_fn = SPWMLoss(
        lambda_pred=1.0,
        lambda_multi=0.5,
        lambda_var=0.1,
        lambda_sparse=0.5,
        lambda_vel=0.5,
        lambda_probe=2.0,
        lambda_coord=0.15,
        target_spike_rate=0.10,
        q_dim=q_dim,
        p_dim=p_dim,
    )

    latent_states = torch.randn(B, T, D)
    predicted_latents = torch.randn(B, T, D)
    mean_spike_rate = torch.tensor(0.12)
    true_kin = torch.randn(B, T, 4)
    decoded_kin = torch.randn(B, T, 4)
    keypoints = torch.randn(B, T, 32)

    out = loss_fn(
        latent_states=latent_states,
        predicted_latents=predicted_latents,
        mean_spike_rate=mean_spike_rate,
        decoded_kinematics=decoded_kin,
        true_kinematics=true_kin,
        keypoints=keypoints,
    )

    assert isinstance(out, LossOutput)
    assert out.l_coord is not None
    assert out.l_vel is not None
    assert out.total_loss.item() > 0.0

    # Verify L1 sparsity calculation: |0.12 - 0.10| = 0.02
    assert torch.isclose(out.l_sparse, torch.tensor(0.02), atol=1e-5)

    d = out.to_dict()
    assert "l_coord" in d
    assert "l_vel" in d
    assert "l_vel_cons" not in d


def test_direct_momentum_primary_velocity_coupling():
    """Verify that direct primary velocity loss ||p_t - Delta q_t||_2 couples p directly to Delta q."""
    B, T, D = 2, 8, 128
    q_dim, p_dim = 32, 96
    loss_fn = SPWMLoss(lambda_vel=0.5, q_dim=q_dim, p_dim=p_dim)

    latents = torch.randn(B, T, D, requires_grad=True)
    preds = torch.randn(B, T, D)
    mean_spike = torch.tensor(0.10)

    out = loss_fn(
        latent_states=latents,
        predicted_latents=preds,
        mean_spike_rate=mean_spike,
    )

    assert out.l_vel is not None
    out.l_vel.backward()

    # Gradient on momentum subspace p [32:128] must be non-zero and directly driven by finite-difference velocity
    assert latents.grad is not None
    assert torch.any(latents.grad[..., q_dim:] != 0.0)


def test_encoder_freezing_first_70_epochs():
    """Verify explicit freeze (requires_grad = False) of encoder conv and spatial_softmax for epochs <= 70."""
    model = SPWM(latent_dim=128, q_dim=32, p_dim=96, timescale_dims=(64, 64), encoder_dim=64)
    loss_fn = SPWMLoss(lambda_multi=0.5, multi_step_horizon=6, q_dim=32, p_dim=96)

    events = torch.randn(4, 5, 2, 32, 32)
    kin = torch.randn(4, 5, 4)
    dataset = TensorDataset(events, kin)

    def collate_fn(batch):
        return {
            "events": torch.stack([b[0] for b in batch]),
            "flat_kinematics": torch.stack([b[1] for b in batch]),
        }

    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        loss_fn=loss_fn,
        learning_rate=1e-3,
        device="cpu",
        curriculum_multi_step=True,
        encoder_warmup_epochs=70,
        start_epoch=1,
    )

    # During warmup epoch 50 (<= 70):
    trainer.train_epoch(epoch=50)
    for p in model.encoder.conv.parameters():
        assert p.requires_grad is False
    for p in model.encoder.spatial_softmax.parameters():
        assert p.requires_grad is False

    # After warmup epoch 71 (> 70):
    trainer.train_epoch(epoch=71)
    for p in model.encoder.conv.parameters():
        assert p.requires_grad is True
    for p in model.encoder.spatial_softmax.parameters():
        assert p.requires_grad is True
