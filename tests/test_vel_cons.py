"""
Unit tests for SPWM-v6.5 Kinematic Velocity Consistency Loss (L_vel_cons).
Verifies:
1. Mathematical calculation and dimension alignment of L_vel_cons.
2. Gradient flow from L_vel_cons to momentum subspace p and projection weights W_vel_cons.
3. Proper dictionary serialization in LossOutput.
4. SPWMLoss forward step with lambda_vel_cons.
"""

import pytest
import torch
import torch.nn as nn
from spwm.learning.losses import SPWMLoss, LossOutput
from spwm.models.world_model import SPWM
from spwm.learning.trainer import Trainer
from torch.utils.data import DataLoader, TensorDataset


def test_velocity_consistency_loss_computation():
    B, T, D = 4, 20, 128
    q_dim, p_dim = 32, 96
    loss_fn = SPWMLoss(
        lambda_vel_cons=0.25,
        q_dim=q_dim,
        p_dim=p_dim,
    )

    latent_states = torch.randn(B, T, D, requires_grad=True)
    l_vel = loss_fn.velocity_consistency_loss(latent_states)

    assert isinstance(l_vel, torch.Tensor)
    assert l_vel.dim() == 0  # scalar
    assert l_vel.item() >= 0.0

    # Test gradient propagation
    l_vel.backward()
    assert latent_states.grad is not None
    assert loss_fn.vel_proj.weight.grad is not None
    # Momentum subspace (dim 32 to 128) must have non-zero gradients
    p_grad = latent_states.grad[:, :-1, q_dim:q_dim + p_dim]
    assert torch.any(p_grad != 0.0)


def test_spwm_loss_forward_with_vel_cons():
    B, T, D = 2, 10, 128
    q_dim, p_dim = 32, 96
    loss_fn = SPWMLoss(
        lambda_pred=1.0,
        lambda_multi=0.5,
        lambda_var=0.1,
        lambda_sparse=0.0025,
        lambda_probe=2.0,
        lambda_coord=0.15,
        lambda_vel_cons=0.25,
        q_dim=q_dim,
        p_dim=p_dim,
    )

    latent_states = torch.randn(B, T, D)
    predicted_latents = torch.randn(B, T, D)
    mean_spike_rate = torch.tensor(0.11)
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
    assert out.l_vel_cons is not None
    assert out.l_vel_cons.item() >= 0.0
    assert out.total_loss.item() > 0.0

    d = out.to_dict()
    assert "l_vel_cons" in d
    assert d["l_vel_cons"] >= 0.0


def test_trainer_optimizer_includes_vel_proj():
    model = SPWM(latent_dim=128, q_dim=32, p_dim=96, timescale_dims=(64, 64), encoder_dim=64)
    loss_fn = SPWMLoss(lambda_vel_cons=0.25, q_dim=32, p_dim=96)

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
    )

    # Check that vel_proj weights are in predictor optimizer
    vel_proj_weights = set(loss_fn.vel_proj.parameters())
    predictor_param_set = set()
    for group in trainer.predictor_optimizer.param_groups:
        for p in group["params"]:
            predictor_param_set.add(p)

    for p in vel_proj_weights:
        assert p in predictor_param_set
