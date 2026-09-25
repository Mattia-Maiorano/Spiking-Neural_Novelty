"""
Mandatory Tiny Overfit Gate for SPWM-v1.
Verifies that the model can successfully memorize and overfit a tiny dataset
(8 trajectories, 10 timesteps) before launching large-scale experiments.
"""

import torch
import pytest
from torch.utils.data import DataLoader

from spwm.data.synthetic_world import MovingObjectsWorld
from spwm.data.event_camera import EventCameraSimulator
from spwm.data.datasets import EventWorldDataset, collate_event_batches
from spwm.models.world_model import SPWM
from spwm.learning.losses import SPWMLoss
from spwm.learning.trainer import Trainer
from spwm.learning.diagnostics import inspect_gradients


def test_tiny_overfit_and_gradient_flow():
    torch.manual_seed(42)

    # 1. Tiny dataset: 8 trajectories, 10 timesteps
    world = MovingObjectsWorld()
    event_sim = EventCameraSimulator(height=32, width=32)
    dataset = EventWorldDataset(
        trajectory_indices=list(range(8)),
        world=world,
        event_simulator=event_sim,
        sequence_length=10,
        num_objects=1,
        cache_data=True,
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_event_batches)

    # 2. Compact SPWM model
    model = SPWM(
        in_channels=2,
        height=32,
        width=32,
        encoder_conv_channels=(16, 32),
        encoder_dim=32,
        latent_dim=32,
        timescale_dims=(16, 16),
        betas=(0.7, 0.95),
        predictor_hidden_dim=64,
        num_objects=1,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    loss_fn = SPWMLoss(
        lambda_pred=1.0,
        lambda_multi=0.2,
        lambda_var=0.01,
        lambda_sparse=0.0001,
        lambda_probe=1.0,
    )

    initial_loss = None
    final_loss = None

    # Train for 40 steps
    for epoch in range(40):
        for batch in dataloader:
            events = batch["events"]
            true_kin = batch["flat_kinematics"]

            optimizer.zero_grad()
            out = model(events)

            loss_out = loss_fn(
                latent_states=out.latent_states,
                predicted_latents=out.predicted_latents,
                mean_spike_rate=out.mean_spike_rate,
                model_predictor=model.predictor,
                decoded_kinematics=out.decoded_kinematics,
                true_kinematics=true_kin,
            )

            current_loss = loss_out.total_loss.item()
            if initial_loss is None:
                initial_loss = current_loss

            loss_out.total_loss.backward()

            # Gradient sanity checks
            diag = inspect_gradients(model)
            assert not diag["has_nan"], "NaN detected in gradients during tiny overfit!"
            assert not diag["has_inf"], "Inf detected in gradients during tiny overfit!"
            assert diag["total_norm"] > 0.0, "Zero gradients detected across model!"

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        final_loss = current_loss

    print(f"\nTiny Overfit: Initial Loss = {initial_loss:.4f} -> Final Loss = {final_loss:.4f}")

    # Check substantial loss decrease
    assert final_loss < initial_loss, f"Loss did not decrease: initial={initial_loss}, final={final_loss}"
    assert final_loss < 0.6 * initial_loss, f"Loss did not decrease sufficiently: {initial_loss} -> {final_loss}"
