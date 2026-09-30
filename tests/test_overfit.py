"""
Mandatory Tiny Overfit Gate for SPWM-v3.
Verifies that the model can successfully memorize and overfit a tiny dataset
(8 trajectories, T=150 timesteps) with monotonic loss decrease and stable spike rate (10% - 20%).
"""

import torch
import torch.nn.functional as F
import pytest
from torch.utils.data import DataLoader

from spwm.data.synthetic_world import MovingObjectsWorld
from spwm.data.event_camera import EventCameraSimulator
from spwm.data.datasets import EventWorldDataset, collate_event_batches
from spwm.models.world_model import SPWM
from spwm.learning.diagnostics import inspect_gradients


def test_tiny_overfit_and_gradient_flow():
    torch.manual_seed(42)

    # 1. Tiny dataset: 8 trajectories, T=150 timesteps
    world = MovingObjectsWorld()
    event_sim = EventCameraSimulator(height=32, width=32)
    dataset = EventWorldDataset(
        trajectory_indices=list(range(8)),
        world=world,
        event_simulator=event_sim,
        sequence_length=150,
        num_objects=1,
        cache_data=True,
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_event_batches)

    # 2. SPWM-v3 ALIF model
    model = SPWM(
        in_channels=2,
        height=32,
        width=32,
        encoder_conv_channels=(16, 32),
        encoder_dim=32,
        latent_dim=32,
        timescale_dims=(16, 16),
        betas=(0.90, 0.985),
        beta_mem=0.80,
        gamma=0.18,
        predictor_hidden_dim=64,
        num_objects=1,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-2)

    losses = []
    spike_rates = []

    # Train for 40 epochs
    for epoch in range(40):
        for batch in dataloader:
            events = batch["events"]
            true_kin = batch["flat_kinematics"]

            optimizer.zero_grad()
            out = model(events, accumulate_local_updates=False)

            # Predictive and kinematic probe losses
            z_in = out.latent_states[:, :-1]
            z_target = out.latent_states[:, 1:]
            pred_res = model.predictor(z_in)
            l_pred = F.mse_loss(pred_res.predicted_latent, z_target)

            decoded = model.physical_decoder(out.latent_states)
            l_probe = F.mse_loss(decoded, true_kin)

            total_loss = l_pred + l_probe
            losses.append(total_loss.item())
            spike_rates.append(out.mean_spike_rate.item())

            total_loss.backward()

            # Gradient sanity checks
            diag = inspect_gradients(model)
            assert not diag["has_nan"], "NaN detected in gradients during tiny overfit!"
            assert not diag["has_inf"], "Inf detected in gradients during tiny overfit!"
            assert diag["total_norm"] > 0.0, "Zero gradients detected across model!"

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    initial_loss = losses[0]
    mid_loss = losses[len(losses) // 2]
    final_loss = losses[-1]
    final_spike_rate = spike_rates[-1]

    print(f"\nTiny Overfit (T=150): Initial = {initial_loss:.4f} -> Mid = {mid_loss:.4f} -> Final = {final_loss:.4f} | Spike Rate = {final_spike_rate:.3f}")

    # Verify monotonic downward trend without divergence / rebounds
    assert mid_loss < initial_loss, f"Mid loss did not decrease: initial={initial_loss}, mid={mid_loss}"
    assert final_loss < mid_loss, f"Final loss did not decrease below mid loss: mid={mid_loss}, final={final_loss}"
    assert final_loss < 0.6 * initial_loss, f"Loss did not decrease sufficiently: {initial_loss} -> {final_loss}"

    # Verify spike rate is bounded in healthy regime (10% to 20%)
    assert 0.10 <= final_spike_rate <= 0.20, f"Spike rate {final_spike_rate:.3f} outside [10%, 20%] regime!"
