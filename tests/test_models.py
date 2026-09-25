import torch
import pytest
from spwm.models.world_model import SPWM


def test_spwm_forward_shapes():
    B, T, C, H, W = 2, 10, 2, 32, 32
    model = SPWM(
        in_channels=C,
        height=H,
        width=W,
        encoder_dim=64,
        latent_dim=64,
        timescale_dims=(32, 32),
        betas=(0.7, 0.95),
        num_objects=1,
    )

    x = torch.zeros(B, T, C, H, W)
    # Add arbitrary sparse events
    x[:, :, 0, 10:15, 10:15] = 1.0

    output = model(x)

    assert output.latent_states.shape == (B, T, 64)
    assert output.predicted_latents.shape == (B, T, 64)
    assert output.prediction_errors.shape == (B, T - 1)
    assert output.fast_spikes.shape == (B, T, 32)
    assert output.slow_spikes.shape == (B, T, 32)
    assert output.decoded_kinematics.shape == (B, T, 4)
    assert 0.0 <= output.mean_spike_rate.item() <= 1.0


def test_spwm_step_vs_sequence_equivalence():
    B, T, C, H, W = 1, 5, 2, 32, 32
    model = SPWM(
        in_channels=C,
        height=H,
        width=W,
        encoder_dim=32,
        latent_dim=32,
        timescale_dims=(16, 16),
        betas=(0.7, 0.95),
    )
    model.eval()

    torch.manual_seed(42)
    x = (torch.rand(B, T, C, H, W) > 0.8).float()

    with torch.no_grad():
        # Sequence forward
        seq_out = model(x)

        # Step by step
        state = model.init_state(B)
        step_latents = []
        for t in range(T):
            step_out, state = model.step(x[:, t], state)
            step_latents.append(step_out.latent)

        step_latents = torch.stack(step_latents, dim=1)

    assert torch.allclose(seq_out.latent_states, step_latents, atol=1e-5)


def test_autonomous_rollout_shapes():
    model = SPWM(latent_dim=32, timescale_dims=(16, 16))
    z0 = torch.randn(2, 32)

    # Rollout over multiple horizons
    for H in [1, 5, 10, 25]:
        pred_rollout = model.predict_future(z0, horizon=H)
        assert pred_rollout.shape == (2, H, 32)
        assert not torch.isnan(pred_rollout).any()
