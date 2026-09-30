import torch
import pytest
from spwm.models.world_model import SPWM


def test_spwm_forward_shapes():
    B, T, C, H, W = 2, 150, 2, 32, 32
    model = SPWM(
        in_channels=C,
        height=H,
        width=W,
        encoder_dim=64,
        latent_dim=64,
        timescale_dims=(32, 32),
        betas=(0.90, 0.985),
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
    B, T, C, H, W = 1, 10, 2, 32, 32
    model = SPWM(
        in_channels=C,
        height=H,
        width=W,
        encoder_dim=32,
        latent_dim=32,
        timescale_dims=(16, 16),
        betas=(0.90, 0.985),
    )
    model.eval()

    torch.manual_seed(42)
    x = (torch.rand(B, T, C, H, W) > 0.8).float()

    with torch.no_grad():
        # Sequence forward
        seq_out = model(x, accumulate_local_updates=False)

        # Step by step
        state = model.init_state(B)
        step_latents = []
        for t in range(T):
            step_out, state = model.step(x[:, t], state, accumulate_local_updates=False)
            step_latents.append(step_out.latent)

        step_latents = torch.stack(step_latents, dim=1)

    assert torch.allclose(seq_out.latent_states, step_latents, atol=1e-5)


def test_autonomous_rollout_shapes():
    model = SPWM(latent_dim=32, timescale_dims=(16, 16))
    z0 = torch.randn(2, 32)

    # Rollout over multiple horizons including long horizon H=50 and H=100
    for H in [1, 5, 10, 25, 50, 100]:
        pred_rollout = model.predict_future(z0, horizon=H)
        assert pred_rollout.shape == (2, H, 32)
        assert not torch.isnan(pred_rollout).any()


def test_cann_attractor_properties():
    from spwm.models.latent_dynamics import ContinuousAttractor
    from spwm.models.predictor import LatentPredictor

    cann = ContinuousAttractor(
        q_dim=32,
        num_basis=64,
        drive_dim=96,
        sigma=0.5,
        temperature=0.1,
    )

    # Test coordinate anchoring within compact bounds [-1, 1]
    # Even when presented with large perturbations outside [-1, 1]
    q_drift = torch.randn(4, 32) * 5.0  # severely drifted coordinates
    drive = torch.randn(4, 96)
    q_anchored = cann(q_drift, drive_input=drive)

    assert q_anchored.shape == (4, 32)
    assert not torch.isnan(q_anchored).any()
    # Continuous attractor manifold bounds: decoded coordinates are strictly within (-1, 1)
    assert (q_anchored >= -1.0).all() and (q_anchored <= 1.0).all()

    # Verify LatentPredictor with CANN
    predictor = LatentPredictor(
        latent_dim=128,
        q_dim=32,
        p_dim=96,
        hidden_dim=256,
        use_cann=True,
        cann_num_basis=64,
    )

    z = torch.randn(4, 10, 128)
    out = predictor(z)
    assert out.predicted_latent.shape == (4, 10, 128)
    assert not torch.isnan(out.predicted_latent).any()
    q_pred = out.predicted_latent[..., :32]
    assert (q_pred >= -1.0).all() and (q_pred <= 1.0).all()
