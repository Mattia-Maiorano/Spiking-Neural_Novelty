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


def test_port_hamiltonian_predictor_properties():
    from spwm.models.predictor import LatentPredictor

    predictor = LatentPredictor(
        latent_dim=128,
        q_dim=32,
        p_dim=96,
        hidden_dim=256,
        use_port_hamiltonian=True,
        eps_diss=1e-4,
    )

    # Verify skew-symmetry of J
    J = 0.5 * (predictor.W_skew - predictor.W_skew.T)
    assert torch.allclose(J, -J.T, atol=1e-6), "J must be skew-symmetric"

    # Verify damping matrix R >= eps_diss
    R_diag = torch.nn.functional.softplus(predictor.gamma_diss) + predictor.eps_diss
    assert (R_diag >= 1e-4).all(), "R diagonal must be strictly positive"

    # Verify W_rec = J - R
    W_rec = predictor.get_port_hamiltonian_w_rec()
    assert W_rec.shape == (96, 96)

    # Test forward pass with batch of 4 sequences of length 10
    z = torch.randn(4, 10, 128)
    out = predictor(z)
    assert out.predicted_latent.shape == (4, 10, 128)
    assert not torch.isnan(out.predicted_latent).any()
