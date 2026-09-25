import torch
import pytest
from spwm.models.world_model import SPWM


def test_no_temporal_leakage_from_future():
    """
    Scientific Leakage Gate:
    Modifying the input events at future timesteps t' > t
    MUST have ZERO impact on latent states and predictions at time <= t.
    """
    B, T, C, H, W = 1, 10, 2, 32, 32
    model = SPWM(
        in_channels=C,
        height=H,
        width=W,
        encoder_dim=32,
        latent_dim=32,
        timescale_dims=(16, 16),
    )
    model.eval()

    torch.manual_seed(100)
    x1 = (torch.rand(B, T, C, H, W) > 0.85).float()
    x2 = x1.clone()

    # Modify future timesteps: change everything from t = 5 to 9
    x2[:, 5:] = (torch.rand(B, 5, C, H, W) > 0.5).float()

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    # Latent states at t=0, 1, 2, 3, 4 must be EXACTLY identical
    past_latents_1 = out1.latent_states[:, :5]
    past_latents_2 = out2.latent_states[:, :5]
    assert torch.allclose(past_latents_1, past_latents_2, atol=1e-6), "Future data leaked into past latent states!"

    # Predictions made at t=0, 1, 2, 3, 4 must also be identical
    past_preds_1 = out1.predicted_latents[:, :5]
    past_preds_2 = out2.predicted_latents[:, :5]
    assert torch.allclose(past_preds_1, past_preds_2, atol=1e-6), "Future data leaked into past predictions!"
