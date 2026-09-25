import torch
import pytest
from spwm.baselines.mlp_dynamics import MLPDynamicsWorldModel
from spwm.baselines.gru_world_model import GRUWorldModel
from spwm.baselines.recurrent_snn import VanillaRecurrentSNN


@pytest.mark.parametrize("model_cls", [MLPDynamicsWorldModel, GRUWorldModel, VanillaRecurrentSNN])
def test_baselines_forward_and_rollout(model_cls):
    B, T, C, H, W = 2, 8, 2, 32, 32
    model = model_cls(in_channels=C, height=H, width=W, latent_dim=32, num_objects=1)

    x = torch.zeros(B, T, C, H, W)
    x[:, :, 0, 5:10, 5:10] = 1.0

    out = model(x)
    assert out.latent_states.shape == (B, T, 32)
    assert out.predicted_latents.shape == (B, T, 32)
    assert out.prediction_errors.shape == (B, T - 1)

    # Rollout check
    z0 = out.latent_states[:, 0]
    rollout = model.predict_future(z0, horizon=5)
    assert rollout.shape == (2, 5, 32)
