"""
TDD Unit Tests for Layer 1 (`spwm.models`) Neural Primitives & Contracts.
Verifies:
1. Dynamic SpatialSoftmax coordinate grids across arbitrary valid sensor resolutions (e.g. 16x16, 32x32, 64x64).
2. Deterministic `encode_step` and `forward` returning `(features, keypoints)` without side-effect attributes (`last_keypoints`).
3. Strict tensor shape assertions on inputs and outputs of `EventEncoder`, `SpikingLatentDynamics`, `CascadedTemporalRelay`, and `PhysicalDecoder`.
4. Absence of dead Hebbian plasticity buffers (`delta_w_buffer`, `B_kin_feedback`) on `SPWM`.
5. Differentiable forward pass through `SPWM` when gradients are enabled (`torch.enable_grad()`).
"""

from __future__ import annotations
import pytest
import torch

from spwm.contracts import (
    ExperimentConfig,
    EnvironmentConfig,
    ModelConfig,
    MemoryConfig,
    EncoderProtocol,
    DynamicsProtocol,
    RelayPredictorProtocol,
    PhysicalDecoderProtocol,
    WorldModelProtocol,
)
from spwm.models.encoder import EventEncoder, SpatialSoftmax
from spwm.models.dynamics import SpikingLatentDynamics
from spwm.models.predictors import (
    LatentPredictor,
    CascadedTemporalRelay,
    PhysicalDecoder,
)
from spwm.models.world_model import SPWM


@pytest.mark.parametrize("resolution", [(16, 16), (32, 32), (64, 48)])
def test_event_encoder_dynamic_grid_and_pure_outputs(resolution):
    """EventEncoder must support any resolution divisible by 4 without hardcoded 8x8 SpatialSoftmax grids."""
    H, W = resolution
    encoder = EventEncoder(
        in_channels=2,
        conv_channels=(16, 32),
        num_keypoints=8,
        out_dim=64,
        input_height=H,
        input_width=W,
    )
    assert isinstance(encoder, EncoderProtocol)
    assert not hasattr(encoder, "last_keypoints"), "EventEncoder must not store mutable side-effect state!"

    # Single frame encode_step
    frame = torch.randn(3, 2, H, W)
    feat, kp = encoder.encode_step(frame)
    assert feat.shape == (3, 64)
    assert kp.shape == (3, 16)
    assert torch.all(kp >= -1.0) and torch.all(kp <= 1.0)

    # Sequence forward
    seq = torch.randn(3, 5, 2, H, W)
    seq_feat, seq_kp = encoder(seq, return_keypoints=True)
    assert seq_feat.shape == (3, 5, 64)
    assert seq_kp.shape == (3, 5, 16)


def test_encoder_rejects_invalid_channel_or_spatial_dimensions():
    """EventEncoder must raise ValueError immediately if input tensor violates channel or spatial contracts."""
    encoder = EventEncoder(
        in_channels=2,
        conv_channels=(16, 32),
        num_keypoints=8,
        out_dim=64,
        input_height=32,
        input_width=32,
    )
    # Wrong channel count
    with pytest.raises(ValueError, match="dim 1: expected 2, got 3"):
        encoder.encode_step(torch.randn(2, 3, 32, 32))

    # Wrong spatial resolution
    with pytest.raises(ValueError, match="dim 2: expected 32, got 16"):
        encoder.encode_step(torch.randn(2, 2, 16, 16))


def test_dynamics_strict_beta_and_tier_validation():
    """SpikingLatentDynamics must raise ValueError if betas length mismatches timescale_dims (never silently fallback)."""
    with pytest.raises(ValueError, match="betas length"):
        SpikingLatentDynamics(
            input_dim=64,
            latent_dim=64,
            q_dim=16,
            p_dim=48,
            timescale_dims=(32, 32),
            betas=(0.90, 0.95, 0.99),  # 3 betas for 2 tiers -> must raise!
        )


def test_physical_decoder_multi_object_contract():
    """PhysicalDecoder must decode 4 * num_objects kinematics and validate latent dimension."""
    decoder = PhysicalDecoder(
        latent_dim=64,
        q_dim=16,
        p_dim=48,
        num_objects=2,
        alpha=0.85,
    )
    assert isinstance(decoder, PhysicalDecoderProtocol)

    z_seq = torch.randn(4, 10, 64)
    kin = decoder(z_seq)
    assert kin.shape == (4, 10, 8)  # 4 * 2 objects

    with pytest.raises(ValueError, match="dim 2: expected 64, got 32"):
        decoder(torch.randn(4, 10, 32))


def test_spwm_pure_model_no_dead_buffers_and_differentiable_forward():
    """
    SPWM must conform to WorldModelProtocol, contain zero dead Hebbian buffers,
    and preserve autograd graph during forward() when grad is enabled.
    """
    cfg = ExperimentConfig(
        environment=EnvironmentConfig(height=32, width=32, num_objects=1),
        model=ModelConfig(latent_dim=32, q_dim=8, p_dim=24, num_keypoints=4, encoder_dim=32),
        memory=MemoryConfig(timescale_dims=(16, 16), betas=(0.90, 0.985)),
    )
    model = SPWM.from_config(cfg)
    assert isinstance(model, WorldModelProtocol)
    assert not hasattr(model, "delta_w_buffer")
    assert not hasattr(model, "B_kin_feedback")
    assert not hasattr(model, "corrector")

    events = torch.randn(2, 6, 2, 32, 32)
    out = model(events)
    assert out.latent_states.shape == (2, 6, 32)
    assert out.keypoints is not None
    assert out.keypoints.shape == (2, 6, 8)
    assert out.decoded_kinematics is not None
    assert out.decoded_kinematics.shape == (2, 6, 4)

    # Verify differentiability through encoder, dynamics, and physical_decoder
    loss = out.latent_states.sum() + out.decoded_kinematics.sum()
    loss.backward()

    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.dynamics.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.physical_decoder.parameters())
