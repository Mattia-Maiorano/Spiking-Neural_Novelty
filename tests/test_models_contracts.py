"""
TDD Unit Tests for Layer 1 (`spwm.models`) Neural Primitives & Contracts.
Verifies:
1. Dynamic SpatialSoftmax coordinate grids across arbitrary valid sensor resolutions (e.g. 16x16, 32x32, 64x64).
2. Deterministic `encode_step` and `forward` returning `(features, keypoints)` without side-effect attributes (`last_keypoints`).
3. Strict tensor shape assertions on inputs and outputs of `EventEncoder`, `SpikingLatentDynamics`, `LatentPredictor`, and `PhysicalDecoder`.
4. Absence of dead Hebbian plasticity buffers (`delta_w_buffer`, `B_kin_feedback`) on `SPWM`.
5. Differentiable forward pass through `SPWM` when gradients are enabled (`torch.enable_grad()`).
"""

from __future__ import annotations
import pytest
import torch

from spwm.contracts import (
    DataConfig,
    ExperimentConfig,
    EnvironmentConfig,
    ModelConfig,
    MemoryConfig,
    EncoderProtocol,
    DynamicsProtocol,
    PredictorProtocol,
    PhysicalDecoderProtocol,
    WorldModelProtocol,
)
from spwm.models.encoder import EventEncoder, SpatialSoftmax
from spwm.models.dynamics import SpikingLatentDynamics
from spwm.models.predictors import (
    LatentPredictor,
    PhysicalDecoder,
)
from spwm.models.world_model import SPWM


@pytest.mark.parametrize("resolution", [(16, 16), (32, 32), (64, 48)])
def test_event_encoder_dynamic_grid_and_pure_outputs(resolution):
    """EventEncoder must support any resolution divisible by 4 and fuse 4-channel dual-camera streams."""
    H, W = resolution
    encoder = EventEncoder(
        in_channels=4,
        conv_channels=(16, 32),
        num_keypoints=8,
        out_dim=64,
        input_height=H,
        input_width=W,
    )
    assert isinstance(encoder, EncoderProtocol)
    assert not hasattr(encoder, "last_keypoints"), "EventEncoder must not store mutable side-effect state!"

    # Single frame encode_step (4 channels: 2 global + 2 egocentric wrist)
    frame = torch.randn(3, 4, H, W)
    feat, kp = encoder.encode_step(frame)
    assert feat.shape == (3, 64)
    assert kp.shape == (3, 16)
    assert torch.all(kp >= -1.0) and torch.all(kp <= 1.0)

    # Sequence forward
    seq = torch.randn(3, 5, 4, H, W)
    seq_feat, seq_kp = encoder(seq, return_keypoints=True)
    assert seq_feat.shape == (3, 5, 64)
    assert seq_kp.shape == (3, 5, 16)


def test_encoder_rejects_invalid_channel_or_spatial_dimensions():
    """EventEncoder must raise ValueError immediately if input tensor violates channel or spatial contracts."""
    encoder = EventEncoder(
        in_channels=4,
        conv_channels=(16, 32),
        num_keypoints=8,
        out_dim=64,
        input_height=32,
        input_width=32,
    )
    # Wrong channel count
    with pytest.raises(ValueError, match="dim 1: expected 4, got 3"):
        encoder.encode_step(torch.randn(2, 3, 32, 32))

    # Wrong spatial resolution
    with pytest.raises(ValueError, match="dim 2: expected 32, got 16"):
        encoder.encode_step(torch.randn(2, 4, 16, 16))


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


def test_physical_decoder_articulated_joint_contract():
    """PhysicalDecoder must decode 2 * joint_count kinematics (q*, p*) and validate latent dimension."""
    decoder = PhysicalDecoder(
        latent_dim=64,
        q_dim=16,
        p_dim=48,
        joint_count=6,
        alpha=0.85,
    )
    assert isinstance(decoder, PhysicalDecoderProtocol)

    z_seq = torch.randn(4, 10, 64)
    kin = decoder(z_seq)
    assert kin.shape == (4, 10, 12)  # 2 * 6 joints

    q_sub = z_seq[..., :16]
    p_sub = z_seq[..., 16:]
    pos_sub = decoder.decode_pos(q_sub)
    vel_sub = decoder.decode_vel(p_sub)
    assert pos_sub.shape == (4, 10, 6)
    assert vel_sub.shape == (4, 10, 6)

    with pytest.raises(ValueError, match="dim 2: expected 64, got 32"):
        decoder(torch.randn(4, 10, 32))


def test_spwm_pure_model_no_dead_buffers_and_differentiable_forward():
    """
    SPWM must conform to WorldModelProtocol, contain zero dead Hebbian buffers,
    and preserve autograd graph during forward() and forward_representation() when grad is enabled.
    """
    cfg = ExperimentConfig(
        environment=EnvironmentConfig(height=32, width=32, joint_count=6, action_dim=6),
        data=DataConfig(joint_count=6, action_dim=6),
        model=ModelConfig(latent_dim=32, q_dim=8, p_dim=24, num_keypoints=4, encoder_dim=32),
        memory=MemoryConfig(timescale_dims=(16, 16), betas=(0.90, 0.985)),
    )
    model = SPWM.from_config(cfg)
    assert isinstance(model, WorldModelProtocol)
    assert not hasattr(model, "delta_w_buffer")
    assert not hasattr(model, "B_kin_feedback")
    assert not hasattr(model, "corrector")
    assert not hasattr(model, "m2")

    events = torch.randn(2, 6, 4, 32, 32)
    actions = torch.randn(2, 6, 6)
    rep_latents, rep_errors = model.forward_representation(events)
    assert rep_latents.shape == (2, 6, 32)
    assert rep_errors.shape == (2, 6, 32)

    out = model(events, actions=actions)
    assert out.latent_states.shape == (2, 6, 32)
    assert out.keypoints is not None
    assert out.keypoints.shape == (2, 6, 8)
    assert out.decoded_kinematics is not None
    assert out.decoded_kinematics.shape == (2, 6, 12)

    # Verify differentiability through encoder (both global & wrist stems + fusion), dynamics, and physical_decoder
    loss = out.latent_states.sum() + out.decoded_kinematics.sum()
    loss.backward()

    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.conv.parameters())
    assert model.encoder.wrist_stem is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.wrist_stem.parameters())
    assert model.encoder.fusion_head is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.fusion_head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.dynamics.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.physical_decoder.parameters())


def test_event_gated_temporal_integration_in_spiking_dynamics():
    """
    Verifies that SpikingLatentDynamics integrates velocity when events are absent (K_gate -> 0),
    and allows sensory_keypoints to correct position when events burst (K_gate -> 1).
    """
    dyn = SpikingLatentDynamics(
        input_dim=32,
        latent_dim=32,
        q_dim=8,
        p_dim=24,
        dt=0.05,
        timescale_dims=(16, 16),
        betas=(0.90, 0.985),
    )
    B = 2
    state = dyn.init_state(B)
    sens_input = torch.zeros(B, 32)
    sens_kp = torch.ones(B, 8) * 5.0

    # Case 1: Empty events (activity = 0)
    z_empty, state_empty = dyn.step(
        sensory_input=sens_input,
        state=state,
        sensory_keypoints=sens_kp,
        event_activity=torch.zeros(B, 1),
    )
    q_empty = z_empty[:, :8]
    # At t=0, q_prev=0, and with K_gate=0, q_next = dt * vel(p) (does NOT jump to 5.0)
    assert not torch.allclose(q_empty, sens_kp)
    assert torch.all(torch.abs(q_empty) < 1.0)

    # Case 2: Burst of events (activity = 0.2)
    z_burst, state_burst = dyn.step(
        sensory_input=sens_input,
        state=state,
        sensory_keypoints=sens_kp,
        event_activity=torch.full((B, 1), 0.2),
    )
    q_burst = z_burst[:, :8]
    # With K_gate -> 1, q_next heavily corrects toward sens_kp (5.0)
    assert torch.allclose(q_burst, sens_kp, atol=0.2)


def test_anti_evanescence_gate_suppression_and_ramp():
    """
    Verifies that the Anti-Evanescence gate strictly clamps K_gate to 0 when event activity
    is below the quiescence threshold, prevents coordinate corruption toward collapsed (0, 0)
    keypoints, and holds ramp accumulation until active motion resumes.
    """
    dyn = SpikingLatentDynamics(
        input_dim=32,
        latent_dim=32,
        q_dim=8,
        p_dim=24,
        dt=0.05,
        timescale_dims=(16, 16),
        betas=(0.90, 0.985),
        gate_ramp_steps=5,
        activity_threshold=0.001,
    )
    B = 2
    state = dyn.init_state(B)
    sens_input = torch.zeros(B, 32)
    collapsed_skp = torch.zeros(B, 8)
    state.z_prev[:, :8] = 2.5

    # 1. Evanescent / quiescent frames (activity = 0.0005 < 0.001)
    for _ in range(5):
        k_gate_eval = dyn.compute_k_gate(torch.full((B, 1), 0.0005), state.ema_spikes)
        assert torch.all(k_gate_eval == 0.0)

        z_next, state = dyn.step(
            sensory_input=sens_input,
            state=state,
            sensory_keypoints=collapsed_skp,
            event_activity=torch.full((B, 1), 0.0005),
        )
        # Verify ramp did not accumulate during silence
        assert torch.all(state.grounded_steps == 0.0)
    # Verify q stayed centered around 2.5 and was not pulled down to collapsed 0.0
    assert torch.all(z_next[:, :8] > 1.8)

    # 2. Motion resumes (activity = 0.040 > 0.001)
    valid_skp = torch.ones(B, 8) * 3.0
    z_active, state_active = dyn.step(
        sensory_input=sens_input,
        state=state,
        sensory_keypoints=valid_skp,
        event_activity=torch.full((B, 1), 0.040),
    )
    assert torch.all(state_active.grounded_steps == 1.0)


def test_reverted_momentum_preserves_membrane_continuity():
    """
    Verifies Option 1 Reversion:
    Clean continuous momentum is maintained: p_{t+1} = LayerNorm(fuse_spikes + fuse_mems).
    No raw finite-difference shocks bypass LayerNorm, ensuring max(|p|) <= 3.5.
    """
    dyn = SpikingLatentDynamics(
        input_dim=32,
        latent_dim=32,
        q_dim=8,
        p_dim=24,
        dt=0.05,
        timescale_dims=(16, 16),
        betas=(0.90, 0.985),
        gate_ramp_steps=4,
        activity_threshold=0.001,
    )
    B = 2
    state = dyn.init_state(B)
    sens_input = torch.zeros(B, 32)
    state.z_prev[:, :8] = 1.0
    state.z_prev[:, 8:] = 0.0

    target_skp = torch.ones(B, 8) * 1.5
    for step in range(4):
        z_next, state = dyn.step(
            sensory_input=sens_input,
            state=state,
            sensory_keypoints=target_skp,
            event_activity=torch.full((B, 1), 0.08),
            in_sensory_phase=True,
        )

    # Coordinate q smoothly coupled toward target
    assert torch.all(z_next[:, :8] > 1.0)
    # Momentum p remains normalized by LayerNorm (never exploding above 3.5)
    p_mem = z_next[:, 8:]
    assert torch.all(torch.abs(p_mem) <= 3.5)


def test_slow_gated_injection_schedule():
    """
    Verifies that _compute_default_phase_schedule produces sensory grounding:
    - Starting 7 steps before contact
    - Kept until 3 steps after contact
    - Released for remainder of sequence
    """
    from spwm.data.datasets import _compute_default_phase_schedule
    import numpy as np

    T = 105
    setup_steps, approach_end, heft_end, flags = _compute_default_phase_schedule(
        T, rng=None, pre_contact_steps=7, post_contact_steps=3
    )

    heft_start = approach_end - 7
    sensory_release = approach_end + 3
    assert flags[heft_start - 1] == 0
    assert np.all(flags[heft_start:sensory_release] == 1)
    assert flags[sensory_release] == 2
    assert np.all(flags[sensory_release:] == 2)



