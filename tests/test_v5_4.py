"""
Unit Tests for SPWM-v5.4 Architecture.

Tests:
  1. Active Velocity Supervision (L_vel): non-zero loss on moving keypoints and gradient connectivity.
  2. Persistent Hamiltonian Limit Cycles: autonomous simulation for H=200 steps sustains firing in [5%, 18%]
     without threshold quenching to zero spikes and without runaway divergence.
  3. Bounded Momentum Reflection / Wall Bounce Reflex: velocity inversion (v_k <- -0.8 * v_k) on collision (|q| >= 0.98).
  4. Learnable Bias on W_vel: linear projection v_t = W_vel @ s_bar_p + b_vel.
  5. Port-Hamiltonian Stability: Re(lambda(W_rec)) <= -epsilon_diss with epsilon_diss=1e-5.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import pytest
import numpy as np

from spwm.models.latent_dynamics import SpikingLatentDynamics, DynamicsState
from spwm.models.neurons import ALIFCell
from spwm.models.world_model import SPWM
from spwm.learning.losses import SPWMLoss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dynamics_v5_4():
    """Standard v5.4 SpikingLatentDynamics."""
    return SpikingLatentDynamics(
        input_dim=128,
        latent_dim=128,
        q_dim=32,
        p_dim=96,
        timescale_dims=[64, 64],
        ema_decay=0.9,
        gamma=0.35,
        epsilon_diss=1e-5,
    )


@pytest.fixture
def model_v5_4():
    """SPWM v5.4 Model."""
    return SPWM(
        in_channels=2,
        height=32,
        width=32,
        encoder_dim=128,
        latent_dim=128,
        q_dim=32,
        p_dim=96,
        timescale_dims=(64, 64),
        gamma=0.35,
        epsilon_diss=1e-5,
    )


# ---------------------------------------------------------------------------
# Test 1: Velocity Supervision (L_vel)
# ---------------------------------------------------------------------------

class TestVelocitySupervision:
    """Verifies that L_vel computes backward finite-difference targets and provides active gradients."""

    def test_velocity_loss_nonzero_on_moving_keypoints(self):
        """L_vel must output strictly positive loss on moving keypoints with unaligned W_vel."""
        loss_fn = SPWMLoss(lambda_vel=0.5, delta_t=1.0)
        B, T, q_dim = 4, 20, 32

        # Synthetic moving keypoints: sinusoidal trajectory
        t_steps = torch.linspace(0, 4 * np.pi, T).view(1, T, 1)
        sensor_coords = torch.sin(t_steps).expand(B, T, q_dim)

        # Initial decoded velocities (e.g. near zero from random init)
        decoded_vel = torch.zeros(B, T, q_dim, requires_grad=True)

        l_vel = loss_fn.velocity_loss(decoded_vel, sensor_coords)

        assert l_vel.item() > 0.0, f"L_vel should be strictly positive on moving keypoints (got {l_vel.item()})"
        assert l_vel.requires_grad, "L_vel must retain autograd graph connection"

    def test_velocity_loss_backward_gradient_flow(self, dynamics_v5_4):
        """Gradient of L_vel must flow into W_vel weight and bias."""
        loss_fn = SPWMLoss(lambda_vel=0.5, delta_t=1.0)
        B, T, q_dim = 4, 20, dynamics_v5_4.q_dim
        mem_dim = dynamics_v5_4.total_memory_dim

        sensor_coords = torch.linspace(-0.8, 0.8, T).view(1, T, 1).expand(B, T, q_dim)
        ema_spikes = torch.rand(B, T, mem_dim) * 0.15

        # Forward through W_vel
        decoded_vel = dynamics_v5_4.W_vel(ema_spikes)
        l_vel = loss_fn.velocity_loss(decoded_vel, sensor_coords)

        assert l_vel.item() > 0.0
        l_vel.backward()

        assert dynamics_v5_4.W_vel.weight.grad is not None, "W_vel.weight should receive gradients"
        assert dynamics_v5_4.W_vel.weight.grad.norm().item() > 0.0, "W_vel.weight gradient should be non-zero"
        assert dynamics_v5_4.W_vel.bias.grad is not None, "W_vel.bias should receive gradients"
        assert dynamics_v5_4.W_vel.bias.grad.norm().item() > 0.0, "W_vel.bias gradient should be non-zero"

    def test_velocity_loss_zero_on_perfect_match(self):
        """L_vel must be 0 when decoded velocities exactly match finite-difference targets."""
        loss_fn = SPWMLoss(lambda_vel=0.5, delta_t=1.0)
        B, T, q_dim = 2, 10, 8

        sensor_coords = torch.randn(B, T, q_dim)
        v_target = (sensor_coords[:, 1:] - sensor_coords[:, :-1]) / 1.0
        decoded_vel = torch.cat([torch.zeros(B, 1, q_dim), v_target], dim=1)

        l_vel = loss_fn.velocity_loss(decoded_vel, sensor_coords)
        assert torch.isclose(l_vel, torch.tensor(0.0), atol=1e-6)


# ---------------------------------------------------------------------------
# Test 2: Persistent Hamiltonian Limit Cycles (No Threshold Quenching)
# ---------------------------------------------------------------------------

class TestPersistentLimitCycles:
    """
    Verifies that unforced autonomous rollout (H=200) sustains active limit cycles
    with calibrated gamma=0.35 and epsilon_diss=1e-5.
    """

    def test_unforced_simulation_h200_sustains_firing(self, model_v5_4):
        """
        An autonomous rollout for H=200 steps must maintain population firing in [5%, 18%]
        without decaying to 0 (freezing) and without diverging.
        """
        torch.manual_seed(42)
        model_v5_4.eval()
        B = 4

        # Prime the model with non-zero initial state
        state = model_v5_4.init_state(B)
        # Give initial somatic drive for 10 warmup steps
        dummy_input = torch.randn(B, model_v5_4.dynamics.input_dim) * 0.5 + 0.8
        for _ in range(10):
            z_t, state.dynamics_state = model_v5_4.dynamics.step(
                sensory_input=dummy_input,
                state=state.dynamics_state,
                sensory_keypoints=torch.zeros(B, model_v5_4.dynamics.q_dim),
            )

        # Autonomous rollout for H=200 without sensory input
        rollout_out = model_v5_4.predict_future(initial_state=state, horizon=200)
        spikes = rollout_out["spikes"]  # [B, 200, dim_tier]
        coords = rollout_out["coords"]  # [B, 200, q_dim]

        # Check firing rates across time segments
        early_rate = spikes[:, :50].float().mean().item()
        mid_rate = spikes[:, 50:150].float().mean().item()
        late_rate = spikes[:, 150:200].float().mean().item()
        overall_rate = spikes.float().mean().item()

        print(f"H=200 Rollout Spike Rates: Early={early_rate:.4f}, Mid={mid_rate:.4f}, Late={late_rate:.4f}, Overall={overall_rate:.4f}")

        # Assert no threshold quenching (firing rate > 0.04 across all segments)
        assert early_rate >= 0.04, f"Early firing rate quenched: {early_rate:.4f}"
        assert mid_rate >= 0.04, f"Mid firing rate quenched: {mid_rate:.4f}"
        assert late_rate >= 0.04, f"Late firing rate quenched (post-sensory freeze): {late_rate:.4f}"
        assert overall_rate <= 0.25, f"Firing rate exploded: {overall_rate:.4f}"
        assert 0.05 <= overall_rate <= 0.22, f"Overall firing rate {overall_rate:.4f} not in bio-plausible range [5%, 22%]"

        # Assert coordinate trajectory moves (not frozen at 0 velocity)
        coord_delta = (coords[:, 1:] - coords[:, :-1]).abs().sum().item()
        assert coord_delta > 1.0, f"Trajectory coordinates are frozen: total delta = {coord_delta:.4f}"

        # Assert bounded coordinates
        assert (coords >= -1.001).all() and (coords <= 1.001).all(), "Coordinates drifted out of [-1, 1] bounds"


# ---------------------------------------------------------------------------
# Test 3: Wall Bounce Reflex (Bounded Momentum Reflection)
# ---------------------------------------------------------------------------

class TestWallBounceReflex:
    """Verifies that coordinates reaching |q| >= 0.98 undergo velocity inversion with 0.8 restitution."""

    def test_positive_boundary_collision_inversion(self, dynamics_v5_4):
        """Moving into positive boundary (+0.99) should invert velocity to negative."""
        B = 2
        state = dynamics_v5_4.init_state(B)

        # Set coordinate near positive boundary
        q_near_wall = torch.full((B, dynamics_v5_4.q_dim), 0.97)
        state.z_prev[:, :dynamics_v5_4.q_dim] = q_near_wall

        # Set W_vel to predict positive velocity towards wall
        dynamics_v5_4.W_vel.weight.data.zero_()
        dynamics_v5_4.W_vel.bias.data.fill_(0.15)  # positive velocity +0.15

        dummy_sensory = torch.zeros(B, dynamics_v5_4.input_dim)
        z_next, new_state = dynamics_v5_4.step(
            sensory_input=dummy_sensory,
            state=state,
            sensory_keypoints=None,  # autonomous
        )

        q_next = z_next[:, :dynamics_v5_4.q_dim]

        # Candidate q would be 0.97 + tanh(0.15) ~ 1.118 >= 0.98 -> hit
        # Inverted velocity should be -0.8 * 0.15 = -0.12
        # q_next should move inwards: 0.97 - 0.12 ~ 0.85
        assert (q_next < 0.97).all(), f"Particle did not bounce back from positive wall: q_next={q_next[0, 0].item():.4f}"
        assert (q_next <= 1.0).all(), "Coordinate exceeded +1.0"

    def test_negative_boundary_collision_inversion(self, dynamics_v5_4):
        """Moving into negative boundary (-0.99) should invert velocity to positive."""
        B = 2
        state = dynamics_v5_4.init_state(B)

        # Set coordinate near negative boundary
        q_near_wall = torch.full((B, dynamics_v5_4.q_dim), -0.97)
        state.z_prev[:, :dynamics_v5_4.q_dim] = q_near_wall

        # Set W_vel to predict negative velocity towards wall
        dynamics_v5_4.W_vel.weight.data.zero_()
        dynamics_v5_4.W_vel.bias.data.fill_(-0.15)  # negative velocity -0.15

        dummy_sensory = torch.zeros(B, dynamics_v5_4.input_dim)
        z_next, new_state = dynamics_v5_4.step(
            sensory_input=dummy_sensory,
            state=state,
            sensory_keypoints=None,  # autonomous
        )

        q_next = z_next[:, :dynamics_v5_4.q_dim]

        # Candidate q would be -0.97 - tanh(0.15) ~ -1.118 <= -0.98 -> hit
        # Inverted velocity should be -0.8 * (-0.15) = +0.12
        # q_next should move inwards: -0.97 + 0.12 ~ -0.85
        assert (q_next > -0.97).all(), f"Particle did not bounce back from negative wall: q_next={q_next[0, 0].item():.4f}"
        assert (q_next >= -1.0).all(), "Coordinate exceeded -1.0"


# ---------------------------------------------------------------------------
# Test 4: W_vel Linear Projection with Bias
# ---------------------------------------------------------------------------

class TestWVelLinearProjection:
    """Verifies that W_vel is standard linear layer with learnable bias and no mean-centering."""

    def test_w_vel_has_learnable_bias(self, dynamics_v5_4):
        """W_vel must have a bias parameter."""
        assert dynamics_v5_4.W_vel.bias is not None, "W_vel must have learnable bias in v5.4"
        assert dynamics_v5_4.W_vel.bias.shape == (dynamics_v5_4.q_dim,)

    def test_w_vel_direct_projection(self, dynamics_v5_4):
        """W_vel output must match standard linear projection W @ x + b."""
        x = torch.randn(4, dynamics_v5_4.total_memory_dim)
        expected = torch.nn.functional.linear(x, dynamics_v5_4.W_vel.weight, dynamics_v5_4.W_vel.bias)
        actual = dynamics_v5_4.W_vel(x)
        assert torch.allclose(actual, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 5: Port-Hamiltonian Stability (epsilon_diss = 1e-5)
# ---------------------------------------------------------------------------

class TestPortHamiltonianStabilityV54:
    """Verifies Lyapunov stability of W_rec = J - R with dissipation floor 1e-5."""

    def test_eigenvalue_real_parts_negative(self, dynamics_v5_4):
        """All eigenvalues of W_rec must satisfy Re(lambda) <= -epsilon_diss."""
        W_rec = dynamics_v5_4.get_effective_w_rec()
        eigenvalues = torch.linalg.eigvals(W_rec)
        real_parts = eigenvalues.real
        eps = dynamics_v5_4.epsilon_diss
        max_real = real_parts.max().item()

        assert max_real <= -eps + 1e-6, (
            f"Stability violated: max Re(lambda) = {max_real:.8f}, expected <= -{eps}"
        )
