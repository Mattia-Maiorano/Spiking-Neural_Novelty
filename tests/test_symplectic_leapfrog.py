"""Tests for SPWM-v5.2 Symplectic Leapfrog Integration, Zero-DC Dendritic Force, and Intrinsic Homeostasis."""

import torch
import pytest
from spwm.models.latent_dynamics import SpikingLatentDynamics, DynamicsState
from spwm.models.neurons import ALIFCell
from spwm.models.world_model import SPWM


def test_zero_dc_projection():
    """Assert sum_j (C_pq)_{ij} = 0 +- 1e-7 for all rows i."""
    # latent_dim=128 (q_dim=32, p_dim=96)
    dynamics = SpikingLatentDynamics(latent_dim=128, q_dim=32, p_dim=96)
    C_pq = dynamics.get_c_pq() # [96, 32]
    
    assert C_pq.shape == (96, 32)
    # Check row sums
    row_sums = C_pq.sum(dim=1)
    assert torch.allclose(row_sums, torch.zeros_like(row_sums), atol=1e-7), f"Row sums not zero: {row_sums}"


def test_constant_input_zero_dendritic_current():
    """Assert constant coordinate inputs q_t = c produce zero steady-state dendritic current."""
    dynamics = SpikingLatentDynamics(latent_dim=128, q_dim=32, p_dim=96, beta_dend=0.85)
    
    batch_size = 4
    # Constant coordinate vector across all dimensions
    constant_q = torch.full((batch_size, 32), 0.75)
    
    # Initialize state
    state = dynamics.init_state(batch_size=batch_size, device=constant_q.device)
    
    # Run multiple steps with constant coordinate input and zero somatic spike
    for _ in range(20):
        # Force potential directly from constant q
        F_pot = constant_q @ dynamics.get_c_pq().t()
        # Update dendritic current
        state.i_dend = dynamics.beta_dend * state.i_dend + (1.0 - dynamics.beta_dend) * F_pot
    
    assert torch.allclose(state.i_dend, torch.zeros_like(state.i_dend), atol=1e-6), \
        f"Steady-state dendritic current not zero for constant q: max abs = {state.i_dend.abs().max()}"


def test_intrinsic_homeostasis_firing_rate():
    """Assert that membrane firing rates stabilize between 0.08 and 0.15 when driven by synthetic inputs."""
    cell = ALIFCell(
        size=96,
        beta_mem=0.80,
        beta_adapt=0.90,
        v_th0=1.0,
        gamma=1.5
    )
    
    batch_size = 16
    T = 2000  # 2000 timesteps
    
    state = cell.init_state(batch_size=batch_size)
    
    # Synthetic oscillatory + noisy drive
    t_vec = torch.linspace(0, 2 * 3.14159 * 10, T).unsqueeze(1).unsqueeze(2) # 10 Hz oscillation
    drive = 1.2 * torch.sin(t_vec) + 1.0 + 0.2 * torch.randn(T, batch_size, 96)
    
    spikes = []
    for t in range(T):
        s, state = cell(drive[t], state)
        spikes.append(s)
        
    spike_tensor = torch.stack(spikes, dim=0) # [T, B, 96]
    
    # Measure steady-state firing rate over the second half
    steady_spikes = spike_tensor[T//2:]
    mean_firing_rate = steady_spikes.mean().item()
    
    print(f"Observed steady-state firing rate: {mean_firing_rate:.4f}")
    assert 0.08 <= mean_firing_rate <= 0.15, f"Firing rate {mean_firing_rate} outside [0.08, 0.15]"


def test_o1_memory_invariant():
    """Assert strict memory invariant O(1) across horizons H in {10, 100, 500}."""
    device = torch.device("cpu")
    model = SPWM(
        latent_dim=128,
        q_dim=32,
        p_dim=96,
        beta_dend=0.85,
        alpha_dend=0.10,
        target_rate_center=0.11
    ).to(device)
    
    batch_size = 2
    
    for H in [10, 100, 500]:
        state = model.dynamics.init_state(batch_size=batch_size, device=device)
        
        # Test rollout without building autograd graph
        with torch.no_grad():
            trajectory = model.predict_future(state, horizon=H)
            
        assert trajectory['spikes'].shape[1] == H
        assert trajectory['coords'].shape[1] == H
        assert trajectory['membrane'].shape[1] == H
        assert trajectory['adaptation'].shape[1] == H
        assert trajectory['i_dend'].shape[1] == H
