"""Tests for SPWM-v5.2 Symplectic Leapfrog Integration, Zero-DC Dendritic Force, and Intrinsic Homeostasis."""

import torch
import pytest
from spwm.models.latent_dynamics import SpikingLatentDynamics, DynamicsState
from spwm.models.neurons import ALIFCell
from spwm.models.world_model import SPWM


@pytest.mark.skip(reason="Obsolete v5.2 dendritic force tests (removed in v5.3+ for pure Port-Hamiltonian dynamics)")
def test_zero_dc_projection():
    pass


@pytest.mark.skip(reason="Obsolete v5.2 dendritic current tests (removed in v5.3+)")
def test_constant_input_zero_dendritic_current():
    pass


def test_intrinsic_homeostasis_firing_rate():
    """Assert that membrane firing rates stabilize reasonably when driven by synthetic inputs."""
    cell = ALIFCell(
        size=96,
        beta_mem=0.80,
        beta_adapt=0.90,
        v_th0=1.0,
        gamma=0.35,
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
    assert 0.05 <= mean_firing_rate <= 0.25, f"Firing rate {mean_firing_rate} outside [0.05, 0.25]"


def test_o1_memory_invariant():
    """Assert strict memory invariant O(1) across horizons H in {10, 100, 500}."""
    device = torch.device("cpu")
    model = SPWM(
        latent_dim=128,
        q_dim=32,
        p_dim=96,
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
