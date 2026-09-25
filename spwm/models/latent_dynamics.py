"""
Recurrent Spiking Latent Dynamics with Multi-Timescale Memory.
Updates latent state z_t -> z_(t+1) via recurrent synaptic connections,
multi-timescale membrane dynamics, and spiking activity.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Sequence
import torch
import torch.nn as nn

from spwm.models.memory import MultiTimescaleMemory, MultiTimescaleState


@dataclass
class DynamicsState:
    """Recurrent state container for SpikingLatentDynamics."""
    memory_state: MultiTimescaleState
    z_prev: torch.Tensor  # [B, latent_dim]


@dataclass
class DynamicsOutput:
    """Output container for dynamics forward pass."""
    latent_states: torch.Tensor  # [B, T, latent_dim]
    fast_spikes: torch.Tensor  # [B, T, fast_dim]
    slow_spikes: torch.Tensor  # [B, T, slow_dim]
    fast_mems: torch.Tensor  # [B, T, fast_dim]
    slow_mems: torch.Tensor  # [B, T, slow_dim]
    mean_spike_rate: torch.Tensor


class SpikingLatentDynamics(nn.Module):
    """
    Recurrent Spiking Latent Dynamics Module.
    Combines incoming sensory input with recurrent latent feedback to drive
    fast and slow spiking neural populations.
    """

    def __init__(
        self,
        input_dim: int = 128,
        latent_dim: int = 128,
        timescale_dims: Sequence[int] = (64, 64),
        betas: Sequence[float] = (0.8, 0.98),
        threshold: float = 1.0,
        reset_mechanism: str = "hard",
        surrogate_name: str = "atan",
        surrogate_alpha: float = 2.0,
        learnable_betas: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.timescale_dims = list(timescale_dims)
        self.total_memory_dim = sum(self.timescale_dims)

        # Multi-timescale spiking memory
        self.memory = MultiTimescaleMemory(
            timescale_dims=self.timescale_dims,
            betas=betas,
            threshold=threshold,
            reset_mechanism=reset_mechanism,
            surrogate_name=surrogate_name,
            surrogate_alpha=surrogate_alpha,
            learnable_betas=learnable_betas,
        )

        # Synaptic projections: sensory input + recurrent feedback -> memory currents
        self.input_proj = nn.Linear(input_dim, self.total_memory_dim)
        self.recurrent_proj = nn.Linear(latent_dim, self.total_memory_dim, bias=False)

        # Fusion layer: transforms multi-timescale spikes & membranes into latent state z_t
        # Dimension is 2 * total_memory_dim (spikes + tanh(membranes))
        self.fuse_spikes = nn.Linear(self.total_memory_dim, latent_dim)
        self.fuse_mems = nn.Linear(self.total_memory_dim, latent_dim, bias=False)
        self.norm = nn.LayerNorm(latent_dim)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> DynamicsState:
        """Initializes quiescent dynamics state."""
        mem_state = self.memory.init_state(batch_size, device=device)
        z_prev = torch.zeros(batch_size, self.latent_dim, device=device)
        return DynamicsState(memory_state=mem_state, z_prev=z_prev)

    def step(
        self,
        sensory_input: torch.Tensor,
        state: Optional[DynamicsState] = None,
    ) -> Tuple[torch.Tensor, DynamicsState]:
        """
        Executes a single recurrent dynamical step:
            e_t, z_(t-1) -> I_t -> MultiTimescaleMemory -> z_t
        sensory_input: [B, input_dim]
        state: previous DynamicsState
        Returns: (z_t, new_state)
        """
        B = sensory_input.shape[0]
        device = sensory_input.device

        if state is None:
            state = self.init_state(B, device=device)

        # Compute synaptic current
        current_input = self.input_proj(sensory_input) + self.recurrent_proj(state.z_prev)

        # Update spiking memory
        spikes, new_mem_state = self.memory(current_input, state.memory_state)

        # Combine spikes and analog membrane charge into unified latent state z_t
        mem_analog = torch.tanh(new_mem_state.concatenated_mems)
        z_t = self.norm(self.fuse_spikes(spikes) + self.fuse_mems(mem_analog))

        new_state = DynamicsState(memory_state=new_mem_state, z_prev=z_t)
        return z_t, new_state

    def forward(
        self,
        sensory_sequence: torch.Tensor,
        initial_state: Optional[DynamicsState] = None,
    ) -> Tuple[DynamicsOutput, DynamicsState]:
        """
        Unrolls recurrent dynamics over an input sequence [B, T, input_dim].
        Returns DynamicsOutput and final state.
        """
        B, T, _ = sensory_sequence.shape
        device = sensory_sequence.device

        if initial_state is None:
            state = self.init_state(B, device=device)
        else:
            state = initial_state

        latent_steps = []
        fast_spk_steps = []
        slow_spk_steps = []
        fast_mem_steps = []
        slow_mem_steps = []

        total_spikes = 0
        total_neurons = 0

        for t in range(T):
            e_t = sensory_sequence[:, t]
            z_t, state = self.step(e_t, state)
            latent_steps.append(z_t)

            # Record internal population activities
            fast_spk = state.memory_state.spikes[0]
            slow_spk = state.memory_state.spikes[1]
            fast_mem = state.memory_state.v_mems[0]
            slow_mem = state.memory_state.v_mems[1]

            fast_spk_steps.append(fast_spk)
            slow_spk_steps.append(slow_spk)
            fast_mem_steps.append(fast_mem)
            slow_mem_steps.append(slow_mem)

            total_spikes += (fast_spk.sum() + slow_spk.sum())
            total_neurons += (fast_spk.numel() + slow_spk.numel())

        latent_seq = torch.stack(latent_steps, dim=1)  # [B, T, latent_dim]
        fast_spikes_seq = torch.stack(fast_spk_steps, dim=1)
        slow_spikes_seq = torch.stack(slow_spk_steps, dim=1)
        fast_mems_seq = torch.stack(fast_mem_steps, dim=1)
        slow_mems_seq = torch.stack(slow_mem_steps, dim=1)

        mean_spike_rate = total_spikes / max(1, total_neurons)

        output = DynamicsOutput(
            latent_states=latent_seq,
            fast_spikes=fast_spikes_seq,
            slow_spikes=slow_spikes_seq,
            fast_mems=fast_mems_seq,
            slow_mems=slow_mems_seq,
            mean_spike_rate=mean_spike_rate,
        )

        return output, state
