"""
Recurrent Spiking Latent Dynamics with ALIF Memory Hierarchy.
Integrates error-routed sensory input with recurrent feedback and dual-timescale
ALIF populations (Reactive 50%, Deep Context 50%).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Sequence, Union
import torch
import torch.nn as nn

from spwm.models.memory import MultiTimescaleMemory, MultiTimescaleState, compute_tier_dims


@dataclass
class DynamicsState:
    """Recurrent state container for SpikingLatentDynamics."""
    memory_state: MultiTimescaleState
    z_prev: torch.Tensor  # [B, latent_dim] (concatenated q and p)
    ema_spikes: torch.Tensor  # [B, total_memory_dim] EMA of spikes


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
    Recurrent Spiking Latent Dynamics with ALIF Core (SPWM-v3).
    """

    def __init__(
        self,
        input_dim: int = 128,
        latent_dim: int = 128,
        q_dim: Optional[int] = None,
        p_dim: Optional[int] = None,
        ema_decay: float = 0.9,
        timescale_dims: Optional[Sequence[int]] = None,
        betas: Sequence[float] = (0.90, 0.985),
        beta_mem: float = 0.80,
        v_th0: float = 1.0,
        gamma: float = 0.18,
        surrogate_name: str = "atan",
        surrogate_alpha: float = 2.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim  # total latent dim (q + p)

        if q_dim is None and p_dim is None:
            q_dim = latent_dim // 4
            p_dim = latent_dim - q_dim
        elif q_dim is None:
            q_dim = latent_dim - p_dim
        elif p_dim is None:
            p_dim = latent_dim - q_dim

        self.q_dim = q_dim
        self.p_dim = p_dim
        self.ema_decay = ema_decay
        assert q_dim + p_dim == latent_dim, f"q_dim ({q_dim}) + p_dim ({p_dim}) must equal latent_dim ({latent_dim})"

        if timescale_dims is not None:
            self.timescale_dims = list(timescale_dims)
        else:
            self.timescale_dims = list(compute_tier_dims(latent_dim, num_tiers=2))

        self.total_memory_dim = sum(self.timescale_dims)

        # Multi-timescale ALIF memory
        self.memory = MultiTimescaleMemory(
            timescale_dims=self.timescale_dims,
            betas=betas if len(betas) == len(self.timescale_dims) else None,
            beta_mem=beta_mem,
            v_th0=v_th0,
            gamma=gamma,
            surrogate_name=surrogate_name,
            surrogate_alpha=surrogate_alpha,
        )

        # Synaptic projections: sensory error + recurrent feedback -> somatic currents
        self.input_proj = nn.Linear(input_dim, self.total_memory_dim)
        # Recurrent projection now receives only p component
        self.recurrent_proj = nn.Linear(self.p_dim, self.total_memory_dim, bias=False)
        # Projection from q component to somatic current
        self.q_proj = nn.Linear(self.q_dim, self.total_memory_dim, bias=False)

        # Linear map from EMA of spikes to q velocity update
        self.W_vel = nn.Linear(self.total_memory_dim, self.q_dim, bias=False)
        nn.init.normal_(self.W_vel.weight, mean=0.0, std=0.01)

        # Latent fusion layer now produces only p component
        self.fuse_spikes = nn.Linear(self.total_memory_dim, self.p_dim)
        self.fuse_mems = nn.Linear(self.total_memory_dim, self.p_dim, bias=False)
        self.norm = nn.LayerNorm(self.p_dim)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> DynamicsState:
        """Initializes quiescent dynamics state."""
        mem_state = self.memory.init_state(batch_size, device=device)
        # Initialize q and p to zeros
        q0 = torch.zeros(batch_size, self.q_dim, device=device)
        p0 = torch.zeros(batch_size, self.p_dim, device=device)
        z0 = torch.cat([q0, p0], dim=1)
        ema0 = torch.zeros(batch_size, self.total_memory_dim, device=device)
        return DynamicsState(memory_state=mem_state, z_prev=z0, ema_spikes=ema0)

    def step(
        self,
        sensory_input: torch.Tensor,
        state: Optional[DynamicsState] = None,
        sensory_keypoints: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, DynamicsState]:
        """
        Executes a single recurrent dynamical step:
            ϵ_t, z_(t-1) -> I_soma -> MultiTimescaleMemory (ALIF) -> z_t

        In observation mode (sensory_keypoints is not None):
            q_t is set directly from SpatialSoftmax keypoints (bounded in [-1, 1]).
        In autonomous mode (sensory_keypoints is None):
            q_t evolves via bounded velocity update: q_t = clamp(q_(t-1) + tanh(W_vel(ema_spikes)), -1, 1).
        """
        B = sensory_input.shape[0]
        device = sensory_input.device

        if state is None:
            state = self.init_state(B, device=device)

        # Split previous concatenated latent into q and p
        q_prev = state.z_prev[:, :self.q_dim]
        p_prev = state.z_prev[:, self.q_dim:]

        # 1. Somatic synaptic current receives sensory input, recurrent from p, and q projection
        soma_current = self.input_proj(sensory_input) + self.recurrent_proj(p_prev) + self.q_proj(q_prev)

        # 2. Multi-timescale ALIF memory update (operates on p component)
        spikes, new_mem_state = self.memory(
            synaptic_inputs=soma_current,
            state=state.memory_state,
        )

        # 3. Compute new p latent via fusion of spikes and analog membranes
        mem_analog = torch.tanh(new_mem_state.concatenated_mems)
        p_next = self.norm(self.fuse_spikes(spikes) + self.fuse_mems(mem_analog))

        # 4. Update EMA of spikes (per batch element)
        ema_spikes = self.ema_decay * state.ema_spikes + (1.0 - self.ema_decay) * spikes

        # 5. Dual-mode update for q:
        # - Observation mode: directly clamp/assign from raw SpatialSoftmax keypoints in [-1, 1]
        # - Autonomous mode: integrate bounded delta from spike EMA
        if sensory_keypoints is not None:
            q_next = sensory_keypoints
        else:
            delta_q = torch.tanh(self.W_vel(ema_spikes))
            q_next = torch.clamp(q_prev + delta_q, -1.0, 1.0)

        # 6. Concatenate updated q and p to form next latent
        z_next = torch.cat([q_next, p_next], dim=1)

        new_state = DynamicsState(memory_state=new_mem_state, z_prev=z_next, ema_spikes=ema_spikes)
        return z_next, new_state

    def forward(
        self,
        sensory_sequence: torch.Tensor,
        initial_state: Optional[DynamicsState] = None,
    ) -> Tuple[DynamicsOutput, DynamicsState]:
        """Unrolls recurrent dynamics over an input sequence [B, T, input_dim]."""
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
            z_t, state = self.step(e_t, state=state)
            latent_steps.append(z_t)

            pop_spks = state.memory_state.spikes
            pop_mems = state.memory_state.v_mems

            fast_spk = pop_spks[0]
            slow_spk = pop_spks[-1]

            fast_mem = pop_mems[0]
            slow_mem = pop_mems[-1]

            fast_spk_steps.append(fast_spk)
            slow_spk_steps.append(slow_spk)
            fast_mem_steps.append(fast_mem)
            slow_mem_steps.append(slow_mem)

            for spk in pop_spks:
                total_spikes += spk.sum()
                total_neurons += spk.numel()

        latent_seq = torch.stack(latent_steps, dim=1)
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
