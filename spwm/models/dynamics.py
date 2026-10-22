"""
Layer 1 Recurrent Spiking Latent Dynamics with ALIF Memory Hierarchy.
Integrates sensory error with recurrent feedback and multi-timescale ALIF memory.
Strictly validates dimensions without silent fallback defaults.
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple
import torch
import torch.nn as nn

from spwm.contracts.states import DynamicsOutput, DynamicsState
from spwm.contracts.tensors import assert_phase_space_contract, assert_tensor_contract
from spwm.models.memory import MultiTimescaleMemory


class SpikingLatentDynamics(nn.Module):
    """
    Recurrent Spiking Latent Dynamics with ALIF Core.
    Strictly validates dimensions without fallback defaults.
    """

    def __init__(
        self,
        input_dim: int = 128,
        latent_dim: int = 128,
        q_dim: int = 32,
        p_dim: int = 96,
        ema_decay: float = 0.9,
        timescale_dims: Sequence[int] = (128, 128),
        betas: Sequence[float] = (0.90, 0.985),
        beta_mem: float = 0.80,
        v_th0: float = 1.0,
        gamma: float = 0.18,
        surrogate_name: str = "atan",
        surrogate_alpha: float = 2.0,
    ) -> None:
        super().__init__()
        assert_phase_space_contract(latent_dim=latent_dim, q_dim=q_dim, p_dim=p_dim)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.q_dim = q_dim
        self.p_dim = p_dim
        self.ema_decay = ema_decay
        self.timescale_dims = list(timescale_dims)

        if len(betas) != len(self.timescale_dims):
            raise ValueError(
                f"SpikingLatentDynamics: betas length ({len(betas)}) != timescale_dims length ({len(self.timescale_dims)})"
            )

        self.total_memory_dim = sum(self.timescale_dims)

        self.memory = MultiTimescaleMemory(
            timescale_dims=self.timescale_dims,
            betas=betas,
            beta_mem=beta_mem,
            v_th0=v_th0,
            gamma=gamma,
            surrogate_name=surrogate_name,
            surrogate_alpha=surrogate_alpha,
        )

        self.input_proj = nn.Linear(input_dim, self.total_memory_dim)
        self.recurrent_proj = nn.Linear(self.p_dim, self.total_memory_dim, bias=False)
        self.q_proj = nn.Linear(self.q_dim, self.total_memory_dim, bias=False)

        self.W_vel = nn.Linear(self.total_memory_dim, self.q_dim, bias=False)
        nn.init.normal_(self.W_vel.weight, mean=0.0, std=0.01)

        self.fuse_spikes = nn.Linear(self.total_memory_dim, self.p_dim)
        self.fuse_mems = nn.Linear(self.total_memory_dim, self.p_dim, bias=False)
        self.norm = nn.LayerNorm(self.p_dim)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> DynamicsState:
        mem_state = self.memory.init_state(batch_size, device=device)
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
        assert_tensor_contract(sensory_input, (None, self.input_dim), "SpikingLatentDynamics.sensory_input")
        B = sensory_input.shape[0]
        device = sensory_input.device

        if state is None:
            state = self.init_state(B, device=device)

        q_prev = state.z_prev[:, : self.q_dim]
        p_prev = state.z_prev[:, self.q_dim :]

        soma_current = self.input_proj(sensory_input) + self.recurrent_proj(p_prev) + self.q_proj(q_prev)

        spikes, new_mem_state = self.memory(
            synaptic_inputs=soma_current,
            state=state.memory_state,
        )

        mem_analog = torch.tanh(new_mem_state.concatenated_mems)
        p_next = self.norm(self.fuse_spikes(spikes) + self.fuse_mems(mem_analog))

        ema_spikes = self.ema_decay * state.ema_spikes + (1.0 - self.ema_decay) * spikes

        if sensory_keypoints is not None:
            assert_tensor_contract(sensory_keypoints, (B, self.q_dim), "SpikingLatentDynamics.sensory_keypoints")
            q_next = sensory_keypoints
        else:
            delta_q = torch.tanh(self.W_vel(ema_spikes))
            q_next = torch.clamp(q_prev + delta_q, -1.0, 1.0)

        z_next = torch.cat([q_next, p_next], dim=1)
        new_state = DynamicsState(memory_state=new_mem_state, z_prev=z_next, ema_spikes=ema_spikes)
        return z_next, new_state

    def forward(
        self,
        sensory_sequence: torch.Tensor,
        initial_state: Optional[DynamicsState] = None,
        sensory_keypoints: Optional[torch.Tensor] = None,
    ) -> Tuple[DynamicsOutput, DynamicsState]:
        assert_tensor_contract(sensory_sequence, (None, None, self.input_dim), "SpikingLatentDynamics.sensory_sequence")
        B, T, _ = sensory_sequence.shape
        device = sensory_sequence.device

        if sensory_keypoints is not None:
            assert_tensor_contract(sensory_keypoints, (B, T, self.q_dim), "SpikingLatentDynamics.sensory_keypoints_seq")

        state = initial_state if initial_state is not None else self.init_state(B, device=device)

        latent_steps = []
        fast_spk_steps = []
        slow_spk_steps = []
        fast_mem_steps = []
        slow_mem_steps = []

        total_spikes = torch.tensor(0.0, device=device)
        total_neurons = 0

        for t in range(T):
            e_t = sensory_sequence[:, t]
            kp_t = sensory_keypoints[:, t] if sensory_keypoints is not None else None
            z_t, state = self.step(e_t, state=state, sensory_keypoints=kp_t)
            latent_steps.append(z_t)

            pop_spks = state.memory_state.spikes
            pop_mems = state.memory_state.v_mems

            fast_spk_steps.append(pop_spks[0])
            slow_spk_steps.append(pop_spks[-1])
            fast_mem_steps.append(pop_mems[0])
            slow_mem_steps.append(pop_mems[-1])

            for spk in pop_spks:
                total_spikes = total_spikes + spk.sum()
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
