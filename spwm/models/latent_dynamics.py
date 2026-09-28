"""
Recurrent Spiking Latent Dynamics with ALIF Memory Hierarchy.
SPWM-v5.4: Persistent Hamiltonian Limit Cycles with Active Velocity Supervision.

Key changes from v5.3:
  - Calibrated ALIF threshold homeostasis: gamma_adapt = 0.35 (prevents post-sensory quenching).
  - Dissipation floor lowered to epsilon_diss = 1e-5 (near-lossless Hamiltonian orbits).
  - Standard linear velocity decoding with learnable bias: v_t = W_vel @ s_bar_p + b_vel (removed destructive mean-centering).
  - Autonomous Wall Bounce Reflex: momentum reflection v_k <- -0.8 * v_k at boundaries |q| >= 0.98.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Sequence, Union
import torch
import torch.nn as nn

from spwm.models.memory import MultiTimescaleMemory, MultiTimescaleState, compute_tier_dims


@dataclass
class DynamicsState:
    """Recurrent state container for SpikingLatentDynamics (SPWM-v5.3)."""
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
    Recurrent Spiking Latent Dynamics with ALIF Core (SPWM-v5.3).

    Port-Hamiltonian recurrence (pure, no somatic coordinate injection):
        W_rec = J - R,  J = 0.5*(S - S^T),  R = diag(softplus(r) + eps_diss)
        I_soma,t = W_rec @ s_bar_p,t

    Zero-mean velocity decoding (eliminates DC bias):
        s_tilde_p = s_bar_p - mean(s_bar_p, dim=-1, keepdim=True)
        v_t = W_vel @ s_tilde_p

    Coordinate update (autonomous rollout):
        q_{t+1} = clamp(q_t + delta_t * tanh(v_t), -1, 1)

    Observation mode:
        q_t = q_sensor,t
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
        gamma: float = 0.35,
        surrogate_name: str = "atan",
        surrogate_alpha: float = 2.0,
        delta_t: float = 1.0,
        epsilon_diss: float = 1e-5,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim  # total latent dim (q + p)
        self.delta_t = delta_t
        self.epsilon_diss = epsilon_diss

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
        assert q_dim + p_dim == latent_dim, (
            f"q_dim ({q_dim}) + p_dim ({p_dim}) must equal latent_dim ({latent_dim})"
        )

        if timescale_dims is not None:
            self.timescale_dims = list(timescale_dims)
        else:
            self.timescale_dims = list(compute_tier_dims(self.p_dim, num_tiers=2))

        self.total_memory_dim = sum(self.timescale_dims)

        # Multi-timescale ALIF memory (with calibrated homeostasis via gamma=0.35)
        self.memory = MultiTimescaleMemory(
            timescale_dims=self.timescale_dims,
            betas=betas if len(betas) == len(self.timescale_dims) else None,
            beta_mem=beta_mem,
            v_th0=v_th0,
            gamma=gamma,
            surrogate_name=surrogate_name,
            surrogate_alpha=surrogate_alpha,
        )

        # Synaptic projection: sensory error -> somatic currents
        self.input_proj = nn.Linear(input_dim, self.total_memory_dim)

        # Port-Hamiltonian parameterization for recurrent dynamics
        # W_rec = J - R where J is skew-symmetric (conservative) and R is positive diagonal (dissipative)
        self.S = nn.Parameter(torch.randn(self.p_dim, self.p_dim))  # unconstrained base matrix
        self.r = nn.Parameter(torch.randn(self.p_dim))  # log-damping vector
        self.rec_proj = nn.Linear(self.p_dim, self.total_memory_dim, bias=False)

        # Direct velocity projection: W_vel maps s_bar_p -> coordinate velocity (with learnable bias)
        self.W_vel = nn.Linear(self.total_memory_dim, self.q_dim, bias=True)
        nn.init.normal_(self.W_vel.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.W_vel.bias)

        # Latent fusion layer produces p component
        self.fuse_spikes = nn.Linear(self.total_memory_dim, self.p_dim)
        self.fuse_mems = nn.Linear(self.total_memory_dim, self.p_dim, bias=False)
        self.norm = nn.LayerNorm(self.p_dim)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> DynamicsState:
        """Initializes quiescent dynamics state."""
        mem_state = self.memory.init_state(batch_size, device=device)
        q0 = torch.zeros(batch_size, self.q_dim, device=device)
        p0 = torch.zeros(batch_size, self.p_dim, device=device)
        z0 = torch.cat([q0, p0], dim=1)
        ema0 = torch.zeros(batch_size, self.total_memory_dim, device=device)
        return DynamicsState(memory_state=mem_state, z_prev=z0, ema_spikes=ema0)

    def _effective_recurrent_weight(self) -> torch.Tensor:
        """Compute skew-symmetric J and positive diagonal R, return W = J - R.
        J = 0.5 * (S - S.T) ensures skew-symmetry (conservative Hamiltonian flow).
        R = diag(softplus(r) + epsilon_diss) ensures positive definiteness (dissipation).
        Re(lambda(W_rec)) <= -epsilon_diss for all eigenvalues -> Lyapunov stable.
        """
        J = 0.5 * (self.S - self.S.T)
        R = torch.diag(torch.nn.functional.softplus(self.r) + self.epsilon_diss)
        return J - R

    def _effective_recurrent_current(self, p: torch.Tensor) -> torch.Tensor:
        """Apply the effective recurrent weight to p: I_soma = W_rec @ p."""
        weight = self._effective_recurrent_weight()
        return torch.nn.functional.linear(p, weight)

    def get_effective_w_rec(self) -> torch.Tensor:
        """Public accessor for the current effective recurrent weight matrix (J - R)."""
        return self._effective_recurrent_weight()

    def step(
        self,
        sensory_input: torch.Tensor,
        state: Optional[DynamicsState] = None,
        sensory_keypoints: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, DynamicsState]:
        """
        Executes a single recurrent dynamical step (SPWM-v5.4 Pure Port-Hamiltonian Flow):

        1. Momentum Evaluation (Port-Hamiltonian):
           I_soma,t = I_sensory + W_rec s_bar_p,t
           p_(t+1), s_p_(t+1) = ALIF_Step(p_t, I_soma,t)
           s_bar_p_(t+1) = ema_decay * s_bar_p,t + (1 - ema_decay) * s_p_(t+1)

        2. Coordinate Update (Dual-mode):
           Observation: q_t = q_sensor,t
           Autonomous:  v_t = W_vel @ s_bar_p + b_vel
                        Bounded Momentum Reflection at walls (|q| >= 0.98 -> v_k = -0.8 * v_k)
                        q_{t+1} = clamp(q_t + delta_t * tanh(v_t), -1, 1)
        """
        B = sensory_input.shape[0]
        device = sensory_input.device

        if state is None:
            state = self.init_state(B, device=device)

        # Split previous concatenated latent into q and p
        q_prev = state.z_prev[:, :self.q_dim]
        p_prev = state.z_prev[:, self.q_dim:]

        # Phase 1: Momentum Evaluation under pure Port-Hamiltonian field
        recurrent_p = self._effective_recurrent_current(p_prev)
        recurrent_current = self.rec_proj(recurrent_p)
        sensory_current = self.input_proj(sensory_input)

        # I_soma = I_sensory + W_rec s_bar_p  (no dendritic current, no coordinate injection)
        soma_current = sensory_current + recurrent_current

        spikes, new_mem_state = self.memory(
            synaptic_inputs=soma_current,
            state=state.memory_state,
        )

        mem_analog = torch.tanh(new_mem_state.concatenated_mems)
        p_next = self.norm(self.fuse_spikes(spikes) + self.fuse_mems(mem_analog))

        # Update filtered spike train (s_bar_p)
        ema_spikes = self.ema_decay * state.ema_spikes + (1.0 - self.ema_decay) * spikes

        # Phase 2: Coordinate update (Dual-mode: sensory observation vs autonomous rollout)
        if sensory_keypoints is not None:
            # Observation mode: ground-truth coordinate injection
            q_next = sensory_keypoints
        else:
            # Autonomous mode: standard linear projection with learnable bias (v5.4)
            v_t = self.W_vel(ema_spikes)

            # Wall Bounce Reflex / Bounded Momentum Reflection (Section 2.C)
            q_cand = q_prev + self.delta_t * torch.tanh(v_t)
            hit_pos = (q_cand >= 0.98) & (v_t > 0)
            hit_neg = (q_cand <= -0.98) & (v_t < 0)
            boundary_hit = hit_pos | hit_neg
            if boundary_hit.any():
                v_t = torch.where(boundary_hit, -0.8 * v_t, v_t)

            q_next = torch.clamp(q_prev + self.delta_t * torch.tanh(v_t), -1.0, 1.0)

        # Concatenate updated q and p to form next latent
        z_next = torch.cat([q_next, p_next], dim=1)

        new_state = DynamicsState(
            memory_state=new_mem_state,
            z_prev=z_next,
            ema_spikes=ema_spikes,
        )
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
