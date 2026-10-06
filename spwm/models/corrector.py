"""
Slow Neuromorphic Corrector Module (SPWM-v8: Two-Timescale Predictor-Corrector).

This module implements the slow timescale corrector population designed to prevent
asymptotic rollout divergence and ALIF spike-rate saturation on extended horizons (H >= 25)
while preserving the local predictive acuity (H <= 10) of the fast predictor.

Key Design Principles:
1. Independent Slow ALIF Population:
   Membrane decay (beta_mem ~ 0.95) and adaptation decay (beta_adapt ~ 0.995)
   are markedly slower than the fast population (beta_mem=0.80, beta_adapt=0.90/0.985).
2. Rarefied Cadence (Delta):
   Executes corrective intervention at regular intervals Delta (e.g. Delta=5),
   maintaining smooth inter-cadence exponential decay for continuous C^1 trajectories.
3. Quiescent Baseline Regime:
   Elevated resting threshold (v_th0 = 1.5) ensures near-zero baseline spiking on
   nominal orbits, firing only upon critical trajectory drift.
4. Non-Destructive Continuous Coupling:
   Modulates kinematic derivative (velocity/steering force) and dampens internal
   momentum p/sub-threshold variables without discontinuous coordinate resets.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn

from spwm.models.neurons import ALIFCell, ALIFState


@dataclass
class SlowCorrectorState:
    """Recurrent state container for SlowCorrector."""
    alif_state: ALIFState
    step_count: int
    last_delta_v: torch.Tensor   # [B, q_dim]
    last_delta_p: torch.Tensor   # [B, p_dim]
    last_damp: torch.Tensor      # [B, p_dim]


@dataclass
class SlowCorrectorOutput:
    """Output container for a single corrector step."""
    corrected_latent: torch.Tensor  # [B, latent_dim]
    delta_v: torch.Tensor           # [B, q_dim]
    delta_p: torch.Tensor           # [B, p_dim]
    damp_factor: torch.Tensor       # [B, p_dim]
    spikes: torch.Tensor            # [B, corrector_dim]
    v_mem: torch.Tensor             # [B, corrector_dim]
    intervened: bool


class SlowCorrector(nn.Module):
    """
    Two-Timescale Slow Neuromorphic Corrector.
    """

    def __init__(
        self,
        q_dim: int = 32,
        p_dim: int = 96,
        corrector_dim: int = 64,
        cadence: int = 5,
        beta_mem: float = 0.95,
        beta_adapt: float = 0.995,
        v_th0: float = 1.5,
        gamma: float = 0.25,
        surrogate_name: str = "atan",
        surrogate_alpha: float = 2.0,
        max_gain_v: float = 0.25,
        max_gain_p: float = 0.25,
        max_damp: float = 0.40,
        inter_step_decay: float = 0.85,
    ) -> None:
        super().__init__()
        self.q_dim = q_dim
        self.p_dim = p_dim
        self.latent_dim = q_dim + p_dim
        self.corrector_dim = corrector_dim
        self.cadence = cadence
        self.max_gain_v = max_gain_v
        self.max_gain_p = max_gain_p
        self.max_damp = max_damp
        self.inter_step_decay = inter_step_decay

        # Independent slow ALIF population
        self.alif = ALIFCell(
            size=corrector_dim,
            beta_mem=beta_mem,
            beta_adapt=beta_adapt,
            v_th0=v_th0,
            gamma=gamma,
            surrogate_name=surrogate_name,
            surrogate_alpha=surrogate_alpha,
        )

        # Synaptic input projection: receives trajectory drift [Δq, p_fast, Δz]
        input_feature_dim = self.latent_dim + self.p_dim  # drift features
        self.input_proj = nn.Linear(input_feature_dim, corrector_dim)

        # Corrective coupling projections
        self.v_proj = nn.Linear(corrector_dim, corrector_dim, bias=False)
        self.s_proj = nn.Linear(corrector_dim, corrector_dim, bias=False)
        self.to_v = nn.Linear(corrector_dim, q_dim)
        self.to_p = nn.Linear(corrector_dim, p_dim)
        self.to_damp = nn.Linear(corrector_dim, p_dim)

        # Initialize coupling weights with small magnitude for gentle, stable initial modulation
        nn.init.normal_(self.to_v.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.to_v.bias)
        nn.init.normal_(self.to_p.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.to_p.bias)
        nn.init.normal_(self.to_damp.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.to_damp.bias, -2.0)  # sigmoid(-2) ~ 0.12 initial damping

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> SlowCorrectorState:
        """Initializes quiescent state for the corrector."""
        alif_state = self.alif.init_state(batch_size, device=device)
        last_delta_v = torch.zeros(batch_size, self.q_dim, device=device)
        last_delta_p = torch.zeros(batch_size, self.p_dim, device=device)
        last_damp = torch.zeros(batch_size, self.p_dim, device=device)
        return SlowCorrectorState(
            alif_state=alif_state,
            step_count=0,
            last_delta_v=last_delta_v,
            last_delta_p=last_delta_p,
            last_damp=last_damp,
        )

    def forward_step(
        self,
        z_prev: torch.Tensor,
        z_fast_proposed: torch.Tensor,
        state: Optional[SlowCorrectorState] = None,
    ) -> Tuple[torch.Tensor, SlowCorrectorOutput, SlowCorrectorState]:
        """
        Executes one time-step of the slow corrector:
        1. Compares proposed high-frequency state against previous trajectory point.
        2. If step is on cadence (step % Delta == 0), activates slow ALIF population.
        3. If step is off cadence, passively decays previous corrective modulation.
        4. Smoothly modifies velocity derivative and internal momentum without jumping q.

        Args:
            z_prev: [B, latent_dim] state at step t
            z_fast_proposed: [B, latent_dim] proposal from fast predictor for step t+1
            state: current SlowCorrectorState
        Returns:
            Tuple of:
            - z_corrected: [B, latent_dim] smoothed, stabilized next state
            - output: SlowCorrectorOutput diagnostic info
            - new_state: updated SlowCorrectorState
        """
        B = z_prev.shape[0]
        device = z_prev.device

        if state is None:
            state = self.init_state(B, device=device)

        q_prev = z_prev[:, :self.q_dim]
        q_fast = z_fast_proposed[:, :self.q_dim]
        p_fast = z_fast_proposed[:, self.q_dim:]

        # Proposed instantaneous displacement from fast predictor
        v_fast = q_fast - q_prev

        is_cadence_step = (state.step_count % self.cadence == 0)

        if is_cadence_step:
            # Macro-intervention on cadence: compute drift features
            drift_features = torch.cat([z_fast_proposed - z_prev, p_fast], dim=-1)
            syn_current = self.input_proj(drift_features)

            spikes, new_alif = self.alif(syn_current, state.alif_state)

            # Combined analog membrane and spike information
            u = torch.tanh(self.v_proj(new_alif.v_mem)) + self.s_proj(spikes)

            # Velocity steering force modulation
            delta_v = torch.tanh(self.to_v(u)) * self.max_gain_v
            # Momentum stabilization modulation
            delta_p = torch.tanh(self.to_p(u)) * self.max_gain_p
            # Sub-threshold damping factor to quench runaway spike cascades
            damp_factor = torch.sigmoid(self.to_damp(u)) * self.max_damp
        else:
            # Inter-cadence smooth exponential decay for C^1 continuity
            delta_v = state.last_delta_v * self.inter_step_decay
            delta_p = state.last_delta_p * self.inter_step_decay
            damp_factor = state.last_damp * self.inter_step_decay

            # Passive membrane leakage without spike emission during inter-cadence steps
            new_v_mem = self.alif.beta_mem * state.alif_state.v_mem
            spikes = torch.zeros_like(state.alif_state.spikes)
            new_alif = ALIFState(
                v_mem=new_v_mem,
                a_adapt=state.alif_state.a_adapt,
                spikes=spikes,
            )

        # Non-destructive continuous integration:
        # Modulate derivative, do NOT replace q
        v_corrected = v_fast + delta_v
        q_corrected = torch.clamp(q_prev + v_corrected, -1.0, 1.0)

        # Dampen internal momentum and inject corrective impulse
        p_corrected = p_fast * (1.0 - damp_factor) + delta_p

        z_corrected = torch.cat([q_corrected, p_corrected], dim=-1)

        output = SlowCorrectorOutput(
            corrected_latent=z_corrected,
            delta_v=delta_v,
            delta_p=delta_p,
            damp_factor=damp_factor,
            spikes=spikes,
            v_mem=new_alif.v_mem,
            intervened=is_cadence_step,
        )

        new_state = SlowCorrectorState(
            alif_state=new_alif,
            step_count=state.step_count + 1,
            last_delta_v=delta_v,
            last_delta_p=delta_p,
            last_damp=damp_factor,
        )

        return z_corrected, output, new_state
