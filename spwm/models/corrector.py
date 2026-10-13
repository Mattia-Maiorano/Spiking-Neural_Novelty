"""
Slow Neuromorphic Corrector Module (SPWM-v9: Hamiltonian Energy & Impulsive Jump System).

This module implements the slow timescale corrector population designed to prevent
asymptotic rollout divergence on extended horizons (H >= 50, H = 100) while preserving
the local predictive acuity of the fast predictor.

Key Design Principles (SPWM-v9):
1. Hamiltonian Energy Modeling & Symplectically Coupled (q, p) Correction:
   - Eliminates uncoupled, free offsets for position and momentum.
   - The slow ALIF population predicts a scalar corrective potential ΔΦ(q_t).
   - Conservative momentum correction is computed as the negative spatial gradient:
     Δp_cons = -∇_q ΔΦ(q_t).
   - Controlled viscous dissipation is applied solely to drain parasitic drift energy:
     Δp = Δp_cons - damp_t * p_t.
   - Position q_t is updated symplectically coherent with the corrected momentum,
     preventing uncoupled coordinate alterations.
2. Impulsive Jump Paradigm (Hybrid Event-Driven System):
   - Continuous inter-step geometric decay is completely eliminated.
   - For t % Δ != 0: Fast Predictor operates in full autonomy (Δq = 0, Δp = 0, damp = 0).
   - For t % Δ == 0: Corrector executes an instantaneous discrete reset (orthogonal jump)
     on the current state, resetting intervention to zero at the next step.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn

from spwm.models.neurons import ALIFCell, ALIFState


@dataclass
class SlowCorrectorState:
    """Recurrent state container for SlowCorrector (SPWM-v9 Jump System)."""
    alif_state: ALIFState
    step_count: int
    last_delta_q: torch.Tensor       # [B, q_dim]
    last_delta_p: torch.Tensor       # [B, p_dim]
    last_damp: torch.Tensor          # [B, p_dim]
    z_cadence_prev: torch.Tensor     # [B, latent_dim] s_{t - cadence}

    def detach(self) -> "SlowCorrectorState":
        return SlowCorrectorState(
            alif_state=self.alif_state.detach(),
            step_count=self.step_count,
            last_delta_q=self.last_delta_q.detach(),
            last_delta_p=self.last_delta_p.detach(),
            last_damp=self.last_damp.detach(),
            z_cadence_prev=self.z_cadence_prev.detach(),
        )


@dataclass
class SlowCorrectorOutput:
    """Output container for a single corrector step (SPWM-v9)."""
    corrected_latent: torch.Tensor           # [B, latent_dim] s_t^corr
    delta_q: torch.Tensor                    # [B, q_dim]
    delta_p: torch.Tensor                    # [B, p_dim]
    damp_factor: torch.Tensor                # [B, p_dim]
    spikes: torch.Tensor                     # [B, corrector_dim]
    v_mem: torch.Tensor                      # [B, corrector_dim]
    intervened: bool
    potential: Optional[torch.Tensor] = None # [B, 1] scalar potential ΔΦ


class SlowCorrector(nn.Module):
    """
    Two-Timescale Slow Neuromorphic Corrector with Hamiltonian Energy Modeling
    and Impulsive Jump System (SPWM-v9).
    """

    def __init__(
        self,
        q_dim: int = 32,
        p_dim: int = 96,
        corrector_dim: int = 128,
        cadence: int = 5,
        beta_mem: float = 0.95,
        beta_adapt: float = 0.995,
        v_th0: float = 1.0,
        gamma: float = 0.25,
        surrogate_name: str = "atan",
        surrogate_alpha: float = 2.0,
        max_gain_p: float = 0.45,
        max_damp: float = 0.15,
        potential_hidden_dim: int = 64,
        **kwargs,
    ) -> None:
        super().__init__()
        self.q_dim = q_dim
        self.p_dim = p_dim
        self.latent_dim = q_dim + p_dim
        self.corrector_dim = corrector_dim
        self.cadence = cadence
        self.max_gain_p = max_gain_p
        self.max_damp = max_damp

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

        # Input feature dimension: Concat(s_t^{fast}, s_{t-cadence})
        input_feature_dim = 2 * self.latent_dim
        self.input_proj = nn.Linear(input_feature_dim, corrector_dim)

        # Neuromorphic representation projections from somatic membrane and spikes
        self.v_proj = nn.Linear(corrector_dim, corrector_dim, bias=False)
        self.s_proj = nn.Linear(corrector_dim, corrector_dim, bias=False)

        # 1. Scalar corrective potential head: ΔΦ(q_t; u_t)
        self.potential_net = nn.Sequential(
            nn.Linear(q_dim + corrector_dim, potential_hidden_dim),
            nn.Tanh(),
            nn.Linear(potential_hidden_dim, potential_hidden_dim),
            nn.Tanh(),
            nn.Linear(potential_hidden_dim, 1, bias=False),
        )

        # Coupling from spatial force -∇_q ΔΦ to momentum space p
        self.force_to_momentum = nn.Linear(q_dim, p_dim, bias=False)

        # Viscous damping projection to drain parasitic drift
        self.to_damp = nn.Linear(corrector_dim, p_dim)

        # Symplectic position update: derives Δq directly and coherently from momentum correction Δp
        self.momentum_to_pos = nn.Linear(p_dim, q_dim, bias=False)

        # Parameter initialization
        for m in self.potential_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.force_to_momentum.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.momentum_to_pos.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.to_damp.weight, mean=0.0, std=0.05)
        nn.init.constant_(self.to_damp.bias, -2.0)

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        initial_latent: Optional[torch.Tensor] = None,
    ) -> SlowCorrectorState:
        """Initializes quiescent state for the corrector."""
        alif_state = self.alif.init_state(batch_size, device=device)
        last_delta_q = torch.zeros(batch_size, self.q_dim, device=device)
        last_delta_p = torch.zeros(batch_size, self.p_dim, device=device)
        last_damp = torch.zeros(batch_size, self.p_dim, device=device)
        if initial_latent is not None:
            z_cadence_prev = initial_latent.clone()
        else:
            z_cadence_prev = torch.zeros(batch_size, self.latent_dim, device=device)

        return SlowCorrectorState(
            alif_state=alif_state,
            step_count=0,
            last_delta_q=last_delta_q,
            last_delta_p=last_delta_p,
            last_damp=last_damp,
            z_cadence_prev=z_cadence_prev,
        )

    def forward_step(
        self,
        z_prev: torch.Tensor,
        z_fast_proposed: torch.Tensor,
        state: Optional[SlowCorrectorState] = None,
        step_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, SlowCorrectorOutput, SlowCorrectorState]:
        """
        Executes one time-step of SPWM-v9 hybrid event-driven stabilization:
        1. Formulates cadence-aware input x_corr(t) = Concat(s_t^{fast}, s_{t-cadence}).
        2. Integrates slow ALIF population dynamics continuously at each step.
        3. On cadence (t % Delta == 0):
           - Predicts scalar corrective potential ΔΦ(q_t).
           - Computes conservative momentum correction: Δp_cons = -∇_q ΔΦ(q_t).
           - Applies viscous damping solely to drain parasitic drift: Δp = Δp_cons - damp_t * p_t.
           - Updates position q_t symplectically coherent with the corrected momentum.
        4. Inter-cadence (t % Delta != 0):
           - Impulsive Jump: Fast Predictor operates in full autonomy (Δq=0, Δp=0, damp=0).
        """
        B = z_prev.shape[0]
        device = z_prev.device

        if state is None:
            state = self.init_state(B, device=device, initial_latent=z_prev)

        curr_step = step_idx if step_idx is not None else state.step_count
        q_fast = z_fast_proposed[:, :self.q_dim]
        p_fast = z_fast_proposed[:, self.q_dim:]

        # Cadence-aware input vector: Concat(s_t^{fast}, s_{t-cadence})
        z_ref = state.z_cadence_prev if state.z_cadence_prev is not None else z_prev
        corr_in = torch.cat([z_fast_proposed, z_ref], dim=-1)

        # Continuous neuromorphic integration: ALIF runs at EVERY step
        syn_current = self.input_proj(corr_in)
        spikes, new_alif = self.alif(syn_current, state.alif_state)

        is_cadence_step = (curr_step % self.cadence == 0)

        if is_cadence_step:
            # 1. Slow population representation
            u = torch.tanh(self.v_proj(new_alif.v_mem)) + self.s_proj(spikes)

            # 2. Predict scalar corrective potential ΔΦ(q_t) and spatial negative gradient -∇_q ΔΦ(q_t)
            with torch.enable_grad():
                q_in = q_fast.clone().requires_grad_(True)
                phi = self.potential_net(torch.cat([q_in, u], dim=-1))  # [B, 1]
                grad_outputs = torch.ones_like(phi)
                grad_q = torch.autograd.grad(
                    outputs=phi,
                    inputs=q_in,
                    grad_outputs=grad_outputs,
                    create_graph=torch.is_grad_enabled(),
                    retain_graph=torch.is_grad_enabled(),
                )[0]  # [B, q_dim]
            f_cons = -grad_q

            # Conservative momentum correction
            delta_p_cons = self.force_to_momentum(f_cons)
            delta_p_cons = torch.tanh(delta_p_cons) * self.max_gain_p

            # Controlled viscous dissipation solely to drain parasitic drift excess
            if self.max_damp > 0.0:
                damp_factor = torch.sigmoid(self.to_damp(u)) * self.max_damp
            else:
                damp_factor = torch.zeros_like(p_fast)

            delta_p = delta_p_cons - damp_factor * p_fast
            p_t_corr = p_fast + delta_p

            # Symplectic position update coherent with corrected momentum
            delta_q = self.momentum_to_pos(delta_p)
            q_t_corr = torch.clamp(q_fast + delta_q, -1.0, 1.0)
            z_t_corr = torch.cat([q_t_corr, p_t_corr], dim=-1)
            potential = phi
        else:
            # Impulsive Jump System: zero interference between cadences
            delta_q = torch.zeros_like(q_fast)
            delta_p = torch.zeros_like(p_fast)
            damp_factor = torch.zeros_like(p_fast)
            z_t_corr = z_fast_proposed
            potential = None

        new_z_cadence_prev = z_t_corr if is_cadence_step else state.z_cadence_prev

        output = SlowCorrectorOutput(
            corrected_latent=z_t_corr,
            delta_q=delta_q,
            delta_p=delta_p,
            damp_factor=damp_factor,
            spikes=spikes,
            v_mem=new_alif.v_mem,
            intervened=is_cadence_step,
            potential=potential,
        )

        new_state = SlowCorrectorState(
            alif_state=new_alif,
            step_count=state.step_count + 1,
            last_delta_q=delta_q,
            last_delta_p=delta_p,
            last_damp=damp_factor,
            z_cadence_prev=new_z_cadence_prev,
        )

        return z_t_corr, output, new_state
