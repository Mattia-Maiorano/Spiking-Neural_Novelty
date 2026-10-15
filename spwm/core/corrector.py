"""
Slow Neuromorphic Corrector Module (SPWM-v0.1.0: Neuromodulatory Gain Field System).
Acts as a neuromodulator: emits a Gain Mask in [-1, 1] that amplifies or dampens
the fast predictor's local dynamics.
Strict: No fallbacks allowed. Parameter misalignments raise immediately.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn

from spwm.core.neurons import ALIFCell, ALIFState


@dataclass
class SlowCorrectorState:
    """Recurrent state container for SlowCorrector (Gain Modulation System)."""
    alif_state: ALIFState
    step_count: int
    last_gain_mask: torch.Tensor     # [B, latent_dim]

    def detach(self) -> "SlowCorrectorState":
        return SlowCorrectorState(
            alif_state=self.alif_state.detach(),
            step_count=self.step_count,
            last_gain_mask=self.last_gain_mask.detach(),
        )


@dataclass
class SlowCorrectorOutput:
    """Output container for a single corrector step (Gain Modulation)."""
    corrected_latent: torch.Tensor           # [B, latent_dim] s_t^corr
    gain_mask: torch.Tensor                  # [B, latent_dim] in [-1, 1]
    spikes: torch.Tensor                     # [B, corrector_dim]
    v_mem: torch.Tensor                      # [B, corrector_dim]
    intervened: bool


class SlowCorrector(nn.Module):
    """
    Continuous Slow Neuromorphic Corrector acting via Field Gain Modulation.
    Maps 1-to-1 with the latent space to apply hot/cold spots on transition dynamics.
    Strictly asserts that p_dim % q_dim == 0 for geometric constraint coupling.
    """

    def __init__(
        self,
        q_dim: int = 32,
        p_dim: int = 96,
        corrector_dim: int = 128,
        beta_mem: float = 0.20,
        beta_adapt: float = 0.05,
        v_th0: float = 0.50,
        gamma: float = 0.08,
        surrogate_name: str = "atan",
        surrogate_alpha: float = 2.0,
        alpha_gain: float = 1.0,
    ) -> None:
        super().__init__()
        if q_dim <= 0:
            raise ValueError(f"q_dim must be positive, got {q_dim}")
        if p_dim % q_dim != 0:
            raise ValueError(f"p_dim ({p_dim}) must be an exact multiple of q_dim ({q_dim}) for spatial boundary coupling.")

        self.q_dim = q_dim
        self.p_dim = p_dim
        self.latent_dim = q_dim + p_dim
        self.corrector_dim = corrector_dim
        self.alpha_gain = alpha_gain
        self.expansion_factor = p_dim // q_dim

        self.alif = ALIFCell(
            size=corrector_dim,
            beta_mem=beta_mem,
            beta_adapt=beta_adapt,
            v_th0=v_th0,
            gamma=gamma,
            surrogate_name=surrogate_name,
            surrogate_alpha=surrogate_alpha,
        )

        input_feature_dim = 2 * self.latent_dim
        self.input_proj = nn.Linear(input_feature_dim, corrector_dim)

        self.v_proj = nn.Linear(corrector_dim, corrector_dim, bias=False)
        self.s_proj = nn.Linear(corrector_dim, corrector_dim, bias=False)

        self.gain_net = nn.Sequential(
            nn.Linear(corrector_dim, self.latent_dim),
            nn.Tanh(),
        )

        nn.init.normal_(self.input_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.s_proj.weight, mean=0.0, std=0.02)
        for m in self.gain_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.05)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        initial_latent: Optional[torch.Tensor] = None,
    ) -> SlowCorrectorState:
        alif_state = self.alif.init_state(batch_size, device=device)
        last_gain_mask = torch.zeros(batch_size, self.latent_dim, device=device)
        return SlowCorrectorState(
            alif_state=alif_state,
            step_count=0,
            last_gain_mask=last_gain_mask,
        )

    def forward_step(
        self,
        z_prev: torch.Tensor,
        z_fast_proposed: torch.Tensor,
        state: Optional[SlowCorrectorState] = None,
        step_idx: Optional[int] = None,
        is_burn_in: bool = False,
    ) -> Tuple[torch.Tensor, SlowCorrectorOutput, SlowCorrectorState]:
        B = z_prev.shape[0]
        device = z_prev.device

        if state is None:
            state = self.init_state(B, device=device)

        delta_z_fast = z_fast_proposed - z_prev
        corr_in = torch.cat([z_fast_proposed, delta_z_fast], dim=-1)

        # 1. ALIF runs continuously at EVERY step
        syn_current = self.input_proj(corr_in)
        spikes, new_alif = self.alif(syn_current, state.alif_state)

        # 2. Representation from membrane potential and spikes
        u = torch.tanh(self.v_proj(new_alif.v_mem)) + self.s_proj(spikes)

        # 3. Continuous Gain Mask in [-1, 1]
        gain_mask = self.gain_net(u)

        # 4. Burn-in bypass vs Gain field modulation
        if is_burn_in:
            z_t_corr = z_fast_proposed
            intervened = False
        else:
            delta_z_mod = delta_z_fast * (1.0 + self.alpha_gain * gain_mask)
            z_t_corr = z_prev + delta_z_mod
            intervened = True

            # 5. Coordinate boundary clamping in [-1, 1] and momentum dissipation
            if self.q_dim > 0:
                q_raw = z_t_corr[:, :self.q_dim]
                p_raw = z_t_corr[:, self.q_dim:]

                hit_upper = (q_raw > 1.0)
                hit_lower = (q_raw < -1.0)
                boundary_mask = (hit_upper | hit_lower).float()

                q_corr = torch.clamp(q_raw, -1.0, 1.0)
                p_mask = boundary_mask.repeat_interleave(self.expansion_factor, dim=-1)
                p_corr = p_raw * (1.0 - 0.5 * p_mask)

                z_t_corr = torch.cat([q_corr, p_corr], dim=-1)


        output = SlowCorrectorOutput(
            corrected_latent=z_t_corr,
            gain_mask=gain_mask,
            spikes=spikes,
            v_mem=new_alif.v_mem,
            intervened=intervened,
        )

        new_state = SlowCorrectorState(
            alif_state=new_alif,
            step_count=state.step_count + 1,
            last_gain_mask=gain_mask,
        )

        return z_t_corr, output, new_state
