"""
Slow Neuromorphic Corrector Module (SPWM-v10: Neuromodulatory Gain Field System).

This module implements the slow timescale corrector population acting as a neuromodulator.
Instead of predicting additive offsets (forces) in the phase space, it emits a 
volume-transmission-like Gain Mask (hot and cold spots) that amplifies or dampens 
the fast predictor's local dynamics.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn

from spwm.models.neurons import ALIFCell, ALIFState


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
    """

    def __init__(
        self,
        q_dim: int = 32,
        p_dim: int = 96,
        corrector_dim: int = 128,
        beta_mem: float = 0.95,
        beta_adapt: float = 0.94,
        v_th0: float = 1.0,
        gamma: float = 0.25,
        surrogate_name: str = "atan",
        surrogate_alpha: float = 2.0,
        alpha_gain: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__()
        self.q_dim = q_dim
        self.p_dim = p_dim
        self.latent_dim = q_dim + p_dim
        self.corrector_dim = corrector_dim
        self.alpha_gain = alpha_gain

        # Independent slow ALIF population (1-to-1 mapping with latent dimensions)
        self.alif = ALIFCell(
            size=corrector_dim,
            beta_mem=beta_mem,
            beta_adapt=beta_adapt,
            v_th0=v_th0,
            gamma=gamma,
            surrogate_name=surrogate_name,
            surrogate_alpha=surrogate_alpha,
        )

        # Input feature dimension: Concat(s_t^{fast}, \Delta s_t^{fast})
        input_feature_dim = 2 * self.latent_dim
        self.input_proj = nn.Linear(input_feature_dim, corrector_dim)

        # Neuromorphic representation projections from somatic membrane and spikes
        self.v_proj = nn.Linear(corrector_dim, corrector_dim, bias=False)
        self.s_proj = nn.Linear(corrector_dim, corrector_dim, bias=False)

        # Neuromodulatory gain projection: Emits hot/cold spots over the latent dimensions
        self.gain_net = nn.Sequential(
            nn.Linear(corrector_dim, self.latent_dim),
            nn.Tanh()  # Bounds the modulation mask to [-1, 1]
        )

        # Parameter initialization
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
        """Initializes quiescent state for the corrector."""
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

        # Continuous local dynamic delta (no cadence)
        delta_z_fast = z_fast_proposed - z_prev
        corr_in = torch.cat([z_fast_proposed, delta_z_fast], dim=-1)

        # 1. Continuous neuromorphic integration: ALIF runs at EVERY step
        syn_current = self.input_proj(corr_in)
        spikes, new_alif = self.alif(syn_current, state.alif_state)

        # 2. Slow population representation (COMPUTED CONTINUOUSLY)
        u = torch.tanh(self.v_proj(new_alif.v_mem)) + self.s_proj(spikes)

        # 3. Predict Gain Mask (hot spots > 0, cold spots < 0) continuously
        gain_mask = self.gain_net(u)  # [B, latent_dim]

        # 4. Apply Neuromodulation to the fast transition
        if is_burn_in:
            # Mask is ignored during burn-in to align hidden states without mutating trajectory
            z_t_corr = z_fast_proposed
            intervened = False
        else:
            # Gain field application: Hot spots amplify (>0), Cold spots dampen (<0)
            delta_z_mod = delta_z_fast * (1.0 + self.alpha_gain * gain_mask)
            z_t_corr = z_prev + delta_z_mod
            intervened = True
        
# Clamp the geometric coordinate portion (q_dim) to maintain physical boundaries [-1, 1]
        if self.q_dim > 0:
            q_raw = z_t_corr[:, :self.q_dim]
            p_raw = z_t_corr[:, self.q_dim:]

            # Detect boundaries
            hit_upper = (q_raw > 1.0)
            hit_lower = (q_raw < -1.0)
            boundary_mask = (hit_upper | hit_lower).float()

            q_corr = torch.clamp(q_raw, -1.0, 1.0)
            
            # Adatta la maschera di collisione spaziale (q) alle dimensioni del momento (p)
            if self.p_dim % self.q_dim == 0:
                # Se p è un multiplo di q (es. 96 e 32), espandi la maschera localmente
                expansion_factor = self.p_dim // self.q_dim
                p_mask = boundary_mask.repeat_interleave(expansion_factor, dim=-1)
            else:
                # Fallback: se non c'è corrispondenza diretta, applica uno smorzamento 
                # globale basato sulla percentuale di keypoint che sbattono contro i bordi
                p_mask = boundary_mask.mean(dim=-1, keepdim=True)
            
            # Momentum dissipation on physical constraint collision
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