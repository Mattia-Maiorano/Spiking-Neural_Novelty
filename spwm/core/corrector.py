"""
Slow Corrector Module (SPWM: Baseline GRU / Residual MLP Control & Neuromorphic Gain Field).
Supports:
1. 'gru': Δz_corr_t = GRU(corr_in_t, h_{t-1}); z_corr_t = z_fast_t + Δz_corr_t (Experimental Control)
2. 'mlp': Δz_corr_t = MLP(corr_in_t); z_corr_t = z_fast_t + Δz_corr_t (Ablation Control)
3. 'alif': ALIF-based continuous gain modulation field (Neuromorphic system)
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
    """Recurrent state container for SlowCorrector."""
    alif_state: Optional[ALIFState] = None
    gru_state: Optional[torch.Tensor] = None   # [B, hidden_dim]
    step_count: int = 0
    last_gain_mask: Optional[torch.Tensor] = None     # [B, latent_dim]

    def detach(self) -> "SlowCorrectorState":
        return SlowCorrectorState(
            alif_state=self.alif_state.detach() if self.alif_state is not None else None,
            gru_state=self.gru_state.detach() if self.gru_state is not None else None,
            step_count=self.step_count,
            last_gain_mask=self.last_gain_mask.detach() if self.last_gain_mask is not None else None,
        )


@dataclass
class SlowCorrectorOutput:
    """Output container for a single corrector step."""
    corrected_latent: torch.Tensor           # [B, latent_dim] s_t^corr
    gain_mask: torch.Tensor                  # [B, latent_dim] (or delta_z in GRU mode)
    spikes: torch.Tensor                     # [B, corrector_dim]
    v_mem: torch.Tensor                      # [B, corrector_dim]
    intervened: bool


class SlowCorrector(nn.Module):
    """
    Slow Corrector acting either via:
    - 'gru': Recurrent GRU computing delta_z_corr added directly to z_fast (Control)
    - 'mlp': Feedforward MLP computing delta_z_corr added directly to z_fast (Control)
    - 'alif': Spiking neuromorphic gain modulation field
    Strictly asserts that p_dim % q_dim == 0 for geometric boundary coupling.
    """

    def __init__(
        self,
        q_dim: int = 32,
        p_dim: int = 96,
        corrector_type: str = "gru",
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

        valid_types = {"gru", "mlp", "alif"}
        if corrector_type not in valid_types:
            raise ValueError(f"corrector_type must be one of {valid_types}, got '{corrector_type}'")

        self.q_dim = q_dim
        self.p_dim = p_dim
        self.latent_dim = q_dim + p_dim
        self.corrector_type = corrector_type
        self.corrector_dim = corrector_dim
        self.alpha_gain = alpha_gain
        self.expansion_factor = p_dim // q_dim

        input_feature_dim = 2 * self.latent_dim

        if self.corrector_type == "gru":
            self.gru_cell = nn.GRUCell(input_feature_dim, corrector_dim)
            self.delta_proj = nn.Sequential(
                nn.Linear(corrector_dim, corrector_dim),
                nn.GELU(),
                nn.Linear(corrector_dim, self.latent_dim),
            )
            # Alias for backward-compatible inspections
            self.gain_net = self.delta_proj
            # Initialize close to zero so initial corrections start nominal
            nn.init.zeros_(self.delta_proj[-1].weight)
            nn.init.zeros_(self.delta_proj[-1].bias)

        elif self.corrector_type == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(input_feature_dim, corrector_dim),
                nn.GELU(),
                nn.Linear(corrector_dim, corrector_dim),
                nn.GELU(),
                nn.Linear(corrector_dim, self.latent_dim),
            )
            self.gain_net = self.mlp
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

        elif self.corrector_type == "alif":
            self.alif = ALIFCell(
                size=corrector_dim,
                beta_mem=beta_mem,
                beta_adapt=beta_adapt,
                v_th0=v_th0,
                gamma=gamma,
                surrogate_name=surrogate_name,
                surrogate_alpha=surrogate_alpha,
            )
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
        if self.corrector_type == "gru":
            gru_state = torch.zeros(batch_size, self.corrector_dim, device=device)
            last_gain_mask = torch.zeros(batch_size, self.latent_dim, device=device)
            return SlowCorrectorState(
                alif_state=None,
                gru_state=gru_state,
                step_count=0,
                last_gain_mask=last_gain_mask,
            )
        elif self.corrector_type == "mlp":
            last_gain_mask = torch.zeros(batch_size, self.latent_dim, device=device)
            return SlowCorrectorState(
                alif_state=None,
                gru_state=None,
                step_count=0,
                last_gain_mask=last_gain_mask,
            )
        else:  # alif
            alif_state = self.alif.init_state(batch_size, device=device)
            last_gain_mask = torch.zeros(batch_size, self.latent_dim, device=device)
            return SlowCorrectorState(
                alif_state=alif_state,
                gru_state=None,
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

        if self.corrector_type == "gru":
            prev_h = state.gru_state if state.gru_state is not None else torch.zeros(B, self.corrector_dim, device=device)
            new_h = self.gru_cell(corr_in, prev_h)
            delta_z_corr = self.delta_proj(new_h)

            spikes = torch.zeros(B, self.corrector_dim, device=device)
            v_mem = new_h
            gain_mask = delta_z_corr

            if is_burn_in:
                z_t_corr = z_fast_proposed
                intervened = False
            else:
                z_t_corr = z_fast_proposed + delta_z_corr
                intervened = True

            new_state = SlowCorrectorState(
                alif_state=None,
                gru_state=new_h,
                step_count=state.step_count + 1,
                last_gain_mask=gain_mask,
            )

        elif self.corrector_type == "mlp":
            delta_z_corr = self.mlp(corr_in)

            spikes = torch.zeros(B, self.corrector_dim, device=device)
            v_mem = delta_z_corr
            gain_mask = delta_z_corr

            if is_burn_in:
                z_t_corr = z_fast_proposed
                intervened = False
            else:
                z_t_corr = z_fast_proposed + delta_z_corr
                intervened = True

            new_state = SlowCorrectorState(
                alif_state=None,
                gru_state=None,
                step_count=state.step_count + 1,
                last_gain_mask=gain_mask,
            )

        else:  # alif
            alif_st = state.alif_state if state.alif_state is not None else self.alif.init_state(B, device=device)
            syn_current = self.input_proj(corr_in)
            spikes, new_alif = self.alif(syn_current, alif_st)

            u = torch.tanh(self.v_proj(new_alif.v_mem)) + self.s_proj(spikes)
            gain_mask = self.gain_net(u)

            if is_burn_in:
                z_t_corr = z_fast_proposed
                intervened = False
            else:
                delta_z_mod = delta_z_fast * (1.0 + self.alpha_gain * gain_mask)
                z_t_corr = z_prev + delta_z_mod
                intervened = True

            new_state = SlowCorrectorState(
                alif_state=new_alif,
                gru_state=None,
                step_count=state.step_count + 1,
                last_gain_mask=gain_mask,
            )

        # Coordinate boundary clamping in [-1, 1] and momentum dissipation
        if intervened and self.q_dim > 0:
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
            v_mem=v_mem,
            intervened=intervened,
        )

        return z_t_corr, output, new_state
