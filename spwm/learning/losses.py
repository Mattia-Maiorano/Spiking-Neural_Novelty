"""
Loss functions and anti-collapse objectives for SPWM (v7).
Implements:
- One-step latent prediction loss L_pred
- Multi-step autoregressive rollout prediction loss L_multi
- Anti-collapse variance regularization L_variance (VICReg style)
- Differentiable neuromorphic spike sparsity loss L_sparse (L1-norm on spike tensor)
- Kinematic velocity coupling L_vel on predictive states (frozen decoder pass)
- Auxiliary Spatial Coordinate Loss L_coord (SpatialSoftmax supervision)

v7 changes vs v6.6:
  - L_vel now explicitly freezes physical_decoder parameters during the velocity
    supervision forward/backward pass so that gradients from L_vel flow *only*
    into the predictor / dynamics generating pred_z, never into the probe weights.
  - Encoder loss is pure geometric ground-truth supervision (L_coord only); the
    sensory-consistency term (MSE vs pred_x) has been completely removed.
  - LossOutput dataclass renamed l_decoder -> l_probe for naming consistency.
"""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility: temporarily freeze / unfreeze a module's parameters
# ---------------------------------------------------------------------------

@contextmanager
def _frozen(module: nn.Module):
    """Context manager that sets requires_grad=False on all parameters of
    *module* for the duration of the block, then restores original state."""
    original = {n: p.requires_grad for n, p in module.named_parameters(recurse=True)}
    try:
        for p in module.parameters():
            p.requires_grad_(False)
        yield
    finally:
        for n, p in module.named_parameters(recurse=True):
            p.requires_grad_(original[n])


# ---------------------------------------------------------------------------
# Loss output dataclass
# ---------------------------------------------------------------------------

@dataclass
class LossOutput:
    """Detailed loss components breakdown."""
    total_loss: torch.Tensor
    l_pred: torch.Tensor
    l_multi: torch.Tensor
    l_var: torch.Tensor
    l_sparse: torch.Tensor
    l_probe: torch.Tensor          # renamed from l_decoder for clarity
    l_vel: Optional[torch.Tensor] = None
    l_coord: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, float]:
        d = {
            "total_loss": self.total_loss.item(),
            "l_pred": self.l_pred.item(),
            "l_multi": self.l_multi.item(),
            "l_var": self.l_var.item(),
            "l_sparse": self.l_sparse.item(),
            "l_probe": self.l_probe.item(),
        }
        if self.l_vel is not None:
            d["l_vel"] = self.l_vel.item()
        if self.l_coord is not None:
            d["l_coord"] = self.l_coord.item()
        return d


# ---------------------------------------------------------------------------
# Main loss module
# ---------------------------------------------------------------------------

class SPWMLoss(nn.Module):
    def __init__(
        self,
        lambda_pred: float = 1.0,
        lambda_multi: float = 0.5,
        lambda_var: float = 0.1,
        lambda_sparse: float = 0.5,
        lambda_vel: float = 0.5,
        lambda_probe: float = 2.0,
        lambda_coord: float = 0.15,
        multi_step_horizon: int = 3,
        target_variance: float = 1.0,
        target_spike_rate: float = 0.10,
        q_dim: int = 32,
        p_dim: int = 96,
    ) -> None:
        super().__init__()
        self.lambda_pred = lambda_pred
        self.lambda_multi = lambda_multi
        self.lambda_var = lambda_var
        self.lambda_sparse = lambda_sparse
        self.lambda_vel = lambda_vel
        self.lambda_probe = lambda_probe
        # Backwards-compat alias so that existing code using lambda_decoder still works
        self.lambda_decoder = lambda_probe
        self.lambda_coord = lambda_coord
        self.multi_step_horizon = multi_step_horizon
        self.target_variance = target_variance
        self.target_spike_rate = target_spike_rate
        self.q_dim = q_dim
        self.p_dim = p_dim

    # ------------------------------------------------------------------
    # Individual sub-losses (public API, callable from trainer directly)
    # ------------------------------------------------------------------

    def variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """VICReg-style anti-collapse variance penalty."""
        flat_z = z.reshape(-1, z.shape[-1])
        std = torch.sqrt(flat_z.var(dim=0) + 1e-4)
        return torch.mean(F.relu(self.target_variance - std))

    def coordinate_loss(
        self,
        keypoints: torch.Tensor,        # [B, T, K*2] in [-1,1]^2
        true_kinematics: torch.Tensor,  # [B, T, 4*N] (x,y,vx,vy)
    ) -> torch.Tensor:
        """Geometric ground-truth coordinate supervision for the encoder."""
        gt_xy = true_kinematics[..., :2]
        K = keypoints.shape[-1] // 2
        kp = keypoints.view(*keypoints.shape[:-1], K, 2)
        gt_expanded = gt_xy.unsqueeze(-2).expand_as(kp)
        return F.mse_loss(kp, gt_expanded)

    def multi_step_rollout_loss(
        self,
        latent_states: torch.Tensor,
        model_predictor: nn.Module,
    ) -> torch.Tensor:
        """Autoregressive multi-step rollout loss (delegated from trainer)."""
        B, T, D = latent_states.shape
        K = self.multi_step_horizon
        if T <= K + 1:
            return torch.tensor(0.0, device=latent_states.device)

        stride = max(1, K)
        t_starts = list(range(0, T - K, stride))
        if not t_starts:
            return torch.tensor(0.0, device=latent_states.device)

        curr_z = latent_states[:, t_starts].detach()
        step_losses = []
        for step in range(1, K + 1):
            target_step_z = torch.stack(
                [latent_states[:, t + step].detach() for t in t_starts], dim=1
            )
            pred_out = model_predictor(curr_z)
            curr_z = pred_out.predicted_latent
            step_losses.append(F.mse_loss(curr_z, target_step_z))

        return torch.stack(step_losses).mean()

    def velocity_loss_frozen_probe(
        self,
        pred_z: torch.Tensor,           # [B, T_pred, latent_dim]
        true_kinematics: torch.Tensor,  # [B, T, 4*N] (x, y, vx, vy)
        physical_decoder: nn.Module,
        stride_k: int = 5,
        dt: float = 0.01,
    ) -> torch.Tensor:
        """Supervisione cinematica multi-passo: calcola la velocità media 
        su finestra (q_{t+k} - q_t) / (k * dt) per alzare il rapporto segnale/rumore.
        """
        B, T_tot, D_kin = true_kinematics.shape
        T_pred = pred_z.shape[1]

        # Posizioni ground truth q (x, y)
        true_pos = true_kinematics[..., :2]

        if T_tot <= stride_k:
            # Fallback a passo singolo se la sequenza è troppo breve
            kin_target = true_kinematics[:, 1 : T_pred + 1, 2:4]
            with _frozen(physical_decoder):
                pred_decoded = physical_decoder(pred_z)
            pred_sub = pred_decoded[..., 2:4].contiguous()
            return F.smooth_l1_loss(pred_sub, kin_target)

        # Target de-noised multi-passo: Delta_k q / (k * dt)
        target_disp = true_pos[:, stride_k:, :] - true_pos[:, :-stride_k, :]
        target_vel = target_disp / (stride_k * dt)  # [B, T_tot - stride_k, 2]

        # Allineamento temporale con pred_z
        T_valid = min(T_pred, target_vel.shape[1])
        pred_z_valid = pred_z[:, :T_valid].contiguous()
        target_vel_valid = target_vel[:, :T_valid].contiguous()

        with _frozen(physical_decoder):
            pred_decoded = physical_decoder(pred_z_valid)

        pred_sub = pred_decoded[..., 2:4].contiguous()

        return F.smooth_l1_loss(pred_sub, target_vel_valid, beta=1.0)


    # ------------------------------------------------------------------
    # Combined forward (kept for compatibility; trainer uses sub-losses)
    # ------------------------------------------------------------------

    def forward(
        self,
        latent_states: torch.Tensor,                        # [B, T, latent_dim]
        predicted_latents: torch.Tensor,                    # [B, T-1, latent_dim]
        spikes: Optional[torch.Tensor] = None,              # [B, T, N]
        model_predictor: Optional[nn.Module] = None,
        physical_decoder: Optional[nn.Module] = None,
        decoded_kinematics: Optional[torch.Tensor] = None,  # from probe on detached z
        true_kinematics: Optional[torch.Tensor] = None,     # [B, T, 4*N]
        keypoints: Optional[torch.Tensor] = None,           # [B, T, K*2]
    ) -> LossOutput:
        device = latent_states.device
        B, T, D = latent_states.shape

        # 1. One-step prediction loss (target detached from graph)
        target_z = latent_states[:, 1:].detach()
        pred_z = predicted_latents[:, : target_z.shape[1]]
        l_pred = F.mse_loss(pred_z, target_z)

        # 2. Multi-step autoregressive rollout
        l_multi = torch.tensor(0.0, device=device)
        if self.lambda_multi > 0.0 and model_predictor is not None and T > self.multi_step_horizon + 1:
            l_multi = self.multi_step_rollout_loss(latent_states, model_predictor)

        # 3. Anti-collapse variance loss
        l_var = self.variance_loss(latent_states)

        # 4. Differentiable L1 sparsity on spike tensor
        l_sparse = torch.tensor(0.0, device=device)
        if self.lambda_sparse > 0.0 and spikes is not None:
            l_sparse = torch.abs(spikes.mean() - self.target_spike_rate)

        # 5. Velocity supervision on pred_z -- probe frozen to block probe gradient
        l_vel = torch.tensor(0.0, device=device)
        if (
            self.lambda_vel > 0.0
            and true_kinematics is not None
            and physical_decoder is not None
            and T > 1
        ):
            l_vel = self.velocity_loss_frozen_probe(pred_z, true_kinematics, physical_decoder)

        # 6. Kinematic probe loss (decoded from detached z)
        l_probe = torch.tensor(0.0, device=device)
        if self.lambda_probe > 0.0 and decoded_kinematics is not None and true_kinematics is not None:
            l_probe = F.mse_loss(decoded_kinematics, true_kinematics)

        # 7. Coordinate loss (geometric encoder supervision)
        l_coord = torch.tensor(0.0, device=device)
        if self.lambda_coord > 0.0 and keypoints is not None and true_kinematics is not None:
            l_coord = self.coordinate_loss(keypoints, true_kinematics)

        total_loss = (
            self.lambda_pred * l_pred
            + self.lambda_multi * l_multi
            + self.lambda_var * l_var
            + self.lambda_sparse * l_sparse
            + self.lambda_vel * l_vel
            + self.lambda_probe * l_probe
            + self.lambda_coord * l_coord
        )

        return LossOutput(
            total_loss=total_loss,
            l_pred=l_pred,
            l_multi=l_multi,
            l_var=l_var,
            l_sparse=l_sparse,
            l_probe=l_probe,
            l_vel=l_vel,
            l_coord=l_coord,
        )
