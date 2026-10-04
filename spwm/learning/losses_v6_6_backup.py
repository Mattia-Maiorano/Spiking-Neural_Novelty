"""
Loss functions and anti-collapse objectives for SPWM (v6.7).
Implements:
- One-step latent prediction loss L_pred
- Multi-step autoregressive rollout prediction loss L_multi
- Anti-collapse variance regularization L_variance (VICReg style)
- Differentiable neuromorphic spike sparsity loss L_sparse (L1-norm on spike tensor)
- Kinematic velocity coupling L_vel on predictive states
- Auxiliary Spatial Coordinate Loss L_coord (SpatialSoftmax supervision)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossOutput:
    """Detailed loss components breakdown."""
    total_loss: torch.Tensor
    l_pred: torch.Tensor
    l_multi: torch.Tensor
    l_var: torch.Tensor
    l_sparse: torch.Tensor
    l_decoder: torch.Tensor
    l_vel: Optional[torch.Tensor] = None
    l_coord: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, float]:
        d = {
            "total_loss": self.total_loss.item(),
            "l_pred": self.l_pred.item(),
            "l_multi": self.l_multi.item(),
            "l_var": self.l_var.item(),
            "l_sparse": self.l_sparse.item(),
            "l_decoder": self.l_decoder.item(),
        }
        if self.l_vel is not None:
            d["l_vel"] = self.l_vel.item()
        if self.l_coord is not None:
            d["l_coord"] = self.l_coord.item()
        return d


class SPWMLoss(nn.Module):
    def __init__(
        self,
        lambda_pred: float = 1.0,
        lambda_multi: float = 0.5,
        lambda_var: float = 0.1,
        lambda_sparse: float = 0.5,
        lambda_vel: float = 0.5,
        lambda_decoder: float = 2.0,
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
        self.lambda_decoder = lambda_decoder
        self.lambda_coord = lambda_coord
        self.multi_step_horizon = multi_step_horizon
        self.target_variance = target_variance
        self.target_spike_rate = target_spike_rate
        self.q_dim = q_dim
        self.p_dim = p_dim

    def variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        flat_z = z.reshape(-1, z.shape[-1])
        std = torch.sqrt(flat_z.var(dim=0) + 1e-4)
        return torch.mean(F.relu(self.target_variance - std))

    def coordinate_loss(
        self,
        keypoints: torch.Tensor,        # [B, T, K*2] in [-1,1]^2
        true_kinematics: torch.Tensor,  # [B, T, 4*N] (x,y,vx,vy)
    ) -> torch.Tensor:
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

    def forward(
        self,
        latent_states: torch.Tensor,                        # [B, T, latent_dim]
        predicted_latents: torch.Tensor,                    # [B, T-1, latent_dim] o [B, T, latent_dim]
        spikes: Optional[torch.Tensor] = None,              # [B, T, N] o [B, N] (differenziabile)
        model_predictor: Optional[nn.Module] = None,
        physical_decoder: Optional[nn.Module] = None,       # Decoder per mappare p -> cinematica
        decoded_kinematics: Optional[torch.Tensor] = None,  # Output del decoder
        true_kinematics: Optional[torch.Tensor] = None,     # [B, T, 4*N]
        keypoints: Optional[torch.Tensor] = None,           # [B, T, K*2]
    ) -> LossOutput:
        device = latent_states.device
        B, T, D = latent_states.shape

        # 1. One-step prediction loss (target disaccoppiato da gradiente)
        target_z = latent_states[:, 1:].detach()
        pred_z = predicted_latents[:, :target_z.shape[1]]
        l_pred = F.mse_loss(pred_z, target_z)

        # 2. Multi-step autoregressive rollout
        l_multi = torch.tensor(0.0, device=device)
        if self.lambda_multi > 0.0 and model_predictor is not None and T > self.multi_step_horizon + 1:
            l_multi = self.multi_step_rollout_loss(latent_states, model_predictor)

        # 3. Anti-collapse variance loss
        l_var = self.variance_loss(latent_states)

        # 4. Sparsità L1 differenziabile sul tensore degli spike
        l_sparse = torch.tensor(0.0, device=device)
        if self.lambda_sparse > 0.0 and spikes is not None:
            l_sparse = torch.abs(spikes.mean() - self.target_spike_rate)

        # 5. Supervisione cinematica di velocità su pred_z
        l_vel = torch.tensor(0.0, device=device)
        if self.lambda_vel > 0.0 and true_kinematics is not None and physical_decoder is not None and T > 1:
            pred_decoded = physical_decoder(pred_z)
            # Indici 2:4 corrispondenti a (vx, vy)
            l_vel = F.mse_loss(pred_decoded[..., 2:4], true_kinematics[:, 1:target_z.shape[1]+1, 2:4])

        # 6. Kinematic decoder loss
        l_decoder = torch.tensor(0.0, device=device)
        if self.lambda_decoder > 0.0 and decoded_kinematics is not None and true_kinematics is not None:
            l_decoder = F.mse_loss(decoded_kinematics, true_kinematics)

        # 7. Coordinate loss
        l_coord = torch.tensor(0.0, device=device)
        if self.lambda_coord > 0.0 and keypoints is not None and true_kinematics is not None:
            l_coord = self.coordinate_loss(keypoints, true_kinematics)

        total_loss = (
            self.lambda_pred * l_pred
            + self.lambda_multi * l_multi
            + self.lambda_var * l_var
            + self.lambda_sparse * l_sparse
            + self.lambda_vel * l_vel
            + self.lambda_decoder * l_decoder
            + self.lambda_coord * l_coord
        )

        return LossOutput(
            total_loss=total_loss,
            l_pred=l_pred,
            l_multi=l_multi,
            l_var=l_var,
            l_sparse=l_sparse,
            l_decoder=l_decoder,
            l_vel=l_vel,
            l_coord=l_coord,
        )
