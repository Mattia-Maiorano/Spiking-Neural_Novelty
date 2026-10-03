"""
Loss functions and anti-collapse objectives for SPWM (v6.6).
Implements:
- One-step latent prediction loss L_pred
- Multi-step rollout prediction loss L_multi
- Anti-collapse variance regularization L_variance (VICReg style)
- Neuromorphic spike sparsity loss L_sparse
- Kinematic physical probe loss L_probe
- v4.2: Auxiliary Spatial Coordinate Loss L_coord
  Directly supervises the 16 raw Spatial-Softmax keypoints (u_k, v_k) ∈ [-1,1]^2
  against the GT object position projected into the same normalized image space,
  providing gradient that steers the Conv2D + SpatialSoftmax frontend toward real
  object geometry instead of high-contrast spurious features.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
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
    l_probe: torch.Tensor
    l_vel: Optional[torch.Tensor] = None        # v6.6: direct primary world model velocity coupling ||p_t - Delta q_t||_2
    l_coord: Optional[torch.Tensor] = None      # v4.2: auxiliary spatial coordinate loss

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


class SPWMLoss(nn.Module):
    """
    Composite Loss for Spiking Predictive World Model (v6.6).

    Formula:
        L_total(t) = λ_pred·L_pred + λ_multi·L_multi + λ_var·L_var
                   + λ_sparse·L_sparse + λ_vel·L_vel + λ_probe·L_probe
                   + λ_coord·L_coord

    L_sparse: Proportional L1 sparsity penalty around target spike rate (0.10):
        L_sparse = |mean(s) - 0.10| with λ_sparse >= 0.5.

    L_vel: Direct primary kinematic velocity coupling in world model phase space:
        L_vel = ||p_t - Δq_t||_2 where Δq_t = q_(t+1) - q_t.

    L_coord: Auxiliary spatial keypoint loss.
        Penalizes the MSE between raw Spatial-Softmax keypoints (u_k, v_k) ∈ [-1,1]^2
        and the GT object xy-position projected into the normalized image frame.
    """

    def __init__(
        self,
        lambda_pred: float = 1.0,
        lambda_multi: float = 0.5,
        lambda_var: float = 0.1,
        lambda_sparse: float = 0.5,
        lambda_vel: float = 0.5,
        lambda_probe: float = 0.5,
        lambda_coord: float = 0.0,    # v4.2: auxiliary coordinate coupling
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
        self.lambda_coord = lambda_coord
        self.multi_step_horizon = multi_step_horizon
        self.target_variance = target_variance
        self.target_spike_rate = target_spike_rate
        self.q_dim = q_dim
        self.p_dim = p_dim

    def variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Anti-collapse variance regularization (VICReg principle).
        Prevents all latent embeddings from collapsing into trivial constant vectors.
        Penalizes features whose standard deviation across (batch, time) is below target_variance.
        """
        # Flatten batch and time dimensions: [B * T, latent_dim]
        flat_z = z.reshape(-1, z.shape[-1])
        std = torch.sqrt(flat_z.var(dim=0) + 1e-4)
        # Hinge loss on standard deviation
        var_loss = torch.mean(F.relu(self.target_variance - std))
        return var_loss

    def coordinate_loss(
        self,
        keypoints: torch.Tensor,   # [B, T, K*2] or [B, K*2]  in [-1,1]^2
        true_kinematics: torch.Tensor,  # [B, T, 4*N] (x,y,vx,vy) in [-1,1]^2
    ) -> torch.Tensor:
        """
        Auxiliary Spatial Coordinate Loss (v4.2 — Strada B).

        Supervises the K raw Spatial-Softmax keypoints directly against the GT
        object position, averaged over all keypoints. The asymmetry (K >> num_objects)
        means multiple keypoints are free to specialize on different object parts;
        the loss drives the *centroid* of the keypoint cloud toward the GT.
        """
        gt_xy = true_kinematics[..., :2]  # [..., 2]

        # Reshape keypoints: [..., K, 2]
        K = keypoints.shape[-1] // 2
        kp = keypoints.view(*keypoints.shape[:-1], K, 2)  # [..., K, 2]

        # Broadcast GT to all keypoints: [..., 1, 2] -> [..., K, 2]
        gt_expanded = gt_xy.unsqueeze(-2).expand_as(kp)

        return F.mse_loss(kp, gt_expanded)

    def multi_step_rollout_loss(
        self,
        latent_states: torch.Tensor,
        model_predictor: nn.Module,
    ) -> torch.Tensor:
        """
        Closed-loop autoregressive multi-step rollout loss over short horizon K.
        During rollout, predicted output z_hat_(t+1) = [q_hat_(t+1), p_hat_(t+1)] is
        re-injected autoregressively as input for z_hat_(t+2), calculating MSE
        against detached ground truth latents over all K steps.
        """
        B, T, D = latent_states.shape
        K = self.multi_step_horizon
        if T <= K + 1:
            return torch.tensor(0.0, device=latent_states.device)

        stride = max(1, K)
        t_starts = list(range(0, T - K, stride))
        if not t_starts:
            return torch.tensor(0.0, device=latent_states.device)

        curr_z = latent_states[:, t_starts].detach()  # [B, S, D]
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
        latent_states: torch.Tensor,     # [B, T, latent_dim]
        predicted_latents: torch.Tensor, # [B, T, latent_dim]
        mean_spike_rate: torch.Tensor,   # scalar
        model_predictor: Optional[nn.Module] = None,
        decoded_kinematics: Optional[torch.Tensor] = None,  # [B, T, 4 * N]
        true_kinematics: Optional[torch.Tensor] = None,     # [B, T, 4 * N]
        keypoints: Optional[torch.Tensor] = None,           # [B, T, K*2] — v4.2
    ) -> LossOutput:
        B, T, D = latent_states.shape
        device = latent_states.device

        # 1. One-step prediction loss
        target_z = latent_states[:, 1:].detach()
        pred_z = predicted_latents[:, :-1]
        l_pred = F.mse_loss(pred_z, target_z)

        # 2. Multi-step autoregressive prediction loss
        l_multi = torch.tensor(0.0, device=device)
        if self.lambda_multi > 0.0 and model_predictor is not None and T > self.multi_step_horizon + 1:
            l_multi = self.multi_step_rollout_loss(latent_states, model_predictor)

        # 3. Anti-collapse variance loss
        l_var = self.variance_loss(latent_states)

        # 4. Proportional L1 Spike sparsity loss: L_sparse = |mean(s) - 0.10|
        l_sparse = torch.abs(mean_spike_rate - self.target_spike_rate)

        # 5. Direct primary kinematic velocity penalty: ||p_t - Δq_t||_2
        # Directly couples momentum p to the finite-difference coordinate velocity Δq_t = q_(t+1) - q_t
        l_vel = torch.tensor(0.0, device=device)
        if self.lambda_vel > 0.0 and T > 1:
            q_states = latent_states[..., :self.q_dim]       # [B, T, q_dim]
            p_states = latent_states[..., self.q_dim:]      # [B, T, p_dim]
            delta_q = q_states[:, 1:] - q_states[:, :-1]    # [B, T-1, q_dim]
            p_vel = p_states[:, :-1, :self.q_dim]           # [B, T-1, q_dim] matched coordinate velocity subspace
            l_vel = F.mse_loss(p_vel, delta_q)

        # 6. Kinematic probe loss (if probe supervision is active)
        l_probe = torch.tensor(0.0, device=device)
        if decoded_kinematics is not None and true_kinematics is not None:
            l_probe = F.mse_loss(decoded_kinematics, true_kinematics)

        # 7. v4.2: Auxiliary spatial coordinate loss (L_coord)
        l_coord = torch.tensor(0.0, device=device)
        if self.lambda_coord > 0.0 and keypoints is not None and true_kinematics is not None:
            l_coord = self.coordinate_loss(keypoints, true_kinematics)

        # Composite total loss — v6.6 formula:
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
