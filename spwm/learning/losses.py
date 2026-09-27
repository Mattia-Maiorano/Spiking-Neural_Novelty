"""
Loss functions and anti-collapse objectives for SPWM.
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

  Total loss (v4.2):
    L_total(t) = L_pred(t) + λ_coord · L_coord(t) + λ_sparse · L_reg(t)
  where λ_coord ∈ [0.1, 0.2] is the coordinate coupling strength.
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
    l_coord: torch.Tensor = None  # v4.2: auxiliary spatial coordinate loss

    def to_dict(self) -> Dict[str, float]:
        d = {
            "total_loss": self.total_loss.item(),
            "l_pred": self.l_pred.item(),
            "l_multi": self.l_multi.item(),
            "l_var": self.l_var.item(),
            "l_sparse": self.l_sparse.item(),
            "l_probe": self.l_probe.item(),
        }
        if self.l_coord is not None:
            d["l_coord"] = self.l_coord.item()
        return d


class SPWMLoss(nn.Module):
    """
    Composite Loss for Spiking Predictive World Model (v4.2).

    Formula:
        L_total(t) = λ_pred·L_pred + λ_multi·L_multi + λ_var·L_var
                   + λ_sparse·L_sparse + λ_probe·L_probe
                   + λ_coord·L_coord          # new in v4.2

    L_coord: Auxiliary spatial keypoint loss.
        Penalizes the MSE between raw Spatial-Softmax keypoints (u_k, v_k) ∈ [-1,1]^2
        and the GT object xy-position projected into the normalized image frame.
        Gradient flows unobstructed through Conv2D → SpatialSoftmax → ALIF input_proj,
        anchoring the visual frontend to Euclidean object coordinates before any
        symplectic / higher-order structure is imposed.

    Args:
        lambda_coord: coupling strength for L_coord (recommended 0.10–0.20).
                      Set to 0.0 to reproduce v4.1 behaviour.
    """

    def __init__(
        self,
        lambda_pred: float = 1.0,
        lambda_multi: float = 0.5,
        lambda_var: float = 0.1,
        lambda_sparse: float = 0.001,
        lambda_probe: float = 0.5,
        lambda_coord: float = 0.0,    # v4.2: auxiliary coordinate coupling
        multi_step_horizon: int = 3,
        target_variance: float = 1.0,
        target_spike_rate: float = 0.05,
    ) -> None:
        super().__init__()
        self.lambda_pred = lambda_pred
        self.lambda_multi = lambda_multi
        self.lambda_var = lambda_var
        self.lambda_sparse = lambda_sparse
        self.lambda_probe = lambda_probe
        self.lambda_coord = lambda_coord
        self.multi_step_horizon = multi_step_horizon
        self.target_variance = target_variance
        self.target_spike_rate = target_spike_rate

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

        Implementation:
            - Extract xy columns from true_kinematics (indices 0,1 per object).
            - Broadcast to all K keypoints: every keypoint is pulled toward GT.
            - Average MSE over the K pairs.

        This provides a smooth, geometry-aware gradient to Conv2D + SpatialSoftmax
        without imposing an explicit assignment (soft coupling), consistent with the
        biological analogy of a distributed dorsal-stream population code.
        """
        # true_kinematics: [..., 4*N] -> take x,y of the first (or only) object
        # positions are assumed to already be in [-1,1]^2 (normalized image coords)
        gt_xy = true_kinematics[..., :2]  # [..., 2]

        # Reshape keypoints: [..., K, 2]
        K = keypoints.shape[-1] // 2
        kp = keypoints.view(*keypoints.shape[:-1], K, 2)  # [..., K, 2]

        # Broadcast GT to all keypoints: [..., 1, 2] -> [..., K, 2]
        gt_expanded = gt_xy.unsqueeze(-2).expand_as(kp)

        return F.mse_loss(kp, gt_expanded)

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

        # 2. Multi-step prediction loss
        l_multi = torch.tensor(0.0, device=device)
        if self.lambda_multi > 0.0 and model_predictor is not None and T > self.multi_step_horizon + 1:
            multi_losses = []
            rollout_start_max = T - self.multi_step_horizon - 1
            start_indices = [0, rollout_start_max // 2, rollout_start_max]
            for t_start in start_indices:
                curr_z = latent_states[:, t_start].detach()
                for step in range(1, self.multi_step_horizon + 1):
                    target_step_z = latent_states[:, t_start + step].detach()
                    pred_out = model_predictor(curr_z)
                    curr_z = pred_out.predicted_latent
                    multi_losses.append(F.mse_loss(curr_z, target_step_z))
            if multi_losses:
                l_multi = torch.stack(multi_losses).mean()

        # 3. Anti-collapse variance loss
        l_var = self.variance_loss(latent_states)

        # 4. Spike sparsity loss
        l_sparse = mean_spike_rate

        # 5. Kinematic probe loss
        l_probe = torch.tensor(0.0, device=device)
        if decoded_kinematics is not None and true_kinematics is not None:
            l_probe = F.mse_loss(decoded_kinematics, true_kinematics)

        # 6. v4.2: Auxiliary spatial coordinate loss (L_coord)
        #    Flows gradient through Conv2D → SpatialSoftmax → ALIF input_proj
        l_coord = torch.tensor(0.0, device=device)
        if self.lambda_coord > 0.0 and keypoints is not None and true_kinematics is not None:
            l_coord = self.coordinate_loss(keypoints, true_kinematics)

        # Composite total loss — v4.2 formula:
        # L_total = L_pred + λ_coord · L_coord + λ_sparse · L_reg
        # (full version with all auxiliary terms)
        total_loss = (
            self.lambda_pred * l_pred
            + self.lambda_multi * l_multi
            + self.lambda_var * l_var
            + self.lambda_sparse * l_sparse
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
            l_coord=l_coord,
        )
