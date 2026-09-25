"""
Loss functions and anti-collapse objectives for SPWM.
Implements:
- One-step latent prediction loss L_pred
- Multi-step rollout prediction loss L_multi
- Anti-collapse variance regularization L_variance (VICReg style)
- Neuromorphic spike sparsity loss L_sparse
- Kinematic physical probe loss L_probe
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

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_loss": self.total_loss.item(),
            "l_pred": self.l_pred.item(),
            "l_multi": self.l_multi.item(),
            "l_var": self.l_var.item(),
            "l_sparse": self.l_sparse.item(),
            "l_probe": self.l_probe.item(),
        }


class SPWMLoss(nn.Module):
    """
    Composite Loss for Spiking Predictive World Model.
    Loss = λ_pred * L_pred + λ_multi * L_multi + λ_var * L_var + λ_sparse * L_sparse + λ_probe * L_probe
    """

    def __init__(
        self,
        lambda_pred: float = 1.0,
        lambda_multi: float = 0.5,
        lambda_var: float = 0.1,
        lambda_sparse: float = 0.001,
        lambda_probe: float = 0.5,
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

    def forward(
        self,
        latent_states: torch.Tensor,  # [B, T, latent_dim]
        predicted_latents: torch.Tensor,  # [B, T, latent_dim]
        mean_spike_rate: torch.Tensor,  # scalar
        model_predictor: Optional[nn.Module] = None,
        decoded_kinematics: Optional[torch.Tensor] = None,  # [B, T, 4 * N]
        true_kinematics: Optional[torch.Tensor] = None,  # [B, T, 4 * N]
    ) -> LossOutput:
        B, T, D = latent_states.shape
        device = latent_states.device

        # 1. One-step prediction loss:
        # z_hat from step t predicts step t+1
        # Stop gradient on target latent so predictor doesn't trivially minimize loss by shrinking z
        target_z = latent_states[:, 1:].detach()
        pred_z = predicted_latents[:, :-1]
        l_pred = F.mse_loss(pred_z, target_z)

        # 2. Multi-step prediction loss (over short horizon K during training)
        l_multi = torch.tensor(0.0, device=device)
        if self.lambda_multi > 0.0 and model_predictor is not None and T > self.multi_step_horizon + 1:
            multi_losses = []
            # Sample starting steps for rollout
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

        # 4. Spike Sparsity loss (penalize excessive firing rate)
        l_sparse = mean_spike_rate

        # 5. Kinematics probe loss (supervision for physical decoding probe)
        l_probe = torch.tensor(0.0, device=device)
        if decoded_kinematics is not None and true_kinematics is not None:
            l_probe = F.mse_loss(decoded_kinematics, true_kinematics)

        # Composite total loss
        total_loss = (
            self.lambda_pred * l_pred
            + self.lambda_multi * l_multi
            + self.lambda_var * l_var
            + self.lambda_sparse * l_sparse
            + self.lambda_probe * l_probe
        )

        return LossOutput(
            total_loss=total_loss,
            l_pred=l_pred,
            l_multi=l_multi,
            l_var=l_var,
            l_sparse=l_sparse,
            l_probe=l_probe,
        )
