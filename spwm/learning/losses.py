"""
Pure, composable loss functions for SPWM.
Single source of truth for:
- Predictor training loss (loss_encoder, loss_decoder, loss_position, loss_velocity)
- Corrector rollout loss (comb-weighted kinematic trajectory loss)
- Frozen parameter context manager
"""

from __future__ import annotations
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from spwm.config.schema import LossConfig


@contextmanager
def _frozen(module: nn.Module):
    """Context manager that temporarily sets requires_grad=False on all module parameters."""
    original = {n: p.requires_grad for n, p in module.named_parameters(recurse=True)}
    try:
        for p in module.parameters():
            p.requires_grad_(False)
        yield
    finally:
        for n, p in module.named_parameters(recurse=True):
            p.requires_grad_(original[n])


class PredictorLoss(nn.Module):
    """
    Simplified loss calculation for representation and predictor optimization.
    Returns exactly:
    - loss_encoder: MSE constraints directly on SpatialSoftmax keypoint distributions
    - loss_decoder: MSE of reconstructing ground-truth input state from immediate latent
    - loss_position: Euclidean distance / MSE between multi-step trajectory positions and ground-truth positions
    - loss_velocity: Euclidean distance / MSE between multi-step trajectory velocities and ground-truth velocities
    - total_loss: weighted sum of the active loss components
    """

    def __init__(
        self,
        lambda_encoder: float = 1.0,
        lambda_decoder: float = 1.0,
        lambda_position: float = 1.0,
        lambda_velocity: float = 1.0,
        multi_step_horizon: int = 5,
    ) -> None:
        super().__init__()
        self.lambda_encoder = lambda_encoder
        self.lambda_decoder = lambda_decoder
        self.lambda_position = lambda_position
        self.lambda_velocity = lambda_velocity
        self.multi_step_horizon = multi_step_horizon

    @classmethod
    def from_config(cls, cfg: LossConfig, **kwargs: Any) -> "PredictorLoss":
        return cls(
            lambda_encoder=cfg.lambda_encoder,
            lambda_decoder=cfg.lambda_decoder,
            lambda_position=cfg.lambda_position,
            lambda_velocity=cfg.lambda_velocity,
            multi_step_horizon=cfg.multi_step_horizon,
        )

    def forward(
        self,
        keypoints: Optional[torch.Tensor] = None,
        decoded_kinematics: Optional[torch.Tensor] = None,
        rollout_positions: Optional[torch.Tensor] = None,
        rollout_velocities: Optional[torch.Tensor] = None,
        target_kinematics: Optional[torch.Tensor] = None,
        target_rollout_kinematics: Optional[torch.Tensor] = None,
        num_objects: int = 1,
        mode: str = "train_predictor",
    ) -> Dict[str, torch.Tensor]:
        """
        Unified loss computation returning exactly:
        - total_loss
        - loss_encoder
        - loss_decoder
        - loss_position
        - loss_velocity
        """
        device = (
            keypoints.device
            if keypoints is not None
            else (
                decoded_kinematics.device
                if decoded_kinematics is not None
                else (
                    rollout_positions.device
                    if rollout_positions is not None
                    else (
                        target_kinematics.device
                        if target_kinematics is not None
                        else torch.device("cpu")
                    )
                )
            )
        )

        loss_encoder = torch.tensor(0.0, device=device)
        loss_decoder = torch.tensor(0.0, device=device)
        loss_position = torch.tensor(0.0, device=device)
        loss_velocity = torch.tensor(0.0, device=device)

        if mode == "train_representation":
            # 1. loss_encoder: MSE constraints on SpatialSoftmax keypoint distributions
            if keypoints is not None and target_kinematics is not None:
                gt_xy = target_kinematics[..., :2]
                k = keypoints.shape[-1] // 2
                kp = keypoints.view(*keypoints.shape[:-1], k, 2)
                gt_expanded = gt_xy.unsqueeze(-2).expand_as(kp)
                loss_encoder = F.mse_loss(kp, gt_expanded)

            # 2. loss_decoder: MSE of reconstructing ground-truth input state from immediate latent
            if decoded_kinematics is not None and target_kinematics is not None:
                dim_kin = decoded_kinematics.shape[-1]
                loss_decoder = F.mse_loss(decoded_kinematics, target_kinematics[..., :dim_kin])

            total_loss = self.lambda_encoder * loss_encoder + self.lambda_decoder * loss_decoder

        elif mode == "train_joint":
            # Joint training mode: representation loss + predictor rollout loss
            if keypoints is not None and target_kinematics is not None:
                gt_xy = target_kinematics[..., :2]
                k = keypoints.shape[-1] // 2
                kp = keypoints.view(*keypoints.shape[:-1], k, 2)
                gt_expanded = gt_xy.unsqueeze(-2).expand_as(kp)
                loss_encoder = F.mse_loss(kp, gt_expanded)

            if decoded_kinematics is not None and target_kinematics is not None:
                dim_kin = decoded_kinematics.shape[-1]
                loss_decoder = F.mse_loss(decoded_kinematics, target_kinematics[..., :dim_kin])

            rollout_target = target_rollout_kinematics if target_rollout_kinematics is not None else target_kinematics

            if rollout_positions is not None and rollout_target is not None:
                t_roll = rollout_positions.shape[1]
                true_pos = rollout_target[:, :t_roll, : 2 * num_objects].reshape(
                    rollout_positions.shape
                )
                loss_position = F.mse_loss(rollout_positions, true_pos)

            if rollout_velocities is not None and rollout_target is not None:
                t_roll = rollout_velocities.shape[1]
                true_vel = rollout_target[:, :t_roll, 2 * num_objects : 4 * num_objects].reshape(
                    rollout_velocities.shape
                )
                loss_velocity = F.mse_loss(rollout_velocities, true_vel)

            total_loss = (
                self.lambda_encoder * loss_encoder
                + self.lambda_decoder * loss_decoder
                + self.lambda_position * loss_position
                + self.lambda_velocity * loss_velocity
            )

        else:
            # mode == "train_predictor" or other predictor evaluation
            # Compute loss_position and loss_velocity over multi-step rollout
            if rollout_positions is not None and target_kinematics is not None:
                t_roll = rollout_positions.shape[1]
                true_pos = target_kinematics[:, :t_roll, : 2 * num_objects].reshape(
                    rollout_positions.shape
                )
                loss_position = F.mse_loss(rollout_positions, true_pos)

            if rollout_velocities is not None and target_kinematics is not None:
                t_roll = rollout_velocities.shape[1]
                true_vel = target_kinematics[:, :t_roll, 2 * num_objects : 4 * num_objects].reshape(
                    rollout_velocities.shape
                )
                loss_velocity = F.mse_loss(rollout_velocities, true_vel)

            total_loss = self.lambda_position * loss_position + self.lambda_velocity * loss_velocity

        return {
            "total_loss": total_loss,
            "loss_encoder": loss_encoder,
            "loss_decoder": loss_decoder,
            "loss_position": loss_position,
            "loss_velocity": loss_velocity,
        }

    compute_loss = forward


class CorrectorRolloutLoss(nn.Module):
    """
    Uses Gaussian temporal weighting centered at multiples of 25 steps to maintain
    smooth, elevated relevance around key transition boundaries (e.g. 1, 2, 3 and 22, 23, 24)
    while gently suppressing mid-range drift.
    Single source of truth used identically in training and validation.
    """

    def __init__(self, beta_v: float = 2.0, comb_sigma: float = 3.5, lambda_reg: float = 0.01) -> None:
        super().__init__()
        self.beta_v = beta_v
        self.comb_sigma = comb_sigma
        self.lambda_reg = lambda_reg

    def forward(
        self,
        decoded_rollout: torch.Tensor,
        target_kinematics: torch.Tensor,
        burn_in_steps: int = 0,
        num_objects: int = 1,
        gain_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = decoded_rollout.device
        pred_pos_active = decoded_rollout[:, burn_in_steps:, : 2 * num_objects]
        true_pos_active = target_kinematics[:, burn_in_steps:, : 2 * num_objects]
        pred_vel_active = decoded_rollout[:, burn_in_steps:, 2 * num_objects : 4 * num_objects]
        true_vel_active = target_kinematics[:, burn_in_steps:, 2 * num_objects : 4 * num_objects]

        active_len = pred_pos_active.shape[1]
        if active_len > 0:
            steps = torch.arange(1 + burn_in_steps, 1 + burn_in_steps + active_len, device=device, dtype=torch.float32)
            rem = steps % 25.0
            dist_to_0 = torch.minimum(rem, 25.0 - rem)
            time_weights = torch.exp(-0.5 * (dist_to_0 / self.comb_sigma) ** 2)
            time_weights = time_weights / time_weights.mean()
            tw_view = time_weights.view(1, -1, 1)

            pos_err_sq = (pred_pos_active - true_pos_active) ** 2
            vel_err_sq = (pred_vel_active - true_vel_active) ** 2

            l_pos = (pos_err_sq * tw_view).mean()
            l_vel = (vel_err_sq * tw_view).mean()
        else:
            l_pos = torch.tensor(0.0, device=device)
            l_vel = torch.tensor(0.0, device=device)

        l_kin = l_pos + self.beta_v * l_vel
        l_reg = torch.tensor(0.0, device=device)
        if gain_masks is not None and self.lambda_reg > 0:
            active_gains = gain_masks[:, burn_in_steps:] if burn_in_steps < gain_masks.shape[1] else gain_masks
            l_reg = (active_gains ** 2).sum(dim=-1).mean()

        total_loss = l_kin + self.lambda_reg * l_reg

        return {
            "total_loss": total_loss,
            "loss_corr": total_loss,
            "l_corr": total_loss,
            "l_corr_asymptotic": l_kin,
            "l_kin": l_kin,
            "l_pos": l_pos,
            "l_vel": l_vel,
            "l_reg": l_reg,
        }

    compute_corrector_rollout_loss = forward


class SPWMLoss(nn.Module):
    """
    Unified loss module combining PredictorLoss and CorrectorRolloutLoss.
    Provides single source of truth for both representation/predictor and corrector training.
    """

    def __init__(
        self,
        lambda_encoder: float = 1.0,
        lambda_decoder: float = 1.0,
        lambda_position: float = 1.0,
        lambda_velocity: float = 2.0,
        multi_step_horizon: int = 5,
        beta_v: float = 2.0,
        comb_sigma: float = 3.5,
        lambda_corrector_reg: float = 0.01,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.lambda_encoder = lambda_encoder
        self.lambda_decoder = lambda_decoder
        self.lambda_position = lambda_position
        self.lambda_velocity = lambda_velocity
        self.multi_step_horizon = multi_step_horizon
        self.beta_v = beta_v
        self.comb_sigma = comb_sigma
        self.lambda_corrector_reg = lambda_corrector_reg

        self.predictor_loss = PredictorLoss(
            lambda_encoder=lambda_encoder,
            lambda_decoder=lambda_decoder,
            lambda_position=lambda_position,
            lambda_velocity=lambda_velocity,
            multi_step_horizon=multi_step_horizon,
        )

        self.corrector_loss = CorrectorRolloutLoss(
            beta_v=beta_v,
            comb_sigma=comb_sigma,
            lambda_reg=lambda_corrector_reg,
        )

    @classmethod
    def from_config(cls, cfg: LossConfig, **kwargs: Any) -> "SPWMLoss":
        return cls(
            lambda_encoder=cfg.lambda_encoder,
            lambda_decoder=cfg.lambda_decoder,
            lambda_position=cfg.lambda_position,
            lambda_velocity=cfg.lambda_velocity,
            multi_step_horizon=cfg.multi_step_horizon,
            beta_v=cfg.beta_v,
            comb_sigma=cfg.comb_sigma,
            lambda_corrector_reg=getattr(cfg, "lambda_corrector_reg", 0.01),
            **kwargs,
        )

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        return self.predictor_loss(*args, **kwargs)

    def compute_loss(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        return self.predictor_loss(*args, **kwargs)

    def compute_corrector_rollout_loss(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        return self.corrector_loss(*args, **kwargs)