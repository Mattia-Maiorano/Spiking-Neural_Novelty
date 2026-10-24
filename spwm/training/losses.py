"""
Pure, Composable Loss Formulations for SPWM (Layer 2 — Training).

Single source of truth for:
- PredictorLoss (representation keypoint/decoder MSE + rollout kinematic MSE)
- SPWMLoss (unified facade for phase training)
- _frozen context manager for temporary parameter freezing

Zero silent `getattr` fallbacks; strictly typed against `LossConfig`.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from spwm.contracts.config import LossConfig
from spwm.contracts.tensors import assert_tensor_contract


def decompose_latent_loss(
    predicted_latent: torch.Tensor,
    target_latent: torch.Tensor,
    q_dim: int,
    lambda_position: float = 1.0,
    lambda_velocity: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Decompose latent trajectory loss into coordinate (q) and momentum/velocity (p) components."""
    assert_tensor_contract(predicted_latent, (None, None, None), "decompose_latent_loss.predicted_latent")
    assert_tensor_contract(target_latent, tuple(predicted_latent.shape), "decompose_latent_loss.target_latent")

    pred_q = predicted_latent[..., :q_dim]
    target_q = target_latent[..., :q_dim]
    loss_pos_latent = F.mse_loss(pred_q, target_q)

    pred_p = predicted_latent[..., q_dim:]
    target_p = target_latent[..., q_dim:]
    loss_vel_latent = F.mse_loss(pred_p, target_p)

    weighted_loss = float(lambda_position) * loss_pos_latent + float(lambda_velocity) * loss_vel_latent
    return {
        "loss_latent": weighted_loss,
        "loss_pos_latent": loss_pos_latent,
        "loss_vel_latent": loss_vel_latent,
    }


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
    """Loss computation for representation and predictor optimization."""

    def __init__(
        self,
        lambda_encoder: float = 1.0,
        lambda_decoder: float = 1.0,
        lambda_sensory: float = 1.0,
        lambda_position: float = 1.0,
        lambda_velocity: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_encoder = float(lambda_encoder)
        self.lambda_decoder = float(lambda_decoder)
        self.lambda_sensory = float(lambda_sensory)
        self.lambda_position = float(lambda_position)
        self.lambda_velocity = float(lambda_velocity)

    @classmethod
    def from_config(cls, cfg: LossConfig) -> "PredictorLoss":
        return cls(
            lambda_encoder=cfg.lambda_encoder,
            lambda_decoder=cfg.lambda_decoder,
            lambda_sensory=cfg.lambda_sensory,
            lambda_position=cfg.lambda_position,
            lambda_velocity=cfg.lambda_velocity,
        )

    def forward(
        self,
        keypoints: Optional[torch.Tensor] = None,
        decoded_kinematics: Optional[torch.Tensor] = None,
        rollout_positions: Optional[torch.Tensor] = None,
        rollout_velocities: Optional[torch.Tensor] = None,
        target_kinematics: Optional[torch.Tensor] = None,
        target_rollout_kinematics: Optional[torch.Tensor] = None,
        joint_count: int = 8,
        mode: str = "train_m1",
    ) -> Dict[str, torch.Tensor]:
        device = torch.device("cpu")
        for tensor_candidate in (
            keypoints,
            decoded_kinematics,
            rollout_positions,
            rollout_velocities,
            target_kinematics,
        ):
            if tensor_candidate is not None:
                device = tensor_candidate.device
                break

        loss_encoder = torch.tensor(0.0, device=device)
        loss_decoder = torch.tensor(0.0, device=device)
        loss_position = torch.tensor(0.0, device=device)
        loss_velocity = torch.tensor(0.0, device=device)

        if keypoints is not None and target_kinematics is not None and mode in ("train_representation", "train_m1"):
            common_d = min(keypoints.shape[-1], target_kinematics.shape[-1], joint_count)
            loss_encoder = F.mse_loss(keypoints[..., :common_d], target_kinematics[..., :common_d])

        if decoded_kinematics is not None and target_kinematics is not None and mode in ("train_representation", "train_m1"):
            dim_kin = decoded_kinematics.shape[-1]
            loss_decoder = F.mse_loss(decoded_kinematics, target_kinematics[..., :dim_kin])

        if mode == "train_representation":
            total_loss = self.lambda_encoder * loss_encoder + self.lambda_decoder * loss_decoder
        else:
            rollout_target = (
                target_rollout_kinematics if target_rollout_kinematics is not None else target_kinematics
            )
            if rollout_positions is not None and rollout_target is not None:
                t_roll = rollout_positions.shape[1]
                true_pos = rollout_target[:, :t_roll, :joint_count].reshape(rollout_positions.shape)
                loss_position = F.mse_loss(rollout_positions, true_pos)

            if rollout_velocities is not None and rollout_target is not None:
                t_roll = rollout_velocities.shape[1]
                true_vel = rollout_target[:, :t_roll, joint_count : 2 * joint_count].reshape(
                    rollout_velocities.shape
                )
                loss_velocity = F.mse_loss(rollout_velocities, true_vel)

            if mode == "train_m1":
                total_loss = (
                    self.lambda_encoder * loss_encoder
                    + self.lambda_decoder * loss_decoder
                    + self.lambda_position * loss_position
                    + self.lambda_velocity * loss_velocity
                )
            else:
                total_loss = self.lambda_position * loss_position + self.lambda_velocity * loss_velocity

        return {
            "total_loss": total_loss,
            "loss_encoder": loss_encoder,
            "loss_decoder": loss_decoder,
            "loss_position": loss_position,
            "loss_velocity": loss_velocity,
        }

    compute_loss = forward


class SPWMLoss(nn.Module):
    """Unified loss module wrapping PredictorLoss."""

    def __init__(
        self,
        lambda_encoder: float = 1.0,
        lambda_decoder: float = 1.0,
        lambda_sensory: float = 1.0,
        lambda_position: float = 1.0,
        lambda_velocity: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.lambda_encoder = float(lambda_encoder)
        self.lambda_decoder = float(lambda_decoder)
        self.lambda_sensory = float(lambda_sensory)
        self.lambda_position = float(lambda_position)
        self.lambda_velocity = float(lambda_velocity)

        self.predictor_loss = PredictorLoss(
            lambda_encoder=self.lambda_encoder,
            lambda_decoder=self.lambda_decoder,
            lambda_sensory=self.lambda_sensory,
            lambda_position=self.lambda_position,
            lambda_velocity=self.lambda_velocity,
        )

    @classmethod
    def from_config(cls, cfg: LossConfig, **kwargs: Any) -> "SPWMLoss":
        return cls(
            lambda_encoder=cfg.lambda_encoder,
            lambda_decoder=cfg.lambda_decoder,
            lambda_sensory=cfg.lambda_sensory,
            lambda_position=cfg.lambda_position,
            lambda_velocity=cfg.lambda_velocity,
            **kwargs,
        )

    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        return self.predictor_loss(*args, **kwargs)

    def compute_loss(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        return self.predictor_loss(*args, **kwargs)


__all__ = [
    "_frozen",
    "decompose_latent_loss",
    "PredictorLoss",
    "SPWMLoss",
]
