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
from spwm.contracts.tensors import J, assert_tensor_contract


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


def compute_critic_mc_loss(
    critic: nn.Module,
    physical_decoder: nn.Module,
    fk: nn.Module,
    pred_latents: torch.Tensor,
    target_latents: torch.Tensor,
    gamma: float = 0.95,
    w_rot_critic: float = 1.0,
) -> torch.Tensor:
    """Computes Monte Carlo discounted cost-to-go MSE loss for LatentCritic with geodesic rotation error.

    Target value backward recursion: V_target_t = step_cost_t + gamma * V_target_{t+1},
    where step_cost_t = pos_err_t + w_rot_critic * rot_err_t.
    """
    with torch.no_grad():
        jc = physical_decoder.joint_count
        pred_q = physical_decoder(pred_latents.detach())[..., :jc]
        tgt_q = physical_decoder(target_latents.detach())[..., :jc]
        pred_ee_homo = fk.forward_homogeneous(pred_q)
        tgt_ee_homo = fk.forward_homogeneous(tgt_q)
        pred_ee = pred_ee_homo[..., :3, 3]
        tgt_ee = tgt_ee_homo[..., :3, 3]
        pos_err = torch.norm(pred_ee - tgt_ee, p=2, dim=-1)  # [B, H]

        pred_R = pred_ee_homo[..., :3, :3]
        tgt_R = tgt_ee_homo[..., :3, :3]
        trace = torch.diagonal(torch.matmul(pred_R.transpose(-1, -2), tgt_R), dim1=-2, dim2=-1).sum(-1)
        cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
        rot_err = 1.0 - cos_theta

        step_cost = pos_err + float(w_rot_critic) * rot_err  # [B, H]
        v_target = torch.zeros_like(step_cost)
        v_next = torch.zeros_like(step_cost[:, 0])
        for t in range(step_cost.shape[1] - 1, -1, -1):
            v_next = step_cost[:, t] + float(gamma) * v_next
            v_target[:, t] = v_next

    predicted_v = critic(pred_latents).squeeze(-1)  # [B, H]
    return F.mse_loss(predicted_v, v_target.detach())


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
        joint_count: int = J,
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
    "compute_critic_mc_loss",
    "decompose_latent_loss",
    "PredictorLoss",
    "SPWMLoss",
]
