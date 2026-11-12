"""
Pure, Composable Loss Formulations for SPWM (Layer 2 — Training).

Single source of truth for:
- compute_representation_loss (Phase 0 grounded articulated representation losses)
- compute_m1_loss (Phase 1 action-conditioned M1 rollout, closed-loop BPTT, kinematic & critic losses)
- compute_critic_mc_loss & decompose_latent_loss
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

    predicted_v = critic(pred_latents.detach()).squeeze(-1)  # [B, H]
    return F.mse_loss(predicted_v, v_target.detach())


def compute_representation_loss(
    q: torch.Tensor,
    p: torch.Tensor,
    pos_pred: torch.Tensor,
    vel_pred: torch.Tensor,
    error_neurons: torch.Tensor,
    flat_kin: torch.Tensor,
    fk: nn.Module,
    joint_count: int = J,
    dt: float = 0.05,
    lambda_kinematic: float = 5.0,
    lambda_derivative: float = 1.0,
    lambda_smoothness: float = 0.1,
    lambda_sensory: float = 1.0,
    cartesian_weight: float = 25.0,
    velocity_loss_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Computes Phase 0 representation losses: kinematic (joint + Cartesian EE), derivative, smoothness, sensory."""
    joint_pos_true = flat_kin[..., :joint_count]
    joint_vel_true = flat_kin[..., joint_count : 2 * joint_count]

    loss_pos_kin = F.mse_loss(pos_pred, joint_pos_true, reduction="none").sum(dim=-1).mean()
    vel_scale = max(1e-3, float(velocity_loss_scale))
    loss_vel_kin = F.mse_loss(vel_pred, joint_vel_true, reduction="none").sum(dim=-1).mean() / (vel_scale ** 2)
    T_pred = fk.forward_homogeneous(pos_pred)
    T_true = fk.forward_homogeneous(joint_pos_true)
    loss_ee_pos = F.mse_loss(T_pred[..., :3, 3], T_true[..., :3, 3], reduction="none").sum(dim=-1).mean()
    loss_ee_rot = F.mse_loss(T_pred[..., :3, :3], T_true[..., :3, :3], reduction="none").sum(dim=(-2, -1)).mean()
    loss_cartesian = loss_ee_pos + 0.1 * loss_ee_rot
    l_kinematic = loss_pos_kin + loss_vel_kin + float(cartesian_weight) * loss_cartesian

    q_dim = q.shape[-1]
    delta_q, p_t = q[:, 1:] - q[:, :-1], p[:, :-1]
    p_mapped = p_t[..., :q_dim] if p_t.shape[-1] != q_dim else p_t
    l_derivative = F.mse_loss(delta_q, float(dt) * p_mapped, reduction="none").sum(dim=-1).mean()

    l_smoothness = F.smooth_l1_loss(p[:, 1:], p[:, :-1], beta=0.1, reduction="none").sum(dim=-1).mean()
    l_sensory = F.mse_loss(error_neurons, torch.zeros_like(error_neurons), reduction="none").sum(dim=-1).mean()

    total_loss = (
        float(lambda_kinematic) * l_kinematic
        + float(lambda_derivative) * l_derivative
        + float(lambda_smoothness) * l_smoothness
        + float(lambda_sensory) * l_sensory
    )
    return {
        "total_loss": total_loss,
        "loss_kinematic": l_kinematic,
        "loss_position": loss_pos_kin,
        "loss_velocity": loss_vel_kin,
        "loss_cartesian": loss_cartesian,
        "loss_derivative": l_derivative,
        "loss_smoothness": l_smoothness,
        "loss_sensory": l_sensory,
    }


def compute_m1_loss(
    m1_traj: torch.Tensor,
    target_latents: torch.Tensor,
    q_dim: int,
    critic: nn.Module,
    physical_decoder: Optional[nn.Module],
    fk: nn.Module,
    flags_rollout: Optional[torch.Tensor] = None,
    cl_pred: Optional[torch.Tensor] = None,
    cl_target: Optional[torch.Tensor] = None,
    target_kin: Optional[torch.Tensor] = None,
    joint_count: int = J,
    curriculum_phase: int = 1,
    gamma: float = 0.95,
    lambda_latent: float = 1.0,
    lambda_position: float = 1.0,
    lambda_velocity: float = 1.5,
    lambda_critic: float = 0.2,
    cartesian_weight: float = 25.0,
) -> Dict[str, torch.Tensor]:
    """Computes Phase 1 (M1) latent rollout, closed-loop mixed BPTT, kinematic, and critic losses."""
    B, horizon, z_dim = m1_traj.shape
    step_weights = torch.ones_like(m1_traj[..., :1])
    vel_step_weights = torch.ones_like(m1_traj[..., :1])
    if flags_rollout is not None:
        for b in range(B):
            heft_idx = (flags_rollout[b] == 1).nonzero(as_tuple=False)
            if len(heft_idx) > 0:
                h0, h_end = int(heft_idx[0, 0].item()), int(heft_idx[-1, 0].item())
                step_weights[b, max(0, h0 - 5) : min(horizon, h_end + 10)] = 2.0
                vel_step_weights[b, max(0, h0) : min(horizon, h_end + 12)] = 3.0

    pos_diff_sq = (m1_traj[..., :q_dim] - target_latents[..., :q_dim]) ** 2
    vel_diff_sq = (m1_traj[..., q_dim:] - target_latents[..., q_dim:]) ** 2
    loss_pos = (step_weights * pos_diff_sq).mean()
    loss_vel = (vel_step_weights * vel_diff_sq).mean()
    loss_critic = compute_critic_mc_loss(
        critic, physical_decoder, fk, m1_traj.detach(), target_latents, gamma=gamma
    )

    if cl_pred is not None and cl_target is not None and cl_pred.numel() > 0:
        cl_flat_pred = cl_pred.reshape(-1, z_dim)
        cl_flat_target = cl_target.reshape(-1, z_dim)
        n_cl = cl_flat_pred.shape[0]
        mixed_pred = torch.cat([m1_traj.reshape(-1, z_dim), cl_flat_pred], dim=0)
        mixed_target = torch.cat([target_latents.reshape(-1, z_dim), cl_flat_target], dim=0)
        ones_cl = torch.ones(n_cl, 1, device=m1_traj.device, dtype=step_weights.dtype)
        mixed_pos_w = torch.cat([step_weights.reshape(-1, 1), ones_cl], dim=0)
        mixed_vel_w = torch.cat([vel_step_weights.reshape(-1, 1), ones_cl], dim=0)
        p_diff = (mixed_pred[..., :q_dim] - mixed_target[..., :q_dim]) ** 2
        v_diff = (mixed_pred[..., q_dim:] - mixed_target[..., q_dim:]) ** 2
        loss_pos = (mixed_pos_w * p_diff).mean()
        loss_vel = (mixed_vel_w * v_diff).mean()
        cl_critic_loss = compute_critic_mc_loss(
            critic, physical_decoder, fk, cl_pred.detach(), cl_target, gamma=gamma
        )
        loss_critic = 0.5 * (loss_critic + cl_critic_loss)

    w_critic = float(lambda_critic) * (5.0 if curriculum_phase >= 3 else 1.0)
    loss_m1_latent = float(lambda_latent) * (
        float(lambda_position) * loss_pos + float(lambda_velocity) * loss_vel
    )
    loss_kin_m1 = torch.zeros((), device=m1_traj.device, dtype=m1_traj.dtype)
    if target_kin is not None and physical_decoder is not None:
        pred_q = physical_decoder.pos_head(m1_traj[..., :q_dim])
        pred_p = physical_decoder.vel_head(m1_traj[..., q_dim:])
        loss_kin_m1 = (
            F.mse_loss(pred_q, target_kin[..., :joint_count])
            + 0.5 * F.mse_loss(pred_p, target_kin[..., joint_count : 2 * joint_count])
            + float(cartesian_weight)
            * F.mse_loss(fk(pred_q), fk(target_kin[..., :joint_count]), reduction="none").sum(dim=-1).mean()
        )

    loss_m1_total = loss_m1_latent + 1.0 * loss_kin_m1
    total_loss = loss_m1_total + w_critic * loss_critic
    return {
        "total_loss": total_loss,
        "loss_m1": loss_m1_total,
        "loss_m1_latent": loss_m1_latent,
        "loss_kin_m1": loss_kin_m1,
        "loss_critic": loss_critic,
        "loss_position": loss_pos,
        "loss_velocity": loss_vel,
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
    """Unified loss module for Phase 0 representation, Phase 1 M1 rollout, and PredictorLoss."""

    def __init__(
        self,
        lambda_encoder: float = 1.0,
        lambda_decoder: float = 1.0,
        lambda_latent: float = 1.0,
        lambda_sensory: float = 1.0,
        lambda_position: float = 1.0,
        lambda_velocity: float = 1.5,
        lambda_kinematic: float = 5.0,
        lambda_derivative: float = 1.0,
        lambda_smoothness: float = 0.1,
        lambda_critic: float = 0.2,
        cartesian_weight: float = 25.0,
        velocity_loss_scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.lambda_encoder = float(lambda_encoder)
        self.lambda_decoder = float(lambda_decoder)
        self.lambda_latent = float(lambda_latent)
        self.lambda_sensory = float(lambda_sensory)
        self.lambda_position = float(lambda_position)
        self.lambda_velocity = float(lambda_velocity)
        self.lambda_kinematic = float(lambda_kinematic)
        self.lambda_derivative = float(lambda_derivative)
        self.lambda_smoothness = float(lambda_smoothness)
        self.lambda_critic = float(lambda_critic)
        self.cartesian_weight = float(cartesian_weight)
        self.velocity_loss_scale = float(velocity_loss_scale)

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
            lambda_latent=cfg.lambda_latent,
            lambda_sensory=cfg.lambda_sensory,
            lambda_position=cfg.lambda_position,
            lambda_velocity=cfg.lambda_velocity,
            lambda_kinematic=cfg.lambda_kinematic,
            lambda_derivative=cfg.lambda_derivative,
            lambda_smoothness=cfg.lambda_smoothness,
            lambda_critic=cfg.lambda_critic,
            cartesian_weight=cfg.cartesian_weight,
            velocity_loss_scale=cfg.velocity_loss_scale,
            **kwargs,
        )

    def compute_representation_loss(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        pos_pred: torch.Tensor,
        vel_pred: torch.Tensor,
        error_neurons: torch.Tensor,
        flat_kin: torch.Tensor,
        fk: nn.Module,
        joint_count: int = J,
        dt: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        return compute_representation_loss(
            q=q,
            p=p,
            pos_pred=pos_pred,
            vel_pred=vel_pred,
            error_neurons=error_neurons,
            flat_kin=flat_kin,
            fk=fk,
            joint_count=joint_count,
            dt=dt,
            lambda_kinematic=self.lambda_kinematic,
            lambda_derivative=self.lambda_derivative,
            lambda_smoothness=self.lambda_smoothness,
            lambda_sensory=self.lambda_sensory,
            cartesian_weight=self.cartesian_weight,
            velocity_loss_scale=self.velocity_loss_scale,
        )

    def compute_m1_loss(
        self,
        m1_traj: torch.Tensor,
        target_latents: torch.Tensor,
        q_dim: int,
        critic: nn.Module,
        physical_decoder: Optional[nn.Module],
        fk: nn.Module,
        flags_rollout: Optional[torch.Tensor] = None,
        cl_pred: Optional[torch.Tensor] = None,
        cl_target: Optional[torch.Tensor] = None,
        target_kin: Optional[torch.Tensor] = None,
        joint_count: int = J,
        curriculum_phase: int = 1,
        gamma: float = 0.95,
    ) -> Dict[str, torch.Tensor]:
        return compute_m1_loss(
            m1_traj=m1_traj,
            target_latents=target_latents,
            q_dim=q_dim,
            critic=critic,
            physical_decoder=physical_decoder,
            fk=fk,
            flags_rollout=flags_rollout,
            cl_pred=cl_pred,
            cl_target=cl_target,
            target_kin=target_kin,
            joint_count=joint_count,
            curriculum_phase=curriculum_phase,
            gamma=gamma,
            lambda_latent=self.lambda_latent,
            lambda_position=self.lambda_position,
            lambda_velocity=self.lambda_velocity,
            lambda_critic=self.lambda_critic,
            cartesian_weight=self.cartesian_weight,
        )

    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        return self.predictor_loss(*args, **kwargs)

    def compute_loss(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        return self.predictor_loss(*args, **kwargs)


__all__ = [
    "_frozen",
    "compute_critic_mc_loss",
    "compute_m1_loss",
    "compute_representation_loss",
    "decompose_latent_loss",
    "PredictorLoss",
    "SPWMLoss",
]
