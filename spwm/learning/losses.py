"""
Pure, composable loss functions for SPWM.
Single source of truth for:
- Predictor training loss (1-step, multi-step autoregressive, velocity probe coupling, spike sparsity, VICReg variance)
- Corrector rollout loss (comb-weighted kinematic trajectory loss)
- Frozen parameter context manager
"""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
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


@dataclass
class LossOutput:
    """Detailed loss components breakdown."""
    total_loss: torch.Tensor
    l_pred: torch.Tensor
    l_multi: torch.Tensor
    l_var: torch.Tensor
    l_sparse: torch.Tensor
    l_probe: torch.Tensor
    l_vel: Optional[torch.Tensor] = None
    l_coord: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, float]:
        d = {
            "total_loss": float(self.total_loss.item()),
            "l_pred": float(self.l_pred.item()),
            "l_multi": float(self.l_multi.item()),
            "l_var": float(self.l_var.item()),
            "l_sparse": float(self.l_sparse.item()),
            "l_probe": float(self.l_probe.item()),
        }
        if self.l_vel is not None:
            d["l_vel"] = float(self.l_vel.item())
        if self.l_coord is not None:
            d["l_coord"] = float(self.l_coord.item())
        return d


class PredictorLoss(nn.Module):
    """
    Composite loss for fast predictor optimization. Single source of truth.
    Modifying this loss applies repo-wide across training and evaluation.
    """

    def __init__(
        self,
        lambda_pred: float = 1.0,
        lambda_multi: float = 0.5,
        lambda_var: float = 0.1,
        lambda_sparse: float = 0.5,
        lambda_vel: float = 0.5,
        lambda_probe: float = 2.0,
        lambda_coord: float = 0.15,
        beta_v: float = 2.0,
        sigma_q: float = 1.0,
        sigma_v: float = 1.0,
        use_empirical_variance: bool = False,
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
        self.beta_v = beta_v
        self.sigma_q = sigma_q
        self.sigma_v = sigma_v
        self.use_empirical_variance = use_empirical_variance
        self.multi_step_horizon = multi_step_horizon
        self.target_variance = target_variance
        self.target_spike_rate = target_spike_rate
        self.q_dim = q_dim
        self.p_dim = p_dim

    @classmethod
    def from_config(cls, cfg: LossConfig, q_dim: int = 32, p_dim: int = 96) -> "PredictorLoss":
        return cls(
            lambda_pred=cfg.lambda_pred,
            lambda_multi=cfg.lambda_multi,
            lambda_var=cfg.lambda_var,
            lambda_sparse=cfg.lambda_sparse,
            lambda_vel=cfg.lambda_vel,
            lambda_probe=cfg.lambda_probe,
            lambda_coord=cfg.lambda_coord,
            beta_v=cfg.beta_v,
            sigma_q=cfg.sigma_q,
            sigma_v=cfg.sigma_v,
            use_empirical_variance=cfg.use_empirical_variance,
            multi_step_horizon=cfg.multi_step_horizon,
            target_variance=cfg.target_variance,
            target_spike_rate=cfg.target_spike_rate,
            q_dim=q_dim,
            p_dim=p_dim,
        )

    def variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        flat_z = z.reshape(-1, z.shape[-1])
        std = torch.sqrt(flat_z.var(dim=0) + 1e-4)
        return torch.mean(F.relu(self.target_variance - std))

    def coordinate_loss(self, keypoints: torch.Tensor, true_kinematics: torch.Tensor) -> torch.Tensor:
        if keypoints.dim() == 2 and true_kinematics.dim() == 3:
            gt_xy = true_kinematics[:, -1, :2]
        else:
            gt_xy = true_kinematics[..., :2]
        k = keypoints.shape[-1] // 2
        kp = keypoints.view(*keypoints.shape[:-1], k, 2)
        gt_expanded = gt_xy.unsqueeze(-2).expand_as(kp)
        return F.mse_loss(kp, gt_expanded)


    def multi_step_rollout_loss(self, latent_states: torch.Tensor, model_predictor: nn.Module) -> torch.Tensor:
        b, t, d = latent_states.shape
        k = self.multi_step_horizon
        if t <= k + 1:
            return torch.tensor(0.0, device=latent_states.device)

        stride = max(1, k)
        t_starts = list(range(0, t - k, stride))
        if not t_starts:
            return torch.tensor(0.0, device=latent_states.device)

        curr_z = latent_states[:, t_starts].detach()
        step_losses = []
        for step in range(1, k + 1):
            target_step_z = torch.stack([latent_states[:, ts + step].detach() for ts in t_starts], dim=1)
            pred_out = model_predictor(curr_z)
            curr_z = pred_out.predicted_latent
            step_losses.append(F.mse_loss(curr_z, target_step_z))

        return torch.stack(step_losses).mean()

    def velocity_loss_frozen_probe(
        self,
        pred_z: torch.Tensor,
        true_kinematics: torch.Tensor,
        physical_decoder: nn.Module,
        stride_k: int = 5,
        dt: float = 0.01,
    ) -> torch.Tensor:
        b, t_tot, d_kin = true_kinematics.shape
        t_pred = pred_z.shape[1]
        true_pos = true_kinematics[..., :2]

        if t_tot <= stride_k:
            kin_target = true_kinematics[:, 1 : t_pred + 1, 2:4]
            with _frozen(physical_decoder):
                pred_decoded = physical_decoder(pred_z)
            pred_sub = pred_decoded[..., 2:4].contiguous()
            return F.smooth_l1_loss(pred_sub, kin_target)

        target_disp = true_pos[:, stride_k:, :] - true_pos[:, :-stride_k, :]
        target_vel = target_disp / (stride_k * dt)
        t_valid = min(t_pred, target_vel.shape[1])
        pred_z_valid = pred_z[:, :t_valid].contiguous()
        target_vel_valid = target_vel[:, :t_valid].contiguous()

        with _frozen(physical_decoder):
            pred_decoded = physical_decoder(pred_z_valid)

        pred_sub = pred_decoded[..., 2:4].contiguous()
        return F.smooth_l1_loss(pred_sub, target_vel_valid, beta=1.0)

    def kinematic_loss(
        self,
        decoded_kinematics: torch.Tensor,
        true_kinematics: torch.Tensor,
        num_objects: int = 1,
    ) -> torch.Tensor:
        pred_pos = decoded_kinematics[..., : 2 * num_objects]
        pred_vel = decoded_kinematics[..., 2 * num_objects : 4 * num_objects]
        true_pos = true_kinematics[..., : 2 * num_objects]
        true_vel = true_kinematics[..., 2 * num_objects : 4 * num_objects]

        if self.use_empirical_variance:
            sigma_q_sq = true_pos.var().clamp(min=1e-4)
            sigma_v_sq = true_vel.var().clamp(min=1e-4)
        else:
            sigma_q_sq = max(1e-4, float(self.sigma_q) ** 2)
            sigma_v_sq = max(1e-4, float(self.sigma_v) ** 2)

        l_pos = F.mse_loss(pred_pos, true_pos) / sigma_q_sq
        l_vel = F.mse_loss(pred_vel, true_vel) / sigma_v_sq
        return l_pos + self.beta_v * l_vel

    def forward(
        self,
        predicted_latent: torch.Tensor,
        target_latents: torch.Tensor,
        all_latents: torch.Tensor,
        model_predictor: Optional[nn.Module] = None,
        physical_decoder: Optional[nn.Module] = None,
        true_kinematics: Optional[torch.Tensor] = None,
        spikes: Optional[torch.Tensor] = None,
        is_warmup: bool = False,
    ) -> Dict[str, torch.Tensor]:
        device = predicted_latent.device
        l_1step = F.mse_loss(predicted_latent, target_latents)

        l_multi = torch.tensor(0.0, device=device)
        lambda_multi = self.lambda_multi if not is_warmup else 0.0
        if lambda_multi > 0.0 and model_predictor is not None:
            l_multi = self.multi_step_rollout_loss(all_latents, model_predictor)

        l_vel = torch.tensor(0.0, device=device)
        if self.lambda_vel > 0.0 and true_kinematics is not None and physical_decoder is not None:
            l_vel_k1 = self.velocity_loss_frozen_probe(predicted_latent, true_kinematics, physical_decoder, stride_k=1, dt=0.01)
            l_vel_k3 = self.velocity_loss_frozen_probe(predicted_latent, true_kinematics, physical_decoder, stride_k=3, dt=0.01)
            l_vel = 0.5 * l_vel_k1 + 0.5 * l_vel_k3

        l_sparse = torch.tensor(0.0, device=device)
        if self.lambda_sparse > 0.0 and spikes is not None:
            l_sparse = torch.abs(spikes.mean() - self.target_spike_rate)

        l_var = torch.tensor(0.0, device=device)
        if self.lambda_var > 0.0:
            l_var = self.variance_loss(predicted_latent)

        total_loss = (
            self.lambda_pred * l_1step
            + lambda_multi * l_multi
            + self.lambda_vel * l_vel
            + self.lambda_sparse * l_sparse
            + self.lambda_var * l_var
        )

        return {
            "total_loss": total_loss,
            "l_1step": l_1step,
            "l_multi": l_multi,
            "l_vel": l_vel,
            "l_sparse": l_sparse,
            "l_var": l_var,
        }

    compute_predictor_loss = forward


class CorrectorRolloutLoss(nn.Module):
    """
    Weighted comb (% 25) kinematic rollout loss for the gain-field corrector.
    Single source of truth used identically in training and validation.
    """

    def __init__(self, beta_v: float = 2.0) -> None:
        super().__init__()
        self.beta_v = beta_v

    def forward(
        self,
        decoded_rollout: torch.Tensor,
        target_kinematics: torch.Tensor,
        burn_in_steps: int = 0,
        num_objects: int = 1,
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
            time_weights = 1.0 / (1.0 + dist_to_0)
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
        loss_corr = l_kin

        return {
            "total_loss": loss_corr,
            "loss_corr": loss_corr,
            "l_corr": loss_corr,
            "l_corr_asymptotic": l_kin,
            "l_kin": l_kin,
            "l_pos": l_pos,
            "l_vel": l_vel,
        }

    compute_corrector_rollout_loss = forward


class SPWMLoss(nn.Module):
    """
    Unified loss module combining PredictorLoss and CorrectorRolloutLoss.
    Provides backward-compatible interface with pure composable components.
    """

    def __init__(
        self,
        lambda_pred: float = 1.0,
        lambda_multi: float = 0.5,
        lambda_var: float = 0.1,
        lambda_sparse: float = 0.5,
        lambda_vel: float = 0.5,
        lambda_probe: float = 2.0,
        lambda_coord: float = 0.15,
        beta_v: float = 2.0,
        sigma_q: float = 1.0,
        sigma_v: float = 1.0,
        use_empirical_variance: bool = False,
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
        self.beta_v = beta_v
        self.multi_step_horizon = multi_step_horizon

        self.predictor_loss = PredictorLoss(
            lambda_pred=lambda_pred,
            lambda_multi=lambda_multi,
            lambda_var=lambda_var,
            lambda_sparse=lambda_sparse,
            lambda_vel=lambda_vel,
            lambda_probe=lambda_probe,
            lambda_coord=lambda_coord,
            beta_v=beta_v,
            sigma_q=sigma_q,
            sigma_v=sigma_v,
            use_empirical_variance=use_empirical_variance,
            multi_step_horizon=multi_step_horizon,
            target_variance=target_variance,
            target_spike_rate=target_spike_rate,
            q_dim=q_dim,
            p_dim=p_dim,
        )

        self.corrector_loss = CorrectorRolloutLoss(
            beta_v=beta_v,
        )
        self.multi_step_horizon = multi_step_horizon
        self.beta_v = beta_v

    @classmethod
    def from_config(cls, cfg: LossConfig, q_dim: int = 32, p_dim: int = 96) -> "SPWMLoss":
        return cls(
            lambda_pred=cfg.lambda_pred,
            lambda_multi=cfg.lambda_multi,
            lambda_var=cfg.lambda_var,
            lambda_sparse=cfg.lambda_sparse,
            lambda_vel=cfg.lambda_vel,
            lambda_probe=cfg.lambda_probe,
            lambda_coord=cfg.lambda_coord,
            beta_v=cfg.beta_v,
            sigma_q=cfg.sigma_q,
            sigma_v=cfg.sigma_v,
            use_empirical_variance=cfg.use_empirical_variance,
            multi_step_horizon=cfg.multi_step_horizon,
            target_variance=cfg.target_variance,
            target_spike_rate=cfg.target_spike_rate,
            q_dim=q_dim,
            p_dim=p_dim,
        )

    def variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor_loss.variance_loss(z)

    def coordinate_loss(self, keypoints: torch.Tensor, true_kinematics: torch.Tensor) -> torch.Tensor:
        return self.predictor_loss.coordinate_loss(keypoints, true_kinematics)

    def multi_step_rollout_loss(self, latent_states: torch.Tensor, model_predictor: nn.Module) -> torch.Tensor:
        return self.predictor_loss.multi_step_rollout_loss(latent_states, model_predictor)

    def velocity_loss_frozen_probe(self, *args, **kwargs) -> torch.Tensor:
        return self.predictor_loss.velocity_loss_frozen_probe(*args, **kwargs)

    def kinematic_loss(self, *args, **kwargs) -> torch.Tensor:
        return self.predictor_loss.kinematic_loss(*args, **kwargs)

    def compute_predictor_loss(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        return self.predictor_loss(*args, **kwargs)

    def compute_corrector_rollout_loss(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        return self.corrector_loss(*args, **kwargs)