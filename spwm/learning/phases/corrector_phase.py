"""
Corrector Phase Handler for SPWM.
Owns the slow corrector optimizer, enforces strict freezing of the base predictor,
and computes asymptotic rollout loss via the unified rollout engine.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import chronicle
import numpy as np
import torch
import torch.nn as nn

from spwm.config.schema import ExperimentConfig
from spwm.learning.losses import SPWMLoss
from spwm.learning.metrics import position_error, velocity_error
from spwm.learning.phases.base import PhaseHandler
from spwm.models.rollout import rollout_latent


class CorrectorPhaseHandler(PhaseHandler):
    """
    Handles Phase 2: Gain Field Slow Corrector Training.
    Predictor and encoder are strictly frozen.
    """

    def setup(self) -> None:
        if not hasattr(self.model, "corrector") or self.model.corrector is None:
            raise ValueError("CorrectorPhaseHandler requires model to have an enabled corrector.")

        # Strictly freeze all non-corrector parameters
        for name, param in self.model.named_parameters():
            if "corrector" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        lr = self.cfg.training.corrector_lr
        wd = self.cfg.training.weight_decay
        self.corrector_optimizer = torch.optim.Adam(
            self.model.corrector.parameters(), lr=lr, weight_decay=wd
        )

    def train_step(self, batch: Dict[str, torch.Tensor], epoch: int) -> Dict[str, float]:
        self.model.eval()  # Base model in eval mode
        self.model.corrector.train()  # Corrector in train mode

        events = batch["events"].to(self.device)
        flat_kin = batch["flat_kinematics"].to(self.device)
        b, t, _, _, _ = events.shape

        with torch.no_grad():
            enc_latents = self.model(events, accumulate_local_updates=False).latent_states

        t_warmup = min(10, max(1, t // 4))
        z_start = enc_latents[:, t_warmup].detach()

        if self.cfg.training.perturb_corrector_states:
            perturb_std = self.cfg.training.corrector_perturbation_std
            z_start = z_start + torch.randn_like(z_start) * perturb_std

        horizon = min(self.cfg.training.corrector_horizon, t - t_warmup - 1)
        burn_in = min(self.cfg.training.corrector_burn_in_steps, max(0, horizon - 1))

        # Execute differentiable rollout via single unified rollout engine

        rollout_res = rollout_latent(
            model=self.model,
            z_init=z_start,
            horizon=horizon,
            use_corrector=True,
            burn_in_steps=burn_in,
            with_grad=True,
            decode_kinematics=True,
        )

        target_kin = flat_kin[:, t_warmup + 1 : t_warmup + 1 + horizon]
        corr_loss_dict = self.loss_fn.compute_corrector_rollout_loss(
            decoded_rollout=rollout_res.decoded_kinematics,
            target_kinematics=target_kin,
            burn_in_steps=burn_in,
            num_objects=self.cfg.environment.num_objects,
        )

        self.corrector_optimizer.zero_grad()
        corr_loss_dict["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.corrector.parameters(), self.cfg.training.grad_clip_norm
        )
        self.corrector_optimizer.step()

        with torch.no_grad():
            pos_err = position_error(rollout_res.decoded_kinematics, target_kin, num_objects=self.cfg.environment.num_objects)
            vel_err = velocity_error(rollout_res.decoded_kinematics, target_kin, num_objects=self.cfg.environment.num_objects)

        return {
            "total_loss": float(corr_loss_dict["total_loss"].item()),
            "loss_corr": float(corr_loss_dict["loss_corr"].item()),
            "l_corr": float(corr_loss_dict["l_corr"].item()),
            "l_corr_asymptotic": float(corr_loss_dict["l_corr_asymptotic"].item()),
            "l_kin": float(corr_loss_dict["l_kin"].item()),
            "l_pos": float(corr_loss_dict["l_pos"].item()),
            "l_vel": float(corr_loss_dict["l_vel"].item()),
            "train_pos_err": pos_err,
            "train_vel_err": vel_err,
        }


    @torch.no_grad()
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.eval()
        events = batch["events"].to(self.device)
        flat_kin = batch["flat_kinematics"].to(self.device)
        b, t, _, _, _ = events.shape

        enc_latents = self.model(events, accumulate_local_updates=False).latent_states
        t_warmup = min(10, max(1, t // 4))
        z_start = enc_latents[:, t_warmup]

        horizon = min(max(self.cfg.evaluation.rollout_horizons), t - t_warmup - 1)
        burn_in = min(self.cfg.training.corrector_burn_in_steps, max(0, horizon - 1))


        rollout_res = rollout_latent(
            model=self.model,
            z_init=z_start,
            horizon=horizon,
            use_corrector=True,
            burn_in_steps=burn_in,
            with_grad=False,
            decode_kinematics=True,
        )

        target_kin = flat_kin[:, t_warmup + 1 : t_warmup + 1 + horizon]
        corr_loss_dict = self.loss_fn.compute_corrector_rollout_loss(
            decoded_rollout=rollout_res.decoded_kinematics,
            target_kinematics=target_kin,
            burn_in_steps=burn_in,
            num_objects=self.cfg.environment.num_objects,
        )

        # Compute multi-horizon position errors
        h1_pos = position_error(rollout_res.decoded_kinematics[:, :1], target_kin[:, :1], num_objects=self.cfg.environment.num_objects)
        h25_len = min(25, horizon)
        h25_pos = position_error(rollout_res.decoded_kinematics[:, :h25_len], target_kin[:, :h25_len], num_objects=self.cfg.environment.num_objects)
        drift = float(h25_pos / max(h1_pos, 1e-6))

        return {
            "val_total_loss": float(corr_loss_dict["total_loss"].item()),
            "val_loss_corr": float(corr_loss_dict["loss_corr"].item()),
            "val_l_kin": float(corr_loss_dict["l_kin"].item()),
            "val_l_pos": float(corr_loss_dict["l_pos"].item()),
            "val_l_vel": float(corr_loss_dict["l_vel"].item()),
            "val_pos_err_h1": h1_pos,
            "val_pos_err_h25": h25_pos,
            "val_drift_ratio": drift,
            "val_pos_err": h25_pos,
        }

    def get_selection_score(self, val_metrics: Dict[str, float]) -> float:
        # Best model selected directly by the %25-weighted position and velocity rollout loss
        return val_metrics.get("val_total_loss", float("inf"))

    def log_epoch(
        self,
        epoch: int,
        end_epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch_duration: float,
        total_elapsed: float,
        is_best: bool,
        best_epoch: Optional[int],
        best_score: float,
    ) -> None:
        best_tag = " * [BEST]" if is_best else ""
        alpha = self.cfg.corrector.alpha_gain
        h = self.cfg.training.corrector_horizon
        b_in = self.cfg.training.corrector_burn_in_steps
        header = (
            f"Epoch [{epoch:03d}/{end_epoch:03d}] [CORRECTOR-PHASE] (H={h}, burn-in={b_in}, alpha={alpha:.2f})  "
            f"Time: {epoch_duration:5.1f}s | Total: {total_elapsed / 60:4.1f}m{best_tag} | "
            f"Remaining: {epoch_duration * (end_epoch - epoch) / 60:4.1f}m"
        )
        chronicle.log_section_header(header)
        chronicle.log_detail("🔒 Predictor Core", "FROZEN (eval mode).", indent_level=1)

        # Log metrics read directly from loss dictionary, NEVER assuming formulas!
        l_kin = train_metrics.get("l_kin", 0.0)
        l_pos = train_metrics.get("l_pos", 0.0)
        l_vel = train_metrics.get("l_vel", 0.0)
        l_total = train_metrics.get("total_loss", 0.0)
        chronicle.log_detail(
            "🧠 Gain Field Corrector",
            f"L_total: {l_total:.5f} | L_kin: {l_kin:.5f} | L_pos: {l_pos:.5f} | L_vel: {l_vel:.5f}",
            indent_level=1,
        )

        h1 = val_metrics.get("val_pos_err_h1", 0.0)
        h25 = val_metrics.get("val_pos_err_h25", 0.0)
        drift = val_metrics.get("val_drift_ratio", 0.0)
        chronicle.log_detail(
            "📈 Autonomous Rollout",
            f"Val L_kin: {val_metrics.get('val_l_kin', 0.0):.5f} | H1 Pos: {h1:.4f} | H25 Pos: {h25:.4f} | Drift: {drift:.2f}x",
            indent_level=1,
        )
        curr_score = self.get_selection_score(val_metrics)
        chronicle.log_detail(
            "🎯 Model Selection",
            f"Score: {curr_score:.5f} | Best: {best_score:.5f} @ Ep {best_epoch}",
            indent_level=1,
        )
        chronicle.log_newline()

    def get_optimizers_state(self) -> Dict[str, Any]:
        return {
            "corrector": self.corrector_optimizer.state_dict(),
        }

    def load_optimizers_state(self, state: Dict[str, Any]) -> None:
        if "corrector" in state:
            self.corrector_optimizer.load_state_dict(state["corrector"])
