"""
Predictor Phase Handler for SPWM.
Owns the 3 fast optimizers (encoder, predictor/dynamics, probe),
warmup curriculum, coordinate supervision, and step-by-step training step.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import chronicle
import torch
import torch.nn as nn
import torch.nn.functional as F

from spwm.config.schema import ExperimentConfig
from spwm.learning.losses import SPWMLoss
from spwm.learning.metrics import position_error, velocity_error
from spwm.learning.phases.base import PhaseHandler


class PredictorPhaseHandler(PhaseHandler):
    """
    Handles Phase 1: Fast Predictor & Kinematic Alignment.
    Corrector is strictly disabled/frozen.
    """

    def setup(self) -> None:
        # Freeze corrector if present
        if hasattr(self.model, "corrector") and self.model.corrector is not None:
            for p in self.model.corrector.parameters():
                p.requires_grad = False

        # Ensure all predictor components are trainable
        trainable_modules = [
            getattr(self.model, "encoder", None),
            getattr(self.model, "dynamics", None),
            getattr(self.model, "predictor", None),
            getattr(self.model, "sensory_predictor", None),
            getattr(self.model, "physical_decoder", None),
        ]
        for mod in trainable_modules:
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = True

        lr = self.cfg.training.learning_rate
        wd = self.cfg.training.weight_decay
        probe_lr = self.cfg.training.probe_lr
        probe_wd = self.cfg.training.probe_weight_decay

        self.encoder_optimizer = torch.optim.Adam(
            self.model.encoder.parameters(), lr=lr, weight_decay=wd
        )

        predictor_params = (
            list(self.model.dynamics.parameters())
            + list(self.model.predictor.parameters())
            + list(self.model.sensory_predictor.parameters())
        )
        self.predictor_optimizer = torch.optim.Adam(
            predictor_params, lr=lr, weight_decay=wd
        )

        self.probe_optimizer = torch.optim.Adam(
            self.model.physical_decoder.parameters(), lr=probe_lr, weight_decay=probe_wd
        )

    def train_step(self, batch: Dict[str, torch.Tensor], epoch: int) -> Dict[str, float]:
        self.model.train()
        events = batch["events"].to(self.device)
        flat_kin = batch["flat_kinematics"].to(self.device)

        is_warmup = epoch <= self.cfg.training.encoder_warmup_epochs
        clip_norm = self.cfg.training.grad_clip_norm

        self.probe_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        self.predictor_optimizer.zero_grad()

        out = self.model(events, accumulate_local_updates=False)

        # 1. SpatialSoftmax Coordinate Supervision (Encoder Optimization)
        l_coord = torch.tensor(0.0, device=self.device)
        if self.loss_fn.lambda_coord > 0.0:
            with torch.enable_grad():
                _, kp_seq = self.model.encoder(events, return_keypoints=True)
                l_coord = self.loss_fn.coordinate_loss(kp_seq, flat_kin)
                (self.loss_fn.lambda_coord * l_coord).backward()
                torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), clip_norm)
                self.encoder_optimizer.step()

        # 2. Physical Decoder Probe Optimization
        decoded_kin = self.model.physical_decoder(out.latent_states.detach())
        l_probe = self.loss_fn.kinematic_loss(decoded_kin, flat_kin, num_objects=self.cfg.environment.num_objects)
        (self.loss_fn.lambda_probe * l_probe).backward()
        torch.nn.utils.clip_grad_norm_(self.model.physical_decoder.parameters(), clip_norm)
        self.probe_optimizer.step()

        # 3. Fast Predictor Optimization
        with torch.enable_grad():
            latents = out.latent_states.detach()
            pred_res = self.model.predictor(latents[:, :-1])
            pred_z = pred_res.predicted_latent

            all_spikes = (
                torch.cat([out.fast_spikes, out.slow_spikes], dim=-1)
                if getattr(out, "fast_spikes", None) is not None and getattr(out, "slow_spikes", None) is not None
                else getattr(out, "fast_spikes", None)
            )

            pred_loss = self.loss_fn.compute_predictor_loss(
                predicted_latent=pred_z,
                target_latents=latents[:, 1:],
                all_latents=latents,
                model_predictor=self.model.predictor,
                physical_decoder=self.model.physical_decoder,
                true_kinematics=flat_kin,
                spikes=all_spikes,
                is_warmup=is_warmup,
            )
            pred_loss["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.dynamics.parameters(), clip_norm)
            torch.nn.utils.clip_grad_norm_(self.model.predictor.parameters(), clip_norm)
            self.predictor_optimizer.step()



        # Compute position & velocity error for tracking
        with torch.no_grad():
            pos_err = position_error(decoded_kin, flat_kin, num_objects=self.cfg.environment.num_objects)
            vel_err = velocity_error(decoded_kin, flat_kin, num_objects=self.cfg.environment.num_objects)

        metrics = {
            "total_loss": float(pred_loss["total_loss"].item()),
            "l_pred": float(pred_loss["l_1step"].item()),
            "l_multi": float(pred_loss["l_multi"].item()),
            "l_vel": float(pred_loss["l_vel"].item()),
            "l_sparse": float(pred_loss["l_sparse"].item()),
            "l_var": float(pred_loss["l_var"].item()),
            "l_coord": float(l_coord.item()),
            "l_probe": float(l_probe.item()),
            "l_corr": 0.0,
            "l_corr_asymptotic": 0.0,
            "train_pos_err": pos_err,
            "train_vel_err": vel_err,
        }
        return metrics


    @torch.no_grad()
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.eval()
        events = batch["events"].to(self.device)
        flat_kin = batch["flat_kinematics"].to(self.device)

        out = self.model(events, accumulate_local_updates=False)

        l_coord = torch.tensor(0.0, device=self.device)
        last_kp = getattr(self.model.encoder, "last_keypoints", None)
        if last_kp is not None:
            l_coord = self.loss_fn.coordinate_loss(last_kp, flat_kin)

        decoded_kin = self.model.physical_decoder(out.latent_states)
        l_probe = self.loss_fn.kinematic_loss(decoded_kin, flat_kin, num_objects=self.cfg.environment.num_objects)

        pred_loss = self.loss_fn.compute_predictor_loss(
            predicted_latent=out.predicted_latents[:, :-1],
            target_latents=out.latent_states[:, 1:],
            all_latents=out.latent_states,
            model_predictor=self.model.predictor,
            physical_decoder=self.model.physical_decoder,
            true_kinematics=flat_kin,
            spikes=out.fast_spikes,
            is_warmup=False,
        )

        pos_err = position_error(decoded_kin, flat_kin, num_objects=self.cfg.environment.num_objects)
        vel_err = velocity_error(decoded_kin, flat_kin, num_objects=self.cfg.environment.num_objects)

        return {
            "val_total_loss": float(pred_loss["total_loss"].item()),
            "val_l_pred": float(pred_loss["l_1step"].item()),
            "val_l_multi": float(pred_loss["l_multi"].item()),
            "val_l_vel": float(pred_loss["l_vel"].item()),
            "val_l_coord": float(l_coord.item()),
            "val_l_probe": float(l_probe.item()),
            "val_pos_err": pos_err,
            "val_vel_err": vel_err,
        }

    def get_selection_score(self, val_metrics: Dict[str, float]) -> float:
        # Score combines total loss and kinematic position error
        return val_metrics.get("val_total_loss", float("inf")) + val_metrics.get("val_pos_err", float("inf"))

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
        warmup_tag = " [WARMUP]" if epoch <= self.cfg.training.encoder_warmup_epochs else ""
        header = (
            f"Epoch [{epoch:03d}/{end_epoch:03d}] [PREDICTOR-PHASE]{warmup_tag}  "
            f"Time: {epoch_duration:5.1f}s | Total: {total_elapsed / 60:4.1f}m{best_tag} | "
            f"Remaining: {epoch_duration * (end_epoch - epoch) / 60:4.1f}m"
        )
        chronicle.log_section_header(header)
        chronicle.log_detail(
            "⚡ Fast Predictor Optimization",
            f"L_1step: {train_metrics.get('l_pred', 0.0):.5f} | "
            f"L_multi: {train_metrics.get('l_multi', 0.0):.5f} | "
            f"L_vel: {train_metrics.get('l_vel', 0.0):.5f} | "
            f"L_sparse: {train_metrics.get('l_sparse', 0.0):.5f}",
            indent_level=1,
        )
        chronicle.log_detail(
            "🎯 Representation & Probe",
            f"L_coord: {train_metrics.get('l_coord', 0.0):.5f} | "
            f"L_probe: {train_metrics.get('l_probe', 0.0):.5f}",
            indent_level=1,
        )
        chronicle.log_detail(
            "📊 Validation Kinematics",
            f"Val Loss: {val_metrics.get('val_total_loss', 0.0):.5f} | "
            f"Pos: {val_metrics.get('val_pos_err', 0.0):.4f} | "
            f"Vel: {val_metrics.get('val_vel_err', 0.0):.4f}",
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
            "encoder": self.encoder_optimizer.state_dict(),
            "predictor": self.predictor_optimizer.state_dict(),
            "probe": self.probe_optimizer.state_dict(),
        }

    def load_optimizers_state(self, state: Dict[str, Any]) -> None:
        if "encoder" in state:
            self.encoder_optimizer.load_state_dict(state["encoder"])
        if "predictor" in state:
            self.predictor_optimizer.load_state_dict(state["predictor"])
        if "probe" in state:
            self.probe_optimizer.load_state_dict(state["probe"])
