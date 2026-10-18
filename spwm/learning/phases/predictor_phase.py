"""
Predictor Phase Handler for SPWM.
Implements Two-Phase Training Curriculum for the Predictor Model:
- Mode "train_representation":
    Freeze the ALIF recurrent predictor / dynamics.
    Train spatial encoder and physical decoder probe.
    Compute and backpropagate only loss_encoder and loss_decoder.
    Model selection tracks total_loss (val_total_loss).
- Mode "train_predictor":
    Freeze the SpatialSoftmax encoder and physical decoder probe.
    Train the ALIF recurrent predictor / dynamics.
    Compute and backpropagate only loss_position and loss_velocity over a multi-step rollout.
    Model selection tracks total_loss (val_total_loss).

Enforces 1:1 symmetry between train_step and validate_step.
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
from spwm.models.rollout import rollout_latent


class PredictorPhaseHandler(PhaseHandler):
    """
    Handles Predictor & Representation phases.
    Supports modes: "train_representation" and "train_predictor".
    Corrector is strictly disabled/frozen.
    """

    def setup(self) -> None:
        mode = self.cfg.training.mode.lower()

        # Freeze corrector if present
        if hasattr(self.model, "corrector") and self.model.corrector is not None:
            for p in self.model.corrector.parameters():
                p.requires_grad = False

        lr = self.cfg.training.learning_rate
        wd = self.cfg.training.weight_decay
        probe_lr = self.cfg.training.probe_lr
        probe_wd = self.cfg.training.probe_weight_decay

        if mode == "train_representation":
            # Freeze ALIF recurrent predictor and dynamics
            if hasattr(self.model, "predictor") and self.model.predictor is not None:
                for p in self.model.predictor.parameters():
                    p.requires_grad = False
            if hasattr(self.model, "dynamics") and self.model.dynamics is not None:
                for p in self.model.dynamics.parameters():
                    p.requires_grad = False
            if hasattr(self.model, "sensory_predictor") and self.model.sensory_predictor is not None:
                for p in self.model.sensory_predictor.parameters():
                    p.requires_grad = False

            # Train encoder and physical decoder
            if hasattr(self.model, "encoder") and self.model.encoder is not None:
                for p in self.model.encoder.parameters():
                    p.requires_grad = True
            if hasattr(self.model, "physical_decoder") and self.model.physical_decoder is not None:
                for p in self.model.physical_decoder.parameters():
                    p.requires_grad = True

            repr_params = []
            if hasattr(self.model, "encoder") and self.model.encoder is not None:
                repr_params.extend(list(self.model.encoder.parameters()))
            if hasattr(self.model, "physical_decoder") and self.model.physical_decoder is not None:
                repr_params.extend(list(self.model.physical_decoder.parameters()))

            self.representation_optimizer = torch.optim.Adam(
                repr_params, lr=lr, weight_decay=wd
            )
            self.predictor_optimizer = None

        elif mode == "train_joint":
            # Joint training mode: both representation and predictor are active
            # Train encoder and physical decoder
            repr_params = []
            if hasattr(self.model, "encoder") and self.model.encoder is not None:
                for p in self.model.encoder.parameters():
                    p.requires_grad = True
                repr_params.extend(list(self.model.encoder.parameters()))
            if hasattr(self.model, "physical_decoder") and self.model.physical_decoder is not None:
                for p in self.model.physical_decoder.parameters():
                    p.requires_grad = True
                repr_params.extend(list(self.model.physical_decoder.parameters()))

            self.representation_optimizer = torch.optim.Adam(
                repr_params, lr=lr, weight_decay=wd
            )

            # Train predictor / dynamics
            pred_params = []
            if hasattr(self.model, "predictor") and self.model.predictor is not None:
                for p in self.model.predictor.parameters():
                    p.requires_grad = True
                pred_params.extend(list(self.model.predictor.parameters()))
            if hasattr(self.model, "dynamics") and self.model.dynamics is not None:
                for p in self.model.dynamics.parameters():
                    p.requires_grad = True
                pred_params.extend(list(self.model.dynamics.parameters()))
            if hasattr(self.model, "sensory_predictor") and self.model.sensory_predictor is not None:
                for p in self.model.sensory_predictor.parameters():
                    p.requires_grad = True
                pred_params.extend(list(self.model.sensory_predictor.parameters()))

            self.predictor_optimizer = torch.optim.Adam(
                pred_params, lr=lr, weight_decay=wd
            )

        else:
            # Mode "train_predictor"
            # Freeze SpatialSoftmax encoder and physical decoder probe
            if hasattr(self.model, "encoder") and self.model.encoder is not None:
                for p in self.model.encoder.parameters():
                    p.requires_grad = False
            if hasattr(self.model, "physical_decoder") and self.model.physical_decoder is not None:
                for p in self.model.physical_decoder.parameters():
                    p.requires_grad = False

            # Train predictor / dynamics
            pred_params = []
            if hasattr(self.model, "predictor") and self.model.predictor is not None:
                for p in self.model.predictor.parameters():
                    p.requires_grad = True
                pred_params.extend(list(self.model.predictor.parameters()))
            if hasattr(self.model, "dynamics") and self.model.dynamics is not None:
                for p in self.model.dynamics.parameters():
                    p.requires_grad = True
                pred_params.extend(list(self.model.dynamics.parameters()))
            if hasattr(self.model, "sensory_predictor") and self.model.sensory_predictor is not None:
                for p in self.model.sensory_predictor.parameters():
                    p.requires_grad = True
                pred_params.extend(list(self.model.sensory_predictor.parameters()))

            self.predictor_optimizer = torch.optim.Adam(
                pred_params, lr=lr, weight_decay=wd
            )
            self.representation_optimizer = None

    @property
    def encoder_optimizer(self):
        return self.representation_optimizer

    @property
    def probe_optimizer(self):
        return self.representation_optimizer

    def train_step(self, batch: Dict[str, torch.Tensor], epoch: int) -> Dict[str, float]:
        mode = self.cfg.training.mode.lower()
        clip_norm = self.cfg.training.grad_clip_norm
        events = batch["events"].to(self.device)
        flat_kin = batch["flat_kinematics"].to(self.device)

        if mode == "train_representation":
            self.model.eval()
            self.model.encoder.train()
            self.model.physical_decoder.train()

            self.representation_optimizer.zero_grad()

            _, keypoints = self.model.encoder(events, return_keypoints=True)
            out = self.model(events, accumulate_local_updates=False)
            decoded_kin = self.model.physical_decoder(out.latent_states)

            loss_dict = self.loss_fn.compute_loss(
                keypoints=keypoints,
                decoded_kinematics=decoded_kin,
                target_kinematics=flat_kin,
                num_objects=self.cfg.environment.num_objects,
                mode="train_representation",
            )

            loss_dict["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), clip_norm)
            torch.nn.utils.clip_grad_norm_(self.model.physical_decoder.parameters(), clip_norm)
            self.representation_optimizer.step()

        elif mode == "train_joint":
            self.model.eval()
            self.model.encoder.train()
            self.model.physical_decoder.train()
            self.model.predictor.train()

            self.representation_optimizer.zero_grad()
            self.predictor_optimizer.zero_grad()

            # 1. Representation forward pass
            _, keypoints = self.model.encoder(events, return_keypoints=True)
            out = self.model(events, accumulate_local_updates=False)
            decoded_kin = self.model.physical_decoder(out.latent_states)

            # 2. Predictor rollout starting from detached encoder latent
            enc_latents = out.latent_states.detach()
            b, t, d = enc_latents.shape
            burn_in = min(self.cfg.training.temporal_burn_in, max(0, t - 2))
            horizon = min(self.loss_fn.multi_step_horizon, t - 1 - burn_in)
            t_start = burn_in
            z_start = enc_latents[:, t_start]

            # Warm up physical decoder momentum filter on prefix latents [0:t_start+1]
            decoder_state = None
            if hasattr(self.model, "physical_decoder") and self.model.physical_decoder is not None:
                z_prefix = enc_latents[:, : t_start + 1]
                decoder_state = self.model.physical_decoder.warm_up_filter(z_prefix)

            rollout_res = rollout_latent(
                model=self.model,
                z_init=z_start,
                horizon=horizon,
                use_corrector=False,
                with_grad=True,
                decode_kinematics=True,
                initial_decoder_state=decoder_state,
            )

            target_rollout_kin = flat_kin[:, t_start + 1 : t_start + 1 + horizon]

            loss_dict = self.loss_fn.compute_loss(
                keypoints=keypoints,
                decoded_kinematics=decoded_kin,
                rollout_positions=rollout_res.positions,
                rollout_velocities=rollout_res.velocities,
                target_kinematics=flat_kin,
                target_rollout_kinematics=target_rollout_kin,
                num_objects=self.cfg.environment.num_objects,
                mode="train_joint",
            )

            # Compute combined loss: L_repr = L_enc + L_dec, L_pred = L_pos + L_vel
            l_repr = self.loss_fn.lambda_encoder * loss_dict["loss_encoder"] + self.loss_fn.lambda_decoder * loss_dict["loss_decoder"]
            l_pred = self.loss_fn.lambda_position * loss_dict["loss_position"] + self.loss_fn.lambda_velocity * loss_dict["loss_velocity"]
            total_loss = l_repr + l_pred

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), clip_norm)
            torch.nn.utils.clip_grad_norm_(self.model.physical_decoder.parameters(), clip_norm)
            torch.nn.utils.clip_grad_norm_(self.model.predictor.parameters(), clip_norm)

            self.representation_optimizer.step()
            self.predictor_optimizer.step()

        else:
            # Mode "train_predictor"
            self.model.eval()
            self.model.predictor.train()

            self.predictor_optimizer.zero_grad()

            with torch.no_grad():
                enc_latents = self.model(events, accumulate_local_updates=False).latent_states

            b, t, d = enc_latents.shape
            burn_in = min(self.cfg.training.temporal_burn_in, max(0, t - 2))
            horizon = min(self.loss_fn.multi_step_horizon, t - 1 - burn_in)
            t_start = burn_in
            z_start = enc_latents[:, t_start]

            decoder_state = None
            if hasattr(self.model, "physical_decoder") and self.model.physical_decoder is not None:
                z_prefix = enc_latents[:, : t_start + 1]
                decoder_state = self.model.physical_decoder.warm_up_filter(z_prefix)

            rollout_res = rollout_latent(
                model=self.model,
                z_init=z_start,
                horizon=horizon,
                use_corrector=False,
                with_grad=True,
                decode_kinematics=True,
                initial_decoder_state=decoder_state,
            )

            target_kin = flat_kin[:, t_start + 1 : t_start + 1 + horizon]

            loss_dict = self.loss_fn.compute_loss(
                rollout_positions=rollout_res.positions,
                rollout_velocities=rollout_res.velocities,
                target_kinematics=target_kin,
                num_objects=self.cfg.environment.num_objects,
                mode="train_predictor",
            )

            loss_dict["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.predictor.parameters(), clip_norm)
            self.predictor_optimizer.step()

        train_pos_err = 0.0
        train_vel_err = 0.0
        if mode in ("train_joint", "train_predictor") and rollout_res.decoded_kinematics is not None:
            train_pos_err = position_error(rollout_res.decoded_kinematics, target_rollout_kin if mode == "train_joint" else target_kin, num_objects=self.cfg.environment.num_objects)
            train_vel_err = velocity_error(rollout_res.decoded_kinematics, target_rollout_kin if mode == "train_joint" else target_kin, num_objects=self.cfg.environment.num_objects)

        return {
            "train_loss": float(loss_dict["total_loss"].item()),
            "train_loss_encoder": float(loss_dict["loss_encoder"].item()),
            "train_loss_decoder": float(loss_dict["loss_decoder"].item()),
            "train_loss_position": float(loss_dict["loss_position"].item()),
            "train_loss_velocity": float(loss_dict["loss_velocity"].item()),
            "train_pos_err": train_pos_err,
            "train_vel_err": train_vel_err,
        }

    @torch.no_grad()
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        mode = self.cfg.training.mode.lower()
        self.model.eval()
        events = batch["events"].to(self.device)
        flat_kin = batch["flat_kinematics"].to(self.device)

        val_pos_err = 0.0
        val_vel_err = 0.0

        if mode == "train_representation":
            _, keypoints = self.model.encoder(events, return_keypoints=True)
            out = self.model(events, accumulate_local_updates=False)
            decoded_kin = self.model.physical_decoder(out.latent_states)

            loss_dict = self.loss_fn.compute_loss(
                keypoints=keypoints,
                decoded_kinematics=decoded_kin,
                target_kinematics=flat_kin,
                num_objects=self.cfg.environment.num_objects,
                mode="train_representation",
            )
            if decoded_kin is not None:
                val_pos_err = position_error(decoded_kin, flat_kin, num_objects=self.cfg.environment.num_objects)
                val_vel_err = velocity_error(decoded_kin, flat_kin, num_objects=self.cfg.environment.num_objects)
        elif mode == "train_joint":
            _, keypoints = self.model.encoder(events, return_keypoints=True)
            out = self.model(events, accumulate_local_updates=False)
            decoded_kin = self.model.physical_decoder(out.latent_states)

            enc_latents = out.latent_states
            b, t, d = enc_latents.shape
            burn_in = min(self.cfg.training.temporal_burn_in, max(0, t - 2))
            horizon = min(self.loss_fn.multi_step_horizon, t - 1 - burn_in)
            t_start = burn_in
            z_start = enc_latents[:, t_start]

            decoder_state = None
            if hasattr(self.model, "physical_decoder") and self.model.physical_decoder is not None:
                z_prefix = enc_latents[:, : t_start + 1]
                decoder_state = self.model.physical_decoder.warm_up_filter(z_prefix)

            rollout_res = rollout_latent(
                model=self.model,
                z_init=z_start,
                horizon=horizon,
                use_corrector=False,
                with_grad=False,
                decode_kinematics=True,
                initial_decoder_state=decoder_state,
            )

            target_rollout_kin = flat_kin[:, t_start + 1 : t_start + 1 + horizon]

            loss_dict = self.loss_fn.compute_loss(
                keypoints=keypoints,
                decoded_kinematics=decoded_kin,
                rollout_positions=rollout_res.positions,
                rollout_velocities=rollout_res.velocities,
                target_kinematics=flat_kin,
                target_rollout_kinematics=target_rollout_kin,
                num_objects=self.cfg.environment.num_objects,
                mode="train_joint",
            )
            if rollout_res.decoded_kinematics is not None:
                val_pos_err = position_error(rollout_res.decoded_kinematics, target_rollout_kin, num_objects=self.cfg.environment.num_objects)
                val_vel_err = velocity_error(rollout_res.decoded_kinematics, target_rollout_kin, num_objects=self.cfg.environment.num_objects)
        else:
            enc_latents = self.model(events, accumulate_local_updates=False).latent_states
            b, t, d = enc_latents.shape
            burn_in = min(self.cfg.training.temporal_burn_in, max(0, t - 2))
            horizon = min(self.loss_fn.multi_step_horizon, t - 1 - burn_in)
            t_start = burn_in
            z_start = enc_latents[:, t_start]

            decoder_state = None
            if hasattr(self.model, "physical_decoder") and self.model.physical_decoder is not None:
                z_prefix = enc_latents[:, : t_start + 1]
                decoder_state = self.model.physical_decoder.warm_up_filter(z_prefix)

            rollout_res = rollout_latent(
                model=self.model,
                z_init=z_start,
                horizon=horizon,
                use_corrector=False,
                with_grad=False,
                decode_kinematics=True,
                initial_decoder_state=decoder_state,
            )

            target_kin = flat_kin[:, t_start + 1 : t_start + 1 + horizon]

            loss_dict = self.loss_fn.compute_loss(
                rollout_positions=rollout_res.positions,
                rollout_velocities=rollout_res.velocities,
                target_kinematics=target_kin,
                num_objects=self.cfg.environment.num_objects,
                mode="train_predictor",
            )
            if rollout_res.decoded_kinematics is not None:
                val_pos_err = position_error(rollout_res.decoded_kinematics, target_kin, num_objects=self.cfg.environment.num_objects)
                val_vel_err = velocity_error(rollout_res.decoded_kinematics, target_kin, num_objects=self.cfg.environment.num_objects)

        # Exact 1:1 symmetry with train_step (returns identical dictionary keys)
        return {
            "val_loss": float(loss_dict["total_loss"].item()),
            "val_loss_encoder": float(loss_dict["loss_encoder"].item()),
            "val_loss_decoder": float(loss_dict["loss_decoder"].item()),
            "val_loss_position": float(loss_dict["loss_position"].item()),
            "val_loss_velocity": float(loss_dict["loss_velocity"].item()),
            "val_pos_err": val_pos_err,
            "val_vel_err": val_vel_err,
        }

    def get_selection_score(self, val_metrics: Dict[str, float]) -> float:
        return val_metrics.get("val_loss", float("inf"))

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
        mode = self.cfg.training.mode.upper()
        header = (
            f"Epoch [{epoch:03d}/{end_epoch:03d}] [{mode}]{best_tag}  "
            f"Time: {epoch_duration:5.1f}s | Total: {total_elapsed / 60:4.1f}m | "
            f"Remaining: {epoch_duration * (end_epoch - epoch) / 60:4.1f}m"
        )
        chronicle.log_section_header(header)
        chronicle.log_detail(
            "⚡ Training Losses",
            f"Train Loss: {train_metrics.get('train_loss', 0.0):.5f} | "
            f"Enc: {train_metrics.get('train_loss_encoder', 0.0):.5f} | "
            f"Dec: {train_metrics.get('train_loss_decoder', 0.0):.5f} | "
            f"Pos MSE: {train_metrics.get('train_loss_position', 0.0):.5f} | "
            f"Vel MSE: {train_metrics.get('train_loss_velocity', 0.0):.5f}",
            indent_level=1,
        )
        chronicle.log_detail(
            "📊 Validation Losses",
            f"Val Loss: {val_metrics.get('val_loss', 0.0):.5f} | "
            f"Enc: {val_metrics.get('val_loss_encoder', 0.0):.5f} | "
            f"Dec: {val_metrics.get('val_loss_decoder', 0.0):.5f} | "
            f"Pos MSE: {val_metrics.get('val_loss_position', 0.0):.5f} | "
            f"Vel MSE: {val_metrics.get('val_loss_velocity', 0.0):.5f}",
            indent_level=1,
        )
        if "val_pos_err" in val_metrics or "train_pos_err" in train_metrics:
            chronicle.log_detail(
                "📏 Euclidean L2 Errors",
                f"Train Pos L2: {train_metrics.get('train_pos_err', 0.0):.5f} | "
                f"Train Vel L2: {train_metrics.get('train_vel_err', 0.0):.5f} || "
                f"Val Pos L2: {val_metrics.get('val_pos_err', 0.0):.5f} | "
                f"Val Vel L2: {val_metrics.get('val_vel_err', 0.0):.5f}",
                indent_level=1,
            )
        curr_score = self.get_selection_score(val_metrics)
        chronicle.log_detail(
            "🎯 Model Selection",
            f"Current: {curr_score:.5f} | Best: {best_score:.5f} @ Ep {best_epoch}",
            indent_level=1,
        )
        chronicle.log_newline()

    def get_optimizers_state(self) -> Dict[str, Any]:
        state = {}
        if self.representation_optimizer is not None:
            state["representation"] = self.representation_optimizer.state_dict()
        if self.predictor_optimizer is not None:
            state["predictor"] = self.predictor_optimizer.state_dict()
        return state

    def load_optimizers_state(self, state: Dict[str, Any]) -> None:
        if "representation" in state and self.representation_optimizer is not None:
            self.representation_optimizer.load_state_dict(state["representation"])
        if "predictor" in state and self.predictor_optimizer is not None:
            self.predictor_optimizer.load_state_dict(state["predictor"])
