"""
Forward-Only Continuous Online Trainer for SPWM (v6.6).
Maintains O(1) memory footprint scaling across long sequence horizons.
Executes online e-prop plasticity with online mini-batch updates.

v6.6 features:
  - Encoder Warmup: Stabilizes SpatialSoftmax keypoint coordinates in the initial phase
    (first encoder_warmup_epochs global epochs) by withholding multi-step rollout loss (K=1)
    and focusing on clean sensory and coordinate representations.
  - Direct Kinematic Velocity Supervision: Active gradient propagation into latent momentum p
    via physical decoder vel_head probe.
  - Removal of indirect L_vel_cons shortcut.
  - Homeostatic sparsity lock (lambda_sparse scaled by an order of magnitude).
"""

from __future__ import annotations
import collections
import csv
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

import chronicle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from spwm.learning.diagnostics import collect_spike_statistics, inspect_gradients
from spwm.learning.losses import LossOutput, SPWMLoss
from spwm.learning.metrics import (
    one_step_mse,
    position_error,
    spike_rate,
    velocity_error,
)


class Trainer:
    """
    Continuous Online Trainer (SPWM-v6.6).
    Streams sequence batches frame-by-frame with O(1) memory footprint and
    executes online forward-only e-prop plasticity.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Optional[SPWMLoss] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        grad_clip_norm: float = 1.0,
        learning_algorithm: str = "online_eprop",
        device: Optional[Union[str, torch.device]] = None,
        save_dir: str = "results/default_run",
        probe_lr: float = 5e-4,
        probe_weight_decay: float = 1e-2,
        tensorboard_logging: bool = False,
        start_epoch: int = 1,
        best_val_loss: float = float("inf"),
        best_val_pos_err: float = float("inf"),
        history: Optional[List[Dict[str, float]]] = None,
        optimizer_state: Optional[Dict] = None,
        curriculum_multi_step: bool = False,
        curriculum_thresholds: Optional[Dict[str, Any]] = None,
        encoder_warmup_epochs: int = 70,
    ) -> None:
        self.best_epoch: Optional[int] = None
        self.best_val_pos_err: float = best_val_pos_err
        self.learning_algorithm = learning_algorithm.lower()
        self.learning_rate = learning_rate
        self.curriculum_multi_step = curriculum_multi_step
        self.encoder_warmup_epochs = encoder_warmup_epochs
        self.curriculum_thresholds = curriculum_thresholds or {
            "phase_1_horizon": 3,
            "phase_2_horizon": 6,
            "phase_2_threshold": 0.20,
            "phase_3_horizon": 10,
            "phase_3_threshold": 0.12,
        }

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.recent_grad_norms = collections.deque(maxlen=10)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = (loss_fn or SPWMLoss()).to(self.device)
        self.grad_clip_norm = grad_clip_norm
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer setup
        probe_params = [
            p for n, p in self.model.named_parameters() if "decoder" in n or "probe" in n
        ]
        # Include latent dynamics p projection/fusion weights if direct kinematics gradient is active
        if hasattr(self.model, "dynamics"):
            dynamics_p_params = [
                p for n, p in self.model.dynamics.named_parameters()
                if "fuse_" in n or "recurrent_proj" in n or "norm" in n
            ]
            probe_params.extend(dynamics_p_params)

        predictor_params = [
            p
            for n, p in self.model.named_parameters()
            if "predictor" in n and "sensory_predictor" not in n
        ]

        encoder_params = list(self.model.encoder.parameters()) if hasattr(self.model, "encoder") else []

        self.encoder_optimizer = (
            torch.optim.AdamW(
                encoder_params,
                lr=2e-4,
                weight_decay=1e-4,
            )
            if encoder_params
            else None
        )

        self.probe_optimizer = (
            torch.optim.AdamW(
                probe_params,
                lr=probe_lr,
                weight_decay=probe_weight_decay,
            )
            if probe_params
            else None
        )

        self.probe_scheduler = None

        self.predictor_optimizer = (
            torch.optim.AdamW(
                predictor_params,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
            if predictor_params
            else None
        )

        self.writer = (
            SummaryWriter(log_dir=str(self.save_dir / "tb"))
            if tensorboard_logging
            else None
        )

        # Resume state
        self.start_epoch = start_epoch
        self.best_val_loss = best_val_loss
        self.history = history if history is not None else []
        if optimizer_state:
            if self.encoder_optimizer and "encoder_optimizer" in optimizer_state:
                self.encoder_optimizer.load_state_dict(
                    optimizer_state["encoder_optimizer"]
                )
            if self.probe_optimizer and "probe_optimizer" in optimizer_state:
                self.probe_optimizer.load_state_dict(optimizer_state["probe_optimizer"])
            if self.predictor_optimizer and "predictor_optimizer" in optimizer_state:
                self.predictor_optimizer.load_state_dict(
                    optimizer_state["predictor_optimizer"]
                )

    def _print_training_header(self, total_epochs: int) -> None:
        """Visualizza i parametri principali prima dell'avvio."""
        chronicle.log_application_title("SPWM-v6.6 CONTINUOUS ONLINE TRAINER")
        chronicle.log_detail("Device", self.device)
        chronicle.log_detail(
            "Epochs",
            f"{self.start_epoch} -> {self.start_epoch + total_epochs - 1}  (Total: {total_epochs})",
        )
        chronicle.log_detail("Learning Rate", f"{self.learning_rate:.2e}")
        chronicle.log_detail("Algorithm", self.learning_algorithm)
        chronicle.log_detail("Checkpoint Dir", str(self.save_dir))
        chronicle.log_detail("Encoder Warmup Epochs", f"{self.encoder_warmup_epochs}")
        chronicle.log_detail("Encoder Optimizer", "Enabled" if self.encoder_optimizer else "Disabled")
        chronicle.log_detail("Probe Optimizer", "Enabled" if self.probe_optimizer else "Disabled")
        chronicle.log_detail("Pred Optimizer", "Enabled" if self.predictor_optimizer else "Disabled")
        chronicle.log_newline()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_losses: Dict[str, float] = {
            "total_loss": 0.0,
            "l_pred": 0.0,
            "l_multi": 0.0,
            "l_probe": 0.0,
            "l_coord": 0.0,   # v4.2: auxiliary keypoint coordinate loss
            "spike_rate": 0.0,
            "grad_norm": 0.0,
        }
        num_batches = 0

        # Visual Encoder Warmup Phase:
        # During the first encoder_warmup_epochs (e.g. 70 epochs), explicitly freeze
        # the convolutional weights and SpatialSoftmax (param.requires_grad = False),
        # unlocking them only after coordinate convergence (epoch > encoder_warmup_epochs).
        # Multi-step rollout loss is also held to 1-step during warmup.
        is_warmup = epoch <= self.encoder_warmup_epochs

        # Enforce explicit freeze/unfreeze on encoder conv and spatial_softmax
        if hasattr(self.model, "encoder"):
            if hasattr(self.model.encoder, "conv"):
                for p in self.model.encoder.conv.parameters():
                    p.requires_grad = not is_warmup
            if hasattr(self.model.encoder, "spatial_softmax"):
                for p in self.model.encoder.spatial_softmax.parameters():
                    p.requires_grad = not is_warmup

        for batch in self.train_loader:
            events = batch["events"].to(self.device)
            true_kin = batch.get("flat_kinematics")
            if true_kin is not None:
                true_kin = true_kin.to(self.device)

            with torch.no_grad():
                kin_for_eprop = true_kin
                out = self.model(
                    events,
                    accumulate_local_updates=True,
                    learning_rate=self.learning_rate,
                    target_kinematics=kin_for_eprop,
                )

            if hasattr(self.model, "apply_accumulated_updates"):
                self.model.apply_accumulated_updates(learning_rate=self.learning_rate)

            # Online encoder update (Sensory prediction + Coordinate Supervision)
            # When frozen (epoch <= encoder_warmup_epochs), only unfrozen parameters (like proj) receive gradients
            # or update step is skipped if all encoder parameters are frozen.
            if self.encoder_optimizer is not None:
                active_params = [p for p in self.model.encoder.parameters() if p.requires_grad]
                if active_params:
                    self.encoder_optimizer.zero_grad()
                    with torch.enable_grad():
                        enc_seq, kp_seq = self.model.encoder(events, return_keypoints=True)  # [B,T,D], [B,T,K*2]
                        B_sz, T_sz, _ = enc_seq.shape

                        # 1. Top-down sensory prediction error
                        z_lat = out.latent_states.detach()
                        z_prev = torch.cat(
                            [torch.zeros(B_sz, 1, z_lat.shape[-1], device=self.device), z_lat[:, :-1]],
                            dim=1,
                        )
                        pred_x = self.model.sensory_predictor(z_prev)
                        loss_sensory = nn.functional.mse_loss(enc_seq, pred_x.detach())

                        loss_enc = loss_sensory

                        # 2. v4.2 — Auxiliary Coordinate Loss (L_coord)
                        lambda_coord = getattr(self.loss_fn, "lambda_coord", 0.0)
                        if lambda_coord > 0.0 and true_kin is not None:
                            l_coord = self.loss_fn.coordinate_loss(kp_seq, true_kin)
                            loss_enc = loss_enc + lambda_coord * l_coord
                            epoch_losses["l_coord"] += l_coord.item()
                        else:
                            epoch_losses["l_coord"] += 0.0

                        loss_enc.backward()
                        if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                active_params, self.grad_clip_norm
                            )
                        self.encoder_optimizer.step()

            # Online local predictor update (1-step Teacher Forcing + K-step Autoregressive Rollout Loss)
            if self.predictor_optimizer is not None:
                self.predictor_optimizer.zero_grad()
                with torch.enable_grad():
                    latents = out.latent_states.detach()  # [B, T, D]
                    z_in = latents[:, :-1]
                    z_target = latents[:, 1:]
                    l_1step = torch.tensor(0.0, device=self.device)
                    pred_res = None
                    if z_in.shape[1] > 0:
                        pred_res = self.model.predictor(z_in)
                        l_1step = nn.functional.mse_loss(
                            pred_res.predicted_latent, z_target
                        )

                    # Multi-step autoregressive rollout loss over horizon K (active only after warmup)
                    K = getattr(self.loss_fn, "multi_step_horizon", 3)
                    lambda_multi = getattr(self.loss_fn, "lambda_multi", 0.5) if not is_warmup else 0.0
                    lambda_pred = getattr(self.loss_fn, "lambda_pred", 1.0)

                    l_multi = torch.tensor(0.0, device=self.device)
                    B_sz, T_sz, _ = latents.shape
                    if lambda_multi > 0.0 and T_sz > K + 1:
                        stride = max(1, K)
                        t_starts = list(range(0, T_sz - K, stride))
                        if t_starts:
                            curr_z = latents[:, t_starts]  # [B, S, D]
                            step_losses = []
                            for step in range(1, K + 1):
                                target_step_z = torch.stack(
                                    [latents[:, t + step] for t in t_starts], dim=1
                                )
                                pred_out = self.model.predictor(curr_z)
                                curr_z = pred_out.predicted_latent
                                step_losses.append(
                                    nn.functional.mse_loss(curr_z, target_step_z)
                                )
                            l_multi = torch.stack(step_losses).mean()

                    pred_loss = (
                        lambda_pred * l_1step
                        + lambda_multi * l_multi
                    )
                    pred_loss.backward()
                    if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                        for group in self.predictor_optimizer.param_groups:
                            torch.nn.utils.clip_grad_norm_(
                                group["params"], self.grad_clip_norm
                            )
                    self.predictor_optimizer.step()

                epoch_losses["l_multi"] += l_multi.item()
                epoch_losses["total_loss"] += lambda_multi * l_multi.item()

            # Online local probe update with direct kinematic supervision on momentum p
            if self.probe_optimizer is not None and true_kin is not None:
                self.probe_optimizer.zero_grad()
                with torch.enable_grad():
                    # Direct kinematic supervision:
                    # q is detached (or supervised by L_coord), while momentum p receives active gradients
                    # forcing p to directly encode real kinematic velocity.
                    z_states = out.latent_states
                    q_dim = getattr(self.loss_fn, "q_dim", 32)
                    q_part = z_states[..., :q_dim].detach()
                    p_part = z_states[..., q_dim:]
                    z_input = torch.cat([q_part, p_part], dim=-1)

                    decoded = self.model.physical_decoder(z_input)
                    probe_loss = nn.functional.mse_loss(decoded, true_kin)
                    probe_loss.backward()
                    if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                        for group in self.probe_optimizer.param_groups:
                            torch.nn.utils.clip_grad_norm_(
                                group["params"], self.grad_clip_norm
                            )
                    self.probe_optimizer.step()
                epoch_losses["l_probe"] += probe_loss.item()
                epoch_losses["total_loss"] += self.loss_fn.lambda_probe * probe_loss.item()

            grad_norm = 0.0
            if self.encoder_optimizer is not None:
                for p in self.encoder_optimizer.param_groups[0]["params"]:
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item()
            if self.predictor_optimizer is not None:
                for p in self.predictor_optimizer.param_groups[0]["params"]:
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item()
            if self.probe_optimizer is not None:
                for p in self.probe_optimizer.param_groups[0]["params"]:
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item()
            epoch_losses["grad_norm"] += grad_norm
            self.recent_grad_norms.append(grad_norm)

            pred_err = (
                out.prediction_errors.mean().item()
                if out.prediction_errors.numel() > 0
                else 0.0
            )
            epoch_losses["l_pred"] += pred_err
            epoch_losses["total_loss"] += pred_err
            if hasattr(out, "mean_spike_rate"):
                epoch_losses["spike_rate"] += out.mean_spike_rate.item()

            num_batches += 1

        for k in epoch_losses:
            epoch_losses[k] /= max(1, num_batches)

        return epoch_losses

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        val_losses: Dict[str, float] = {
            "val_total_loss": 0.0,
            "val_l_pred": 0.0,
            "val_l_multi": 0.0,
            "val_l_probe": 0.0,
            "val_l_coord": 0.0,   # v4.2: coordinate loss on val set
            "val_pos_err": 0.0,
            "val_vel_err": 0.0,
            "val_spike_rate": 0.0,
        }
        num_batches = 0

        for batch in self.val_loader:
            events = batch["events"].to(self.device)
            true_kin = batch.get("flat_kinematics")
            if true_kin is not None:
                true_kin = true_kin.to(self.device)

            out = self.model(events, accumulate_local_updates=False)

            pred_err = (
                out.prediction_errors.mean().item()
                if out.prediction_errors.numel() > 0
                else 0.0
            )
            val_losses["val_total_loss"] += pred_err
            val_losses["val_l_pred"] += pred_err

            # Multi-step autoregressive rollout loss on val set
            K = getattr(self.loss_fn, "multi_step_horizon", 3)
            lambda_multi = getattr(self.loss_fn, "lambda_multi", 0.5)
            B_sz, T_sz, _ = out.latent_states.shape
            if lambda_multi > 0.0 and T_sz > K + 1:
                stride = max(1, K)
                t_starts = list(range(0, T_sz - K, stride))
                if t_starts:
                    curr_z = out.latent_states[:, t_starts]
                    val_step_losses = []
                    for step in range(1, K + 1):
                        target_step_z = torch.stack(
                            [out.latent_states[:, t + step] for t in t_starts], dim=1
                        )
                        pred_out = self.model.predictor(curr_z)
                        curr_z = pred_out.predicted_latent
                        val_step_losses.append(
                            nn.functional.mse_loss(curr_z, target_step_z)
                        )
                    val_l_multi = torch.stack(val_step_losses).mean().item()
                    val_losses["val_l_multi"] += val_l_multi
                    val_losses["val_total_loss"] += lambda_multi * val_l_multi

            # Compute probe (kinematic) loss if decoder output is available
            if out.decoded_kinematics is not None and true_kin is not None:
                probe_loss = nn.functional.mse_loss(out.decoded_kinematics, true_kin)
                val_losses["val_l_probe"] += probe_loss.item()
                if hasattr(self.loss_fn, "lambda_probe"):
                    val_losses["val_total_loss"] += self.loss_fn.lambda_probe * probe_loss.item()
                else:
                    val_losses["val_total_loss"] += probe_loss.item()

            if out.decoded_kinematics is not None and true_kin is not None:
                val_losses["val_pos_err"] += position_error(
                    out.decoded_kinematics, true_kin
                )
                val_losses["val_vel_err"] += velocity_error(
                    out.decoded_kinematics, true_kin
                )

            # v4.2: evaluate coordinate loss on val set (no_grad — diagnostic only)
            lambda_coord = getattr(self.loss_fn, "lambda_coord", 0.0)
            if lambda_coord > 0.0 and true_kin is not None:
                enc_out, kp_val = self.model.encoder(events, return_keypoints=True)
                l_coord_val = self.loss_fn.coordinate_loss(kp_val, true_kin)
                val_losses["val_l_coord"] += l_coord_val.item()

            if hasattr(out, "mean_spike_rate"):
                val_losses["val_spike_rate"] += out.mean_spike_rate.item()

            num_batches += 1

        for k in val_losses:
            val_losses[k] /= max(1, num_batches)

        return val_losses

    def fit(
        self,
        epochs: int = 50,
        print_every: int = 1,
    ) -> List[Dict[str, float]]:
        start_time = time.time()
        base_elapsed = self.history[-1].get("elapsed_time", 0.0) if self.history else 0.0
        end_epoch = self.start_epoch + epochs - 1
        self._print_training_header(epochs)

        try:
            for epoch in range(self.start_epoch, end_epoch + 1):
                t_epoch_start = time.time()
                train_metrics = self.train_epoch(epoch)
                val_metrics = self.evaluate()
                epoch_duration = time.time() - t_epoch_start
                total_elapsed = base_elapsed + (time.time() - start_time)

                recent_norm = (
                    sum(self.recent_grad_norms) / max(1, len(self.recent_grad_norms))
                )

                record = {
                    "epoch": epoch,
                    "elapsed_time": total_elapsed,
                    **train_metrics,
                    **val_metrics,
                    "grad_norm_recent": recent_norm,
                }
                self.history.append(record)

                if self.writer is not None:
                    for k, v in record.items():
                        if k != "epoch":
                            self.writer.add_scalar(k, v, epoch)

                # Step probe learning rate scheduler
                if self.probe_scheduler is not None:
                    self.probe_scheduler.step()

                # Dynamic Adaptive Curriculum for Multi-Step Horizon (K)
                # (Active on global epoch count > encoder_warmup_epochs)
                if self.curriculum_multi_step and hasattr(self.loss_fn, "multi_step_horizon"):
                    if epoch > self.encoder_warmup_epochs:
                        p1_k = self.curriculum_thresholds.get("phase_1_horizon", 3)
                        p2_k = self.curriculum_thresholds.get("phase_2_horizon", 6)
                        p2_th = self.curriculum_thresholds.get("phase_2_threshold", 0.20)
                        p3_k = self.curriculum_thresholds.get("phase_3_horizon", 10)
                        p3_th = self.curriculum_thresholds.get("phase_3_threshold", 0.12)

                        curr_val_pos = val_metrics["val_pos_err"]
                        if curr_val_pos < p3_th:
                            target_k = p3_k
                        elif curr_val_pos < p2_th:
                            target_k = p2_k
                        else:
                            target_k = p1_k

                        if target_k != self.loss_fn.multi_step_horizon:
                            old_k = self.loss_fn.multi_step_horizon
                            self.loss_fn.multi_step_horizon = target_k
                            chronicle.log_info(
                                f"[Curriculum Horizon Shift] val_pos_err={curr_val_pos:.4f}: "
                                f"K transitioned from {old_k} to {target_k}"
                            )
                    else:
                        # Warmup phase: hold K=1 for predictor stability
                        if self.loss_fn.multi_step_horizon != 1:
                            self.loss_fn.multi_step_horizon = 1

                # Controllo Best Validation Model vincolato alla minima Pos Err di validazione
                current_metric = val_metrics["val_pos_err"]
                is_best = current_metric < self.best_val_pos_err
                self.save_dir.mkdir(parents=True, exist_ok=True)
                if is_best:
                    self.best_val_pos_err = current_metric
                    self.best_epoch = epoch
                    torch.save(self.model.state_dict(), self.save_dir / "model.pt")

                # Salva Checkpoint periodico
                checkpoint_path = self.save_dir / "checkpoint.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "best_val_loss": self.best_val_loss,
                        "best_val_pos_err": self.best_val_pos_err,
                        "history": self.history,
                        "optimizer_state": {
                            "encoder_optimizer": self.encoder_optimizer.state_dict()
                            if self.encoder_optimizer
                            else None,
                            "probe_optimizer": self.probe_optimizer.state_dict()
                            if self.probe_optimizer
                            else None,
                            "predictor_optimizer": self.predictor_optimizer.state_dict()
                            if self.predictor_optimizer
                            else None,
                        },
                    },
                    checkpoint_path,
                )

                # Stampa formattata a schermo
                if epoch % print_every == 0 or epoch == end_epoch:
                    best_tag = " ★ [BEST]" if is_best else ""
                    current_k = getattr(self.loss_fn, "multi_step_horizon", 1)
                    warmup_tag = " [WARMUP]" if epoch <= self.encoder_warmup_epochs else ""
                    header = (
                        f"Epoch [{epoch:03d}/{end_epoch:03d}]"
                        f" (K={current_k}){warmup_tag}"
                        f"  Time: {epoch_duration:5.1f}s"
                        f"  Total: {total_elapsed / 60:4.1f}m"
                        f"{best_tag}"
                    )
                    chronicle.log_section_header(header)
                    chronicle.log_detail(
                        "Train Loss",
                        (
                            f"{train_metrics['total_loss']:.5f}"
                            f"  (Pred: {train_metrics['l_pred']:.5f},"
                            f" Multi: {train_metrics.get('l_multi', 0.0):.5f},"
                            f" Probe: {train_metrics['l_probe']:.5f})"
                        ),
                        indent_level=1,
                    )
                    chronicle.log_detail(
                        "Val Loss",
                        (
                            f"{val_metrics['val_total_loss']:.5f}"
                            f"  (Pred: {val_metrics['val_l_pred']:.5f},"
                            f" Multi: {val_metrics.get('val_l_multi', 0.0):.5f},"
                            f" Probe: {val_metrics.get('val_l_probe', 0.0):.5f})"
                        ),
                        indent_level=1,
                    )
                    chronicle.log_detail(
                        "Pos Err",
                        f"{val_metrics['val_pos_err']:.5f}",
                        indent_level=1,
                    )
                    chronicle.log_detail(
                        "Vel Err",
                        f"{val_metrics['val_vel_err']:.5f}",
                        indent_level=1,
                    )
                    chronicle.log_detail(
                        "L_coord",
                        (
                            f"Train {train_metrics['l_coord']:.5f}"
                            f" / Val {val_metrics.get('val_l_coord', 0.0):.5f}"
                        ),
                        indent_level=1,
                    )
                    chronicle.log_detail(
                        "Spike Rate",
                        (
                            f"Train {train_metrics['spike_rate']:.3f}"
                            f" / Val {val_metrics['val_spike_rate']:.3f}"
                        ),
                        indent_level=1,
                    )
                    chronicle.log_detail(
                        "Grad Norm",
                        f"{recent_norm:.4f}",
                        indent_level=1,
                    )
                    chronicle.log_newline()

        except KeyboardInterrupt:
            chronicle.log_warning(
                "[INTERRUPT] Training interrotto manualmente dall'utente."
            )
            try:
                chronicle.log_info(
                    "Valutazione stato attuale del modello in corso..."
                )
                val_metrics = self.evaluate()
                current_pos_err = val_metrics.get("val_pos_err", float("inf"))
                if current_pos_err < self.best_val_pos_err:
                    self.best_val_pos_err = current_pos_err
                    model_path = self.save_dir / "model.pt"
                    torch.save(self.model.state_dict(), model_path)
                    chronicle.log_success(
                        f"Nuovo record ottenuto (Pos Err: {current_pos_err:.5f})."
                        f" Modello salvato in: {model_path}"
                    )
                else:
                    chronicle.log_info(
                        f"Nessun miglioramento"
                        f" (Attuale: {current_pos_err:.5f}"
                        f" vs Best: {self.best_val_pos_err:.5f})."
                    )
            except Exception as e:
                chronicle.log_error(
                    f"Errore durante la validazione post-interrupt: {e}"
                )
                model_path = self.save_dir / "model.pt"
                torch.save(self.model.state_dict(), model_path)
                chronicle.log_success(
                    f"Modello salvato (fallback) in: {model_path}"
                )

        self.save_training_log()
        chronicle.log_success(
            f"Sessione completata. Log salvati in '{self.save_dir}'."
            f" Best Val Pos Err: {self.best_val_pos_err:.5f}"
        )
        return self.history

    def save_training_log(self) -> None:
        if not self.history:
            return

        json_path = self.save_dir / "training_log.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        csv_path = self.save_dir / "training_log.csv"
        keys = list(self.history[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)

        txt_path = self.save_dir / "training_log.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for rec in self.history:
                f.write(
                    f"Epoch {rec.get('epoch'):03d} | TrainLoss {rec.get('total_loss'):.4f} | "
                    f"ValLoss {rec.get('val_total_loss'):.4f} | SpikeRate {rec.get('val_spike_rate'):.3f} | "
                    f"Time {rec.get('elapsed_time'):.1f}s\n"
                )
                loss_items = [
                    f"total_loss={rec.get('total_loss'):.4f}",
                    f"l_pred={rec.get('l_pred'):.4f}",
                    f"l_multi={rec.get('l_multi', 0.0):.4f}",
                    f"l_probe={rec.get('l_probe'):.4f}",
                    f"l_coord={rec.get('l_coord', 0.0):.4f}",
                    f"spike_rate={rec.get('spike_rate'):.3f}",
                    f"val_total_loss={rec.get('val_total_loss'):.4f}",
                    f"val_l_pred={rec.get('val_l_pred'):.4f}",
                    f"val_l_multi={rec.get('val_l_multi', 0.0):.4f}",
                    f"val_pos_err={rec.get('val_pos_err'):.4f}",
                    f"val_vel_err={rec.get('val_vel_err'):.4f}",
                    f"val_spike_rate={rec.get('val_spike_rate'):.3f}",
                ]
                f.write("  Losses: " + ", ".join(loss_items) + "\n")
                f.write(
                    f"  GradNormOverall={rec.get('grad_norm'):.4f}, "
                    f"GradNormRecent10={rec.get('grad_norm_recent'):.4f}\n\n"
                )
