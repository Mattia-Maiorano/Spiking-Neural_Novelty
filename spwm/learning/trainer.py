"""
Forward-Only Continuous Online Trainer for SPWM (v3 / v4.2).
Maintains O(1) memory footprint scaling across long sequence horizons.
Executes online e-prop plasticity with online mini-batch updates.

v4.2 additions:
  - Auxiliary coordinate loss L_coord (lambda_coord * MSE of Spatial-Softmax keypoints
    vs GT position in normalised image space) is computed inside the encoder_optimizer
    block with torch.enable_grad(), letting the gradient flow through:
      Conv2D -> SpatialSoftmax -> ALIF input_proj
    without touching the predictor or the e-prop plasticity path.
  - New per-epoch scalars logged: l_coord (train) and val_l_coord (val).
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
    Continuous Online Trainer (SPWM-v3).
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
        # Resume support (optional)
        start_epoch: int = 1,
        best_val_loss: float = float("inf"),
        history: Optional[List[Dict[str, float]]] = None,
        optimizer_state: Optional[Dict] = None,
    ) -> None:
        self.best_epoch: Optional[int] = None
        self.best_val_pos_err: float = float("inf")
        self.learning_algorithm = learning_algorithm.lower()
        self.learning_rate = learning_rate

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
        self.loss_fn = loss_fn or SPWMLoss()
        self.grad_clip_norm = grad_clip_norm
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer setup
        probe_params = [
            p for n, p in self.model.named_parameters() if "decoder" in n or "probe" in n
        ]
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

        self.probe_scheduler = (
            torch.optim.lr_scheduler.StepLR(
                self.probe_optimizer, step_size=5, gamma=0.5
            )
            if self.probe_optimizer
            else None
        )

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
        chronicle.log_application_title("SPWM-v4.3 CONTINUOUS ONLINE TRAINER")
        chronicle.log_detail("Device", self.device)
        chronicle.log_detail(
            "Epochs",
            f"{self.start_epoch} -> {self.start_epoch + total_epochs - 1}  (Total: {total_epochs})",
        )
        chronicle.log_detail("Learning Rate", f"{self.learning_rate:.2e}")
        chronicle.log_detail("Algorithm", self.learning_algorithm)
        chronicle.log_detail("Checkpoint Dir", str(self.save_dir))
        chronicle.log_detail("Encoder Optimizer", "Enabled" if self.encoder_optimizer else "Disabled")
        chronicle.log_detail("Probe Optimizer", "Enabled" if self.probe_optimizer else "Disabled")
        chronicle.log_detail("Pred Optimizer", "Enabled" if self.predictor_optimizer else "Disabled")
        chronicle.log_newline()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_losses: Dict[str, float] = {
            "total_loss": 0.0,
            "l_pred": 0.0,
            "l_probe": 0.0,
            "l_coord": 0.0,   # v4.2: auxiliary keypoint coordinate loss
            "spike_rate": 0.0,
            "grad_norm": 0.0,
        }
        num_batches = 0

        for batch in self.train_loader:
            events = batch["events"].to(self.device)
            true_kin = batch.get("flat_kinematics")
            if true_kin is not None:
                true_kin = true_kin.to(self.device)

            with torch.no_grad():
                # v3.4: flat_kinematics is already [B, T, 4*N]; pass directly for per-step kinematic feedback
                kin_for_eprop = true_kin  # [B, T, 4] or None
                out = self.model(
                    events,
                    accumulate_local_updates=True,
                    learning_rate=self.learning_rate,
                    target_kinematics=kin_for_eprop,
                )

            if hasattr(self.model, "apply_accumulated_updates"):
                self.model.apply_accumulated_updates(learning_rate=self.learning_rate)

            # Online encoder update (Sensory prediction + Kinematic / Latent feedback)
            if self.encoder_optimizer is not None:
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

                    # 2. Kinematic / probe alignment
                    if true_kin is not None:
                        decoded_enc = self.model.physical_decoder(enc_seq)
                        loss_kin_enc = nn.functional.mse_loss(decoded_enc, true_kin)
                        loss_enc = loss_sensory + loss_kin_enc
                    else:
                        loss_enc = loss_sensory

                    # 3. v4.2 — Auxiliary Coordinate Loss (L_coord)
                    #    Gradient path: GT_xy -> MSE -> kp_seq (SpatialSoftmax output)
                    #                              -> conv weights (Conv2D)
                    #    This is the "Strada B" geometric rectification step.
                    lambda_coord = getattr(self.loss_fn, "lambda_coord", 0.0)
                    if lambda_coord > 0.0 and true_kin is not None:
                        l_coord = self.loss_fn.coordinate_loss(kp_seq, true_kin)
                        loss_enc = loss_enc + lambda_coord * l_coord
                        epoch_losses["l_coord"] += l_coord.item()
                    else:
                        # still accumulate zero so the key exists in every epoch record
                        epoch_losses["l_coord"] += 0.0

                    loss_enc.backward()
                    if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.encoder.parameters(), self.grad_clip_norm
                        )
                    self.encoder_optimizer.step()

            # Online local predictor update
            if self.predictor_optimizer is not None:
                self.predictor_optimizer.zero_grad()
                with torch.enable_grad():
                    z_in = out.latent_states[:, :-1].detach()
                    z_target = out.latent_states[:, 1:].detach()
                    if z_in.shape[1] > 0:
                        pred_res = self.model.predictor(z_in)
                        pred_loss = nn.functional.mse_loss(
                            pred_res.predicted_latent, z_target
                        )
                        pred_loss.backward()
                        if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                            for group in self.predictor_optimizer.param_groups:
                                torch.nn.utils.clip_grad_norm_(
                                    group["params"], self.grad_clip_norm
                                )
                        self.predictor_optimizer.step()

            # Online local probe update
            if self.probe_optimizer is not None and true_kin is not None:
                self.probe_optimizer.zero_grad()
                with torch.enable_grad():
                    z_detached = out.latent_states.detach()
                    decoded = self.model.physical_decoder(z_detached)
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

            # Compute probe (kinematic) loss if decoder output is available
            if out.decoded_kinematics is not None and true_kin is not None:
                probe_loss = nn.functional.mse_loss(out.decoded_kinematics, true_kin)
                val_losses["val_l_probe"] += probe_loss.item()
                # Apply weighting from loss function if defined
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
        end_epoch = self.start_epoch + epochs - 1
        self._print_training_header(epochs)

        try:
            for epoch in range(self.start_epoch, end_epoch + 1):
                t_epoch_start = time.time()
                train_metrics = self.train_epoch(epoch)
                val_metrics = self.evaluate()
                epoch_duration = time.time() - t_epoch_start
                total_elapsed = time.time() - start_time

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

                # Controllo Best Validation Model vincolato alla minima Pos Err di validazione
                current_metric = val_metrics["val_pos_err"]
                is_best = current_metric < self.best_val_pos_err
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
                    header = (
                        f"Epoch [{epoch:03d}/{end_epoch:03d}]"
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
                            f" Probe: {train_metrics['l_probe']:.5f})"
                        ),
                        indent_level=1,
                    )
                    chronicle.log_detail(
                        "Val Loss",
                        (
                            f"{val_metrics['val_total_loss']:.5f}"
                            f"  (Pred: {val_metrics['val_l_pred']:.5f},"
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
            raise

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
                    f"l_probe={rec.get('l_probe'):.4f}",
                    f"l_coord={rec.get('l_coord', 0.0):.4f}",
                    f"spike_rate={rec.get('spike_rate'):.3f}",
                    f"val_total_loss={rec.get('val_total_loss'):.4f}",
                    f"val_l_pred={rec.get('val_l_pred'):.4f}",
                    f"val_pos_err={rec.get('val_pos_err'):.4f}",
                    f"val_vel_err={rec.get('val_vel_err'):.4f}",
                    f"val_spike_rate={rec.get('val_spike_rate'):.3f}",
                ]
                f.write("  Losses: " + ", ".join(loss_items) + "\n")
                f.write(
                    f"  GradNormOverall={rec.get('grad_norm'):.4f}, "
                    f"GradNormRecent10={rec.get('grad_norm_recent'):.4f}\n\n"
                )
