"""
Modular Trainer for SPWM and Baseline World Models.
Implements surrogate-gradient BPTT, gradient diagnostics, metric tracking,
and artifact checkpointing. Pluggable for future local learning rules.
"""

from __future__ import annotations
import os
import time
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from spwm.learning.losses import SPWMLoss, LossOutput
from spwm.learning.diagnostics import inspect_gradients, collect_spike_statistics
from spwm.learning.metrics import one_step_mse, position_error, velocity_error, spike_rate


class Trainer:
    """
    Research Trainer for Predictive World Models.
    Decoupled learning algorithm design (currently BPTT, ready for V3 local learning).
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
        learning_algorithm: str = "bptt",
        device: Optional[Union[str, torch.device]] = None,
        save_dir: str = "results/default_run",
        tensorboard_logging: bool = True,
    ) -> None:
        self.learning_algorithm = learning_algorithm.lower()
        if self.learning_algorithm != "bptt":
            raise NotImplementedError(
                f"Learning algorithm '{learning_algorithm}' is reserved for future versions. V1 uses 'bptt'."
            )

        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn or SPWMLoss()
        self.grad_clip_norm = grad_clip_norm
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.writer = SummaryWriter(log_dir=str(self.save_dir / "tb")) if tensorboard_logging else None
        self.history: List[Dict[str, float]] = []

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Runs one full training epoch over train_loader."""
        self.model.train()
        epoch_losses: Dict[str, float] = {
            "total_loss": 0.0,
            "l_pred": 0.0,
            "l_multi": 0.0,
            "l_var": 0.0,
            "l_sparse": 0.0,
            "l_probe": 0.0,
            "grad_norm": 0.0,
        }
        num_batches = 0

        for batch in self.train_loader:
            events = batch["events"].to(self.device)  # [B, T, 2, H, W]
            true_kin = batch["flat_kinematics"].to(self.device)  # [B, T, 4 * N]

            self.optimizer.zero_grad()

            # Forward pass through model
            out = self.model(events)

            # Compute composite loss
            predictor_head = getattr(self.model, "predictor", None)
            loss_out = self.loss_fn(
                latent_states=out.latent_states,
                predicted_latents=out.predicted_latents,
                mean_spike_rate=getattr(out, "mean_spike_rate", torch.tensor(0.0, device=self.device)),
                model_predictor=predictor_head,
                decoded_kinematics=getattr(out, "decoded_kinematics", None),
                true_kinematics=true_kin,
            )

            # Backpropagation
            loss_out.total_loss.backward()

            # Gradient diagnostics & clipping
            grad_diag = inspect_gradients(self.model)
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.optimizer.step()

            # Accumulate
            for k in ["total_loss", "l_pred", "l_multi", "l_var", "l_sparse", "l_probe"]:
                epoch_losses[k] += getattr(loss_out, k).item()
            epoch_losses["grad_norm"] += grad_diag["total_norm"]
            num_batches += 1

        for k in epoch_losses:
            epoch_losses[k] /= max(1, num_batches)

        return epoch_losses

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluates model on validation dataset."""
        self.model.eval()
        val_losses: Dict[str, float] = {
            "val_total_loss": 0.0,
            "val_l_pred": 0.0,
            "val_pos_err": 0.0,
            "val_vel_err": 0.0,
            "val_spike_rate": 0.0,
        }
        num_batches = 0

        for batch in self.val_loader:
            events = batch["events"].to(self.device)
            true_kin = batch["flat_kinematics"].to(self.device)

            out = self.model(events)

            loss_out = self.loss_fn(
                latent_states=out.latent_states,
                predicted_latents=out.predicted_latents,
                mean_spike_rate=getattr(out, "mean_spike_rate", torch.tensor(0.0, device=self.device)),
                model_predictor=getattr(self.model, "predictor", None),
                decoded_kinematics=getattr(out, "decoded_kinematics", None),
                true_kinematics=true_kin,
            )

            val_losses["val_total_loss"] += loss_out.total_loss.item()
            val_losses["val_l_pred"] += loss_out.l_pred.item()

            if getattr(out, "decoded_kinematics", None) is not None:
                val_losses["val_pos_err"] += position_error(out.decoded_kinematics, true_kin)
                val_losses["val_vel_err"] += velocity_error(out.decoded_kinematics, true_kin)

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
        """Executes full training loop, logs metrics, and saves best model."""
        best_val_loss = float("inf")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate()

            record = {
                "epoch": epoch,
                "elapsed_time": time.time() - start_time,
                **train_metrics,
                **val_metrics,
            }
            self.history.append(record)

            # TensorBoard logging
            if self.writer is not None:
                for k, v in record.items():
                    if k != "epoch":
                        self.writer.add_scalar(k, v, epoch)

            # Checkpoint best model
            if val_metrics["val_total_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_total_loss"]
                torch.save(self.model.state_dict(), self.save_dir / "model.pt")

            if epoch % print_every == 0 or epoch == epochs:
                print(
                    f"Epoch {epoch:03d}/{epochs:03d} | "
                    f"Train Loss: {train_metrics['total_loss']:.4f} | "
                    f"Pred Loss: {train_metrics['l_pred']:.4f} | "
                    f"Val Loss: {val_metrics['val_total_loss']:.4f} | "
                    f"Val PosErr: {val_metrics['val_pos_err']:.4f} | "
                    f"SpikeRate: {val_metrics['val_spike_rate']:.3f} | "
                    f"Time: {record['elapsed_time']:.1f}s"
                )

        # Save training log CSV
        self.save_training_log()

        if self.writer is not None:
            self.writer.close()

        return self.history

    def save_training_log(self) -> None:
        """Saves CSV log of all recorded metrics."""
        if not self.history:
            return
        csv_path = self.save_dir / "training_log.csv"
        fieldnames = list(self.history[0].keys())
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)
