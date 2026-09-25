"""
Forward-Only Continuous Online Trainer for SPWM-v3.
Maintains O(1) memory footprint scaling across long sequence horizons.
Executes online e-prop plasticity with online mini-batch updates.
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
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from spwm.learning.losses import SPWMLoss, LossOutput
from spwm.learning.diagnostics import inspect_gradients, collect_spike_statistics
from spwm.learning.metrics import one_step_mse, position_error, velocity_error, spike_rate


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
        tensorboard_logging: bool = False,
        # Resume support (optional)
        start_epoch: int = 1,
        best_val_loss: float = float("inf"),
        history: Optional[List[Dict[str, float]]] = None,
        optimizer_state: Optional[Dict] = None,
    ) -> None:
        # Track which epoch produced the best validation loss
        self.best_epoch: int | None = None
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

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn or SPWMLoss()
        self.grad_clip_norm = grad_clip_norm
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer for linear probes / decoders and predictor head
        probe_params = [p for n, p in self.model.named_parameters() if "decoder" in n or "probe" in n]
        predictor_params = [p for n, p in self.model.named_parameters() if "predictor" in n and "sensory_predictor" not in n]

        self.probe_optimizer = torch.optim.AdamW(
            probe_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        ) if probe_params else None

        self.predictor_optimizer = torch.optim.AdamW(
            predictor_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        ) if predictor_params else None

        self.writer = SummaryWriter(log_dir=str(self.save_dir / "tb")) if tensorboard_logging else None
        # Resume state
        self.start_epoch = start_epoch
        self.best_val_loss = best_val_loss
        self.history = history if history is not None else []
        if optimizer_state:
            if self.probe_optimizer and "probe_optimizer" in optimizer_state:
                self.probe_optimizer.load_state_dict(optimizer_state["probe_optimizer"])
            if self.predictor_optimizer and "predictor_optimizer" in optimizer_state:
                self.predictor_optimizer.load_state_dict(optimizer_state["predictor_optimizer"])


    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Runs one forward-only training epoch with O(1) graph memory.
        No loss.backward() over time dimension is performed.
        """
        self.model.train()
        epoch_losses: Dict[str, float] = {
            "total_loss": 0.0,
            "l_pred": 0.0,
            "l_probe": 0.0,
            "spike_rate": 0.0,
            "grad_norm": 0.0,
        }
        num_batches = 0
        batch_idx = 0

        for batch in self.train_loader:
            batch_idx += 1
            events = batch["events"].to(self.device)  # [B, T, 2, H, W]
            true_kin = batch.get("flat_kinematics")
            if true_kin is not None:
                true_kin = true_kin.to(self.device)

            # Streaming forward pass with forward-only e-prop update accumulation
            # Entire sequence unrolls forward without autograd computation graph over time
            with torch.no_grad():
                out = self.model(
                    events,
                    accumulate_local_updates=True,
                    learning_rate=self.learning_rate,
                )

            # Apply accumulated online plasticity updates at end of mini-batch
            if hasattr(self.model, "apply_accumulated_updates"):
                self.model.apply_accumulated_updates(learning_rate=self.learning_rate)

            # Online local predictor update (predicts next latent state z_(t+1) from z_t)
            if self.predictor_optimizer is not None:
                self.predictor_optimizer.zero_grad()
                with torch.enable_grad():
                    z_in = out.latent_states[:, :-1].detach()
                    z_target = out.latent_states[:, 1:].detach()
                    if z_in.shape[1] > 0:
                        pred_res = self.model.predictor(z_in)
                        pred_loss = nn.functional.mse_loss(pred_res.predicted_latent, z_target)
                        pred_loss.backward()
                        if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                            for group in self.predictor_optimizer.param_groups:
                                torch.nn.utils.clip_grad_norm_(group["params"], self.grad_clip_norm)
                        self.predictor_optimizer.step()

            # Online local probe update (instantaneous frame-level MSE for decoder)
            if self.probe_optimizer is not None and true_kin is not None:
                self.probe_optimizer.zero_grad()
                with torch.enable_grad():
                    z_detached = out.latent_states.detach()
                    decoded = self.model.physical_decoder(z_detached)
                    probe_loss = nn.functional.mse_loss(decoded, true_kin)
                    probe_loss.backward()
                    if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                        for group in self.probe_optimizer.param_groups:
                            torch.nn.utils.clip_grad_norm_(group["params"], self.grad_clip_norm)
                    self.probe_optimizer.step()
                epoch_losses["l_probe"] += probe_loss.item()
                # Compute overall gradient norm for predictor and probe optimizers
                grad_norm = 0.0
                if self.predictor_optimizer is not None:
                    for p in self.predictor_optimizer.param_groups[0]["params"]:
                        if p.grad is not None:
                            grad_norm += p.grad.norm().item()
                if self.probe_optimizer is not None:
                    for p in self.probe_optimizer.param_groups[0]["params"]:
                        if p.grad is not None:
                            grad_norm += p.grad.norm().item()
                epoch_losses["grad_norm"] += grad_norm

            # Record metrics
            pred_err = out.prediction_errors.mean().item() if out.prediction_errors.numel() > 0 else 0.0
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
        """Evaluates model on validation dataset without plasticity accumulation."""
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
            true_kin = batch.get("flat_kinematics")
            if true_kin is not None:
                true_kin = true_kin.to(self.device)

            out = self.model(events, accumulate_local_updates=False)

            pred_err = out.prediction_errors.mean().item() if out.prediction_errors.numel() > 0 else 0.0
            val_losses["val_total_loss"] += pred_err
            val_losses["val_l_pred"] += pred_err

            if out.decoded_kinematics is not None and true_kin is not None:
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
        """Executes online training stream, logs metrics, and saves model checkpoints."""
        start_time = time.time()

        try:
            for epoch in range(self.start_epoch, self.start_epoch + epochs):
                train_metrics = self.train_epoch(epoch)
                val_metrics = self.evaluate()

                record = {
                    "epoch": epoch,
                    "elapsed_time": time.time() - start_time,
                    **train_metrics,
                    **val_metrics,
                }
                self.history.append(record)

                if self.writer is not None:
                    for k, v in record.items():
                        if k != "epoch":
                            self.writer.add_scalar(k, v, epoch)

                # Save best model based on validation loss
                if val_metrics["val_total_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_total_loss"]
                    torch.save(self.model.state_dict(), self.save_dir / "model.pt")

                # Periodic checkpoint (each epoch) – includes optimizer state, epoch, best loss, and history
                checkpoint_path = self.save_dir / "checkpoint.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "best_val_loss": self.best_val_loss,
                        "history": self.history,
                        "optimizer_state": {
                            "probe_optimizer": self.probe_optimizer.state_dict() if self.probe_optimizer else None,
                            "predictor_optimizer": self.predictor_optimizer.state_dict() if self.predictor_optimizer else None,
                        },
                    },
                    checkpoint_path,
                )

                if epoch % print_every == 0 or epoch == self.start_epoch + epochs - 1:
                    print(
                        f"Epoch {epoch:03d}/{self.start_epoch + epochs - 1:03d} | "
                        f"Train Loss: {train_metrics['total_loss']:.4f} | "
                        f"Pred Loss: {train_metrics['l_pred']:.4f} | "
                        f"Val Loss: {val_metrics['val_total_loss']:.4f} | "
                        f"SpikeRate: {val_metrics['val_spike_rate']:.3f} | "
                        f"Time: {record['elapsed_time']:.1f}s"
                    )
        except KeyboardInterrupt:
            # Evaluate validation loss and save model only if it improves over the best checkpoint
            try:
                val_metrics = self.evaluate()
                current_val_loss = val_metrics.get("val_total_loss", float('inf'))
                if current_val_loss < self.best_val_loss:
                    self.best_val_loss = current_val_loss
                    model_path = self.save_dir / "model.pt"
                    torch.save(self.model.state_dict(), model_path)
                    print("\n⚠️ Training interrupted – improved model saved to", model_path)
                else:
                    print("\n⚠️ Training interrupted – no improvement (val loss: %.4f), keeping existing model.pt" % current_val_loss)
            except Exception as e:
                print("\n⚠️ Training interrupted – evaluation failed:", e)
                # Fallback: still save current model state
                model_path = self.save_dir / "model.pt"
                torch.save(self.model.state_dict(), model_path)
                print("Saved current model to", model_path)
            raise

        self.save_training_log()
        return self.history

    def save_training_log(self) -> None:
        """Saves CSV and JSON history logs to save_dir."""
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
