"""
BaseTrainer Engine for SPWM.
Single source of truth for the training epoch loop, timing, checkpointing,
metrics logging, and phase dispatch.
"""

from __future__ import annotations
import csv
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import chronicle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spwm.config.schema import ExperimentConfig
from spwm.learning.losses import SPWMLoss
from spwm.learning.phases import PhaseHandler, PredictorPhaseHandler, CorrectorPhaseHandler


class BaseTrainer:
    """
    Standardized, modular training engine.
    Orchestrates epoch lifecycle, checkpoint management, and delegates
    step-level mechanics to specialized PhaseHandlers.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: ExperimentConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Optional[SPWMLoss] = None,
        device: Optional[Union[str, torch.device]] = None,
        save_dir: Union[str, Path] = "results/default_run",
        start_epoch: int = 1,
        best_score: float = float("inf"),
        best_epoch: Optional[int] = None,
        history: Optional[List[Dict[str, float]]] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if loss_fn is None:
            self.loss_fn = SPWMLoss.from_config(
                cfg.loss, q_dim=cfg.model.q_dim, p_dim=cfg.model.p_dim
            ).to(self.device)
        else:
            self.loss_fn = loss_fn.to(self.device)

        self.mode = cfg.training.mode.lower()
        if self.mode == "train_predictor":
            self.handler: PhaseHandler = PredictorPhaseHandler(
                model=self.model, cfg=self.cfg, loss_fn=self.loss_fn, device=self.device
            )
        elif self.mode == "train_corrector":
            self.handler = CorrectorPhaseHandler(
                model=self.model, cfg=self.cfg, loss_fn=self.loss_fn, device=self.device
            )
        else:
            raise ValueError(f"Unsupported training mode '{self.mode}'. Allowed: 'train_predictor', 'train_corrector'")

        self.start_epoch = start_epoch
        self.best_score = best_score
        self.best_epoch = best_epoch
        self.history: List[Dict[str, float]] = history or []

    @property
    def predictor_optimizer(self):
        return getattr(self.handler, "predictor_optimizer", None)

    @property
    def encoder_optimizer(self):
        return getattr(self.handler, "encoder_optimizer", None)

    @property
    def probe_optimizer(self):
        return getattr(self.handler, "probe_optimizer", None)

    @property
    def corrector_optimizer(self):
        return getattr(self.handler, "corrector_optimizer", None)

    def verify_gradient_isolation(self) -> None:
        """Verifies that in train_corrector mode, non-corrector parameters are strictly frozen."""
        if self.mode == "train_corrector":
            for name, p in self.model.named_parameters():
                if "corrector" not in name and p.requires_grad:
                    raise AssertionError(f"Gradient leak! Parameter '{name}' has requires_grad=True in train_corrector mode.")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Executes a single epoch over train_loader using the active PhaseHandler."""
        train_accum: Dict[str, float] = {}
        batch_count = 0
        for batch in self.train_loader:
            step_metrics = self.handler.train_step(batch, epoch=epoch)
            for k, v in step_metrics.items():
                train_accum[k] = train_accum.get(k, 0.0) + v
            batch_count += 1
        return {k: v / max(1, batch_count) for k, v in train_accum.items()}

    def fit(self, epochs: Optional[int] = None) -> List[Dict[str, float]]:

        total_epochs = epochs if epochs is not None else self.cfg.training.epochs
        end_epoch = self.start_epoch + total_epochs - 1

        chronicle.log_application_title(f"Starting Training: {self.cfg.project.name} [{self.mode.upper()}]")
        chronicle.log_detail("Device", str(self.device))
        chronicle.log_detail("Epochs", f"{self.start_epoch} -> {end_epoch}")
        chronicle.log_detail("Save Directory", str(self.save_dir))

        start_time = time.time()
        try:
            for epoch in range(self.start_epoch, end_epoch + 1):
                t_ep_start = time.time()

                # Train epoch loop
                train_accum: Dict[str, float] = {}
                batch_count = 0
                for batch in self.train_loader:
                    step_metrics = self.handler.train_step(batch, epoch=epoch)
                    for k, v in step_metrics.items():
                        train_accum[k] = train_accum.get(k, 0.0) + v
                    batch_count += 1

                train_metrics = {k: v / max(1, batch_count) for k, v in train_accum.items()}

                # Validation epoch loop
                val_metrics = self.evaluate(self.val_loader)

                score = self.handler.get_selection_score(val_metrics)
                is_best = score < self.best_score
                if is_best:
                    self.best_score = score
                    self.best_epoch = epoch
                    self.save_checkpoint(epoch, is_best=True)

                ep_duration = time.time() - t_ep_start
                total_elapsed = time.time() - start_time

                log_entry = {
                    "epoch": epoch,
                    **train_metrics,
                    **val_metrics,
                    "score": score,
                    "is_best": is_best,
                    "duration": ep_duration,
                }
                self.history.append(log_entry)

                # Periodic checkpoint
                self.save_checkpoint(epoch, is_best=False)

                # Chronicle logging
                self.handler.log_epoch(
                    epoch=epoch,
                    end_epoch=end_epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    epoch_duration=ep_duration,
                    total_elapsed=total_elapsed,
                    is_best=is_best,
                    best_epoch=self.best_epoch,
                    best_score=self.best_score,
                )

        except KeyboardInterrupt:
            chronicle.log_warning("[INTERRUPT] Training interrupted manually.")

        self.save_training_log()
        chronicle.log_success(
            f"Session complete. Logs saved in '{self.save_dir}'. "
            f"Best Selection Score: {self.best_score:.5f} @ Epoch {self.best_epoch}"
        )
        return self.history

    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        target_loader = loader if loader is not None else self.val_loader
        val_accum: Dict[str, float] = {}
        batch_count = 0
        for batch in target_loader:
            step_metrics = self.handler.validate_step(batch)
            for k, v in step_metrics.items():
                val_accum[k] = val_accum.get(k, 0.0) + v
            batch_count += 1

        return {k: v / max(1, batch_count) for k, v in val_accum.items()}

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        ckpt = {
            "epoch": epoch,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "model_state": self.model.state_dict(),
            "optimizers_state": self.handler.get_optimizers_state(),
            "config": self.cfg.to_dict(),
            "history": self.history,
        }
        if is_best:
            torch.save(ckpt, self.save_dir / "model.pt")
        torch.save(ckpt, self.save_dir / "checkpoint.pt")

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            self.model.load_state_dict(ckpt["model_state"], strict=False)
            self.start_epoch = ckpt.get("epoch", 0) + 1
            self.best_score = ckpt.get("best_score", float("inf"))
            self.best_epoch = ckpt.get("best_epoch", None)
            self.history = ckpt.get("history", []) or []
            if "optimizers_state" in ckpt:
                self.handler.load_optimizers_state(ckpt["optimizers_state"])
        else:
            self.model.load_state_dict(ckpt, strict=False)

    def save_training_log(self) -> None:
        if not self.history:
            return

        json_path = self.save_dir / "training_log.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        csv_path = self.save_dir / "training_log.csv"
        fieldnames = []
        for rec in self.history:
            for k in rec.keys():
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.history)
