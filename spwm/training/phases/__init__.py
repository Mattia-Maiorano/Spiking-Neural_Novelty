"""
Isolated Training Phase Strategies (Layer 2 — Training).

Exposes:
- `RepresentationPhaseStrategy` (`spwm/training/phases/representation.py`)
- `M1PhaseStrategy` (`spwm/training/phases/m1_nominal.py`)
- `build_phase_strategy` factory
- `PredictorPhaseHandler` dispatcher for unified lifecycle compatibility
"""

from __future__ import annotations

from typing import Dict, Optional
import chronicle
import torch
import torch.nn as nn

from spwm.contracts.config import ExperimentConfig
from spwm.contracts.protocols import PhaseStrategy
from spwm.training.phases.representation import RepresentationPhaseStrategy
from spwm.training.phases.m1_nominal import M1PhaseStrategy

PhaseHandler = PhaseStrategy


def build_phase_strategy(
    model: nn.Module,
    cfg: ExperimentConfig,
    loss_fn: nn.Module,
    device: torch.device,
) -> PhaseStrategy:
    """Instantiate the isolated PhaseStrategy corresponding to `cfg.training.mode`."""
    mode = cfg.training.mode.lower()
    if mode in ("train_representation", "train_phase0", "representation"):
        return RepresentationPhaseStrategy(model=model, cfg=cfg, loss_fn=loss_fn, device=device)
    if mode in ("train_m1", "m1"):
        return M1PhaseStrategy(model=model, cfg=cfg, loss_fn=loss_fn, device=device)
    raise ValueError(
        f"Unsupported training mode '{mode}'. Supported modes: 'train_representation', 'train_m1'"
    )


class PredictorPhaseHandler(PhaseStrategy):
    """Thin dispatcher that delegates to RepresentationPhaseStrategy or M1PhaseStrategy."""

    def setup(self) -> None:
        self.strategy: PhaseStrategy = build_phase_strategy(
            model=self.model,
            cfg=self.cfg,
            loss_fn=self.loss_fn,
            device=self.device,
        )
        self.predictor_optimizer = self.strategy.predictor_optimizer
        self.representation_optimizer = None

    @property
    def encoder_optimizer(self) -> Optional[torch.optim.Optimizer]:
        return self.representation_optimizer

    @property
    def probe_optimizer(self) -> Optional[torch.optim.Optimizer]:
        return self.representation_optimizer

    def verify_gradient_isolation(self) -> None:
        self.strategy.verify_gradient_isolation()

    def train_step(self, batch: Dict[str, torch.Tensor], epoch: int) -> Dict[str, float]:
        return self.strategy.train_step(batch, epoch=epoch)

    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        return self.strategy.validate_step(batch)

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
        train_loss = train_metrics.get("train_loss", 0.0)
        train_pos_l = train_metrics.get("train_loss_position", 0.0)
        train_vel_l = train_metrics.get("train_loss_velocity", 0.0)
        train_pos_e = train_metrics.get("train_pos_err", 0.0)
        train_vel_e = train_metrics.get("train_vel_err", 0.0)
        chronicle.log_detail(
            "⚡ Training Losses",
            f"Total: {train_loss:.5f} [q-MSE: {train_pos_l:.5f} | p-MSE: {train_vel_l:.5f}] | "
            f"Eval: Joint Angle L2: {train_pos_e:.5f} | Joint Vel L2: {train_vel_e:.5f}",
            indent_level=1,
        )

        val_loss = val_metrics.get("val_loss", 0.0)
        val_pos_l = val_metrics.get("val_loss_position", 0.0)
        val_vel_l = val_metrics.get("val_loss_velocity", 0.0)
        val_pos_e = val_metrics.get("val_pos_err", 0.0)
        val_vel_e = val_metrics.get("val_vel_err", 0.0)
        chronicle.log_detail(
            "📊 Validation Losses",
            f"Total: {val_loss:.5f} [q-MSE: {val_pos_l:.5f} | p-MSE: {val_vel_l:.5f}] | "
            f"Eval: Joint Angle L2: {val_pos_e:.5f} | Joint Vel L2: {val_vel_e:.5f}",
            indent_level=1,
        )
        curr_score = self.get_selection_score(val_metrics)
        chronicle.log_detail(
            "🎯 Model Selection",
            f"Current: {curr_score:.5f} | Best: {best_score:.5f} @ Ep {best_epoch}",
            indent_level=1,
        )
        chronicle.log_newline()


__all__ = [
    "PhaseStrategy",
    "PhaseHandler",
    "RepresentationPhaseStrategy",
    "M1PhaseStrategy",
    "build_phase_strategy",
    "PredictorPhaseHandler",
]
