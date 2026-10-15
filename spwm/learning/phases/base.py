"""
Abstract Base Class for Specialized Training Phase Handlers.
Defines the uniform lifecycle contract for training phases (Predictor, Corrector, etc.).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spwm.config.schema import ExperimentConfig
from spwm.learning.losses import SPWMLoss


class PhaseHandler(ABC):
    """
    Uniform contract for phase-specific training dynamics.
    Enforces isolation between training paradigms without sprawling conditional checks.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: ExperimentConfig,
        loss_fn: SPWMLoss,
        device: torch.device,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.loss_fn = loss_fn
        self.device = device
        self.setup()

    @abstractmethod
    def setup(self) -> None:
        """Sets parameter requires_grad states and initializes phase-specific optimizers."""
        pass

    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor], epoch: int) -> Dict[str, float]:
        """Executes one training step on the given batch and returns metrics dictionary."""
        pass

    @abstractmethod
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Executes one validation step and returns metrics dictionary."""
        pass

    @abstractmethod
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
        """Logs epoch metrics to chronicle, reading directly from source-of-truth metrics dictionaries."""
        pass

    @abstractmethod
    def get_selection_score(self, val_metrics: Dict[str, float]) -> float:
        """Computes model selection score for checkpointing (lower is better)."""
        pass

    @abstractmethod
    def get_optimizers_state(self) -> Dict[str, Any]:
        """Returns serializable dictionary of optimizer states."""
        pass

    @abstractmethod
    def load_optimizers_state(self, state: Dict[str, Any]) -> None:
        """Restores optimizer states from checkpoint."""
        pass
