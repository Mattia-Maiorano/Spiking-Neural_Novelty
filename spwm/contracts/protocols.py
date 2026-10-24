"""
Layer 0 Protocols and Abstract Base Contracts for SPWM.
Defines strict structural interfaces for all neural components and training phase strategies,
eliminating duck-typed reflection (`getattr`/`hasattr`) and circular imports.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol, Tuple, Union, runtime_checkable
import torch
import torch.nn as nn

from spwm.contracts.config import ExperimentConfig
from spwm.contracts.states import (
    DynamicsOutput,
    DynamicsState,
    PredictorOutput,
    SPWMSequenceOutput,
    SPWMState,
    SPWMStepOutput,
)


@runtime_checkable
class EncoderProtocol(Protocol):
    """Contract for visual event encoders with SpatialSoftmax keypoint bottlenecks."""
    in_channels: int
    num_keypoints: int
    out_dim: int

    def encode_step(self, event_frame: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes a single frame [B, C, H, W] -> (features [B, D_enc], keypoints [B, 2*K])."""
        ...

    def __call__(
        self,
        x: torch.Tensor,
        return_keypoints: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        ...


@runtime_checkable
class DynamicsProtocol(Protocol):
    """Contract for recurrent spiking latent dynamics."""
    input_dim: int
    latent_dim: int
    q_dim: int
    p_dim: int

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> DynamicsState:
        ...

    def step(
        self,
        sensory_input: torch.Tensor,
        state: Optional[DynamicsState] = None,
        sensory_keypoints: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, DynamicsState]:
        ...

    def __call__(
        self,
        sensory_sequence: torch.Tensor,
        initial_state: Optional[DynamicsState] = None,
        sensory_keypoints: Optional[torch.Tensor] = None,
    ) -> Tuple[DynamicsOutput, DynamicsState]:
        ...


@runtime_checkable
class PredictorModuleProtocol(Protocol):
    """Contract for a single stateless latent transition predictor (M1, M2, or M3)."""
    latent_dim: int

    def __call__(self, z: torch.Tensor) -> PredictorOutput:
        ...


@runtime_checkable
class PredictorProtocol(Protocol):
    """Contract for the symplectic latent predictor (M1)."""
    latent_dim: int
    hidden_dim: int
    symplectic: bool
    dt: float

    def __call__(self, z: torch.Tensor) -> PredictorOutput:
        ...


@runtime_checkable
class PhysicalDecoderProtocol(Protocol):
    """Contract for kinematic position & velocity readout probe."""
    latent_dim: int
    q_dim: int
    p_dim: int
    num_objects: int

    def decode_pos(self, q: torch.Tensor) -> torch.Tensor:
        ...

    def decode_vel(self, p: torch.Tensor) -> torch.Tensor:
        ...

    def warm_up_filter(
        self,
        z_prefix: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ...

    def __call__(
        self,
        z: torch.Tensor,
        initial_filter_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ...


@runtime_checkable
class WorldModelProtocol(Protocol):
    """Contract for the composite SPWM world model."""
    latent_dim: int
    q_dim: int
    p_dim: int
    num_objects: int
    encoder: nn.Module
    dynamics: nn.Module
    predictor: nn.Module
    physical_decoder: PhysicalDecoderProtocol

    @property
    def m1(self) -> nn.Module:
        ...

    @property
    def m2(self) -> nn.Module:
        ...

    def forward_representation(
        self,
        event_sequence: torch.Tensor,
        initial_state: Optional[SPWMState] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    def __call__(
        self,
        event_sequence: torch.Tensor,
        initial_state: Optional[SPWMState] = None,
    ) -> SPWMSequenceOutput:
        ...


class PhaseStrategy(ABC):
    """
    Uniform contract for isolated phase-specific training strategies (M1, M2, M3).
    Each concrete strategy lives in its own module (< 150 lines) so small LLMs can
    inspect and modify a single training phase with < 3 files in context.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: ExperimentConfig,
        loss_fn: nn.Module,
        device: torch.device,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.loss_fn = loss_fn
        self.device = device
        self.predictor_optimizer: Optional[torch.optim.Optimizer] = None
        self.setup()

    @abstractmethod
    def setup(self) -> None:
        """Sets parameter requires_grad states and initializes phase-specific optimizer(s)."""
        pass

    @abstractmethod
    def verify_gradient_isolation(self) -> None:
        """Asserts that only authorized parameters have requires_grad=True for this phase."""
        pass

    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor], epoch: int) -> Dict[str, float]:
        """Executes one training step on the given batch and returns metrics dictionary."""
        pass

    @abstractmethod
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Executes one validation step and returns metrics dictionary."""
        pass

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
        """Logs epoch metrics (may be overridden by subclasses for phase-specific formatting)."""
        pass

    def get_selection_score(self, val_metrics: Dict[str, float]) -> float:
        """Computes model selection score for checkpointing (lower is better)."""
        return val_metrics["val_loss"]

    def get_optimizers_state(self) -> Dict[str, Any]:
        """Returns serializable dictionary of optimizer states."""
        if self.predictor_optimizer is not None:
            return {"predictor": self.predictor_optimizer.state_dict()}
        return {}

    def load_optimizers_state(self, state: Dict[str, Any]) -> None:
        """Restores optimizer states from checkpoint."""
        if "predictor" in state and self.predictor_optimizer is not None:
            self.predictor_optimizer.load_state_dict(state["predictor"])


# Alias for backward compatibility
PhaseHandler = PhaseStrategy
