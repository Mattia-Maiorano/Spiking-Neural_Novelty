"""
MPPI Action Noise Injection & 4-Phase Latch Curriculum Utilities (Layer 2 — Training).

Provides AR(1) Gaussian action noise sampling matching LatentMPPIController,
irreversible 4-Phase Latch Curriculum state machine, and DAgger scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple
import torch


@dataclass
class LatchCurriculumState:
    """State-machine-based 4-Phase Curriculum with irreversible threshold latches.

    Phases:
      1. Warmup Nominal: 100% Open-Loop, alpha = 0.0, w_t = 0.60.
         Trigger -> Phase 2: train_loss < 0.01.
      2. Sensory Fade-Out: 100% Open-Loop, alpha = 0.0, linearly decay w_t from 0.60 to 0.0 over 7 epochs.
         Trigger -> Phase 3: w_t == 0.0 and train_loss < 0.008.
      3. Noise Ramp-Up: 100% Open-Loop, w_t = 0.0, linearly ramp alpha from 0.0 to 0.4 over 10 epochs.
         Trigger -> Phase 4: alpha == 0.4.
      4. Fast DAgger Integration: Mixed batch scaling DAgger ratio from 10% to 50% over 2 * tier2_frequency epochs.
         alpha fixed at 0.4, w_t fixed at 0.0. Open-loop portion still receives alpha = 0.4 noise injection.
         Runs indefinitely until Tier 2 Early Stopping halts training.
    """

    current_curriculum_phase: int = 1
    phase_epochs: int = 0
    w_t: float = 0.60
    alpha: float = 0.0
    dagger_ratio: float = 0.0
    tier2_frequency: int = 5
    sensory_fade_epochs: int = 7
    noise_ramp_epochs: int = 20
    target_alpha: float = 0.4
    phase1_loss_threshold: float = 0.01
    phase2_loss_threshold: float = 0.008
    min_dagger_ratio: float = 0.10
    max_dagger_ratio: float = 0.50

    def begin_epoch(self) -> Tuple[int, float, float, float]:
        """Advance epoch counter within active phase and return (phase, w_t, alpha, dagger_ratio)."""
        if self.current_curriculum_phase == 1:
            self.phase_epochs += 1
            self.w_t = 0.60
            self.alpha = 0.0
            self.dagger_ratio = 0.0
        elif self.current_curriculum_phase == 2:
            self.phase_epochs += 1
            progress = min(1.0, self.phase_epochs / float(max(1, self.sensory_fade_epochs)))
            self.w_t = round(max(0.0, 0.60 * (1.0 - progress)), 6)
            self.alpha = 0.0
            self.dagger_ratio = 0.0
        elif self.current_curriculum_phase == 3:
            self.phase_epochs += 1
            self.w_t = 0.0
            progress = min(1.0, self.phase_epochs / float(max(1, self.noise_ramp_epochs)))
            self.alpha = round(min(self.target_alpha, self.target_alpha * progress), 6)
            self.dagger_ratio = 0.0
        else:
            self.current_curriculum_phase = 4
            self.w_t = 0.0
            self.alpha = float(self.target_alpha)
            span = max(1, 2 * self.tier2_frequency)
            progress = min(1.0, self.phase_epochs / float(span))
            self.dagger_ratio = round(
                self.min_dagger_ratio + progress * (self.max_dagger_ratio - self.min_dagger_ratio), 6
            )
            self.phase_epochs += 1

        return self.current_curriculum_phase, self.w_t, self.alpha, self.dagger_ratio

    def end_epoch(self, train_loss: float) -> int:
        """Evaluate irreversible latch transitions at the conclusion of a training epoch."""
        if self.current_curriculum_phase == 1:
            if train_loss < self.phase1_loss_threshold:
                self.current_curriculum_phase = 2
                self.phase_epochs = 0
        elif self.current_curriculum_phase == 2:
            if self.w_t == 0.0 and train_loss < self.phase2_loss_threshold:
                self.current_curriculum_phase = 3
                self.phase_epochs = 0
        elif self.current_curriculum_phase == 3:
            if abs(self.alpha - self.target_alpha) < 1e-6 or self.alpha >= self.target_alpha:
                self.current_curriculum_phase = 4
                self.phase_epochs = 0
        return self.current_curriculum_phase

    def state_dict(self) -> Dict[str, Any]:
        return {
            "current_curriculum_phase": self.current_curriculum_phase,
            "phase_epochs": self.phase_epochs,
            "w_t": self.w_t,
            "alpha": self.alpha,
            "dagger_ratio": self.dagger_ratio,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "current_curriculum_phase" in state:
            self.current_curriculum_phase = max(self.current_curriculum_phase, int(state["current_curriculum_phase"]))
        if "phase_epochs" in state:
            self.phase_epochs = int(state["phase_epochs"])
        if "w_t" in state:
            self.w_t = float(state["w_t"])
        if "alpha" in state:
            self.alpha = float(state["alpha"])
        if "dagger_ratio" in state:
            self.dagger_ratio = float(state["dagger_ratio"])


def sample_ar1_noise(
    batch_size: int,
    horizon: int,
    action_dim: int,
    device: torch.device,
    beta: float = 0.75,
    base_sigma: float = 0.28,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Samples temporally correlated Gaussian noise following an AR(1) process matching LatentMPPIController."""
    sigmas = torch.full((action_dim,), float(base_sigma), device=device, dtype=torch.float32)
    if action_dim >= 3:
        # Match LatentMPPIController anisotropic covariance boost: Shoulder_Lift (idx 1) +250%, Elbow_Flex (idx 2) +100%
        sigmas[1] = base_sigma * 3.5
        sigmas[2] = base_sigma * 2.0

    decay = math.sqrt(max(0.0, 1.0 - beta ** 2))
    eta = torch.randn(batch_size, horizon, action_dim, generator=generator, device=device) * sigmas.view(1, 1, action_dim)
    noise = torch.zeros(batch_size, horizon, action_dim, device=device)
    curr_eps = eta[:, 0]
    noise[:, 0] = curr_eps
    for tau in range(1, horizon):
        curr_eps = beta * curr_eps + decay * eta[:, tau]
        noise[:, tau] = curr_eps
    return noise


def compute_mppi_noise_alpha(
    epoch: int,
    warmup_epochs: int = 10,
    ramp_epochs: int = 40,
    target_alpha: float = 0.5,
) -> float:
    """Computes curriculum-scheduled noise intensity alpha for MPPI-aware training."""
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return float(target_alpha)
    progress = min(1.0, (epoch - warmup_epochs) / float(ramp_epochs))
    return float(progress * target_alpha)


def compute_dagger_ratio(
    epoch: int,
    warmup_epochs: int = 10,
    ramp_end_epoch: int = 40,
    min_ratio: float = 0.15,
    max_ratio: float = 0.60,
) -> float:
    """Computes dynamic epoch-based DAgger batch ratio for closed-loop training.

    Rule:
      - Epochs 0..warmup_epochs (e.g. 0-10): strictly 0% closed-loop data (warmup).
      - From warmup_epochs + 1 (e.g. 11) onward: linearly scale from min_ratio (15%)
        up to hard cap max_ratio (60%) at ramp_end_epoch (40).
    """
    if epoch <= warmup_epochs:
        return 0.0
    if epoch >= ramp_end_epoch:
        return float(max_ratio)
    span = ramp_end_epoch - (warmup_epochs + 1)
    if span <= 0:
        return float(max_ratio)
    fraction = (epoch - (warmup_epochs + 1)) / float(span)
    ratio = min_ratio + fraction * (max_ratio - min_ratio)
    return float(min(max(ratio, min_ratio), max_ratio))
