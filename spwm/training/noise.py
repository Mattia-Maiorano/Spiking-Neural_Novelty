"""
MPPI Action Noise Injection & Curriculum Utilities (Layer 2 — Training).

Provides AR(1) Gaussian action noise sampling matching LatentMPPIController
and linear warmup curriculum scheduling for MPPI-aware training.
"""

from __future__ import annotations

import math
from typing import Optional
import torch


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
