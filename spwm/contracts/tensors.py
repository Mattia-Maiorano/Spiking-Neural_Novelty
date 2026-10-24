"""
Layer 0 Tensor & Phase-Space Mathematical Contracts for SPWM.
Enforces strict runtime dimensionality, rank, finiteness, and action-conditioned phase-space invariants.
Zero external dependencies outside Python stdlib and PyTorch.
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch


def assert_tensor_contract(
    tensor: torch.Tensor,
    expected_shape: Tuple[Optional[int], ...],
    name: str,
    check_finite: bool = False,
) -> None:
    """
    Validates that `tensor` matches `expected_shape` (where `None` denotes a wildcard dimension).
    Raises `ValueError` immediately on rank or size mismatch.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name}: expected torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim != len(expected_shape):
        raise ValueError(
            f"{name}: expected {len(expected_shape)}D tensor with shape {expected_shape}, "
            f"got {tensor.ndim}D tensor with shape {tuple(tensor.shape)}"
        )
    for idx, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
        if expected is not None and actual != expected:
            raise ValueError(
                f"{name} dim {idx}: expected {expected}, got {actual} "
                f"(full shape {tuple(tensor.shape)} vs expected {expected_shape})"
            )
    if check_finite and not torch.isfinite(tensor).all():
        raise ValueError(f"{name}: tensor contains NaN or Inf values.")


def assert_phase_space_contract(
    latent_dim: int,
    q_dim: int,
    p_dim: int,
    num_keypoints: Optional[int] = None,
    action_dim: Optional[int] = None,
    actions: Optional[torch.Tensor] = None,
) -> None:
    """
    Enforces the action-conditioned phase-space latent decomposition invariants:
      1. q_dim > 0 and p_dim > 0 and latent_dim > 0
      2. q_dim + p_dim == latent_dim
      3. p_dim % q_dim == 0
      4. If num_keypoints is provided: 2 * num_keypoints == q_dim
      5. If action_dim is provided: action_dim > 0
      6. If actions tensor u_t is provided: validates rank >= 2, last dim == action_dim, and finiteness
    """
    if q_dim <= 0 or p_dim <= 0 or latent_dim <= 0:
        raise ValueError(
            f"Phase space dimensions must be strictly positive: "
            f"latent_dim={latent_dim}, q_dim={q_dim}, p_dim={p_dim}"
        )
    if q_dim + p_dim != latent_dim:
        raise ValueError(
            f"q_dim ({q_dim}) + p_dim ({p_dim}) != latent_dim ({latent_dim})"
        )
    if num_keypoints is not None and num_keypoints * 2 != q_dim:
        raise ValueError(
            f"num_keypoints ({num_keypoints}) * 2 must equal q_dim ({q_dim})"
        )
    if p_dim % q_dim != 0:
        raise ValueError(
            f"p_dim ({p_dim}) must be an integer multiple of q_dim ({q_dim}) for spatial boundary coupling."
        )
    if action_dim is not None and action_dim <= 0:
        raise ValueError(f"action_dim must be strictly positive, got {action_dim}")
    if actions is not None:
        if not isinstance(actions, torch.Tensor):
            raise TypeError(f"actions: expected torch.Tensor, got {type(actions).__name__}")
        if actions.ndim < 2:
            raise ValueError(
                f"actions: expected at least 2D tensor [B, ..., action_dim], got {actions.ndim}D tensor"
            )
        if action_dim is not None and actions.shape[-1] != action_dim:
            raise ValueError(
                f"actions last dim: expected action_dim={action_dim}, got {actions.shape[-1]} "
                f"(full shape {tuple(actions.shape)})"
            )
        if actions.shape[-1] <= 0:
            raise ValueError(f"actions last dim must be > 0, got {actions.shape[-1]}")
        if not torch.isfinite(actions).all():
            raise ValueError("actions: tensor contains NaN or Inf values.")


def assert_euclidean_coordinates(
    q: torch.Tensor,
    name: str = "q_coordinates",
    tolerance: float = 1e-5,
) -> None:
    """
    Verifies that geometric coordinates `q` lie within the normalized sensor plane [-1, 1]^q_dim.
    """
    if not isinstance(q, torch.Tensor):
        raise TypeError(f"{name}: expected torch.Tensor, got {type(q).__name__}")
    max_abs = float(q.abs().max().item()) if q.numel() > 0 else 0.0
    if max_abs > 1.0 + tolerance:
        raise ValueError(
            f"{name}: coordinates outside Euclidean arena bounds [-1.0, 1.0] (max |q| = {max_abs:.6f})"
        )
