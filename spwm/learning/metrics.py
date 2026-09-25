"""
Scientific evaluation metrics for SPWM.
Covers latent prediction accuracy, physical kinematic decoding,
spike sparsity, parameter capacity, and neuromorphic energy estimation.
"""

from __future__ import annotations
from typing import Dict, Union
import numpy as np
import torch
import torch.nn as nn


def one_step_mse(z_pred: torch.Tensor, z_target: torch.Tensor) -> float:
    """Computes Mean Squared Error for 1-step latent prediction."""
    return torch.mean((z_pred - z_target) ** 2).item()


def multi_step_mse(rollout_predictions: torch.Tensor, target_latents: torch.Tensor) -> float:
    """
    Computes average MSE across autonomous rollout horizon.
    rollout_predictions: [B, H, D]
    target_latents: [B, H, D]
    """
    return torch.mean((rollout_predictions - target_latents) ** 2).item()


def position_error(decoded_kinematics: torch.Tensor, true_kinematics: torch.Tensor, num_objects: int = 1) -> float:
    """
    Mean L2 Euclidean distance between predicted and true object positions.
    decoded_kinematics: [..., 4 * N] (x, y, vx, vy)
    """
    # Position occupies first 2 * num_objects coordinates
    pred_pos = decoded_kinematics[..., : 2 * num_objects].reshape(-1, num_objects, 2)
    true_pos = true_kinematics[..., : 2 * num_objects].reshape(-1, num_objects, 2)
    dist = torch.norm(pred_pos - true_pos, dim=-1)  # [..., N]
    return torch.mean(dist).item()


def velocity_error(decoded_kinematics: torch.Tensor, true_kinematics: torch.Tensor, num_objects: int = 1) -> float:
    """
    Mean L2 Euclidean distance between predicted and true object velocities.
    """
    pred_vel = decoded_kinematics[..., 2 * num_objects : 4 * num_objects].reshape(-1, num_objects, 2)
    true_vel = true_kinematics[..., 2 * num_objects : 4 * num_objects].reshape(-1, num_objects, 2)
    dist = torch.norm(pred_vel - true_vel, dim=-1)
    return torch.mean(dist).item()


def spike_rate(spikes: torch.Tensor) -> float:
    """Calculates mean spike rate per neuron per timestep (in [0.0, 1.0])."""
    return torch.mean(spikes.float()).item()


def active_neuron_fraction(spikes: torch.Tensor, min_activity_threshold: float = 0.01) -> float:
    """
    Fraction of neurons that spiked at least once or above min_activity_threshold across the sequence.
    spikes: [B, T, N_neurons]
    """
    # Average firing across batch and time for each neuron
    neuron_activity = spikes.float().mean(dim=(0, 1))
    active_mask = (neuron_activity >= min_activity_threshold).float()
    return torch.mean(active_mask).item()


def parameter_count(model: nn.Module) -> int:
    """Returns total number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def energy_estimate(total_spikes: int, avg_fan_out: int = 128, energy_per_sop_pj: float = 0.9) -> float:
    """
    Neuromorphic Synaptic Operations (SOP) Energy Estimate.
    On neuromorphic architectures (e.g. Intel Loihi, TrueNorth), energy consumption is event-driven:
        Energy = Total_Spikes * FanOut * Energy_per_SOP
    Standard estimate ~0.9 pJ per synaptic operation on 14nm neuromorphic silicon.
    Returns: estimated energy in nanojoules (nJ).
    """
    total_sops = total_spikes * avg_fan_out
    energy_pj = total_sops * energy_per_sop_pj
    return energy_pj / 1000.0  # nanojoules
