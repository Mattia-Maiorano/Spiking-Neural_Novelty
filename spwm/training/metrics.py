"""
Scientific Evaluation & Training Metrics for SPWM (Layer 2 — Training).

Covers latent prediction accuracy, articulated joint angle & angular velocity L2 errors,
spike sparsity, parameter capacity, and neuromorphic energy estimation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spwm.contracts.tensors import J, assert_tensor_contract


def one_step_mse(z_pred: torch.Tensor, z_target: torch.Tensor) -> float:
    """Computes Mean Squared Error for 1-step latent prediction."""
    return float(torch.mean((z_pred - z_target) ** 2).item())


def multi_step_mse(rollout_predictions: torch.Tensor, target_latents: torch.Tensor) -> float:
    """Computes average MSE across autonomous rollout horizon [B, H, D]."""
    return float(torch.mean((rollout_predictions - target_latents) ** 2).item())


def position_error(
    decoded_kinematics: torch.Tensor,
    true_kinematics: torch.Tensor,
    joint_count: int = J,
) -> float:
    """Mean L2 Euclidean distance between predicted and ground-truth joint angles q*."""
    pred_pos = decoded_kinematics[..., :joint_count].reshape(-1, joint_count)
    true_pos = true_kinematics[..., :joint_count].reshape(-1, joint_count)
    assert_tensor_contract(pred_pos, (None, joint_count), "position_error.pred_pos")
    assert_tensor_contract(true_pos, (None, joint_count), "position_error.true_pos")
    dist = torch.norm(pred_pos - true_pos, dim=-1)
    return float(torch.mean(dist).item())


def velocity_error(
    decoded_kinematics: torch.Tensor,
    true_kinematics: torch.Tensor,
    joint_count: int = J,
) -> float:
    """Mean L2 Euclidean distance between predicted and ground-truth joint angular velocities p*."""
    pred_vel = decoded_kinematics[..., joint_count : 2 * joint_count].reshape(-1, joint_count)
    true_vel = true_kinematics[..., joint_count : 2 * joint_count].reshape(-1, joint_count)
    assert_tensor_contract(pred_vel, (None, joint_count), "velocity_error.pred_vel")
    assert_tensor_contract(true_vel, (None, joint_count), "velocity_error.true_vel")
    dist = torch.norm(pred_vel - true_vel, dim=-1)
    return float(torch.mean(dist).item())


def spike_rate(spikes: torch.Tensor) -> float:
    """Calculates mean spike rate per neuron per timestep (in [0.0, 1.0])."""
    return float(torch.mean(spikes.float()).item())


def active_neuron_fraction(spikes: torch.Tensor, min_activity_threshold: float = 0.01) -> float:
    """Fraction of neurons spiking at or above min_activity_threshold across [B, T, N]."""
    neuron_activity = spikes.float().mean(dim=(0, 1))
    active_mask = (neuron_activity >= min_activity_threshold).float()
    return float(torch.mean(active_mask).item())


def parameter_count(model: nn.Module) -> int:
    """Returns total number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def energy_estimate(total_spikes: int, avg_fan_out: int = 128, energy_per_sop_pj: float = 0.9) -> float:
    """Neuromorphic Synaptic Operations (SOP) Energy Estimate in nanojoules (nJ)."""
    total_sops = total_spikes * avg_fan_out
    energy_pj = total_sops * energy_per_sop_pj
    return float(energy_pj / 1000.0)


def compute_sops(spike_count: int, fan_out: int) -> int:
    """Computes total Synaptic Operations (SOPs) given emitted spike count and synaptic fan-out.

    Formula: SOPs = sum_{l} s_{l-1} * F_{out, l}
    """
    if spike_count < 0 or fan_out < 0:
        raise ValueError(f"spike_count ({spike_count}) and fan_out ({fan_out}) must be non-negative.")
    return int(spike_count * fan_out)


def compute_neuromorphic_energy_nj(
    sops: int,
    num_neurons: int,
    energy_per_sop_pj: float = 0.9,
    energy_leak_pj: float = 0.1,
) -> float:
    """Computes neuromorphic hardware energy in nanojoules (nJ) using Intel Loihi-2 benchmarks.

    Formula: E_neuro = SOPs * E_SOP + N_neurons * E_leak
    """
    if sops < 0 or num_neurons < 0:
        raise ValueError(f"sops ({sops}) and num_neurons ({num_neurons}) must be non-negative.")
    energy_pj = float(sops * energy_per_sop_pj + num_neurons * energy_leak_pj)
    return float(energy_pj / 1000.0)


def compute_dense_energy_nj(
    flops: int,
    energy_per_flop_pj: float = 4.0,
) -> float:
    """Computes GPU/Edge accelerator dense energy in nanojoules (nJ) (e.g. Nvidia Jetson Orin / RTX).

    Formula: E_GPU = FLOPs * E_FLOP
    """
    if flops < 0:
        raise ValueError(f"flops ({flops}) must be non-negative.")
    energy_pj = float(flops * energy_per_flop_pj)
    return float(energy_pj / 1000.0)


def calculate_linear_macs_and_flops(in_features: int, out_features: int, has_bias: bool = True) -> tuple[int, int]:
    """Calculates MACs and FLOPs for a Linear layer."""
    macs = in_features * out_features
    flops = 2 * macs + (out_features if has_bias else 0)
    return macs, flops


def calculate_conv2d_macs_and_flops(
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, int],
    out_hw: tuple[int, int],
    has_bias: bool = True,
) -> tuple[int, int]:
    """Calculates MACs and FLOPs for a Conv2D layer."""
    kh, kw = kernel_size
    oh, ow = out_hw
    macs = in_channels * out_channels * kh * kw * oh * ow
    flops = 2 * macs + (out_channels * oh * ow if has_bias else 0)
    return macs, flops


__all__ = [
    "one_step_mse",
    "multi_step_mse",
    "position_error",
    "velocity_error",
    "spike_rate",
    "active_neuron_fraction",
    "parameter_count",
    "energy_estimate",
    "compute_sops",
    "compute_neuromorphic_energy_nj",
    "compute_dense_energy_nj",
    "calculate_linear_macs_and_flops",
    "calculate_conv2d_macs_and_flops",
]
