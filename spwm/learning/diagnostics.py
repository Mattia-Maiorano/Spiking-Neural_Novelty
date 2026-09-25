"""
SNN Gradient and Spiking Diagnostics.
Essential tools for monitoring surrogate gradient backpropagation,
detecting vanishing/exploding gradients, and analyzing spike rate distributions.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn


def inspect_gradients(model: nn.Module) -> Dict[str, Any]:
    """
    Inspects parameter gradients across the entire model.
    Reports total gradient norm, individual parameter norms, NaN/Inf checks,
    and the fraction of dead (zero-gradient) parameters.
    """
    param_diagnostics = {}
    total_norm_sq = 0.0
    has_nan = False
    has_inf = False
    zero_grad_elements = 0
    total_elements = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.grad is None:
            param_diagnostics[name] = {
                "norm": 0.0,
                "has_grad": False,
                "is_nan": False,
                "is_inf": False,
                "zero_ratio": 1.0,
            }
            zero_grad_elements += param.numel()
            total_elements += param.numel()
            continue

        grad = param.grad.data
        grad_norm = grad.norm(2).item()
        total_norm_sq += grad_norm ** 2

        nan_present = torch.isnan(grad).any().item()
        inf_present = torch.isinf(grad).any().item()
        if nan_present:
            has_nan = True
        if inf_present:
            has_inf = True

        zero_count = (grad == 0.0).sum().item()
        zero_ratio = zero_count / max(1, grad.numel())
        zero_grad_elements += zero_count
        total_elements += grad.numel()

        param_diagnostics[name] = {
            "norm": grad_norm,
            "has_grad": True,
            "is_nan": nan_present,
            "is_inf": inf_present,
            "zero_ratio": zero_ratio,
        }

    total_norm = total_norm_sq ** 0.5
    overall_zero_ratio = zero_grad_elements / max(1, total_elements)

    return {
        "total_norm": total_norm,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "zero_gradient_ratio": overall_zero_ratio,
        "parameters": param_diagnostics,
    }


def collect_spike_statistics(spikes: torch.Tensor) -> Dict[str, Any]:
    """
    Computes statistical properties of spiking activity.
    spikes: [B, T, N_neurons] or [T, N_neurons]
    Returns:
        mean, median, min, max, fraction of silent neurons (<1% firing),
        fraction of hyperactive neurons (>50% firing), and histogram bins.
    """
    spk_np = spikes.detach().cpu().float().numpy()

    # Activity per neuron across time and batch
    # If [B, T, N], reduce over (0, 1) -> [N]
    reduce_dims = tuple(range(len(spk_np.shape) - 1))
    neuron_rates = np.mean(spk_np, axis=reduce_dims)

    mean_rate = float(np.mean(neuron_rates))
    median_rate = float(np.median(neuron_rates))
    min_rate = float(np.min(neuron_rates))
    max_rate = float(np.max(neuron_rates))

    silent_fraction = float(np.mean(neuron_rates < 0.01))
    hyperactive_fraction = float(np.mean(neuron_rates > 0.50))

    hist_counts, bin_edges = np.histogram(neuron_rates, bins=10, range=(0.0, 1.0))

    return {
        "mean_spike_rate": mean_rate,
        "median_spike_rate": median_rate,
        "min_spike_rate": min_rate,
        "max_spike_rate": max_rate,
        "fraction_silent_neurons": silent_fraction,
        "fraction_hyperactive_neurons": hyperactive_fraction,
        "histogram_counts": hist_counts.tolist(),
        "histogram_bin_edges": bin_edges.tolist(),
    }
