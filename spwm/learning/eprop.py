"""
Forward-Only e-prop Plasticity Engine for ALIF Spiking Neurons.
Implements dual eligibility traces (membrane potential V and threshold adaptation A)
under deterministic feedback for O(1) memory footprint scaling over long sequence horizons.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from spwm.models.neurons import ALIFCell, ALIFState


@dataclass
class ALIFEpropTraces:
    """Eligibility trace container for an ALIF synaptic connection."""
    trace_v: torch.Tensor  # Filtered presynaptic input [B, in_dim]
    trace_a: torch.Tensor  # Synaptic adaptation trace [B, out_dim, in_dim]
    psi_prev: torch.Tensor  # Previous surrogate derivative [B, out_dim]


def init_eprop_traces(
    batch_size: int,
    out_dim: int,
    in_dim: int,
    device: Optional[torch.device] = None,
) -> ALIFEpropTraces:
    """Initializes quiescent e-prop eligibility traces."""
    return ALIFEpropTraces(
        trace_v=torch.zeros(batch_size, in_dim, device=device, dtype=torch.float32),
        trace_a=torch.zeros(batch_size, out_dim, in_dim, device=device, dtype=torch.float32),
        psi_prev=torch.zeros(batch_size, out_dim, device=device, dtype=torch.float32),
    )


def step_alif_eprop_traces(
    cell: ALIFCell,
    curr_state: ALIFState,
    x_pre: torch.Tensor,
    traces: Optional[ALIFEpropTraces] = None,
) -> Tuple[torch.Tensor, ALIFEpropTraces]:
    """
    Computes instantaneous ALIF eligibility trace e_ij(t) and propagates internal traces forward:
        x_bar_j^v(t) = beta_mem * x_bar_j^v(t-1) + x_j(t)
        eps_ij^a(t) = beta_adapt_i * eps_ij^a(t-1) + psi_i(t-1) * (x_bar_j^v(t-1) - gamma * eps_ij^a(t-1))
        e_ij(t) = psi_i(t) * (x_bar_j^v(t) - gamma * eps_ij^a(t))

    Args:
        cell: ALIFCell instance
        curr_state: Current post-synaptic ALIFState (after forward step at time t)
        x_pre: Pre-synaptic activation/input [B, in_dim] at time t
        traces: Eligibility traces from time t-1

    Returns:
        e_trace: Instantaneous eligibility matrix [B, out_dim, in_dim]
        new_traces: Updated ALIFEpropTraces container
    """
    B, in_dim = x_pre.shape
    out_dim = cell.size
    device = x_pre.device

    if traces is None:
        traces = init_eprop_traces(B, out_dim, in_dim, device=device)

    # 1. Propagate filtered presynaptic input
    # beta_mem: [out_dim] -> use mean or uniform beta_mem value for presynaptic filter
    beta_mem_scalar = cell.beta_mem[0].item() if cell.beta_mem.numel() > 0 else 0.80
    new_trace_v = beta_mem_scalar * traces.trace_v + x_pre  # [B, in_dim]

    # 2. Update adaptation trace eps_ij^a(t)
    # beta_adapt: [out_dim] -> reshape to [1, out_dim, 1]
    beta_adapt_3d = cell.beta_adapt.view(1, out_dim, 1)  # [1, out_dim, 1]
    psi_prev_3d = traces.psi_prev.unsqueeze(-1)  # [B, out_dim, 1]
    trace_v_prev_3d = traces.trace_v.unsqueeze(1)  # [B, 1, in_dim]

    new_trace_a = (
        beta_adapt_3d * traces.trace_a
        + psi_prev_3d * (trace_v_prev_3d - cell.gamma * traces.trace_a)
    )  # [B, out_dim, in_dim]

    # 3. Compute instantaneous surrogate derivative psi_i(t)
    psi_curr = cell.compute_surrogate_derivative(curr_state)  # [B, out_dim]
    psi_curr_3d = psi_curr.unsqueeze(-1)  # [B, out_dim, 1]
    new_trace_v_3d = new_trace_v.unsqueeze(1)  # [B, 1, in_dim]

    # 4. Instantaneous eligibility trace e_ij(t)
    e_trace = psi_curr_3d * (new_trace_v_3d - cell.gamma * new_trace_a)  # [B, out_dim, in_dim]

    new_traces = ALIFEpropTraces(
        trace_v=new_trace_v,
        trace_a=new_trace_a,
        psi_prev=psi_curr,
    )
    return e_trace, new_traces


def compute_eprop_weight_update(
    learning_signal: torch.Tensor,
    eligibility_trace: torch.Tensor,
) -> torch.Tensor:
    """
    Computes synaptic parameter update:
        Delta W_ij = (1 / B) * sum_{b=1}^B L_i,b(t) * e_ij,b(t)

    Args:
        learning_signal: Post-synaptic error/learning signal L_i(t) [B, out_dim]
        eligibility_trace: Instantaneous eligibility trace e_ij(t) [B, out_dim, in_dim]

    Returns:
        delta_w: Synaptic weight update [out_dim, in_dim]
    """
    B = learning_signal.shape[0]
    # learning_signal: [B, out_dim, 1]
    L_3d = learning_signal.unsqueeze(-1)
    # Product: [B, out_dim, in_dim]
    delta_w_batch = L_3d * eligibility_trace
    delta_w = delta_w_batch.sum(dim=0) / max(1, B)
    return delta_w
