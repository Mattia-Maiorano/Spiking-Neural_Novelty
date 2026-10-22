"""
Layer 1 Multi-Timescale Spiking Memory Hierarchy using ALIF Neurons.
Maintains neural populations with distinct membrane and adaptation time constants.
Imports strictly from Layer 0 (`spwm.contracts`) and Layer 1 (`spwm.models.neurons`).
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn

from spwm.contracts.states import ALIFState, MultiTimescaleState
from spwm.contracts.tensors import assert_tensor_contract
from spwm.models.neurons import ALIFCell


def compute_tier_dims(total_dim: int, num_tiers: int = 2) -> Tuple[int, ...]:
    """Partitions total dimension into balanced tiers (e.g. 50% reactive, 50% deep context)."""
    if total_dim <= 0 or num_tiers <= 0:
        raise ValueError(f"total_dim ({total_dim}) and num_tiers ({num_tiers}) must be > 0")
    base = total_dim // num_tiers
    dims = [base] * num_tiers
    dims[-1] += total_dim - sum(dims)
    return tuple(dims)


class MultiTimescaleMemory(nn.Module):
    """
    Multi-Timescale Spiking Memory Module.
    Partitions latent population into multiple timescale tiers with strict dimension validation.
    """

    def __init__(
        self,
        timescale_dims: Sequence[int] = (128, 128),
        betas: Sequence[float] = (0.90, 0.985),
        beta_mem: float = 0.80,
        v_th0: float = 1.0,
        gamma: float = 0.18,
        surrogate_name: str = "atan",
        surrogate_alpha: float = 2.0,
    ) -> None:
        super().__init__()
        self.timescale_dims = list(timescale_dims)
        self.num_timescales = len(self.timescale_dims)
        if self.num_timescales == 0 or any(d <= 0 for d in self.timescale_dims):
            raise ValueError(f"timescale_dims must be non-empty positive integers, got {self.timescale_dims}")

        if len(betas) != self.num_timescales:
            raise ValueError(
                f"betas length ({len(betas)}) does not match num_timescales ({self.num_timescales})"
            )

        self.total_dim = sum(self.timescale_dims)
        self.pool_betas_adapt = [float(b) for b in betas]

        self.cells = nn.ModuleList([
            ALIFCell(
                size=dim,
                beta_mem=beta_mem,
                beta_adapt=b_adapt,
                v_th0=v_th0,
                gamma=gamma,
                surrogate_name=surrogate_name,
                surrogate_alpha=surrogate_alpha,
            )
            for dim, b_adapt in zip(self.timescale_dims, self.pool_betas_adapt)
        ])

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> MultiTimescaleState:
        states = [cell.init_state(batch_size, device=device) for cell in self.cells]
        return MultiTimescaleState(population_states=states)

    def forward(
        self,
        synaptic_inputs: Union[torch.Tensor, Sequence[torch.Tensor]],
        state: Optional[MultiTimescaleState] = None,
    ) -> Tuple[torch.Tensor, MultiTimescaleState]:
        if isinstance(synaptic_inputs, torch.Tensor):
            assert_tensor_contract(synaptic_inputs, (None, self.total_dim), "MultiTimescaleMemory.synaptic_inputs")
            inputs = torch.split(synaptic_inputs, self.timescale_dims, dim=-1)
        else:
            inputs = list(synaptic_inputs)
            if len(inputs) != self.num_timescales:
                raise ValueError(
                    f"Expected {self.num_timescales} input tensors, got {len(inputs)}"
                )

        batch_size = inputs[0].shape[0]
        device = inputs[0].device

        if state is None:
            state = self.init_state(batch_size, device=device)

        new_pop_states: List[ALIFState] = []
        out_spikes: List[torch.Tensor] = []

        for cell, inp, prev_s in zip(self.cells, inputs, state.population_states):
            spk, n_s = cell(inp, state=prev_s)
            new_pop_states.append(n_s)
            out_spikes.append(spk)

        new_state = MultiTimescaleState(population_states=new_pop_states)
        combined_spikes = torch.cat(out_spikes, dim=-1)
        return combined_spikes, new_state
