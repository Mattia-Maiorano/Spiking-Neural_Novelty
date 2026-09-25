"""
Multi-Timescale Spiking Memory Hierarchy using ALIF Neurons.
Maintains neural populations with distinct membrane and adaptation time constants:
- Reactive Pool (50% of neurons): β_adapt = 0.90 (fast behavioral adjustments)
- Deep Context Memory Pool (50% of neurons): β_adapt = 0.985 (long-horizon temporal integration)
- Uniform membrane decay: β_mem = 0.80 across all neurons.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Sequence, Union
import torch
import torch.nn as nn
from spwm.models.neurons import ALIFCell, ALIFState


@dataclass
class MultiTimescaleState:
    """State container for multi-timescale ALIF memory populations."""
    population_states: List[ALIFState]

    @property
    def v_mems(self) -> List[torch.Tensor]:
        return [s.v_mem for s in self.population_states]

    @property
    def a_adapts(self) -> List[torch.Tensor]:
        return [s.a_adapt for s in self.population_states]

    @property
    def spikes(self) -> List[torch.Tensor]:
        return [s.spikes for s in self.population_states]

    @property
    def concatenated_spikes(self) -> torch.Tensor:
        """Concatenates spikes across all timescale tiers [B, sum(dim_k)]."""
        return torch.cat(self.spikes, dim=-1)

    @property
    def concatenated_mems(self) -> torch.Tensor:
        """Concatenates membrane potentials across all tiers [B, sum(dim_k)]."""
        return torch.cat(self.v_mems, dim=-1)

    @property
    def fast_spikes(self) -> torch.Tensor:
        return self.spikes[0]

    @property
    def slow_spikes(self) -> torch.Tensor:
        return self.spikes[-1]


def compute_tier_dims(total_dim: int, num_tiers: int = 2) -> Tuple[int, ...]:
    """Partitions total dimension into balanced tiers (e.g. 50% reactive, 50% deep context)."""
    if num_tiers == 2:
        fast = total_dim // 2
        slow = total_dim - fast
        return fast, slow
    elif num_tiers == 3:
        fast = total_dim // 3
        mid = total_dim // 3
        slow = total_dim - (fast + mid)
        return fast, mid, slow
    else:
        base = total_dim // num_tiers
        dims = [base] * num_tiers
        dims[-1] += total_dim - sum(dims)
        return tuple(dims)


class MultiTimescaleMemory(nn.Module):
    """
    Multi-Timescale Spiking Memory Module (SPWM-v3).
    Partitions latent population into Reactive (β_adapt=0.90) and Deep Context (β_adapt=0.985) pools.
    """

    DEFAULT_BETAS_ADAPT: Tuple[float, float] = (0.90, 0.985)

    def __init__(
        self,
        timescale_dims: Optional[Sequence[int]] = None,
        total_dim: Optional[int] = None,
        betas: Optional[Sequence[float]] = None,
        beta_mem: float = 0.80,
        v_th0: float = 1.0,
        gamma: float = 0.18,
        surrogate_name: str = "atan",
        surrogate_alpha: float = 2.0,
    ) -> None:
        super().__init__()

        if timescale_dims is not None:
            self.timescale_dims = list(timescale_dims)
        else:
            dim = total_dim or 128
            self.timescale_dims = list(compute_tier_dims(dim, num_tiers=2))

        self.num_timescales = len(self.timescale_dims)
        self.total_dim = sum(self.timescale_dims)

        if betas is not None:
            self.pool_betas_adapt = list(betas)
        else:
            if len(self.timescale_dims) == 2:
                self.pool_betas_adapt = list(self.DEFAULT_BETAS_ADAPT)
            elif len(self.timescale_dims) == 3:
                self.pool_betas_adapt = [0.85, 0.95, 0.99]
            else:
                self.pool_betas_adapt = [0.95] * len(self.timescale_dims)

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
        """Initializes quiescent states across all timescale pools."""
        states = [
            cell.init_state(batch_size, device=device)
            for cell in self.cells
        ]
        return MultiTimescaleState(population_states=states)

    def forward(
        self,
        synaptic_inputs: Union[torch.Tensor, Sequence[torch.Tensor]],
        state: Optional[MultiTimescaleState] = None,
    ) -> Tuple[torch.Tensor, MultiTimescaleState]:
        """
        Updates each timescale pool with its respective somatic input currents.
        """
        if isinstance(synaptic_inputs, torch.Tensor):
            inputs = torch.split(synaptic_inputs, self.timescale_dims, dim=-1)
        else:
            inputs = list(synaptic_inputs)

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
