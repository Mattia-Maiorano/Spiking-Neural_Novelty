"""
Multi-Timescale Spiking Memory.
Maintains neural populations with distinct membrane time constants (τ_fast, τ_slow, ..., τ_N)
to support both high-frequency reactivity and sustained multi-step predictive memory.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Sequence
import torch
import torch.nn as nn
from spwm.models.neurons import LIFCell, NeuronState


@dataclass
class MultiTimescaleState:
    """State of multi-timescale memory populations."""
    population_states: List[NeuronState]

    @property
    def v_mems(self) -> List[torch.Tensor]:
        return [s.v_mem for s in self.population_states]

    @property
    def spikes(self) -> List[torch.Tensor]:
        return [s.spikes for s in self.population_states]

    @property
    def concatenated_spikes(self) -> torch.Tensor:
        """Concatenates spikes across all timescales [B, sum(dim_k)]."""
        return torch.cat(self.spikes, dim=-1)

    @property
    def concatenated_mems(self) -> torch.Tensor:
        """Concatenates membrane potentials across all timescales [B, sum(dim_k)]."""
        return torch.cat(self.v_mems, dim=-1)


class MultiTimescaleMemory(nn.Module):
    """
    Spiking memory module with N timescales.
    Maintains fast and slow LIF populations, capturing rapid event transitions and sustained dynamics.
    """

    def __init__(
        self,
        timescale_dims: Sequence[int] = (64, 64),
        betas: Sequence[float] = (0.8, 0.98),
        threshold: float = 1.0,
        reset_mechanism: str = "hard",
        surrogate_name: str = "atan",
        surrogate_alpha: float = 2.0,
        learnable_betas: bool = False,
    ) -> None:
        super().__init__()
        assert len(timescale_dims) == len(betas), "Mismatch between timescale dimensions and betas."
        self.num_timescales = len(betas)
        self.timescale_dims = list(timescale_dims)
        self.total_dim = sum(self.timescale_dims)

        self.lif_cells = nn.ModuleList([
            LIFCell(
                beta=beta,
                threshold=threshold,
                reset_mechanism=reset_mechanism,
                surrogate_name=surrogate_name,
                surrogate_alpha=surrogate_alpha,
                learnable_beta=learnable_betas,
            )
            for beta in betas
        ])

    @property
    def betas(self) -> List[float]:
        return [cell.beta.item() if isinstance(cell.beta, torch.Tensor) else float(cell.beta) for cell in self.lif_cells]

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> MultiTimescaleState:
        """Initializes quiescent states for all timescale populations."""
        states = [
            cell.init_state(batch_size, dim, device=device)
            for cell, dim in zip(self.lif_cells, self.timescale_dims)
        ]
        return MultiTimescaleState(population_states=states)

    def forward(
        self,
        synaptic_inputs: Sequence[torch.Tensor],
        state: Optional[MultiTimescaleState] = None,
    ) -> Tuple[torch.Tensor, MultiTimescaleState]:
        """
        Updates each timescale population with its respective synaptic input.
        synaptic_inputs: list of tensors [B, dim_k] or single tensor [B, total_dim].
        Returns:
            fused_representation: combined latent activity [B, total_dim]
            new_state: updated MultiTimescaleState
        """
        if isinstance(synaptic_inputs, torch.Tensor):
            # Split tensor into chunks corresponding to each timescale dimension
            inputs = torch.split(synaptic_inputs, self.timescale_dims, dim=-1)
        else:
            inputs = list(synaptic_inputs)

        batch_size = inputs[0].shape[0]
        device = inputs[0].device

        if state is None:
            state = self.init_state(batch_size, device=device)

        new_pop_states: List[NeuronState] = []
        out_spikes: List[torch.Tensor] = []

        for cell, inp, prev_s in zip(self.lif_cells, inputs, state.population_states):
            spk, n_s = cell(inp, prev_s)
            new_pop_states.append(n_s)
            out_spikes.append(spk)

        new_state = MultiTimescaleState(population_states=new_pop_states)
        combined_spikes = torch.cat(out_spikes, dim=-1)
        return combined_spikes, new_state
