"""
Spiking Neuron Models (Leaky Integrate-and-Fire - LIF).
Implements membrane potential dynamics, surrogate thresholding, and state management.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
from spwm.models.surrogate import get_surrogate, SurrogateSpike


@dataclass
class NeuronState:
    """State of a spiking neuron layer."""
    v_mem: torch.Tensor  # Membrane potential
    spikes: torch.Tensor  # Emitted spikes {0, 1}


class LIFCell(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) neuron cell.
    Dynamics:
        V_t = β * V_(t-1) + I_t
        S_t = Heaviside(V_t - V_th)  (with surrogate gradient)
        V_t_post = V_t * (1 - S_t)   (hard reset) or V_t - V_th * S_t (soft reset)
    """

    def __init__(
        self,
        beta: float = 0.9,
        threshold: float = 1.0,
        reset_mechanism: str = "hard",
        surrogate_name: str = "atan",
        surrogate_alpha: float = 2.0,
        learnable_beta: bool = False,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.reset_mechanism = reset_mechanism.lower()
        self.surrogate = get_surrogate(surrogate_name, alpha=surrogate_alpha)

        if learnable_beta:
            # Parametrized via inverse sigmoid (logit) to constrain beta in (0, 1)
            init_logit = torch.logit(torch.tensor(beta).clamp(1e-4, 1.0 - 1e-4))
            self.beta_raw = nn.Parameter(init_logit)
        else:
            self.register_buffer("beta_raw", torch.tensor(beta))
        self.learnable_beta = learnable_beta

    @property
    def beta(self) -> torch.Tensor:
        if self.learnable_beta:
            return torch.sigmoid(self.beta_raw)
        return self.beta_raw

    def init_state(self, *shape: int, device: Optional[torch.device] = None) -> NeuronState:
        """Initializes quiescent membrane potentials and zero spikes."""
        v_mem = torch.zeros(*shape, device=device, dtype=torch.float32)
        spikes = torch.zeros(*shape, device=device, dtype=torch.float32)
        return NeuronState(v_mem=v_mem, spikes=spikes)

    def forward(
        self,
        synaptic_input: torch.Tensor,
        state: Optional[NeuronState] = None,
    ) -> Tuple[torch.Tensor, NeuronState]:
        """
        Single-step LIF forward update.
        synaptic_input: [B, ...]
        state: previous NeuronState
        Returns: (spikes, new_state)
        """
        if state is None:
            state = self.init_state(*synaptic_input.shape, device=synaptic_input.device)

        # Membrane integration
        v_decayed = self.beta * state.v_mem
        v_mem = v_decayed + synaptic_input

        # Spike generation via surrogate
        spikes = self.surrogate(v_mem - self.threshold)

        # Reset mechanism
        if self.reset_mechanism == "hard":
            v_post = v_mem * (1.0 - spikes)
        elif self.reset_mechanism == "soft":
            v_post = v_mem - (self.threshold * spikes)
        else:
            raise ValueError(f"Unknown reset mechanism: {self.reset_mechanism}")

        new_state = NeuronState(v_mem=v_post, spikes=spikes)
        return spikes, new_state
