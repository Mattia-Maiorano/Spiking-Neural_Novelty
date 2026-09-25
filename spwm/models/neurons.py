"""
Spiking Neuron Models (LIF and ALIF).
Implements membrane potential dynamics, adaptive thresholds, surrogate gradient thresholding,
and state management for forward-only learning.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Sequence
import torch
import torch.nn as nn
from spwm.models.surrogate import get_surrogate, surrogate_derivative, SurrogateSpike


@dataclass
class NeuronState:
    """State of a standard LIF spiking neuron layer."""
    v_mem: torch.Tensor  # Membrane potential [B, ...]
    spikes: torch.Tensor  # Emitted spikes {0, 1} [B, ...]


@dataclass
class ALIFState:
    """State of an Adaptive Leaky Integrate-and-Fire (ALIF) neuron layer."""
    v_mem: torch.Tensor  # Membrane potential V_t [B, D]
    a_adapt: torch.Tensor  # Adaptive threshold variable A_t [B, D]
    spikes: torch.Tensor  # Emitted spikes z_t [B, D]
    e_trace_v: Optional[torch.Tensor] = None  # Filtered presynaptic input trace on V
    e_trace_a: Optional[torch.Tensor] = None  # Adaptation eligibility trace on A


class LIFCell(nn.Module):
    """
    Standard Leaky Integrate-and-Fire (LIF) neuron cell.
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


class ALIFCell(nn.Module):
    """
    Adaptive Leaky Integrate-and-Fire (ALIF) neuron cell.
    Combines fast membrane potential dynamics with slow threshold adaptation:
        V_t = β_mem * V_(t-1) + I_t - z_(t-1) * V_th0
        A_t = β_adapt * A_(t-1) + z_(t-1)
        V_th,t = V_th0 + γ * A_t
        z_t = Heaviside(V_t - V_th,t)  (with surrogate gradient)
    
    Heterogeneity:
        Supports controlled heterogeneity across population (50% reactive β_adapt=0.90,
        50% deep context memory β_adapt=0.985).
    """

    def __init__(
        self,
        size: int,
        beta_mem: float = 0.80,
        beta_adapt: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
        v_th0: float = 1.0,
        gamma: float = 0.18,
        surrogate_name: str = "atan",
        surrogate_alpha: float = 2.0,
    ) -> None:
        super().__init__()
        self.size = size
        self.v_th0 = float(v_th0)
        self.gamma = float(gamma)
        self.surrogate_name = surrogate_name.lower()
        self.surrogate_alpha = float(surrogate_alpha)
        self.surrogate = get_surrogate(self.surrogate_name, alpha=self.surrogate_alpha)

        # Membrane decay constant (uniform across population)
        self.register_buffer("beta_mem", torch.full((size,), float(beta_mem), dtype=torch.float32))

        # Adaptation decay constants
        if beta_adapt is None:
            # Controlled heterogeneity: 50% reactive (0.90), 50% deep context (0.985)
            n_reactive = size // 2
            n_deep = size - n_reactive
            adapt_vals = [0.90] * n_reactive + [0.985] * n_deep
            self.register_buffer("beta_adapt", torch.tensor(adapt_vals, dtype=torch.float32))
        elif isinstance(beta_adapt, (int, float)):
            self.register_buffer("beta_adapt", torch.full((size,), float(beta_adapt), dtype=torch.float32))
        else:
            adapt_tensor = torch.as_tensor(beta_adapt, dtype=torch.float32)
            if adapt_tensor.numel() == 1:
                adapt_tensor = adapt_tensor.repeat(size)
            elif adapt_tensor.numel() != size:
                raise ValueError(f"beta_adapt size {adapt_tensor.numel()} does not match cell size {size}")
            self.register_buffer("beta_adapt", adapt_tensor)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> ALIFState:
        """Initializes quiescent state for ALIF layer."""
        v_mem = torch.zeros(batch_size, self.size, device=device, dtype=torch.float32)
        a_adapt = torch.zeros(batch_size, self.size, device=device, dtype=torch.float32)
        spikes = torch.zeros(batch_size, self.size, device=device, dtype=torch.float32)
        return ALIFState(v_mem=v_mem, a_adapt=a_adapt, spikes=spikes)

    def forward(
        self,
        synaptic_input: torch.Tensor,
        state: Optional[ALIFState] = None,
    ) -> Tuple[torch.Tensor, ALIFState]:
        """
        Executes single ALIF forward time-step:
            V_t = β_mem * V_(t-1) + I_t - z_(t-1) * V_th0
            A_t = β_adapt * A_(t-1) + z_(t-1)
            V_th,t = V_th0 + γ * A_t
            z_t = Θ(V_t - V_th,t)
        Args:
            synaptic_input: [B, size] input current I_t
            state: previous ALIFState
        Returns:
            (spikes, new_state) where spikes is [B, size]
        """
        B = synaptic_input.shape[0]
        device = synaptic_input.device

        if state is None:
            state = self.init_state(B, device=device)

        # 1. Update membrane potential with reset from previous step
        v_mem = self.beta_mem * state.v_mem + synaptic_input - state.spikes * self.v_th0

        # 2. Update dynamic adaptive threshold variable
        a_adapt = self.beta_adapt * state.a_adapt + state.spikes
        v_th_t = self.v_th0 + self.gamma * a_adapt

        # 3. Emit spikes via surrogate gradient
        spikes = self.surrogate(v_mem - v_th_t)

        new_state = ALIFState(
            v_mem=v_mem,
            a_adapt=a_adapt,
            spikes=spikes,
            e_trace_v=state.e_trace_v,
            e_trace_a=state.e_trace_a,
        )
        return spikes, new_state

    def compute_surrogate_derivative(self, state: ALIFState) -> torch.Tensor:
        """Computes ψ_t = dS/dV for e-prop eligibility calculation."""
        v_th_t = self.v_th0 + self.gamma * state.a_adapt
        return surrogate_derivative(
            state.v_mem - v_th_t,
            surrogate_name=self.surrogate_name,
            alpha=self.surrogate_alpha,
        )
