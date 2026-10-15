"""
Spiking Neuron Primitives and Surrogate Gradients for SPWM.
Single source of truth for:
- Surrogate gradient functions (Atan, FastSigmoid, Sigmoid)
- Standard LIF and Adaptive Leaky Integrate-and-Fire (ALIF) cells
- State containers and analytical derivative functions for e-prop.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn


class FastSigmoidSurrogate(torch.autograd.Function):
    """
    Fast Sigmoid surrogate gradient function.
    Forward: S = (x >= 0).float()
    Backward: dS/dx = 1 / (1 + alpha * |x|)^2
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0.0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output / ((1.0 + alpha * torch.abs(x)) ** 2)
        return grad_input, None


class AtanSurrogate(torch.autograd.Function):
    """
    Arctangent (Atan) surrogate gradient function.
    Forward: S = (x >= 0).float()
    Backward: dS/dx = (alpha / 2) / (1 + (pi/2 * alpha * x)^2)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0.0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        denom = 1.0 + ((torch.pi / 2.0) * alpha * x) ** 2
        grad_input = grad_output * (alpha / 2.0) / denom
        return grad_input, None


class SigmoidSurrogate(torch.autograd.Function):
    """
    Sigmoid surrogate gradient function.
    Forward: S = (x >= 0).float()
    Backward: dS/dx = alpha * sigmoid(alpha*x) * (1 - sigmoid(alpha*x))
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0.0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        sg = torch.sigmoid(alpha * x)
        grad_input = grad_output * alpha * sg * (1.0 - sg)
        return grad_input, None


class SurrogateSpike(nn.Module):
    """Configurable surrogate spike activation module."""

    def __init__(self, surrogate_name: str = "atan", alpha: float = 2.0) -> None:
        super().__init__()
        self.surrogate_name = surrogate_name.lower()
        self.alpha = float(alpha)

        if self.surrogate_name == "atan":
            self.fn = AtanSurrogate.apply
        elif self.surrogate_name in ("fast_sigmoid", "fastsigmoid"):
            self.fn = FastSigmoidSurrogate.apply
        elif self.surrogate_name == "sigmoid":
            self.fn = SigmoidSurrogate.apply
        else:
            raise ValueError(f"Unknown surrogate function: '{surrogate_name}'. Supported: 'atan', 'fast_sigmoid', 'sigmoid'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x, self.alpha)


def surrogate_derivative(x: torch.Tensor, surrogate_name: str = "atan", alpha: float = 2.0) -> torch.Tensor:
    """Analytical surrogate derivative psi(x) = dS/dV where x = V_mem - V_th."""
    name = surrogate_name.lower()
    if name == "atan":
        denom = 1.0 + ((torch.pi / 2.0) * alpha * x) ** 2
        return (alpha / 2.0) / denom
    elif name in ("fast_sigmoid", "fastsigmoid"):
        return 1.0 / ((1.0 + alpha * torch.abs(x)) ** 2)
    elif name == "sigmoid":
        sg = torch.sigmoid(alpha * x)
        return alpha * sg * (1.0 - sg)
    else:
        raise ValueError(f"Unknown surrogate function: '{surrogate_name}'. Supported: 'atan', 'fast_sigmoid', 'sigmoid'")


def get_surrogate(surrogate_name: str = "atan", alpha: float = 2.0) -> SurrogateSpike:
    return SurrogateSpike(surrogate_name=surrogate_name, alpha=alpha)


@dataclass
class NeuronState:
    """State of a standard LIF spiking neuron layer."""
    v_mem: torch.Tensor
    spikes: torch.Tensor

    def detach(self) -> "NeuronState":
        return NeuronState(
            v_mem=self.v_mem.detach(),
            spikes=self.spikes.detach(),
        )


@dataclass
class ALIFState:
    """State of an Adaptive Leaky Integrate-and-Fire (ALIF) neuron layer."""
    v_mem: torch.Tensor
    a_adapt: torch.Tensor
    spikes: torch.Tensor
    e_trace_v: Optional[torch.Tensor] = None
    e_trace_a: Optional[torch.Tensor] = None

    def detach(self) -> "ALIFState":
        return ALIFState(
            v_mem=self.v_mem.detach(),
            a_adapt=self.a_adapt.detach(),
            spikes=self.spikes.detach(),
            e_trace_v=self.e_trace_v.detach() if self.e_trace_v is not None else None,
            e_trace_a=self.e_trace_a.detach() if self.e_trace_a is not None else None,
        )


class ALIFCell(nn.Module):
    """
    Adaptive Leaky Integrate-and-Fire (ALIF) neuron cell.
    Combines fast membrane potential dynamics with slow threshold adaptation:
        V_t = beta_mem * V_(t-1) + I_t - z_(t-1) * V_th0
        A_t = beta_adapt * A_(t-1) + z_(t-1)
        V_th,t = V_th0 + gamma * A_t
        z_t = Heaviside(V_t - V_th,t)  (with surrogate gradient)
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

        self.register_buffer("beta_mem", torch.full((size,), float(beta_mem), dtype=torch.float32))

        if beta_adapt is None:
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
        v_mem = torch.zeros(batch_size, self.size, device=device, dtype=torch.float32)
        a_adapt = torch.zeros(batch_size, self.size, device=device, dtype=torch.float32)
        spikes = torch.zeros(batch_size, self.size, device=device, dtype=torch.float32)
        return ALIFState(v_mem=v_mem, a_adapt=a_adapt, spikes=spikes)

    def forward(
        self,
        synaptic_input: torch.Tensor,
        state: Optional[ALIFState] = None,
    ) -> Tuple[torch.Tensor, ALIFState]:
        B = synaptic_input.shape[0]
        device = synaptic_input.device

        if state is None:
            state = self.init_state(B, device=device)

        v_mem = self.beta_mem * state.v_mem + synaptic_input - state.spikes * self.v_th0
        a_adapt = self.beta_adapt * state.a_adapt + state.spikes
        v_th_t = self.v_th0 + self.gamma * a_adapt
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
        v_th_t = self.v_th0 + self.gamma * state.a_adapt
        return surrogate_derivative(
            state.v_mem - v_th_t,
            surrogate_name=self.surrogate_name,
            alpha=self.surrogate_alpha,
        )
