"""
Surrogate Gradient Functions for Spiking Neural Networks.
Implements surrogate derivatives for the discontinuous Heaviside step function:
- FastSigmoid
- Atan (Arctangent)
- Sigmoid
"""

from __future__ import annotations
from typing import Callable
import torch
import torch.nn as nn


class FastSigmoidSurrogate(torch.autograd.Function):
    """
    Fast Sigmoid surrogate gradient function.
    Forward: S = (x >= 0).float()
    Backward: dS/dx = 1 / (1 + α * |x|)^2
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
    Backward: dS/dx = (α / 2) / (1 + (π/2 * α * x)^2)
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
    Backward: dS/dx = α * σ(αx) * (1 - σ(αx))
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
    """
    Configurable surrogate spike activation module.
    Computes Heaviside thresholding in forward and surrogate gradient in backward.
    """

    def __init__(self, surrogate_name: str = "atan", alpha: float = 2.0) -> None:
        super().__init__()
        self.surrogate_name = surrogate_name.lower()
        self.alpha = float(alpha)

        if self.surrogate_name == "atan":
            self.fn = AtanSurrogate.apply
        elif self.surrogate_name in ["fast_sigmoid", "fastsigmoid"]:
            self.fn = FastSigmoidSurrogate.apply
        elif self.surrogate_name == "sigmoid":
            self.fn = SigmoidSurrogate.apply
        else:
            raise ValueError(f"Unknown surrogate function: {surrogate_name}. Supported: 'atan', 'fast_sigmoid', 'sigmoid'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x, self.alpha)


def get_surrogate(surrogate_name: str = "atan", alpha: float = 2.0) -> SurrogateSpike:
    """Factory helper to obtain a surrogate activation."""
    return SurrogateSpike(surrogate_name=surrogate_name, alpha=alpha)
