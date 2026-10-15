import torch
import pytest
from spwm.core.neurons import FastSigmoidSurrogate, AtanSurrogate, SigmoidSurrogate, get_surrogate


@pytest.mark.parametrize("surrogate_name", ["fast_sigmoid", "atan", "sigmoid"])
def test_surrogate_gradients_flow(surrogate_name):
    surrogate = get_surrogate(surrogate_name, alpha=2.0)
    x = torch.tensor([-1.5, -0.5, 0.0, 0.5, 1.5], requires_grad=True)
    spikes = surrogate(x)

    # Check forward values are binary {0.0, 1.0}
    assert torch.all((spikes == 0.0) | (spikes == 1.0))
    assert spikes[0].item() == 0.0
    assert spikes[2].item() == 1.0  # at 0.0, step is 1
    assert spikes[3].item() == 1.0

    # Check gradient flow: non-zero, finite gradients
    loss = (spikes * torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])).sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()
    assert torch.all(x.grad > 0.0)  # surrogate derivative is strictly positive around threshold
