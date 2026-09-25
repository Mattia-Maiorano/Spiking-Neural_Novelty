import torch
import pytest
from spwm.models.surrogate import FastSigmoidSurrogate, AtanSurrogate, SigmoidSurrogate, get_surrogate
from spwm.models.neurons import LIFCell


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


def test_lif_cell_membrane_update_and_reset():
    lif = LIFCell(beta=0.5, threshold=1.0, reset_mechanism="hard")
    state = lif.init_state(1, 2)

    # Step 1: Sub-threshold input (0.6) -> no spike, v_mem = 0.6
    inp1 = torch.tensor([[0.6, 0.6]])
    spk1, state1 = lif(inp1, state)
    assert torch.equal(spk1, torch.zeros(1, 2))
    assert torch.allclose(state1.v_mem, torch.tensor([[0.6, 0.6]]))

    # Step 2: Next input (0.8) -> v = 0.5 * 0.6 + 0.8 = 1.1 >= 1.0 -> spike emitted, hard reset to 0
    inp2 = torch.tensor([[0.8, 0.8]])
    spk2, state2 = lif(inp2, state1)
    assert torch.equal(spk2, torch.ones(1, 2))
    assert torch.allclose(state2.v_mem, torch.tensor([[0.0, 0.0]]))


def test_lif_soft_reset():
    lif = LIFCell(beta=0.5, threshold=1.0, reset_mechanism="soft")
    state = lif.init_state(1, 1)

    inp = torch.tensor([[1.3]])
    spk, state_post = lif(inp, state)
    assert spk.item() == 1.0
    # Soft reset: V_post = 1.3 - 1.0 = 0.3
    assert torch.isclose(state_post.v_mem, torch.tensor([[0.3]]))
