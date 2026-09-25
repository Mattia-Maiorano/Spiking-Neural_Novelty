import torch
import pytest
from spwm.models.memory import MultiTimescaleMemory


def test_fast_decays_faster_than_slow():
    # fast beta = 0.5, slow beta = 0.95
    mem = MultiTimescaleMemory(timescale_dims=(1, 1), betas=(0.5, 0.95), threshold=10.0)
    state = mem.init_state(1)

    # Initial pulse of equal magnitude to both populations
    inp = torch.tensor([[5.0, 5.0]])
    _, state1 = mem(inp, state)

    # Decay over 3 silent steps (input = 0)
    zero_inp = torch.zeros(1, 2)
    s = state1
    for _ in range(3):
        _, s = mem(zero_inp, s)

    fast_v = s.v_mems[0].item()
    slow_v = s.v_mems[1].item()

    # Fast: 5.0 * 0.5^3 = 0.625
    # Slow: 5.0 * 0.95^3 = 4.2868
    assert fast_v < slow_v
    assert pytest.approx(fast_v, rel=1e-3) == 5.0 * (0.5 ** 3)
    assert pytest.approx(slow_v, rel=1e-3) == 5.0 * (0.95 ** 3)


def test_batch_independence():
    mem = MultiTimescaleMemory(timescale_dims=(4, 4), betas=(0.7, 0.98))
    # Batch size 2 with completely different inputs
    inp_batch = torch.randn(2, 8)
    spk_batch, state_batch = mem(inp_batch)

    # Compute item 0 individually
    spk_item0, state_item0 = mem(inp_batch[0:1])
    assert torch.allclose(spk_batch[0:1], spk_item0)
    assert torch.allclose(state_batch.v_mems[0][0:1], state_item0.v_mems[0])
    assert torch.allclose(state_batch.v_mems[1][0:1], state_item0.v_mems[1])
