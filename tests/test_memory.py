import torch
import pytest
from spwm.models.memory import MultiTimescaleMemory


def test_reactive_adapts_and_decays_faster_than_deep_context():
    # reactive beta_adapt = 0.90, deep context beta_adapt = 0.985
    mem = MultiTimescaleMemory(
        timescale_dims=(1, 1),
        betas=(0.90, 0.985),
        beta_mem=0.80,
        v_th0=10.0,
    )
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

    # Membrane decays with beta_mem=0.80: 5.0 * 0.8^3 = 2.56
    assert pytest.approx(fast_v, rel=1e-3) == 5.0 * (0.80 ** 3)
    assert pytest.approx(slow_v, rel=1e-3) == 5.0 * (0.80 ** 3)


def test_batch_independence():
    mem = MultiTimescaleMemory(timescale_dims=(4, 4), betas=(0.90, 0.985))
    # Batch size 2 with completely different inputs
    inp_batch = torch.randn(2, 8)
    spk_batch, state_batch = mem(inp_batch)

    # Compute item 0 individually
    spk_item0, state_item0 = mem(inp_batch[0:1])
    assert torch.allclose(spk_batch[0:1], spk_item0)
    assert torch.allclose(state_batch.v_mems[0][0:1], state_item0.v_mems[0])
    assert torch.allclose(state_batch.v_mems[1][0:1], state_item0.v_mems[1])
