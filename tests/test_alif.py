import torch
import pytest
from spwm.models.neurons import ALIFCell, ALIFState
from spwm.learning.eprop import (
    ALIFEpropTraces,
    init_eprop_traces,
    step_alif_eprop_traces,
    compute_eprop_weight_update,
)


def test_alif_cell_membrane_adaptation_and_reset():
    """Verify ALIF membrane integration, threshold adaptation on spike, and post-spike reset."""
    cell = ALIFCell(size=2, beta_mem=0.80, beta_adapt=0.90, v_th0=1.0, gamma=0.20)
    state = cell.init_state(batch_size=1)

    # Step 1: Sub-threshold input (0.5) -> V_1 = 0.5 < 1.0 -> no spike, A_1 = 0, V_th = 1.0
    inp1 = torch.tensor([[0.5, 0.5]])
    spk1, state1 = cell(inp1, state=state)
    assert torch.equal(spk1, torch.zeros(1, 2))
    assert torch.allclose(state1.v_mem, torch.tensor([[0.5, 0.5]]))
    assert torch.allclose(state1.a_adapt, torch.tensor([[0.0, 0.0]]))

    # Step 2: Strong input (0.8) -> V_2 = 0.8 * 0.5 + 0.8 = 1.2 >= 1.0 -> spike emitted
    inp2 = torch.tensor([[0.8, 0.8]])
    spk2, state2 = cell(inp2, state=state1)
    assert torch.equal(spk2, torch.ones(1, 2))
    assert torch.allclose(state2.v_mem, torch.tensor([[1.2, 1.2]]))
    assert torch.allclose(state2.a_adapt, torch.tensor([[0.0, 0.0]]))  # A_t updates from previous spikes z_(t-1)

    # Step 3: Zero input -> V_3 = 0.8 * 1.2 + 0.0 - 1.0 * 1.0 = -0.04
    # A_3 = 0.9 * 0.0 + 1.0 = 1.0 -> V_th,3 = 1.0 + 0.2 * 1.0 = 1.2
    inp3 = torch.zeros(1, 2)
    spk3, state3 = cell(inp3, state=state2)
    assert torch.equal(spk3, torch.zeros(1, 2))
    assert pytest.approx(state3.v_mem[0, 0].item(), rel=1e-3) == -0.04
    assert pytest.approx(state3.a_adapt[0, 0].item(), rel=1e-3) == 1.0


def test_alif_controlled_heterogeneity():
    """Verify 50% reactive (beta_adapt=0.90) and 50% deep context (beta_adapt=0.985) decay dynamics."""
    cell = ALIFCell(size=4, beta_mem=0.80, v_th0=1.0, gamma=0.20)
    # Default should partition 4 neurons into 2 reactive (0.90) and 2 deep (0.985)
    assert torch.allclose(cell.beta_adapt[:2], torch.tensor([0.90, 0.90]))
    assert torch.allclose(cell.beta_adapt[2:], torch.tensor([0.985, 0.985]))

    # Inject spike into all 4 neurons
    state = cell.init_state(batch_size=1)
    # Force spike at t=0
    state = ALIFState(
        v_mem=torch.zeros(1, 4),
        a_adapt=torch.zeros(1, 4),
        spikes=torch.ones(1, 4),
    )

    # Let adaptation decay over 10 silent steps
    for _ in range(10):
        _, state = cell(torch.zeros(1, 4), state=state)

    a_reactive = state.a_adapt[0, 0].item()
    a_deep = state.a_adapt[0, 3].item()

    # Reactive adaptation decays much faster than deep context adaptation
    assert a_reactive < a_deep
    assert pytest.approx(a_reactive, rel=1e-2) == (0.90 ** 9)
    assert pytest.approx(a_deep, rel=1e-2) == (0.985 ** 9)


def test_alif_threshold_stability_under_sustained_input():
    """Verify that dynamic threshold homeostasis prevents runaway firing under sustained DC input."""
    cell = ALIFCell(size=2, beta_mem=0.80, beta_adapt=0.90, v_th0=1.0, gamma=0.5)
    state = cell.init_state(batch_size=1)

    # Sustained input
    sustained_input = torch.tensor([[1.2, 1.2]])
    spikes_collected = []

    for _ in range(50):
        spk, state = cell(sustained_input, state=state)
        spikes_collected.append(spk)

    spk_tensor = torch.stack(spikes_collected, dim=1)  # [1, 50, 2]
    mean_rate = spk_tensor.mean().item()

    # Rate should adapt to a sparse / bounded firing rate due to threshold increase, not saturated at 100%
    assert 0.0 < mean_rate < 0.8
    # Adaptive threshold variable A should be strictly positive and bounded
    assert torch.all(state.a_adapt > 0.0)
    assert torch.all(state.a_adapt < 10.0)


def test_alif_eprop_dual_eligibility_traces():
    """Verify that e-prop dual traces (trace_v and trace_a) update and produce valid weight updates."""
    cell = ALIFCell(size=2, beta_mem=0.80, beta_adapt=0.90, v_th0=1.0, gamma=0.18)
    in_dim = 3
    B = 2

    state = cell.init_state(batch_size=B)
    traces = init_eprop_traces(batch_size=B, out_dim=cell.size, in_dim=in_dim)

    x_pre = torch.rand(B, in_dim)
    syn_input = torch.rand(B, cell.size)

    # Step forward
    spk, new_state = cell(syn_input, state=state)
    e_trace, new_traces = step_alif_eprop_traces(cell, new_state, x_pre, traces=traces)

    assert e_trace.shape == (B, cell.size, in_dim)
    assert not torch.isnan(e_trace).any()
    assert not torch.isinf(e_trace).any()

    # Compute weight update with learning signal L
    learning_signal = torch.randn(B, cell.size)
    delta_w = compute_eprop_weight_update(learning_signal, e_trace)

    assert delta_w.shape == (cell.size, in_dim)
    assert not torch.isnan(delta_w).any()
    assert not torch.isinf(delta_w).any()
