"""
Unit tests for Neuromorphic Synaptic Operations (SOP) & Dense Energy Profiling.
Verifies exact calculations, non-negative constraints, and absence of fallbacks.
"""

import pytest
from spwm.training.metrics import (
    calculate_conv2d_macs_and_flops,
    calculate_linear_macs_and_flops,
    compute_dense_energy_nj,
    compute_neuromorphic_energy_nj,
    compute_sops,
    energy_estimate,
)


def test_compute_sops_exact():
    # 32 spikes with fan-out 256
    sops = compute_sops(spike_count=32, fan_out=256)
    assert sops == 8192
    assert isinstance(sops, int)

    # 0 spikes -> 0 SOPs
    assert compute_sops(0, 256) == 0


def test_compute_sops_negative_raises():
    with pytest.raises(ValueError, match="must be non-negative"):
        compute_sops(-1, 256)
    with pytest.raises(ValueError, match="must be non-negative"):
        compute_sops(10, -5)


def test_compute_neuromorphic_energy():
    # SOPs = 10,000, 256 neurons, E_SOP = 0.9 pJ, E_leak = 0.1 pJ
    # Total pJ = 10,000 * 0.9 + 256 * 0.1 = 9000 + 25.6 = 9025.6 pJ
    # In nJ = 9.0256 nJ
    energy_nj = compute_neuromorphic_energy_nj(
        sops=10000,
        num_neurons=256,
        energy_per_sop_pj=0.9,
        energy_leak_pj=0.1,
    )
    assert abs(energy_nj - 9.0256) < 1e-5


def test_compute_dense_energy():
    # 1,000,000 FLOPs * 4.0 pJ = 4,000,000 pJ = 4000.0 nJ = 4.0 uJ
    energy_nj = compute_dense_energy_nj(flops=1000000, energy_per_flop_pj=4.0)
    assert abs(energy_nj - 4000.0) < 1e-5


def test_calculate_linear_macs_and_flops():
    macs, flops = calculate_linear_macs_and_flops(in_features=128, out_features=256, has_bias=True)
    assert macs == 128 * 256
    assert flops == 2 * 128 * 256 + 256

    macs_no_bias, flops_no_bias = calculate_linear_macs_and_flops(in_features=128, out_features=256, has_bias=False)
    assert macs_no_bias == 128 * 256
    assert flops_no_bias == 2 * 128 * 256


def test_calculate_conv2d_macs_and_flops():
    macs, flops = calculate_conv2d_macs_and_flops(
        in_channels=4,
        out_channels=16,
        kernel_size=(3, 3),
        out_hw=(32, 32),
        has_bias=True,
    )
    assert macs == 4 * 16 * 9 * 32 * 32
    assert flops == 2 * macs + (16 * 32 * 32)

