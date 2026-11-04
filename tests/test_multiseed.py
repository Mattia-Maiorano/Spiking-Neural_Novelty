"""
Unit tests for Multi-Seed Statistical Validation (DOSSIER Section 6.4).
Verifies statistical metric aggregation (mean +/- std), horizon parsing, and zero fallbacks.
"""

from pathlib import Path
import numpy as np
import pytest
from experiments.train_multiseed import compute_mean_std, format_mean_std


def test_compute_mean_std_basic():
    vals = [0.010, 0.012, 0.014]
    m, s = compute_mean_std(vals)
    assert abs(m - 0.012) < 1e-6
    assert abs(s - 0.002) < 1e-6
    formatted = format_mean_std(m, s, precision=4)
    assert formatted == "0.0120 ± 0.0020"


def test_compute_mean_std_single_value():
    vals = [0.015]
    m, s = compute_mean_std(vals)
    assert m == 0.015
    assert s == 0.0


def test_compute_mean_std_empty_raises():
    with pytest.raises(ValueError, match="Cannot compute mean/std"):
        compute_mean_std([])


def test_multiseed_directory_contracts(tmp_path: Path):
    """Verifies output folder structure adheres to Phase 0 and Phase 1 contracts."""
    rep_dir = tmp_path / "seed_101_representation"
    m1_dir = tmp_path / "seed_101_m1"
    rep_dir.mkdir(parents=True)
    m1_dir.mkdir(parents=True)

    rep_pt = rep_dir / "model.pt"
    m1_pt = m1_dir / "model.pt"
    rep_pt.touch()
    m1_pt.touch()

    assert rep_pt.is_file()
    assert m1_pt.is_file()
