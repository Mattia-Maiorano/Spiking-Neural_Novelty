"""TDD Verification Tests for Global Version Solver (spwm.contracts.version).

Verifies:
1. VersionSolver dataclass properties (version/major, minor, patch, semver, short_tag, tag, formatted, snake_case, tuple).
2. Canonical training paths resolution (m1_dir, m1_checkpoint, m1_config, m1_metrics, representation_checkpoint, representation_config).
3. String parsing across all supported variants ('0.7.3', 'v0.7.3', 'SPWM-v0.7.3', 'spwm_v0_7_3', 'spwm_v0_7_3_train_m1').
4. Tuple construction and strict validation (immutability, negative values, malformed inputs).
5. CLI runner execution (`python -m spwm.version`).
"""

from __future__ import annotations
from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys
import pytest

from spwm.contracts.version import (
    GLOBAL_VERSION,
    GLOBAL_VERSION_SOLVER,
    VERSION_SOLVER,
    VersionSolver,
)


def test_global_version_defaults():
    """GLOBAL_VERSION must default to v0.7.3 with full SSOT properties."""
    assert GLOBAL_VERSION.version == 0
    assert GLOBAL_VERSION.major == 0
    assert GLOBAL_VERSION.minor == 7
    assert GLOBAL_VERSION.patch == 3
    assert GLOBAL_VERSION.semver == "0.7.3"
    assert GLOBAL_VERSION.short_tag == "v0.7.3"
    assert GLOBAL_VERSION.tag == "SPWM-v0.7.3"
    assert GLOBAL_VERSION.formatted == "SPWM-v0.7.3"
    assert GLOBAL_VERSION.snake_case == "spwm_v0_7_3"
    assert GLOBAL_VERSION.tuple == (0, 7, 3)
    assert GLOBAL_VERSION is VERSION_SOLVER
    assert GLOBAL_VERSION is GLOBAL_VERSION_SOLVER


def test_version_solver_path_resolvers():
    """VersionSolver must resolve all canonical training directories and artifacts."""
    v = VersionSolver(version=0, minor=7, patch=3)
    res_dir = Path("results")

    assert v.m1_dir_name() == "spwm_v0_7_3_train_m1"
    assert v.m1_dir(res_dir) == res_dir / "spwm_v0_7_3_train_m1"
    assert v.m1_checkpoint(res_dir) == res_dir / "spwm_v0_7_3_train_m1" / "model.pt"
    assert v.m1_config(res_dir) == res_dir / "spwm_v0_7_3_train_m1" / "config.yaml"
    assert v.m1_metrics(res_dir) == res_dir / "spwm_v0_7_3_train_m1" / "metrics_spwm_v0_7_3.json"

    assert v.representation_dir_name() == "spwm_v0_7_3_train_representation"
    assert v.representation_dir(res_dir) == res_dir / "spwm_v0_7_3_train_representation"
    assert v.representation_checkpoint(res_dir) == res_dir / "spwm_v0_7_3_train_representation" / "model.pt"
    assert v.representation_config(res_dir) == res_dir / "spwm_v0_7_3_train_representation" / "config.yaml"


@pytest.mark.parametrize(
    "raw_string,expected_tuple,expected_tag",
    [
        ("0.7.3", (0, 7, 3), "SPWM-v0.7.3"),
        ("v0.7.3", (0, 7, 3), "SPWM-v0.7.3"),
        ("SPWM-v0.7.3", (0, 7, 3), "SPWM-v0.7.3"),
        ("spwm_v0_7_3", (0, 7, 3), "SPWM-v0.7.3"),
        ("spwm_v0_7_3_train_m1", (0, 7, 3), "SPWM-v0.7.3"),
        ("spwm_v0_8_0", (0, 8, 0), "SPWM-v0.8.0"),
        ("v1.2.4", (1, 2, 4), "SPWM-v1.2.4"),
    ],
)
def test_from_string_parsing(raw_string: str, expected_tuple: tuple[int, int, int], expected_tag: str):
    """from_string must correctly resolve all valid string patterns."""
    solver = VersionSolver.from_string(raw_string)
    assert solver.tuple == expected_tuple
    assert solver.formatted == expected_tag


def test_from_tuple():
    """from_tuple must construct VersionSolver accurately."""
    solver = VersionSolver.from_tuple((1, 0, 0))
    assert solver.version == 1
    assert solver.minor == 0
    assert solver.patch == 0
    assert solver.formatted == "SPWM-v1.0.0"


def test_version_solver_immutability():
    """VersionSolver must be frozen and reject in-place mutations."""
    v = VersionSolver(0, 7, 3)
    with pytest.raises(FrozenInstanceError):
        v.version = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"version": -1, "minor": 0, "patch": 0},
        {"version": 0, "minor": -1, "patch": 0},
        {"version": 0, "minor": 0, "patch": -1},
        {"version": "0", "minor": 7, "patch": 3},  # type: ignore[dict-item]
        {"prefix": ""},
    ],
)
def test_strict_validation_rejects_invalid_values(bad_kwargs: dict):
    """Validation must strictly reject negative, empty, or non-integer inputs without fallback."""
    with pytest.raises(ValueError):
        VersionSolver(**bad_kwargs)


@pytest.mark.parametrize("bad_str", ["", "invalid", "0.7", "v0.7", "spwm_version", "v-1.0.0"])
def test_from_string_rejects_malformed_strings(bad_str: str):
    """from_string must fail loudly on malformed strings."""
    with pytest.raises(ValueError):
        VersionSolver.from_string(bad_str)


def test_cli_execution():
    """CLI execution via python -m spwm.version must return expected strings."""
    res_formatted = subprocess.run(
        [sys.executable, "-m", "spwm.version", "--format", "formatted"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert res_formatted.stdout.strip() == "SPWM-v0.7.3"

    res_snake = subprocess.run(
        [sys.executable, "-m", "spwm.version", "--format", "snake"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert res_snake.stdout.strip() == "spwm_v0_7_3"

    res_checkpoint = subprocess.run(
        [sys.executable, "-m", "spwm.version", "--format", "m1-checkpoint"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert res_checkpoint.stdout.strip() == "results/spwm_v0_7_3_train_m1/model.pt"

