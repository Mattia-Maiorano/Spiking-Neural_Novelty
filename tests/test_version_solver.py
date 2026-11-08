"""TDD Verification Tests for Global Version Solver (spwm.contracts.version).

Verifies contracts, properties, path resolvers, parsers, immutability,
and CLI interface dynamically against the SSOT without fragile hardcoded versions.
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


def test_global_version_contracts():
    """GLOBAL_VERSION must satisfy all SSOT properties and aliases dynamically."""
    assert GLOBAL_VERSION.version >= 0
    assert GLOBAL_VERSION.major == GLOBAL_VERSION.version
    assert GLOBAL_VERSION.minor >= 0
    assert GLOBAL_VERSION.patch >= 0
    assert GLOBAL_VERSION.semver == f"{GLOBAL_VERSION.version}.{GLOBAL_VERSION.minor}.{GLOBAL_VERSION.patch}"
    assert GLOBAL_VERSION.short_tag == f"v{GLOBAL_VERSION.semver}"
    assert GLOBAL_VERSION.tag == f"{GLOBAL_VERSION.prefix}-{GLOBAL_VERSION.short_tag}"
    assert GLOBAL_VERSION.formatted == GLOBAL_VERSION.tag
    assert (
        GLOBAL_VERSION.snake_case
        == f"{GLOBAL_VERSION.prefix.lower()}_v{GLOBAL_VERSION.version}_{GLOBAL_VERSION.minor}_{GLOBAL_VERSION.patch}"
    )
    assert GLOBAL_VERSION.tuple == (GLOBAL_VERSION.version, GLOBAL_VERSION.minor, GLOBAL_VERSION.patch)
    assert GLOBAL_VERSION is VERSION_SOLVER
    assert GLOBAL_VERSION is GLOBAL_VERSION_SOLVER


def test_version_solver_path_resolvers():
    """VersionSolver must resolve all canonical training directories and artifacts dynamically."""
    v = VersionSolver(version=1, minor=2, patch=3, prefix="TEST")
    res_dir = Path("results")

    expected_snake = "test_v1_2_3"
    assert v.m1_dir_name() == f"{expected_snake}_train_m1"
    assert v.m1_dir(res_dir) == res_dir / f"{expected_snake}_train_m1"
    assert v.m1_checkpoint(res_dir) == res_dir / f"{expected_snake}_train_m1" / "model.pt"
    assert v.m1_config(res_dir) == res_dir / f"{expected_snake}_train_m1" / "config.yaml"
    assert v.m1_metrics(res_dir) == res_dir / f"{expected_snake}_train_m1" / f"metrics_{expected_snake}.json"

    assert v.representation_dir_name() == f"{expected_snake}_train_representation"
    assert v.representation_dir(res_dir) == res_dir / f"{expected_snake}_train_representation"
    assert v.representation_checkpoint(res_dir) == res_dir / f"{expected_snake}_train_representation" / "model.pt"
    assert v.representation_config(res_dir) == res_dir / f"{expected_snake}_train_representation" / "config.yaml"


@pytest.mark.parametrize(
    "raw_string,expected_tuple,expected_prefix",
    [
        ("0.7.3", (0, 7, 3), "SPWM"),
        ("v0.7.3", (0, 7, 3), "SPWM"),
        ("SPWM-v0.7.3", (0, 7, 3), "SPWM"),
        ("spwm_v0_7_3", (0, 7, 3), "SPWM"),
        ("spwm_v0_7_3_train_m1", (0, 7, 3), "SPWM"),
        ("spwm_v0_8_3", (0, 8, 3), "SPWM"),
        ("v1.2.4", (1, 2, 4), "SPWM"),
    ],
)
def test_from_string_parsing(raw_string: str, expected_tuple: tuple[int, int, int], expected_prefix: str):
    """from_string must correctly resolve valid string patterns across arbitrary versions."""
    solver = VersionSolver.from_string(raw_string, default_prefix=expected_prefix)
    assert solver.tuple == expected_tuple
    assert solver.prefix == expected_prefix


def test_from_tuple():
    """from_tuple must construct VersionSolver accurately."""
    solver = VersionSolver.from_tuple((2, 4, 6), prefix="SPWM")
    assert solver.version == 2
    assert solver.minor == 4
    assert solver.patch == 6
    assert solver.formatted == "SPWM-v2.4.6"


def test_version_solver_immutability():
    """VersionSolver must be frozen and reject in-place mutations."""
    v = VersionSolver(1, 0, 0)
    with pytest.raises(FrozenInstanceError):
        v.version = 2  # type: ignore[misc]


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
    """CLI execution via python -m spwm.version must dynamically match GLOBAL_VERSION."""
    res_formatted = subprocess.run(
        [sys.executable, "-m", "spwm.version", "--format", "formatted"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert res_formatted.stdout.strip() == GLOBAL_VERSION.formatted

    res_snake = subprocess.run(
        [sys.executable, "-m", "spwm.version", "--format", "snake"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert res_snake.stdout.strip() == GLOBAL_VERSION.snake_case

    res_checkpoint = subprocess.run(
        [sys.executable, "-m", "spwm.version", "--format", "m1-checkpoint"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert res_checkpoint.stdout.strip() == str(GLOBAL_VERSION.m1_checkpoint())
