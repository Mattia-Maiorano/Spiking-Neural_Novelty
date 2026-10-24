"""Automated dead code and deprecated architecture detection test suite.

Verifies:
1. Complete elimination of Cascaded Temporal Relay, M2, M3, Convergence Funnel,
   and cascade training modes across all implementation files in spwm/ and experiments/.
2. Physical deletion of obsolete modules from disk.
3. Runtime model and config schemas are clean of phantom/dead attributes.
"""

from __future__ import annotations

import ast
from pathlib import Path
import pytest
import torch

from spwm.contracts.config import ExperimentConfig, TrainingConfig
from spwm.models.world_model import SPWM

DEAD_SYMBOLS = {
    "CascadedTemporalRelay",
    "ConvergenceFunnelLoss",
    "compute_convergence_funnel_target",
    "compute_convergence_funnel_loss",
    "extrapolate_latent",
    "M3PhaseStrategy",
    "load_m2_checkpoint",
    "RelayPredictorProtocol",
}

DEAD_STRING_LITERALS = {
    "train_m3",
    "train_cascade",
    "max_rollout_horizon",
    "m2_checkpoint",
    "m2_num_phases",
}

DEAD_ATTRIBUTES = {
    "relay",
    "m3",
}

OBSOLETE_FILE_PATHS = [
    Path("spwm/training/phases/m3_sustained.py"),
    Path("spwm/core/corrector.py"),
    Path("spwm/learning/eprop.py"),
    Path("spwm/learning/phases/corrector_phase.py"),
]


def test_obsolete_files_physically_deleted():
    """Ensure obsolete phase and architecture implementation files do not exist on disk."""
    repo_root = Path(__file__).resolve().parent.parent
    for rel_path in OBSOLETE_FILE_PATHS:
        full_path = repo_root / rel_path
        assert not full_path.exists(), f"Obsolete dead code file still exists on disk: {rel_path}"


def test_no_dead_symbols_in_codebase():
    """Scan AST across all production Python code in spwm/ and experiments/ for dead symbols and attributes."""
    repo_root = Path(__file__).resolve().parent.parent
    target_dirs = [repo_root / "spwm", repo_root / "experiments"]

    violations: list[str] = []

    for target_dir in target_dirs:
        for py_file in target_dir.rglob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content, filename=str(py_file))
            except Exception as e:
                violations.append(f"Failed to parse {py_file}: {e}")
                continue

            rel_file = py_file.relative_to(repo_root)

            for node in ast.walk(tree):
                # Check symbol names (imports, functions, classes, variables)
                if isinstance(node, ast.Name) and node.id in DEAD_SYMBOLS:
                    violations.append(f"{rel_file}:{node.lineno} references dead symbol '{node.id}'")
                elif isinstance(node, ast.alias) and node.name in DEAD_SYMBOLS:
                    violations.append(f"{rel_file}:{node.lineno} imports dead symbol '{node.name}'")

                # Check attribute accesses (e.g., model.m2, model.m3, model.relay)
                if isinstance(node, ast.Attribute) and node.attr in DEAD_ATTRIBUTES:
                    violations.append(f"{rel_file}:{node.lineno} accesses dead attribute '.{node.attr}'")

                # Check string literals (dead modes, dead config options)
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if node.value in DEAD_STRING_LITERALS:
                        violations.append(f"{rel_file}:{node.lineno} contains dead literal '{node.value}'")

    assert not violations, (
        f"Found {len(violations)} dead code violation(s):\n" + "\n".join(violations)
    )


def test_runtime_model_has_no_dead_attributes():
    """Verify runtime SPWM instance contains only predictor and stateless M2 projector without phantom m3/relay."""
    cfg = ExperimentConfig()
    model = SPWM.from_config(cfg)

    # Pruned attributes must not exist
    for attr in ("m3", "relay"):
        assert not hasattr(model, attr), f"SPWM instance still has pruned attribute '{attr}'"

    # New architecture assertions
    assert hasattr(model, "predictor"), "SPWM instance missing 'predictor'"
    assert hasattr(model, "m2"), "SPWM instance missing 'm2' manifold projector"
    assert model.m1 is model.predictor, "SPWM 'm1' must alias 'predictor'"
    assert model.predictor.symplectic is True, "SPWM predictor must be symplectic"


def test_runtime_config_has_no_dead_fields():
    """Verify runtime TrainingConfig and ExperimentConfig contain no dead fields and reject dead modes."""
    cfg = ExperimentConfig()
    assert not hasattr(cfg.training, "max_rollout_horizon")
    assert not hasattr(cfg.training, "m2_checkpoint")

    # Reject dead modes
    for dead_mode in ("train_m3", "train_cascade"):
        with pytest.raises(ValueError, match="Invalid training mode"):
            TrainingConfig(mode=dead_mode)

    # Verify train_m2 is valid
    m2_cfg = TrainingConfig(mode="train_m2")
    assert m2_cfg.mode == "train_m2"
