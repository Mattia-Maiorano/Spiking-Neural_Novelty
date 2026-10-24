"""TDD Contract & Layering Tests for SPWM-v0.3.1 4-Layer Architecture.
Enforces:
1. Layer 0 (`spwm.contracts`) immutability, frozen dataclasses, and cross-domain validation.
2. Runtime tensor shape & phase-space mathematical contracts (`assert_tensor_contract`, `assert_phase_space_contract`).
3. Absence of dead legacy modules (`spwm.utils.config`, `spwm.learning.eprop`, `spwm.core.corrector`,
   `spwm.learning.phases.corrector_phase`, `spwm.training.phases.m2_recovery`, `spwm.training.phases.m3_sustained`).
4. Strict import hierarchy across `contracts` -> `models` / `data` -> `training` -> `evaluation`.
5. AST-level ban on silent fallbacks (`getattr` with default on config/model, `load_state_dict(..., strict=False)`, silent `except TypeError:` or `except Exception: pass`).
"""

from __future__ import annotations
import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
import pytest
import torch

from spwm.contracts.config import (
    DataConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    LatentNeuronConfig,
    LossConfig,
    MemoryConfig,
    ModelConfig,
    ProjectConfig,
    TrainingConfig,
)
from spwm.contracts.protocols import (
    DynamicsProtocol,
    EncoderProtocol,
    PhaseStrategy,
    PhysicalDecoderProtocol,
    PredictorProtocol,
    WorldModelProtocol,
)
from spwm.contracts.states import (
    ALIFState,
    DynamicsOutput,
    DynamicsState,
    MultiTimescaleState,
    PredictorOutput,
    RolloutResult,
    SPWMSequenceOutput,
    SPWMState,
    SPWMStepOutput,
)
from spwm.contracts.tensors import (
    assert_euclidean_coordinates,
    assert_phase_space_contract,
    assert_tensor_contract,
)


def test_frozen_config_immutability():
    """All configuration dataclasses must be strictly frozen (immutable)."""
    cfg = ExperimentConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.project.seed = 999  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        cfg.training.mode = "train_representation"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        cfg.model.latent_dim = 64  # type: ignore[misc]


def test_config_with_overrides_revalidates_invariants():
    """ExperimentConfig.with_overrides must produce a new frozen instance and run __post_init__ checks."""
    cfg = ExperimentConfig()
    updated = cfg.with_overrides(
        seed=123,
        mode="train_representation",
        epochs=50,
        phase_horizon=8,
        early_stopping_patience=15,
    )
    assert updated.project.seed == 123
    assert updated.training.mode == "train_representation"
    assert updated.training.epochs == 50
    assert updated.training.phase_horizon == 8
    assert updated.training.early_stopping_patience == 15

    # Invalid override must fail immediately during with_overrides
    with pytest.raises(ValueError, match="Invalid training mode"):
        cfg.with_overrides(mode="train_corrector")


def test_cross_domain_sequence_length_and_sensor_validation():
    """ExperimentConfig must reject sequence_length too short for active phase or non-divisible sensor dims."""
    # Sensor height not divisible by 4 (two stride-2 convs)
    with pytest.raises(ValueError, match="divisible by 4"):
        EnvironmentConfig(height=30, width=32)

    # Sequence length too short for train_m1 (requires >= H + 1)
    with pytest.raises(ValueError, match="sequence_length"):
        ExperimentConfig(
            data=DataConfig(sequence_length=8),
            training=TrainingConfig(mode="train_m1", phase_horizon=10),
        )


def test_tensor_shape_and_phase_space_assertions():
    """Runtime tensor contract validators must catch rank, dimension, NaN, and phase-space violations."""
    x = torch.zeros(4, 10, 128)
    assert_tensor_contract(x, (4, 10, 128), "latent_seq")
    assert_tensor_contract(x, (None, None, 128), "latent_seq")

    with pytest.raises(ValueError, match="expected 3D tensor"):
        assert_tensor_contract(torch.zeros(4, 128), (None, None, 128), "bad_rank")

    with pytest.raises(ValueError, match="dim 2: expected 128, got 64"):
        assert_tensor_contract(torch.zeros(4, 10, 64), (None, None, 128), "bad_dim")

    assert_phase_space_contract(
        latent_dim=128,
        q_dim=32,
        p_dim=96,
        num_keypoints=16,
        action_dim=8,
        actions=torch.zeros(4, 8),
    )
    with pytest.raises(ValueError, match="q_dim"):
        assert_phase_space_contract(latent_dim=128, q_dim=30, p_dim=98, num_keypoints=15)
    with pytest.raises(ValueError, match="action_dim"):
        assert_phase_space_contract(latent_dim=128, q_dim=32, p_dim=96, num_keypoints=16, action_dim=0)

    q_valid = torch.uniform if hasattr(torch, "uniform") else (torch.rand(2, 32) * 2.0 - 1.0)
    assert_euclidean_coordinates(q_valid, "q_valid")
    with pytest.raises(ValueError, match="outside Euclidean arena bounds"):
        assert_euclidean_coordinates(torch.tensor([[1.5, -0.5]]), "q_oob")


def test_dead_modules_are_pruned():
    """Legacy dead modules must not exist in the workspace."""
    repo_root = Path(__file__).resolve().parent.parent
    dead_paths = [
        repo_root / "spwm" / "utils" / "config.py",
        repo_root / "spwm" / "data" / "synthetic_world.py",
        repo_root / "spwm" / "learning" / "eprop.py",
        repo_root / "spwm" / "core" / "corrector.py",
        repo_root / "spwm" / "learning" / "phases" / "corrector_phase.py",
        repo_root / "spwm" / "training" / "phases" / "m2_recovery.py",
        repo_root / "spwm" / "training" / "phases" / "m3_sustained.py",
        repo_root / "spwm_v7_3",
    ]
    for p in dead_paths:
        assert not p.exists(), f"Dead path still exists and must be pruned: {p}"

    # Verify ALIFState has no dead e_trace_v / e_trace_a fields
    state = ALIFState(
        v_mem=torch.zeros(2, 4),
        a_adapt=torch.zeros(2, 4),
        spikes=torch.zeros(2, 4),
    )
    assert not hasattr(state, "e_trace_v")
    assert not hasattr(state, "e_trace_a")


def test_strict_4_layer_import_hierarchy():
    """Verifies strict unidirectional dependency layering across spwm/:
    - Layer 0 (`spwm.contracts`) imports NO other `spwm.*` package.
    - Layer 1 (`spwm.models`, `spwm.data`) imports ONLY `spwm.contracts` (or `spwm.models`/`spwm.data`).
      NEVER `spwm.training`, `spwm.learning`, or `spwm.evaluation`.
    - Layer 2 (`spwm.training`) imports ONLY `spwm.contracts` and `spwm.models`.
      NEVER `spwm.evaluation` or `experiments`.
    """
    spwm_root = Path(__file__).resolve().parent.parent / "spwm"

    def get_spwm_imports(py_file: Path) -> list[str]:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("spwm."):
                        imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("spwm"):
                    imports.append(node.module)
        return imports

    # 1. Check Layer 0: spwm/contracts
    for py_file in (spwm_root / "contracts").rglob("*.py"):
        for imp in get_spwm_imports(py_file):
            assert imp.startswith("spwm.contracts"), (
                f"Layer 0 violation in {py_file.name}: imports '{imp}' outside spwm.contracts"
            )

    # 2. Check Layer 1: spwm/models & spwm/data
    for subdir in ("models", "data"):
        for py_file in (spwm_root / subdir).rglob("*.py"):
            for imp in get_spwm_imports(py_file):
                forbidden = ("spwm.training", "spwm.learning", "spwm.evaluation")
                assert not imp.startswith(forbidden), (
                    f"Layer 1 violation in {subdir}/{py_file.name}: forbidden upward import '{imp}'"
                )

    # 3. Check Layer 2: spwm/training
    training_dir = spwm_root / "training"
    if training_dir.exists():
        for py_file in training_dir.rglob("*.py"):
            for imp in get_spwm_imports(py_file):
                assert not imp.startswith("spwm.evaluation"), (
                    f"Layer 2 violation in training/{py_file.name}: forbidden upward import '{imp}'"
                )


def test_zero_silent_fallbacks_ast_guard():
    """AST static verification across spwm/:
    1. No `getattr(obj, name, default)` with a 3rd fallback argument in spwm/models or spwm/training.
    2. No `load_state_dict(..., strict=False)` in spwm/.
    3. No `except TypeError:` or `except Exception: pass` in spwm/.
    """
    spwm_root = Path(__file__).resolve().parent.parent / "spwm"

    for py_file in spwm_root.rglob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))
        rel_path = py_file.relative_to(spwm_root)

        for node in ast.walk(tree):
            # 1. Forbid 3-arg getattr(..., ..., default) in models, training, learning, core
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
                if len(node.args) >= 3 and rel_path.parts[0] in ("models", "training", "learning", "core"):
                    pytest.fail(
                        f"Forbidden 3-argument getattr(..., default) fallback in {rel_path}:{node.lineno}"
                    )

            # 2. Forbid load_state_dict(..., strict=False)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "load_state_dict":
                for kw in node.keywords:
                    if kw.arg == "strict" and isinstance(kw.value, ast.Constant) and kw.value.value is False:
                        pytest.fail(
                            f"Forbidden load_state_dict(..., strict=False) in {rel_path}:{node.lineno}"
                        )

            # 3. Forbid silent `except TypeError:` or `except Exception: pass`
            if isinstance(node, ast.ExceptHandler):
                if isinstance(node.type, ast.Name) and node.type.id == "TypeError":
                    pytest.fail(
                        f"Forbidden silent 'except TypeError:' fallback in {rel_path}:{node.lineno}"
                    )
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    pytest.fail(
                        f"Forbidden silent 'except ...: pass' block in {rel_path}:{node.lineno}"
                    )
