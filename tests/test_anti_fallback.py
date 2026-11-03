"""TDD Anti-Fallback Verification Tests.

Verifies:
1. Complete absence of silent fallbacks across `spwm/` via AST static inspection.
2. AST Anti-Fallback Scanner correctly spots and catches all forbidden fallback variants:
   - 3-argument `getattr(obj, name, default)`
   - Ternary `hasattr(...)` fallback expressions
   - `load_state_dict(..., strict=False)`
   - Silent exception handlers (`except ...: pass`, `except TypeError:`, `except AttributeError:`)
3. Runtime configuration and contract enforcement: unknown parameters raise `ValueError` immediately.
4. Checkpoints load strictly with `strict=True`.
"""

from __future__ import annotations
import ast
from pathlib import Path
import pytest
import torch

from scripts.scan_fallbacks import FallbackVisitor, scan_directory
from spwm.contracts.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    _strict_dict_to_dataclass,
)


def test_spwm_package_has_zero_fallbacks():
    """Entire spwm/ package must be completely free of fallbacks according to AST scanner."""
    repo_root = Path(__file__).resolve().parent.parent
    spwm_dir = repo_root / "spwm"
    violations = scan_directory(spwm_dir)
    if violations:
        msg = "\n".join(f"{v.file_path.name}:{v.line} [{v.rule}] {v.message}" for v in violations)
        pytest.fail(f"AST Anti-Fallback violations detected in spwm/:\n{msg}")


@pytest.mark.parametrize(
    "code_snippet,expected_rule",
    [
        ("val = getattr(model, 'encoder_dim', 64)", "NO_3ARG_GETATTR"),
        ("val = getattr(obj, attr, None)", "NO_3ARG_GETATTR"),
        ("x = model.dim if hasattr(model, 'dim') else 128", "NO_HASATTR_FALLBACK"),
        ("model.load_state_dict(ckpt, strict=False)", "NO_STRICT_FALSE_LOAD"),
        ("try:\n    do_work()\nexcept TypeError:\n    val = fallback", "NO_SILENT_EXCEPT_TYPEERROR"),
        ("try:\n    do_work()\nexcept AttributeError:\n    val = fallback", "NO_SILENT_EXCEPT_ATTRIBUTEERROR"),
        ("try:\n    do_work()\nexcept Exception:\n    pass", "NO_EXCEPT_PASS"),
    ],
)
def test_ast_scanner_catches_all_fallback_variants(code_snippet: str, expected_rule: str):
    """The AST scanner must detect every prohibited fallback variant."""
    tree = ast.parse(code_snippet)
    visitor = FallbackVisitor(file_path=Path("synthetic_test.py"), source_lines=code_snippet.splitlines())
    visitor.visit(tree)
    rules_detected = [v.rule for v in visitor.violations]
    assert expected_rule in rules_detected, (
        f"Expected rule '{expected_rule}' not detected for code snippet:\n{code_snippet}\n"
        f"Detected rules: {rules_detected}"
    )


def test_config_rejects_unknown_keys_without_fallback():
    """Configurations must reject unexpected dictionary keys loudly rather than falling back or ignoring."""
    bad_dict = {
        "project": {"name": "test_exp", "seed": 42},
        "model": {"latent_dim": 128, "unknown_fallback_param": 999},
    }
    with pytest.raises(ValueError, match="Unknown configuration parameter"):
        ExperimentConfig.from_dict(bad_dict)


def test_strict_dataclass_parser_rejects_unknown_fields():
    """_strict_dict_to_dataclass must raise ValueError on unknown fields without fallback."""
    with pytest.raises(ValueError, match="Unknown configuration parameter"):
        _strict_dict_to_dataclass(ModelConfig, {"latent_dim": 128, "non_existent_field": True})


def test_checkpoint_strict_loading_invariants():
    """Models must refuse to load mismatched state dicts and never silently fallback."""
    from spwm.models.world_model import SPWM

    model = SPWM(latent_dim=32, q_dim=8, p_dim=24, timescale_dims=(16, 16), phase_horizon=4)
    bad_state_dict = {"non_existent_layer.weight": torch.zeros(10, 10)}

    # Must raise RuntimeError on key mismatch (strict=True contract)
    with pytest.raises(RuntimeError):
        model.load_state_dict(bad_state_dict, strict=True)

