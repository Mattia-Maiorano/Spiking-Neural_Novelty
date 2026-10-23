"""
Contract test ensuring that every configuration parameter defined in the
contracts/config.py dataclasses is actively consumed/referenced in the repository codebase.
Detects dead configuration options, deprecated hyperparameters, and orphaned config fields.
"""

from __future__ import annotations
import ast
import dataclasses
from pathlib import Path
from typing import Dict, Set, Type
import pytest

from spwm.contracts.config import (
    ProjectConfig,
    EnvironmentConfig,
    DataConfig,
    ModelConfig,
    LatentNeuronConfig,
    MemoryConfig,
    TrainingConfig,
    LossConfig,
    EvaluationConfig,
    ExperimentConfig,
)


CONFIG_SECTIONS: Dict[str, Type] = {
    "project": ProjectConfig,
    "environment": EnvironmentConfig,
    "data": DataConfig,
    "model": ModelConfig,
    "neuron": LatentNeuronConfig,
    "memory": MemoryConfig,
    "training": TrainingConfig,
    "loss": LossConfig,
    "evaluation": EvaluationConfig,
}


def _gather_codebase_ast_identifiers(repo_root: Path) -> Set[str]:
    """Parse all implementation Python files (excluding config definition and tests)

    and collect all accessed attribute names, keyword argument names, and name identifiers.
    """
    py_files = [
        p
        for p in repo_root.rglob("*.py")
        if ".venv" not in p.parts
        and ".git" not in p.parts
        and "tests" not in p.parts
        and p.name != "config.py"
    ]

    accessed_symbols: Set[str] = set()

    for file_path in py_files:
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        except Exception as exc:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                accessed_symbols.add(node.attr)
            elif isinstance(node, ast.keyword) and node.arg:
                accessed_symbols.add(node.arg)
            elif isinstance(node, ast.Name):
                accessed_symbols.add(node.id)

    return accessed_symbols


def test_no_unused_config_variables():
    """Verify that every field defined across all contract configuration dataclasses

    is actively referenced/consumed in the repository codebase.
    """
    repo_root = Path(__file__).resolve().parent.parent
    accessed_symbols = _gather_codebase_ast_identifiers(repo_root)

    unused_fields = []

    for section_name, cfg_cls in CONFIG_SECTIONS.items():
        for f in dataclasses.fields(cfg_cls):
            field_name = f.name
            if field_name not in accessed_symbols:
                unused_fields.append(f"{cfg_cls.__name__}.{field_name}")

    assert not unused_fields, (
        f"Found unused/dead configuration parameter(s) defined in spwm.contracts.config: {unused_fields}. "
        f"Either utilize them in the model/training pipeline or prune them from config schemas."
    )
