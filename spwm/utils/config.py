"""
Configuration management utilities.
Loads, validates, merges, and serializes YAML experiment configurations.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads a YAML configuration file into a dictionary."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """Saves a dictionary configuration to a YAML file."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges override dictionary into base dictionary."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result
