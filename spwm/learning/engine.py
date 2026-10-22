"""Compatibility facade for `spwm.learning.engine` -> `spwm.training.engine`."""

from spwm.training.engine import (
    BaseTrainer,
    enforce_phase_prerequisites,
    load_m1_checkpoint,
    load_m2_checkpoint,
    resolve_version_checkpoint,
)

__all__ = [
    "BaseTrainer",
    "load_m1_checkpoint",
    "load_m2_checkpoint",
    "resolve_version_checkpoint",
    "enforce_phase_prerequisites",
]
