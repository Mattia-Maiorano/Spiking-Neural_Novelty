"""Compatibility facade for `spwm.learning.engine` -> `spwm.training.engine`."""

from spwm.training.engine import (
    BaseTrainer,
    enforce_phase_prerequisites,
    resolve_version_checkpoint,
)

__all__ = [
    "BaseTrainer",
    "resolve_version_checkpoint",
    "enforce_phase_prerequisites",
]
