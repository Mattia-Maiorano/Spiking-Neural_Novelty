"""Compatibility facade for `spwm.core.memory` -> `spwm.models.memory`."""

from spwm.models.memory import MultiTimescaleMemory, MultiTimescaleState

__all__ = ["MultiTimescaleMemory", "MultiTimescaleState"]
