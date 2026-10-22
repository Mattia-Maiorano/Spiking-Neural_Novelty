"""Compatibility facade for `spwm.learning.diagnostics` -> `spwm.training.diagnostics`."""

from spwm.training.diagnostics import collect_spike_statistics, inspect_gradients

__all__ = ["inspect_gradients", "collect_spike_statistics"]
