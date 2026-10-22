"""Compatibility facade for `spwm.core.dynamics` -> `spwm.models.dynamics`."""

from spwm.models.dynamics import DynamicsOutput, DynamicsState, SpikingLatentDynamics

__all__ = ["SpikingLatentDynamics", "DynamicsState", "DynamicsOutput"]
