"""Compatibility facade for `spwm.learning.metrics` -> `spwm.training.metrics`."""

from spwm.training.metrics import (
    active_neuron_fraction,
    energy_estimate,
    multi_step_mse,
    one_step_mse,
    parameter_count,
    position_error,
    spike_rate,
    velocity_error,
)

__all__ = [
    "one_step_mse",
    "multi_step_mse",
    "position_error",
    "velocity_error",
    "spike_rate",
    "active_neuron_fraction",
    "parameter_count",
    "energy_estimate",
]
