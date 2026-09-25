"""
Learning modules: losses, metrics, diagnostics, and trainer.
"""

from spwm.learning.losses import SPWMLoss, LossOutput
from spwm.learning.metrics import (
    one_step_mse,
    multi_step_mse,
    position_error,
    velocity_error,
    spike_rate,
    active_neuron_fraction,
    parameter_count,
    energy_estimate,
)
from spwm.learning.diagnostics import inspect_gradients, collect_spike_statistics
from spwm.learning.trainer import Trainer

__all__ = [
    "SPWMLoss",
    "LossOutput",
    "one_step_mse",
    "multi_step_mse",
    "position_error",
    "velocity_error",
    "spike_rate",
    "active_neuron_fraction",
    "parameter_count",
    "energy_estimate",
    "inspect_gradients",
    "collect_spike_statistics",
    "Trainer",
]
