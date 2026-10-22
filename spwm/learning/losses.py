"""Compatibility facade for `spwm.learning.losses` -> `spwm.training.losses`."""

from spwm.training.losses import (
    ConvergenceFunnelLoss,
    PredictorLoss,
    SPWMLoss,
    _frozen,
    compute_convergence_funnel_target,
    extrapolate_latent,
)

__all__ = [
    "_frozen",
    "PredictorLoss",
    "ConvergenceFunnelLoss",
    "SPWMLoss",
    "extrapolate_latent",
    "compute_convergence_funnel_target",
]