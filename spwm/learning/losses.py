"""Compatibility facade for `spwm.learning.losses` -> `spwm.training.losses`."""

from spwm.training.losses import (
    ConvergenceFunnelLoss,
    PredictorLoss,
    SPWMLoss,
    _frozen,
    compute_convergence_funnel_target,
    decompose_latent_loss,
    extrapolate_latent,
)

__all__ = [
    "_frozen",
    "decompose_latent_loss",
    "PredictorLoss",
    "ConvergenceFunnelLoss",
    "SPWMLoss",
    "extrapolate_latent",
    "compute_convergence_funnel_target",
]