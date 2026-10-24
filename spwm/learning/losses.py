"""Compatibility facade for `spwm.learning.losses` -> `spwm.training.losses`."""

from spwm.training.losses import (
    PredictorLoss,
    SPWMLoss,
    _frozen,
    decompose_latent_loss,
)

__all__ = [
    "_frozen",
    "decompose_latent_loss",
    "PredictorLoss",
    "SPWMLoss",
]