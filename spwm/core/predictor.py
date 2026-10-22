"""Compatibility facade for `spwm.core.predictor` -> `spwm.models.predictors`."""

from spwm.models.predictors import (
    CascadedTemporalRelay,
    LatentPredictor,
    PhysicalDecoder,
    PredictorOutput,
    compute_convergence_funnel_target,
    extrapolate_latent,
)

__all__ = [
    "LatentPredictor",
    "CascadedTemporalRelay",
    "PhysicalDecoder",
    "PredictorOutput",
    "extrapolate_latent",
    "compute_convergence_funnel_target",
]
