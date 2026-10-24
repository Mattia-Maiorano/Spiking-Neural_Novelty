"""Compatibility facade for `spwm.core.predictor` -> `spwm.models.predictors`."""

from spwm.models.predictors import (
    LatentPredictor,
    PhysicalDecoder,
    PredictorOutput,
)

__all__ = [
    "LatentPredictor",
    "PhysicalDecoder",
    "PredictorOutput",
]
