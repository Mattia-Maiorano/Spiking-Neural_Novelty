"""Compatibility facade for `spwm.core.encoder` -> `spwm.models.encoder`."""

from spwm.models.encoder import EventEncoder, SpatialSoftmax

__all__ = ["EventEncoder", "SpatialSoftmax"]
