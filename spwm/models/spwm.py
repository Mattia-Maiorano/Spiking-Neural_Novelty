"""
Compatibility facade for `spwm.models.spwm`.
Canonical implementation lives in `spwm.models.world_model`.
"""

from spwm.models.world_model import (
    SPWM,
    SPWMSequenceOutput,
    SPWMState,
    SPWMStepOutput,
)

__all__ = [
    "SPWM",
    "SPWMState",
    "SPWMStepOutput",
    "SPWMSequenceOutput",
]
