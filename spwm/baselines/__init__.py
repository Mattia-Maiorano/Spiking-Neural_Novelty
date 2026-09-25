"""
Baselines for comparative scientific evaluation against SPWM-v1.
"""

from spwm.baselines.mlp_dynamics import MLPDynamicsWorldModel, BaselineOutput
from spwm.baselines.gru_world_model import GRUWorldModel
from spwm.baselines.recurrent_snn import VanillaRecurrentSNN

__all__ = [
    "MLPDynamicsWorldModel",
    "BaselineOutput",
    "GRUWorldModel",
    "VanillaRecurrentSNN",
]
