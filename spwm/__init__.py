"""
SPWM: Spiking Predictive World Model.
A modular, reproducible, neuromorphic recurrent state-space world model.
"""

from spwm.contracts.version import GLOBAL_VERSION, VersionSolver

__version__ = GLOBAL_VERSION.semver

__all__ = [
    "__version__",
    "GLOBAL_VERSION",
    "VersionSolver",
]
