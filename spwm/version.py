"""
SPWM Version Solver Facade.
Re-exports the Layer 0 Global Version Solver and provides the CLI runner.
"""

from spwm.contracts.version import (
    GLOBAL_VERSION,
    GLOBAL_VERSION_SOLVER,
    VERSION_SOLVER,
    VersionSolver,
    main,
)

__all__ = [
    "GLOBAL_VERSION",
    "GLOBAL_VERSION_SOLVER",
    "VERSION_SOLVER",
    "VersionSolver",
    "main",
]

if __name__ == "__main__":
    main()

