"""
SPWM Training Package (Layer 2 — Training).

Contains losses, metrics, isolated phase strategies (M1, M2, M3),
and the BaseTrainer / Trainer orchestration engine.
"""

from spwm.training.engine import (
    BaseTrainer,
    ClosedLoopBuffer,
    Trainer,
    compute_dagger_ratio,
    load_representation_checkpoint,
    resolve_closed_loop_buffer,
)
from spwm.training.noise import (
    compute_mppi_noise_alpha,
    sample_ar1_noise,
)
from spwm.training.losses import (
    PredictorLoss,
    SPWMLoss,
    _frozen,
)
from spwm.training.metrics import (
    active_neuron_fraction,
    energy_estimate,
    multi_step_mse,
    one_step_mse,
    parameter_count,
    position_error,
    spike_rate,
    velocity_error,
)
from spwm.training.phases import (
    RepresentationPhaseStrategy,
    M1PhaseStrategy,
    PhaseHandler,
    PhaseStrategy,
    PredictorPhaseHandler,
    build_phase_strategy,
)
from spwm.training.rollout import RolloutResult, TelemetryContainer, rollout_latent

__all__ = [
    "BaseTrainer",
    "ClosedLoopBuffer",
    "Trainer",
    "compute_dagger_ratio",
    "compute_mppi_noise_alpha",
    "sample_ar1_noise",
    "load_representation_checkpoint",
    "resolve_closed_loop_buffer",
    "_frozen",
    "PredictorLoss",
    "SPWMLoss",
    "one_step_mse",
    "multi_step_mse",
    "position_error",
    "velocity_error",
    "spike_rate",
    "active_neuron_fraction",
    "parameter_count",
    "energy_estimate",
    "PhaseStrategy",
    "PhaseHandler",
    "RepresentationPhaseStrategy",
    "M1PhaseStrategy",
    "build_phase_strategy",
    "PredictorPhaseHandler",
    "rollout_latent",
    "RolloutResult",
    "TelemetryContainer",
]
