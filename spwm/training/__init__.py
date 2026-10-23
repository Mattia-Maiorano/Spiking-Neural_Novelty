"""
SPWM Training Package (Layer 2 — Training).

Contains losses, metrics, isolated phase strategies (M1, M2, M3),
and the BaseTrainer / Trainer orchestration engine.
"""

from spwm.training.engine import (
    BaseTrainer,
    Trainer,
    load_representation_checkpoint,
    load_m1_checkpoint,
    load_m2_checkpoint,
)
from spwm.training.losses import (
    ConvergenceFunnelLoss,
    PredictorLoss,
    SPWMLoss,
    _frozen,
    compute_convergence_funnel_target,
    extrapolate_latent,
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
    M2PhaseStrategy,
    M3PhaseStrategy,
    PhaseHandler,
    PhaseStrategy,
    PredictorPhaseHandler,
    build_phase_strategy,
)
from spwm.training.rollout import RolloutResult, TelemetryContainer, rollout_latent

__all__ = [
    "BaseTrainer",
    "Trainer",
    "load_representation_checkpoint",
    "load_m1_checkpoint",
    "load_m2_checkpoint",
    "_frozen",
    "PredictorLoss",
    "ConvergenceFunnelLoss",
    "SPWMLoss",
    "extrapolate_latent",
    "compute_convergence_funnel_target",
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
    "M2PhaseStrategy",
    "M3PhaseStrategy",
    "build_phase_strategy",
    "PredictorPhaseHandler",
    "rollout_latent",
    "RolloutResult",
    "TelemetryContainer",
]
