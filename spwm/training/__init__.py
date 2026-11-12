"""
SPWM Training Package (Layer 2 — Training).

Contains losses, metrics, isolated phase strategies (M1, M2, M3),
and the BaseTrainer / Trainer orchestration engine.
"""

from spwm.training.checkpoints import (
    enforce_phase_prerequisites,
    load_representation_checkpoint,
    resolve_closed_loop_buffer,
    resolve_version_checkpoint,
)
from spwm.training.engine import (
    BaseTrainer,
    Trainer,
)
from spwm.training.replay_buffer import (
    ClosedLoopBuffer,
    EMATracker,
)
from spwm.training.noise import (
    compute_dagger_ratio,
    compute_mppi_noise_alpha,
    sample_ar1_noise,
)
from spwm.training.losses import (
    PredictorLoss,
    SPWMLoss,
    _frozen,
    compute_critic_mc_loss,
    compute_m1_loss,
    compute_representation_loss,
    decompose_latent_loss,
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
from spwm.models.rollout import RolloutResult, TelemetryContainer, rollout_latent

__all__ = [
    "BaseTrainer",
    "ClosedLoopBuffer",
    "EMATracker",
    "Trainer",
    "compute_dagger_ratio",
    "compute_mppi_noise_alpha",
    "sample_ar1_noise",
    "enforce_phase_prerequisites",
    "load_representation_checkpoint",
    "resolve_closed_loop_buffer",
    "resolve_version_checkpoint",
    "_frozen",
    "compute_critic_mc_loss",
    "compute_m1_loss",
    "compute_representation_loss",
    "decompose_latent_loss",
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
