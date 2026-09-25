"""
Evaluation and plotting utilities for SPWM.
"""

from spwm.evaluation.rollout import RolloutEvaluator, RolloutEvaluationResult
from spwm.evaluation.plots import (
    plot_architecture_diagram,
    plot_training_curves,
    plot_multi_step_degradation,
    plot_spike_raster,
    plot_fast_vs_slow_memory,
    plot_baseline_comparison,
    plot_ablation_comparison,
)

__all__ = [
    "RolloutEvaluator",
    "RolloutEvaluationResult",
    "plot_architecture_diagram",
    "plot_training_curves",
    "plot_multi_step_degradation",
    "plot_spike_raster",
    "plot_fast_vs_slow_memory",
    "plot_baseline_comparison",
    "plot_ablation_comparison",
]
