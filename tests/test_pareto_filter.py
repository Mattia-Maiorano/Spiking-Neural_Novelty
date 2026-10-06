import pytest
import numpy as np
from spwm.evaluation.rollout import RolloutEvaluationResult


def test_pareto_gate_and_rank():
    # Model A: Low H=1 error, but high drift (diverges at H=50) -> Fails Gate
    res_diverging = RolloutEvaluationResult(
        horizons=[1, 5, 10, 25, 50],
        latent_mse_per_horizon={1: 0.05, 5: 0.1, 10: 0.2, 25: 0.4, 50: 0.8},
        position_error_per_horizon={1: 0.08, 5: 0.12, 10: 0.20, 25: 0.40, 50: 0.60},
        velocity_error_per_horizon={1: 0.5, 5: 0.5, 10: 0.5, 25: 0.5, 50: 0.5},
        teacher_forcing_mse=0.05,
        mean_spike_rate=0.15,
    )

    # Model B: Slightly higher H=1 error, but stable drift (H50/H1 <= 5.0) -> Passes Gate
    res_stable = RolloutEvaluationResult(
        horizons=[1, 5, 10, 25, 50],
        latent_mse_per_horizon={1: 0.10, 5: 0.12, 10: 0.15, 25: 0.25, 50: 0.40},
        position_error_per_horizon={1: 0.10, 5: 0.13, 10: 0.17, 25: 0.28, 50: 0.45},
        velocity_error_per_horizon={1: 0.5, 5: 0.5, 10: 0.5, 25: 0.5, 50: 0.5},
        teacher_forcing_mse=0.10,
        mean_spike_rate=0.13,
    )

    # Model A: drift ratio = 0.60 / 0.08 = 7.5x (> 5.0x) -> FAIL
    assert res_diverging.drift_ratio == pytest.approx(7.5, rel=1e-3)
    assert not res_diverging.is_gate_passed(max_drift_ratio=5.0)

    # Model B: drift ratio = 0.45 / 0.10 = 4.5x (<= 5.0x) -> PASS
    assert res_stable.drift_ratio == pytest.approx(4.5, rel=1e-3)
    assert res_stable.is_gate_passed(max_drift_ratio=5.0)

    # Rollout MAE
    assert res_stable.rollout_mae == pytest.approx(np.mean([0.10, 0.13, 0.17, 0.28, 0.45]), rel=1e-3)
    assert res_diverging.rollout_mae == pytest.approx(np.mean([0.08, 0.12, 0.20, 0.40, 0.60]), rel=1e-3)


def test_dynamic_score_weighting():
    # Model 1: Low H1 (0.05), but high drift ratio (7.5x)
    res_diverging = RolloutEvaluationResult(
        horizons=[1, 50],
        latent_mse_per_horizon={1: 0.05, 50: 0.8},
        position_error_per_horizon={1: 0.05, 50: 0.375},  # drift = 7.5
        velocity_error_per_horizon={1: 0.5, 50: 0.5},
        teacher_forcing_mse=0.05,
        mean_spike_rate=0.15,
    )

    # Model 2: Balanced H1 (0.10) and low drift ratio (1.5x)
    res_balanced = RolloutEvaluationResult(
        horizons=[1, 50],
        latent_mse_per_horizon={1: 0.10, 50: 0.20},
        position_error_per_horizon={1: 0.10, 50: 0.15},  # drift = 1.5
        velocity_error_per_horizon={1: 0.5, 50: 0.5},
        teacher_forcing_mse=0.10,
        mean_spike_rate=0.12,
    )

    # Model 3: Poor H1 (0.35) near max (0.4), even with ideal drift (1.0)
    res_poor_h1 = RolloutEvaluationResult(
        horizons=[1, 50],
        latent_mse_per_horizon={1: 0.35, 50: 0.35},
        position_error_per_horizon={1: 0.35, 50: 0.35},  # drift = 1.0
        velocity_error_per_horizon={1: 0.5, 50: 0.5},
        teacher_forcing_mse=0.35,
        mean_spike_rate=0.10,
    )

    score_div = res_diverging.dynamic_score
    score_bal = res_balanced.dynamic_score
    score_poor_h1 = res_poor_h1.dynamic_score

    # Balanced model should beat severely diverging and poor H1 models
    assert score_bal < score_div
    assert score_bal < score_poor_h1
