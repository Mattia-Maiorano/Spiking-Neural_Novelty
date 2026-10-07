"""
Evaluation and Autonomous Rollout Engine.
Evaluates world models across varying prediction horizons (H = 1, 5, 10, 25, 50)
and quantifies long-horizon degradation, teacher forcing vs autonomous rollout,
and physical kinematic decoding accuracy.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spwm.learning.metrics import position_error, velocity_error, multi_step_mse


@dataclass
class RolloutEvaluationResult:
    """Contains multi-step evaluation metrics across horizons."""
    horizons: List[int]
    latent_mse_per_horizon: Dict[int, float]
    position_error_per_horizon: Dict[int, float]
    velocity_error_per_horizon: Dict[int, float]
    teacher_forcing_mse: float
    mean_spike_rate: float

    @property
    def drift_ratio(self) -> float:
        """
        Computes the Drift Ratio between long-horizon (H=50 or max horizon)
        and short-horizon (H=1 or min horizon).
        """
        if not self.position_error_per_horizon:
            return float("inf")
        min_k = 1 if 1 in self.position_error_per_horizon else min(self.position_error_per_horizon.keys())
        max_k = 50 if 50 in self.position_error_per_horizon else max(self.position_error_per_horizon.keys())
        h1 = self.position_error_per_horizon.get(min_k, 0.0)
        h_max = self.position_error_per_horizon.get(max_k, float("inf"))
        if h1 <= 0 or not np.isfinite(h_max) or not np.isfinite(h1):
            return float("inf")
        return float(h_max / max(h1, 1e-6))

    @property
    def rollout_mae(self) -> float:
        """Computes the mean absolute position error across all evaluated rollout horizons."""
        if not self.position_error_per_horizon:
            return float("inf")
        return float(np.mean(list(self.position_error_per_horizon.values())))

    @property
    def rollout_vel_mae(self) -> float:
        """Computes the mean absolute velocity error across all evaluated rollout horizons."""
        if not self.velocity_error_per_horizon:
            return float("inf")
        return float(np.mean(list(self.velocity_error_per_horizon.values())))

    def compute_dynamic_score(
        self,
        max_h1: float = 0.4,
        ideal_drift: float = 1.0,
        max_drift_delta: float = 4.0,
        epsilon: float = 1e-6,
    ) -> float:
        """
        Computes dynamic score weighting based on distance from global goals:
        - Absolute single-step loss target: 0.0 (normalized against max_h1 ~ 0.4)
        - Deviation / drift ratio target: 1.0 (normalized against max_drift_delta ~ 4.0, reaching up to 5.0)

        Distance weights dynamically penalize the axis that is further from its optimal goal.
        Lower score is better.
        """
        min_k = 1 if 1 in self.position_error_per_horizon else (min(self.position_error_per_horizon.keys()) if self.position_error_per_horizon else 1)
        h1_loss = self.position_error_per_horizon.get(min_k, float("inf"))
        drift = self.drift_ratio

        if not np.isfinite(h1_loss) or not np.isfinite(drift):
            return float("inf")

        # Normalized distances from target:
        # d_loss in [0, 1] relative to max_h1 (0.4)
        d_loss = max(0.0, float(h1_loss)) / max(max_h1, epsilon)
        # d_dev in [0, 1] relative to ideal_drift (1.0) up to max_drift (5.0 -> delta 4.0)
        d_dev = max(0.0, float(drift) - ideal_drift) / max(max_drift_delta, epsilon)

        # Dynamic weighting: weight is higher if further from goal
        w_loss = d_loss + epsilon
        w_dev = d_dev + epsilon

        score = (w_loss * d_loss + w_dev * d_dev) / (w_loss + w_dev)
        return float(score)

    @property
    def dynamic_score(self) -> float:
        """Computes default dynamic score."""
        return self.compute_dynamic_score()

    def is_gate_passed(self, max_drift_ratio: float = 5.0, max_long_err: float = 2.0) -> bool:
        """
        Soft/Legacy Gate compatibility check.
        """
        ratio = self.drift_ratio
        max_k = 50 if 50 in self.position_error_per_horizon else max(self.position_error_per_horizon.keys(), default=1)
        h_max_val = self.position_error_per_horizon.get(max_k, float("inf"))
        return bool((ratio <= max_drift_ratio) and (h_max_val < max_long_err) and np.isfinite(ratio) and (ratio > 0))



class RolloutEvaluator:
    """
    Evaluates autonomous rollouts and long-horizon degradation curves.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[Union[str, torch.device]] = None,
        horizons: Sequence[int] = (1, 5, 10, 25, 50),
        num_objects: int = 1,
        use_corrector: bool = True,
    ) -> None:
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()
        self.horizons = list(horizons)
        self.num_objects = num_objects
        self.use_corrector = use_corrector

    @torch.no_grad()
    def evaluate_dataset(self, data_loader: DataLoader, max_batches: Optional[int] = None) -> RolloutEvaluationResult:
        """
        Runs comprehensive rollout evaluation across all batches in data_loader.
        """
        latent_errors: Dict[int, List[float]] = {h: [] for h in self.horizons}
        pos_errors: Dict[int, List[float]] = {h: [] for h in self.horizons}
        vel_errors: Dict[int, List[float]] = {h: [] for h in self.horizons}
        tf_errors: List[float] = []
        spike_rates: List[float] = []

        batch_count = 0
        for batch in data_loader:
            events = batch["events"].to(self.device)  # [B, T, 2, H, W]
            true_kin = batch["flat_kinematics"].to(self.device)  # [B, T, 4 * N]
            B, T, _, _, _ = events.shape

            out = self.model(events)
            latent_states = out.latent_states  # [B, T, D]

            # Teacher forcing one-step error
            pred_latents = out.predicted_latents[:, :-1]
            target_latents = latent_states[:, 1:]
            tf_mse = torch.mean((pred_latents - target_latents) ** 2).item()
            tf_errors.append(tf_mse)

            if hasattr(out, "mean_spike_rate"):
                spike_rates.append(out.mean_spike_rate.item())

            # Evaluate autonomous rollouts for each horizon
            # Ensure warm-up step allows maximum possible rollout without index out of bounds
            t_warmup = min(10, max(1, T // 4))
            if t_warmup < T - 1:
                z_warmup = latent_states[:, t_warmup]
                max_eval_h = min(max(self.horizons), T - t_warmup - 1)

                if max_eval_h > 0:
                    if hasattr(self.model, "predict_future"):
                        rollout_predictions = self.model.predict_future(
                            z_warmup, horizon=max_eval_h, use_corrector=self.use_corrector
                        )  # [B, max_h, D]
                    else:
                        rollout_predictions = self.model.predict_rollout(
                            z_warmup, horizon=max_eval_h, use_corrector=self.use_corrector
                        )[0]

                    for h in self.horizons:
                        if h <= max_eval_h and (t_warmup + 1 + h) <= T:
                            pred_h = rollout_predictions[:, :h]
                            target_h = latent_states[:, t_warmup + 1 : t_warmup + 1 + h]

                            h_mse = torch.mean((pred_h - target_h) ** 2).item()
                            latent_errors[h].append(h_mse)

                            # Decoded physical errors from autonomous predictions
                            if hasattr(self.model, "physical_decoder"):
                                decoded_h = self.model.physical_decoder(pred_h)
                                kin_target_h = true_kin[:, t_warmup + 1 : t_warmup + 1 + h]
                                pos_err = position_error(decoded_h, kin_target_h, self.num_objects)
                                vel_err = velocity_error(decoded_h, kin_target_h, self.num_objects)
                                pos_errors[h].append(pos_err)
                                vel_errors[h].append(vel_err)

            batch_count += 1
            if max_batches is not None and batch_count >= max_batches:
                break

        # Compute averages
        avg_latent_mse = {h: float(np.mean(latent_errors[h])) if latent_errors[h] else 0.0 for h in self.horizons}
        avg_pos_err = {h: float(np.mean(pos_errors[h])) if pos_errors[h] else 0.0 for h in self.horizons}
        avg_vel_err = {h: float(np.mean(vel_errors[h])) if vel_errors[h] else 0.0 for h in self.horizons}
        avg_tf = float(np.mean(tf_errors)) if tf_errors else 0.0
        avg_spk = float(np.mean(spike_rates)) if spike_rates else 0.0

        return RolloutEvaluationResult(
            horizons=self.horizons,
            latent_mse_per_horizon=avg_latent_mse,
            position_error_per_horizon=avg_pos_err,
            velocity_error_per_horizon=avg_vel_err,
            teacher_forcing_mse=avg_tf,
            mean_spike_rate=avg_spk,
        )
