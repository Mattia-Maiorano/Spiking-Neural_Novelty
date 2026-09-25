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
                    rollout_predictions = self.model.predict_future(z_warmup, horizon=max_eval_h)  # [B, max_h, D]

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
