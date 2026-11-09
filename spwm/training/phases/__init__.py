"""
Isolated Training Phase Strategies (Layer 2 — Training).

Exposes:
- `RepresentationPhaseStrategy` (`spwm/training/phases/representation.py`)
- `M1PhaseStrategy` (`spwm/training/phases/m1_nominal.py`)
- `build_phase_strategy` factory
- `PredictorPhaseHandler` dispatcher for unified lifecycle compatibility
"""

from __future__ import annotations

from typing import Dict, Optional
import chronicle
import torch
import torch.nn as nn

from spwm.contracts.config import ExperimentConfig
from spwm.contracts.protocols import PhaseStrategy
from spwm.training.phases.representation import RepresentationPhaseStrategy
from spwm.training.phases.m1_nominal import M1PhaseStrategy

PhaseHandler = PhaseStrategy


def build_phase_strategy(
    model: nn.Module,
    cfg: ExperimentConfig,
    loss_fn: nn.Module,
    device: torch.device,
) -> PhaseStrategy:
    """Instantiate the isolated PhaseStrategy corresponding to `cfg.training.mode`."""
    mode = cfg.training.mode.lower()
    if mode in ("train_representation", "train_phase0", "representation"):
        return RepresentationPhaseStrategy(model=model, cfg=cfg, loss_fn=loss_fn, device=device)
    if mode in ("train_m1", "m1"):
        return M1PhaseStrategy(model=model, cfg=cfg, loss_fn=loss_fn, device=device)
    raise ValueError(
        f"Unsupported training mode '{mode}'. Supported modes: 'train_representation', 'train_m1'"
    )


class PredictorPhaseHandler(PhaseStrategy):
    """Thin dispatcher that delegates to RepresentationPhaseStrategy or M1PhaseStrategy."""

    def setup(self) -> None:
        self.strategy: PhaseStrategy = build_phase_strategy(
            model=self.model,
            cfg=self.cfg,
            loss_fn=self.loss_fn,
            device=self.device,
        )
        self.predictor_optimizer = self.strategy.predictor_optimizer
        self.representation_optimizer = None

    @property
    def encoder_optimizer(self) -> Optional[torch.optim.Optimizer]:
        return self.representation_optimizer

    @property
    def probe_optimizer(self) -> Optional[torch.optim.Optimizer]:
        return self.representation_optimizer

    def verify_gradient_isolation(self) -> None:
        self.strategy.verify_gradient_isolation()

    def train_step(self, batch: Dict[str, torch.Tensor], epoch: int) -> Dict[str, float]:
        return self.strategy.train_step(batch, epoch=epoch)

    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        return self.strategy.validate_step(batch)

    def log_epoch(
        self,
        epoch: int,
        end_epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch_duration: float,
        total_elapsed: float,
        is_best: bool,
        best_epoch: Optional[int],
        best_score: float,
    ) -> None:
        best_tag = " * [BEST]" if is_best else ""
        mode = self.cfg.training.mode.upper()
        header = (
            f"Epoch [{epoch:03d}/{end_epoch:03d}] [{mode}]{best_tag}  "
            f"Time: {epoch_duration:5.1f}s | Total: {total_elapsed / 60:4.1f}m | "
            f"Remaining: {epoch_duration * (end_epoch - epoch) / 60:4.1f}m - Patience: {epoch_duration * 10 / 60:4.1}"
        )
        chronicle.log_section_header(header)
        train_loss = train_metrics.get("train_loss", 0.0)
        train_pos_l = train_metrics.get("train_loss_position", 0.0)
        train_vel_l = train_metrics.get("train_loss_velocity", 0.0)
        train_critic_l = train_metrics.get("train_loss_critic", None)
        train_pos_e = train_metrics.get("train_pos_err", 0.0)
        train_vel_e = train_metrics.get("train_vel_err", 0.0)
        train_alpha = train_metrics.get("train_mppi_noise_alpha", 0.0)
        curr_phase = int(train_metrics.get("curriculum_phase", 1))
        w_t = train_metrics.get("active_perception_w_t", 0.60)
        dagger_ratio = train_metrics.get("dagger_ratio", 0.0)
        curric_str = (
            f" | Curriculum Phase {curr_phase}/4 (w_t: {w_t:.2f}, Noise α: {train_alpha:.2f}, DAgger: {dagger_ratio * 100:.0f}%)"
            if mode in ("TRAIN_M1", "M1")
            else ""
        )
        chronicle.log_detail(
            "⚡ Training Losses",
            f"Total: {train_loss:.5f} [q-MSE: {train_pos_l:.5f} | p-MSE: {train_vel_l:.5f}]{curric_str} | "
            f"Eval: Joint Angle L2: {train_pos_e:.5f} | Joint Vel L2: {train_vel_e:.5f}",
            indent_level=1,
        )
        if train_critic_l is not None:
            chronicle.log_detail(
                "⚡ Critic Loss",
                f"Critic-MSE: {train_critic_l:.5f}",
                indent_level=1,
            )

        cl_trajs = int(train_metrics.get("closed_loop_trajs_epoch", 0))
        cl_trans = int(train_metrics.get("closed_loop_trans_epoch", 0))
        buf_trajs = int(train_metrics.get("closed_loop_buffer_trajs", 0))
        buf_total = int(train_metrics.get("closed_loop_buffer_total", 0))
        traj_desc = f"{buf_trajs} trajs" if buf_trajs > 0 else (f"{max(1, buf_total // 100)} trajs" if buf_total > 0 else "0 trajs")
        if cl_trajs > 0:
            chronicle.log_detail(
                "🔄 Closed-Loop MPPI BPTT",
                f"Trained in Epoch: {cl_trajs} trajectory rollouts ({cl_trans} transitions) with BPTT | Replay Buffer: {traj_desc} ({buf_total} transitions)",
                indent_level=1,
            )
        elif buf_total > 0:
            chronicle.log_detail(
                "🔄 Closed-Loop MPPI BPTT",
                f"Replay Buffer: {traj_desc} ({buf_total} transitions, threshold >= 64)",
                indent_level=1,
            )
        elif mode in ("TRAIN_M1", "M1"):
            chronicle.log_detail(
                "🔄 Closed-Loop MPPI BPTT",
                "Replay Buffer: 0 trajs (0 transitions, waiting for Tier 2 evaluation)",
                indent_level=1,
            )

        val_loss = val_metrics.get("val_loss", 0.0)
        val_pos_l = val_metrics.get("val_loss_position", 0.0)
        val_vel_l = val_metrics.get("val_loss_velocity", 0.0)
        val_critic_l = val_metrics.get("val_loss_critic", None)
        val_pos_e = val_metrics.get("val_pos_err", 0.0)
        val_vel_e = val_metrics.get("val_vel_err", 0.0)
        chronicle.log_detail(
            "📊 Validation Losses (Tier 1 Open-Loop)",
            f"Total: {val_loss:.5f} [q-MSE: {val_pos_l:.5f} | p-MSE: {val_vel_l:.5f}] | "
            f"Eval: Joint Angle L2: {val_pos_e:.5f} | Joint Vel L2: {val_vel_e:.5f}",
            indent_level=1,
        )
        if val_critic_l is not None:
            chronicle.log_detail(
                "📊 Critic Validation (Tier 1)",
                f"Critic-MSE: {val_critic_l:.5f}",
                indent_level=1,
            )

        if "mppi_score" in val_metrics:
            mppi_s = val_metrics["mppi_score"]
            mppi_succ = val_metrics.get("mppi_success_rate", 0.0) * 100.0
            mppi_err = val_metrics.get("mppi_mean_error_cm", 0.0)
            mppi_rot = val_metrics.get("mppi_rot_err_rad", val_metrics.get("mppi_final_rot", 0.0))
            mppi_trajs = int(val_metrics.get("mppi_evaluated_trajectories", 0))
            traj_info = f" ({mppi_trajs} Eval Trajectories)" if mppi_trajs > 0 else ""
            chronicle.log_detail(
                f"🎯 Tier 2 (Closed-Loop MPPI){traj_info}",
                f"MPPI Cost: {mppi_s:.5f} | Mean Cartesian Err (cm): {mppi_err:.2f}cm | "
                f"Rotational Err (rad): {mppi_rot:.4f} rad | Success Rate: {mppi_succ:.1f}%",
                indent_level=1,
            )
            curr_score = mppi_s
            sel_desc = f"Tier 2 MPPI Cost: {curr_score:.5f}"
        else:
            curr_score = self.get_selection_score(val_metrics)
            sel_desc = f"Tier 1 Open-Loop MSE: {curr_score:.5f} (Ignored for Best Model Selection)" if mode in ("TRAIN_M1", "M1") else f"Tier 1 Open-Loop MSE: {curr_score:.5f}"

        chronicle.log_detail(
            "🎯 Model Selection",
            f"Current: {sel_desc} | Best Checkpoint: {best_score:.5f} @ Ep {best_epoch}",
            indent_level=1,
        )
        chronicle.log_newline()


__all__ = [
    "PhaseStrategy",
    "PhaseHandler",
    "RepresentationPhaseStrategy",
    "M1PhaseStrategy",
    "build_phase_strategy",
    "PredictorPhaseHandler",
]
