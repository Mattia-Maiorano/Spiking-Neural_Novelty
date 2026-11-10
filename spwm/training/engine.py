"""
BaseTrainer & Trainer Engine for SPWM (Layer 2 — Training).

Single source of truth for:
- Epoch lifecycle orchestration (`fit`, `train_epoch`, `evaluate`)
- Scoped `strict=True` checkpoint management (`save_checkpoint`, `load_checkpoint`,
  `load_representation_checkpoint`)
- Gradient isolation verification (`verify_gradient_isolation`)

Zero upward imports into `spwm.evaluation` and zero silent `getattr` or `strict=False` fallbacks.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import inspect
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Union
import chronicle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spwm.contracts.config import (
    DataConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    LatentNeuronConfig,
    LossConfig,
    MemoryConfig,
    ModelConfig,
    ProjectConfig,
    TrainingConfig,
)
from spwm.contracts.protocols import PhaseStrategy, Tier2EvaluatorProtocol
from spwm.contracts.version import GLOBAL_VERSION
from spwm.training.losses import SPWMLoss
from spwm.training.noise import LatchCurriculumState, compute_dagger_ratio
from spwm.training.phases import PredictorPhaseHandler


def _extract_submodel_state(full_state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Extract keys starting with `prefix.` and strip the prefix."""
    p_dot = f"{prefix}."
    sub_state = {k[len(p_dot) :]: v for k, v in full_state.items() if k.startswith(p_dot)}
    if not sub_state:
        raise RuntimeError(f"Checkpoint state_dict contains no keys for submodule prefix '{prefix}.'")
    return sub_state


def load_representation_checkpoint(
    model: nn.Module,
    ckpt_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
) -> None:
    """Strictly load Phase 0 (`train_representation`) trained components into `model`.

    Loads `encoder`, `sensory_predictor`, `dynamics`, and `physical_decoder`
    each with `strict=True`.
    """
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Representation checkpoint not found at: {path}")
    raw = torch.load(path, map_location=device or "cpu", weights_only=False)
    state_dict = raw["model_state"] if isinstance(raw, dict) and "model_state" in raw else raw

    model.encoder.load_state_dict(_extract_submodel_state(state_dict, "encoder"), strict=True)
    model.sensory_predictor.load_state_dict(
        _extract_submodel_state(state_dict, "sensory_predictor"), strict=True
    )
    model.dynamics.load_state_dict(_extract_submodel_state(state_dict, "dynamics"), strict=True)
    model.physical_decoder.load_state_dict(
        _extract_submodel_state(state_dict, "physical_decoder"), strict=True
    )


def resolve_version_checkpoint(
    version_name: str,
    phase_mode: str,
    explicit_path: Optional[Union[str, Path]] = None,
    results_root: Union[str, Path] = "results",
) -> Path:
    """Resolve and strictly verify the best validation `model.pt` for a given version and phase.

    Searches `explicit_path` if provided, otherwise searches `<results_root>/*<version_name>*_<phase_mode>/model.pt`
    (never `checkpoint.pt`). Raises `FileNotFoundError` if no matching `model.pt` exists.
    """
    if explicit_path is not None:
        cand = Path(explicit_path)
        if cand.is_dir():
            cand = cand / "model.pt"
        if not cand.is_file():
            raise FileNotFoundError(
                f"Required best-validation checkpoint 'model.pt' for {phase_mode} not found at: {cand}"
            )
        return cand

    root = Path(results_root)
    exact_cand = root / f"{version_name}_{phase_mode}" / "model.pt"
    if exact_cand.is_file():
        return exact_cand

    clean_ver = version_name[5:] if version_name.startswith("spwm_") else version_name
    alt_cand = root / f"spwm_{clean_ver}_{phase_mode}" / "model.pt"
    if alt_cand.is_file():
        return alt_cand

    if root.is_dir():
        matches = sorted(root.glob(f"*{clean_ver}*_{phase_mode}/model.pt"))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Auto-enforced prerequisite checkpoint 'model.pt' for phase '{phase_mode}' and version '{version_name}' "
        f"not found in '{root}'. Ensure '{phase_mode}' has been trained first (expected: {exact_cand})."
    )


def resolve_closed_loop_buffer(
    version_name: str,
    phase_mode: str = "train_m1",
    explicit_path: Optional[Union[str, Path]] = None,
    save_dir: Optional[Union[str, Path]] = None,
    results_root: Union[str, Path] = "results",
    is_resuming: bool = False,
) -> Optional[Path]:
    """Resolve existing closed_loop_buffer.pt from explicit path or save_dir when resuming.

    Buffers must be initialized completely fresh for every new training run to eliminate
    cross-run contamination. A buffer is loaded if and only if:
      1. An explicit path is provided via explicit_path (e.g. CLI --closed-loop-buffer), OR
      2. The run is explicitly resuming an interrupted checkpoint (is_resuming=True)
         and closed_loop_buffer.pt exists in save_dir.
    """
    if explicit_path is not None:
        cand = Path(explicit_path)
        if cand.is_dir():
            cand = cand / "closed_loop_buffer.pt"
        if not cand.is_file():
            raise FileNotFoundError(f"Explicit closed_loop_buffer.pt not found at: {cand}")
        return cand

    if is_resuming and save_dir is not None:
        save_path = Path(save_dir) / "closed_loop_buffer.pt"
        if save_path.is_file() and save_path.stat().st_size > 2500:
            return save_path

    return None


def enforce_phase_prerequisites(
    model: nn.Module,
    cfg: ExperimentConfig,
    device: Union[str, torch.device],
    results_root: Union[str, Path] = "results",
) -> Dict[str, Path]:
    """Auto-enforce and strictly load prerequisite `model.pt` checkpoint for `train_m1`.

    - `train_m1`: Requires and loads pre-trained representation (`train_representation` `model.pt`).
    """
    mode = cfg.training.mode.lower()
    loaded: Dict[str, Path] = {}

    if mode == "train_m1":
        if cfg.training.representation_checkpoint:
            rep_path = resolve_version_checkpoint(
                cfg.project.name,
                "train_representation",
                explicit_path=cfg.training.representation_checkpoint,
                results_root=results_root,
            )
            load_representation_checkpoint(model, rep_path, device=device)
            loaded["representation"] = rep_path
            chronicle.log_success(f"Strictly loaded representation model.pt: {rep_path}")
        else:
            try:
                rep_path = resolve_version_checkpoint(
                    cfg.project.name,
                    "train_representation",
                    results_root=results_root,
                )
                load_representation_checkpoint(model, rep_path, device=device)
                loaded["representation"] = rep_path
                chronicle.log_success(f"Strictly loaded representation model.pt: {rep_path}")
            except FileNotFoundError as err:
                raise FileNotFoundError(
                    f"Phase 1 ('train_m1') strictly requires a pre-trained representation checkpoint. "
                    f"No checkpoint found for project '{cfg.project.name}' at results_root='{results_root}'. "
                    f"Pass --representation-checkpoint <path> (e.g. {GLOBAL_VERSION.representation_checkpoint()}) "
                    f"or run Phase 0 first."
                ) from err

    return loaded


class EMATracker:
    """
    Exponential Moving Average (EMA) tracker for model weights.

    Maintains shadow weights tracking an exponential moving average:
        theta_ema <- tau * theta_ema + (1 - tau) * theta
    Provides strict loading of EMA weights into the model for Tier 2 closed-loop MPPI evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.990,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if not (0.0 <= decay <= 1.0):
            raise ValueError(f"EMA decay must be in [0.0, 1.0], got {decay}")
        self.decay = float(decay)
        self.device = torch.device(device) if device is not None else None
        self.shadow: Dict[str, torch.Tensor] = {}
        for k, v in model.state_dict().items():
            t = v.clone().detach()
            if self.device is not None:
                t = t.to(self.device)
            self.shadow[k] = t

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA shadow parameters with current model parameters."""
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                t = v.clone().detach()
                if self.device is not None:
                    t = t.to(self.device)
                self.shadow[k] = t
                continue
            if torch.is_floating_point(v):
                target_v = v.detach()
                if self.device is not None and target_v.device != self.shadow[k].device:
                    target_v = target_v.to(self.shadow[k].device)
                self.shadow[k].mul_(self.decay).add_(target_v, alpha=1.0 - self.decay)
            else:
                target_v = v.detach()
                if self.device is not None and target_v.device != self.shadow[k].device:
                    target_v = target_v.to(self.shadow[k].device)
                self.shadow[k].copy_(target_v)

    def copy_to(self, model: nn.Module) -> None:
        """Loads EMA weights into model with strict=True."""
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        for k, v in state_dict.items():
            t = v.clone().detach()
            if self.device is not None:
                t = t.to(self.device)
            self.shadow[k] = t


class ClosedLoopBuffer:
    """DAgger-style replay buffer for closed-loop MPPI trajectories (Layer 2 — Training).

    Stores closed-loop trajectories and transition tuples collected during Tier 2
    validation to mitigate exposure bias during Phase 1 (train_m1) training via BPTT.
    Enforces a strict FIFO rolling window across evaluation generations to eliminate
    distributional drift and cross-run contamination.
    """

    def __init__(self, capacity: int = 50_000, max_eval_generations: int = 3) -> None:
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        if max_eval_generations <= 0:
            raise ValueError(f"max_eval_generations must be positive, got {max_eval_generations}")
        self.capacity = int(capacity)
        self.max_eval_generations = int(max_eval_generations)
        self.trajectory_capacity = max(100, int(capacity // 50))
        self.z: List[torch.Tensor] = []
        self.u: List[torch.Tensor] = []
        self.z_next: List[torch.Tensor] = []
        self.trajectories: List[Dict[str, Any]] = []
        self.eval_generations: List[List[Dict[str, Any]]] = []
        self._ptr = 0
        self._traj_ptr = 0

    def clear(self) -> None:
        """Clear all stored transitions, trajectories, and evaluation generations."""
        self.z.clear()
        self.u.clear()
        self.z_next.clear()
        self.trajectories.clear()
        self.eval_generations.clear()
        self._ptr = 0
        self._traj_ptr = 0

    def push(
        self,
        transitions: Union[
            Sequence[Dict[str, torch.Tensor]],
            Dict[str, Sequence[torch.Tensor]],
            Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        ],
    ) -> None:
        """Push transitions into the replay buffer with FIFO eviction on overflow."""
        items: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        if isinstance(transitions, dict):
            zs = transitions["z"]
            us = transitions["u"]
            z_nexts = transitions["z_next"]
            n = len(zs)
            for i in range(n):
                items.append((zs[i], us[i], z_nexts[i]))
        elif isinstance(transitions, (list, tuple)):
            for item in transitions:
                if isinstance(item, dict):
                    items.append((item["z"], item["u"], item["z_next"]))
                elif isinstance(item, (tuple, list)) and len(item) == 3:
                    items.append((item[0], item[1], item[2]))
                else:
                    raise ValueError(f"Unsupported transition item format: {type(item)}")
        else:
            raise ValueError(f"Unsupported transitions container: {type(transitions)}")

        for z, u, z_next in items:
            z_t = z.detach().cpu().view(-1)
            u_t = u.detach().cpu().view(-1)
            zn_t = z_next.detach().cpu().view(-1)

            if len(self.z) < self.capacity:
                self.z.append(z_t)
                self.u.append(u_t)
                self.z_next.append(zn_t)
            else:
                self.z[self._ptr] = z_t
                self.u[self._ptr] = u_t
                self.z_next[self._ptr] = zn_t
                self._ptr = (self._ptr + 1) % self.capacity

    def push_trajectories(self, trajectories: Sequence[Dict[str, Any]]) -> None:
        """Push a generation of closed-loop MPPI trajectories into buffer.

        Maintains a strict FIFO rolling window across Tier 2 evaluation generations.
        When new trajectories are ingested, any generation older than
        max_eval_generations is cleanly evicted to eliminate obsolete on-policy errors.
        """
        if not trajectories:
            return

        clean_gen: List[Dict[str, Any]] = []
        for traj in trajectories:
            actions_t = traj["actions"].detach().cpu()
            targets_t = traj["targets"].detach().cpu()
            if actions_t.ndim == 3 and actions_t.shape[1] == 1:
                actions_t = actions_t.squeeze(1)
            if targets_t.ndim == 3 and targets_t.shape[1] == 1:
                targets_t = targets_t.squeeze(1)
            clean_traj = {
                "z_init": traj["z_init"].detach().cpu().view(-1),
                "actions": actions_t,
                "targets": targets_t,
                "length": int(traj["length"]),
                "trajectory_id": int(traj["trajectory_id"]),
                "is_expert": bool(traj["is_expert"]) if "is_expert" in traj else False,
                "elevation_plane": str(traj["elevation_plane"]) if "elevation_plane" in traj else "Unknown",
            }
            clean_gen.append(clean_traj)

        # Cleanly discard trajectories from generation N - max_eval_generations (e.g. N-3) before ingesting new ones
        while len(self.eval_generations) >= self.max_eval_generations:
            self.eval_generations.pop(0)
        self.eval_generations.append(clean_gen)

        # Flatten active generations into active trajectories
        self.trajectories = [t for gen in self.eval_generations for t in gen]
        if len(self.trajectories) > self.trajectory_capacity:
            self.trajectories = self.trajectories[-self.trajectory_capacity:]

        # Synchronize transition-level storage with the active generations
        self.z = []
        self.u = []
        self.z_next = []
        for clean_traj in self.trajectories:
            curr = clean_traj["z_init"]
            actions = clean_traj["actions"]
            targets = clean_traj["targets"]
            for step_idx in range(clean_traj["length"]):
                u_s = actions[step_idx].detach().cpu().view(-1)
                z_next_s = targets[step_idx].detach().cpu().view(-1)
                self.z.append(curr.detach().cpu().view(-1))
                self.u.append(u_s)
                self.z_next.append(z_next_s)
                curr = z_next_s

        if len(self.z) > self.capacity:
            self.z = self.z[-self.capacity:]
            self.u = self.u[-self.capacity:]
            self.z_next = self.z_next[-self.capacity:]
        self._ptr = 0
        self._traj_ptr = 0

    def sample(
        self,
        batch_size: int,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions uniformly at random."""
        cur_len = len(self.z)
        if cur_len == 0:
            raise ValueError("Cannot sample from an empty ClosedLoopBuffer.")
        k = min(batch_size, cur_len)
        indices = torch.randint(0, cur_len, (k,))
        dev = torch.device(device) if device is not None else None
        sampled_z = torch.stack([self.z[i].to(dev) for i in indices]) if dev is not None else torch.stack([self.z[i] for i in indices])
        sampled_u = torch.stack([self.u[i].to(dev) for i in indices]) if dev is not None else torch.stack([self.u[i] for i in indices])
        sampled_zn = torch.stack([self.z_next[i].to(dev) for i in indices]) if dev is not None else torch.stack([self.z_next[i] for i in indices])
        return {
            "z": sampled_z,
            "u": sampled_u,
            "z_next": sampled_zn,
        }

    def sample_trajectories(
        self,
        batch_size: int,
        horizon: int,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Sample trajectory sub-windows for multi-step BPTT training."""
        dev = torch.device(device) if device is not None else None
        if not self.trajectories:
            if len(self.z) == 0:
                raise ValueError("Cannot sample trajectories from an empty ClosedLoopBuffer.")
            k = min(batch_size, len(self.z))
            indices = torch.randint(0, len(self.z), (k,))
            sampled_z = torch.stack([self.z[i].to(dev) for i in indices]) if dev is not None else torch.stack([self.z[i] for i in indices])
            sampled_u = torch.stack([self.u[i].to(dev) for i in indices]).unsqueeze(1) if dev is not None else torch.stack([self.u[i] for i in indices]).unsqueeze(1)
            sampled_zn = torch.stack([self.z_next[i].to(dev) for i in indices]).unsqueeze(1) if dev is not None else torch.stack([self.z_next[i] for i in indices]).unsqueeze(1)
            return {"z_init": sampled_z, "actions": sampled_u, "targets": sampled_zn}

        k = int(batch_size)
        indices = torch.randint(0, len(self.trajectories), (k,))
        sampled_trajs = [self.trajectories[int(i.item())] for i in indices]
        H = min(horizon, min(t["length"] for t in sampled_trajs))

        z_inits: List[torch.Tensor] = []
        acts_list: List[torch.Tensor] = []
        targets_list: List[torch.Tensor] = []
        for t in sampled_trajs:
            L = t["length"]
            max_start = max(0, L - H)
            start_idx = int(torch.randint(0, max_start + 1, (1,)).item()) if max_start > 0 else 0
            z_i = (t["z_init"] if start_idx == 0 else t["targets"][start_idx - 1]).view(-1)
            u_i = t["actions"][start_idx : start_idx + H]
            tgt_i = t["targets"][start_idx : start_idx + H]
            if u_i.ndim == 3 and u_i.shape[1] == 1:
                u_i = u_i.squeeze(1)
            if tgt_i.ndim == 3 and tgt_i.shape[1] == 1:
                tgt_i = tgt_i.squeeze(1)
            z_inits.append(z_i)
            acts_list.append(u_i)
            targets_list.append(tgt_i)

        b_z = torch.stack([z.to(dev) for z in z_inits]) if dev is not None else torch.stack(z_inits)
        b_u = torch.stack([u.to(dev) for u in acts_list]) if dev is not None else torch.stack(acts_list)
        b_tgt = torch.stack([tgt.to(dev) for tgt in targets_list]) if dev is not None else torch.stack(targets_list)
        return {"z_init": b_z, "actions": b_u, "targets": b_tgt}

    def __len__(self) -> int:
        return len(self.z)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "max_eval_generations": self.max_eval_generations,
            "trajectory_capacity": self.trajectory_capacity,
            "ptr": self._ptr,
            "traj_ptr": self._traj_ptr,
            "z": torch.stack(self.z) if self.z else torch.empty(0),
            "u": torch.stack(self.u) if self.u else torch.empty(0),
            "z_next": torch.stack(self.z_next) if self.z_next else torch.empty(0),
            "trajectories": self.trajectories,
            "eval_generations": self.eval_generations,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.capacity = int(state["capacity"])
        if "max_eval_generations" in state:
            self.max_eval_generations = int(state["max_eval_generations"])
        if "trajectory_capacity" in state:
            self.trajectory_capacity = int(state["trajectory_capacity"])
        self._ptr = int(state["ptr"])
        if "traj_ptr" in state:
            self._traj_ptr = int(state["traj_ptr"])
        z_t = state["z"]
        u_t = state["u"]
        zn_t = state["z_next"]
        if z_t.numel() > 0:
            self.z = [z_t[i] for i in range(z_t.shape[0])]
            self.u = [u_t[i] for i in range(u_t.shape[0])]
            self.z_next = [zn_t[i] for i in range(zn_t.shape[0])]
        else:
            self.z = []
            self.u = []
            self.z_next = []
        if "eval_generations" in state and state["eval_generations"]:
            self.eval_generations = [list(gen) for gen in state["eval_generations"]]
            while len(self.eval_generations) > self.max_eval_generations:
                self.eval_generations.pop(0)
            self.trajectories = [t for gen in self.eval_generations for t in gen]
        elif "trajectories" in state and state["trajectories"]:
            self.trajectories = list(state["trajectories"])
            self.eval_generations = [self.trajectories]
        else:
            self.trajectories = []
            self.eval_generations = []

        # Ensure backward compatibility when extending runs from older configurations
        # Mark unannotated legacy trajectories explicitly with verbose status; never fabricate boolean origins
        for gen in self.eval_generations:
            for t in gen:
                if "is_expert" not in t:
                    t["is_expert"] = "UNANNOTATED_LEGACY_ORIGIN"
                if "elevation_plane" not in t:
                    t["elevation_plane"] = "UNANNOTATED_LEGACY_ELEVATION"
        for t in self.trajectories:
            if "is_expert" not in t:
                t["is_expert"] = "UNANNOTATED_LEGACY_ORIGIN"
            if "elevation_plane" not in t:
                t["elevation_plane"] = "UNANNOTATED_LEGACY_ELEVATION"

    def save(self, path: Union[str, Path]) -> None:
        torch.save(self.state_dict(), Path(path))

    def load(self, path: Union[str, Path]) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Buffer checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(state)


class BaseTrainer:
    """Standardized, modular training engine delegating phase mechanics to `PhaseStrategy`."""

    def __init__(
        self,
        model: nn.Module,
        cfg: ExperimentConfig,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[SPWMLoss] = None,
        device: Optional[Union[str, torch.device]] = None,
        save_dir: Union[str, Path] = "results/default_run",
        start_epoch: int = 1,
        best_score: float = float("inf"),
        best_epoch: Optional[int] = None,
        history: Optional[List[Dict[str, float]]] = None,
        auto_enforce_checkpoints: bool = False,
        results_root: Union[str, Path] = "results",
        tier2_evaluator: Optional[Tier2EvaluatorProtocol] = None,
        closed_loop_buffer: Optional[Union[str, Path]] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.results_root = Path(results_root)
        self.tier2_evaluator = tier2_evaluator
        self.tier2_frequency = cfg.training.tier2_frequency
        self.tier2_candidate_selection = cfg.training.tier2_candidate_selection
        self.tier2_patience = cfg.training.tier2_patience
        self.tier2_min_delta = cfg.training.tier2_min_delta
        self.ema_decay = cfg.training.ema_decay

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.ema_tracker: Optional[EMATracker] = (
            EMATracker(model=self.model, decay=self.ema_decay, device=self.device)
            if self.tier2_candidate_selection == "ema"
            else None
        )
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if loss_fn is None:
            self.loss_fn = SPWMLoss.from_config(cfg.loss).to(self.device)
        else:
            self.loss_fn = loss_fn.to(self.device)

        self.mode = cfg.training.mode.lower()
        if auto_enforce_checkpoints and self.mode == "train_m1":
            enforce_phase_prerequisites(
                self.model, self.cfg, device=self.device, results_root=self.results_root
            )

        self.handler: PhaseStrategy = PredictorPhaseHandler(
            model=self.model, cfg=self.cfg, loss_fn=self.loss_fn, device=self.device
        )

        self.replay_buffer: Optional[ClosedLoopBuffer] = (
            ClosedLoopBuffer(
                capacity=50_000,
                max_eval_generations=self.cfg.training.max_eval_generations,
            )
            if self.mode in ("train_m1", "m1")
            else None
        )
        if self.replay_buffer is not None and closed_loop_buffer is not None:
            resolved_buffer = resolve_closed_loop_buffer(
                version_name=self.cfg.project.name,
                phase_mode=self.mode,
                explicit_path=closed_loop_buffer,
                save_dir=self.save_dir,
                results_root=self.results_root,
                is_resuming=False,
            )
            if resolved_buffer is not None:
                self.replay_buffer.load(resolved_buffer)
                chronicle.log_info(
                    f"Loaded explicit ClosedLoopBuffer with {len(self.replay_buffer)} transitions "
                    f"({len(self.replay_buffer.trajectories)} trajectories) from {resolved_buffer}"
                )

        self.start_epoch = start_epoch
        self.best_score = best_score
        self.best_epoch = best_epoch
        self.history: List[Dict[str, float]] = history or []
        self.curriculum = LatchCurriculumState(
            tier2_frequency=self.tier2_frequency,
            noise_ramp_epochs=self.cfg.training.mppi_noise_ramp_epochs,
            target_alpha=self.cfg.training.mppi_noise_target_alpha,
        )
        self.current_curriculum_phase: int = 1

    @property
    def predictor_optimizer(self) -> Optional[torch.optim.Optimizer]:
        return self.handler.predictor_optimizer

    @property
    def representation_optimizer(self) -> Optional[torch.optim.Optimizer]:
        return None

    @property
    def encoder_optimizer(self) -> Optional[torch.optim.Optimizer]:
        return None

    @property
    def probe_optimizer(self) -> Optional[torch.optim.Optimizer]:
        return None

    def verify_gradient_isolation(self) -> None:
        """Delegates gradient isolation assertion to the active PhaseStrategy."""
        self.handler.verify_gradient_isolation()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Executes a single epoch over train_loader using the 4-Phase Latch Curriculum and active PhaseStrategy."""
        if self.train_loader is None:
            raise RuntimeError("Cannot call train_epoch when train_loader is None.")
        train_accum: Dict[str, float] = {}
        batch_count = 0
        min_buffer_samples = 1
        cl_trajs_epoch = 0
        cl_trans_epoch = 0

        if self.mode in ("train_m1", "m1"):
            active_phase, w_t, alpha, dagger_ratio = self.curriculum.begin_epoch()
            self.current_curriculum_phase = active_phase
            target_lr = 0.00005 if active_phase == 4 else self.cfg.training.learning_rate
            if self.handler.predictor_optimizer is not None:
                for pg in self.handler.predictor_optimizer.param_groups:
                    pg["lr"] = target_lr
            if active_phase == 4 and dagger_ratio > 0.0 and self.replay_buffer is not None and len(self.replay_buffer.trajectories) > 0:
                buf_nat = [t["trajectory_id"] for t in self.replay_buffer.trajectories if t.get("is_expert") is False]
                buf_exp = [t["trajectory_id"] for t in self.replay_buffer.trajectories if t.get("is_expert") is True]
                buf_untracked = [
                    t["trajectory_id"]
                    for t in self.replay_buffer.trajectories
                    if t.get("is_expert") not in (True, False)
                ]
                untracked_desc = (
                    f" | Unannotated Legacy Trajectories (excluded from natural/expert breakdown): {buf_untracked}"
                    if buf_untracked
                    else ""
                )
                chronicle.log_info(
                    f"[DAgger Training] Epoch {epoch} training on buffer ({len(self.replay_buffer.trajectories)} trajs, ratio={dagger_ratio:.2f}) | "
                    f"Verified Natural IDs: {buf_nat} | Verified Expert IDs: {buf_exp}{untracked_desc}"
                )
        else:
            active_phase, w_t, alpha, dagger_ratio = 1, 0.60, 0.0, 0.0

        for raw_batch in self.train_loader:
            batch = dict(raw_batch)
            if self.mode in ("train_m1", "m1"):
                batch["curriculum_phase"] = active_phase
                batch["curriculum_w_t"] = w_t
                batch["curriculum_alpha"] = alpha

            if (
                active_phase == 4
                and dagger_ratio > 0.0
                and self.replay_buffer is not None
                and len(self.replay_buffer) >= min_buffer_samples
                and self.mode in ("train_m1", "m1")
            ):
                b_size = batch["events"].shape[0] if "events" in batch else 16
                dagger_size = max(1, int(round(b_size * dagger_ratio)))
                open_size = max(1, b_size - dagger_size)
                for key in ("events", "actions", "flat_kinematics", "phase_flags", "joint_positions", "joint_velocities", "trajectory_ids"):
                    if key in batch and isinstance(batch[key], torch.Tensor) and batch[key].shape[0] == b_size:
                        batch[key] = batch[key][:open_size]

                cl_batch = self.replay_buffer.sample_trajectories(
                    dagger_size, horizon=self.cfg.training.phase_horizon, device=self.device
                )
                batch["closed_loop"] = cl_batch
                n_t = cl_batch["actions"].shape[0]
                h_t = cl_batch["actions"].shape[1]
                cl_trajs_epoch += n_t
                cl_trans_epoch += (n_t * h_t)
            else:
                batch.pop("closed_loop", None)

            step_metrics = self.handler.train_step(batch, epoch=epoch)
            for k, v in step_metrics.items():
                train_accum[k] = train_accum.get(k, 0.0) + v
            batch_count += 1
            if self.ema_tracker is not None:
                self.ema_tracker.update(self.model)

        epoch_metrics = {k: v / max(1, batch_count) for k, v in train_accum.items()}
        if self.mode in ("train_m1", "m1"):
            prev_phase = self.current_curriculum_phase

            kinematic_loss = epoch_metrics.get("train_loss_position", float("inf")) + epoch_metrics.get("train_loss_velocity", float("inf"))
            next_phase = self.curriculum.end_epoch(kinematic_loss)

            self.current_curriculum_phase = next_phase
            if next_phase > prev_phase:
                chronicle.log_info(
                    f"[Latch Curriculum] Irreversible transition triggered at Epoch {epoch}: "
                    f"Phase {prev_phase} -> Phase {next_phase} (train_loss={epoch_metrics.get('train_loss', 0.0):.5f})."
                )
                self.save_checkpoint(epoch, is_best=False, phase_label=f"checkpoint_phase_{prev_phase}")

        epoch_metrics["curriculum_phase"] = float(active_phase)
        epoch_metrics["next_curriculum_phase"] = float(self.current_curriculum_phase)
        epoch_metrics["active_perception_w_t"] = float(w_t)
        epoch_metrics["train_mppi_noise_alpha"] = float(alpha)
        epoch_metrics["closed_loop_trajs_epoch"] = float(cl_trajs_epoch)
        epoch_metrics["closed_loop_trans_epoch"] = float(cl_trans_epoch)
        epoch_metrics["closed_loop_buffer_trajs"] = float(len(self.replay_buffer.trajectories)) if self.replay_buffer is not None else 0.0
        epoch_metrics["closed_loop_buffer_total"] = float(len(self.replay_buffer)) if self.replay_buffer is not None else 0.0
        epoch_metrics["dagger_ratio"] = float(dagger_ratio)
        return epoch_metrics

    def _fit_single_phase(
        self,
        epochs: Optional[int] = None,
        patience: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        total_epochs = epochs if epochs is not None else self.cfg.training.epochs
        end_epoch = self.start_epoch + total_epochs - 1
        max_no_improve = patience if patience is not None else self.cfg.training.early_stopping_patience

        # For Tier 2, early stopping is governed strictly by consecutive Tier 2 checks
        if self.tier2_evaluator is not None:
            if patience is not None:
                tier2_patience_limit = max(1, patience // self.tier2_frequency) if patience >= self.tier2_frequency else max(1, patience)
            else:
                tier2_patience_limit = self.tier2_patience
        else:
            tier2_patience_limit = max_no_improve

        chronicle.log_application_title(
            f"Starting Training: {self.cfg.project.name} [{self.mode.upper()}]"
        )
        chronicle.log_detail("Device", str(self.device))
        if self.tier2_evaluator is not None:
            chronicle.log_detail(
                "Epochs",
                f"{self.start_epoch} -> {end_epoch} (Tier 2 Patience: {tier2_patience_limit} checks / "
                f"{tier2_patience_limit * self.tier2_frequency} epochs, min_delta={self.tier2_min_delta})",
            )
            chronicle.log_detail("Tier 2 Frequency", f"Every {self.tier2_frequency} epochs")
            chronicle.log_detail("Candidate Selection", f"{self.tier2_candidate_selection} (EMA decay: {self.ema_decay})")
        else:
            chronicle.log_detail("Epochs", f"{self.start_epoch} -> {end_epoch} (Patience: {max_no_improve} epochs)")
        chronicle.log_detail("Save Directory", str(self.save_dir))

        epochs_without_improvement = 0
        tier2_checks_without_improvement = 0
        start_time = time.time()
        try:
            for epoch in range(self.start_epoch, end_epoch + 1):
                t_ep_start = time.time()
                train_metrics = self.train_epoch(epoch)
                val_metrics = self.evaluate(self.val_loader)
                tier1_loss = self.handler.get_selection_score(val_metrics)

                is_tier2_epoch = (self.tier2_evaluator is not None) and (epoch % self.tier2_frequency == 0)
                if self.tier2_evaluator is not None:
                    if is_tier2_epoch:
                        current_epoch_state = None
                        if self.tier2_candidate_selection == "ema" and self.ema_tracker is not None:
                            current_epoch_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                            self.ema_tracker.copy_to(self.model)
                            chronicle.log_info(
                                f"Tier 2 Evaluating EMA model weights (Epoch {epoch}, decay={self.ema_decay:.3f})."
                            )
                        elif self.tier2_candidate_selection == "latest":
                            chronicle.log_info(
                                f"Tier 2 Evaluating latest model weights (Epoch {epoch})."
                            )

                        active_curric_phase = int(train_metrics.get("curriculum_phase", self.current_curriculum_phase))
                        sig = inspect.signature(self.tier2_evaluator.__call__)
                        if "curriculum_phase" in sig.parameters or any(
                            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                        ):
                            tier2_metrics = self.tier2_evaluator(
                                self.model, epoch, curriculum_phase=active_curric_phase
                            )
                        else:
                            tier2_metrics = self.tier2_evaluator(self.model, epoch)
                        cl_trajectories = tier2_metrics.pop("trajectories", None)
                        cl_transitions = tier2_metrics.pop("transitions", None)
                        if self.replay_buffer is not None:
                            if cl_trajectories is not None:
                                self.replay_buffer.push_trajectories(cl_trajectories)
                            elif cl_transitions is not None:
                                self.replay_buffer.push(cl_transitions)
                            self.replay_buffer.save(self.save_dir / "closed_loop_buffer.pt")
                            if "mppi_evaluated_trajectories" not in tier2_metrics:
                                raise KeyError(
                                    "Tier 2 evaluator output is missing contractually required 'mppi_evaluated_trajectories'. "
                                    "Cannot proceed with unverified trajectory counts."
                                )
                            if "mppi_score" not in tier2_metrics:
                                raise KeyError(
                                    "Tier 2 evaluator output is missing contractually required 'mppi_score'. "
                                    "Cannot proceed with unverified evaluation score."
                                )
                            num_mppi_eval = int(tier2_metrics["mppi_evaluated_trajectories"])
                            num_expert_eval = int(tier2_metrics.get("mppi_expert_trajectories", 0))
                            expert_blocks = list(tier2_metrics.get("mppi_expert_blocks", []))
                            expert_ids = list(tier2_metrics.get("mppi_expert_trajectory_ids", []))
                            natural_blocks = list(tier2_metrics.get("mppi_natural_blocks", []))
                            natural_ids = list(tier2_metrics.get("mppi_natural_trajectory_ids", []))
                            chronicle.log_info(
                                f"[DAgger Buffer] Training set updated with {len(self.replay_buffer.trajectories)} trajectories "
                                f"({len(self.replay_buffer)} total transitions) | "
                                f"Natural [{num_mppi_eval}]: IDs={natural_ids} ({natural_blocks}) | "
                                f"Expert [{num_expert_eval}]: IDs={expert_ids} ({expert_blocks})."
                            )
                            val_metrics.update(tier2_metrics)
                            score = float(tier2_metrics["mppi_score"])
                            chronicle.log_info(
                                f"[DAgger Buffer] Training set updated with {len(self.replay_buffer.trajectories)} trajectories "
                                f"({len(self.replay_buffer)} total transitions) | "
                                f"Natural [{num_mppi_eval}]: IDs={natural_ids} ({natural_blocks}) | "
                                f"Expert [{num_expert_eval}]: IDs={expert_ids} ({expert_blocks})."
                            )
                        val_metrics.update(tier2_metrics)
                        score = tier2_metrics["mppi_score"]
                        is_best = score < (self.best_score - self.tier2_min_delta)
                        if is_best:
                            self.best_score = score
                            self.best_epoch = epoch
                            epochs_without_improvement = 0
                            tier2_checks_without_improvement = 0
                            self.save_checkpoint(epoch, is_best=True)
                        else:
                            tier2_checks_without_improvement += 1
                            epochs_without_improvement += self.tier2_frequency

                        # Restore current training weights so optimization continues forward
                        if current_epoch_state is not None:
                            self.model.load_state_dict(current_epoch_state, strict=True)
                    else:
                        score = self.best_score
                        is_best = False
                else:
                    score = tier1_loss
                    is_best = score < self.best_score
                    if is_best:
                        self.best_score = score
                        self.best_epoch = epoch
                        epochs_without_improvement = 0
                        self.save_checkpoint(epoch, is_best=True)
                    else:
                        epochs_without_improvement += 1

                ep_duration = time.time() - t_ep_start
                total_elapsed = time.time() - start_time

                log_entry = {
                    "epoch": epoch,
                    "phase": self.mode,
                    **train_metrics,
                    **val_metrics,
                    "score": score,
                    "is_best": is_best,
                    "epochs_without_improvement": epochs_without_improvement,
                    "tier2_checks_without_improvement": tier2_checks_without_improvement if self.tier2_evaluator is not None else epochs_without_improvement,
                    "duration": ep_duration,
                }
                self.history.append(log_entry)
                self.save_checkpoint(epoch, is_best=False)

                self.handler.log_epoch(
                    epoch=epoch,
                    end_epoch=end_epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    epoch_duration=ep_duration,
                    total_elapsed=total_elapsed,
                    is_best=is_best,
                    best_epoch=self.best_epoch,
                    best_score=self.best_score,
                )

                # Early stopping evaluation: strictly driven by the active validation tier
                if self.tier2_evaluator is not None:
                    if is_tier2_epoch and tier2_checks_without_improvement >= tier2_patience_limit:
                        chronicle.log_info(
                            f"Early stopping triggered in [{self.mode.upper()}] after "
                            f"{tier2_checks_without_improvement} consecutive Tier 2 evaluations ({epochs_without_improvement} epochs) "
                            f"without MPPI structural improvement (Best Score: {self.best_score:.5f} @ Epoch {self.best_epoch})."
                        )
                        break
                else:
                    if epochs_without_improvement >= max_no_improve:
                        chronicle.log_info(
                            f"Early stopping triggered in [{self.mode.upper()}] after {epochs_without_improvement} "
                            f"epochs without validation improvement (Best: {self.best_score:.5f} @ Epoch {self.best_epoch})."
                        )
                        break
        except KeyboardInterrupt:
            chronicle.log_warning("[INTERRUPT] Training interrupted manually.")

        best_model_path = self.save_dir / "model.pt"
        if not best_model_path.is_file() and self.history:
            self.save_checkpoint(self.history[-1]["epoch"], is_best=True)

        # Save final phase checkpoint
        if self.mode in ("train_m1", "m1"):
            last_epoch = self.history[-1]["epoch"] if self.history else self.start_epoch
            self.save_checkpoint(
                last_epoch,
                is_best=False,
                phase_label=f"checkpoint_phase_{self.current_curriculum_phase}",
            )

        self.save_training_log()
        chronicle.log_success(
            f"Session complete [{self.mode.upper()}]. Logs saved in '{self.save_dir}'. "
            f"Best Selection Score: {self.best_score:.5f} @ Epoch {self.best_epoch}"
        )
        return self.history

    def fit(
        self,
        epochs: Optional[int] = None,
        patience: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        return self._fit_single_phase(epochs=epochs, patience=patience)

    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        target_loader = loader if loader is not None else self.val_loader
        if target_loader is None:
            raise RuntimeError("Cannot call evaluate when val_loader is None.")
        val_accum: Dict[str, float] = {}
        batch_count = 0
        for batch in target_loader:
            step_metrics = self.handler.validate_step(batch)
            for k, v in step_metrics.items():
                val_accum[k] = val_accum.get(k, 0.0) + v
            batch_count += 1
        return {k: v / max(1, batch_count) for k, v in val_accum.items()}

    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        path: Optional[Union[str, Path]] = None,
        phase_label: Optional[str] = None,
    ) -> None:
        ckpt = {
            "epoch": epoch,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "model_state": self.model.state_dict(),
            "optimizers_state": self.handler.get_optimizers_state(),
            "config": self.cfg.to_dict(),
            "history": self.history,
            "curriculum_state": self.curriculum.state_dict(),
        }
        if self.ema_tracker is not None:
            ckpt["ema_state"] = self.ema_tracker.state_dict()
        if self.replay_buffer is not None:
            ckpt["replay_buffer"] = self.replay_buffer.state_dict()
            buffer_dest = self.save_dir / "closed_loop_buffer.pt"
            if len(self.replay_buffer) > 0 or not buffer_dest.is_file() or buffer_dest.stat().st_size <= 2500:
                self.replay_buffer.save(buffer_dest)
        if path is not None:
            torch.save(ckpt, Path(path))
            return
        if phase_label is not None:
            label_filename = f"{phase_label}.pt" if not phase_label.endswith(".pt") else phase_label
            torch.save(ckpt, self.save_dir / label_filename)
            chronicle.log_success(f"Saved phase checkpoint: {self.save_dir / label_filename}")
        if is_best:
            torch.save(ckpt, self.save_dir / "model.pt")
            torch.save(ckpt, self.save_dir / "best_model.pt")
        torch.save(ckpt, self.save_dir / "checkpoint.pt")

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load checkpoint strictly (`strict=True`). Raises RuntimeError on any key mismatch."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            self.model.load_state_dict(ckpt["model_state"], strict=True)
            self.start_epoch = ckpt.get("epoch", 0) + 1
            self.best_score = ckpt.get("best_score", float("inf"))
            self.best_epoch = ckpt.get("best_epoch", None)
            self.history = ckpt.get("history", []) or []
            if "optimizers_state" in ckpt:
                self.handler.load_optimizers_state(ckpt["optimizers_state"])
            if self.handler.predictor_optimizer is not None:
                for pg in self.handler.predictor_optimizer.param_groups:
                    pg["lr"] = self.cfg.training.learning_rate
            if "curriculum_state" in ckpt and isinstance(ckpt["curriculum_state"], dict):
                self.curriculum.load_state_dict(ckpt["curriculum_state"])
                self.current_curriculum_phase = self.curriculum.current_curriculum_phase
            else:
                import re
                m = re.search(r"checkpoint_phase_(\d+)", path.name)
                if m:
                    loaded_phase = int(m.group(1))
                    self.curriculum.current_curriculum_phase = loaded_phase
                    self.current_curriculum_phase = loaded_phase
                    chronicle.log_info(
                        f"Parsed curriculum phase {loaded_phase} from phase checkpoint filename '{path.name}'."
                    )
            if self.ema_tracker is not None and "ema_state" in ckpt:
                self.ema_tracker.load_state_dict(ckpt["ema_state"])
            if self.replay_buffer is not None:
                if "replay_buffer" in ckpt:
                    self.replay_buffer.load_state_dict(ckpt["replay_buffer"])
                else:
                    ckpt_dir = Path(path).parent
                    resolved_buf = resolve_closed_loop_buffer(
                        version_name=self.cfg.project.name,
                        phase_mode=self.mode,
                        save_dir=ckpt_dir,
                        results_root=self.results_root,
                        is_resuming=True,
                    )
                    if resolved_buf is not None:
                        self.replay_buffer.load(resolved_buf)
        else:
            self.model.load_state_dict(ckpt, strict=True)

    def save_training_log(self) -> None:
        if not self.history:
            return

        json_path = self.save_dir / "training_log.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        csv_path = self.save_dir / "training_log.csv"
        fieldnames: List[str] = []
        for rec in self.history:
            for k in rec.keys():
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.history)


class Trainer(BaseTrainer):
    """Typed Trainer constructor building an immutable ExperimentConfig when not passed explicitly."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        cfg: Optional[ExperimentConfig] = None,
        loss_fn: Optional[SPWMLoss] = None,
        mode: str = "train_m1",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        grad_clip_norm: float = 1.0,
        device: Optional[Union[str, torch.device]] = None,
        save_dir: Union[str, Path] = "results/default_run",
        probe_lr: float = 5e-4,
        probe_weight_decay: float = 1e-4,
        start_epoch: int = 1,
        best_val_loss: float = float("inf"),
        best_epoch: Optional[int] = None,
        history: Optional[List[Dict[str, float]]] = None,
        optimizer_state: Optional[Dict[str, Any]] = None,
        rollout_horizons: Optional[Sequence[int]] = None,
        joint_count: int = 6,
        action_dim: int = 6,
        phase_horizon: int = 10,
        early_stopping_patience: int = 20,
        auto_enforce_checkpoints: bool = False,
        results_root: Union[str, Path] = "results",
        tier2_evaluator: Optional[Tier2EvaluatorProtocol] = None,
        **kwargs: Any,
    ) -> None:
        if cfg is None:
            latent_dim = model.latent_dim
            q_dim = model.q_dim
            p_dim = model.p_dim
            num_kp = model.num_keypoints
            timescale_dims = model.timescale_dims
            roll_h = tuple(rollout_horizons) if rollout_horizons is not None else (1, 5, 10, 25, 50, 100)

            seq_len = max(45, 2 * phase_horizon + 5)
            cfg = ExperimentConfig(
                project=ProjectConfig(name=Path(save_dir).stem),
                environment=EnvironmentConfig(
                    height=model.height,
                    width=model.width,
                    joint_count=joint_count,
                    action_dim=action_dim,
                ),
                data=DataConfig(
                    sequence_length=seq_len,
                    joint_count=joint_count,
                    action_dim=action_dim,
                ),
                model=ModelConfig(
                    in_channels=model.in_channels,
                    encoder_dim=model.encoder_dim,
                    latent_dim=latent_dim,
                    q_dim=q_dim,
                    p_dim=p_dim,
                    num_keypoints=num_kp,
                ),
                neuron=LatentNeuronConfig(),
                memory=MemoryConfig(timescale_dims=timescale_dims),
                training=TrainingConfig(
                    mode=mode,
                    early_stopping_patience=early_stopping_patience,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    grad_clip_norm=grad_clip_norm,
                    probe_lr=probe_lr,
                    probe_weight_decay=probe_weight_decay,
                    phase_horizon=phase_horizon,
                ),
                loss=LossConfig(),
                evaluation=EvaluationConfig(rollout_horizons=roll_h),
            )

        super().__init__(
            model=model,
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            save_dir=save_dir,
            start_epoch=start_epoch,
            best_score=best_val_loss,
            best_epoch=best_epoch,
            history=history,
            auto_enforce_checkpoints=auto_enforce_checkpoints,
            results_root=results_root,
            tier2_evaluator=tier2_evaluator,
        )

        if optimizer_state is not None:
            self.handler.load_optimizers_state(optimizer_state)


__all__ = [
    "BaseTrainer",
    "ClosedLoopBuffer",
    "Trainer",
    "load_representation_checkpoint",
    "resolve_version_checkpoint",
    "resolve_closed_loop_buffer",
    "enforce_phase_prerequisites",
    "compute_dagger_ratio",
]
