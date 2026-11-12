"""
DAgger Closed-Loop Replay Buffer & EMA Weight Tracker (Layer 2 — Training).

Contains:
- `EMATracker`: Exponential Moving Average shadow weight tracker with strict loading.
- `ClosedLoopBuffer`: DAgger-style FIFO rolling-window trajectory & transition buffer.

All `state_dict()` outputs contain strictly plain Python primitives (`dict`, `list`,
`int`, `float`, `str`, `bool`) and `torch.Tensor` objects—never class instances.
Zero upward imports into `spwm.evaluation` and zero silent fallbacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn

from spwm.contracts.states import ClosedLoopTrajectory


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
        return {str(k): v.detach().clone() for k, v in self.shadow.items()}

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
        self.trajectories: List[ClosedLoopTrajectory] = []
        self.eval_generations: List[List[ClosedLoopTrajectory]] = []
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

    def push_trajectories(
        self,
        trajectories: Sequence[Union[ClosedLoopTrajectory, Dict[str, Any]]],
    ) -> None:
        """Push a generation of closed-loop MPPI trajectories into buffer.

        Maintains a strict FIFO rolling window across Tier 2 evaluation generations.
        When new trajectories are ingested, any generation older than
        max_eval_generations is cleanly evicted to eliminate obsolete on-policy errors.
        """
        if not trajectories:
            return

        clean_gen: List[ClosedLoopTrajectory] = []
        for traj in trajectories:
            if isinstance(traj, ClosedLoopTrajectory):
                actions_t = traj.actions.detach().cpu()
                targets_t = traj.targets.detach().cpu()
                if actions_t.ndim == 3 and actions_t.shape[1] == 1:
                    actions_t = actions_t.squeeze(1)
                if targets_t.ndim == 3 and targets_t.shape[1] == 1:
                    targets_t = targets_t.squeeze(1)
                clean_traj = ClosedLoopTrajectory(
                    z_init=traj.z_init.detach().cpu().view(-1),
                    actions=actions_t,
                    targets=targets_t,
                    length=int(traj.length),
                    trajectory_id=int(traj.trajectory_id),
                    is_expert=traj.is_expert,
                    elevation_plane=str(traj.elevation_plane),
                )
            else:
                clean_traj = ClosedLoopTrajectory.from_dict(traj)
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
            curr = clean_traj.z_init
            actions = clean_traj.actions
            targets = clean_traj.targets
            for step_idx in range(clean_traj.length):
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
        H = min(horizon, min(t.length for t in sampled_trajs))

        z_inits: List[torch.Tensor] = []
        acts_list: List[torch.Tensor] = []
        targets_list: List[torch.Tensor] = []
        for t in sampled_trajs:
            L = t.length
            max_start = max(0, L - H)
            start_idx = int(torch.randint(0, max_start + 1, (1,)).item()) if max_start > 0 else 0
            z_i = (t.z_init if start_idx == 0 else t.targets[start_idx - 1]).view(-1)
            u_i = t.actions[start_idx : start_idx + H]
            tgt_i = t.targets[start_idx : start_idx + H]
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

    @staticmethod
    def _serialize_trajectory(t: Union[ClosedLoopTrajectory, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(t, ClosedLoopTrajectory):
            return t.to_dict()
        clean: Dict[str, Any] = {
            "z_init": t["z_init"].detach().cpu().clone(),
            "actions": t["actions"].detach().cpu().clone(),
            "targets": t["targets"].detach().cpu().clone(),
            "length": int(t["length"]),
            "trajectory_id": int(t["trajectory_id"]),
        }
        if "is_expert" in t:
            clean["is_expert"] = bool(t["is_expert"]) if isinstance(t["is_expert"], bool) else str(t["is_expert"])
        if "elevation_plane" in t:
            clean["elevation_plane"] = str(t["elevation_plane"])
        return clean

    @staticmethod
    def _deserialize_trajectory(t: Union[ClosedLoopTrajectory, Dict[str, Any]]) -> ClosedLoopTrajectory:
        if isinstance(t, ClosedLoopTrajectory):
            return t
        return ClosedLoopTrajectory.from_dict(t, legacy_unannotated=True)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "capacity": int(self.capacity),
            "max_eval_generations": int(self.max_eval_generations),
            "trajectory_capacity": int(self.trajectory_capacity),
            "ptr": int(self._ptr),
            "traj_ptr": int(self._traj_ptr),
            "z": torch.stack(self.z).detach().cpu() if self.z else torch.empty(0),
            "u": torch.stack(self.u).detach().cpu() if self.u else torch.empty(0),
            "z_next": torch.stack(self.z_next).detach().cpu() if self.z_next else torch.empty(0),
            "trajectories": [self._serialize_trajectory(t) for t in self.trajectories],
            "eval_generations": [[self._serialize_trajectory(t) for t in gen] for gen in self.eval_generations],
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
            self.eval_generations = [
                [self._deserialize_trajectory(t) for t in gen]
                for gen in state["eval_generations"]
            ]
            while len(self.eval_generations) > self.max_eval_generations:
                self.eval_generations.pop(0)
            self.trajectories = [t for gen in self.eval_generations for t in gen]
        elif "trajectories" in state and state["trajectories"]:
            self.trajectories = [self._deserialize_trajectory(t) for t in state["trajectories"]]
            self.eval_generations = [self.trajectories]
        else:
            self.trajectories = []
            self.eval_generations = []

    def save(self, path: Union[str, Path]) -> None:
        torch.save(self.state_dict(), Path(path))

    def load(self, path: Union[str, Path]) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Buffer checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(state)


__all__ = [
    "EMATracker",
    "ClosedLoopBuffer",
]

