"""
Scoped Strict Checkpoint Resolution & Loading Utilities (Layer 2 — Training).

Single source of truth for:
- Strict submodule state extraction (`_extract_submodel_state`)
- Phase 0 representation checkpoint loading (`load_representation_checkpoint`)
- Versioned best-validation `model.pt` resolution (`resolve_version_checkpoint`)
- Closed-loop replay buffer path resolution (`resolve_closed_loop_buffer`)
- Phase prerequisite enforcement (`enforce_phase_prerequisites`)

Zero upward imports into `spwm.evaluation` and zero silent `strict=False` fallbacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union
import chronicle
import torch
import torch.nn as nn

from spwm.contracts.config import ExperimentConfig
from spwm.contracts.version import GLOBAL_VERSION


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


__all__ = [
    "_extract_submodel_state",
    "load_representation_checkpoint",
    "resolve_version_checkpoint",
    "resolve_closed_loop_buffer",
    "enforce_phase_prerequisites",
]

