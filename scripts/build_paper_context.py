#!/usr/bin/env python3
"""Build a lossless, token-efficient PAPER_CONTEXT.md from SSOT metrics and docs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "PAPER_CONTEXT.md"

SECTION_1_FILES: list[tuple[str, str]] = [
    (
        "results/corl_final_benchmark_summary.md",
        "CoRL final grounded 3-seed, 50-episode PIL, 5-stage ablation, equalized baselines, robustness, and energy summary + LaTeX tables",
    ),
    (
        "results/spwm_v0_12_4_train_m1/comparative_metrics_v0_12_4.json",
        "Multi-seed N=400 open-loop baseline comparison (SPWM, ANN-Rate, Continuous RSSM, and equalized Deep GRU across 3 seeds + ConvLSTM)",
    ),
    (
        "results/multiseed/multiseed_metrics.json",
        "3-seed (42, 101, 2024) SPWM, ANN-Rate, Continuous RSSM, and equalized Deep GRU open-loop, 50-ep PIL (open-loop, ungated, residual-gated), 5-stage ablation, and energy summary",
    ),
    (
        "results/diagnostics/forensic_rssm_gru_failure.json",
        "Forensic audit of RSSM vs. equalized Deep GRU across Open-Loop, Un-Gated Online (w_t=1.0), and Residual-Gated Online PIL regimes",
    ),
    (
        "results/energy_profiling/energy_profiling_metrics.json",
        "Neuromorphic SOPs (Loihi-2 standby recurrent core) vs. GPU MACs (Jetson Orin E2E MPPI) profiling across SPWM and baselines",
    ),
    (
        "results/baselines/classical_and_bc_pil/classical_bc_pil_metrics.json",
        "50-episode PIL benchmarks for Privileged Fixed-Gain PD and Reactive BC",
    ),
    (
        "results/spwm_v0_12_4_train_m1/metrics_spwm_v0_12_4.json",
        "Phase 1 rollout, horizon metrics, training progression, and MPPI progression",
    ),
    (
        "results/spwm_v0_12_4_train_m1/diagnostics.json",
        "ALIF firing-rate histogram, silent/hyperactive fractions, and gradient diagnostics",
    ),
    (
        "results/ssot_eval_metrics.json",
        "N=400 open-loop vs. phasic handover curves and analytical energy estimates",
    ),
    (
        "results/mppi_benchmark/mppi_benchmark_metrics.json",
        "50-episode closed-loop Physics-in-the-Loop (PIL, t >= t_c) MPPI benchmark breakdown",
    ),
    (
        "results/robustness_benchmark/robustness_metrics.json",
        "400-trajectory sensor sweep and 50-episode/condition PIL MPPI stress tests",
    ),
]

SECTION_2_FILES: list[tuple[str, str]] = [
    (
        "docs_verified/README.md",
        "Verified master overview, CoRL grounded PIL benchmarks, 3-seed validation, and documentation index",
    ),
    ("docs_verified/01_SYSTEM_AND_ARCHITECTURE.md", "Verified system, Unity PhysX environment, and neural architecture specification"),
    ("docs_verified/02_TRAINING_CURRICULUM_AND_MPPI.md", "Verified loss formulations, 2-phase/4-latch curriculum, MPPI controls, and equalized baseline training regimes"),
    ("docs_verified/03_BENCHMARKS_EVIDENCE_AND_REPLICATION.md", "Verified SSOT benchmarks, residual-gated baseline fairness, 3-seed ablation, neuromorphic standby energy framing, and replication commands"),
    ("results/spwm_v0_12_4_train_m1/config.yaml", "Exact hyperparameters used for the v0.12.4 run"),
]


class _TupleSafeLoader(yaml.SafeLoader):
    """YAML SafeLoader supporting !!python/tuple tags."""


def _construct_python_tuple(loader: yaml.SafeLoader, node: yaml.SequenceNode) -> list[Any]:
    return loader.construct_sequence(node)


_TupleSafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_python_tuple)


def _is_primitive(val: Any) -> bool:
    return val is None or isinstance(val, (str, int, float, bool))


def _is_primitive_list(val: Any) -> bool:
    return isinstance(val, list) and all(_is_primitive(x) for x in val)


def _format_scalar(val: Any) -> str:
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return repr(val)
    if isinstance(val, str):
        cleaned = val.replace("\r\n", "\n").replace("\n", "\\n").replace("|", "\\|")
        return cleaned
    raise TypeError(f"Expected scalar value, got {type(val).__name__}")


def _format_inline(val: Any) -> str:
    if _is_primitive(val):
        return _format_scalar(val)
    if isinstance(val, list):
        return "[" + ",".join(_format_inline(x) for x in val) + "]"
    if isinstance(val, dict):
        items = [f"{k}:{_format_inline(v)}" for k, v in val.items()]
        return "{" + ",".join(items) + "}"
    raise TypeError(f"Unsupported type for inline formatting: {type(val).__name__}")


def _is_flat_dict(val: Any) -> bool:
    if not isinstance(val, dict) or not val:
        return False
    return all(_is_primitive(v) or _is_primitive_list(v) and len(v) <= 8 for v in val.values())


def _homogeneous_dict_keys(items: list[Any]) -> list[str] | None:
    if not items or not all(isinstance(x, dict) for x in items):
        return None
    keys: list[str] = []
    seen: set[str] = set()
    for item in items:
        for k in item:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


def _can_render_as_table(items: list[Any]) -> list[str] | None:
    keys = _homogeneous_dict_keys(items)
    if keys is None or len(items) < 2:
        return None
    for item in items:
        for k in keys:
            if k in item:
                v = item[k]
                if isinstance(v, dict) and not _is_flat_dict(v):
                    return None
                if isinstance(v, list) and not _is_primitive_list(v):
                    return None
    return keys


def _render_table(keys: list[str], rows: list[dict[str, Any]], indent: str = "") -> list[str]:
    lines = [
        indent + "|" + "|".join(keys) + "|",
        indent + "|" + "|".join("---" for _ in keys) + "|",
    ]
    for row in rows:
        cells = [_format_inline(row[k]) if k in row else "" for k in keys]
        lines.append(indent + "|" + "|".join(cells) + "|")
    return lines


def _render_evaluations_history(history: list[dict[str, Any]], indent: str) -> list[str]:
    """Losslessly split nested evaluations_history into 3 compact tables."""
    summary_keys = [
        k for k in history[0] if k not in ("elevation_breakdown", "trials")
    ]
    lines: list[str] = [f"{indent}evaluations_history_summary ({len(history)} epochs):"]
    lines.extend(_render_table(summary_keys, history, indent=indent + "  "))

    elev_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    for entry in history:
        epoch = entry["epoch"]
        for tier, stats in entry["elevation_breakdown"].items():
            row = {"epoch": epoch, "tier": tier}
            row.update(stats)
            elev_rows.append(row)
        for trial in entry["trials"]:
            row = {"epoch": epoch}
            row.update(trial)
            trial_rows.append(row)

    elev_keys = _homogeneous_dict_keys(elev_rows)
    if elev_keys is None:
        raise ValueError("elevation_breakdown produced empty rows")
    lines.append(f"{indent}evaluations_history_elevation_breakdown ({len(elev_rows)} rows):")
    lines.extend(_render_table(elev_keys, elev_rows, indent=indent + "  "))

    trial_keys = _homogeneous_dict_keys(trial_rows)
    if trial_keys is None:
        raise ValueError("trials produced empty rows")
    lines.append(f"{indent}evaluations_history_trials ({len(trial_rows)} rows):")
    lines.extend(_render_table(trial_keys, trial_rows, indent=indent + "  "))
    return lines


def _render_structured(data: Any, indent_level: int = 0) -> list[str]:
    indent = "  " * indent_level
    lines: list[str] = []

    if isinstance(data, dict):
        for k, v in data.items():
            if k == "evaluations_history" and isinstance(v, list) and v and "trials" in v[0]:
                lines.extend(_render_evaluations_history(v, indent))
            elif _is_primitive(v):
                lines.append(f"{indent}{k}: {_format_scalar(v)}")
            elif _is_primitive_list(v):
                lines.append(f"{indent}{k}: {_format_inline(v)}")
            elif isinstance(v, dict):
                if _is_flat_dict(v) and len(_format_inline(v)) <= 160:
                    lines.append(f"{indent}{k}: {_format_inline(v)}")
                else:
                    lines.append(f"{indent}{k}:")
                    lines.extend(_render_structured(v, indent_level + 1))
            elif isinstance(v, list):
                table_keys = _can_render_as_table(v)
                if table_keys is not None:
                    lines.append(f"{indent}{k} ({len(v)} rows):")
                    lines.extend(_render_table(table_keys, v, indent=indent + "  "))
                else:
                    lines.append(f"{indent}{k} ({len(v)} items):")
                    for idx, item in enumerate(v):
                        if _is_primitive(item) or _is_primitive_list(item):
                            lines.append(f"{indent}  - {_format_inline(item)}")
                        else:
                            lines.append(f"{indent}  - [{idx}]")
                            lines.extend(_render_structured(item, indent_level + 2))
            else:
                raise TypeError(f"Unsupported value type for key {k}: {type(v).__name__}")
        return lines

    if isinstance(data, list):
        table_keys = _can_render_as_table(data)
        if table_keys is not None:
            lines.extend(_render_table(table_keys, data, indent=indent))
            return lines
        for idx, item in enumerate(data):
            if _is_primitive(item) or _is_primitive_list(item):
                lines.append(f"{indent}- {_format_inline(item)}")
            else:
                lines.append(f"{indent}- [{idx}]")
                lines.extend(_render_structured(item, indent_level + 1))
        return lines

    lines.append(f"{indent}{_format_scalar(data)}")
    return lines


def _compact_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    collapsed = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return collapsed.strip() + "\n"


def _count_scalars(obj: Any) -> int:
    if _is_primitive(obj):
        return 1
    if isinstance(obj, list):
        return sum(_count_scalars(x) for x in obj)
    if isinstance(obj, dict):
        return sum(_count_scalars(v) for v in obj.values())
    raise TypeError(f"Unexpected type in scalar count: {type(obj).__name__}")


def _render_file_block(rel_path: str, desc: str) -> tuple[str, int, int]:
    full_path = REPO_ROOT / rel_path
    if not full_path.exists():
        raise FileNotFoundError(f"Required source file missing: {full_path}")

    raw_text = full_path.read_text(encoding="utf-8")
    raw_bytes = len(raw_text.encode("utf-8"))

    header = f"### `{rel_path}`\n> {desc}\n\n"
    suffix = full_path.suffix.lower()

    if suffix == ".md":
        body = _compact_markdown(raw_text)
        rendered = header + body
    elif suffix == ".json":
        data = json.loads(raw_text)
        body = "\n".join(_render_structured(data)) + "\n"
        rendered = header + "```yaml\n" + body + "```\n"
    elif suffix in (".yaml", ".yml"):
        data = yaml.load(raw_text, Loader=_TupleSafeLoader)
        body = "\n".join(_render_structured(data)) + "\n"
        rendered = header + "```yaml\n" + body + "```\n"
    else:
        raise ValueError(f"Unsupported file extension {suffix} for {rel_path}")

    return rendered, raw_bytes, len(rendered.encode("utf-8"))


def build_paper_context() -> None:
    parts: list[str] = [
        "# PAPER_CONTEXT — SPWM-v0.12.4 Complete SSOT Metrics, Architecture & Training Context\n",
        "> Lossless, token-compacted compilation of all SSOT benchmark JSONs, configuration YAML, and architecture/training/world documentation.\n",
        "> JSON/YAML files are compacted into schema-aligned Markdown tables and key-value blocks without dropping any fields or values.\n\n",
        "## 1. SSOT Metrics & Benchmark Data\n\n",
    ]

    total_raw_bytes = 0
    for rel_path, desc in SECTION_1_FILES:
        block, raw_b, out_b = _render_file_block(rel_path, desc)
        total_raw_bytes += raw_b
        parts.append(block + "\n")
        print(f"  {rel_path}: {raw_b:,} B -> {out_b:,} B ({100.0 * out_b / raw_b:.1f}%)")

    parts.append("## 2. Architecture, Training & Environment Context\n\n")
    for rel_path, desc in SECTION_2_FILES:
        block, raw_b, out_b = _render_file_block(rel_path, desc)
        total_raw_bytes += raw_b
        parts.append(block + "\n")
        print(f"  {rel_path}: {raw_b:,} B -> {out_b:,} B ({100.0 * out_b / raw_b:.1f}%)")

    output_text = "".join(parts)
    OUTPUT_PATH.write_text(output_text, encoding="utf-8")
    total_out_bytes = len(output_text.encode("utf-8"))
    print(
        f"\nWrote {OUTPUT_PATH} ({total_out_bytes:,} B vs {total_raw_bytes:,} B raw; "
        f"{100.0 * (1.0 - total_out_bytes / total_raw_bytes):.1f}% reduction)"
    )


if __name__ == "__main__":
    build_paper_context()
