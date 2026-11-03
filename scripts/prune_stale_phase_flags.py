#!/usr/bin/env python3
"""
Prunes stale Phase 1 (Sensory Heft) flags across exported Unity trajectories (SPWM-v0.7.2).

Sanitizes all trajectory_*.json.gz files on disk:
  1. Identifies the original contact step: t_contact = t_old_start + 5
     (where t_old_start was based on pre_contact_steps = 5 in v0.7.0).
  2. Applies the updated pre_contact_sensory_steps = 7 and post_contact_sensory_steps = 3:
     t_inj_start = max(1, t_contact - 7)
     t_inj_end   = min(T - 1, t_contact + 3)
  3. Overwrites phase_flags:
     - 0 for t < t_inj_start
     - 1 for t_inj_start <= t < t_inj_end (10 steps)
     - 2 for t >= t_inj_end
  4. Updates both doc["phase_flags"] and doc["steps"][t]["phase_flag"].
  5. Preserves 100% of the underlying physical kinematics, randomize seeds, and sensor frames.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def sanitize_trajectory_file(
    file_path: str,
    pre_contact_sensory_steps: int = 7,
    post_contact_sensory_steps: int = 3,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Sanitize a single trajectory .json.gz file."""
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        doc = json.load(f)

    steps = doc.get("steps", [])
    T = len(steps)
    if T == 0:
        return {"processed": 1, "modified": 0, "stale_found": 0, "new_p1_len": 0}

    # Extract old flags
    old_flags = [int(s.get("phase_flag", 0)) for s in steps]
    old_start = next((idx for idx, flg in enumerate(old_flags) if flg == 1), -1)
    old_p1_count = old_flags.count(1)

    if old_start == -1:
        # No phase 1 found; compute from default
        t_contact = max(2, int(round(0.38 * T)))
    else:
        # Original contact was at old_start + 5
        t_contact = old_start + 5

    t_inj_start = max(1, t_contact - pre_contact_sensory_steps)
    t_inj_end = min(T - 1, t_contact + post_contact_sensory_steps)

    new_flags = [0] * T
    for t in range(t_inj_start, t_inj_end):
        new_flags[t] = 1
    for t in range(t_inj_end, T):
        new_flags[t] = 2

    new_p1_len = new_flags.count(1)
    stale_found = 1 if old_p1_count == 17 else 0

    if new_flags != old_flags:
        if not dry_run:
            doc["phase_flags"] = new_flags
            for t, s in enumerate(steps):
                s["phase_flag"] = new_flags[t]
            with gzip.open(file_path, "wt", encoding="utf-8") as f:
                json.dump(doc, f, separators=(",", ":"))
        return {"processed": 1, "modified": 1, "stale_found": stale_found, "new_p1_len": new_p1_len}

    return {"processed": 1, "modified": 0, "stale_found": stale_found, "new_p1_len": new_p1_len}


def _worker_fn(args: Tuple[str, int, int, bool]) -> Dict[str, int]:
    path, pre_c, post_c, dry = args
    return sanitize_trajectory_file(path, pre_contact_sensory_steps=pre_c, post_contact_sensory_steps=post_c, dry_run=dry)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanitize Phase 1 sensory flags across cached Unity trajectories.")
    parser.add_argument(
        "--trajectories-dir",
        type=str,
        default="sim/ExportedTrajectories",
        help="Directory containing exported trajectory_*.json.gz files.",
    )
    parser.add_argument(
        "--pre-contact-steps",
        type=int,
        default=7,
        help="Number of pre-contact steps for Phase 1 (default: 7).",
    )
    parser.add_argument(
        "--post-contact-steps",
        type=int,
        default=3,
        help="Number of post-contact steps for Phase 1 (default: 3).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report without overwriting files.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number of parallel worker processes.",
    )
    args = parser.parse_args()

    traj_dir = Path(args.trajectories_dir)
    pattern = str(traj_dir / "trajectory_*.json.gz")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[!] No trajectory files found matching: {pattern}")
        return 1

    print(f"[*] Found {len(files)} trajectory files in {traj_dir}")
    print(f"[*] Target schedule: [t_contact - {args.pre_contact_steps}, t_contact + {args.post_contact_steps}]")
    print(f"[*] Parallel workers: {args.num_workers} (Dry run: {args.dry_run})")

    tasks = [(p, args.pre_contact_steps, args.post_contact_steps, args.dry_run) for p in files]

    if args.num_workers > 1:
        with mp.Pool(processes=args.num_workers) as pool:
            results = pool.map(_worker_fn, tasks)
    else:
        results = [_worker_fn(t) for t in tasks]

    total_proc = sum(r["processed"] for r in results)
    total_mod = sum(r["modified"] for r in results)
    stale_found = sum(r["stale_found"] for r in results)
    p1_lens = [r["new_p1_len"] for r in results]

    print("\n--- Sanitization Summary ---")
    print(f"Total files processed: {total_proc}")
    print(f"Files modified:        {total_mod}")
    print(f"Stale 17-frame blocks: {stale_found}")
    print(f"New Phase 1 durations: min={min(p1_lens)}, max={max(p1_lens)}, mean={sum(p1_lens)/len(p1_lens):.2f}")
    print("----------------------------")
    if not args.dry_run:
        print("[✓] All trajectory files successfully sanitized on disk.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

