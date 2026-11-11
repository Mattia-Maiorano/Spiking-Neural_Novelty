"""
Layer 0 (`spwm.contracts.version`): Global Version Solver & Version Contracts.
Single source of truth (SSOT) for architecture and pipeline version resolution.
Zero upward dependencies.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Optional, Tuple, Union


@dataclass(frozen=True)
class VersionSolver:
    """Global Version Solver for SPWM.

    Encapsulates version resolution based upon (version, minor, patch),
    providing strict validation and standardized representations:
    - semver: "0.7.3"
    - short_tag: "v0.7.3"
    - formatted / tag: "SPWM-v0.7.3"
    - snake_case: "spwm_v0_7_3"
    - canonical path resolvers for checkpoints, configs, and metrics across training phases.
    """

    version: int = 0
    minor: int = 11
    patch: int = 0
    prefix: str = "SPWM"

    def __post_init__(self) -> None:
        if not isinstance(self.version, int) or self.version < 0:
            raise ValueError(f"Version (major) must be a non-negative integer, got {self.version!r}")
        if not isinstance(self.minor, int) or self.minor < 0:
            raise ValueError(f"Minor must be a non-negative integer, got {self.minor!r}")
        if not isinstance(self.patch, int) or self.patch < 0:
            raise ValueError(f"Patch must be a non-negative integer, got {self.patch!r}")
        if not isinstance(self.prefix, str) or not self.prefix.strip():
            raise ValueError(f"Prefix must be a non-empty string, got {self.prefix!r}")

    @property
    def major(self) -> int:
        """Alias for version (major version number)."""
        return self.version

    @property
    def semver(self) -> str:
        """Standard semantic version string, e.g. '0.7.3'."""
        return f"{self.version}.{self.minor}.{self.patch}"

    @property
    def short_tag(self) -> str:
        """Prefixed version tag, e.g. 'v0.7.3'."""
        return f"v{self.semver}"

    @property
    def tag(self) -> str:
        """Full formatted tag, e.g. 'SPWM-v0.7.3'."""
        return f"{self.prefix}-{self.short_tag}"

    @property
    def formatted(self) -> str:
        """Canonical formatted string alias for tag, e.g. 'SPWM-v0.7.3'."""
        return self.tag

    @property
    def snake_case(self) -> str:
        """Snake-case directory tag, e.g. 'spwm_v0_7_3'."""
        return f"{self.prefix.lower()}_v{self.version}_{self.minor}_{self.patch}"

    @property
    def tuple(self) -> Tuple[int, int, int]:
        """Version components as (version, minor, patch)."""
        return (self.version, self.minor, self.patch)

    def format(self, prefix: Optional[str] = None, suffix: str = "") -> str:
        """Format version with a custom prefix or suffix."""
        p = prefix if prefix is not None else f"{self.prefix}-v"
        return f"{p}{self.semver}{suffix}"

    def m1_dir_name(self) -> str:
        """Directory name for M1 training phase, e.g. 'spwm_v0_7_3_train_m1'."""
        return f"{self.snake_case}_train_m1"

    def m1_dir(self, results_dir: Union[Path, str] = "results") -> Path:
        """Directory path for M1 training results."""
        return Path(results_dir) / self.m1_dir_name()

    def m1_checkpoint(self, results_dir: Union[Path, str] = "results") -> Path:
        """M1 model checkpoint path."""
        return self.m1_dir(results_dir) / "model.pt"

    def m1_config(self, results_dir: Union[Path, str] = "results") -> Path:
        """M1 experiment config path."""
        return self.m1_dir(results_dir) / "config.yaml"

    def m1_metrics(self, results_dir: Union[Path, str] = "results") -> Path:
        """Canonical M1 metrics JSON path."""
        return self.m1_dir(results_dir) / f"metrics_{self.snake_case}.json"

    def representation_dir_name(self) -> str:
        """Directory name for representation training phase, e.g. 'spwm_v0_7_3_train_representation'."""
        return f"{self.snake_case}_train_representation"

    def representation_dir(self, results_dir: Union[Path, str] = "results") -> Path:
        """Directory path for representation training results."""
        return Path(results_dir) / self.representation_dir_name()

    def representation_checkpoint(self, results_dir: Union[Path, str] = "results") -> Path:
        """Representation model checkpoint path."""
        return self.representation_dir(results_dir) / "model.pt"

    def representation_config(self, results_dir: Union[Path, str] = "results") -> Path:
        """Representation experiment config path."""
        return self.representation_dir(results_dir) / "config.yaml"

    @classmethod
    def from_string(cls, version_str: str, default_prefix: str = "SPWM") -> VersionSolver:
        """Strictly parse a version string into a VersionSolver instance.

        Supported patterns:
        - '0.7.3'
        - 'v0.7.3'
        - 'SPWM-v0.7.3'
        - 'spwm_v0_7_3'
        - 'spwm_v0_7_3_train_m1'

        Raises:
            ValueError: If the string format cannot be strictly parsed.
        """
        if not isinstance(version_str, str) or not version_str.strip():
            raise ValueError(f"Version string must be a non-empty string, got {version_str!r}")
        s = version_str.strip()

        # Try 'spwm_v0_7_3' or 'spwm_v0_7_3_train_m1'
        snake_m = re.match(r"^([a-zA-Z0-9]+)_v(\d+)_(\d+)_(\d+)(?:_.*)?$", s)
        if snake_m:
            prefix = snake_m.group(1).upper()
            return cls(
                version=int(snake_m.group(2)),
                minor=int(snake_m.group(3)),
                patch=int(snake_m.group(4)),
                prefix=prefix,
            )

        # Try 'SPWM-v0.7.3'
        tag_m = re.match(r"^([a-zA-Z0-9]+)-v(\d+)\.(\d+)\.(\d+)$", s)
        if tag_m:
            prefix = tag_m.group(1)
            return cls(
                version=int(tag_m.group(2)),
                minor=int(tag_m.group(3)),
                patch=int(tag_m.group(4)),
                prefix=prefix,
            )

        # Try 'v0.7.3'
        short_m = re.match(r"^v(\d+)\.(\d+)\.(\d+)$", s)
        if short_m:
            return cls(
                version=int(short_m.group(1)),
                minor=int(short_m.group(2)),
                patch=int(short_m.group(3)),
                prefix=default_prefix,
            )

        # Try '0.7.3'
        semver_m = re.match(r"^(\d+)\.(\d+)\.(\d+)$", s)
        if semver_m:
            return cls(
                version=int(semver_m.group(1)),
                minor=int(semver_m.group(2)),
                patch=int(semver_m.group(3)),
                prefix=default_prefix,
            )

        raise ValueError(
            f"Cannot parse version string '{version_str}'. Expected semver (e.g. '0.7.3'), "
            f"tagged (e.g. 'SPWM-v0.7.3', 'v0.7.3'), or snake-case (e.g. 'spwm_v0_7_3')."
        )

    @classmethod
    def from_tuple(cls, t: Tuple[int, int, int], prefix: str = "SPWM") -> VersionSolver:
        """Construct VersionSolver from a (version, minor, patch) tuple."""
        if not isinstance(t, (tuple, list)) or len(t) != 3:
            raise ValueError(f"Version tuple must contain exactly 3 integers (version, minor, patch), got {t!r}")
        return cls(version=int(t[0]), minor=int(t[1]), patch=int(t[2]), prefix=prefix)


# Global SSOT Version Instance
GLOBAL_VERSION = VersionSolver()
GLOBAL_VERSION_SOLVER = GLOBAL_VERSION
VERSION_SOLVER = GLOBAL_VERSION


def main(argv: Optional[list[str]] = None) -> None:
    """CLI Entrypoint for shell scripts and external queries."""
    parser = argparse.ArgumentParser(description="Global Version Solver for SPWM.")
    parser.add_argument(
        "--format",
        choices=[
            "semver",
            "tag",
            "formatted",
            "short",
            "snake",
            "m1-checkpoint",
            "m1-config",
            "m1-metrics",
            "m1-dir",
            "rep-checkpoint",
            "rep-config",
            "rep-dir",
        ],
        default="formatted",
        help="Output format to resolve (default: formatted)",
    )
    parser.add_argument("--version", type=int, default=None, help="Override version (major)")
    parser.add_argument("--minor", type=int, default=None, help="Override minor")
    parser.add_argument("--patch", type=int, default=None, help="Override patch")
    parser.add_argument("--results-dir", type=str, default="results", help="Results directory path")
    args = parser.parse_args(argv)

    v = GLOBAL_VERSION
    if args.version is not None or args.minor is not None or args.patch is not None:
        v_num = args.version if args.version is not None else v.version
        m_num = args.minor if args.minor is not None else v.minor
        p_num = args.patch if args.patch is not None else v.patch
        v = VersionSolver(version=v_num, minor=m_num, patch=p_num, prefix=v.prefix)

    if args.format in ("tag", "formatted"):
        print(v.formatted)
    elif args.format == "semver":
        print(v.semver)
    elif args.format == "short":
        print(v.short_tag)
    elif args.format == "snake":
        print(v.snake_case)
    elif args.format == "m1-checkpoint":
        print(v.m1_checkpoint(args.results_dir))
    elif args.format == "m1-config":
        print(v.m1_config(args.results_dir))
    elif args.format == "m1-metrics":
        print(v.m1_metrics(args.results_dir))
    elif args.format == "m1-dir":
        print(v.m1_dir(args.results_dir))
    elif args.format == "rep-checkpoint":
        print(v.representation_checkpoint(args.results_dir))
    elif args.format == "rep-config":
        print(v.representation_config(args.results_dir))
    elif args.format == "rep-dir":
        print(v.representation_dir(args.results_dir))


if __name__ == "__main__":
    main(sys.argv[1:])

