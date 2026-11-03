#!/usr/bin/env python3
"""AST Anti-Fallback Scanner.

Static analysis tool to spot and prevent silent fallbacks across the codebase:
1. 3-argument `getattr(obj, name, default)`: banned in favor of explicit typed attributes.
2. Ternary `hasattr(...)` fallback expressions: banned in favor of strict protocol contracts.
3. Checkpoint loading with `load_state_dict(..., strict=False)`: banned in favor of strict=True.
4. Silent exception swallowing: `except ...: pass`, `except TypeError:`, `except AttributeError:` banned.

Usage:
    python scripts/scan_fallbacks.py [paths...]
    python scripts/scan_fallbacks.py spwm/
"""

from __future__ import annotations
import ast
from pathlib import Path
import sys
from typing import List, NamedTuple


class Violation(NamedTuple):
    file_path: Path
    line: int
    rule: str
    message: str
    snippet: str


class FallbackVisitor(ast.NodeVisitor):
    def __init__(self, file_path: Path, source_lines: List[str]) -> None:
        self.file_path = file_path
        self.source_lines = source_lines
        self.violations: List[Violation] = []

    def _get_snippet(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1].strip()
        return ""

    def visit_Call(self, node: ast.Call) -> None:
        # Rule 1: Forbid 3-argument getattr(..., ..., default)
        if isinstance(node.func, ast.Name) and node.func.id == "getattr":
            if len(node.args) >= 3:
                self.violations.append(
                    Violation(
                        file_path=self.file_path,
                        line=node.lineno,
                        rule="NO_3ARG_GETATTR",
                        message=(
                            "Forbidden 3-argument getattr(..., default) fallback detected. "
                            "Access attributes directly on the typed contract or use typing.get_origin."
                        ),
                        snippet=self._get_snippet(node.lineno),
                    )
                )

        # Rule 2: Forbid load_state_dict(..., strict=False)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "load_state_dict":
            for kw in node.keywords:
                if kw.arg == "strict" and isinstance(kw.value, ast.Constant) and kw.value.value is False:
                    self.violations.append(
                        Violation(
                            file_path=self.file_path,
                            line=node.lineno,
                            rule="NO_STRICT_FALSE_LOAD",
                            message=(
                                "Forbidden load_state_dict(..., strict=False) detected. "
                                "Checkpoints must be loaded strictly (strict=True)."
                            ),
                            snippet=self._get_snippet(node.lineno),
                        )
                    )

        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        # Rule 3: Forbid ternary hasattr(...) fallback expressions (e.g., x if hasattr(obj, 'x') else fallback)
        test_src = ast.unparse(node.test)
        if "hasattr" in test_src:
            self.violations.append(
                Violation(
                    file_path=self.file_path,
                    line=node.lineno,
                    rule="NO_HASATTR_FALLBACK",
                    message=(
                        f"Forbidden ternary hasattr fallback expression detected ('{test_src}'). "
                        "Rely on strict typed contracts and protocols instead of duck-typed fallbacks."
                    ),
                    snippet=self._get_snippet(node.lineno),
                )
            )
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        # Rule 4: Forbid silent except blocks
        if isinstance(node.type, ast.Name) and node.type.id in ("TypeError", "AttributeError"):
            self.violations.append(
                Violation(
                    file_path=self.file_path,
                    line=node.lineno,
                    rule=f"NO_SILENT_EXCEPT_{node.type.id.upper()}",
                    message=(
                        f"Forbidden silent 'except {node.type.id}:' fallback block detected. "
                        "Validate parameters up-front or propagate contract exceptions."
                    ),
                    snippet=self._get_snippet(node.lineno),
                )
            )

        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            self.violations.append(
                Violation(
                    file_path=self.file_path,
                    line=node.lineno,
                    rule="NO_EXCEPT_PASS",
                    message=(
                        "Forbidden 'except ...: pass' block detected. "
                        "Never silently swallow exceptions; propagate or raise with context."
                    ),
                    snippet=self._get_snippet(node.lineno),
                )
            )

        self.generic_visit(node)


def scan_file(file_path: Path) -> List[Violation]:
    """Scan a single Python file for anti-fallback violations."""
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError) as e:
        return [
            Violation(
                file_path=file_path,
                line=1,
                rule="PARSE_ERROR",
                message=f"Failed to parse {file_path}: {e}",
                snippet="",
            )
        ]

    visitor = FallbackVisitor(file_path=file_path, source_lines=source.splitlines())
    visitor.visit(tree)
    return visitor.violations


def scan_directory(target_dir: Path) -> List[Violation]:
    """Recursively scan directory for anti-fallback violations."""
    all_violations: List[Violation] = []
    for py_file in sorted(target_dir.rglob("*.py")):
        # Skip git, venv, and cache dirs
        if any(part in py_file.parts for part in (".git", ".venv", "__pycache__", "build", "dist")):
            continue
        all_violations.extend(scan_file(py_file))
    return all_violations


def main() -> int:
    targets = sys.argv[1:]
    if not targets:
        repo_root = Path(__file__).resolve().parent.parent
        targets = [str(repo_root / "spwm")]

    violations: List[Violation] = []
    for target_str in targets:
        target_path = Path(target_str)
        if target_path.is_file():
            violations.extend(scan_file(target_path))
        elif target_path.is_dir():
            violations.extend(scan_directory(target_path))
        else:
            print(f"Error: Target path does not exist: {target_str}", file=sys.stderr)
            return 2

    if not violations:
        print(f"AST Anti-Fallback Scan PASSED: Clean. Zero fallbacks detected across {len(targets)} target(s).")
        return 0

    print(f"\n[!] AST Anti-Fallback Scan FAILED: {len(violations)} violation(s) found:\n")
    for v in violations:
        rel = v.file_path
        try:
            rel = v.file_path.relative_to(Path.cwd())
        except ValueError:
            pass
        print(f"  {rel}:{v.line}: [{v.rule}]")
        print(f"    Message: {v.message}")
        if v.snippet:
            print(f"    Code:    {v.snippet}")
        print()

    return 1


if __name__ == "__main__":
    sys.exit(main())

