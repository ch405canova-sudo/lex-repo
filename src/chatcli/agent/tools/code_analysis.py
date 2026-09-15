"""Code analysis tool: static analysis (mypy, bandit, pylint, radon, vulture).

With this, Lex can check its own code for type errors, security issues,
complexity and dead code — without burning shell rounds.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from .base import Tool, ToolResult

_TIMEOUT = 120  # Sekunden


def _build_cmd(tool: str, path: str) -> list[str]:
    """Builds the CLI command list for the given tool."""
    target = path or "."
    match tool:
        case "mypy":
            return ["uv", "run", "mypy", "--strict", target]
        case "bandit":
            return ["uv", "run", "bandit", "-r", target, "-f", "txt"]
        case "pylint":
            return ["uv", "run", "pylint", "--score=y", target]
        case "radon":
            # cc = cognitive complexity, -a = alle Funktionen, -s = sortiert
            return ["uv", "run", "radon", "cc", "-a", "-s", target]
        case "vulture":
            return ["uv", "run", "vulture", "--min-confidence", "60", target]
        case _:
            raise ValueError(f"Unknown tool: '{tool}'. Possible: mypy, bandit, pylint, radon, vulture")


class CodeAnalysisTool(Tool):
    """Runs static code analysis tools on the Lex codebase.

    Available tools:
    - mypy:     Static type checking (finds runtime bugs)
    - bandit:   Security linter (finds insecure patterns)
    - pylint:   Code quality score + error list
    - radon:    Cognitive complexity per function (A–F scale)
    - vulture:  Dead code (unused imports, methods, variables)

    Recommended usage:
    1. After code changes: ``code_analysis(tool="mypy")`` for type check
    2. Before commits: ``code_analysis(tool="bandit")`` + ``code_analysis(tool="vulture")``
    3. Architecture review: ``code_analysis(tool="radon", path="src/chatcli/agent/loop.py")``
    4. Quality check: ``code_analysis(tool="pylint")``
    """

    name = "code_analysis"
    description = (
        "Static code analysis on the Lex codebase. "
        "Tool: mypy|bandit|pylint|radon|vulture. "
        "Path optional (default: src/chatcli/)."
    )
    args_schema: dict[str, Any] = {
        "tool": "str — which analysis tool: mypy, bandit, pylint, radon or vulture",
        "path": "str optional — target path relative to chatcli root (default: src/chatcli/)",
    }

    def __init__(self, project_root: str = ".") -> None:
        """project_root: absolute path to the chatcli project (where pyproject.toml is)."""
        self._root = Path(project_root).resolve()

    async def run(self, **kwargs: Any) -> ToolResult:
        tool_name: str = kwargs.get("tool", "")
        target: str = kwargs.get("path", "src/chatcli/")

        if not tool_name:
            return ToolResult(ok=False, output="", error="missing tool parameter (mypy|bandit|pylint|radon|vulture)")

        try:
            cmd = _build_cmd(tool_name, target)
        except ValueError as e:
            return ToolResult(ok=False, output="", error=str(e))

        # Pfad relativ zum Projekt-Root auflösen
        abs_target = (self._root / target).resolve()
        if not abs_target.exists():
            return ToolResult(
                ok=False,
                output="",
                error=f"Path '{target}' does not exist under {self._root}",
            )

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self._root),
                env={**os.environ, "LC_ALL": "C"},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=_TIMEOUT)
            output = out.decode(errors="replace").strip()
            code = proc.returncode or 0

            # Ausgabe begrenzen (LLM-Kontext sparen)
            max_chars = 8000
            if len(output) > max_chars:
                output = output[:max_chars] + f"\n… [cut off, {len(output)} chars total]"

            prefix = f"[{tool_name}] exit_code={code}\n"
            return ToolResult(ok=(code == 0), output=prefix + output)

        except asyncio.TimeoutError:
            return ToolResult(ok=False, output="", error=f"{tool_name} timeout after {_TIMEOUT}s")
        except OSError as e:
            return ToolResult(ok=False, output="", error=f"OSError in {tool_name}: {e}")
