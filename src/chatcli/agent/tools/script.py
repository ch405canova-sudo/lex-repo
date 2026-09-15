"""Script tool: Programmatic Tool Calling.

Instead of driving N individual tool calls over N inference passes, the model
writes ONE script (bash or Python) that orchestrates multiple operations in
a single execution (loops, conditionals, pipes).
Result: fewer steps, less context blowup, more deterministic intermediates.

Execution:
- ``lang='bash'``  → ``bash --noprofile --norc -c <code>``
- ``lang='python'`` → ``python3 -c <code>``

Both run in the sandbox (project CWD), with timeout and output
truncation. Bash scripts go through the same guardrails as the
shell tool (hard gates, no confirmation needed — the script is code
that the user sees in context; destructive single commands remain
blocked).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from .base import Tool, ToolResult
from .guardrails import check_command

_MAX_OUTPUT = 16 * 1024


class ScriptTool(Tool):
    name = "script"
    description = (
        "Executes a bash or Python script ONCE (Programmatic Tool "
        "Calling). Use it when multiple operations logically belong together "
        "(loops, conditionals, pipes, bulk checks) — instead of 10 individual "
        "shell calls. Example: 'Check 5 files for pattern X' as ONE "
        "script. stdout+stderr+exit_code come back."
    )
    args_schema: dict[str, Any] = {
        "code": "str — the bash or Python script",
        "lang": "str (optional, default 'bash') — 'bash' or 'python'",
        "timeout": "int (optional, default 120) — timeout in seconds",
    }

    def __init__(self, cwd: str = ".") -> None:
        self._base = cwd

    async def run(self, **kwargs: Any) -> ToolResult:
        code: str = str(kwargs.get("code", "")).strip()
        lang: str = str(kwargs.get("lang", "bash")).strip().lower()
        try:
            timeout: int = int(kwargs.get("timeout", 120))
        except (ValueError, TypeError):
            return ToolResult(
                ok=False, output="",
                error="Invalid value for 'timeout' (must be a number).",
            )
        if not code:
            return ToolResult(ok=False, output="", error="No 'code' specified.")
        if lang not in ("bash", "python"):
            return ToolResult(
                ok=False, output="", error="lang must be 'bash' or 'python'."
            )

        # Guardrails (nur bash): harte Gates gelten auch für Skripte.
        if lang == "bash":
            hit = check_command(code)
            if hit and hit[0] == "hard":
                return ToolResult(
                    ok=False, output="",
                    error=(
                        f"BLOCKED (guardrail): {hit[1]}. "
                        "This script will not be executed."
                    ),
                )

        env = {**os.environ, "LC_ALL": "C"}
        if lang == "bash":
            argv = ["bash", "--noprofile", "--norc", "-c", code]
        else:
            argv = ["python3", "-c", code]

        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=self._base,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout_b, _ = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            if proc:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    await proc.wait()
                except Exception:
                    pass
            return ToolResult(
                ok=False, output="",
                error=f"Timeout after {timeout}s. Script was aborted.",
            )
        except Exception as e:
            return ToolResult(ok=False, output="", error=str(e))

        out = stdout_b.decode(errors="replace")
        if len(out) > _MAX_OUTPUT:
            total = len(out)
            out = out[:_MAX_OUTPUT] + f"\n… [truncated, {total} chars total]"
        rc = proc.returncode if proc.returncode is not None else -1
        if out:
            out += "\n"
        out += f"[exit_code: {rc}]"
        return ToolResult(ok=rc == 0, output=out)
