"""Jobs tool: run long commands in the background.

The shell tool blocks until timeout — a build or server start eats the
entire step. This tool starts commands as background jobs: keep working
immediately, query/kill later.

State: process-local (``_jobs``) — jobs die with the REPL.
Output: ring buffer (last 150 lines per job), so ``read`` never
floods the context.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from ..tools import Tool, ToolResult

#: Maximum number of concurrent jobs (old, finished jobs
#: are automatically discarded when the limit is reached).
_MAX_JOBS = 10
#: Lines per job in the ring buffer.
_MAX_LINES = 150
#: Max. output bytes per ``read`` call (context protection).
_MAX_READ_CHARS = 8000


@dataclass
class _Job:
    id: str
    command: str
    proc: asyncio.subprocess.Process
    lines: deque[str] = field(default_factory=lambda: deque(maxlen=_MAX_LINES))
    returncode: int | None = None
    started_at: float = field(default_factory=time.time)

    @property
    def finished(self) -> bool:
        return self.returncode is not None

    @property
    def age(self) -> float:
        return time.time() - self.started_at


class JobsTool(Tool):
    """Long commands in the background: start/status/read/kill."""

    name = "jobs"
    description = (
        "Run long commands (builds, scans, server starts) IN THE BACKGROUND "
        "— instead of blocking the shell tool until timeout.\n"
        "  jobs(action='start', command='make -j')          → job ID\n"
        "  jobs(action='status')                             → all jobs + status\n"
        "  jobs(action='read', job='job-1')                  → last output lines\n"
        "  jobs(action='kill', job='job-1')                  → terminate job\n"
        "IMPORTANT: Jobs only live in this session (die with the REPL)."
    )
    args_schema = {
        "action": "str — 'start' (new job), 'status' (list), 'read' (output), 'kill' (terminate)",
        "command": "str (only for action='start') — the command to execute",
        "job": "str (optional) — job ID (e.g. 'job-1'); default: newest job",
    }

    def __init__(self, on_line: "Callable[[str], None] | None" = None) -> None:
        self._jobs: dict[str, _Job] = {}
        self._counters = itertools.count(1)
        # Optional live callback (REPL live box): shows job lines
        # in the terminal, just like ShellTool does.
        self._on_line = on_line

    async def run(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "")).strip().lower()
        if action == "start":
            return await self._start(kwargs)
        if action == "status":
            return self._status()
        if action == "read":
            return self._read(kwargs)
        if action == "kill":
            return await self._kill(kwargs)
        return ToolResult(
            ok=False, output="",
            error=f"jobs: unknown action '{action}' (start/status/read/kill).",
        )

    # ------------------------------------------------------------------

    async def _start(self, kwargs: Any) -> ToolResult:
        command = str(kwargs.get("command", "")).strip()
        if not command:
            return ToolResult(ok=False, output="", error="jobs: action='start' requires a command.")

        if len(self._jobs) >= _MAX_JOBS:
            # Discard oldest FINISHED job, otherwise rejected.
            finished = [j for j in self._jobs.values() if j.finished]
            if not finished:
                return ToolResult(
                    ok=False, output="",
                    error=f"jobs: {_MAX_JOBS} active jobs at limit — kill one first.",
                )
            victim = min(finished, key=lambda j: j.started_at)
            del self._jobs[victim.id]

        job_id = f"job-{next(self._counters)}"
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except (OSError, ValueError) as exc:
            return ToolResult(ok=False, output="", error=f"jobs: start failed: {exc}")

        job = _Job(id=job_id, command=command, proc=proc)
        self._jobs[job_id] = job
        asyncio.get_running_loop().create_task(self._pump(job))
        return ToolResult(
            ok=True,
            output=(
                f"{job_id} started: {command}\n"
                f"Query: jobs(action='read', job='{job_id}') · "
                f"Terminate: jobs(action='kill', job='{job_id}')"
            ),
        )

    async def _pump(self, job: _Job) -> None:
        """Pump the job's output into the ring buffer (until process end)."""
        assert job.proc.stdout is not None
        try:
            async for raw in job.proc.stdout:
                line = raw.decode(errors="replace").rstrip("\n")
                if line:
                    job.lines.append(line)
                    if self._on_line:
                        try:
                            self._on_line(f"[{job.id}] {line}")
                        except Exception:
                            pass
        except (OSError, ValueError):
            pass
        job.returncode = await job.proc.wait()

    def _status(self) -> ToolResult:
        if not self._jobs:
            return ToolResult(ok=True, output="No jobs (all finished or none started).")
        lines = []
        for job in self._jobs.values():
            state = (
                f"finished (exit {job.returncode})"
                if job.finished else "running"
            )
            lines.append(
                f"{job.id}: {state} · {job.age:.0f}s · {job.command[:80]}"
            )
        return ToolResult(ok=True, output="\n".join(lines))

    def _read(self, kwargs: Any) -> ToolResult:
        job = self._resolve(kwargs)
        if job is None:
            return ToolResult(ok=False, output="", error="jobs: no job found.")
        out = "\n".join(job.lines) or "(no output yet)"
        header = f"{job.id}: {'finished (exit %d)' % job.returncode if job.finished else 'running'}"
        body = f"{header}\n{out}"
        if len(body) > _MAX_READ_CHARS:
            body = "…" + body[-_MAX_READ_CHARS:]
        return ToolResult(ok=True, output=body)

    async def _kill(self, kwargs: Any) -> ToolResult:
        job = self._resolve(kwargs)
        if job is None:
            return ToolResult(ok=False, output="", error="jobs: no job found.")
        if job.finished:
            return ToolResult(
                ok=True,
                output=f"{job.id} is already finished (exit {job.returncode}).",
            )
        try:
            job.proc.terminate()
            try:
                await asyncio.wait_for(job.proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                job.proc.kill()
                await job.proc.wait()
        except ProcessLookupError:
            pass
        job.returncode = job.proc.returncode
        self._jobs.pop(job.id, None)
        return ToolResult(
            ok=True,
            output=f"{job.id} terminated (exit {job.returncode}).",
        )

    def _resolve(self, kwargs: Any) -> _Job | None:
        ref = str(kwargs.get("job", "")).strip()
        if ref:
            return self._jobs.get(ref)
        if not self._jobs:
            return None
        # Default: the NEWEST job (usually the one you're interested in).
        return max(self._jobs.values(), key=lambda j: j.started_at)
