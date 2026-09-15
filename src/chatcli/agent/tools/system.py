"""System tool: compact system snapshot in a single call.

Instead of probing ps, df, free, ss, journalctl individually, this tool
provides a condensed snapshot: load, RAM, disk, top processes, listening
ports and recent log errors. Goal: diagnostics in 1 step instead of 5.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from ..tools import Tool, ToolResult

# A single bash call that collects all sources. Each block is
# self-healing: if a command is missing (ss, journalctl, free ...), only
# that block drops out — the tool as a whole stays green.
_SNAPSHOT_SCRIPT = r"""
set -u
out() { printf '%s\n' "$*"; }

out "## System-Snapshot"

# Last + Uptime
if [ -r /proc/loadavg ]; then
    read -r l1 l5 l15 _ < /proc/loadavg
    out "Uptime/Last: up=$(uptime -p 2>/dev/null || echo '?')  load1=$l1 load5=$l5 load15=$l15"
fi

# RAM
if command -v free >/dev/null 2>&1; then
    free -h | awk 'NR==1{print "RAM: "$0} NR==2{print "  " $0}' | head -2
elif [ -r /proc/meminfo ]; then
    awk '/MemTotal|MemAvailable/{printf "RAM: %s\n", $0}' /proc/meminfo
fi

# Disk
if command -v df >/dev/null 2>&1; then
    df -h --output=target,size,used,avail,pcent 2>/dev/null | grep -v tmpfs | head -8 | sed 's/^/Disk: /'
fi

# Top-5 Prozesse (nach RSS)
if command -v ps >/dev/null 2>&1; then
    ps aux --sort=-%mem 2>/dev/null | head -6 | awk 'NR>1{printf "TopProc: %s (pid %s, %s%% mem, %s%% cpu)\n", $11, $2, $4, $3}'
fi

# Lauschende Ports
if command -v ss >/dev/null 2>&1; then
    ss -tln 2>/dev/null | tail -n +2 | awk '{print "Port: " $4}' | head -15
fi

# Letzter System-Log (Fehler + Warnung, max. 5 Zeilen)
if command -v journalctl >/dev/null 2>&1; then
    logs=$(journalctl -p warning --since "10 min ago" --no-pager -q 2>/dev/null | tail -5)
    if [ -n "$logs" ]; then
        out "## Letzte Log-Warnungen (10 min)"
        printf '%s\n' "$logs"
    fi
fi
"""


class SystemTool(Tool):
    """One call instead of five: system state as a condensed snapshot."""

    name = "system"
    description = (
        "Compact system snapshot in one call (instead of ps/df/free/journalctl individually):\n"
        "Load, RAM, disk, top-5 processes, listening ports, recent log warnings.\n"
        "Call: system() — optional argument: sections list[str] (default: all).\n"
        "Sections: 'load', 'ram', 'disk', 'top', 'ports', 'logs'.\n"
        "Use BEFORE diagnostics (Why is X hanging? What's running on port Y?)"
    )
    args_schema = {
        "sections": "list[str] (optional) — only deliver these sections; default: all "
        "('load', 'ram', 'disk', 'top', 'ports', 'logs')",
    }

    #: Max. output per section block (chars), so the snapshot
    #: never floods the context (tool outputs are capped at 16 KB).
    _MAX_CHARS = 6000

    async def run(self, **kwargs: Any) -> ToolResult:
        sections = kwargs.get("sections")
        if not isinstance(sections, list) or not sections:
            sections = None
        wanted = {str(s).strip().lower() for s in sections} if sections else None

        env = {**os.environ, "LC_ALL": "C", "LANG": "C"}
        try:
            proc = await asyncio.create_subprocess_exec(
                "bash", "-c", _SNAPSHOT_SCRIPT,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            proc.kill()
            return ToolResult(ok=False, output="", error="system: snapshot not finished after 30s")
        except (OSError, ValueError) as exc:
            return ToolResult(ok=False, output="", error=f"system: {exc}")

        raw = out_b.decode(errors="replace")
        if wanted:
            raw = self._filter_sections(raw, wanted)
        if len(raw) > self._MAX_CHARS:
            raw = raw[: self._MAX_CHARS] + "\n[... snapshot truncated]"
        if proc.returncode != 0:
            err = err_b.decode(errors="replace").strip()[:300]
            return ToolResult(
                ok=False, output=raw,
                error=f"system: Bash exit {proc.returncode}: {err}",
            )
        return ToolResult(ok=True, output=raw.strip() or "(empty snapshot)")

    @staticmethod
    def _filter_sections(raw: str, wanted: set[str]) -> str:
        """Filter the snapshot to the requested sections."""
        keep = set()
        mapping = {
            "load": ("Uptime/Last",),
            "ram": ("RAM:",),
            "disk": ("Disk:",),
            "top": ("TopProc:",),
            "ports": ("Port:",),
            "logs": ("## Letzte Log-Warnungen",),
        }
        for name, prefixes in mapping.items():
            if name in wanted:
                keep.update(prefixes)
        out_lines: list[str] = []
        in_block = False
        for line in raw.splitlines():
            is_header = line.startswith("##")
            if line.startswith(("Uptime/Last", "RAM:", "Disk:", "TopProc:", "Port:", "## Letzte")):
                in_block = any(line.startswith(p) for p in keep)
            elif is_header:
                in_block = line in keep or ("logs" in wanted and "Log" in line)
            if in_block:
                out_lines.append(line)
        return "\n".join(out_lines)
