"""Planner tool: the agent creates its own step-by-step list.

The plan is STATEFUL: ``plan`` sets/replaces, ``complete`` marks
steps as done. The current progress is automatically injected into
the prompt by the AgentLoop at EVERY model step (see
``AgentLoop._inject_plan``) so Lex doesn't lose the thread on long tasks.
"""

from __future__ import annotations

import re
from typing import Any

from ..tools import Tool, ToolResult


class PlannerTool(Tool):
    name = "plan"
    description = (
        "Make yourself a plan? Name the steps you will execute before executing them. "
        "With action='complete' and step=1..n you mark steps as done "
        "(progress is automatically injected). "
        "With action='reset' you delete the plan (topic change: new task). "
        "The tool gives you a confirmation.\n"
        "Examples:\n"
        "  plan(action='set', goal='Research: bwrap sandbox', steps=['Read sources', 'Check path', 'Implementation'])\n"
        "  plan(action='complete', step=1)\n"
        "  plan(action='reset')"
    )
    args_schema = {
        "action": "str (optional, default 'set') — 'set' (create/replace plan), 'complete' (step done) or 'reset' (delete plan on topic change)",
        "steps": "list[str] — list of steps (min 1, max 10, only for action='set')",
        "step": "int (optional) — step number that is considered done with action='complete'",
        "goal": "str (optional) — one-sentence summary of the goal",
    }

    # Ein Plan-Schritt, der sudo/Dienst-Start enthält, braucht einen
    # vorgelagerten Pre-Check (Existenz/Status), sonst startet der Plan
    # ohne Existenz-Check — genau der Fehler, der Lex bei
    # `sudo systemctl start tor` (tor nie installiert) gekostet hat.
    _SUDO_STEP_RE = re.compile(
        r"\bsudo\b|\bsystemctl\s+(start|restart|stop)\b", re.IGNORECASE
    )
    _PRECHECK_RE = re.compile(
        r"pre-?check|command -v|is-active|list-unit-files|existenz|prüfen|vorhanden|installiert",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        self._last_plan: list[str] = []
        self._goal: str = ""
        self._done: set[int] = set()  # 1-basierte Schritt-Nummern

    async def run(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "set")).strip().lower()
        steps: list[str] = kwargs.get("steps", [])
        goal: str = kwargs.get("goal", "")

        if action == "reset":
            # Plan löschen (Themenwechsel: neuer Task beginnt).
            self._last_plan = []
            self._goal = ""
            self._done = set()
            return ToolResult(
                ok=True,
                output="Plan deleted. A new task can start with action='set'.",
            )

        if action == "complete":
            if not self._last_plan:
                return ToolResult(
                    ok=False, output="",
                    error="No active plan — create one first with action='set'.",
                )
            # 'step' fehlt oder ist 0 → ALLE Schritte als erledgt markieren.
            # Grund: Das Modell ruft action='complete' ohne step-Arg auf,
            # wenn es den Plan als Ganzes beendet (z. B. finale Antwort
            # läuft). Ohne dieses Fallback bleibt der Plan "aktiv", wird
            # bei jedem Step injiziert und treibt eine Endlos-Schleife
            # an ("Plan komplett, aber step 2 noch offen …").
            raw_step = kwargs.get("step", 0)
            if raw_step is None or raw_step == 0:
                self._done = set(range(1, len(self._last_plan) + 1))
                hint = (
                    "\nNote: All steps were marked as done "
                    "(step not specified). The plan is now complete."
                )
                return ToolResult(ok=True, output=self._render() + hint)
            try:
                n = int(raw_step)
            except (ValueError, TypeError):
                return ToolResult(
                    ok=False, output="",
                    error="Invalid value for 'step' (must be a number).",
                )
            if not 1 <= n <= len(self._last_plan):
                return ToolResult(
                    ok=False, output="",
                    error=f"Step {n} does not exist (1..{len(self._last_plan)}).",
                )
            self._done.add(n)
            return ToolResult(ok=True, output=self._render())

        # action == "set"
        if not isinstance(steps, list) or len(steps) == 0:
            return ToolResult(ok=False, output="", error="Field 'steps' (non-empty list) is missing.")
        if len(steps) > 10:
            return ToolResult(ok=False, output="", error="Maximum 10 steps allowed.")

        self._last_plan = [str(s) for s in steps]
        self._goal = goal
        self._done = set()

        warning = self._sudo_precheck_warning()
        if warning:
            return ToolResult(ok=True, output=self._render() + "\n" + warning)
        return ToolResult(ok=True, output=self._render())

    def _sudo_precheck_warning(self) -> str:
        """Checks: does every sudo/service-start step have a preceding
        pre-check step (existence/status)? If not → warning, so the plan
        never starts without an existence check.
        """
        for i, step in enumerate(self._last_plan, 1):
            if not self._SUDO_STEP_RE.search(step):
                continue
            # Gibt es einen frühreren Schritt mit Pre-Check?
            has_prior_check = any(
                self._PRECHECK_RE.search(self._last_plan[j])
                for j in range(i - 1)
            )
            if not has_prior_check:
                return (
                    f"⚠ Pre-check required: Step {i} ({step.strip()[:40]}…) "
                    "starts sudo/a service without a preceding existence check. "
                    "Add BEFORE it a step like 'Pre-check: command -v <binary> + "
                    "systemctl is-active; if missing → sudo apt install -y <package>'."
                )
        return ""

    # ------------------------------------------------------------------

    def _render(self) -> str:
        lines = [f"✓ Plan active: {self._goal or '(no goal description)'}"]
        lines.append("  Steps:")
        for i, s in enumerate(self._last_plan, 1):
            mark = "✓" if i in self._done else " "
            lines.append(f"    {mark} {i}. {s}")
        if self._done:
            lines.append("Progress is automatically injected at every step.")
        return "\n".join(lines)

    def status_block(self) -> str:
        """Compact plan status for automatic prompt injection.

        Empty string if no active plan exists OR all steps are done
        (no context noise for completed plans).
        """
        if not self._last_plan:
            return ""
        # Alle Schritte erledigt → nicht mehr injizieren.
        if self._done >= set(range(1, len(self._last_plan) + 1)):
            return ""
        lines = ["[Active Plan]"]
        if self._goal:
            lines.append(f"Goal: {self._goal}")
        for i, s in enumerate(self._last_plan, 1):
            mark = "✓" if i in self._done else "•"
            lines.append(f"  {mark} {i}. {s}")
        return "\n".join(lines)

    @property
    def last_plan(self) -> list[str]:
        return list(self._last_plan)
