"""Tests für die optimierte Action-Trigger-Logik (Streak-Reset, konkreter
Nudge, Escape-Hatch: Modell kann Plan selbst abschließen).

Kernproblem vor dem Fix: _text_only_streak wurde nach Tool-Calls NICHT
zurückgesetzt → Muster "Text → Nudge → echte Arbeit → finale Antwort"
feuerte fälschlich Ralph-Loop/Abbruch (streak=2), obwohl zwischen den
Text-Antworten legitime Arbeit stattgefunden hatte.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Mocks aus test_stabilization.py wiederverwenden (gleiche Fakes).
from test_stabilization import (
    _FakeStreamChunk,
    _FakeLlamaClient,
    _FakeToolRegistry,
    _FakeConfig,
)

from chatcli.agent.loop import AgentLoop
from chatcli.agent.tools.base import ToolResult
from chatcli.agent.tools.planner import PlannerTool


def _make_loop(responses=None, max_steps=8):
    config = _FakeConfig(max_steps=max_steps)
    client = _FakeLlamaClient(responses=responses)
    registry = _FakeToolRegistry()
    loop = AgentLoop(client, registry, config)
    return loop, client, registry


async def _open_plan(registry, steps=("Quellen lesen", "Pfad prüfen")) -> PlannerTool:
    """Plan direkt im laufenden Event-Loop ausführen (anyio-Tests)."""
    planner = PlannerTool()
    await planner.run(action="set", steps=list(steps), goal="Testziel")
    registry._tools["plan"] = planner
    return planner


class _FakeShell:
    def __init__(self, ok=True):
        self._ok = ok

    async def run(self, **kwargs):
        if self._ok:
            return ToolResult(ok=True, output="ok [exit_code: 0]", error="")
        return ToolResult(ok=False, output="", error="command failed")


def _tool_chunk(name: str, args_json: str) -> _FakeStreamChunk:
    return _FakeStreamChunk(
        tool_call_index=0, tool_call_id=f"tc-{name}",
        tool_call_name=name, tool_call_args_delta=args_json,
        finish_reason="stop",
    )


def _text_chunk(text: str) -> _FakeStreamChunk:
    return _FakeStreamChunk(content=text, finish_reason="stop")


# ── Kernfix: Streak-Reset nach legitimer Arbeit ───────────────────

@pytest.mark.anyio
async def test_streak_reset_after_tool_call_no_false_abort():
    """Text → Nudge → echte Arbeit (Tool) → Text → Nudge → Plan-Abschluss
    → finale Antwort. Kein Ralph/Abbruch, obwohl 2× Text-only vorkamen
    (nicht aufeinanderfolgend)."""
    responses = [
        [_tool_chunk("shell", '{"command": "ls"}')],          # Step 0: Arbeit
        [_text_chunk("Ich schaue mir das noch an.")],          # Step 1: streak=1 → Nudge
        [_tool_chunk("shell", '{"command": "cat a.txt"}')],    # Step 2: Arbeit → Reset
        [_text_chunk("Alles erledigt, ich bin fertig.")],      # Step 3: streak=1 → Nudge
        [_tool_chunk("plan", '{"action": "complete", "step": 1}')],  # Step 4: Escape-Hatch
        [_tool_chunk("plan", '{"action": "complete", "step": 2}')],  # Step 5: Plan komplett
        [_text_chunk("Fertig: Ergebnis ist in a.txt.")],       # Step 6: finale Antwort
    ]
    loop, client, registry = _make_loop(responses=responses, max_steps=8)
    await _open_plan(registry)
    registry._tools["shell"] = _FakeShell(ok=True)

    collected: list[str] = []
    async for chunk in loop.ask("Erledige die Aufgabe"):
        collected.append(chunk)

    joined = "".join(collected)
    assert "Fertig: Ergebnis ist in a.txt" in joined
    assert "Abbruch" not in joined          # kein Action-Trigger-Abbruch
    assert "Ralph Loop" not in joined
    # Finale Antwort wurde normal akzeptiert (Client 7× aufgerufen).
    assert client._call_count == 7


# ── Echte Stagnation: 2× Text ohne Arbeit → Abbruch bleibt erhalten ─

@pytest.mark.anyio
async def test_consecutive_text_only_still_aborts():
    """Ohne Ralph-Loop (Fake-Config ohne enable_ralph_loop): zwei
    aufeinanderfolgende Text-Antworten mit offenem Plan → Abbruch."""
    responses = [
        [_tool_chunk("shell", '{"command": "ls"}')],   # Step 0: Arbeit
        [_text_chunk("Ich werde das jetzt analysieren.")],  # Step 1: streak=1 → Nudge
        [_text_chunk("Und hier ist meine Analyse.")],      # Step 2: streak=2 → Abbruch
    ]
    loop, client, registry = _make_loop(responses=responses, max_steps=5)
    await _open_plan(registry)
    registry._tools["shell"] = _FakeShell(ok=True)

    collected: list[str] = []
    async for chunk in loop.ask("Erledige die Aufgabe"):
        collected.append(chunk)

    joined = "".join(collected)
    assert "aborting" in joined
    # Der Client lief nicht bis max_steps durch.
    assert client._call_count <= 3


# ── Konkreter Nudge zitiert den nächsten offenen Plan-Schritt ─────

@pytest.mark.anyio
async def test_nudge_cites_next_open_step():
    """Der Nudge beim ersten Text-only-Step nennt den konkreten
    nächsten offenen Schritt und das Escape-Hatch."""
    responses = [
        [_tool_chunk("shell", '{"command": "ls"}')],
        [_text_chunk("Ich bin noch am Arbeiten.")],
        [_tool_chunk("plan", '{"action": "complete", "step": 1}')],
        [_tool_chunk("plan", '{"action": "complete", "step": 2}')],
        [_text_chunk("Fertig.")],
    ]
    loop, client, registry = _make_loop(responses=responses, max_steps=8)
    await _open_plan(registry, steps=("Quellen lesen", "Pfad prüfen"))
    registry._tools["shell"] = _FakeShell(ok=True)

    async for chunk in loop.ask("Erledige die Aufgabe"):
        pass

    nudge_msgs = [
        m for m in loop._history
        if m.get("content", "").startswith("[System note]")
        and "Execute the NEXT plan step NOW" in m["content"]
    ]
    assert len(nudge_msgs) == 1
    nudge_text = nudge_msgs[0]["content"]
    # Konkreter Schritt wird zitiert.
    assert "Quellen lesen" in nudge_text
    # Escape-Hatch: Plan selbst abschließen.
    assert "action='complete'" in nudge_text


# ── Escape-Hatch: Modell schließt Plan ab → finale Antwort direkt ─

@pytest.mark.anyio
async def test_escape_hatch_plan_complete_accepts_final():
    """Modell ruft plan(action='complete') auf, danach finale Text-Antwort:
    wird ohne Nudge/Abbruch akzeptiert (Plan-Block leer → normaler Exit)."""
    responses = [
        [_tool_chunk("shell", '{"command": "ls"}')],
        [_tool_chunk("plan", '{"action": "complete", "step": 1}')],
        [_tool_chunk("plan", '{"action": "complete", "step": 2}')],
        [_text_chunk("Erledigt, Ergebnis: OK.")],
    ]
    loop, client, registry = _make_loop(responses=responses, max_steps=6)
    await _open_plan(registry)
    registry._tools["shell"] = _FakeShell(ok=True)

    collected: list[str] = []
    async for chunk in loop.ask("Erledige die Aufgabe"):
        collected.append(chunk)

    joined = "".join(collected)
    assert "Erledigt, Ergebnis: OK." in joined
    assert "Abbruch" not in joined
    # Kein Action-Trigger-Nudge, weil der Plan vor der finalen Antwort
    # abgeschlossen war.
    nudge_msgs = [
        m for m in loop._history
        if m.get("content", "").startswith("[System-Hinweis]")
        and "NÄCHSTEN Plan-Schritt" in m["content"]
    ]
    assert len(nudge_msgs) == 0


# ── Ralph-Loop-Message enthält Escape-Hatch ───────────────────────

@pytest.mark.anyio
async def test_ralph_message_includes_escape_hatch():
    """Ralph-Loop aktiv (Config mit enable_ralph_loop=True): Bei streak=2
    wird Kontext resetet; die Fortsetzungs-Anweisung muss das Escape-Hatch
    enthalten (Plan abschließen, wenn Arbeit erledigt ist)."""

    class _FakeConfigRalph(_FakeConfig):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.enable_ralph_loop = True
            self.ralph_max_loops = 2

    config = _FakeConfigRalph(max_steps=8)
    client = _FakeLlamaClient(responses=[
        [_tool_chunk("shell", '{"command": "ls"}')],
        [_text_chunk("Ich schaue mir das an.")],
        [_text_chunk("Und jetzt die Analyse.")],
        # Nach Ralph-Reset: Modell schließt Plan ab + finale Antwort.
        [_tool_chunk("plan", '{"action": "complete", "step": 1}')],
        [_tool_chunk("plan", '{"action": "complete", "step": 2}')],
        [_text_chunk("Fertig nach Ralph-Loop.")],
    ])
    registry = _FakeToolRegistry()
    loop = AgentLoop(client, registry, config)
    await _open_plan(registry)
    registry._tools["shell"] = _FakeShell(ok=True)

    collected: list[str] = []
    async for chunk in loop.ask("Erledige die Aufgabe"):
        collected.append(chunk)

    joined = "".join(collected)
    assert "Fertig nach Ralph-Loop." in joined
    # Die Ralph-Message (User-Message in der History) enthält Escape-Hatch.
    ralph_msgs = [
        m for m in loop._history
        if "[Ralph Loop" in m.get("content", "")
    ]
    assert len(ralph_msgs) == 1
    assert "action='complete'" in ralph_msgs[0]["content"]
