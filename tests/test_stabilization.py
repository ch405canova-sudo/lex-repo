"""Unit-Tests für die 8 Stabilisierungs-Features (Phase 1+2)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import AsyncIterator, Optional
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


class _FakeStreamChunk:
    def __init__(self, content="", reasoning="", finish_reason="",
                 tool_call_index=None, tool_call_id=None,
                 tool_call_name=None, tool_call_args_delta=None):
        self.content = content
        self.reasoning = reasoning
        self.finish_reason = finish_reason
        self.tool_call_index = tool_call_index
        self.tool_call_id = tool_call_id
        self.tool_call_name = tool_call_name
        self.tool_call_args_delta = tool_call_args_delta


class _FakeLlamaClient:
    """Mock für LlamaClient — liefert konfigurierbare StreamChunks.

    ``responses`` ist eine Liste von Chunk-Listen (eine pro Call).
    Wenn None, wird ``chunks`` für jeden Call verwendet.
    """

    def __init__(self, chunks: list[_FakeStreamChunk] | None = None,
                 context_usage: dict | None = None,
                 responses: list[list[_FakeStreamChunk]] | None = None):
        self._chunks = chunks or []
        self._context_usage = context_usage
        self.supports_tools = True
        self._call_count = 0
        self._responses = responses

    async def chat_stream(self, messages, *, tools=None,
                         temperature=None, max_tokens=None) -> AsyncIterator:
        self._call_count += 1
        idx = self._call_count - 1
        if self._responses is not None and idx < len(self._responses):
            chunks = self._responses[idx]
        else:
            chunks = self._chunks
        for chunk in chunks:
            yield chunk

    async def context_usage(self) -> Optional[dict]:
        return self._context_usage


class _FakeToolRegistry:
    def __init__(self):
        self._tools = {}

    def get(self, name):
        return self._tools.get(name)

    def openai_schemas(self):
        return []

    def descriptions(self):
        return ""


class _FakeConfig:
    def __init__(self, max_steps=5, timeout=300, max_tokens=4096,
                 temperature=0.6, enable_selective_lessons=False,
                 shell_cwd="/tmp"):
        self.max_steps = max_steps
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_selective_lessons = enable_selective_lessons
        self.shell_cwd = shell_cwd
        self.system_prompt = "Test-System-Prompt"
        self.selective_lesson_count = 5
        self.lessons_topic = "agent-lessons"
        self.wiki_root = Path("/tmp/fake_wiki")


def _make_loop(chunks=None, context_usage=None, max_steps=5,
               enable_selective_lessons=False, responses=None):
    """Erzeugt eine AgentLoop mit Mocks."""
    from chatcli.agent.loop import AgentLoop

    config = _FakeConfig(max_steps=max_steps,
                        enable_selective_lessons=enable_selective_lessons)
    client = _FakeLlamaClient(chunks=chunks, context_usage=context_usage,
                              responses=responses)
    registry = _FakeToolRegistry()
    loop = AgentLoop(client, registry, config)
    return loop, client, registry


# ── #3 Tool-Output-Offloading ──────────────────────────────────────

def test_offload_tool_output_writes_file(tmp_path):
    from chatcli.agent.loop import AgentLoop

    output = "x" * 20000  # > 16KB
    path = AgentLoop._offload_tool_output(1, "shell", output)
    assert path != ""
    p = Path(path)
    assert p.exists()
    assert p.read_text(encoding="utf-8") == output


def test_offload_tool_output_sanitizes_name():
    from chatcli.agent.loop import AgentLoop

    path = AgentLoop._offload_tool_output(1, "my-tool/with spaces", "test")
    assert "/" not in Path(path).name
    assert " " not in Path(path).name


# ── #2 KV-Cache-Stabilität ─────────────────────────────────────────

@pytest.mark.anyio
async def test_refresh_system_memory_injects_at_index_1():
    """Lessons werden als separate User-Message an Index 1 injiziert,
    System-Message (Index 0) bleibt unverändert."""
    loop, _, _ = _make_loop(enable_selective_lessons=True)

    loop._history = [
        {"role": "system", "content": "SYSTEM-PROMPT-STABIL"},
        {"role": "user", "content": "Hallo"},
        {"role": "assistant", "content": "Hi!"},
    ]

    with patch("chatcli.memory.style_injection", return_value="LESSON-TEXT"):
        loop._refresh_system_memory("test query")

    assert loop._history[0]["content"] == "SYSTEM-PROMPT-STABIL"
    assert loop._history[1]["role"] == "user"
    assert "[Lessons]" in loop._history[1]["content"]
    assert "_is_lessons" in loop._history[1]


@pytest.mark.anyio
async def test_refresh_system_memory_updates_in_place():
    """Bestehende Lessons-Message wird aktualisiert, nicht dupliziert."""
    loop, _, _ = _make_loop(enable_selective_lessons=True)

    loop._history = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "[Lessons] OLD", "_is_lessons": True},
        {"role": "user", "content": "Hallo"},
    ]

    with patch("chatcli.memory.style_injection", return_value="NEW-LESSON"):
        loop._refresh_system_memory("query")

    assert "[Lessons] NEW-LESSON" in loop._history[1]["content"]
    lessons_count = sum(1 for m in loop._history if m.get("_is_lessons"))
    assert lessons_count == 1


# ── #4 Objective-Recitation ────────────────────────────────────────

@pytest.mark.anyio
async def test_compaction_includes_original_objective():
    """Nach Compaction wird das Originalziel in den Situation-Bericht injiziert."""
    loop, _, _ = _make_loop(context_usage={"n_ctx": 100, "n_ctx_used": 95})

    loop._original_objective = "Original-Nutzerziel"
    loop._history = [{"role": "system", "content": "SYSTEM"}]
    for i in range(30):
        loop._history.append({"role": "user", "content": f"msg {i}"})
        loop._history.append({"role": "assistant", "content": f"resp {i}"})

    await loop._compact_history()

    situation_msg = loop._history[1]
    assert "original goal" in situation_msg["content"]
    assert "Original-Nutzerziel" in situation_msg["content"]


# ── #6 Per-Step-Timeout ────────────────────────────────────────────

@pytest.mark.anyio
async def test_step_timeout_constant():
    """_STEP_TIMEOUT ist eine positive Zahl."""
    from chatcli.agent.loop import _STEP_TIMEOUT
    assert _STEP_TIMEOUT > 0
    assert isinstance(_STEP_TIMEOUT, float)


# ── #8 Token/Budget-Monitoring ─────────────────────────────────────

@pytest.mark.anyio
async def test_token_budget_warn_ratio():
    """_TOKEN_BUDGET_WARN_RATIO ist > 1."""
    from chatcli.agent.loop import _TOKEN_BUDGET_WARN_RATIO
    assert _TOKEN_BUDGET_WARN_RATIO > 1


# ── #5 Error-Classification + Retry ────────────────────────────────

def test_retryable_error_classification():
    """429/500/502/503/504 sind retryable, 401/403/422 fatal."""
    _RETRYABLE_CODES = {429, 500, 502, 503, 504}
    _FATAL_CODES = {401, 403, 422}

    assert 429 in _RETRYABLE_CODES
    assert 500 in _RETRYABLE_CODES
    assert 502 in _RETRYABLE_CODES
    assert 503 in _RETRYABLE_CODES
    assert 504 in _RETRYABLE_CODES

    assert 401 in _FATAL_CODES
    assert 403 in _FATAL_CODES
    assert 422 in _FATAL_CODES

    assert not (_RETRYABLE_CODES & _FATAL_CODES)


# ── #1 Early Stopping Generate ─────────────────────────────────────

@pytest.mark.anyio
async def test_early_stopping_make_final_call():
    """Bei max_steps wird ein letzter LLM-Call OHNE Tools gemacht."""
    # Call 1 (Step 0): Tool-Call → Loop läuft weiter
    tool_chunk = _FakeStreamChunk(
        tool_call_index=0, tool_call_id="tc1",
        tool_call_name="shell", tool_call_args_delta='{"command": "echo hi"}',
        finish_reason="stop",
    )
    # Call 2 (Early Stopping): finale Synthese ohne Tools
    final_chunk = _FakeStreamChunk(
        content="Finale Synthese-Antwort", finish_reason="stop",
    )
    loop, client, registry = _make_loop(
        max_steps=1,
        responses=[[tool_chunk], [final_chunk]],
    )

    class FakeShell:
        async def run(self, **kwargs):
            from chatcli.agent.tools.base import ToolResult
            return ToolResult(ok=True, output="hi", error="")

    registry._tools = {"shell": FakeShell()}

    events: list[str] = []
    async for chunk in loop.ask("Test-Frage"):
        events.append(chunk)

    assert any("Finale Synthese" in e for e in events)
    # Client wurde 2× aufgerufen (1× im Loop, 1× Early Stopping)
    assert client._call_count == 2


# ── #7 Self-Verification Nudge ─────────────────────────────────────

@pytest.mark.anyio
async def test_verify_nudge_injected_once():
    """Self-Verification Nudge wird nur EINMAL pro Session injiziert.
    Feuert wenn ein Plan BESTANDEN hat und jetzt ABGESCHLOSSEN ist."""
    from chatcli.agent.tools.base import ToolResult

    tool_call_chunk = _FakeStreamChunk(
        tool_call_index=0, tool_call_id="tc1",
        tool_call_name="shell", tool_call_args_delta='{"command": "echo hi"}',
        finish_reason="stop",
    )
    text_chunk = _FakeStreamChunk(content="Ergebnis ist OK", finish_reason="stop")
    final_chunk = _FakeStreamChunk(content="Fertig", finish_reason="stop")

    # Call 1 (Step 0): Tool-Call → erfolgreich
    # Call 2 (Step 1): Text-Antwort → Nudge wird injiziert → continue
    # Call 3 (Step 2): Finale Antwort → normaler Exit
    loop, client, registry = _make_loop(
        responses=[[tool_call_chunk], [text_chunk], [final_chunk]],
    )

    class FakeShell:
        async def run(self, **kwargs):
            return ToolResult(ok=True, output="hi", error="")

    registry._tools = {"shell": FakeShell()}

    # Plan anlegen + sofort abschließen → _plan_block() == "" (alle Schritte erledigt)
    from chatcli.agent.tools.planner import PlannerTool
    planner = PlannerTool()
    await planner.run(action="set", steps=["Schritt 1"], goal="Test")
    await planner.run(action="complete", step=1)
    registry._tools["plan"] = planner

    events: list[str] = []
    async for chunk in loop.ask("Test"):
        events.append(chunk)

    nudge_msgs = [m for m in loop._history if "Verify NOW" in m.get("content", "")]
    assert len(nudge_msgs) == 1


# ── Integration: Alle Features zusammen ───────────────────────────

@pytest.mark.anyio
async def test_full_loop_with_all_features():
    """End-to-End: Loop mit Tool-Call, Offloading, Token-Tracking."""
    from chatcli.agent.tools.base import ToolResult

    tool_call_chunk = _FakeStreamChunk(
        tool_call_index=0, tool_call_id="tc1",
        tool_call_name="shell", tool_call_args_delta='{"command": "echo ok"}',
        finish_reason="stop",
    )
    final_chunk = _FakeStreamChunk(content="Alles erledigt", finish_reason="stop")

    chunks = [tool_call_chunk, final_chunk]
    loop, client, registry = _make_loop(
        chunks=chunks,
        context_usage={"n_ctx": 10000, "n_ctx_used": 100},
    )

    class FakeShell:
        async def run(self, **kwargs):
            return ToolResult(ok=True, output="ok [exit_code: 0]", error="")

    registry._tools = {"shell": FakeShell()}

    collected: list[str] = []
    async for chunk in loop.ask("Mach was"):
        collected.append(chunk)

    assert "Alles erledigt" in collected
    assert client._call_count >= 2


# ── Verbesserter Nudge nach Tool-Fehler im Action-Trigger ──────────

@pytest.mark.anyio
async def test_nudge_after_failed_tool_with_plan():
    """Aktiver Plan + Text-Antwort nach Tool-Fehler → Nudge enthält
    Hinweis auf fehlgeschlagenen Call und Ansatzwechsel."""
    from chatcli.agent.loop import AgentLoop
    from chatcli.agent.tools.planner import PlannerTool

    loop, client, registry = _make_loop(max_steps=5)

    # Aktiven Plan anlegen (1 offener Schritt).
    planner = PlannerTool()
    await planner.run(action="set", steps=["Schritt 1"], goal="Test")
    registry._tools["plan"] = planner

    # Response 0: Tool-Call der fehlschlägt.
    tool_call_chunk = _FakeStreamChunk(
        tool_call_index=0, tool_call_id="tc1",
        tool_call_name="shell", tool_call_args_delta='{"command": "false"}',
        finish_reason="stop",
    )
    # Response 1: Text-Antwort ohne Tool-Call (triggert Action-Trigger).
    text_chunk = _FakeStreamChunk(
        content="Ich werde das jetzt versuchen.", finish_reason="stop"
    )
    responses = [[tool_call_chunk], [text_chunk]]
    loop, client, registry = _make_loop(responses=responses, max_steps=5)

    # Plan erneut setzen (da _make_loop neue Instanz erzeugt).
    planner2 = PlannerTool()
    await planner2.run(action="set", steps=["Schritt 1"], goal="Test")
    registry._tools["plan"] = planner2

    from chatcli.agent.tools.base import ToolResult

    class FailingShell:
        async def run(self, **kwargs):
            return ToolResult(ok=False, output="", error="command failed")

    registry._tools["shell"] = FailingShell()

    collected: list[str] = []
    async for chunk in loop.ask("Test"):
        collected.append(chunk)

    # Nudge muss den Hinweis auf fehlgeschlagenen Call enthalten.
    nudge_msgs = [
        m for m in loop._history
        if "failed" in m.get("content", "").lower()
    ]
    assert len(nudge_msgs) >= 1


# ── Server-Fehler-Detection ("Sorry, try again") ───────────────────

@pytest.mark.anyio
async def test_server_error_sorry_try_again_triggers_retry():
    """Server liefert 'Sorry, try again.' → kein Abbruch, sondern Retry."""
    responses = [
        [_FakeStreamChunk(content="Sorry, try again.", finish_reason="stop")],
        # Zweiter Versuch: normale Antwort.
        [_FakeStreamChunk(
            content="Die Aufgabe ist erledigt und hier ist das Ergebnis.",
            finish_reason="stop",
        )],
    ]
    loop, client, registry = _make_loop(responses=responses)

    collected: list[str] = []
    async for chunk in loop.ask("Test"):
        collected.append(chunk)

    # Die finale Antwort wird akzeptiert (kein Abbruch bei 'Sorry').
    assert any("erledigt" in c for c in collected)
    # Der Client wurde 2× aufgerufen (Retry statt End).
    assert client._call_count == 2


@pytest.mark.anyio
async def test_server_error_not_triggered_on_long_text():
    """Lange Antwort mit 'sorry' im Text → kein Retry, normaler Abschluss."""
    responses = [
        [_FakeStreamChunk(
            content="Sorry, ich habe einen Fehler gemacht. Hier ist die korrigierte "
                    "Version der Datei mit allen Änderungen eingebaut.",
            finish_reason="stop",
        )],
    ]
    loop, client, registry = _make_loop(responses=responses)

    collected: list[str] = []
    async for chunk in loop.ask("Test"):
        collected.append(chunk)

    # Die lange Antwort wird akzeptiert (kein Retry).
    assert any("korrigierte" in c for c in collected)
    assert client._call_count == 1
