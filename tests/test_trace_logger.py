"""Unit-Tests für den JSONL-Trace-Logger (Session-Nachvollziehbarkeit)."""

from __future__ import annotations

import json
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
    def __init__(self, chunks=None, context_usage=None, responses=None):
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
                 temperature=0.6, shell_cwd="/tmp"):
        self.max_steps = max_steps
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = "Test-System-Prompt"
        self.selective_lesson_count = 5
        self.lessons_topic = "agent-lessons"
        self.shell_cwd = shell_cwd
        self.wiki_root = Path("/tmp/fake_wiki")
        self.enable_selective_lessons = False


def _make_loop(responses=None, max_steps=5):
    from chatcli.agent.loop import AgentLoop

    config = _FakeConfig(max_steps=max_steps)
    client = _FakeLlamaClient(responses=responses)
    registry = _FakeToolRegistry()
    loop = AgentLoop(client, registry, config)
    return loop, client, registry


# ── Konstante & Initialisierung ───────────────────────────────────

def test_trace_dir_is_under_local_share():
    from chatcli.agent.loop import _TRACE_DIR

    assert isinstance(_TRACE_DIR, Path)
    parts = _TRACE_DIR.parts
    assert ".local" in parts and "chatcli" in parts and "traces" in parts


def test_init_creates_trace_file(tmp_path):
    """AgentLoop legt beim Start eine leere .jsonl-Datei an."""
    from chatcli.agent.loop import AgentLoop, _TRACE_DIR

    loop = AgentLoop(_FakeLlamaClient(), _FakeToolRegistry(), _FakeConfig())
    try:
        assert loop._trace_path is not None
        assert loop._trace_path.suffix == ".jsonl"
        assert loop._trace_path.parent == _TRACE_DIR
        assert loop._trace_path.exists()
        assert loop._trace_path.read_text(encoding="utf-8") == ""
    finally:
        loop._trace_path.unlink(missing_ok=True)


# ── _trace(): JSONL-Schreibverhalten ───────────────────────────────

def test_trace_appends_valid_jsonl(tmp_path):
    """Jeder Aufruf hängt genau eine valide JSON-Zeile an."""
    from chatcli.agent.loop import AgentLoop

    loop = AgentLoop(_FakeLlamaClient(), _FakeToolRegistry(), _FakeConfig())
    try:
        loop._trace({"event": "step_start", "step": 0, "history_len": 1})
        loop._trace({"event": "tool_result", "step": 1, "tool": "shell", "ok": True})

        lines = loop._trace_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

        for line in lines:
            entry = json.loads(line)  # muss parsen
            assert "ts" in entry and isinstance(entry["ts"], float)

        assert json.loads(lines[0])["event"] == "step_start"
        assert json.loads(lines[1])["tool"] == "shell"
    finally:
        loop._trace_path.unlink(missing_ok=True)


def test_trace_preserves_unicode(tmp_path):
    """ensure_ascii=False → Umlaute bleiben lesbar."""
    from chatcli.agent.loop import AgentLoop

    loop = AgentLoop(_FakeLlamaClient(), _FakeToolRegistry(), _FakeConfig())
    try:
        loop._trace({"event": "nudge", "text": "Überprüfe äöü — ß"})
        content = loop._trace_path.read_text(encoding="utf-8")
        assert "Überprüfe äöü — ß" in content
    finally:
        loop._trace_path.unlink(missing_ok=True)


def test_trace_noop_when_disabled():
    """_trace_path=None (kein Schreibrecht) → kein Fehler, kein Write."""
    from chatcli.agent.loop import AgentLoop

    loop = AgentLoop(_FakeLlamaClient(), _FakeToolRegistry(), _FakeConfig())
    loop._trace_path = None  # simuliert OSError beim Init
    loop._trace({"event": "step_start", "step": 0})  # darf nicht werfen


def test_trace_survives_write_error(tmp_path):
    """OSError beim Schreiben darf den Loop nie blockieren."""
    from chatcli.agent.loop import AgentLoop

    loop = AgentLoop(_FakeLlamaClient(), _FakeToolRegistry(), _FakeConfig())
    try:
        with patch("builtins.open", side_effect=OSError("disk full")):
            loop._trace({"event": "step_start", "step": 0})  # muss still bleiben
    finally:
        loop._trace_path.unlink(missing_ok=True)


# ── Integration: finale Antwort triggert Trace-Event ───────────────

@pytest.mark.anyio
async def test_final_answer_traces_event():
    """End-to-End: Text-Antwort → 'final_answer'-Eintrag in der JSONL-Datei.
    Validiert den Fix für die kaputte Call-Signatur (TypeError)."""
    responses = [
        [_FakeStreamChunk(content="Alles erledigt", finish_reason="stop")],
    ]
    loop, client, _ = _make_loop(responses=responses)
    try:
        collected: list[str] = []
        async for chunk in loop.ask("Test-Frage"):
            collected.append(chunk)

        assert any("Alles erledigt" in c for c in collected)

        entries = [
            json.loads(line)
            for line in loop._trace_path.read_text(encoding="utf-8").splitlines()
        ]
        final_entries = [e for e in entries if e.get("event") == "final_answer"]
        assert len(final_entries) == 1
        assert final_entries[0]["text_len"] > 0
        assert "ts" in final_entries[0]
    finally:
        loop._trace_path.unlink(missing_ok=True)


@pytest.mark.anyio
async def test_tool_result_traces_event():
    """Tool-Call → 'tool_result'-Eintrag mit Tool-Name und ok-Flag."""
    from chatcli.agent.tools.base import ToolResult

    tool_call_chunk = _FakeStreamChunk(
        tool_call_index=0, tool_call_id="tc1",
        tool_call_name="shell", tool_call_args_delta='{"command": "echo hi"}',
        finish_reason="stop",
    )
    final_chunk = _FakeStreamChunk(content="Fertig", finish_reason="stop")

    loop, client, registry = _make_loop(
        responses=[[tool_call_chunk], [final_chunk]],
    )

    class FakeShell:
        async def run(self, **kwargs):
            return ToolResult(ok=True, output="hi", error="")

    registry._tools = {"shell": FakeShell()}

    try:
        collected: list[str] = []
        async for chunk in loop.ask("Test"):
            collected.append(chunk)

        entries = [
            json.loads(line)
            for line in loop._trace_path.read_text(encoding="utf-8").splitlines()
        ]
        tool_entries = [e for e in entries if e.get("event") == "tool_result"]
        assert len(tool_entries) == 1
        assert tool_entries[0]["tool"] == "shell"
        assert tool_entries[0]["ok"] is True
    finally:
        loop._trace_path.unlink(missing_ok=True)
