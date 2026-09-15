"""Tests für das Sub-Agent-Tool (Schema, Registry-Integration, Grundlauf)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chatcli.agent.tools.subagent import SubAgentTool, _EXCLUDED_TOOLS
from chatcli.agent.tools.base import ToolRegistry


# ── Schema-Tests ──────────────────────────────────────────────────

def test_subagent_schema_has_task_required():
    """Schema muss 'task' als required parameter definieren."""
    tool = SubAgentTool(client=None, config=None, parent_registry=ToolRegistry())
    schema = tool.schema()
    assert schema["parameters"]["required"] == ["task"]
    props = schema["parameters"]["properties"]
    assert "task" in props
    assert props["task"]["type"] == "string"


def test_subagent_schema_has_context_optional():
    """Schema muss 'context' als optionalen Parameter definieren."""
    tool = SubAgentTool(client=None, config=None, parent_registry=ToolRegistry())
    schema = tool.schema()
    props = schema["parameters"]["properties"]
    assert "context" in props
    assert props["context"]["type"] == "string"
    # context ist NICHT in required
    assert "context" not in schema["parameters"]["required"]


def test_subagent_schema_description_mentions_tools():
    """Description muss die reduzierten Tools erwähnen."""
    tool = SubAgentTool(client=None, config=None, parent_registry=ToolRegistry())
    schema = tool.schema()
    desc = schema["description"]
    assert "shell" in desc
    assert "read_file" in desc


def test_subagent_args_schema_matches_json_schema():
    """args_schema (für Prompt-Serialisierung) muss zum JSON-Schema passen."""
    tool = SubAgentTool(client=None, config=None, parent_registry=ToolRegistry())
    # args_schema ist ein dict mit Beschreibung pro Parameter
    assert "task" in tool.args_schema
    assert "context" in tool.args_schema
    # task ist required (steht im JSON-Schema)
    schema = tool.schema()
    assert "task" in schema["parameters"]["required"]


# ── Excluded Tools ─────────────────────────────────────────────────

def test_excluded_tools_prevents_recursion():
    """subagent selbst muss in _EXCLUDED_TOOLS sein (keine Rekursion)."""
    assert "subagent" in _EXCLUDED_TOOLS
    assert "plan" in _EXCLUDED_TOOLS
    assert "self_improve" in _EXCLUDED_TOOLS


# ── Registry-Integration ──────────────────────────────────────────

def test_subagent_not_in_default_registry():
    """subagent wird NICHT in build_default_registry registriert
    (nur explizit in repl.py bei enable_subagent=True)."""
    from chatcli.agent.tools.base import build_default_registry
    reg = build_default_registry(shell_cwd="/tmp")
    names = {t.name for t in reg.all()}
    assert "subagent" not in names


@pytest.mark.anyio
async def test_subagent_empty_task_rejected():
    """Leerer task → saubere Fehlermeldung, kein Crash."""
    tool = SubAgentTool(client=None, config=None, parent_registry=ToolRegistry())
    r = await tool.run(task="")
    assert not r.ok
    assert "empty" in (r.error or "").lower()


@pytest.mark.anyio
async def test_subagent_whitespace_task_rejected():
    """Nur Whitespace → ebenfalls abgelehnt."""
    tool = SubAgentTool(client=None, config=None, parent_registry=ToolRegistry())
    r = await tool.run(task="   ")
    assert not r.ok


# ── Mock-Integration (Sub-Agent-Loop) ─────────────────────────────

@pytest.mark.anyio
async def test_subagent_starts_isolated_loop():
    """Sub-Agent erstellt einen frischen AgentLoop mit reduziertem Registry."""
    from chatcli.agent.loop import AgentLoop

    # Vollständiger Config-Mock (alle Attribute, die AgentLoop + style_injection brauchen)
    class MockConfig:
        subagent_max_steps = 3
        max_steps = 3
        max_tokens = 100
        temperature = 0.7
        shell_cwd = "/tmp"
        wiki_root = Path("/tmp/nonexistent-wiki")
        skill_dir = Path("/tmp/nonexistent-skill")
        wiki_dir = Path("/tmp/nonexistent-wiki/wiki")
        system_prompt = "Du bist ein Test-Agent."
        lessons_topic = "agent-lessons"
        style_topic = "lex-behavior"
        enable_selective_lessons = False
        max_injected_lessons = 5
        selective_lesson_count = 3

    # Mock-Client: liefert sofort eine Antwort (StreamChunk-basiert)
    from chatcli.llama_client import StreamChunk

    class MockClient:
        supports_tools = True

        async def chat_stream(self, messages, **kw):
            yield StreamChunk(content="Erledigt: 2+2=4", finish_reason="stop")

        async def context_usage(self):
            return {"n_ctx": 100, "n_ctx_used": 10}

    parent_reg = ToolRegistry()
    from chatcli.agent.tools.shell import ShellTool
    parent_reg.register(ShellTool(cwd="/tmp"))
    # subagent selbst wird ausgeschlossen
    parent_reg.register(SubAgentTool(client=None, config=None, parent_registry=parent_reg))

    tool = SubAgentTool(client=MockClient(), config=MockConfig(), parent_registry=parent_reg)
    r = await tool.run(task="Berechne 2+2")
    assert r.ok
    assert "[Sub-Agent Result]" in r.output
    assert "2+2" in r.output
