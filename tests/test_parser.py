"""Unit-Tests für chatcli.agent.parser."""

from __future__ import annotations

import sys
import os
from pathlib import Path

import pytest

# src-Pfad ins sys.path legen, damit Tests ohne pip install laufen
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chatcli.agent.parser import (
    ToolCall,
    parse_tool_calls,
)


# ── parse_tool_calls ──────────────────────────────────────────────

def test_single_tool_call_json():
    raw = '{"tool": "shell", "args": {"command": "ls -la"}}'
    calls = parse_tool_calls(raw)
    assert calls is not None
    assert len(calls) == 1
    assert calls[0].tool == "shell"
    assert calls[0].args == {"command": "ls -la"}


def test_single_tool_call_no_args():
    raw = '{"tool": "plan", "args": {}}'
    calls = parse_tool_calls(raw)
    assert calls is not None
    assert calls[0].tool == "plan"
    assert calls[0].args == {}


def test_tool_call_with_markdown_block():
    raw = 'Ich führe das aus:\n```json\n{"tool": "read_file", "args": {"path": "/tmp/x.txt"}}\n```\nErledigt.'
    calls = parse_tool_calls(raw)
    assert calls is not None
    assert calls[0].tool == "read_file"
    assert calls[0].args == {"path": "/tmp/x.txt"}


def test_tool_call_array():
    raw = '[{"tool": "shell", "args": {"command": "ls"}}, {"tool": "plan", "args": {"steps": ["a"]}}]'
    calls = parse_tool_calls(raw)
    assert calls is not None
    assert len(calls) == 2
    assert calls[0].tool == "shell"
    assert calls[1].tool == "plan"


def test_nested_json_in_markdown_block():
    """Regression #4: Verschachteltes JSON in Codeblock muss komplett
    extrahiert werden (alter Regex-Ansatz brach am ersten } ab)."""
    raw = 'Ausführen:\n```json\n{"tool": "shell", "args": {"command": "ls", "cwd": ".", "timeout": 30}}\n```'
    calls = parse_tool_calls(raw)
    assert calls is not None
    assert len(calls) == 1
    assert calls[0].tool == "shell"
    assert calls[0].args == {"command": "ls", "cwd": ".", "timeout": 30}


def test_nested_array_in_markdown_block():
    """Regression #4: Array mit verschachtelten Objekten im Codeblock."""
    raw = '```json\n[{"tool": "a", "args": {"x": 1}}, {"tool": "b", "args": {"y": [1, 2]}}]\n```'
    calls = parse_tool_calls(raw)
    assert calls is not None
    assert len(calls) == 2
    assert calls[0].args == {"x": 1}
    assert calls[1].args == {"y": [1, 2]}


def test_normal_text_returns_none():
    assert parse_tool_calls("Hallo, wie geht's?") is None


def test_empty_string_returns_none():
    assert parse_tool_calls("") is None


def test_whitespace_around_json():
    raw = '  {"tool": "shell", "args": {"command": "echo hi"}}  '
    calls = parse_tool_calls(raw)
    assert calls is not None
    assert calls[0].tool == "shell"


def test_malformed_json_returns_none():
    assert parse_tool_calls("{{{{") is None


def test_dict_without_tool_key_returns_none():
    assert parse_tool_calls('{"foo": "bar"}') is None


def test_list_without_tool_key_returns_none():
    assert parse_tool_calls('[{"foo": "bar"}]') is None


# ── ToolCall-Dataclass ────────────────────────────────────────────

def test_toolcall_dataclass_fields():
    tc = ToolCall(tool="read_file", args={"path": "x.txt"})
    assert tc.tool == "read_file"
    assert tc.args == {"path": "x.txt"}
