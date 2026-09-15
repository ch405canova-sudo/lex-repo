"""Tests für das Script-Tool (Programmatic Tool Calling)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chatcli.agent.tools.script import ScriptTool


@pytest.mark.anyio
async def test_script_bash_echo():
    t = ScriptTool(cwd="/tmp")
    r = await t.run(code="echo hello-script")
    assert r.ok
    assert "hello-script" in r.output
    assert "exit_code: 0" in r.output


@pytest.mark.anyio
async def test_script_bash_loop():
    """Kernnutzen: Loop in EINER Ausführung (statt N Tool-Calls)."""
    t = ScriptTool(cwd="/tmp")
    r = await t.run(code="for i in 1 2 3; do echo line-$i; done")
    assert r.ok
    assert "line-1" in r.output and "line-3" in r.output


@pytest.mark.anyio
async def test_script_python():
    t = ScriptTool(cwd="/tmp")
    r = await t.run(code="print('py-ok')", lang="python")
    assert r.ok
    assert "py-ok" in r.output


@pytest.mark.anyio
async def test_script_failure_exit_code():
    t = ScriptTool(cwd="/tmp")
    r = await t.run(code="exit 7")
    assert not r.ok
    assert "exit_code: 7" in r.output


@pytest.mark.anyio
async def test_script_empty_rejected():
    t = ScriptTool(cwd="/tmp")
    r = await t.run(code="")
    assert not r.ok


@pytest.mark.anyio
async def test_script_bad_lang_rejected():
    t = ScriptTool(cwd="/tmp")
    r = await t.run(code="ls", lang="sh")
    assert not r.ok
    assert "bash" in r.error or "python" in r.error


@pytest.mark.anyio
async def test_script_timeout():
    t = ScriptTool(cwd="/tmp")
    r = await t.run(code="sleep 5", timeout=1)
    assert not r.ok
    assert "Timeout" in r.error


@pytest.mark.anyio
async def test_script_hard_blocked():
    """Harte Guardrails gelten auch für Skripte."""
    t = ScriptTool(cwd="/tmp")
    r = await t.run(code="mkfs.ext4 /dev/sda1")
    assert not r.ok
    assert "BLOCKED" in r.error


@pytest.mark.anyio
async def test_script_stderr_merged(tmp_path):
    t = ScriptTool(cwd=str(tmp_path))
    r = await t.run(code="echo out; echo err 1>&2")
    assert r.ok
    assert "out" in r.output and "err" in r.output
