"""Regressionstest: confirm_ask-Callback in ShellTool.

Verifiziert, dass destruktive Befehle NUR nach direkter Nutzer-Bestätigung
ausgeführt werden — unabhängig vom Modell-Verhalten (confirm=true).

Testfälle:
  1. rm -rf OHNE confirm_ask, OHNE confirm=true → blockiert
  2. rm -rf MIT confirm_ask="Ja" → ausgeführt
  3. rm -rf MIT confirm_ask="Nein" → blockiert
  4. rm -rf MIT confirm=true (Modell hat selbst gefragt) → ausgeführt
  5. Hard-Block (mkfs) → immer blockiert, egal ob confirm=true
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chatcli.agent.tools.shell import ShellTool


@pytest.fixture
def shell_tool():
    """ShellTool mit persistent=False (kein bwrap nötig) und tmp-CWD."""
    import tempfile
    return ShellTool(
        cwd=tempfile.mkdtemp(prefix="confirm-test-"),
        persistent=False,
        sandbox=False,
    )


# ── 1. Ohne Callback, ohne confirm → blockiert ─────────────────────

async def test_rm_rf_without_confirm_blocked(shell_tool):
    """rm -rf ohne confirm=true und ohne Callback → Error."""
    result = await shell_tool.run(command="rm -rf ./build")
    assert result.ok is False
    assert "Confirmation required" in result.error


# ── 2. MIT confirm_ask=Ja → ausgeführt ─────────────────────────────

async def test_rm_rf_confirm_ask_yes(shell_tool):
    """rm -rf mit confirm_ask-Callback der 'ja' sagt → Befehl läuft."""
    shell_tool._confirm_ask = AsyncMock(return_value="ja")
    # Erstelle ein Testverzeichnis, das dann gelöscht wird.
    import tempfile, os
    d = Path(shell_tool._base) / "build"
    d.mkdir()
    (d / "file.txt").write_text("test")

    result = await shell_tool.run(command="rm -rf ./build")
    assert result.ok is True
    assert not d.exists()
    # Callback wurde mit (command, reason) aufgerufen.
    shell_tool._confirm_ask.assert_called_once()


# ── 3. MIT confirm_ask=Nein → blockiert ────────────────────────────

async def test_rm_rf_confirm_ask_no(shell_tool):
    """rm -rf mit confirm_ask-Callback der 'nein' sagt → blockiert."""
    shell_tool._confirm_ask = AsyncMock(return_value="nein")

    result = await shell_tool.run(command="rm -rf ./build")
    assert result.ok is False
    assert "REJECTED" in result.error


# ── 4. MIT confirm=true (Modell hat selbst gefragt) → ausgeführt ───

async def test_rm_rf_with_confirm_true(shell_tool):
    """rm -rf mit confirm=true → direkt ausgeführt, kein Callback nötig."""
    import tempfile, os
    d = Path(shell_tool._base) / "build"
    d.mkdir()
    (d / "file.txt").write_text("test")

    result = await shell_tool.run(command="rm -rf ./build", confirm=True)
    assert result.ok is True
    assert not d.exists()


# ── 5. Hard-Block bleibt immer blockiert ───────────────────────────

async def test_hard_block_ignores_confirm(shell_tool):
    """mkfs wird NIEMALS ausgeführt, auch mit confirm=true."""
    shell_tool._confirm_ask = AsyncMock(return_value="ja")

    result = await shell_tool.run(command="mkfs.ext4 /dev/sda1", confirm=True)
    assert result.ok is False
    assert "BLOCKED" in result.error


# ── 6. Erlaubter Befehl braucht keine Bestätigung ──────────────────

async def test_allowed_command_no_confirm_needed(shell_tool):
    """ls -la läuft ohne confirm_ask und ohne confirm=true."""
    result = await shell_tool.run(command="echo hello")
    assert result.ok is True
    assert "hello" in result.output


# ── 7. Callback-Timeout → blockiert (nicht ausgeführt) ─────────────

async def test_confirm_ask_timeout_blocks(shell_tool):
    """Wenn der Callback timeoutt, wird der Befehl NICHT ausgeführt."""
    import asyncio
    async def slow_cb(cmd, reason):
        await asyncio.sleep(10)  # simuliert Timeout
        return "ja"

    shell_tool._confirm_ask = slow_cb
    shell_tool._confirm_ask_timeout = 1.0  # kurzer Timeout für Test

    result = await shell_tool.run(command="rm -rf ./build")
    assert result.ok is False
    assert "Confirmation required" in result.error
