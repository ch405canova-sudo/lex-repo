"""Tests für das Top-4-Paket: apply_diff, stateful shell, parallel loop, plan-injection."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chatcli.agent.tools.shell import ShellTool
from chatcli.agent.tools.apply_diff import ApplyDiffTool, _parse_blocks
from chatcli.agent.tools.planner import PlannerTool
from chatcli.agent.tools.base import Tool, ToolRegistry, build_default_registry


# ── _parse_blocks ────────────────────────────────────────────────────

def test_parse_blocks_single():
    diff = (
        "<<<<<<< SEARCH\n"
        "def foo():\n"
        "    pass\n"
        "=======\n"
        "def foo():\n"
        "    return 1\n"
        ">>>>>>> REPLACE"
    )
    blocks, err = _parse_blocks(diff)
    assert err == ""
    assert len(blocks) == 1
    assert blocks[0][0] == "def foo():\n    pass"
    assert blocks[0][1] == "def foo():\n    return 1"


def test_parse_blocks_multiple():
    diff = (
        "<<<<<<< SEARCH\nA\n=======\nB\n>>>>>>> REPLACE\n"
        "irgendwas dazwischen\n"
        "<<<<<<< SEARCH\nC\n=======\nD\n>>>>>>> REPLACE"
    )
    blocks, err = _parse_blocks(diff)
    assert err == ""
    assert len(blocks) == 2
    assert blocks[0] == ("A", "B")
    assert blocks[1] == ("C", "D")


def test_parse_blocks_empty():
    blocks, err = _parse_blocks("kein block hier")
    assert err == ""
    assert blocks == []


def test_parse_blocks_malformed_middle():
    """Fehlendes '=======' → klarer Fehler statt stiller Datei-Korruption."""
    blocks, err = _parse_blocks("<<<<<<< SEARCH\nA\nB\n>>>>>>> REPLACE")
    assert blocks == []
    assert "=======" in err


def test_parse_blocks_malformed_footer():
    """Verkürzter Footer (6x '>') → klarer Fehler statt Datei-Korruption."""
    blocks, err = _parse_blocks("<<<<<<< SEARCH\nA\n=======\nB\n>>>>>>")
    assert blocks == []
    assert "REPLACE" in err


# ── ApplyDiffTool ────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_apply_diff_basic(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\nbeta\ngamma\n")
    t = ApplyDiffTool(base_dir=str(tmp_path))
    diff = (
        "<<<<<<< SEARCH\nbeta\n=======\nBETA\n>>>>>>> REPLACE"
    )
    r = await t.run(path="a.txt", diff=diff)
    assert r.ok
    assert (tmp_path / "a.txt").read_text() == "alpha\nBETA\ngamma\n"


@pytest.mark.anyio
async def test_apply_diff_not_found(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\n")
    t = ApplyDiffTool(base_dir=str(tmp_path))
    r = await t.run(
        path="a.txt",
        diff="<<<<<<< SEARCH\nnicht_da\n=======\nneue\n>>>>>>> REPLACE",
    )
    assert not r.ok
    assert "not found" in r.error.lower()


@pytest.mark.anyio
async def test_apply_diff_ambiguous(tmp_path):
    (tmp_path / "a.txt").write_text("x\nx\n")
    t = ApplyDiffTool(base_dir=str(tmp_path))
    r = await t.run(
        path="a.txt",
        diff="<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE",
    )
    assert not r.ok
    assert "multiple times" in r.error


@pytest.mark.anyio
async def test_apply_diff_replace_all(tmp_path):
    (tmp_path / "a.txt").write_text("foo bar foo")
    t = ApplyDiffTool(base_dir=str(tmp_path))
    r = await t.run(
        path="a.txt",
        diff="<<<<<<< SEARCH\nfoo\n=======\nbar\n>>>>>>> REPLACE",
        replace_all=True,
    )
    assert r.ok
    assert (tmp_path / "a.txt").read_text() == "bar bar bar"


@pytest.mark.anyio
async def test_apply_diff_missing_file(tmp_path):
    t = ApplyDiffTool(base_dir=str(tmp_path))
    r = await t.run(
        path="kein.txt",
        diff="<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE",
    )
    assert not r.ok
    assert "not found" in r.error.lower()


@pytest.mark.anyio
async def test_apply_diff_sandbox_escape_blocked(tmp_path):
    t = ApplyDiffTool(base_dir=str(tmp_path))
    r = await t.run(
        path="../../etc/passwd",
        diff="<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE",
    )
    assert not r.ok


# ── Stateful Shell ───────────────────────────────────────────────────

@pytest.mark.anyio
async def test_shell_stateful_env_persists():
    """Env-Variablen überleben mehrere Tool-Calls (stateful bash)."""
    t = ShellTool(cwd="/tmp")
    try:
        r1 = await t.run(command="export CHATCLI_TEST_VAR=lebenslang")
        assert r1.ok
        r2 = await t.run(command="echo $CHATCLI_TEST_VAR")
        assert r2.ok
        assert "lebenslang" in r2.output
    finally:
        await t.close()


@pytest.mark.anyio
async def test_shell_stateful_cd_persists():
    """cd überlebt mehrere Tool-Calls."""
    t = ShellTool(cwd="/tmp")
    try:
        r1 = await t.run(command="mkdir -p /tmp/chatcli_cd_test && cd /tmp/chatcli_cd_test")
        assert r1.ok
        r2 = await t.run(command="pwd")
        assert r2.ok
        assert "chatcli_cd_test" in r2.output
    finally:
        await t.close()


@pytest.mark.anyio
async def test_shell_reset():
    """reset=true startet eine frische Shell (alte Env-Vars weg)."""
    t = ShellTool(cwd="/tmp")
    try:
        await t.run(command="export CHATCLI_RESET_VAR=alt")
        r = await t.run(command="echo $CHATCLI_RESET_VAR", reset=True)
        assert r.ok
        assert "alt" not in r.output
    finally:
        await t.close()


@pytest.mark.anyio
async def test_shell_exit_code_in_output():
    t = ShellTool(cwd="/tmp")
    try:
        r = await t.run(command="false")
        assert not r.ok
        assert "exit_code: 1" in r.output
    finally:
        await t.close()


# ── PlannerTool (stateful) ───────────────────────────────────────────

@pytest.mark.anyio
async def test_planner_complete_action():
    t = PlannerTool()
    await t.run(steps=["A", "B", "C"], goal="Test")
    r = await t.run(action="complete", step=1)
    assert r.ok
    assert "✓" in r.output
    assert "1. A" in r.output


@pytest.mark.anyio
async def test_planner_complete_without_plan_rejected():
    t = PlannerTool()
    r = await t.run(action="complete", step=1)
    assert not r.ok


@pytest.mark.anyio
async def test_planner_complete_out_of_range():
    t = PlannerTool()
    await t.run(steps=["A"])
    r = await t.run(action="complete", step=5)
    assert not r.ok


@pytest.mark.anyio
async def test_planner_complete_without_step_marks_all_done():
    """Regression: complete OHNE step-Arg (wie es das Modell am
    Session-Ende sendet) markiert ALLE Schritte als erledgt — sonst
    bleibt der Plan 'aktiv' und treibt eine Wiederholungs-Schleife an.
    Seit 2026-09-09: status_block() ist bei abgeschlossenem Plan LEER
    (Fix 4: kein Kontext-Rauschen)."""
    t = PlannerTool()
    await t.run(steps=["A", "B", "C"], goal="Test")
    r = await t.run(action="complete")
    assert r.ok
    # Alle Schritte erledgt → status_block ist leer (Fix 4).
    block = t.status_block()
    assert block == ""
    assert "complete" in r.output.lower()


@pytest.mark.anyio
async def test_planner_status_block():
    t = PlannerTool()
    assert t.status_block() == ""
    await t.run(steps=["A", "B"], goal="Ziel")
    await t.run(action="complete", step=1)
    block = t.status_block()
    assert "Ziel" in block
    assert "✓ 1. A" in block
    assert "• 2. B" in block


@pytest.mark.anyio
async def test_planner_sudo_step_without_precheck_warns():
    """Pre-Check-Zwang: sudo-/Dienst-Step ohne vorgelagerten
    Existenz-Check → Warnung im Output (Plan läuft trotzdem an)."""
    t = PlannerTool()
    r = await t.run(
        steps=["Script schreiben", "sudo systemctl start tor"],
        goal="Tor starten",
    )
    assert r.ok
    assert "Pre-check" in r.output
    assert "sudo apt install" in r.output


@pytest.mark.anyio
async def test_planner_sudo_step_with_precheck_no_warning():
    """Mit vorgelagertem Pre-Check-Step → keine Warnung mehr."""
    t = PlannerTool()
    r = await t.run(
        steps=[
            "Pre-Check: command -v tor + systemctl is-active tor",
            "sudo systemctl start tor",
        ],
        goal="Tor starten (sauber)",
    )
    assert r.ok
    assert "Pre-Check-Zwang" not in r.output


@pytest.mark.anyio
async def test_planner_non_sudo_plan_no_warning():
    """Plan ohne sudo/Dienst-Start → keine Pre-Check-Warnung."""
    t = PlannerTool()
    r = await t.run(steps=["Quellen lesen", "Zusammenfassen"], goal="Recherche")
    assert r.ok
    assert "Pre-Check-Zwang" not in r.output


# ── Planner: status_block bei abgeschlossenem Plan (Fix 4) ───────────

@pytest.mark.anyio
async def test_planner_status_block_empty_when_all_done():
    """Regression (2026-09-09): [Aktiver Plan] wird NICHT injiziert, wenn
    alle Schritte erledigt sind — sonst kontaminiert der abgeschlossene
    Plan den Kontext bei einem neuen Task."""
    t = PlannerTool()
    await t.run(steps=["A", "B"], goal="Test")
    await t.run(action="complete", step=1)
    # Noch nicht alle erledigt → Block vorhanden.
    assert t.status_block() != ""
    await t.run(action="complete", step=2)
    # Alle erledigt → Block leer.
    assert t.status_block() == ""


# ── Planner: reset-Action (Fix 5) ─────────────────────────────────────

@pytest.mark.anyio
async def test_planner_reset_clears_plan():
    """Regression (2026-09-09): action='reset' löscht den Plan komplett,
    damit ein neuer Task nicht vom alten Plan kontaminiert wird."""
    t = PlannerTool()
    await t.run(steps=["Altes Thema"], goal="Alt")
    assert t.status_block() != ""

    r = await t.run(action="reset")
    assert r.ok
    assert "deleted" in r.output.lower()
    # Plan ist jetzt leer.
    assert t.status_block() == ""
    assert t.last_plan == []


# ── Registry (erweitert) ──────────────────────────────────────────────

def test_build_default_registry_includes_apply_diff():
    reg = build_default_registry(
        enable_shell=True,
        enable_file_ops=True,
        enable_planner=True,
        shell_cwd="/tmp",
    )
    names = {t.name for t in reg.all()}
    assert "apply_diff" in names
