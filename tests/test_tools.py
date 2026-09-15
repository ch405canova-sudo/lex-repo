"""Unit-Tests für die Tool-Implementierungen (Shell, File, Planner)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chatcli.agent.tools.shell import ShellTool, _needs_sudo
from chatcli.agent.tools.file_ops import (
    ReadFileTool,
    ListDirTool,
    WriteFileTool,
    SearchRegexTool,
)
from chatcli.agent.tools.apply_diff import ApplyDiffTool, _parse_blocks
from chatcli.agent.tools.patch import PatchLineTool
from chatcli.agent.tools.planner import PlannerTool
from chatcli.agent.tools.base import Tool, ToolRegistry, build_default_registry


# ── ShellTool ─────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_shell_echo():
    t = ShellTool(cwd="/tmp")
    r = await t.run(command="echo test123")
    assert r.ok
    assert "test123" in r.output


@pytest.mark.anyio
async def test_shell_no_double_wait_warning(caplog):
    """Regression: _pump() darf nach EOF nicht nochmal wait() aufrufen,
    wenn der Transport den Child bereits reaped hat. Sonst loggt asyncio
    'exit status already read' und meldet returncode 255."""
    import logging
    caplog.set_level(logging.WARNING, logger="asyncio")
    t = ShellTool(cwd="/tmp")
    r = await t.run(command="echo ok")
    assert r.ok
    assert not any(
        "exit status already read" in rec.getMessage() for rec in caplog.records
    )


@pytest.mark.anyio
async def test_shell_on_line_streams_lines():
    """on_line-Callback erhält jede Ausgabe-Zeile live (Live-Terminal-Box)."""
    seen: list[str] = []
    t = ShellTool(cwd="/tmp", on_line=seen.append)
    r = await t.run(command="echo zeile1; echo zeile2")
    assert r.ok
    assert "zeile1" in seen
    assert "zeile2" in seen
    # Reihenfolge erhalten
    assert seen.index("zeile1") < seen.index("zeile2")


@pytest.mark.anyio
async def test_shell_on_line_none_is_noop():
    """Ohne on_line läuft das Tool wie bisher (kein Crash)."""
    t = ShellTool(cwd="/tmp", on_line=None)
    r = await t.run(command="echo ok")
    assert r.ok
    assert "ok" in r.output


@pytest.mark.anyio
async def test_shell_bad_command_fails():
    t = ShellTool(cwd="/tmp")
    r = await t.run(command="exit 42")
    assert not r.ok
    assert "42" in r.output


@pytest.mark.anyio
async def test_shell_empty_command_rejected():
    t = ShellTool(cwd="/tmp")
    r = await t.run(command="")
    assert not r.ok


@pytest.mark.anyio
async def test_shell_timeout():
    t = ShellTool(cwd="/tmp")
    r = await t.run(command="sleep 5", timeout=1)
    assert not r.ok
    assert "Timeout" in r.error or "timeout" in r.error.lower()


# ── Sudo-Erkennung ────────────────────────────────────────────────

def test_needs_sudo_positive():
    assert _needs_sudo("sudo ls /root")
    assert _needs_sudo("  sudo   -v")
    assert _needs_sudo("FOO=bar sudo whoami")
    assert _needs_sudo("ls; sudo reboot")
    assert _needs_sudo("ls && sudo systemctl restart nginx")
    assert _needs_sudo("sudo apt install curl | tee log")


def test_needs_sudo_negative():
    assert not _needs_sudo("ls -la")
    assert not _needs_sudo("echo sudo")          # nur als Argument
    assert not _needs_sudo("cat file.txt")
    assert not _needs_sudo("")


@pytest.mark.anyio
async def test_shell_sudo_without_callback_fails_cleanly():
    """sudo ohne Callback (non-interaktiv) → saubere Fehlermeldung,
    kein Crash, kein Passwort im Output."""
    t = ShellTool(cwd="/tmp", sudo_ask=None)
    r = await t.run(command="sudo whoami")
    # Entweder: sudo läuft ohnehin ohne Passwort (NOPASSWD/Root) → ok,
    # oder: saubere Fehlermeldung wegen fehlender Abfrage.
    if not r.ok:
        assert "Passwort" in r.error or "sudo" in r.error


@pytest.mark.anyio
async def test_shell_sudo_async_callback_invoked():
    """sudo_ask ist ein async Callback: wird korrekt per await aufgerufen,
    Passwort landet im RAM-Cache und nie im Tool-Output."""
    calls = []

    async def ask(cmd: str):
        calls.append(cmd)
        return "geheim-pw-123"

    t = ShellTool(cwd="/tmp", sudo_ask=ask)
    r = await t.run(command="sudo whoami")
    # Passwort darf nie in der Tool-Ausgabe auftauchen
    assert "geheim-pw-123" not in (r.output or "")
    assert "geheim-pw-123" not in (r.error or "")
    if r.ok:
        # NOPASSWD/Root: sudo lief ohne Passwort → kein Callback nötig
        assert not calls
    else:
        assert "Passwort" in r.error or "sudo" in r.error
        # Der async-Callback wurde korrekt per await aufgerufen;
        # das (falsche) Test-PW wurde danach aus dem Cache geworfen.
        assert calls
        assert t._sudo_pw is None


@pytest.mark.anyio
async def test_shell_sudo_ask_timeout_clean_abort():
    """Passwort-Abfrage mit Timeout: wartet der Callback zu lange, bricht
    der sudo-Pfad sauber ab (keine Ewig-Blockade, saubere Meldung)."""
    import asyncio as _asyncio

    async def slow_ask(cmd: str):
        await _asyncio.sleep(5)  # länger als der Test-Timeout
        return "spätes-pw"

    t = ShellTool(cwd="/tmp", sudo_ask=slow_ask, sudo_ask_timeout=0.5)
    r = await t.run(command="sudo whoami")
    assert not r.ok
    assert "timed out" in (r.error or "")
    assert "sudo" in (r.error or "")
    # Passwort nie gesetzt (war nie da)
    assert t._sudo_pw is None


@pytest.mark.anyio
async def test_shell_sudo_ask_exception_reported_not_swallowed():
    """Regression (2026-09-09): Wenn der sudo-Callback eine Exception wirft,
    muss die Fehlermeldung den konkreten Grund nennen statt nur
    'kein interaktiver Modus'."""
    async def raising_ask(cmd: str):
        raise RuntimeError("Terminal gestört durch Rich Live-Box")

    t = ShellTool(cwd="/tmp", sudo_ask=raising_ask)
    r = await t.run(command="sudo whoami")
    assert not r.ok
    # Der konkrete Fehler muss in der Meldung stehen.
    assert "Terminal gestört" in (r.error or "")
    assert "failed" in (r.error or "")
    # Not the generic 'no interactive mode' message.
    assert "no interactive mode" not in (r.error or "").lower()


@pytest.mark.anyio
async def test_shell_sudo_ask_error_attr_set_on_exception():
    """_sudo_ask_error wird gesetzt, wenn der Callback eine Exception wirft."""
    async def raising_ask(cmd: str):
        raise ValueError("test-error-42")

    t = ShellTool(cwd="/tmp", sudo_ask=raising_ask)
    await t.run(command="sudo whoami")
    assert t._sudo_ask_error is not None
    assert "test-error-42" in t._sudo_ask_error


def test_missing_service_hint_triggers_install_path():
    """'Unit not found' / 'No such file' in stderr → Install-Lern-Hinweis
    (Pre-Check + apt install), damit der Agent nicht abbricht."""
    from chatcli.agent.tools.shell import _missing_service_hint

    assert "apt install" in _missing_service_hint(
        "Unit tor.service could not be found."
    )
    assert "apt install" in _missing_service_hint("No such file or directory")
    assert "command -v" in _missing_service_hint("not found")
    # Kein 'not found' → kein Hinweis
    assert _missing_service_hint("everything is fine") == ""


# ── File-Tools ────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_write_and_read(tmp_path):
    w = WriteFileTool(base_dir=str(tmp_path))
    r = await w.run(path="a.txt", content="Hallo")
    assert r.ok

    rd = ReadFileTool(base_dir=str(tmp_path))
    r = await rd.run(path="a.txt")
    assert r.ok
    assert r.output == "Hallo"


@pytest.mark.anyio
async def test_read_missing_file(tmp_path):
    rd = ReadFileTool(base_dir=str(tmp_path))
    r = await rd.run(path="nonexistent.txt")
    assert not r.ok


@pytest.mark.anyio
async def test_list_dir(tmp_path):
    (tmp_path / "x.txt").write_text("x")
    ld = ListDirTool(base_dir=str(tmp_path))
    r = await ld.run(path=".")
    assert r.ok
    assert "x.txt" in r.output


@pytest.mark.anyio
async def test_sandbox_escape_blocked(tmp_path):
    rd = ReadFileTool(base_dir=str(tmp_path))
    r = await rd.run(path="../../etc/shadow")
    assert not r.ok


@pytest.mark.anyio
async def test_search_regex(tmp_path):
    (tmp_path / "a.py").write_text("import os\ndef main():\n    pass\n")
    sr = SearchRegexTool(base_dir=str(tmp_path))
    r = await sr.run(pattern="def main", path=".")
    assert r.ok
    assert "main" in r.output


@pytest.mark.anyio
async def test_search_bad_regex(tmp_path):
    sr = SearchRegexTool(base_dir=str(tmp_path))
    r = await sr.run(pattern="[unclosed", path=".")
    assert not r.ok



# ── Absolute Pfade in resolve_in_roots ─────────────────────────────

@pytest.mark.anyio
async def test_absolute_path_in_memory_root(tmp_path):
    """Absoluter Pfad /home/.../memory/wiki/x.md muss funktionieren,
    wenn er unter einem Sandbox-Root liegt (Prompt nutzt absolute Pfade)."""
    cwd = tmp_path / "projekt"
    mem = tmp_path / "memory"
    (mem / "wiki").mkdir(parents=True)
    (mem / "wiki" / "index.md").write_text("test")
    cwd.mkdir(parents=True)

    rd = ReadFileTool(base_dir=str(cwd), extra_roots=[mem])
    # Absoluter Pfad ins Memory-Root
    abs_path = str(mem / "wiki" / "index.md")
    r = await rd.run(path=abs_path)
    assert r.ok
    assert r.output == "test"


@pytest.mark.anyio
async def test_absolute_path_outside_sandbox_blocked(tmp_path):
    """Absoluter Pfad AUSSERHALB der Sandbox muss blockiert werden."""
    cwd = tmp_path / "projekt"
    cwd.mkdir(parents=True)
    rd = ReadFileTool(base_dir=str(cwd))
    r = await rd.run(path="/etc/shadow")
    assert not r.ok


@pytest.mark.anyio
async def test_write_absolute_path_in_memory_root(tmp_path):
    """write_file mit absolutem Pfad ins Memory-Root."""
    cwd = tmp_path / "projekt"
    mem = tmp_path / "memory"
    (mem / "wiki").mkdir(parents=True)
    cwd.mkdir(parents=True)

    w = WriteFileTool(base_dir=str(cwd), extra_roots=[mem])
    abs_path = str(mem / "wiki" / "test.md")
    r = await w.run(path=abs_path, content="absolut")
    assert r.ok
    assert (mem / "wiki" / "test.md").read_text() == "absolut"


# ── ListDir mit prefer_existing_dir ──────────────────────────────────

@pytest.mark.anyio
async def test_list_dir_wiki_routed_to_memory(tmp_path):
    """list_dir('wiki') muss das Memory-Root zeigen, nicht CWD."""
    cwd = tmp_path / "projekt"
    mem = tmp_path / "memory"
    (mem / "wiki").mkdir(parents=True)
    (mem / "wiki" / "index.md").write_text("x")
    cwd.mkdir(parents=True)

    ld = ListDirTool(base_dir=str(cwd), extra_roots=[mem])
    r = await ld.run(path="wiki")
    assert r.ok
    assert "index.md" in r.output


# ── Multi-Root-Routing (CWD + Memory-Root) ────────────────────────

@pytest.mark.anyio
async def test_write_wiki_path_routed_to_memory_root(tmp_path):
    """wiki/... must be routed to the memory root, not the project CWD.

    Regression: Lex wrote wiki/lex-behavior/... into the CWD (/home/chaos),
    because resolve_in_roots blindly took the first root (CWD).
    """
    cwd = tmp_path / "projekt"
    mem = tmp_path / "memory"
    (mem / "wiki").mkdir(parents=True)
    (mem / "raw").mkdir(parents=True)
    cwd.mkdir(parents=True)

    w = WriteFileTool(base_dir=str(cwd), extra_roots=[mem])
    r = await w.run(path="wiki/agent-lessons/test.md", content="x")
    assert r.ok
    assert (mem / "wiki" / "agent-lessons" / "test.md").is_file()
    assert not (cwd / "wiki").exists()


@pytest.mark.anyio
async def test_write_plain_file_stays_in_cwd(tmp_path):
    """Neue Dateien OHNE Memory-Root-Verzeichnis bleiben im CWD."""
    cwd = tmp_path / "projekt"
    mem = tmp_path / "memory"
    (mem / "wiki").mkdir(parents=True)
    cwd.mkdir(parents=True)

    w = WriteFileTool(base_dir=str(cwd), extra_roots=[mem])
    r = await w.run(path="notiz.txt", content="hallo")
    assert r.ok
    assert (cwd / "notiz.txt").is_file()


@pytest.mark.anyio
async def test_apply_diff_existing_wiki_file_routed(tmp_path):
    """apply_diff auf wiki/index.md: Datei im Memory-Root gewinnt."""
    cwd = tmp_path / "projekt"
    mem = tmp_path / "memory"
    (mem / "wiki").mkdir(parents=True)
    (mem / "wiki" / "index.md").write_text("zeile A\n")
    cwd.mkdir(parents=True)

    d = ApplyDiffTool(base_dir=str(cwd), extra_roots=[mem])
    diff = (
        "<<<<<<< SEARCH\nzeile A\n=======\n"
        "zeile B\n>>>>>>> REPLACE"
    )
    r = await d.run(path="wiki/index.md", diff=diff)
    assert r.ok
    assert (mem / "wiki" / "index.md").read_text() == "zeile B\n"


@pytest.mark.anyio
async def test_apply_diff_malformed_footer_no_corruption(tmp_path):
    """Regression (2026-09-11): Fehlender/verkürzter Footer (6x '>' statt 7x)
    muss einen Fehler liefern und darf NICHT den Rest des Diff-Texts
    wörtlich in die Datei schreiben (das hatte loop.py korrupt gemacht)."""
    cwd = tmp_path / "projekt"
    cwd.mkdir(parents=True)
    f = cwd / "ziel.txt"
    f.write_text("zeile A\n")

    d = ApplyDiffTool(base_dir=str(cwd))
    diff = "<<<<<<< SEARCH\nzeile A\n=======\nneuer Text\n>>>>>>"
    r = await d.run(path="ziel.txt", diff=diff)
    assert not r.ok
    assert "REPLACE" in r.error
    # Datei bleibt unangetastet.
    assert f.read_text() == "zeile A\n"

    # Fehlendes '=======' wird ebenfalls sauber abgelehnt.
    diff2 = "<<<<<<< SEARCH\nzeile A\nneuer Text\n>>>>>>> REPLACE"
    r2 = await d.run(path="ziel.txt", diff=diff2)
    assert not r2.ok
    assert f.read_text() == "zeile A\n"


# ── PlannerTool ───────────────────────────────────────────────────


@pytest.mark.anyio
async def test_planner_basic():
    t = PlannerTool()
    r = await t.run(steps=["A", "B"], goal="Test")
    assert r.ok
    assert "A" in r.output


@pytest.mark.anyio
async def test_planner_empty_rejected():
    t = PlannerTool()
    r = await t.run(steps=[])
    assert not r.ok


@pytest.mark.anyio
async def test_planner_too_many_steps():
    t = PlannerTool()
    r = await t.run(steps=[f"S{i}" for i in range(11)])
    assert not r.ok


# ── Registry ──────────────────────────────────────────────────────



def test_registry_includes_semantic_search():
    reg = build_default_registry(
        enable_shell=False, enable_file_ops=False, enable_planner=False,
        enable_scrape=False, enable_web_search=False,
    )
    names = {t.name for t in reg.all()}
    assert "semantic_search" in names


def test_build_default_registry_all():
    reg = build_default_registry(
        enable_shell=True,
        enable_file_ops=True,
        enable_planner=True,
        shell_cwd="/tmp",
    )
    names = {t.name for t in reg.all()}
    assert {"shell", "read_file", "write_file", "list_dir", "search", "plan"} <= names


@pytest.mark.anyio
async def test_registry_unknown_tool():
    reg = ToolRegistry()
    r = await reg.execute("nonexistent", {})
    assert not r.ok
    assert "unknown tool" in r.error.lower()

def test_registry_descriptions():
    reg = build_default_registry(shell_cwd="/tmp")
    desc = reg.descriptions()
    assert "shell" in desc
    # Format rules are in the system_prompt (config.py),
    # here only the tool list is delivered.
    assert "Available tools" in desc


# ── PatchLineTool ─────────────────────────────────────────────────

@pytest.mark.anyio
async def test_patch_line_replace(tmp_path):
    """Einzeilen-Ersetzung via Zeilennummer — das Kernversprechen."""
    f = tmp_path / "a.py"
    f.write_text("zeile1\nzeile2\nzeile3\n", encoding="utf-8")
    t = PatchLineTool(base_dir=str(tmp_path))
    r = await t.run(path="a.py", line=2, replace="NEU")
    assert r.ok
    assert f.read_text(encoding="utf-8") == "zeile1\nNEU\nzeile3\n"


@pytest.mark.anyio
async def test_patch_line_out_of_range(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("a\nb\n", encoding="utf-8")
    t = PatchLineTool(base_dir=str(tmp_path))
    r = await t.run(path="a.py", line=99, replace="x")
    assert not r.ok
    assert "out of range" in r.error

@pytest.mark.anyio
async def test_patch_line_ambiguous_context(tmp_path):
    """Falsche Zeile + Kontext passt nicht → klare Fehlermeldung."""
    f = tmp_path / "a.py"
    f.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    t = PatchLineTool(base_dir=str(tmp_path))
    r = await t.run(path="a.py", line=2, replace="x", context="nicht_da")
    assert not r.ok
    assert "context" in r.error.lower()

@pytest.mark.anyio
async def test_patch_line_context_ok(tmp_path):
    """Kontext passt → Ersetzung erlaubt."""
    f = tmp_path / "a.py"
    f.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    t = PatchLineTool(base_dir=str(tmp_path))
    r = await t.run(path="a.py", line=2, replace="BETA", context="beta")
    assert r.ok
    assert "BETA" in f.read_text(encoding="utf-8")


@pytest.mark.anyio
async def test_patch_line_no_file(tmp_path):
    t = PatchLineTool(base_dir=str(tmp_path))
    r = await t.run(path="missing.py", line=1, replace="x")
    assert not r.ok
    assert "not found" in r.error.lower()


@pytest.mark.anyio
async def test_patch_line_multi_replace(tmp_path):
    """Mehrzeilen-Ersetzung: line..line+n-1 wird ersetzt."""
    f = tmp_path / "a.py"
    f.write_text("l1\nl2\nl3\nl4\n", encoding="utf-8")
    t = PatchLineTool(base_dir=str(tmp_path))
    r = await t.run(path="a.py", line=2, replace="X1\nX2", count=2)
    assert r.ok
    assert f.read_text(encoding="utf-8") == "l1\nX1\nX2\nl4\n"


@pytest.mark.anyio
async def test_patch_in_registry():
    reg = build_default_registry(shell_cwd="/tmp", enable_file_ops=True)
    assert "patch" in {t.name for t in reg.all()}
