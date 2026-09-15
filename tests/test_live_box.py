"""Tests für die Live-Terminal-Box (repl._LiveShellBox) — ohne Live-View.

Die Box wird ohne start() getestet, damit kein Rich-Live-Rendering im
Test-Terminal läuft; die Logik (Transkript, Shell-Zeilen, Exit-Code-
Erkennung) bleibt geprüft.
"""

from __future__ import annotations

from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console
from chatcli.repl import _LiveShellBox, _SHELL_CMD_RE, _ERR_LINE_RE


def _box() -> _LiveShellBox:
    return _LiveShellBox(Console(record=True, file=open("/dev/null", "w")))


def test_shell_command_regex_extracts_command():
    m = _SHELL_CMD_RE.search('shell  command="ls -la", timeout=30')
    assert m is not None
    assert m.group(1) == '"ls -la"'
    m2 = _SHELL_CMD_RE.search("shell  command=ls -la")
    assert m2 is not None
    assert m2.group(1) == "ls -la"


def test_error_line_regex():
    assert _ERR_LINE_RE.search("Error: not found")
    assert _ERR_LINE_RE.search("Permission denied")
    assert not _ERR_LINE_RE.search("alles gut")


def test_note_tool_and_output_transcript():
    """Shell-Events schreiben raw auf stdout, nicht ins Transkript."""
    b = _box()
    b.note("tool", 'shell  command="ls -la"', tool="shell")
    b.note("tool_output", "total 2\nfile1\n[exit_code: 0]", tool="shell")
    # Shell-Events erzeugen KEINE Transkript-Einträge (raw stdout).
    assert len(b._transcript) == 0
    assert b._active == 0


def test_shell_lines_buffered_only_while_active():
    """Im Raw-Modus werden Zeilen auf stdout geschrieben, nicht in _lines."""
    import io
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        b = _box()
        b.shell_line("vorher")  # keine aktive Box → wird ignoriert
        assert len(b._lines) == 0
        b.note("tool", 'shell  command="echo hi"', tool="shell")
        b.shell_line("hi")
        # Im Raw-Modus: "hi" landet auf stdout, nicht in _lines.
        assert "hi" not in b._lines
        out = buf.getvalue()
        assert "hi" in out
        b.note("tool_output", "[exit_code: 0]", tool="shell")
        assert b._active == 0
    finally:
        sys.stdout = old_stdout


def test_shell_error_decrements_active():
    b = _box()
    b.note("tool", 'shell  command="false"', tool="shell")
    b.note("tool_error", "FEHLER: kaputt", tool="shell")
    assert b._active == 0


def test_render_spinner_frame_no_typeerror():
    """Regression: Spinner-Frame muss rendern (rich-15 Signatur: style, kein spinner_style)."""
    b = _box()
    from rich.console import Console
    out = Console(record=True, width=80)
    out.print(b._render())
    assert out.export_text().strip()


def test_render_shell_frame_no_typeerror():
    b = _box()
    b.note("tool", 'shell  command="echo hi"', tool="shell")
    from rich.console import Console
    out = Console(record=True, width=80)
    out.print(b._render())
    assert out.export_text().strip()


def test_answer_markdown_themed_console_renders():
    """Regression: Rich >= 14 hat Markdown(theme=) entfernt — das Theme
    sitzt jetzt auf _themed_console. Antwort-Pfad muss rendern."""
    import io
    from rich.markdown import Markdown
    from rich.panel import Panel
    from chatcli.repl import _themed_console, _ANSWER_THEME

    md = Markdown("# Titel\n\n**fett** und `code`\n\n- Punkt 1")
    buf = io.StringIO()
    old_file = _themed_console.file
    _themed_console.file = buf
    try:
        _themed_console.print(md)
        _themed_console.print(
            Panel(md, title="✦ Antwort", border_style="bright_green")
        )
    finally:
        _themed_console.file = old_file
    assert buf.getvalue().strip()
    from rich.style import Style
    assert _ANSWER_THEME.styles["markdown.h1"] == Style.parse("bold bright_yellow")


def _span_styles(t) -> list[str]:
    """Style-Namen der Rich-Text-Spans (rich 15: _spans, Span(start, end, style))."""
    spans = getattr(t, "_spans", None) or []
    return [str(s.style) for s in spans if getattr(s, "style", None)]


def test_note_tool_line_has_no_literal_markup():
    """Regression: Nicht-Shell-Tools erzeugen Transkript-Zeilen ohne Literal-Markup."""
    b = _box()
    b.note("tool", 'web_search  query="test"', tool="web_search")
    t = b._transcript[-1]
    assert "[cyan]" not in t.plain and "[/]" not in t.plain
    assert "⚙" in t.plain
    assert any("cyan" in s for s in _span_styles(t))


def test_note_warning_has_no_literal_markup():
    b = _box()
    b.note("warning", "History komprimiert (Context-Rot-Schutz)")
    t = b._transcript[-1]
    assert "[yellow]" not in t.plain and "[/]" not in t.plain
    assert any("yellow" in s for s in _span_styles(t))


def test_note_shell_exit_status_no_literal_markup():
    """Shell-Exit-Status wird raw auf stdout geschrieben, nicht ins Transkript."""
    import io
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        b = _box()
        b.note("tool", 'shell  command="ls -la"', tool="shell")
        b.note("tool_output", "[exit_code: 1]", tool="shell")
        # Kein Transkript-Eintrag für Shell-Exit (raw stdout).
        assert len(b._transcript) == 0
        out = buf.getvalue()
        assert "exit 1" in out
    finally:
        sys.stdout = old_stdout


# ── Passwort-Prompt-History-Leak (sudo) ────────────────────────────

def test_pw_session_uses_in_memory_history_not_file():
    """Regression: Die sudo-Passwort-Session MUSS InMemoryHistory nutzen.

    prompt_toolkit speichert JEDER Prompt in die Session-History — auch
    password=True-Prompts. Würde _make_pw_session() eine FileHistory
    (history.txt) verwenden, läge das sudo-Passwort im Klartext auf Disk
    und wäre per ↑/↓ wiederabrufbar (Enter sendet es als Chat-Nachricht).
    """
    from prompt_toolkit.history import InMemoryHistory, FileHistory

    from chatcli.repl import _make_pw_session

    session = _make_pw_session()
    assert isinstance(session.history, InMemoryHistory)
    assert not isinstance(session.history, FileHistory)
