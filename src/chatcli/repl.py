"""Interaktiver REPL mit prompt_toolkit (History, Farben, Slash-Kommandos).

Falls prompt_toolkit nicht installiert ist (z. B. System-Python ohne venv),
greift automatisch ein einfacher input()-Fallback ohne Autocomplete.
"""

from __future__ import annotations

import asyncio
import getpass
import html
import json
import logging
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.history import FileHistory, InMemoryHistory
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False

from .config import Config
from .llama_client import LlamaClient
from .agent.loop import AgentLoop
from .agent.tools import ToolRegistry
from .output import (
    banner, user_msg,
    error_msg, info_msg, slash_help,
)

log = logging.getLogger(__name__)

console = Console()

HISTORY_PATH = Path.home() / ".local" / "share" / "chatcli" / "history.txt"
HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

# Auto-Session-Dump (Forensik): nach JEDEM Turn landet die komplette
# Agent-History hier. Lex ist ein LLM — sein Gedächtnis ist nur der
# RAM-Buffer; nach Crash/Abbruch wäre die Session sonst unerreichbar.
SESSION_DIR = HISTORY_PATH.parent / "sessions"
SESSION_KEEP = 20

# Antwort wird erst als Panel gerendert, wenn sie länger als das ist.
_ANSWER_PANEL_MIN_LINES = 4

class _AnswerBuffer:
    """Sammelt nur den Text SEIT dem letzten Tool-Event.

    Der Loop yieldet jede Prosa live — auch Zwischen-Prosa zwischen
    Tool-Calls. Ohne dieses Buffer würde die finale Antwort vermischt
    Zwischengedanken + Endergebnis enthalten. Bei jedem Tool-Event wird
    der Puffer geleert; am Ende steht nur die finale Antwort drin.
    """

    def __init__(self) -> None:
        self._parts: list[str] = []

    def on_event(self, kind: str, text: str) -> None:
        if kind in ("tool", "tool_output", "tool_error"):
            self._parts.clear()

    def add(self, chunk: str) -> None:
        self._parts.append(chunk)

    @property
    def answer(self) -> str:
        return "".join(self._parts)


import re

# Regex für "wichtige" Terminal-Zeilen (Fehler/Warnings) → rot markieren.
_ERR_LINE_RE = re.compile(
    r"(?i)\b(error|failed|failure|denied|permission|refused|not found|no such|"
    r"traceback|exception|fatal|panic|segfault|killed|timed out|timeout)\b"
)


class _LiveShellBox:
    """Zweistufiges Rendering: permanent + kleiner Live-Frame.

    Bestätigte Events (Thinking, Tool-Calls, Tool-Outputs, Warnungen)
    werden PERMANENT auf stdout geschrieben — wie in einem echten
    Terminal (natürliches Scrollen, kein Re-Render). Nur der aktive
    Teil (Spinner oder laufende Shell-Box) bleibt als kleiner Live-Frame.

    Damit entfällt das Rich-Live-Flackern: Ein ``Live``-Frame flackert,
    wenn er höher ist als das Terminal — dann verschiebt sich das
    Terminal beim Scrollen und die "Cursor-N-Zeilen-hoch"-Rechnung von
    Richs Redraw landet versetzt. Hier kann der Frame nie größer sein
    als das Terminal (dynamische Höhengrenze), und permanenter Text
    wird außerhalb des Live-Modus geschrieben.
    """

    _MAX_LINES = 14
    _MAX_LINE_LEN = 240

    def __init__(self, console: Console) -> None:
        self._console = console
        self._lines: deque[str] = deque(maxlen=self._MAX_LINES)
        self._command = ""
        self._active = 0  # parallele Shell-Calls zählen
        self._live: Optional[Live] = None
        # Letzte permanent geschriebene Renderables (für Tests + Debugging).
        self._permanent: list = []
        # Raw-Terminal-Modus: Shell-Ausgabe wird direkt auf stdout
        # geschrieben (wie in einem echten Terminal). Die Live-Ansicht
        # ist währenddessen pausiert und danach wieder aktiv.
        self._raw_mode = False
        # Throttle für Fallback-Modus (parallele Calls in Rich-Box).
        self._last_render = 0.0

    def is_active(self) -> bool:
        return self._live is not None

    def start(self) -> None:
        self._lines.clear()
        self._command = ""
        self._active = 0
        self._raw_mode = False
        self._permanent = []
        self._last_render = 0.0
        self._start_live_quiet()

    def stop(self) -> None:
        if self._live is not None:
            # Letztes Frame leeren, damit kein verwaister Spinner/Box
            # zwischen den Turns hängen bleibt.
            self._live.update(Text(""))
            self._live.stop()
            self._live = None
        # Terminal-State-Reset: Richs Live-Rendering kann den Cursor
        # ausblenden (?25l) und Attribute setzen (?25h = Cursor sichtbar,
        # \x1b[0m = alle SGR-Attribute zurücksetzen). prompt_toolkit
        # braucht einen sauberen Terminal-State für prompt_async().
        sys.stdout.write("\x1b[?25h\x1b[0m")
        sys.stdout.flush()

    def _push(self) -> None:
        """Neues Frame an die Live-Ansicht geben (sofort sichtbar)."""
        if self._live is not None:
            self._live.update(self._render())

    def _print_permanent(self, renderable) -> None:
        """Ein bestätigtes Event permanent auf stdout schreiben.

        Rich 15 unterstützt "print while live": Der Live-Frame wird
        automatisch nach unten geschoben, wenn permanent Text daneben
        geschrieben wird — kein stop/start-Zyklus nötig (weniger
        Terminal-State-Wechsel = stabiler, kein Flackern).
        """
        self._permanent.append(renderable)
        if len(self._permanent) > 40:
            del self._permanent[: len(self._permanent) - 40]
        self._console.print(renderable)

    @property
    def _transcript(self) -> list:
        """Kompatibilität: alias für die permanenten Renderables."""
        return self._permanent

    # -- Transkript-Events ------------------------------------------------

    def note(self, kind: str, text: str, tool: str = "") -> None:
        """Ein Agent-Event aufnehmen und permanent rendern."""
        if kind == "thinking":
            raw = text.strip()
            if len(raw) > 500:
                raw = raw[:497] + "…"
            self._print_permanent(
                Panel(
                    Text(raw),
                    title="[dim]💭 Thinking[/dim]",
                    border_style="dim",
                    expand=True,
                    padding=(0, 1),
                )
            )
        elif kind == "tool":
            if tool == "shell":
                m = _SHELL_CMD_RE.search(text)
                cmd = (m.group(1) if m else text).strip()
                self._active += 1
                if self._active == 1:
                    self._command = cmd
                    self._lines.clear()
                    # Raw-Terminal-Modus aktivieren: Live-Ansicht pausieren,
                    # Befehlszeile direkt auf stdout schreiben.
                    if not self._raw_mode:
                        self._stop_live_quiet()
                        self._raw_mode = True
                    sys.stdout.write(f"\n$ {cmd}\n")
                    sys.stdout.flush()
            else:
                # Nicht-Shell-Tool: permanente Transkript-Zeile.
                line = Text("  ")
                line.append("⚙ ", style="cyan")
                line.append(text)
                self._print_permanent(line)
        elif kind == "tool_output":
            if tool == "shell":
                if self._active > 0:
                    self._active -= 1
                # Exit-Status direkt auf stdout (Raw-Terminal-Stil).
                from .agent.tools.base import EXIT_CODE_RE
                code_m = EXIT_CODE_RE.search(text)
                code = code_m.group(1) if code_m else "?"
                ok = code == "0"
                mark = "✓" if ok else "✗"
                sys.stdout.write(f"  {mark} exit {code}\n")
                sys.stdout.flush()
                # Raw-Modus beenden, wenn kein paralleler Shell-Call mehr aktiv.
                if self._active <= 0 and self._raw_mode:
                    self._raw_mode = False
                    self._start_live_quiet()
            else:
                lines = [l for l in text.strip().splitlines() if l.strip()]
                preview = " | ".join(l for l in lines[:3])
                if len(preview) > 200:
                    preview = preview[:197] + "..."
                if not preview:
                    preview = "(empty)"
                line = Text("    ")
                line.append("↳ ", style="dim")
                line.append(preview)
                self._print_permanent(line)
        elif kind == "tool_error":
            if tool == "shell" and self._active > 0:
                self._active -= 1
                # Fehler direkt auf stdout (Raw-Terminal-Stil).
                sys.stdout.write(f"  ✗ {text[:200]}\n")
                sys.stdout.flush()
                if self._active <= 0 and self._raw_mode:
                    self._raw_mode = False
                    self._start_live_quiet()
        elif kind == "warning":
            line = Text("  ")
            line.append("⚠ ", style="yellow")
            line.append(text)
            self._print_permanent(line)

    # -- Live-Terminal-Box (Shell-Zeilen) ----------------------------------

    def shell_line(self, line: str) -> None:
        if self._active <= 0:
            return
        if self._raw_mode:
            # Echtes Terminal: Zeile raw auf stdout (natürliches Scrollen).
            sys.stdout.write(line[: self._MAX_LINE_LEN] + "\n")
            sys.stdout.flush()
        else:
            # Fallback (z. B. parallele Calls): Rich-Box mit Throttling.
            self._lines.append(line[: self._MAX_LINE_LEN])
            now = time.monotonic()
            if now - self._last_render >= 0.15:
                self._last_render = now
                self._push()

    def _stop_live_quiet(self) -> None:
        """Live-Ansicht stoppen, ohne leeres Frame zu schreiben."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _start_live_quiet(self) -> None:
        """Live-Ansicht neu starten (nur der kleine aktive Frame)."""
        if self._live is None:
            self._live = Live(
                self._render(), console=self._console,
                auto_refresh=True, vertical_overflow="hidden",
            )
            self._live.start(refresh=True)

    def _render(self):
        """Nur der AKTIVE Teil (Spinner oder laufende Shell-Box).

        Die Shell-Box wird dynamisch auf die Terminalhöhe begrenzt,
        damit der Live-Frame nie höher ist als das Fenster — sonst
        scrollt das Terminal beim Redraw und Richs Cursor-Positionierung
        landet versetzt (Flackern).
        """
        if self._active > 0:
            term_h = self._console.size.height
            max_lines = max(3, min(self._MAX_LINES, term_h - 3))
            shown = list(self._lines)[-max_lines:]
            text = Text()
            for ln in shown:
                style = "red" if _ERR_LINE_RE.search(ln) else "white"
                text.append(ln, style=style)
                text.append("\n")
            if not shown:
                text.append("… running …", style="dim")
            return Panel(
                text,
                title=f"[bold bright_blue]⚡ {self._command}[/]",
                border_style="bright_blue",
                expand=True,
                padding=(0, 1),
            )
        from rich.spinner import Spinner
        return Spinner("dots", "Lex thinking…", style="bright_green")


# Reflection-Step (Session-Consolidation, MemGPT-Stil): Am Session-Ende
# eine dedizierte LLM-Passage, die die Session durchsucht und daraus
# (a) neue Lessons, (b) Fakten für die Wiki, (c) Archivierungs-Vorschläge
# ableitet — statt nur reaktiv auf Fehler-Hinweise zu schreiben.
_REFLECTION_PROMPT = (
    "[Session Reflection] The session ends now. Perform a short "
    "consolidation (max. 3 tool calls):\n"
    "1. Check the session: Is there a SOLVED ERROR whose solution is "
    "not yet a lesson? If so, write ONE short lesson "
    "(5-15 lines: Error / Cause / Solution) as a new file in the wiki "
    "topic agent-lessons (path relative to the memory root) and register "
    "it in index.md + log.md.\n"
    "2. Is there a reusable SUCCESS PATTERN (unconventional approach that "
    "worked)? Then max. 5 lines success lesson.\n"
    "3. No new info? Reply ONLY with: NONE\n"
    "End with ONE line: what you wrote (or NONE)."
)


# Custom Theme für die finale Antwort: wichtige Bereiche (Überschriften,
# Code, Fett, Links) in kräftigen Farben — der Rest bleibt normal.
from rich.theme import Theme

_ANSWER_THEME = Theme({
    "markdown.h1": "bold bright_yellow",
    "markdown.h2": "bold bright_yellow",
    "markdown.h3": "bold yellow",
    "markdown.h4": "bold magenta",
    "markdown.code": "bold bright_cyan on grey23",
    "markdown.code_block": "bright_cyan on grey23",
    "markdown.inline_code": "bold bright_magenta on grey23",
    "markdown.link": "bright_blue underline",
    "markdown.link_text": "bright_blue",
    "markdown.bold": "bold bright_white",
    "markdown.strong": "bold bright_yellow",
    "markdown.emphasis": "italic bright_yellow",
    "markdown.list.item": "white",
    "markdown.hr": "dim",
})
# Rich >= 14: Theme gehört an die Console, nicht an Markdown — eigene
# Console für die finale Antwort (Theming ohne Seiteneffekte auf
# die übrigen console.print-Aufrufe).
_themed_console = Console(theme=_ANSWER_THEME)


# Regex, um aus dem "tool"-Event den Shell-Befehl für die Box-Titel zu holen.
_SHELL_CMD_RE = re.compile(r"command=(.*?)(?:,\s\w+=|$)")


def _make_pw_session() -> "PromptSession":
    """PromptSession NUR für Passwort-Prompts (sudo-Abfrage).

    WICHTIG: InMemoryHistory, NICHT die FileHistory der Main-Session!
    prompt_toolkit speichert JEDER abgeschlossenen Prompt in die
    Session-History — auch password=True-Prompts. Ohne diese Trennung
    würde das sudo-Passwort in history.txt landen und per ↑/↓ als
    normale History-Eingabe wiederabrufbar sein (Enter sendet es dann
    als Chat-Nachricht). InMemory = kein Disk, kein Pfeiltasten-Zugriff,
    verfliegt mit der Session.
    """
    return PromptSession(history=InMemoryHistory())


class ChatCLI:
    """Der Haupt-REPL-Loop."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client: Optional[LlamaClient] = None
        self._loop: Optional[AgentLoop] = None
        self._registry: Optional[ToolRegistry] = None
        # Kontext-Nutzung des Servers (live abgefragt, immer sichtbar)
        self._ctx_usage: Optional[dict] = None
        # PromptSession wird in _run_interactive gesetzt — die
        # sudo-Abfrage braucht sie für die verdeckte Passwort-Eingabe.
        self._session: Optional["PromptSession"] = None
        # Separate Session NUR für Passwort-Prompts (InMemoryHistory!):
        # prompt_toolkit speichert JEDER Eingabe in die History — auch
        # Passwort-Prompts. Ohne diese Trennung würde das sudo-Passwort
        # in history.txt landen und wäre per ↑/↓ wiederabrufbar.
        self._pw_session: Optional["PromptSession"] = None
        # Live-Terminal-Box: zeigt laufende Shell-Befehle zeilenweise an.
        self._live_box = _LiveShellBox(console)

    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Startet die komplette CLI (non-interaktiver oder interaktiver Modus)."""
        from .agent.tools import build_default_registry
        from .llama_client import LlamaClient

        self._client = LlamaClient(self._config)

        # Health-Check
        if not await self._client.health():
            error_msg(
                f"llama-server not reachable at {self._config.base_url}.\n"
                f"  → Start a llama-server (OpenAI-compatible) on that address first."
            )
            await self._client.close()
            sys.exit(1)

        info_msg(f"Connected to {self._config.base_url}  (model: {self._config.model})")

        # Sampling-Parameter + Kontextgrösse: llama-server ist Single Source
        # of Truth (ai.sh setzt --temp/--top-k/--ctx …). Explizite
        # CHATCLI_*-Overrides des Nutzers bleiben unverändert.
        params = await self._client.server_params()
        if params:
            self._config.sync_sampling_from_server(params)
            n_ctx = params.get("n_ctx")
            info_msg(
                f"Sampling from server: temp={self._config.temperature} "
                f"top_k={self._config.top_k} top_p={self._config.top_p} "
                f"min_p={self._config.min_p}"
                + (f" · Context: {n_ctx} tokens" if isinstance(n_ctx, int) else "")
            )

        # Registry + Loop (sudo-Abfrage geht über die REPL, damit das
        # Passwort verdeckt eingegeben wird)
        self._registry = build_default_registry(
            enable_shell=self._config.enable_shell,
            enable_file_ops=self._config.enable_file_ops,
            enable_planner=self._config.enable_planner,
            enable_scrape=self._config.enable_scrape,
            enable_web_search=self._config.enable_web_search,
            enable_semantic_search=self._config.enable_semantic_search,
            enable_script=self._config.enable_script,
            enable_system=self._config.enable_system,
            enable_self_improve=self._config.enable_self_improve,
            shell_cwd=self._config.shell_cwd,
            wiki_dir=str(self._config.wiki_root),
            skill_dir=str(self._config.skill_dir),
            sudo_ask=self._ask_sudo_password,
            confirm_ask=self._ask_confirm,
            shell_on_line=self._live_box.shell_line,
        )
        # Phase 3: Sub-Agent-Isolation — registriert das subagent-Tool,
        # das einen isolierten AgentLoop mit frischem Kontext startet.
        if self._config.enable_subagent:
            from .agent.tools.subagent import SubAgentTool
            self._registry.register(
                SubAgentTool(
                    client=self._client,
                    config=self._config,
                    parent_registry=self._registry,
                )
            )
        self._loop = AgentLoop(
            client=self._client,
            registry=self._registry,
            config=self._config,
            on_event=self._event,
        )

        try:
            if self._config.non_interactive and self._config.prompt_text:
                await self._run_single(self._config.prompt_text)
            else:
                await self._run_interactive()
            # Reflection-Step (Session-Consolidation) am Session-Ende.
            await self._run_reflection()
        finally:
            # Stateful Shell: persistenten bash-Prozess sauber beenden.
            for t in (self._registry.all() if self._registry else []):
                closer = getattr(t, "close", None)
                if closer is not None:
                    try:
                        await closer()
                    except Exception:
                        pass
            await self._client.close()

    # ------------------------------------------------------------------

    async def _run_reflection(self) -> None:
        """Reflection-Step (Session-Consolidation) am Session-Ende.

        Gate: nur wenn die Session echte Tool-Arbeit hatte (≥1 Tool-
        Resultat + ≥4 Non-System-Messages) — sonst wäre es nur Latenz.
        Die Reflexion läuft im bestehenden Loop (gleiche History,
        gleiche Tools) und kann dadurch Lessons direkt ins Wiki
        schreiben. Fehler dürfen den Ausstieg nicht blockieren.
        """
        if not self._config.enable_reflection or self._loop is None:
            return
        hist = self._loop.history
        tool_results = sum(
            1 for m in hist
            if m["role"] == "user" and m["content"].startswith("[Tool-Resultat")
        )
        non_system = sum(1 for m in hist if m["role"] != "system")
        if tool_results < 1 or non_system < 4:
            return
        console.print("\n[dim]🧠 Session-Reflexion (Consolidation) …[/dim]")
        buf = _AnswerBuffer()

        def _ev(kind: str, text: str, tool: str = "") -> None:
            buf.on_event(kind, text)
            self._event(kind, text, tool)

        self._loop.on_event(_ev)
        try:
            async for chunk in self._loop.ask(_REFLECTION_PROMPT):
                buf.add(chunk)
        except Exception as e:
            console.print(f"[yellow]⚠ Reflexion fehlgeschlagen: {e}[/]")
            return
        finally:
            self._loop.on_event(self._event)
        answer = buf.answer
        summary = " ".join(answer.split())
        console.print(f"[dim]🧠 Reflexion: {summary[:200]}[/dim]")

    async def _run_single(self, text: str) -> None:
        """Single-Shot: eine Frage, finale Antwort auf stdout, beenden.

        Zwischentext (Prosa zwischen Tool-Calls) wird NICHT live nach
        stdout geschrieben — nur die finale Antwort seit dem letzten
        Tool-Event (Tool-Aktivität läuft über die Event-Zeilen).
        """
        user_msg(text)
        buf = _AnswerBuffer()

        def _ev(kind: str, text: str, tool: str = "") -> None:
            buf.on_event(kind, text)
            self._event(kind, text, tool)

        self._loop.on_event(_ev)
        try:
            async for chunk in self._loop.ask(text):
                buf.add(chunk)
        except Exception as e:
            self._loop.on_event(self._event)
            error_msg(str(e))
            return
        finally:
            self._loop.on_event(self._event)
        sys.stdout.write(buf.answer)
        sys.stdout.flush()
        console.print()

    _CTX_CACHE_TTL = 5.0  # Sekunden — Kontext-Nutzung ändert sich langsam

    async def _refresh_context(self) -> None:
        """Frage live die Kontext-Nutzung des Servers ab (GET /slots).

        Mit TTL-Cache: In langen Sessions wird dieser Call sonst JEDEN
        Turn ausgeführt und blockiert den Event-Loop mit einem HTTP-
        Roundtrip (~100-300ms). 5s TTL reicht — die Kontext-Nutzung
        ändert sich nur pro Agent-Turn.
        """
        if self._client is None:
            return
        now = time.monotonic()
        if (
            self._ctx_usage is not None
            and hasattr(self, "_ctx_cache_time")
            and now - self._ctx_cache_time < self._CTX_CACHE_TTL
        ):
            return
        try:
            self._ctx_usage = await self._client.context_usage()
            self._ctx_cache_time = now
        except Exception:
            self._ctx_usage = None

    def _print_context_bar(self) -> None:
        """Druckt die kompakte Kontext-Balken-Zeile (dim)."""
        usage = self._ctx_usage
        if not usage:
            console.print("  [dim]Context: n/a (server provides no /slots info)[/dim]")
            return
        total = usage["n_ctx"]
        used = usage["n_ctx_used"]
        free = max(total - used, 0)
        pct = (used / total * 100) if total else 0.0
        if pct < 60:
            color = "green"
        elif pct < 85:
            color = "yellow"
        else:
            color = "red"
        bar_w = 30
        filled = int(bar_w * used / total) if total else 0
        bar = "█" * filled + "░" * (bar_w - filled)
        console.print(
            f"  [bold]Context:[/] [{color}]{bar}[/] "
            f"[{color}]{used}[/] / {total} tokens ({pct:.1f}% used, {free} free)"
        )

    async def _prompt_line(self, session: "PromptSession | None") -> str:
        """Liest eine Zeile ein — prompt_toolkit wenn vorhanden, sonst input().

        ASYNC + prompt_async: Die REPL läuft in einem aktiven asyncio-
        Event-Loop — das synchrone prompt_toolkit prompt() ruft intern
        asyncio.run() auf und würde CRASHEN ("asyncio.run() cannot be
        called from a running event loop"). Dieselbe Fehlerklasse wie der
        sudo-Callback (siehe agent-lessons/2026-09-08-async-callback-
        in-running-event-loop.md).
        """
        if HAS_PROMPT_TOOLKIT:
            assert session is not None
            # Terminal-State-Reset vor prompt_async: Richs Live.stop()
            # kann Bracketed-Paste-Mode oder Cursor-Key-Modi zurücklassen.
            # Ohne Reset "frisst" prompt_toolkit die ersten 1-2 Tastendrücke
            # (veraltete Escape-Sequenzen im Input-Buffer).
            sys.stdout.write("\x1b[?2004l\x1b[?1l\x1b[?25h\x1b[0m")
            sys.stdout.flush()
            # Event-Loop drain: Nach dem Streaming-Ende gibt es noch
            # ausstehende I/O-Callbacks (httpx-Socket-Buffers, Connection
            # Pool-Cleanup). Diese konkurrieren mit prompt_toolkit's
            # stdin-Reader und machen die ersten Tastendrücke träge.
            # Ein kurzer Yield lässt den Loop ausruhen.
            await asyncio.sleep(0)
            line = await session.prompt_async(
                HTML("<b><blue>Du</blue></b> <dim>(/help)</dim> > "),
            )
            return line.strip()
        # Fallback: input() blockiert den Loop — im Thread laufen lassen.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: input("\x1b[1;34mDu\x1b[0m \x1b[2m(/help)\x1b[0m > ").strip(),
        )

    # ------------------------------------------------------------------
    # Sudo-Abfrage
    # ------------------------------------------------------------------

    async def _ask_sudo_password(self, command: str) -> Optional[str]:
        """Frage das sudo-Passwort ab (verdeckt). None = abgebrochen.

        Wird von der ShellTool aufgerufen, wenn ein Befehl sudo braucht.
        Das Passwort landet nur im RAM-Cache der ShellTool, nie in der
        Chat-History oder in Logs.

        ASYNC, weil der Agent-Loop in einem aktiven asyncio-Event-Loop
        läuft — ein synchroner prompt()-Aufruf würde dort crashen.
        prompt_toolkit: prompt_async(); Fallback: getpass in einem
        Thread (damit der Event-Loop nicht blockiert).

        WICHTIG: Die Rich-Live-Box wird vor dem Prompt gestoppt und
        danach wieder gestartet — sonst ist der Terminal-Zustand durch
        Richs Rendering gestört und prompt_async schlägt fehl.
        """
        short = command[:80]
        # Live-Box stoppen, damit das Terminal für die Passwort-Eingabe
        # in einem sauberen Zustand ist (Rich rendert sonst aktiv).
        live_was_active = self._live_box.is_active()
        if live_was_active:
            self._live_box.stop()
        try:
            pw: Optional[str] = None
            prompt_failed = False
            if HAS_PROMPT_TOOLKIT and self._session is not None:
                # EIGENE Session mit InMemoryHistory: das Passwort darf
                # NICHT in die FileHistory (history.txt) — sonst wäre es
                # per ↑/↓ wiederabrufbar und läge im Klartext auf Disk.
                if self._pw_session is None:
                    self._pw_session = _make_pw_session()
                try:
                    pw = await self._pw_session.prompt_async(
                        HTML(
                            "<b><red>🔒 sudo</red></b> <dim>Password for:</dim> "
                            f"<b>{html.escape(short)}</b>"
                        ),
                        is_password=True,
                    )
                except Exception:
                    # prompt_async kann fehlschlagen wenn der Terminal-Zustand
                    # durch Richs Live-Rendering gestört ist (selbst nach
                    # stop()). Fallback auf getpass im Thread.
                    prompt_failed = True
            if pw is None and not prompt_failed:
                # Kein prompt_toolkit oder Session fehlt → getpass-Fallback.
                pw = await asyncio.get_running_loop().run_in_executor(
                    None, getpass.getpass, f"🔒 sudo password for: {short} > "
                )
            elif prompt_failed:
                # prompt_async hat eine Exception geworfen → getpass-Fallback.
                pw = await asyncio.get_running_loop().run_in_executor(
                    None, getpass.getpass, f"🔒 sudo password for: {short} > "
                )
            return pw or None
        except (EOFError, KeyboardInterrupt):
            return None
        finally:
            # Live-Box wieder starten, wenn sie vorher aktiv war.
            if live_was_active:
                self._live_box.start()

    async def _ask_confirm(self, command: str, reason: str) -> bool:
        """Frage den Nutzer direkt, ob er einen destruktiven Befehl bestätigt.

        Wird von der ShellTool aufgerufen, wenn ein Guardrail-CONFIRM-Hit
        ohne ``confirm=true`` passiert. Liefert ``True`` (Ja) oder
        ``False`` (Nein/Abbruch).

        ASYNC, weil der Agent-Loop in einem aktiven asyncio-Event-Loop
        läuft — analog zu ``_ask_sudo_password``.
        """
        short = command[:80]
        live_was_active = self._live_box.is_active()
        if live_was_active:
            self._live_box.stop()
        try:
            if HAS_PROMPT_TOOLKIT and self._session is not None:
                answer = await self._session.prompt_async(
                    HTML(
                        "<b><yellow>⚠️ Confirmation required</yellow></b> "
                        f"<dim>({reason})</dim>\n"
                        f"Command: <b>{html.escape(short)}</b>\n"
                        "Execute? (y/n) > "
                    ),
                )
            else:
                answer = await asyncio.get_running_loop().run_in_executor(
                    None,
                    input,
                    f"\x1b[1;33m⚠️ Confirmation required ({reason})\x1b[0m\n"
                    f"Command: {short}\nExecute? (y/n) > ",
                )
            return answer.strip().lower() in ("j", "y", "ja", "yes")
        except (EOFError, KeyboardInterrupt):
            return False
        finally:
            if live_was_active:
                self._live_box.start()

    # ------------------------------------------------------------------
    # Agent-Run
    # ------------------------------------------------------------------

    @staticmethod
    def _dump_session(loop: "AgentLoop") -> None:
        """Kleine Session-Sicherung: last.json + timestampiertes Archiv.

        Läuft im finally-Block von _run_agent — auch bei Exceptions,
        damit ein Abbruch (Loop-Detekt, Ctrl+C, Tool-Fehler) nie eine
        unerreichbare Session hinterlässt.
        """
        if loop is None:
            return
        try:
            SESSION_DIR.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y-%m-%dT%H-%M-%S")
            data = {"saved_at": ts, "history": loop.history}
            (SESSION_DIR / "last.json").write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            archived = SESSION_DIR / f"{ts}.json"
            archived.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            # Archiv begrenzen (älteste löschen).
            olds = sorted(SESSION_DIR.glob("2*.json"))
            olds = [p for p in olds if p.name != "last.json"]
            for p in olds[:-SESSION_KEEP]:
                p.unlink(missing_ok=True)
        except OSError as e:
            # Session-Dump darf den Agent nie blockieren.
            log.debug("Session-Dump fehlgeschlagen: %s", e)

    async def _dump_session_async(self, loop: "AgentLoop") -> None:
        """Async-Wrapper für _dump_session: läuft im Executor, blockiert
        den Event-Loop nicht. In langen Sessions kann die JSON-Serie
        100-500ms dauern — das würde prompt_toolkit's stdin-Reader
        träge machen (User muss Tasten mehrmals drücken)."""
        await asyncio.get_running_loop().run_in_executor(
            None, self._dump_session, loop
        )

    async def _run_agent(self, prompt: str) -> None:
        """Sendet eine Nachricht an den Agent.

        Während des Laufs: Live-Terminal-Box (Shell-Befehle zeilenweise,
        Spinner dazwischen) + Tool-/Thinking-Zeilen. Die finale Antwort
        wird hervorgehoben: fettes Panel + farbiges Markdown-Theme.
        """
        buf = _AnswerBuffer()

        def _ev(kind: str, text: str, tool: str = "") -> None:
            buf.on_event(kind, text)
            self._event(kind, text, tool)

        self._loop.on_event(_ev)
        self._live_box.start()
        try:
            async for chunk in self._loop.ask(prompt):
                buf.add(chunk)
        except Exception as e:
            self._loop.on_event(self._event)
            error_msg(str(e))
            return
        finally:
            self._live_box.stop()
            self._loop.on_event(self._event)
            # Async-Dump: blockiert den Event-Loop nicht (in langen
            # Sessions kann die JSON-Serie 100-500ms dauern).
            await self._dump_session_async(self._loop)
        answer = buf.answer

        if not answer.strip():
            console.print("[dim](leere Antwort)[/dim]")
            return

        # Antwort als Markdown mit Hervorhebungs-Theme rendern
        # (Überschriften gelb, Code cyan, Fett/Links kräftig).
        # Rich >= 14: Theme gehört an die Console, nicht an Markdown.
        md = Markdown(answer.strip())
        lines = [l for l in answer.strip().splitlines() if l.strip()]
        if len(lines) < _ANSWER_PANEL_MIN_LINES and len(answer.strip()) < 200:
            # Kurzantwort: direkt mit Markdown (keine Box nötig)
            _themed_console.print(md)
        else:
            # Längere Antwort: fettes, hervorgehobenes Panel
            _themed_console.print(
                Panel(
                    md,
                    title="[bold bright_white]✦ Antwort[/]",
                    border_style="bright_green",
                    box=box.ROUNDED,
                    expand=True,
                    padding=(1, 1),
                )
            )

    async def _run_interactive(self) -> None:
        """Der interaktive Loop."""
        banner()
        info_msg("Type a message or [dim]/help[/] for commands. [dim]Ctrl+D[/] to exit.")
        if not HAS_PROMPT_TOOLKIT:
            info_msg("[dim](prompt_toolkit missing — no autocomplete/history)[/dim]")
        console.print()

        session = None
        if HAS_PROMPT_TOOLKIT:
            session = PromptSession(
                history=FileHistory(str(HISTORY_PATH)),
                complete_while_typing=True,
            )
        self._session = session

        while True:
            # Kontext-Balken vor jedem Prompt aktualisieren — so ist er
            # auch zwischen den Turns (nach jeder Antwort/Kommando) sichtbar.
            await self._refresh_context()
            self._print_context_bar()

            try:
                text = await self._prompt_line(session)
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Bye![/dim]")
                break

            if not text:
                continue

            # --- Slash-Kommandos ---
            if text.startswith("/"):
                cmd = text.split()[0].lower()

                if cmd == "/exit" or cmd == "/quit":
                    console.print("[dim]Bye![/dim]")
                    break

                elif cmd == "/help":
                    slash_help()
                    continue

                elif cmd == "/clear":
                    self._loop.clear()
                    info_msg("History cleared.")
                    continue

                elif cmd == "/learn":
                    parts = text.split(maxsplit=1)
                    if len(parts) < 2:
                        console.print("[yellow]Usage: /learn <topic>[/]")
                        continue
                    from .memory import learn_prompt
                    prompt = learn_prompt(self._config, parts[1].strip())
                    info_msg(f"📚 Learning: {parts[1].strip()}")
                    await self._run_agent(prompt)
                    continue

                elif cmd == "/ingest":
                    parts = text.split(maxsplit=1)
                    if len(parts) < 2:
                        console.print("[yellow]Usage: /ingest <text or URL>[/]")
                        continue
                    info_msg("💾 Saving to the wiki (Ingest).")
                    from .memory import skill_dir
                    await self._run_agent(
                        f"Perform a wiki ingest (schema: {skill_dir(self._config)}/SKILL.md). "
                        f"Source = the following user instruction/information "
                        "(original text):\n\n" + parts[1].strip() + "\n"
                    )
                    continue

                elif cmd == "/tools":
                    if self._registry:
                        for t in self._registry.all():
                            console.print(f"  [cyan]{t.name}[/] — {t.description}")
                    else:
                        info_msg("No tools registered.")
                    continue

                elif cmd == "/history":
                    if self._loop:
                        for m in self._loop.history:
                            if m["role"] != "system":
                                console.print(f"  [{m['role']}] {m['content'][:120]}")
                    continue

                elif cmd == "/save":
                    parts = text.split(maxsplit=1)
                    path = parts[1].strip() if len(parts) > 1 else "chat_history.json"
                    hist = self._loop.history if self._loop else []
                    try:
                        with open(path, "w", encoding="utf-8") as f:
                            json.dump(hist, f, indent=2, ensure_ascii=False)
                    except OSError as e:
                        console.print(f"[red]Error saving:[/] {e}")
                        continue
                    info_msg(f"✓ History saved to {path}.")
                    continue

                elif cmd == "/system":
                    info_msg("System prompt modification not supported during a run.")
                    continue

                elif cmd == "/tokens":
                    if self._loop:
                        n = len(self._loop.history)
                        info_msg(f"{n} messages in history.")
                    continue

                elif cmd == "/context":
                    await self._refresh_context()
                    self._print_context_bar()
                    continue

                else:
                    console.print(f"[yellow]Unknown command: {cmd}[/] — /help for list.")
                    continue

            # --- Normale Nachricht an den Agent ---
            try:
                await self._run_agent(text)
            except KeyboardInterrupt:
                # Ctrl+C während Agent-Run: Graceful Abort.
                # Live-Box stoppen, Terminal-State resetten, zum Prompt
                # zurückkehren — ohne die gesamte REPL zu crasen.
                self._live_box.stop()
                console.print("\n[yellow]⚠ Agent run aborted (Ctrl+C).[/]")
                continue

    # ------------------------------------------------------------------

    def _event(self, kind: str, text: str, tool: str = "") -> None:
        """Events aus der Agent-Loop rendern (Thinking, Tools, Warnungen).

        Während eines Agent-Turns läuft alles durch die Live-Box
        (Transkript + Live-Terminal-Box); sonst (z. B. Reflection ohne
        Live-View) direkt auf die Konsole.
        """
        if self._live_box.is_active():
            self._live_box.note(kind, text, tool)
            return
        if kind == "thinking":
            raw = text.strip()
            if len(raw) > 500:
                raw = raw[:497] + "…"
            console.print(
                Panel(
                    Text(raw),
                    title="[dim]💭 Thinking[/dim]",
                    border_style="dim",
                    expand=True,
                    padding=(0, 1),
                )
            )
        elif kind == "tool":
            console.print(f"  [cyan]⚙ {text}[/]")
        elif kind == "tool_output":
            lines = [l for l in text.strip().splitlines() if l.strip()]
            preview = " | ".join(l for l in lines[:3])
            if len(preview) > 200:
                preview = preview[:197] + "..."
            if not preview:
                preview = "(empty)"
            console.print(f"    [dim]↳ {preview}[/]")
        elif kind == "tool_error":
            console.print(f"    [dim red]✗ {text[:200]}[/]")
        elif kind == "warning":
            console.print(f"  [yellow]⚠ {text}[/]")
