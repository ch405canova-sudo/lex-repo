"""Agent-Loop: Plan → Act → Observe.

Der Loop:
1. Sendet die aktuelle Message-History an das Modell.
2. Prüft die Antwort auf Tool-Calls.
3. Führt Tools PARALLEl aus (asyncio.gather), hängt die Ergebnisse in
   der ursprünglichen Reihenfolge als 'user'-Messages an.
4. Injiziert nach jedem Step den aktuellen Plan-Status (wenn ein
   Planner-Tool aktiv ist).
5. Wiederholt, bis das Modell eine normale Antwort liefert
   oder max_steps erreicht.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import AsyncGenerator, Callable, Optional

from ..config import Config
from ..llama_client import LlamaClient
from .tools import ToolRegistry, ToolResult
from .tools.base import EXIT_CODE_RE
from .tools.planner import PlannerTool
from .parser import ToolCall, parse_tool_calls

log = logging.getLogger(__name__)

# Harte Obergrenze für Tool-Outputs in der History (KB):
# verhindert Kontext-Blowup bei `shell: cat große_datei` & Co.
_MAX_TOOL_OUTPUT = 16 * 1024
# Ingestion-Filter: Bei großen Outputs werden nur die wichtigsten Zeilen
# behalten (Fehler, Konklusionen, Kopf/Ende) statt blind abzuschneiden.
_INGEST_HEAD_LINES = 20   # erste Zeilen (Kontext/Kopf)
_INGEST_TAIL_LINES = 10   # letzte Zeilen (Ergebnis/Fazit)
# N-maliger identischer (Tool + Args)-Wiederholung → Schleifen-Abbruch.
# Einziger harter Guard: fängt echte Infinite Loops (exakt derselbe Call).
_LOOP_ABORT_COUNT = 3
# History-Compaction (Context-Rot-Schutz): wird ausgelöst, wenn der
# Server-Kontext >= _COMPACT_CONTEXT_PCT belegt ist (via context_usage).
# Fallback bei fehlender Server-Info: Message-Zahl >= _COMPACT_TRIGGER.
# Der mittere Teil wird zu EINER Situation-Bericht-Message komprimiert;
# KEEP Messages bleiben vollständig erhalten (aktuelle Arbeit).
# Früher Trigger (70% statt 95%): Context-Drift beginnt bei >30K Tokens,
# Performance degradiert LANGE vor dem harten Limit.
_COMPACT_CONTEXT_PCT = 0.70
_COMPACT_TRIGGER = 60   # Fallback: Message-Zahl, wenn Server keine Kontext-Info liefert
_COMPACT_KEEP = 16
# Drift-Detektion: Wenn das Modell in aufeinanderfolgenden Steps ähnliche
# Textmuster wiederholt (Re-Statements, gleiche Formulierungen), wird ein
# Nudge injiziert statt blind weiterzulaufen.
_DRIFT_SIMILARITY_THRESHOLD = 0.7  # Jaccard-ähnlichkeits-Schwelle (Token-Basis)
_DRIFT_STREAK_LIMIT = 3            # consecutive Steps mit hoher Ähnlichkeit

# --- Stabilisierungs-Features (Phase 1+2) ---
# Tool-Output-Offloading: vollständige Outputs > _MAX_TOOL_OUTPUT werden
# in eine Datei geschrieben; der Pfad wird in der History referenziert.
_TOOL_OUTPUT_DIR = Path.home() / ".local" / "share" / "chatcli" / "tool_outputs"
# JSONL-Trace-Logger: Schreibt jeden Agent-Step als JSON-Zeile in eine
# Session-Datei (~/.local/share/chatcli/traces/<timestamp>.jsonl).
# Macht lange Sessions nachvollziehbar (Debugging, Forensik, Replay).
_TRACE_DIR = Path.home() / ".local" / "share" / "chatcli" / "traces"
# Per-Step-Timeout: max. Sekunden pro einzelner Tool-Aufruf (verhindert
# hängende Befehle wie `sleep 3600` die den gesamten Loop blockieren).
# Der effektive Wert kommt aus Config.step_timeout (Env: CHATCLI_STEP_TIMEOUT);
# diese Konstante ist der Fallback, falls Config kein step_timeout-Feld hat.
_STEP_TIMEOUT = 60.0
# Token-Budget-Monitoring: Warnschwelle für quadratisches Wachstum.
_TOKEN_BUDGET_WARN_RATIO = 2.0


class AgentLoop:
    def __init__(
        self,
        client: LlamaClient,
        registry: ToolRegistry,
        config: Config,
        on_event: Optional[Callable[[str, str, str], None]] = None,
    ) -> None:
        self._client = client
        self._registry = registry
        self._config = config
        self._history: list[dict[str, str]] = []
        # Optionales Event-Callback: (kind, text, tool_name)
        # kind ∈ {"thinking","tool","tool_output","tool_error","warning"}
        self._on_event = on_event
        # Mid-Session Lesson-Re-Injektion: Snapshot der Lessons-Dateien
        # (Name → mtime) zum Start; neue/geänderte Dateien werden im Lauf
        # der Session als Hinweis injiziert (sonst "blinder Fleck").
        self._lesson_snapshot = self._snapshot_lessons(config)
        self._json_protocol_injected = False
        self._last_tool_success = False  # #7 Self-Verification Nudge
        self._tracked_shell_cwd: Optional[str] = None  # CWD der stateful Shell
        # Session-State-Anchor: Persistente 4-Felder-Listen, die bei jeder
        # Compaction inkrementell aktualisiert werden (Anchored Iterative
        # Summarization). Verhindert Wissensverlust bei wiederholter Kompression.
        self._anchor_insights: list[str] = []
        self._anchor_actions: list[str] = []
        self._anchor_decisions: list[str] = []
        self._anchor_next_steps: list[str] = []

        # JSONL-Trace-Logger: Session-ID + Trace-Datei.
        _ts = time.strftime("%Y%m%d-%H%M%S")
        self._trace_path = _TRACE_DIR / f"{_ts}.jsonl"
        try:
            _TRACE_DIR.mkdir(parents=True, exist_ok=True)
            self._trace_path.write_text("", encoding="utf-8")
        except OSError:
            self._trace_path = None  # Trace deaktiviert (kein Schreibrecht)

        # System-Prompt (Kontext wird dynamisch injiziert,
        # damit der Prompt nicht hart auf Pfade verweist).
        # _system_base = der stabile Teil OHNE Memory — die Memory-Blöcke
        # (Lessons/Stil) werden pro Turn neu injiziert (selektive
        # Lessons), der Rest bleibt als Prefix erhalten.
        self._system_base = config.system_prompt
        if registry.descriptions():
            self._system_base += "\n\n" + registry.descriptions()
        cwd = config.shell_cwd or os.getcwd()
        self._system_base += (
            f"\n\n## Context\n"
            f"Working directory: {cwd}\n"
            f"IMPORTANT: Relative paths in read_file, list_dir, write_file, search "
            f"are resolved against the CURRENT working directory. "
            f"When the shell enters a different directory via `cd`, "
            f"the relative paths of file tools change accordingly. "
            f"Use absolute paths or check with list_dir('.') where you are."
        )

        # Gedächtnis: Memory-Zeiger + Stil-Regeln (z. B. Humor/Emojis) aus
        # der Wiki werden bei JEDEM Start automatisch injiziert.
        system = self._system_base
        try:
            from ..memory import style_injection
            system += style_injection(config)
        except Exception:
            pass  # Memory darf den Start nicht blockieren

        self._history.append({"role": "system", "content": system})

    # ------------------------------------------------------------------

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)

    def clear(self) -> None:
        # Behalte nur die System-Message
        self._history = [h for h in self._history if h["role"] == "system"]

    def on_event(self, cb) -> None:
        """Event-Callback setzen (None = deaktiviert)."""
        self._on_event = cb

    # ------------------------------------------------------------------

    @staticmethod
    def _lesson_dir(config: Config):
        from ..memory import wiki_root
        return wiki_root(config) / "wiki" / config.lessons_topic

    @staticmethod
    def _snapshot_lessons(config: Config) -> dict[str, float]:
        """Lessons-Dateien als {name: mtime} ({} wenn Topic fehlt)."""
        d = AgentLoop._lesson_dir(config)
        if not d.is_dir():
            return {}
        snap: dict[str, float] = {}
        for f in d.glob("*.md"):
            try:
                snap[f.name] = f.stat().st_mtime
            except OSError:
                pass
        return snap

    def _lesson_delta(self) -> tuple[list[str], list[str]]:
        """(neue_geänderte, archiviert_entfernte) seit Session-Start."""
        d = self._lesson_dir(self._config)
        if not d.is_dir():
            return [], []
        now: dict[str, float] = {}
        for f in d.glob("*.md"):
            try:
                now[f.name] = f.stat().st_mtime
            except OSError:
                pass
        added = [n for n, t in now.items()
                 if self._lesson_snapshot.get(n) != t]
        removed = [n for n in self._lesson_snapshot if n not in now]
        self._lesson_snapshot = now
        return added, removed

    def _refresh_system_memory(self, user_text: str) -> None:
        """Selektive Lessons-Injektion (KV-Cache-stabil).

        Die System-Message (Index 0) bleibt VOLLSTÄNDIG UNVERÄNDERT —
        nur der stabile Base + Style wird einmalig beim Start gesetzt.
        Die Lessons werden als SEPARATE User-Message (Index 1) injiziert,
        die pro Turn aktualisiert wird. So bleibt das System-Prefix
        KV-Cache-stabil (kein 10×-Kosten-Aufschlag).

        Layout:
          [0] system  → _system_base + style_injection (stabil, nie geändert)
          [1] user    → "[Lessons] ..." (pro Turn aktualisiert)
          [2+] user/assistant → normale History
        """
        if not self._config.enable_selective_lessons:
            return
        try:
            from ..memory import style_injection
            lessons_block = style_injection(self._config, query=user_text)
            # System-Message (Index 0) bleibt stabil — wird NIE pro Turn geändert.
            # Stattdessen: Lessons als separate User-Message an Index 1 setzen.
            if len(self._history) >= 2 and self._history[1].get("_is_lessons"):
                # Bestehende Lessons-Message aktualisieren (in-place, kein Appending).
                self._history[1]["content"] = f"[Lessons] {lessons_block}"
            else:
                # Neue Lessons-Message nach der System-Message einfügen.
                self._history.insert(
                    1,
                    {"role": "user", "content": f"[Lessons] {lessons_block}", "_is_lessons": True},
                )
        except Exception:
            log.debug("Selektive Lessons-Injektion fehlgeschlagen", exc_info=True)

    @staticmethod
    def _offload_tool_output(step: int, tool_name: str, output: str) -> str:
        """Schreibt das vollständige Tool-Output in eine Datei und liefert den Pfad.

        Wird aufgerufen, wenn ein Tool-Output > _MAX_TOOL_OUTPUT ist.
        Das Modell kann die Datei per read_file nachlesen.
        """
        try:
            _TOOL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_name)
            path = _TOOL_OUTPUT_DIR / f"step{step}_{safe_name}.txt"
            path.write_text(output, encoding="utf-8")
            return str(path)
        except OSError:
            return ""  # Fallback: kein Offload möglich

    def _trace(self, event: dict) -> None:
        """Schreibt einen Trace-Eintrag als JSON-Zeile in die Session-Datei.

        Event-Felder (typisch):
          step, type ("model_response"|"tool_result"|"nudge"|"compaction"
          |"ralph"|"final"), tool, args, ok, error, tokens_used, ...
        """
        if self._trace_path is None:
            return
        try:
            entry = {"ts": time.time(), **event}
            with open(self._trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass  # Trace darf den Loop nie blockieren

    @staticmethod
    def _ingest_filter(output: str) -> str:
        """Ingestion-Filter: Extrahiert die wichtigsten Zeilen aus einem großen
        Tool-Output, statt blind abzuschneiden.

        Strategie (basierend auf Factory.ai / Zylos Research 2026):
        - Kopf-Zeilen (Kontext, Header)
        - Fehler-/Warnungs-Zeilen (stderr, Error, Warning, FAIL)
        - Schluss-Zeilen (Ergebnis, Konklusion)
        - Mittlere Zeilen werden durch einen Platzhalter ersetzt

        Das vollständige Output wird weiterhin offgeladen — hier wird nur
        die Version für den Kontext komprimiert.
        """
        if len(output) <= _MAX_TOOL_OUTPUT:
            return output  # Passt rein, kein Filter nötig

        lines = output.splitlines()
        total_lines = len(lines)

        # Wichtige Zeilen identifizieren (Fehler, Warnungen, Konklusionen).
        important_patterns = re.compile(
            r"(?i)(error|warn|fail|denied|not found|missing|refused|"
            r"exception|traceback|fatal|✗|❌|\[exit_code: [^0])"
        )

        head = lines[:_INGEST_HEAD_LINES]
        tail = lines[-_INGEST_TAIL_LINES:] if total_lines > _INGEST_HEAD_LINES else []

        # Fehler-/Warnungs-Zeilen aus dem Mittelteil extrahieren.
        middle_start = _INGEST_HEAD_LINES
        middle_end = total_lines - len(tail) if tail else total_lines
        error_lines: list[str] = []
        for i in range(middle_start, middle_end):
            if important_patterns.search(lines[i]):
                error_lines.append(f"  {lines[i].strip()}")
                if len(error_lines) >= 15:
                    break

        # Zusammenbauen.
        parts: list[str] = []
        parts.extend(head)
        omitted = total_lines - len(head) - len(tail) - len(error_lines)
        if omitted > 0:
            parts.append(f"  [... {omitted} lines truncated — important errors/warnings below ...]")
        parts.extend(error_lines)
        if tail:
            parts.append("  --- End of output ---")
            parts.extend(tail)

        filtered = "\n".join(parts)
        # Fallback: Wenn der Filter das Output NICHT kleiner macht,
        # einfach hart truncieren (sollte extrem selten passieren).
        if len(filtered) > _MAX_TOOL_OUTPUT:
            filtered = output[:_MAX_TOOL_OUTPUT] + f"\n... [truncated, {len(output)} chars total]"
        return filtered

    def _drift_check(self, text: str) -> bool:
        """Prüft, ob der aktuelle Assistant-Text stark mit den letzten
        Assistant-Antworten übereinstimmt (Drift-Signal).

        Nutzt eine einfache Token-Basis-Jaccard-Ähnlichkeit. Wenn in
        _DRIFT_STREAK_LIMIT aufeinanderfolgenden Steps die Ähnlichkeit
        über _DRIFT_SIMILARITY_THRESHOLD liegt → True (Drift erkannt).
        """
        if not hasattr(self, "_recent_texts"):
            self._recent_texts: list[str] = []
        self._recent_texts.append(text)
        # Nur die letzten N Texte behalten.
        if len(self._recent_texts) > _DRIFT_STREAK_LIMIT + 1:
            self._recent_texts.pop(0)

        if len(self._recent_texts) < _DRIFT_STREAK_LIMIT:
            return False

        # Jaccard-Ähnlichkeit zwischen dem aktuellen und dem vorherigen.
        def _tokens(s: str) -> set[str]:
            return set(re.findall(r"\w+", s.lower()))

        current = _tokens(text)
        prev = _tokens(self._recent_texts[-2]) if len(self._recent_texts) >= 2 else set()
        if not current or not prev:
            return False
        intersection = len(current & prev)
        union = len(current | prev)
        similarity = intersection / union if union > 0 else 0.0
        return similarity > _DRIFT_SIMILARITY_THRESHOLD

    @staticmethod
    def _looks_like_announcement(text: str) -> bool:
        """Erkennt 'beschreiben statt handeln': Das Modell kündigt eine
        Aktion an (oder imitiert das Insight-Format), ohne ein Tool aufzurufen.

        Starke Signale (jedes für sich ausreichend):
        1. >=2 sichtbare [Insight: ...]-Blöcke — Format-Imitation.
        2. Typische Ankündigungsphrasen am Textanfang (DE/EN).
        """
        t = text.lower()
        if len(re.findall(r"\[insight:", t)) >= 2:
            return True
        head = t[:150]
        markers = (
            "lass mich ", "schauen wir uns", "schau ich mir", "ich schaue mir",
            "ich werde das", "ich werde jetzt", "jetzt rufe ich",
            "let me check", "let me read", "let me look", "let me see",
            "i will now", "now i will", "now let me",
        )
        return any(m in head for m in markers)

    def _inject_json_protocol_if_needed(self) -> None:
        """Fix 4: Wenn der Server nativen Tool-Calls ablehnt, wird das
        JSON-Text-Protokoll NACHTRÄGLICH in die System-Message injiziert
        (sonst widerspricht der Prompt der Realität)."""
        if self._json_protocol_injected or self._client.supports_tools:
            return
        try:
            from ..config import JSON_TOOL_PROTOCOL
        except ImportError:
            return
        sys_msg = self._history[0]
        if JSON_TOOL_PROTOCOL.splitlines()[0] in sys_msg["content"]:
            self._json_protocol_injected = True
            return
        sys_msg["content"] += "\n\n" + JSON_TOOL_PROTOCOL
        self._json_protocol_injected = True
        log.info("JSON-Tool-Protokoll nachträglich injiziert (Server lehnt Tools ab)")

    # ------------------------------------------------------------------

    def _planner(self) -> Optional[PlannerTool]:
        """Das PlannerTool aus der Registry (wenn registriert)."""
        tool = self._registry.get("plan")
        return tool if isinstance(tool, PlannerTool) else None

    async def _compact_history(self) -> None:
        """Komprimiert die mittlere History, wenn der Kontext >= 70% belegt ist.

        Strategie (Anchored Iterative Summarization):
        - System-Message + letzte KEEP Messages bleiben vollständig.
        - Der mittlere Teil wird zu EINEM "[Situation-Bericht]" (Session-State-Anchor)
          reduziert, strukturiert in 4 Feldern: Intent, Änderungen, Entscheidungen,
          Nächste Schritte.
        - Bei wiederholter Compaction wird der Anchor INKREMENTELL aktualisiert:
          Neue Erkenntnisse werden in den bestehenden Anchor gemerged statt ihn
          von Grund auf neu zu generieren (verhindert Wissensverlust).
        """
        if len(self._history) <= 1:
            return

        # Trigger: Kontext-Nutzung >= _COMPACT_CONTEXT_PCT (via Server),
        # Fallback: Message-Zahl.
        trigger_hit = False
        try:
            usage = await self._client.context_usage()
            if usage and usage["n_ctx"] > 0:
                pct = usage["n_ctx_used"] / usage["n_ctx"]
                if pct >= _COMPACT_CONTEXT_PCT:
                    trigger_hit = True
                    log.info(
                        "Compaction-Trigger: Kontext %.1f%% belegt (%d/%d Tokens)",
                        pct * 100, usage["n_ctx_used"], usage["n_ctx"],
                    )
            else:
                non_system = len(self._history) - 1
                if non_system >= _COMPACT_TRIGGER:
                    trigger_hit = True
        except Exception:
            non_system = len(self._history) - 1
            if non_system >= _COMPACT_TRIGGER:
                trigger_hit = True

        if not trigger_hit:
            return

        # KEEP-Grenze an Tool-Paar-Grenzen ausrichten.
        keep_start = len(self._history) - _COMPACT_KEEP
        while keep_start > 1:
            first_keep = self._history[keep_start]
            is_tool_result = (
                first_keep["role"] == "user"
                and first_keep["content"].startswith("[Tool Result |")
            )
            if not is_tool_result:
                break
            keep_start -= 1
        keep = self._history[keep_start:]
        middle = self._history[1:keep_start]
        if not middle:
            return

        # --- Neue Erkenntnisse aus dem mittleren Abschnitt extrahieren ---
        new_insights: list[str] = []
        new_actions: list[str] = []
        new_decisions: list[str] = []
        new_next_steps: list[str] = []

        for m in middle:
            content = m["content"]
            if m["role"] == "assistant":
                if content.strip():
                    insight_m = re.search(r"\[Insight: ([^\]]+)\]", content)
                    if insight_m:
                        new_insights.append(insight_m.group(1))
                    # Entscheidungen erkennen (z. B. "Entscheidung:", "Ich werde")
                    decision_m = re.search(
                        r"(?i)(entscheidung|decision|ich werde|wir sollten|entschieden)\s*[:—-]?\s*(.+)",
                        content,
                    )
                    if decision_m:
                        new_decisions.append(decision_m.group(2).strip()[:100])
            else:
                if content.startswith("[Insight:"):
                    new_insights.append(re.sub(r"^\[Insight:\s*", "", content))
                elif content.startswith("[System note]"):
                    new_next_steps.append(content[len("[System note]"):][:200])
                elif content.startswith("[Tool Result |"):
                    lines = content.splitlines()
                    status_line = lines[0] if lines else ""
                    body_lines = [l for l in lines[1:] if l.strip()][:2]
                    new_actions.append(status_line + " | " + " ".join(body_lines)[:120])
                elif content.startswith("[Memory-Update]"):
                    pass  # transient, nicht relevant für Situation
                else:
                    new_next_steps.append(" ".join(content.split())[:120])

        # --- Inkrementeller Merge in persistente Anchor-Listen ---
        for ins in new_insights:
            if ins not in self._anchor_insights:
                self._anchor_insights.append(ins)
        for act in new_actions:
            if act not in self._anchor_actions:
                self._anchor_actions.append(act)
        for dec in new_decisions:
            if dec not in self._anchor_decisions:
                self._anchor_decisions.append(dec)
        for ns in new_next_steps:
            if ns not in self._anchor_next_steps:
                self._anchor_next_steps.append(ns)

        # --- 4-Felder-Anker aus den persistenten Listen aufbauen ---
        parts: list[str] = []
        objective = getattr(self, "_original_objective", "")
        if objective:
            parts.append(f"## Intent (original goal):\n{objective}")
        else:
            parts.append("## Intent:")

        # Erkenntnisse (Insights) haben Vorrang — werden zuerst gelistet
        if self._anchor_insights:
            parts.append("## Key insights:")
            parts.extend(self._anchor_insights[-15:])  # max. 15 Insights

        if self._anchor_actions:
            parts.append("## Actions performed:")
            parts.extend(self._anchor_actions[-20:])  # max. 20 Aktionen

        if self._anchor_decisions:
            parts.append("## Decisions:")
            parts.extend(self._anchor_decisions[-10:])  # max. 10 Entscheidungen

        if self._anchor_next_steps:
            parts.append("## Next steps:")
            parts.extend(self._anchor_next_steps[-5:])  # max. 5 nächste Schritte

        summary_final = "\n".join(parts)
        if len(summary_final) > 4000:
            summary_final = "… [older entries truncated]\n" + summary_final[-4000:]

        self._history = (
            [self._history[0],
             {"role": "user",
              "content": "[Situation Report] Persistent session-state anchor. "
                         "IMPORTANT insights are here — use them, "
                         "instead of re-deriving them:\n"
                         + summary_final}]
            + keep
        )
        if self._on_event:
            self._on_event(
                "warning", "History compacted (session-state anchor updated)", ""
            )

    # ------------------------------------------------------------------

    def _plan_block(self) -> str:
        """Aktueller Plan-Status, "" wenn kein aktiver Plan existiert."""
        planner = self._planner()
        if planner is None:
            return ""
        try:
            return planner.status_block()
        except Exception:
            return ""

    @staticmethod
    def _next_open_plan_step(plan_block: str) -> str:
        """Erster offener Plan-Schritt (Zeile mit '•') aus dem Status-Block."""
        for line in plan_block.splitlines():
            if "•" in line:
                return line.strip().lstrip("•").strip()
        return "(next open step)"

    # ------------------------------------------------------------------

    async def ask(self, user_text: str) -> AsyncGenerator[str, None]:
        """
        Führe einen vollständigen Agent-Schritt aus.
        Streamt die finale Antwort (ohne Tool-Zwischenschritte).
        """
        self._history.append({"role": "user", "content": user_text})

        # Selektive Lessons-Injektion: System-Message auf die aktuelle
        # Aufgabe abstimmen (semantisch relevante Lessons, Prefix-stabil).
        self._refresh_system_memory(user_text)

        # Schleifen-Erkennung: Einzelne (tool, args)-Paare werden über
        # aufeinanderfolgende Steps gezählt. Wenn ein spezifischer Call
        # N-mal in Folge in JEDEM Step erscheint → Abbruch. Erkennt auch
        # alternierende Muster (A-B-A-B), die der alte Step-Vergleich verpasst.
        _call_streak: dict[tuple, int] = {}  # (tool, args_json) → consecutive count
        _text_only_streak = 0  # consecutive text-only answers after tool results
        _announce_streak = 0  # "Ankündigung ohne Aktion"-Nudges in diesem Run
        _plan_completed_this_run = False  # Escape-Hatch: Plan wurde im Lauf abgeschlossen
        # OpenAI-Tool-Definitionen für native Tool-Calls (der Client
        # erkennt lazily, ob der Server das unterstützt, und fällt
        # sonst auf das JSON-Text-Protokoll zurück).
        schemas = self._registry.openai_schemas()

        # #8 Token/Budget-Monitoring: Kontext-Nutzung pro Step tracken.
        _token_history: list[int] = []  # n_ctx_used nach jedem Step

        # #4 Objective-Recitation: Original-Nutzer-Nachricht für Compaction.
        self._original_objective = user_text

        # #7 Self-Verification Nudge: nur EINMAL pro Session.
        _verify_nudged = False

        # #10 (Phase 3) Ralph Loop: Zählt, wie oft der Kontext bereits
        # resetet wurde (bei vorzeitigem "Fertig" mit offenem Plan).
        _ralph_count = 0

        for step in range(self._config.max_steps):
            # Fix 4: Server lehnt native Tools ab → JSON-Protokoll nachtragen.
            self._inject_json_protocol_if_needed()

            # History-Compaction: vor jeder Modell-Aufruf prüfen, ob die
            # History komprimiert werden muss (Context Rot).
            compacted = await self._compact_history()

            # Trace: Step-Start (vor Modell-Aufruf)
            self._trace({
                "event": "step_start",
                "step": step,
                "history_len": len(self._history),
                "compacted": compacted,
            })

            # Fix 5: Mid-Session Lesson-Re-Injektion — Lessons, die SEIT
            # Session-Start neu geschrieben/geändert wurden (z. B. von Lex
            # selbst während dieser Session), werden als Hinweis injiziert.
            added, removed = self._lesson_delta()
            if added or removed:
                bits = []
                if added:
                    bits.append(
                        "New/updated lessons in this session: "
                        + ", ".join(sorted(added))
                    )
                if removed:
                    bits.append(
                        "Removed/relocated lessons: " + ", ".join(sorted(removed))
                    )
                bits.append(
                    "These are NOW active (content is automatically injected "
                    "selectively in the next turn — no manual read_file needed)."
                )
                self._history.append(
                    {"role": "user", "content": "[Memory-Update] " + " ".join(bits)}
                )
                if self._on_event:
                    self._on_event("warning", f"Memory update: {len(added)} new lesson(s)", "")

            # Streamende Anfrage: Prosa wird Chunk-für-Chunk live
            # durchgereicht. Native Tool-Calls kommen strukturiert als
            # Deltas (tool_call_index/name/args) und werden hier pro
            # Index akkumuliert — kein Text-Parsing nötig.
            full = ""
            reasoning = ""
            finish_reason = ""
            # index → {id, name, args} (args wird per Delta angestückt)
            tc_acc: dict[int, dict] = {}
            async for chunk in self._client.chat_stream(
                self._history,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                tools=schemas,
            ):
                if chunk.reasoning:
                    reasoning += chunk.reasoning
                if chunk.finish_reason:
                    finish_reason = chunk.finish_reason
                c = chunk.content
                if c:
                    full += c
                    yield c
                if chunk.tool_call_index is not None:
                    tc = tc_acc.setdefault(
                        chunk.tool_call_index, {"id": "", "name": "", "args": ""}
                    )
                    tc["id"] = tc["id"] or chunk.tool_call_id
                    tc["name"] = tc["name"] or chunk.tool_call_name
                    tc["args"] += chunk.tool_call_args_delta
            answer = full
            log.debug("Step %d Antwort: %s", step, answer[:500])

            # #8 Token/Budget-Monitoring: Kontext-Nutzung tracken.
            # Quadratisches Wachstum (> _TOKEN_BUDGET_WARN_RATIO× pro Step)
            # signalisiert Context-Rot — vorzeitige Compaction wird ausgelöst.
            try:
                usage = await self._client.context_usage()
                if usage and usage["n_ctx"] > 0:
                    _token_history.append(usage["n_ctx_used"])
                    if len(_token_history) >= 3:
                        prev = _token_history[-2]
                        curr = _token_history[-1]
                        if prev > 0 and curr / prev > _TOKEN_BUDGET_WARN_RATIO:
                            log.warning(
                                "Token-Budget: quadratisches Wachstum (%d → %d, %.1f×). "
                                "Vorzeitige Compaction empfohlen.",
                                prev, curr, curr / prev,
                            )
                            if self._on_event:
                                self._on_event(
                                    "warning",
                                    f"Token-Wachstum: {prev}→{curr} ({curr/prev:.1f}×) — "
                                    f"Context-Rot-Risiko.",
                                    "",
                                )
            except Exception:
                pass  # Token-Tracking darf den Loop nicht blockieren

            # Denken des Modells anzeigen (zählt gegen max_tokens)
            # UND persistieren: Kurzer Insight an die Assistant-Message anhängen,
            # damit das Modell seine eigenen Erkenntnisse nicht verliert
            # (Context-Rot-Schutz). Ohne dies würde Compaction "Tor fehlt"
            # aus der History löschen und das Modell blind neu proben.
            if self._on_event and reasoning:
                self._on_event("thinking", reasoning, "")
            # Insight wird an die assistant-Message angehängt (nicht als
            # separate Message), damit die History kompakt bleibt.
            insight_suffix = ""
            if reasoning:
                insight = " ".join(reasoning.split())[:200]
                if len(insight) > 20:
                    insight_suffix = f"\n[Insight: {insight}]"

            # Token-Limit erreicht?
            if self._on_event and finish_reason == "length":
                self._on_event(
                    "warning",
                    f"Token limit reached (max_tokens={self._config.max_tokens}) — "
                    f"answer was truncated. Increase CHATCLI_MAX_TOKENS "
                    f"(thinking tokens count too!).",
                    "",
                )

            # Tool-Calls: primär nativ (strukturierte Deltas), Fallback
            # auf das JSON-Text-Protokoll, wenn der Server keine
            # nativen Tools unterstützt.
            calls: Optional[list[ToolCall]] = None
            json_error: Optional[str] = None
            if tc_acc:
                calls = []
                for idx in sorted(tc_acc):
                    tc = tc_acc[idx]
                    args_raw = tc["args"].strip()
                    try:
                        args = json.loads(args_raw) if args_raw else {}
                    except json.JSONDecodeError as e:
                        # Kaputtes/trunciertes Args-JSON (z. B. Token-Limit
                        # mitten im Stream): NICHT mit leeren Args ausfuehren
                        # (sonst laeuft shell mit leerem Command, apply_diff
                        # ohne Pfad still durch) — als Fehler zurueckgeben.
                        json_error = (
                            f"Tool-Call args for '{tc['name']}' were broken "
                            f"JSON ({e}); the call was NOT executed. "
                            f"Please resend the tool call with complete, "
                            f"untruncated args."
                        )
                        calls.append(ToolCall(tool=tc["name"], args={}))
                        continue
                    if not isinstance(args, dict):
                        args = {}
                    calls.append(ToolCall(tool=tc["name"], args=args))
            elif not self._client.supports_tools:
                # Text-Protokoll: Modell gibt JSON als Content aus.
                calls = parse_tool_calls(answer)

            if calls is None:
                # Leere Antwort ohne Tool-Calls nach einem Tool-Resultat:
                # Das Modell hat offenbar den Faden verloren. Statt die
                # Loop zu beenden, injizieren wir einen Nudge und retryen.
                if not answer.strip() and step > 0:
                    # Sanfter Nudge: Das Modell denkt vielleicht noch.
                    # Nicht drängen, sondern einen Hinweis geben.
                    nudge = (
                        "[System note] Your last answer was empty. "
                        "Think about what to do next — "
                        "respond with text or call a tool."
                    )
                    self._history.append({"role": "user", "content": nudge})
                    if self._on_event:
                        self._on_event("warning", "Empty answer — gentle nudge, retry.", "")
                    continue

                # #7 Self-Verification Nudge: Wenn ein Plan BESTANDEN hat und
                # jetzt ABGESCHLOSSEN ist (alle Schritte erledigt), wird das
                # Modell EINMAL aufgefordert, seine Arbeit zu verifizieren.
                # (Mutually exclusive mit Action-Trigger: bei offenem Plan
                #  feuert der Action-Trigger, bei abgeschlossenem Plan
                #  der Self-Verification Nudge.)
                _planner = self._planner()
                _plan_complete = (
                    _planner is not None
                    and len(_planner.last_plan) > 0
                    and self._plan_block() == ""  # alle Schritte erledigt
                )
                if (
                    step > 0
                    and not _verify_nudged
                    and self._last_tool_success  # letzter Tool-Call war erfolgreich
                    and answer.strip()
                    and _plan_complete
                    # Escape-Hatch: Wenn das Modell den Plan selbst im Lauf
                    # abgeschlossen hat, wird die finale Antwort direkt
                    # akzeptiert (kein Verifizierungs-Nudge).
                    and not _plan_completed_this_run
                ):
                    _verify_nudged = True
                    verify_nudge = (
                        "[System note] You have completed your work. "
                        "Verify NOW briefly that the result is correct: "
                        "Check with a tool (e.g. read_file, shell) whether the "
                        "result is actually correct before you answer."
                    )
                    self._history.append({"role": "user", "content": verify_nudge})
                    if self._on_event:
                        self._on_event(
                            "warning",
                            "Self-verification: model should verify work.",
                            "",
                        )
                    continue

                # --- Action-Trigger ---
                # (1) Offener Plan + Text-only → Nudge / Ralph / Abbruch.
                # (2) Kein aktiver Plan, aber "Ankündigung ohne Aktion"
                #     ("Lass mich lesen …", Insight-Format-Imitation) →
                #     EIN Nudge pro Run; danach wird die Antwort akzeptiert
                #     (die Aktion könnte legitim erledigt sein).
                if answer.strip():
                    plan_block = self._plan_block()
                    if plan_block and step > 0:
                        _text_only_streak += 1
                        # Fix 3: Hard-Gate — progressive Eskalation statt
                        # passivem Warten bis streak >= 2.
                        if _text_only_streak == 1:
                            # Erste Text-only-Antwort: imperativer Nudge mit
                            # dem konkreten nächsten Plan-Schritt + Escape-Hatch
                            # (das Modell kann den Plan selbst abschließen).
                            next_step = self._next_open_plan_step(plan_block)
                            nudge = (
                                "[System note] You gave text only, "
                                "but the plan still has open steps. "
                                f"Execute the NEXT plan step NOW: "
                                f"\"{next_step}\" — call a tool "
                                "(read_file, shell, list_dir). "
                                "Do NOT describe what you would do — DO it."
                            )
                            nudge += (
                                "\nIf you are TRULY done, close the plan "
                                "with plan(action='complete', step=<N>) BEFORE "
                                "giving your final answer."
                            )
                            if not self._last_tool_success:
                                nudge += (
                                    "\nYour last tool call failed. "
                                    "Try a different approach instead of repeating "
                                    "the same command."
                                )
                            self._history.append({"role": "user", "content": nudge})
                            if self._on_event:
                                self._on_event(
                                    "warning",
                                    "Action-Trigger: 1× text without tool call, nudge.",
                                    "",
                                )
                            continue
                        if _text_only_streak >= 2:
                            # #10 (Phase 3) Ralph Loop: Statt Abbruch wird der
                            # Kontext resetet und die Arbeit fortgesetzt.
                            # Bedingung: Ralph Loop aktiv + max. Loops nicht erreicht.
                            ralph_enabled = getattr(
                                self._config, "enable_ralph_loop", False
                            )
                            ralph_max = getattr(self._config, "ralph_max_loops", 2)

                            if ralph_enabled and _ralph_count < ralph_max:
                                _ralph_count += 1
                                # Kontext-Reset: Nur System-Prompt + Originalziel
                                # + aktueller Plan-Status + Fortsetzungs-Anweisung.
                                # Das Modell liest den State aus dem Dateisystem
                                # und setzt die Arbeit fort (frischer Kontext).
                                self._history = [
                                    {"role": "system", "content": self._history[0]["content"]},
                                    {
                                        "role": "user",
                                        "content": (
                                            f"[Ralph Loop #{_ralph_count}] "
                                            f"You said you are done — but the "
                                            f"plan still has open steps. "
                                            f"Continue the work NOW.\n\n"
                                            f"## ORIGINAL GOAL:\n{self._original_objective}\n\n"
                                            f"## Current plan status:\n{plan_block}\n\n"
                                            f"## Instructions:\n"
                                            f"1. Read the current state from the filesystem "
                                            f"(read_file, list_dir).\n"
                                            f"2. Execute the NEXT open plan step.\n"
                                            f"3. If you are truly done (all steps "
                                            f"completed), close the plan with "
                                            f"plan(action='complete') and then give "
                                            f"the final text answer."
                                        ),
                                    },
                                ]
                                _text_only_streak = 0
                                if self._on_event:
                                    self._on_event(
                                        "warning",
                                        f"Ralph Loop #{_ralph_count}: context reset, "
                                        f"work will continue.",

                                    ""                                    )
                                continue

                            # Ralph Loop deaktiviert oder Limit erreicht → Abbruch.
                            self._history.append({
                                "role": "assistant",
                                "content": answer + insight_suffix,
                            })
                            if self._on_event:
                                reason = (
                                    f"Ralph Loop limit ({ralph_max}) reached."
                                    if ralph_enabled
                                    else "Ralph Loop disabled."
                                )
                                self._on_event(
                                    "warning",
                                    f"Action-Trigger: 2× text without tool call — aborting. {reason}",

                                ""                                )
                            yield "(Agent: Action-Trigger — 2× text without tool call, aborting.)"
                            return
                        # Announcement-Detection auch bei aktivem Plan:
                        # Das Modell kündigt eine Aktion an, statt sie
                        # auszuführen → spezifischerer Nudge (statt nur
                        # generischem "nächsten Schritt"-Hinweis). Nach
                        # einem fehlgeschlagenen Tool-Call zusätzlich:
                        # Ansatz wechseln statt denselben Call zu wiederholen.
                        if self._looks_like_announcement(answer) and _announce_streak == 0:
                            _announce_streak = 1
                            nudge = (
                                "[System note] You announced what you "
                                "will do, but did not execute a tool call. "
                                "Execute the announced action NOW — "
                                "call the tool."
                            )
                            if not self._last_tool_success:
                                nudge += (
                                    " The last tool call failed — "
                                    "do NOT repeat the same call, try "
                                    "a different approach (e.g. corrected path)."
                                )
                        else:
                            nudge = (
                                "[System note] You analyzed or planned, but "
                                "did not yet execute the next step. "
                                "Execute the NEXT plan step NOW — call a tool."
                            )
                            if not self._last_tool_success:
                                nudge += (
                                    " The last tool call failed — "
                                    "do NOT repeat the same call, try "
                                    "a different approach (e.g. corrected path, "
                                    "alternative command)."
                                )
                        self._history.append({"role": "user", "content": nudge})
                        if self._on_event:
                            self._on_event(
                                "warning",
                                "Action-Trigger: text without action — gentle nudge.",

                            ""                            )
                        continue
                    elif (
                        self._looks_like_announcement(answer)
                        and _announce_streak == 0
                    ):
                        # Kein aktiver Plan, aber das Modell kündigt an,
                        # statt zu handeln. EIN Nudge pro Run — danach wird
                        # die Text-Antwort akzeptiert (Schleifenschutz).
                        _announce_streak = 1
                        nudge = (
                            "[System note] You announced what you will do, "
                            "but did not execute a tool call. Execute the "
                            "announced action NOW — call the tool. "
                            "If there is nothing left to do, give your final answer."
                        )
                        self._history.append({"role": "user", "content": nudge})
                        if self._on_event:
                            self._on_event(
                                "warning",
                                "Announcement without action — nudge injected.",

                            ""                            )
                        continue

                # Server-Fehler-Detection: Der llama-server liefert bei internen
                # Fehlern kurze Text-Antworten wie "Sorry, try again." — das
                # ist KEINE finale Antwort, sondern ein Retry-Trigger.
                stripped = answer.strip().lower()
                if stripped in (
                    "sorry, try again.",
                    "sorry, try again",
                    "try again.",
                ):
                    self._history.append({"role": "assistant", "content": answer + insight_suffix})
                    if self._on_event:
                        self._on_event(
                            "warning",
                            "Server error detected ('Sorry, try again') — retry.",
                        )
                    continue

                # Normale Antwort — fertig (Content war live gestreamt).
                self._history.append({"role": "assistant", "content": answer + insight_suffix})
                self._trace({"event": "final_answer", "step": step, "text_len": len(answer)})
                return

            # Es wurden Tool-Calls erkannt — echte Arbeit wurde geleistet,
            # daher wird der Text-only-Streak zurückgesetzt. Der Trigger
            # feuert nur bei AUFENEINANDERFOLGENDEN Text-Antworten (echtes
            # Stagnieren), nicht über legitime Tool-Work hinweg.
            self._history.append({"role": "assistant", "content": answer + insight_suffix})
            _announce_streak = 0
            _text_only_streak = 0

            # Drift-Detektion: Wenn das Modell in aufeinanderfolgenden Steps
            # ähnliche Texte wiederholt (Re-Statements), wird ein Nudge
            # injiziert, um den Loop zu durchbrechen.
            if self._drift_check(answer):
                drift_nudge = (
                    "[Drift warning: Your recent answers are very similar. "
                    "You may be repeating the same approach. "
                    "Try a DIFFERENT path or summarize what you "
                    "already know before trying again.]"
                )
                self._history.append({"role": "user", "content": drift_nudge})
                if self._on_event:
                    self._on_event("warning", "Drift detected — nudge injected.", "")

            # Schleifen-Erkennung: Pro (tool, args)-Pair wird gezählt, wie oft
            # es in aufeinanderfolgenden Steps erscheint. Erkennt auch
            # alternierende Muster (Step A: tool1(x), tool2(y) → Step B: tool1(x), tool2(z)
            # → Step A wieder …) — der alte Step-Vergleich sah nur die volle Signatur.
            assert calls is not None  # mypy: oben bereits None-Case behandelt
            current_sigs = set()
            for tc in calls:
                sig = (tc.tool, json.dumps(tc.args, sort_keys=True, ensure_ascii=False))
                current_sigs.add(sig)
            # Zähler aktualisieren: nur Sigs, die JETZT vorkommen, zählen weiter;
            # alle anderen werden auf 0 zurückgesetzt.
            for sig in list(_call_streak):
                if sig in current_sigs:
                    _call_streak[sig] += 1
                else:
                    del _call_streak[sig]
            for sig in current_sigs:
                if sig not in _call_streak:
                    _call_streak[sig] = 1
            # Abbruch, wenn EINER der Sigs die Schwelle erreicht.
            loop_hit = None
            for sig, count in _call_streak.items():
                if count >= _LOOP_ABORT_COUNT:
                    loop_hit = (sig, count)
                    break
            if loop_hit:
                tool_name = loop_hit[0][0]
                count = loop_hit[1]
                if self._on_event:
                    self._on_event(
                        "warning",
                        f"Loop detected: '{tool_name}' {count}× in a row — aborting.",

                    ""                    )
                self._history.append({
                    "role": "assistant",
                    "content": "(Agent: loop detected, aborting.)",
                })
                yield "(Agent: loop detected — same tool calls repeating, aborting.)"
                return

            # Alle Tool-Calls dieses Steps PARALLEl ausführen (sie sind
            # per Definition unabhängig — das Modell hat sie bewusst in
            # einem Schritt gebündelt). Ergebnisse kommen in der
            # ursprünglichen Reihenfolge zurück.
            for call in calls:
                # Kompakte Anzeige des Tool-Calls (vor Ausführung)
                if self._on_event:
                    args_brief = ", ".join(
                        f"{k}={json.dumps(v, ensure_ascii=False)}"
                        for k, v in call.args.items()
                    )
                    if len(args_brief) > 160:
                        args_brief = args_brief[:157] + "..."
                    self._on_event(
                        "tool", f"{call.tool}  {args_brief}".rstrip(), call.tool
                    )

            async def _run_one(call: ToolCall) -> ToolResult:
                if json_error:
                    # Args waren kaputtes JSON → Tool NICHT ausführen.
                    return ToolResult(ok=False, output="", error=json_error)
                tool = self._registry.get(call.tool)
                if tool is None:
                    return ToolResult(
                        ok=False, output="",
                        error=f"Tool '{call.tool}' not available.",
                    )
                # #6 Per-Step-Timeout: verhindert hängende Befehle
                # (z. B. `sleep 3600`) die den gesamten Loop blockieren.
                step_timeout = getattr(self._config, "step_timeout", 60.0)
                try:
                    return await asyncio.wait_for(
                        tool.run(**call.args), timeout=step_timeout
                    )
                except asyncio.TimeoutError:
                    return ToolResult(
                        ok=False, output="",
                        error=(
                            f"Timeout after {step_timeout:.0f}s — "
                            f"command '{call.tool}' may be hanging. "
                            f"Try a different approach or set "
                            f"CHATCLI_STEP_TIMEOUT (env) for longer limits."
                        ),
                    )
                except TypeError as e:
                    return ToolResult(ok=False, output="", error=f"Arg error: {e}")
                except Exception as e:
                    return ToolResult(ok=False, output="", error=str(e))

            results = await asyncio.gather(
                *[_run_one(call) for call in calls]
            )

            # Ergebnisse (in Reihenfolge) als User-Messages an das Modell.
            for call, result in zip(calls, results):
                # #7 Self-Verification Nudge: letztes Tool-Resultat merken.
                self._last_tool_success = result.ok
                # Escape-Hatch: Plan wurde vom Modell im Lauf vollständig
                # abgeschlossen → nächste Text-Antwort ist die finale.
                if call.tool == "plan" and result.ok:
                    _planner_chk = self._planner()
                    if (_planner_chk is not None
                            and len(_planner_chk.last_plan) > 0
                            and self._plan_block() == ""):
                        _plan_completed_this_run = True
                if result.ok:
                    tool_output = result.output
                    # Leeres-Output-Hint: Wenn der Befehl erfolgreich war aber
                    # keine Ausgabe lieferte (z.B. grep ohne Match, test -f),
                    # bekommt das Modell einen expliziten Hinweis, den Ansatz
                    # zu wechseln statt denselben Befehl blind zu wiederholen.
                    stripped = tool_output.strip() if tool_output else ""
                    if not stripped or re.match(r"^\[exit_code: -?\d+\]\s*$", stripped):
                        exit_m = EXIT_CODE_RE.search(tool_output) if tool_output else None
                        code_str = exit_m.group(1) if exit_m else "?"
                        # Tool-spezifische Empty-Output-Hints (Schritt 3):
                        # Jeder Tool-Typ bekommt einen gezielten Hinweis,
                        # statt des generischen "Versuche was anderes".
                        _empty_hints: dict[str, str] = {
                            "shell": (
                                f"Command produced no output (exit_code={code_str}). "
                                "No match / empty. Do NOT repeat the same command. "
                                "Try instead: read_file on the file, a "
                                "different search pattern (e.g. broader grep), or "
                                "list_dir to verify the path."
                            ),
                            "search": (
                                f"Regex search produced no matches (exit_code={code_str}). "
                                "Try a BROADER pattern (less specific), "
                                "or list_dir to check whether the files are even "
                                "in the search path."
                            ),
                            "list_dir": (
                                f"Directory is empty or does not exist (exit_code={code_str}). "
                                "Try the PARENT directory (e.g. list_dir('.') "
                                "instead of list_dir('sub/')) or check the path with shell ls."
                            ),
                            "read_file": (
                                f"File is empty (exit_code={code_str}). "
                                "Check whether it is the right file — the "
                                "content may be in a different file. list_dir in the same "
                                "directory shows candidates."
                            ),
                            "script": (
                                f"Script produced no output (exit_code={code_str}). "
                                "Check the logic: Is an echo/print missing? Is the "
                                "condition never true? Run the script with bash -x "
                                "via shell to trace execution."
                            ),
                            "scrape": (
                                f"URL produced no text content (exit_code={code_str}). "
                                "Possibly a JS-rendered page or 404. "
                                "Try a different URL or web_search for the "
                                "correct source."
                            ),
                            "semantic_search": (
                                f"Sematic search produced no matches (exit_code={code_str}). "
                                "Try different search terms (synonyms, broader) or "
                                "regex-search (search tool) as fallback."
                            ),
                            "web_search": (
                                f"Web search produced no matches (exit_code={code_str}). "
                                "Try different search terms or a more direct "
                                "URL with scrape."
                            ),
                        }
                        tool_output = _empty_hints.get(
                            call.tool,
                            f"Command produced no output (exit_code={code_str}). "
                            "Do NOT repeat the same command. Try a "
                            "different approach.",
                        )
                else:
                    # WICHTIG: Auch bei ok=False kann der output NICHT leer
                    # sein (z. B. Shell-Befehl mit exit_code != 0 liefert
                    # stdout/stderr als Output, aber keinen Error-Text).
                    # Früher wurde dann NUR "FEHLER: " (leer) ans Modell
                    # geschickt — das Modell hatte KEINE Information über den
                    # Zustand und hat blind dieselben Probes wiederholt
                    # (Loop). Jetzt: Fehler + Output zusammen.
                    if result.output and result.output.strip() != "[exit_code: 1]" and not all(
                        l.strip().startswith("[exit_code:") or not l.strip()
                        for l in result.output.splitlines()
                    ):
                        # Output hat echten Inhalt (nicht nur Exit-Code)
                        tool_output = (
                            f"ERROR: {result.error}\n"
                            f"[Tool output despite error]\n{result.output}"
                        ) if result.error else result.output
                    elif result.error:
                        tool_output = f"ERROR: {result.error}"
                    else:
                        # BEIDE leer (nur Exit-Code, kein stdout/stderr):
                        # Modell bekommt KEINE Information → blindes Probing.
                        # Hilfreicher Hinweis statt leeren Fehlers.
                        exit_m = EXIT_CODE_RE.search(result.output) if result.output else None
                        code_str = exit_m.group(1) if exit_m else "?"
                        tool_output = (
                            f"Command failed (exit_code={code_str}) without output. "
                            f"Check whether the command is correct (syntax, path, typos). "
                            f"Try a different approach instead of repeating the same one."
                        )
                # Ingestion-Filter + Offloading: Bei großen Outputs werden
                # nur die wichtigsten Zeilen (Fehler, Kopf, Ende) in den
                # Kontext aufgenommen; der Rest wird offgeladen.
                # AUSNAHME: read_file und script liefern Code/Text, der
                # VOLLSTÄNDIG gelesen werden muss — hier kein Filter.
                _NO_FILTER_TOOLS = {"read_file", "script"}
                if len(tool_output) > _MAX_TOOL_OUTPUT and call.tool not in _NO_FILTER_TOOLS:
                    total = len(tool_output)
                    offload_path = self._offload_tool_output(step, call.tool, tool_output)
                    tool_output = self._ingest_filter(tool_output)
                    trunc_note = f"\n... [Ingestion filter active, {total} chars total]"
                    if offload_path:
                        trunc_note += f"\n[Full output: {offload_path} — readable with read_file.]"
                    tool_output += trunc_note
                # Bei Fehlern: Lern-Hinweis — der Loop soll aus dem Fehler
                # eine Lesson ableiten (compounding intelligence), statt
                # denselben Ansatz nur stur zu wiederholen.
                feedback = (
                    f"[Tool Result | {call.tool} | {'OK' if result.ok else 'ERROR'}]\n"
                    f"{tool_output}"
                )
                if not result.ok:
                    # Relevante Lessons zum FEHLER injizieren (semantisch zum
                    # Fehler + Tool-Name, NICHT zur User-Query). So sieht das
                    # Modell sofort die bereits bekannte Lösung.
                    _error_query = f"{call.tool} {result.error or ''}"
                    try:
                        from ..memory import select_lesson_bodies
                        _rel_lessons = select_lesson_bodies(
                            self._config, _error_query, 2
                        )
                    except Exception:
                        _rel_lessons = []
                    if _rel_lessons:
                        feedback += (
                            "\n[Known lessons for this error:]\n"
                            + "\n\n".join(_rel_lessons)
                        )
                    feedback += (
                        "\n[Learn hint: If you solve the error and the solution "
                        f"is not yet a lesson, write it as a short lesson "
                        f"in wiki topic {self._config.lessons_topic} "
                        "(Error / Cause / Solution)."
                    )
                status = "OK" if result.ok else "ERROR"
                log.info("Tool %s → %s", call.tool, status)
                if self._on_event:
                    if result.ok:
                        self._on_event("tool_output", result.output, call.tool)
                    else:
                        self._on_event(
                            "tool_error",
                            result.error or f"unknown error (tool: {call.tool})",
                            call.tool,
                        )

                # Plan-Status: Nach dem letzten Tool-Resultat dieses Steps
                # wird der aktuelle Plan-Status angehängt, damit das
                # Modell bei langen Aufgaben den Fortschritt behält.
                # CWD-Tracking: Nach einem Shell-Befehl aktualisieren wir den
                # Arbeitsverzeichnis-Hinweis, damit das Modell weiß, wo die
                # stateful Shell steht (cd bleibt erhalten).
                if call.tool == "shell" and result.ok:
                    tool = self._registry.get("shell")
                    if hasattr(tool, "current_cwd"):
                        new_cwd = tool.current_cwd
                        old_cwd = getattr(self, "_tracked_shell_cwd", None)
                        if new_cwd and new_cwd != old_cwd:
                            self._tracked_shell_cwd = new_cwd
                            feedback += f"\n[CWD updated: {new_cwd}]"
                            # Live-CWD-Sync: File-Tools lösen relative Pfade
                            # jetzt gegen das aktuelle CWD auf (nicht nur
                            # gegen den statischen Startup-Root).
                            self._registry.set_live_cwd(new_cwd)
                # Sequenzielle Abhängigkeit-Warnung: File-Tools (read_file,
                # list_dir, write_file, search, apply_diff) lösen relative
                # Pfade gegen den Projekt-Root auf — NICHT gegen die
                # Shell-CWD. Wenn die Shell-CWD abweicht, warnt das Modell,
                # dass der Pfad relativ zum Projekt-Root ist (nicht zum CWD).
                _FILE_TOOLS = {"read_file", "list_dir", "write_file", "search", "apply_diff"}
                if (call.tool in _FILE_TOOLS
                        and self._tracked_shell_cwd
                        and self._tracked_shell_cwd != (self._config.shell_cwd or os.getcwd())):
                    rel_arg = call.args.get("path", "")
                    if rel_arg and not rel_arg.startswith("/"):
                        project_root = self._config.shell_cwd or os.getcwd()
                        feedback += (
                            f"\n[CWD hint] File tools resolve relative paths "
                            f"against the shell CWD ({self._tracked_shell_cwd}) AND "
                            f"the project root ({project_root}). "
                            f"Repo-relative paths (e.g. 'lex/chatcli/...') are "
                            f"resolved against the project root; paths without repo prefix "
                            f"are resolved against the shell CWD."
                        )
                if call is calls[-1]:
                    plan_block = self._plan_block()
                    if plan_block:
                        feedback += f"\n\n{plan_block}"
                self._history.append({"role": "user", "content": feedback})
                # Trace: Tool-Resultat
                self._trace({
                    "step": step,
                    "event": "tool_result",
                    "tool": call.tool,
                    "ok": result.ok,
                    "output_len": len(result.output),
                    "error": (result.error or "")[:200],
                })


        # #1 Early Stopping Generate: Statt blindem Abbruch bei max_steps
        # wird ein letzter LLM-Call OHNE Tools gemacht, um eine sinnvolle
        # Synthese-Antwort zu produzieren (statt nur "Abbruch").
        log.info("max_steps (%d) erreicht — Early Stopping Generate", self._config.max_steps)
        self._history.append({
            "role": "user",
            "content": (
                "You have reached the maximum number of steps. "
                "Now summarize your findings and results "
                "in a clear, complete answer. "
                "Use NO more tools."
            ),
        })
        try:
            final_chunks: list[str] = []
            async for chunk in self._client.chat_stream(
                messages=self._history, tools=None
            ):
                if chunk.content:
                    final_chunks.append(chunk.content)
            final_text = "".join(final_chunks).strip()
            if final_text:
                self._history.append({"role": "assistant", "content": final_text})
                self._trace({"event": "final_answer", "step": step, "text_len": len(final_text)})
                yield final_text
            else:
                yield "(Agent: max steps reached, no final answer generated.)"
        except Exception as e:
            log.warning("Early Stopping Generate fehlgeschlagen: %s", e)
            self._history.append({
                "role": "assistant",
                "content": "(Agent: max steps reached.)",
            })
            yield "(Agent: max steps reached, aborting.)"

    # ------------------------------------------------------------------

