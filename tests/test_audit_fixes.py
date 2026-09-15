"""Tests für die Audit-Fixes: kaputte Tool-Args-JSON, Multi-Tool-Call-Deltas,
Compaction an Tool-Paar-Grenzen, Guardrail-Regex-Dedup, REPL-Answer-Buffer."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chatcli.config import Config
from chatcli.llama_client import LlamaClient, StreamChunk
from chatcli.agent.loop import AgentLoop
from chatcli.agent.tools.base import Tool, ToolRegistry, ToolResult
from chatcli.agent.tools.planner import PlannerTool


# ── Test-Dummies ─────────────────────────────────────────────────────


class _EchoTool(Tool):
    """Tool, das sein Argument echo-t (zur Ausführungsnachweis-Prüfung)."""

    name = "echo"
    description = "echo"
    args_schema = {"msg": "str — zu echo-ender Text"}

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def run(self, **kwargs) -> ToolResult:
        self.calls.append(kwargs)
        return ToolResult(ok=True, output=str(kwargs.get("msg", "")))


class _FakeClient:
    """Duck-typed LlamaClient: liefert pro ask()-Aufruf die nächste
    'Antwort' aus dem Skript (Liste von StreamChunk-Listen)."""

    def __init__(self, script: list[list[StreamChunk]],
                 context_used_pct: float = 0.95) -> None:
        self._script = script
        self.supports_tools = True
        self._ctx_pct = context_used_pct

    async def chat_stream(self, messages, **kwargs):
        if not self._script:
            raise AssertionError("FakeClient: kein Skript mehr vorhanden")
        for chunk in self._script.pop(0):
            yield chunk

    async def context_usage(self):
        """Simuliert Kontext-Nutzung (für Compaction-Trigger in Tests)."""
        return {"n_ctx": 131072, "n_ctx_used": int(131072 * self._ctx_pct)}


def _tc_chunk(index: int, name: str = "", args: str = "") -> StreamChunk:
    return StreamChunk(
        tool_call_index=index,
        tool_call_id=f"call_{index}",
        tool_call_name=name,
        tool_call_args_delta=args,
    )


def _make_loop(
    script: list[list[StreamChunk]],
    tmp_path: Path,
) -> tuple[AgentLoop, _EchoTool]:
    cfg = Config(wiki_dir=str(tmp_path), enable_self_improve=False)
    tool = _EchoTool()
    registry = ToolRegistry()
    registry.register(tool)
    client = _FakeClient(script)
    loop = AgentLoop(client=client, registry=registry, config=cfg)  # type: ignore[arg-type]
    return loop, tool


# ── Fix 1: kaputtes Args-JSON → ToolResult(ok=False), kein stiller Lauf ──


@pytest.mark.anyio
async def test_broken_args_json_not_executed(tmp_path):
    """Trunciertes Args-JSON (Token-Limit) darf das Tool NICHT mit leeren
    Args ausführen — es muss ein Fehler-ToolResult zurückkommen."""
    script = [
        # Step 1: Tool-Call mit kaputtem JSON (mitten im String abgeschnitten)
        [_tc_chunk(0, "echo", '{"msg": "hallo')],
        # Step 2: normale Abschluss-Antwort
        [StreamChunk(content="Erledigt.", finish_reason="stop")],
    ]
    loop, tool = _make_loop(script, tmp_path)
    out = "".join([c async for c in loop.ask("test")])
    assert out == "Erledigt."
    # Das Tool wurde NICHT ausgeführt (kein stiller Lauf mit leeren Args).
    assert tool.calls == []
    # Das Tool-Resultat in der History meldet den JSON-Fehler.
    results = [
        m for m in loop.history
        if m["role"] == "user" and m["content"].startswith("[Tool Result |")
    ]
    assert len(results) == 1
    assert "ERROR" in results[0]["content"]
    assert "broken" in results[0]["content"]


@pytest.mark.anyio
async def test_valid_args_json_executed(tmp_path):
    """Valides Args-JSON läuft normal durch (Regression: der Fehler-Pfad
    darf den normalen Pfad nicht brechen)."""
    script = [
        [_tc_chunk(0, "echo", '{"msg": "hallo"}')],
        [StreamChunk(content="Fertig.", finish_reason="stop")],
    ]
    loop, tool = _make_loop(script, tmp_path)
    out = "".join([c async for c in loop.ask("test")])
    assert out == "Fertig."
    assert tool.calls == [{"msg": "hallo"}]


# ── Fix 2: mehrere tool_calls-Einträge in einem Delta ─────────────────


def test_parse_tool_call_delta_multiple():
    """Ein Delta mit zwei tool_calls-Einträgen liefert beide (vorher
    ging der zweite still verloren)."""
    delta = {
        "tool_calls": [
            {"index": 0, "id": "a", "function": {"name": "echo", "arguments": '{"m": 1'}},
            {"index": 1, "id": "b", "function": {"name": "echo", "arguments": '{"m": 2'}},
        ]
    }
    result = LlamaClient._parse_tool_call_delta(delta)
    assert len(result) == 2
    assert result[0] == (0, "a", "echo", '{"m": 1')
    assert result[1] == (1, "b", "echo", '{"m": 2')


def test_parse_tool_call_delta_empty():
    assert LlamaClient._parse_tool_call_delta({}) == []
    assert LlamaClient._parse_tool_call_delta({"content": "hi"}) == []


@pytest.mark.anyio
async def test_two_tool_calls_in_one_step(tmp_path):
    """Zwei parallele Tool-Calls (unterschiedliche Indizes) werden beide
    ausgeführt — auch wenn sie in derselben Stream-Runde ankommen."""
    script = [
        [
            _tc_chunk(0, "echo", '{"msg": "eins"}'),
            _tc_chunk(1, "echo", '{"msg": "zwei"}'),
        ],
        [StreamChunk(content="Beide fertig.", finish_reason="stop")],
    ]
    loop, tool = _make_loop(script, tmp_path)
    out = "".join([c async for c in loop.ask("test")])
    assert out == "Beide fertig."
    assert tool.calls == [{"msg": "eins"}, {"msg": "zwei"}]


# ── Compaction: KEEP-Grenze an Tool-Paar-Grenzen ─────────────────────


def test_compaction_does_not_split_tool_pair(tmp_path):
    """Die KEEP-Grenze darf nicht zwischen einer Assistant-Message (mit
    Tool-Call) und ihrem Tool-Resultat schneiden — sonst landet ein
    verwaistes Tool-Resultat am Anfang der KEEP-Zone."""
    cfg = Config(wiki_dir=str(tmp_path), enable_self_improve=False)
    registry = ToolRegistry()
    loop = AgentLoop(client=_FakeClient([]), registry=registry, config=cfg)  # type: ignore[arg-type]

    # History: System + 12× (Assistant + Tool-Resultat) + 1 Assistant
    # = 25 Non-System-Messages (Trigger 24 → Compaction greift).
    # KEEP = 10 → blinde Grenze (len-10) fällt auf Index 16 =
    # Tool-Resultat R7 → ohne Fix wäre das Paar (A7, R7) getrennt
    # und R7 läge verwaist am Anfang der KEEP-Zone.
    hist = [{"role": "system", "content": "sys"}]
    for i in range(12):
        hist.append({"role": "assistant", "content": f"A{i}"})
        hist.append(
            {"role": "user",
             "content": f"[Tool Result | echo | OK]\nR{i}"}
        )
    hist.append({"role": "assistant", "content": "A-last"})
    loop._history = hist

    asyncio.run(loop._compact_history())

    # Die erste KEEP-Message darf KEIN Tool-Resultat sein.
    non_system = [m for m in loop._history if m["role"] != "system"]
    # Nach Compaction: System + 1 Zusammenfassung + KEEP-Messages.
    keep_part = [
        m for m in loop._history
        if not (
            m["role"] == "user"
            and m["content"].startswith("[Situation Report]")
        )
    ]
    keep_msgs = keep_part[1:]  # ohne System
    assert not keep_msgs[0]["content"].startswith("[Tool Result |"), (
        "Compaction hat ein Tool-Paar getrennt: erste KEEP-Message ist "
        "ein verwaistes Tool-Resultat."
    )
    # Das Paar ist komplett in der KEEP-Zone (Assistant direkt davor).
    assert keep_msgs[0]["role"] == "assistant"


# ── Fix 6: Guardrail-Regex-Dedup ─────────────────────────────────────


def test_guardrail_rm_rf_single_rule():
    """Die doppelte rm -rf-Regel ist zu EINER zusammengeführt — alle
    relevanten Fälle werden weiterhin hart blockiert."""
    from chatcli.agent.tools.guardrails import _HARD_BLOCK, check_command

    rm_rf_rules = [
        (p, r) for p, r in _HARD_BLOCK if "rm -rf" in r
    ]
    assert len(rm_rf_rules) == 1, "Es muss genau eine rm -rf-Regel geben."

    assert check_command("rm -rf /") == ("hard", "rm -rf on /, ~ or $HOME")
    assert check_command("rm -rf ~") == ("hard", "rm -rf on /, ~ or $HOME")
    assert check_command("rm -rf $HOME") == ("hard", "rm -rf on /, ~ or $HOME")
    assert check_command("sudo rm -rf / --no-preserve-root") == (
        "hard", "rm -rf on /, ~ or $HOME"
    )
    # Legitime Nutzung bleibt bei confirm (nicht hard).
    assert check_command("rm -rf ./build")[0] == "confirm"


# ── Fix 3: REPL-Answer-Buffer ─────────────────────────────────────────


def test_answer_buffer_resets_on_tool_event(tmp_path):
    """Zwischentext vor einem Tool-Event wird verworfen; nur der Text
    seit dem letzten Tool-Event bleibt als finale Antwort."""
    from chatcli.repl import _AnswerBuffer

    buf = _AnswerBuffer()
    buf.add("Ich schaue mal nach… ")
    buf.add(" ")
    buf.on_event("tool", "shell  command=ls")
    buf.add("Die Ausgabe zeigt ")
    buf.add("alles.")
    assert buf.answer == "Die Ausgabe zeigt alles."


def test_answer_buffer_no_events(tmp_path):
    from chatcli.repl import _AnswerBuffer

    buf = _AnswerBuffer()
    buf.add("Hallo ")
    buf.add("Welt.")
    assert buf.answer == "Hallo Welt."


# ── semantic_search: numpy-Scoring + mtime-Cache ──────────────────


def _make_index_db(tmp_path: Path, dim: int = 4) -> Path:
    """Kleine Embedding-DB im echten Schema (embed_index.py-kompatibel)."""
    import sqlite3
    import struct

    db_path = tmp_path / "test_embed.db"
    db = sqlite3.connect(str(db_path))
    db.execute(
        "CREATE TABLE chunks (id INTEGER PRIMARY KEY, path TEXT, "
        "chunk_idx INTEGER, line_no INTEGER, text TEXT, emb BLOB, mtime REAL)"
    )
    # Drei Chunks: A liegt in Query-Richtung, B mittig, C entgegengesetzt.
    chunks = [
        ("a.md", 1, "alpha", [1.0, 0.0, 0.0, 0.0]),
        ("b.md", 2, "beta", [0.5, 0.5, 0.0, 0.0]),
        ("c.md", 3, "gamma", [-1.0, 0.0, 0.0, 0.0]),
    ]
    for i, (path, ln, text, vec) in enumerate(chunks):
        db.execute(
            "INSERT INTO chunks (id, path, chunk_idx, line_no, text, emb) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (i, path, 0, ln, text, struct.pack(f"{dim}f", *vec)),
        )
    db.commit()
    db.close()
    return db_path


@pytest.mark.anyio
async def test_semantic_search_numpy_scores_and_cache(tmp_path, monkeypatch):
    """numpy-Bruteforce liefert korrekte Top-k-Reihenfolge, und der
    mtime-Cache liefert beim zweiten Laden dasselbe Array (kein
    zweites DB-Laden)."""
    import struct

    from chatcli.agent.tools import semantic_search
    from chatcli.agent.tools.semantic_search import SemanticSearchTool

    db_path = _make_index_db(tmp_path)
    tool = SemanticSearchTool(db_path=str(db_path))

    # _embed monkeypatchen: Query-Vektor = [1, 0, 0, 0] (Richtung Chunk A).
    # (synchron — wird über asyncio.to_thread aufgerufen)
    def _fake_embed(text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]

    monkeypatch.setattr(semantic_search, "_embed", _fake_embed)

    r = await tool.run(query="was auch immer", max_results=3)
    assert r.ok
    lines = r.output.splitlines()
    assert len(lines) == 3
    # Reihenfolge: A (Kosinus 1.0) > B (0.7071) > C (-1.0, entgegengesetzt).
    assert lines[0].startswith("1.0000  a.md:1")
    assert lines[1].startswith("0.7071  b.md:2")
    assert lines[2].startswith("-1.0000  c.md:3")

    # Cache: zweites _load_embeddings liefert dasselbe Array-Objekt.
    rows1, embs1 = tool._load_embeddings()
    rows2, embs2 = tool._load_embeddings()
    assert embs1 is embs2
    assert rows1 == rows2
    assert embs1.shape == (3, 4)


@pytest.mark.anyio
async def test_semantic_search_cache_invalidated_on_reindex(tmp_path, monkeypatch):
    """Nach einem Reindex (neue DB-Mtime) wird der Cache verworfen und
    die neuen Vektoren geladen."""
    from chatcli.agent.tools import semantic_search
    from chatcli.agent.tools.semantic_search import SemanticSearchTool

    db_path = _make_index_db(tmp_path)
    tool = SemanticSearchTool(db_path=str(db_path))

    def _fake_embed(text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]

    monkeypatch.setattr(semantic_search, "_embed", _fake_embed)

    tool._load_embeddings()
    mtime_before = os.path.getmtime(str(db_path))

    # DB "reindexen": mtime in die Zukunft schieben.
    future = mtime_before + 10
    os.utime(str(db_path), (future, future))
    rows, embs = tool._load_embeddings()
    assert embs.shape == (3, 4)
    assert tool._emb_cache[0] == future  # Cache auf neue Mtime gesetzt


# ── Fix 3: Leere Antwort → Nudge statt Abbruch ────────────────────────


@pytest.mark.anyio
async def test_empty_answer_triggers_nudge_not_abort(tmp_path):
    """Regression (2026-09-09): Wenn das Modell nach einem Tool-Resultat
    eine leere Antwort liefert (kein Text, kein Tool-Call), injiziert die
    Loop einen Nudge und retryt — statt die Session zu beenden."""
    script = [
        # Step 1: Tool-Call
        [_tc_chunk(0, "echo", '{"msg": "hallo"}')],
        # Step 2: Leere Antwort (Modell verliert Faden)
        [StreamChunk(content="", finish_reason="stop")],
        # Step 3: Nach Nudge → normale Antwort
        [StreamChunk(content="Erledigt.", finish_reason="stop")],
    ]
    loop, tool = _make_loop(script, tmp_path)
    out = "".join([c async for c in loop.ask("test")])
    # Die finale Antwort kommt durch (nicht Abbruch).
    assert "Erledigt." in out
    # Der Nudge wurde injiziert (als User-Message in der History).
    nudges = [
        m for m in loop.history
        if m["role"] == "user" and "[System note]" in m.get("content", "")
    ]
    assert len(nudges) == 1


# ── Compaction: Situation-Bericht + Insight-Persistenz ───────────────


@pytest.mark.anyio
async def test_compaction_preserves_insights(tmp_path):
    """Insights aus Reasoning werden im Situation-Bericht erhalten."""
    cfg = Config(wiki_dir=str(tmp_path), enable_self_improve=False)
    registry = ToolRegistry()
    # >95% Kontext belegt → Compaction wird triggered.
    loop = AgentLoop(
        client=_FakeClient([], context_used_pct=0.96),
        registry=registry, config=cfg,
    )  # type: ignore[arg-type]

    hist = [{"role": "system", "content": "sys"}]
    # 30 Paare (genug für Fallback-Trigger) mit Insights in Assistant-Messages.
    for i in range(30):
        hist.append({
            "role": "assistant",
            "content": f"A{i}\n[Insight: Tor ist nicht installiert auf diesem System]"
        })
        hist.append({"role": "user", "content": f"[Tool Result | shell | OK]\nR{i}"})
    hist.append({"role": "assistant", "content": "A-last"})
    loop._history = hist

    await loop._compact_history()

    # Der Situation-Bericht muss Insights enthalten.
    summary_msg = next(
        m for m in loop._history
        if m["role"] == "user" and m["content"].startswith("[Situation Report]")
    )
    assert "Tor ist nicht installiert" in summary_msg["content"]


@pytest.mark.anyio
async def test_compaction_not_triggered_below_95(tmp_path):
    """Unter 95% Kontext wird NICHT komprimiert (selbst bei vielen Messages)."""
    cfg = Config(wiki_dir=str(tmp_path), enable_self_improve=False)
    registry = ToolRegistry()
    # Nur 50% Kontext belegt → kein Trigger.
    loop = AgentLoop(
        client=_FakeClient([], context_used_pct=0.50),
        registry=registry, config=cfg,
    )  # type: ignore[arg-type]

    hist = [{"role": "system", "content": "sys"}]
    for i in range(100):
        hist.append({"role": "assistant", "content": f"A{i}"})
        hist.append({"role": "user", "content": f"[Tool Result | echo | OK]\nR{i}"})
    loop._history = hist

    await loop._compact_history()

    # History sollte unverändert sein (kein [Situation-Bericht]).
    assert not any(
        m["role"] == "user" and m["content"].startswith("[Situation Report]")
        for m in loop._history
    )


@pytest.mark.anyio
async def test_insight_appended_to_assistant_message(tmp_path):
    """Reasoning wird als [Insight: ...] an die Assistant-Message angehängt."""
    cfg = Config(wiki_dir=str(tmp_path), enable_self_improve=False)
    registry = ToolRegistry()
    script = [
        [StreamChunk(reasoning="Tor fehlt auf diesem System, wir brauchen Dante.",
                      finish_reason="stop")],
    ]
    loop, _ = _make_loop(script, tmp_path)
    list([c async for c in loop.ask("test")])

    # Die Assistant-Message muss den Insight enthalten.
    last_assistant = [m for m in loop.history if m["role"] == "assistant"][-1]
    assert "[Insight:" in last_assistant["content"]
    assert "Tor fehlt" in last_assistant["content"]


# ── Action-Trigger: Text ohne Tool-Call nach aktiver Plan ─────────────


def _make_loop_with_plan(
    script: list[list[StreamChunk]],
    tmp_path: Path,
    **cfg_kwargs,
) -> tuple[AgentLoop, PlannerTool]:
    """Loop mit PlannerTool (aktiver Plan mit offenen Schritten)."""
    cfg = Config(wiki_dir=str(tmp_path), enable_self_improve=False, **cfg_kwargs)
    tool = _EchoTool()
    planner = PlannerTool()
    registry = ToolRegistry()
    registry.register(tool)
    registry.register(planner)
    client = _FakeClient(script)
    loop = AgentLoop(client=client, registry=registry, config=cfg)  # type: ignore[arg-type]
    return loop, planner


@pytest.mark.anyio
async def test_action_trigger_nudges_on_text_only_with_active_plan(tmp_path):
    """Action-Trigger (2026-09-10): Nach Tool-Erfolg + Textantwort (kein
    weiterer Tool-Call) wird ein sanfter Nudge injiziert, wenn noch ein
    aktiver Plan mit offenen Schritten existiert."""
    script = [
        # Step 0: Modell legt Plan an (plan tool call)
        [_tc_chunk(0, "plan", '{"action": "set", "goal": "Test", "steps": ["Schritt 1"]}'),
         _tc_chunk(1, "echo", '{"msg": "hallo"}')],
        # Step 1: Modell gibt Textantwort (kein Tool-Call) → Action-Trigger Nudge
        [StreamChunk(content="Ich habe die Datei erstellt.", finish_reason="stop")],
        # Step 2: Nach Nudge → Modell schließt Plan ab + ruft Tool auf
        [_tc_chunk(0, "plan", '{"action": "complete"}'),
         _tc_chunk(1, "echo", '{"msg": "weiter"}')],
        # Step 3: Finale Antwort (Plan abgeschlossen → kein Action-Trigger)
        [StreamChunk(content="Alles erledigt.", finish_reason="stop")],
        # Step 4: Nach Self-Verification Nudge → Bestätigung
        [StreamChunk(content="Ja, alles ist korrekt.", finish_reason="stop")],
    ]
    loop, planner = _make_loop_with_plan(
        script, tmp_path, enable_ralph_loop=False
    )
    out = "".join([c async for c in loop.ask("test")])

    # Finale Antwort kommt durch.
    assert "Alles erledigt." in out
    # Der Action-Trigger-Nudge wurde injiziert.
    nudges = [
        m for m in loop.history
        if m["role"] == "user" and "Execute the NEXT plan step NOW" in m.get("content", "")
    ]
    assert len(nudges) == 1


@pytest.mark.anyio
async def test_action_trigger_aborts_on_second_text_only(tmp_path):
    """Action-Trigger (Ralph Loop deaktiviert): Wenn das Modell ZWEIMAL Text
    ohne Tool-Call gibt (mit aktivem Plan), wird die Loop abgebrochen."""
    script = [
        # Step 0: Plan + Tool-Call
        [_tc_chunk(0, "plan", '{"action": "set", "goal": "Test", "steps": ["Schritt 1"]}'),
         _tc_chunk(1, "echo", '{"msg": "hallo"}')],
        # Step 1: Text ohne Tool-Call → 1. Nudge
        [StreamChunk(content="Ich habe analysiert.", finish_reason="stop")],
        # Step 2: Nochmal Text ohne Tool-Call → Abbruch (Ralph Loop aus)
        [StreamChunk(content="Ich weiß nicht was ich tun soll.", finish_reason="stop")],
    ]
    loop, planner = _make_loop_with_plan(
        script, tmp_path, enable_ralph_loop=False
    )
    out = "".join([c async for c in loop.ask("test")])

    # Abbruch-Meldung ist in der Ausgabe.
    assert "aborting" in out
    # Two Nudges were injected (1st + 2nd → abort).
    nudges = [
        m for m in loop.history
        if m["role"] == "user" and "Execute the NEXT plan step NOW" in m.get("content", "")
    ]
    assert len(nudges) == 1  # Nur der erste Nudge, dann Abbruch


@pytest.mark.anyio
async def test_ralph_loop_resets_context_on_second_text_only(tmp_path):
    """Ralph Loop (Phase 3): Wenn das Modell ZWEIMAL Text ohne Tool-Call gibt
    (mit aktivem Plan), wird der Kontext resetet und die Arbeit fortgesetzt."""
    script = [
        # Step 0: Plan + Tool-Call
        [_tc_chunk(0, "plan", '{"action": "set", "goal": "Test", "steps": ["Schritt 1", "Schritt 2"]}'),
         _tc_chunk(1, "echo", '{"msg": "hallo"}')],
        # Step 1: Text ohne Tool-Call → 1. Nudge (streak=1)
        [StreamChunk(content="Ich habe analysiert.", finish_reason="stop")],
        # Step 2: Nochmal Text ohne Tool-Call → streak=2 → Ralph Loop (Kontext-Reset)
        [StreamChunk(content="Ich weiß nicht was ich tun soll.", finish_reason="stop")],
        # Step 3: Nach Ralph-Loop → Modell schließt Plan ab
        [_tc_chunk(0, "plan", '{"action": "complete"}')],
        # Step 4: Finale Antwort (Plan abgeschlossen)
        [StreamChunk(content="Alles erledigt.", finish_reason="stop")],
        # Step 5: Nach Self-Verification Nudge → Bestätigung
        [StreamChunk(content="Ja, alles ist korrekt.", finish_reason="stop")],
    ]
    loop, planner = _make_loop_with_plan(script, tmp_path)
    out = "".join([c async for c in loop.ask("test")])

    # Finale Antwort kommt durch (Ralph Loop hat die Arbeit fortgesetzt).
    assert "Alles erledigt." in out
    # Ralph-Loop-Meldung ist in der History.
    ralph_msgs = [
        m for m in loop.history
        if m["role"] == "user" and "[Ralph Loop" in m.get("content", "")
    ]
    assert len(ralph_msgs) == 1
    # Der Ralph-Loop-Reset enthält das Originalziel.
    assert "ORIGINAL GOAL" in ralph_msgs[0]["content"]


@pytest.mark.anyio
async def test_action_trigger_no_nudge_without_active_plan(tmp_path):
    """Action-Trigger feuert NICHT ohne aktiven Plan: Textantwort nach
    Tool-Erfolg ist eine normale finale Antwort."""
    script = [
        # Step 0: Tool-Call (kein plan)
        [_tc_chunk(0, "echo", '{"msg": "hallo"}')],
        # Step 1: Textantwort → normale Endung (kein Plan aktiv)
        [StreamChunk(content="Alles erledigt.", finish_reason="stop")],
    ]
    loop, planner = _make_loop_with_plan(script, tmp_path)
    out = "".join([c async for c in loop.ask("test")])

    assert "Alles erledigt." in out
    # Kein Action-Trigger-Nudge injiziert.
    nudges = [
        m for m in loop.history
        if m["role"] == "user" and "Execute the NEXT plan step NOW" in m.get("content", "")
    ]
    assert len(nudges) == 0
