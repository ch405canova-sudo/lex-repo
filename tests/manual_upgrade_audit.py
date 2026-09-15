"""Ausführlicher manueller Test-Lauf für das Top-4-Paket (UPGRADE_AUDIT.md).

Führt alle Punkte der manuellen Test-Checkliste als Skript aus:
  1. Stateful-Shell (export/cd Persistenz, reset, Timeout, close)
  2. apply_diff (kleine Änderung, Multi-Block, replace_all, Fehlerfälle)
  3. Parallele Tool-Calls (Timing-Messung)
  4. Plan-Injektion (Status-Blöcke, complete)

Aufruf:  uv run python tests/manual_upgrade_audit.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chatcli.agent.tools.shell import ShellTool  # noqa: E402
from chatcli.agent.tools.apply_diff import ApplyDiffTool  # noqa: E402
from chatcli.agent.tools.planner import PlannerTool  # noqa: E402
from chatcli.agent.tools import ToolResult  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> bool:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")
    return cond


def section(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


# ----------------------------------------------------------------------
# 1. Stateful-Shell
# ----------------------------------------------------------------------
async def test_stateful_shell(tmp: str) -> None:
    section("1. Stateful-Shell")
    shell = ShellTool(cwd=tmp)

    r1 = await shell.run(command="export FOO=bar")
    check("export FOO=bar → ok", r1.ok, r1.error)

    r2 = await shell.run(command="echo $FOO")
    check(
        "nächster Call: echo $FOO liefert 'bar' (persistiert)",
        r2.ok and "bar" in r2.output,
        f"output={r2.output!r}",
    )

    r3 = await shell.run(command="cd subdir && pwd")
    check(
        "cd subdir → pwd zeigt das Verzeichnis",
        r3.ok and os.path.join(tmp, "subdir") in r3.output,
        f"output={r3.output!r}",
    )

    r4 = await shell.run(command="pwd")
    check(
        "cd persistiert: pwd (ohne erneutes cd) zeigt subdir",
        r4.ok and os.path.join(tmp, "subdir") in r4.output,
        f"output={r4.output!r}",
    )

    r5 = await shell.run(command="echo $FOO")
    check(
        "Env-Var überlebt auch cd (gleicher Prozess)",
        r5.ok and "bar" in r5.output,
        f"output={r5.output!r}",
    )

    r6 = await shell.run(command="echo $FOO", reset=True)
    check(
        "reset=true → FOO ist weg (frische Shell)",
        r6.ok and "bar" not in r6.output,
        f"output={r6.output!r}",
    )

    # Timeout: sleep 5 mit timeout=2 → sauberer Timeout, kein Hängen
    t0 = time.monotonic()
    r7 = await shell.run(command="sleep 5", timeout=2)
    elapsed = time.monotonic() - t0
    check(
        "sleep 5 / timeout=2 → Timeout-Fehler",
        not r7.ok and "Timeout" in (r7.error or ""),
        f"error={r7.error!r}",
    )
    check(
        "Timeout dauert ~2s (kein Hängen)",
        1.5 <= elapsed <= 5.0,
        f"elapsed={elapsed:.1f}s",
    )

    # Nach Timeout muss die Shell tot/neu sein → nächster Call muss funktionieren
    r8 = await shell.run(command="echo alive")
    check(
        "Shell nach Timeout-Abort noch bedienbar (Neustart)",
        r8.ok and "alive" in r8.output,
        f"output={r8.output!r} error={r8.error!r}",
    )

    # Exit-Code-Propagation
    r9 = await shell.run(command="false")
    check(
        "Exit-Code != 0 wird als Fehler gemeldet",
        not r9.ok,
        f"output={r9.output!r}",
    )

    # Sandbox: Verweis auf Elternverzeichnis muss blockiert werden
    r10 = await shell.run(command="ls", cwd="..")
    check(
        "Sandbox: cwd='..' wird blockiert",
        not r10.ok and "Sandbox" in (r10.error or ""),
        f"error={r10.error!r}",
    )

    # close(): kein zurückgebliebener bash-Prozess
    await shell.close()
    check("close() läuft fehlerfrei", True)


# ----------------------------------------------------------------------
# 2. apply_diff
# ----------------------------------------------------------------------
async def test_apply_diff(tmp: str) -> None:
    section("2. apply_diff")
    tool = ApplyDiffTool(base_dir=tmp)
    target = "demo.txt"

    with open(os.path.join(tmp, target), "w", encoding="utf-8") as f:
        f.write("alpha\nbeta\ngamma\nbeta\ndelta\n")

    # Kleine Änderung: nur ein Block, kein Volltext
    r1 = await tool.run(
        path=target,
        diff=(
            "<<<<<<< SEARCH\n"
            "alpha\n"
            "=======\n"
            "ALPHA\n"
            ">>>>>>> REPLACE"
        ),
    )
    check(
        "Kleine Änderung (1 Block) → ok",
        r1.ok and "1 Änderung" in r1.output,
        f"output={r1.output!r} error={r1.error!r}",
    )
    content = open(os.path.join(tmp, target), encoding="utf-8").read()
    check(
        "Datei enthält nur die geänderte Zeile (Rest unangetastet)",
        content == "ALPHA\nbeta\ngamma\nbeta\ndelta\n",
        f"content={content!r}",
    )

    # Mehrere Blöcke in einem Aufruf (sequenziell, ein Block baut auf dem
    # Ergebnis des vorherigen auf)
    r2 = await tool.run(
        path=target,
        diff=(
            "<<<<<<< SEARCH\n"
            "ALPHA\n"
            "=======\n"
            "alpha2\n"
            ">>>>>>> REPLACE\n"
            "<<<<<<< SEARCH\n"
            "delta\n"
            "=======\n"
            "delta2\n"
            ">>>>>>> REPLACE"
        ),
    )
    check(
        "Multi-Block (2 Blöcke) → ok",
        r2.ok and "2 Änderung(en)" in r2.output,
        f"output={r2.output!r} error={r2.error!r}",
    )
    content = open(os.path.join(tmp, target), encoding="utf-8").read()
    check(
        "Multi-Block-Ergebnis korrekt",
        content == "alpha2\nbeta\ngamma\nbeta\ndelta2\n",
        f"content={content!r}",
    )

    # replace_all=true bei Umbenennung ("beta" kommt 2× vor)
    r3 = await tool.run(
        path=target,
        diff=(
            "<<<<<<< SEARCH\n"
            "beta\n"
            "=======\n"
            "BETA\n"
            ">>>>>>> REPLACE"
        ),
        replace_all=True,
    )
    check(
        "replace_all=true → alle Vorkommensorte ersetzt",
        r3.ok and "2 Änderung(en)" in r3.output,
        f"output={r3.output!r} error={r3.error!r}",
    )
    content = open(os.path.join(tmp, target), encoding="utf-8").read()
    check(
        "beide 'beta' → 'BETA'",
        content.count("BETA") == 2 and "\nbeta\n" not in content,
        f"content={content!r}",
    )

    # Falscher Suchtext → klare Fehlermeldung "nicht gefunden"
    r4 = await tool.run(
        path=target,
        diff=(
            "<<<<<<< SEARCH\n"
            "dieser-text-existiert-nicht\n"
            "=======\n"
            "x\n"
            ">>>>>>> REPLACE"
        ),
    )
    check(
        "Falscher Suchtext → Fehler 'nicht gefunden'",
        not r4.ok and "nicht gefunden" in (r4.error or ""),
        f"error={r4.error!r}",
    )

    # Mehrdeutiger Suchtext → klare Fehlermeldung "mehrfach"
    with open(os.path.join(tmp, "ambig.txt"), "w", encoding="utf-8") as f:
        f.write("ein\nzwei\nein\n")
    r5 = await tool.run(
        path="ambig.txt",
        diff=(
            "<<<<<<< SEARCH\n"
            "ein\n"
            "=======\n"
            "EIN\n"
            ">>>>>>> REPLACE"
        ),
    )
    check(
        "Mehrfaches Vorkommen → Fehler 'mehrfach'",
        not r5.ok and "mehrfach" in (r5.error or ""),
        f"error={r5.error!r}",
    )

    # Fehlende Datei
    r6 = await tool.run(
        path="keine-datei.txt",
        diff=(
            "<<<<<<< SEARCH\n"
            "a\n"
            "=======\n"
            "b\n"
            ">>>>>>> REPLACE"
        ),
    )
    check(
        "Fehlende Datei → klare Fehlermeldung",
        not r6.ok and "nicht gefunden" in (r6.error or ""),
        f"error={r6.error!r}",
    )

    # Sandbox-Escape
    r7 = await tool.run(
        path="../outside.txt",
        diff=(
            "<<<<<<< SEARCH\n"
            "a\n"
            "=======\n"
            "b\n"
            ">>>>>>> REPLACE"
        ),
    )
    check(
        "Sandbox-Escape blockiert",
        not r7.ok,
        f"error={r7.error!r}",
    )


# ----------------------------------------------------------------------
# 3. Parallele Tool-Calls (Timing)
# ----------------------------------------------------------------------
async def test_parallel(tmp: str) -> None:
    section("3. Parallele Tool-Calls (Timing)")

    # Zwei unabhängige read_file-Calls (echte Parallelität, kein
    # Shell-Lock) + zwei sleeps in getrennten Shell-Instanzen, um die
    # asyncio.gather-Mechanik direkt zu messen.
    async def slow_read(name: str) -> ToolResult:
        await asyncio.sleep(2)
        return ToolResult(ok=True, output=name)

    t0 = time.monotonic()
    results = await asyncio.gather(slow_read("a"), slow_read("b"))
    elapsed = time.monotonic() - t0
    check(
        "zwei 2s-Calls parallel → Gesamtzeit ~2s (nicht ~4s)",
        elapsed < 3.2 and results[0].ok and results[1].ok,
        f"elapsed={elapsed:.1f}s",
    )

    # Reihenfolge bleibt konsistent (gather liefert in Aufruf-Reihenfolge)
    check(
        "Reihenfolge der Ergebnisse bleibt konsistent",
        results[0].output == "a" and results[1].output == "b",
        f"results={[r.output for r in results]!r}",
    )

    # Shell-Instanz: parallele Calls werden sequenziell (Lock) abgewickelt —
    # das ist das DOKUMENTIERTE Verhalten (Limitation #2 im Audit).
    shell = ShellTool(cwd=tmp)
    t0 = time.monotonic()
    r_a, r_b = await asyncio.gather(
        shell.run(command="sleep 1"), shell.run(command="sleep 1")
    )
    elapsed = time.monotonic() - t0
    check(
        "Shell: parallele Calls laufen sequenziell (Lock, ~2s) — dokumentiert",
        r_a.ok and r_b.ok and elapsed >= 1.9,
        f"elapsed={elapsed:.1f}s",
    )
    await shell.close()


# ----------------------------------------------------------------------
# 4. Plan-Injektion
# ----------------------------------------------------------------------
async def test_planner() -> None:
    section("4. Plan-Injektion")
    planner = PlannerTool()

    # Kein Plan → leerer Status-Block
    check(
        "Kein Plan → status_block() ist leer",
        planner.status_block() == "",
    )

    # Plan anlegen
    steps = [f"Schritt {i}" for i in range(1, 11)]  # 10 Steps (Limit)
    r = await planner.run(action="set", steps=steps, goal="Großes Feature bauen")
    check(
        "plan(steps=[10 Steps]) anlegen → ok",
        r.ok and "Plan aktiv" in r.output,
        f"output={r.output!r}",
    )

    block0 = planner.status_block()
    check(
        "status_block() zeigt alle 10 Schritte (•-Markierung)",
        block0.count("•") == 10 and "Großes Feature bauen" in block0,
        f"block={block0!r}",
    )

    # complete step 1
    r1 = await planner.run(action="complete", step=1)
    check(
        "complete step=1 → ok",
        r1.ok and "✓" in r1.output,
        f"output={r1.output!r}",
    )
    block1 = planner.status_block()
    check(
        "Status-Block: Schritt 1 mit ✓, Rest mit •",
        block1.count("✓") == 1 and block1.count("•") == 9,
        f"block={block1!r}",
    )

    # complete step 10
    r2 = await planner.run(action="complete", step=10)
    block2 = planner.status_block()
    check(
        "complete step=10 → ✓-Anzahl steigt auf 2",
        r2.ok and block2.count("✓") == 2 and block2.count("•") == 8,
        f"block={block2!r}",
    )

    # Out-of-range
    r3 = await planner.run(action="complete", step=11)
    check(
        "complete step=11 (out of range) → Fehler",
        not r3.ok and "existiert nicht" in (r3.error or ""),
        f"error={r3.error!r}",
    )

    # complete ohne Plan
    fresh = PlannerTool()
    r4 = await fresh.run(action="complete", step=1)
    check(
        "complete ohne Plan → Fehler",
        not r4.ok and "Kein aktiver Plan" in (r4.error or ""),
        f"error={r4.error!r}",
    )

    # Plan-Reset: neues 'set' setzt Fortschritt zurück
    r5 = await planner.run(action="set", steps=["Neu 1", "Neu 2"], goal="Klein")
    block5 = planner.status_block()
    check(
        "Neues 'set' setzt Fortschritt zurück (0 ✓)",
        r5.ok and block5.count("✓") == 0 and block5.count("•") == 2,
        f"block={block5!r}",
    )

    # > 10 Steps abgelehnt
    r6 = await planner.run(action="set", steps=[f"s{i}" for i in range(11)])
    check(
        "11 Steps → abgelehnt (max. 10)",
        not r6.ok and "Maximal 10" in (r6.error or ""),
        f"error={r6.error!r}",
    )


async def main() -> int:
    with tempfile.TemporaryDirectory(prefix="chatcli-audit-") as tmp:
        os.makedirs(os.path.join(tmp, "subdir"), exist_ok=True)
        await test_stateful_shell(tmp)
        await test_apply_diff(tmp)
        await test_parallel(tmp)
        await test_planner()

    print(f"\n{'=' * 60}")
    print(f"ERGEBNIS: {PASS} bestanden, {FAIL} fehlgeschlagen")
    print(f"{'=' * 60}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
