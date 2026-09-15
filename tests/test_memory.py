"""Unit-Tests für memory.py (Lessons-Injektion) und Pfad-Auflösung."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chatcli.config import Config
from chatcli import memory


def test_config_default_prompt_uses_absolute_paths(tmp_path):
    """Der Default-System-Prompt muss absolute Pfade enthalten (kein
    'lex/memory/...'), sonst bricht das Tool-Sandbox-CWD die Pfade."""
    c = Config(wiki_dir=str(tmp_path / "memory"))
    assert c.system_prompt, "Default-System-Prompt wurde nicht gebaut"
    assert str(c.wiki_root) in c.system_prompt
    assert "lex/memory/raw" not in c.system_prompt
    assert "lex/wiki-skill/SKILL.md" not in c.system_prompt


def test_config_skill_dir_derived_from_wiki_dir(tmp_path):
    c = Config(wiki_dir=str(tmp_path / "memory"))
    assert c.skill_dir == tmp_path / "wiki-skill"


def test_lessons_block_loads_topic(tmp_path):
    wiki = tmp_path / "memory" / "wiki"
    topic = wiki / "agent-lessons"
    topic.mkdir(parents=True)
    (topic / "some-error.md").write_text(
        "# Some Error\n\n> Collected: 2026-09-06\n\n**Fehler:** X\n",
        encoding="utf-8",
    )
    c = Config(wiki_dir=str(tmp_path / "memory"))
    block = memory.lessons_block(c)
    assert "Some Error" in block
    assert "Collected" not in block  # Metadata-Header wird gefiltert


def test_lessons_block_empty_when_missing(tmp_path):
    c = Config(wiki_dir=str(tmp_path / "memory"))
    assert memory.lessons_block(c) == ""


def test_style_injection_no_lessons_without_query(tmp_path):
    """Ohne Query (Start): KEINE Lessons injiziert — nur Stil + Memory-Anleitung."""
    wiki = tmp_path / "memory" / "wiki"
    (wiki / "agent-lessons").mkdir(parents=True)
    (wiki / "agent-lessons" / "some-error.md").write_text(
        "# Some Error\n\n**Fehler:** X\n", encoding="utf-8"
    )
    c = Config(wiki_dir=str(tmp_path / "memory"))
    block = memory.style_injection(c)
    assert "Injizierte Lessons" not in block
    assert "Some Error" not in block
    # Stil-Regeln und Memory-Anleitung sind trotzdem da
    assert "Active Memory" in block


def test_style_injection_contains_lessons_with_query(tmp_path):
    """Mit Query (pro Turn): Lessons WERDEN injiziert (selektiv)."""
    wiki = tmp_path / "memory" / "wiki"
    (wiki / "agent-lessons").mkdir(parents=True)
    (wiki / "agent-lessons" / "some-error.md").write_text(
        "# Some Error\n\n**Fehler:** X\n", encoding="utf-8"
    )
    c = Config(wiki_dir=str(tmp_path / "memory"))
    # Mit Query: selektive Injektion aktiv (Fallback auf neueste N,
    # weil Embedding-Server im Test nicht erreichbar ist).
    block = memory.style_injection(c, query="some error about X")
    assert "Injected Lessons" in block
    assert "Some Error" in block


@pytest.mark.anyio
async def test_scrape_save_to_raw_writes_file(tmp_path):
    """save_to_raw muss eine Raw-Datei mit Metadata-Header schreiben und
    nur einen kurzen Echo zurückgeben (Kontext-Schutz)."""
    from chatcli.agent.tools.scrape import ScrapeTool

    t = ScrapeTool(python="python3", wiki_dir=str(tmp_path / "memory"))
    r = t._save_to_raw(
        "https://example.com/foo-bar",
        "# Foo Bar\n\nInhalt der Seite.",
        None, None,
    )
    assert r.ok
    raw_dir = tmp_path / "memory" / "raw" / "examplecom"
    files = list(raw_dir.glob("*.md"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert "# Foo Bar" in content
    assert "> Source: https://example.com/foo-bar" in content
    assert "> Collected:" in content
    # Echo ist kurz, nicht der Volltext
    assert len(r.output) < 4000
    assert "Source saved" in r.output


@pytest.mark.anyio
async def test_scrape_save_to_raw_requires_wiki_dir(tmp_path):
    from chatcli.agent.tools.scrape import ScrapeTool

    t = ScrapeTool(python="python3")  # ohne wiki_dir
    result = await t.run(url="https://example.com", save_to_raw=True)
    assert not result.ok
    assert "wiki_dir" in result.error


# --- Selektive Lessons-Injektion -------------------------------------------


def _make_lessons_topic(tmp_path, n=5):
    wiki = tmp_path / "memory" / "wiki"
    topic = wiki / "agent-lessons"
    topic.mkdir(parents=True)
    for i in range(n):
        (topic / f"lesson-{i}.md").write_text(
            f"# Lesson {i}\n\n> Collected: 2026-09-{i:02d}\n\n"
            f"**Fehler:** Fehler Nummer {i}\n",
            encoding="utf-8",
        )
    return tmp_path / "memory"


def test_lessons_block_without_query_is_latest_n(tmp_path):
    """Ohne Query: neueste N (altes Start-Verhalten)."""
    mem = _make_lessons_topic(tmp_path, n=5)
    c = Config(wiki_dir=str(mem), max_injected_lessons=2)
    block = memory.lessons_block(c)
    assert "Lesson 4" in block  # neueste
    assert "Lesson 3" in block
    assert "Lesson 0" not in block


def test_select_lesson_bodies_fallback_when_server_down(tmp_path, monkeypatch):
    """Embedding-Server weg → Fallback auf neueste N (kein Crash)."""
    mem = _make_lessons_topic(tmp_path, n=5)
    monkeypatch.setattr(
        memory, "_EMBED_URL", "http://127.0.0.1:1/embedding"
    )
    c = Config(wiki_dir=str(mem), selective_lesson_count=2)
    bodies = memory.select_lesson_bodies(c, "irgendeine Aufgabe", 2)
    assert len(bodies) == 2
    assert "Lesson 4" in bodies[0]  # Fallback = neueste
    assert "Lesson 3" in bodies[1]


def test_select_lesson_bodies_empty_topic(tmp_path):
    mem = _make_lessons_topic(tmp_path, n=0)
    c = Config(wiki_dir=str(mem))
    assert memory.select_lesson_bodies(c, "x", 3) == []


def test_style_injection_with_query_falls_back_cleanly(tmp_path, monkeypatch):
    """style_injection(query=…) darf bei Server-Ausfall nicht crashen."""
    mem = _make_lessons_topic(tmp_path, n=3)
    monkeypatch.setattr(
        memory, "_EMBED_URL", "http://127.0.0.1:1/embedding"
    )
    c = Config(wiki_dir=str(mem), selective_lesson_count=2)
    block = memory.style_injection(c, query="Pfad-Sandbox-Problem")
    assert "Injected Lessons" in block
    assert "Lesson 2" in block  # Fallback neueste
