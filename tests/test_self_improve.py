"""Unit-Tests für das Self-Improve-Tool (Lesson-Modus + Code-Modus-Gates)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chatcli.agent.tools.self_improve import SelfImproveTool, _slugify


def _make_wiki(tmp_path: Path) -> Path:
    """Minimal-Wiki-Struktur (wiki/index.md + wiki/log.md) anlegen."""
    wiki = tmp_path / "memory"
    (wiki / "wiki").mkdir(parents=True)
    (wiki / "wiki" / "agent-lessons").mkdir()
    (wiki / "wiki" / "index.md").write_text(
        "# Knowledge Base Index\n\n"
        "## agent-lessons\n\n"
        "| Article | Summary | Updated |\n"
        "|---------|---------|---------|\n",
        encoding="utf-8",
    )
    (wiki / "wiki" / "log.md").write_text("# Wiki Log\n", encoding="utf-8")
    return wiki


@pytest.mark.anyio
async def test_lesson_writes_file_and_updates_index(tmp_path):
    wiki = _make_wiki(tmp_path)
    t = SelfImproveTool(wiki_dir=str(wiki), shell_cwd=str(tmp_path))
    r = await t.run(
        mode="lesson",
        slug="Test-Slug!",
        title="Test-Lesson",
        error="Etwas ist schiefgelaufen.",
        cause="Die Ursache.",
        solution="Die Lösung.",
        reindex=False,
    )
    assert r.ok, r.error
    lessons = list((wiki / "wiki" / "agent-lessons").glob("*.md"))
    assert len(lessons) == 1
    body = lessons[0].read_text(encoding="utf-8")
    assert "Test-Lesson" in body
    assert "**Error:**" in body and "**Cause:**" in body and "**Solution:**" in body
    # index.md: Zeile nach dem Separator eingefügt
    idx = (wiki / "wiki" / "index.md").read_text(encoding="utf-8").splitlines()
    assert any("Test-Lesson" in l for l in idx)
    # log.md: append-only ergänzt
    log = (wiki / "wiki" / "log.md").read_text(encoding="utf-8")
    assert "lesson | Test-Lesson" in log


@pytest.mark.anyio
async def test_lesson_duplicate_slug_rejected(tmp_path):
    wiki = _make_wiki(tmp_path)
    t = SelfImproveTool(wiki_dir=str(wiki), shell_cwd=str(tmp_path))
    args = dict(
        mode="lesson", slug="doppelt", title="A",
        error="f", cause="u", solution="l", reindex=False,
    )
    r1 = await t.run(**args)
    assert r1.ok
    r2 = await t.run(**args)
    assert not r2.ok
    assert "already exists" in r2.error


@pytest.mark.anyio
async def test_lesson_missing_fields_rejected(tmp_path):
    wiki = _make_wiki(tmp_path)
    t = SelfImproveTool(wiki_dir=str(wiki), shell_cwd=str(tmp_path))
    r = await t.run(mode="lesson", slug="x", title="T", error="f")
    assert not r.ok
    assert "required fields" in r.error


def test_slugify_normalizes():
    assert _slugify("Hello World! 123") == "hello-world-123"
    assert _slugify("  a--b  ") == "a-b"


@pytest.mark.anyio
async def test_code_mode_without_changes_fails(tmp_path):
    t = SelfImproveTool(wiki_dir="", shell_cwd=str(tmp_path))
    r = await t.run(mode="code", commit_message="Test")
    assert not r.ok
    # Kein Git-Repo im tmp_path → klare Fehlermeldung
    assert "git" in r.error.lower()


@pytest.mark.anyio
async def test_code_mode_ignores_dirty_files_outside_chatcli(tmp_path):
    """Code-Modus darf NICHT blockiert werden, wenn nur Dateien außerhalb
    von lex/chatcli/ dirty sind (z. B. Wiki/Memory-Updates)."""
    import subprocess

    # Mini-Git-Repo anlegen mit lex/chatcli/ Struktur
    repo = tmp_path / "repo"
    (repo / "lex" / "chatcli" / "src").mkdir(parents=True)
    (repo / "lex" / "memory" / "wiki").mkdir(parents=True)

    def git(*args):
        subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True)

    git("init")
    git("config", "user.email", "test@test.com")
    git("config", "user.name", "Test")

    # Initiale Datei committen
    (repo / "lex" / "chatcli" / "src" / "mod.py").write_text("x = 1\n")
    (repo / "lex" / "memory" / "wiki" / "index.md").write_text("# Index\n")
    git("add", "-A")
    git("commit", "-m", "init")

    # Ändern: eine Datei in lex/chatcli/ UND eine außerhalb
    (repo / "lex" / "chatcli" / "src" / "mod.py").write_text("x = 2\n")
    (repo / "lex" / "memory" / "wiki" / "index.md").write_text("# Index v2\n")

    t = SelfImproveTool(wiki_dir="", shell_cwd=str(repo))
    r = await t.run(mode="code", commit_message="fix: mod.py geändert")
    # Sollte NICHT blockiert sein — die Wiki-Datei wird ignoriert.
    # (Tests können hier nicht laufen, da kein pytest im Repo → Test-Fehler,
    #  aber der BLOCKIERT-Fehler darf NICHT auftreten.)
    if not r.ok:
        assert "BLOCKED" not in r.error, f"Sollte nicht blockiert sein: {r.error}"
        # Expected error: no changed files or tests failed
        assert "No changed files" in r.error or "Tests FAILED" in r.error or "uv" in r.error
