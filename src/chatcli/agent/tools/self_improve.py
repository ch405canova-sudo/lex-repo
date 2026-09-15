"""Self-improve tool: Lex documents learnings and fixes its own code.

Two modes:

1. ``mode='lesson'`` (default): Writes a lesson (error/cause/solution)
   to ``wiki/agent-lessons/``, registers it merge-last in ``index.md`` +
   ``log.md`` and triggers the incremental re-index. The lesson is loaded
   LIVE by the agent loop in the running session (``_lesson_delta``) —
   the feedback loop closes without a restart.

2. ``mode='code'``: After a fix in its own code (``lex/chatcli/**``),
   the tool runs the test suite and commits ONLY if all tests pass.
   Changed files outside ``lex/chatcli/`` are rejected (sandbox boundary:
   never llama.cpp/, model/, .embed_index.db).

The tool is deliberately DUMB: it makes no value judgments about the quality
of the lesson or the fix — that's the model's job. It only enforces the
procedure (format, merge-last, tests-before-commit, commit discipline).
"""

from __future__ import annotations

import asyncio
import os
import re
from datetime import date
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..tools import Tool, ToolResult

# Guardrail: code mode may ONLY commit files under this path (relative to
# repo root). llama.cpp/, model/, .embed_index.db stay outside.
_CODE_PREFIX = "src/chatcli/"


@dataclass(frozen=True)
class _CmdResult:
    """Result of a subprocess call (stdout already decoded)."""
    returncode: int
    stdout: str
# Test run: chatcli test suite from repo root (uv venv, no system Python).
_TEST_CMD = "uv run pytest tests/ -q"
_TEST_TIMEOUT = 600.0

# Quality gate: static analysis before commit (mypy, pylint, bandit, radon, vulture).
# Failure = no commit (same as tests).
_QUALITY_CMDS: list[str] = [
    "uv run mypy src/chatcli --no-error-summary 2>&1 | tail -5",
    "uv run pylint src/chatcli --score=n 2>&1 | tail -5",
    "uv run bandit -r src/chatcli --format=json -q 2>&1 | tail -3",
    "uv run radon cc src/chatcli -a 2>&1 | tail -5",
    "uv run vulture src/chatcli 2>&1 | tail -5",
]


def _slugify(slug: str) -> str:
    """Normalize slug: lowercase, hyphens, max 80 chars."""
    s = slug.strip().lower()
    s = re.sub(r"[^a-z0-9\-]+", "-", s).strip("-")
    s = re.sub(r"-{2,}", "-", s)  # consecutive hyphens collapse
    return s[:80]


def _one_line(text: str, limit: int = 120) -> str:
    """Truncate text to one line (for the index.md summary)."""
    flat = " ".join(text.split())
    if len(flat) > limit:
        flat = flat[: limit - 1].rstrip() + "…"
    return flat


class SelfImproveTool(Tool):
    name = "self_improve"
    description = (
        "Self-improvement: document a learning or fix your own code. "
        "mode='lesson': writes a lesson (error/cause/solution) to "
        "wiki/agent-lessons/, updates index.md + log.md (merge-last) and "
        "reindexes — the lesson is injected LIVE in the session. "
        "mode='code': runs the test suite and commits your changes to "
        "lex/chatcli/** (ONLY if all tests pass; changed files outside are rejected)."
    )
    args_schema: dict[str, Any] = {
        "mode": "str (optional, default 'lesson') — 'lesson' or 'code'",
        "slug": "str (lesson only) — unique filename, e.g. 'sudo-timeout-race'",
        "title": "str (lesson only) — one-line heading of the lesson",
        "error": "str (lesson only) — what went wrong (1-3 sentences)",
        "cause": "str (lesson only) — the root cause (1-3 sentences)",
        "solution": "str (lesson only) — the solution/rule (1-5 sentences, numbered ok)",
        "reindex": "bool (optional, default true) — run incremental re-index",
        "commit_message": "str (code only) — commit message (1 logical step)",
    }

    def __init__(self, wiki_dir: str = "", shell_cwd: str = ".") -> None:
        self._wiki_root = Path(wiki_dir).expanduser().resolve() if wiki_dir else None
        self._cwd = Path(shell_cwd or ".").resolve()

    # ------------------------------------------------------------------

    async def run(self, **kwargs: Any) -> ToolResult:
        mode = str(kwargs.get("mode", "lesson")).strip().lower()
        if mode == "lesson":
            return await self._run_lesson(kwargs)
        if mode == "code":
            return await self._run_code(kwargs)
        return ToolResult(
            ok=False, output="",
            error=f"Unknown mode '{mode}' (allowed: 'lesson', 'code').",
        )

    # ------------------------------------------------------------------
    # Mode 1: write a lesson
    # ------------------------------------------------------------------

    async def _run_lesson(self, kwargs: dict[str, Any]) -> ToolResult:
        if self._wiki_root is None or not self._wiki_root.is_dir():
            return ToolResult(
                ok=False, output="",
                error="Memory root not set (wiki_dir missing).",
            )
        slug = _slugify(str(kwargs.get("slug", "")))
        titel = " ".join(str(kwargs.get("title", "")).split())
        fehler = str(kwargs.get("error", "")).strip()
        ursache = str(kwargs.get("cause", "")).strip()
        loesung = str(kwargs.get("solution", "")).strip()
        if not slug:
            return ToolResult(ok=False, output="", error="'slug' missing (lesson).")
        if not titel:
            return ToolResult(ok=False, output="", error="'title' missing (lesson).")
        if not (fehler and ursache and loesung):
            return ToolResult(
                ok=False, output="",
                error="'error', 'cause' and 'solution' are required fields.",
            )

        lessons_dir = self._wiki_root / "wiki" / "agent-lessons"
        lessons_dir.mkdir(parents=True, exist_ok=True)
        today = date.today().isoformat()
        fname = f"{today}-{slug}.md"
        fpath = lessons_dir / fname

        # Never silently overwrite: a lesson with the same name
        # already exists (today or earlier) → Lex should rename it
        # or maintain the existing lesson via apply_diff.
        existing = sorted(lessons_dir.glob(f"*-{slug}.md"))
        if fpath.exists() or existing:
            return ToolResult(
                ok=False, output="",
                error=(
                    f"Lesson with this slug already exists "
                    f"({', '.join(p.name for p in existing) or fname}). "
                    "Choose a new slug or update the existing file with "
                    "apply_diff."
                ),
            )

        content = (
            f"# {titel}\n"
            f"\n"
            f"> Collected: {today}\n"
            f"\n"
            f"**Error:** {fehler}\n"
            f"\n"
            f"**Cause:** {ursache}\n"
            f"\n"
            f"**Solution:** {loesung}\n"
        )
        fpath.write_text(content, encoding="utf-8")

        notes: list[str] = [f"Lesson written: wiki/agent-lessons/{fname}"]

        # Merge-last: index.md (table) + log.md (append-only).
        idx = self._wiki_root / "wiki" / "index.md"
        if idx.is_file():
            _, note = self._update_index(idx, titel, fname, fehler, today)
            notes.append(note)
        else:
            notes.append("WARNING: wiki/index.md not found — not updated.")

        logf = self._wiki_root / "wiki" / "log.md"
        if logf.is_file():
            entry = (
                f"\n## [{today}] lesson | {titel}\n"
                f"- Disposition: New\n"
                f"- Wiki: agent-lessons/{fname}\n"
            )
            with open(logf, "a", encoding="utf-8") as f:
                f.write(entry)
            notes.append("log.md appended (append-only).")
        else:
            notes.append("WARNING: wiki/log.md not found — not updated.")

        notes.append(
            "The lesson is injected LIVE in this session (delta detection)."
        )

        # Incremental re-index so that semantic_search can find the lesson.
        if str(kwargs.get("reindex", "true")).strip().lower() not in ("0", "false", "no"):
            repo_root = self._wiki_root.parent.parent
            script = repo_root / "lex" / "embed_index.py"
            if script.is_file():
                proc = await asyncio.create_subprocess_exec(
                    "python3", str(script), "reindex",
                    cwd=str(repo_root),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                out, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
                tail = " ".join(out.decode(errors="replace").split())[-200:]
                if proc.returncode == 0:
                    notes.append(f"Re-index ok: {tail}")
                else:
                    notes.append(f"WARNING: re-index failed (rc={proc.returncode}): {tail}")
            else:
                notes.append(f"WARNING: re-index script not found: {script}")

        # Automatic lint after ingest: checks index consistency + raw references.
        lint_script = self._wiki_root.parent / "wiki-skill" / "scripts" / "check_evidence.py"
        if lint_script.is_file():
            try:
                proc = await asyncio.create_subprocess_exec(
                    "python3", str(lint_script), str(self._wiki_root.parent),
                    cwd=str(self._wiki_root.parent),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                out, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
                output = " ".join(out.decode(errors="replace").split())[-300:]
                if proc.returncode == 0:
                    notes.append(f"Lint ok: {output or 'no issues'}")
                else:
                    notes.append(f"Lint note (rc={proc.returncode}): {output}")
            except (OSError, asyncio.TimeoutError) as e:
                notes.append(f"Lint skipped ({e}).")

        return ToolResult(ok=True, output="\n".join(notes))

    @staticmethod
    def _update_index(
        idx: Path, titel: str, fname: str, fehler: str, today: str
    ) -> tuple[bool, str]:
        """Insert the lesson row into the agent-lessons table in index.md.

        Merge-last: file is read, ONLY the row after the
        table header separator is inserted, nothing else touched.
        """
        try:
            text = idx.read_text(encoding="utf-8")
        except OSError as e:
            return False, f"WARNING: index.md not readable ({e})."
        lines = text.splitlines()
        # Find section '## agent-lessons'
        sec = None
        for i, l in enumerate(lines):
            if l.strip() == "## agent-lessons":
                sec = i
                break
        if sec is None:
            return False, "WARNING: section '## agent-lessons' not found in index.md."
        # Find table header + separator after the section
        sep = None
        for i in range(sec + 1, len(lines)):
            if lines[i].strip().startswith("|---"):
                sep = i
                break
        if sep is None:
            return False, "WARNING: agent-lessons table not found in index.md."
        row = (
            f"| [{titel}](agent-lessons/{fname}) "
            f"| {_one_line(fehler)} | {today} |"
        )
        lines.insert(sep + 1, row)
        idx.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return True, "index.md updated (merge-last, row after header)."

    # ------------------------------------------------------------------
    # Mode 2: fix own code (tests + commit)
    # ------------------------------------------------------------------

    async def _run_code(self, kwargs: dict[str, Any]) -> ToolResult:
        msg = " ".join(str(kwargs.get("commit_message", "")).split())
        if not msg:
            return ToolResult(
                ok=False, output="",
                error="'commit_message' missing (code) — 1 logical step.",
            )
        repo_root = await self._git_root()
        if repo_root is None:
            return ToolResult(
                ok=False, output="",
                error="No git repository found in working directory.",
            )

        # Guardrail: only lex/chatcli/** is committed. Other dirty files
        # (wiki, memory, raw/) are legitimate and do NOT block the commit —
        # they simply are not staged.
        status = await self._git(repo_root, ["status", "--porcelain"])
        if status is None:
            return ToolResult(ok=False, output="", error="git status failed.")
        # Porcelain format: "XY path" (2 status chars + space + path).
        changed = [
            l[3:].strip() for l in status.splitlines()
            if l.strip() and not l.startswith("??")
        ]
        code_changed = [p for p in changed if p.startswith(_CODE_PREFIX)]
        if not code_changed:
            return ToolResult(
                ok=False, output="",
                error="No changed files found under lex/chatcli/ — apply the fix first (apply_diff).",
            )
        # Files outside lex/chatcli/ are ignored (not blocked).
        other = [p for p in changed if not p.startswith(_CODE_PREFIX)]
        ignored_note = ""
        if other:
            ignored_note = f"\nIgnored (not committed): {', '.join(other)}"

        # Tests BEFORE commit — failure = no commit.
        test = await self._run_cmd(repo_root, _TEST_CMD)
        if test is None or test.returncode != 0:
            tail = " ".join((test.stdout or "").split())[-500:] if test else ""
            return ToolResult(
                ok=False, output="",
                error=(
                    "Tests FAILED — nothing was committed. "
                    f"Test output (tail): {tail}"
                ),
            )
        tests_ok = " ".join((test.stdout or "").split())[-120:]

        # Quality gate: static analysis (mypy, pylint, bandit, radon, vulture).
        # SOFT report: results are shown in output but do NOT block
        # the commit (only tests are hard blockers). This lets Lex iterate
        # quickly and find issues in the output.
        quality_notes: list[str] = []
        for qcmd in _QUALITY_CMDS:
            qres = await self._run_cmd(repo_root, qcmd)
            if qres is None:
                quality_notes.append("QUALITY GATE: timeout/error")
                break
            tail_q = " ".join((qres.stdout or "").split())[-150:]
            quality_notes.append(tail_q)
        quality_summary = " | ".join(quality_notes) if quality_notes else "OK"

        # Commit: stage only lex/chatcli/**.
        add = await self._git(repo_root, ["add", "lex/chatcli"])
        if add is None:
            return ToolResult(ok=False, output="", error="git add failed.")
        commit = await self._git(repo_root, ["commit", "-m", msg])
        if commit is None:
            return ToolResult(ok=False, output="", error="git commit failed.")

        return ToolResult(
            ok=True,
            output=(
                f"Self-code fix committed: {msg}\n"
                f"Files: {len(code_changed)} changed\n"
                f"Tests: {tests_ok}\n"
                f"Quality gate: {quality_summary}"
                f"{ignored_note}"
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _git_root(self) -> Path | None:
        proc = await self._run_cmd(self._cwd, "git rev-parse --show-toplevel")
        if proc is not None and proc.returncode == 0:
            root = (proc.stdout or "").strip()
            return Path(root) if root else None
        # Fallback: repo is below the tool CWD (e.g. /home/user/project/.git,
        # CWD=/home/user). Test next subdirectory with .git.
        try:
            for d in sorted(self._cwd.iterdir()):
                if d.is_dir() and (d / ".git").is_dir():
                    p = await self._run_cmd(d, "git rev-parse --show-toplevel")
                    if p is not None and p.returncode == 0:
                        root = (p.stdout or "").strip()
                        return Path(root) if root else None
        except OSError:
            pass
        return None

    async def _git(self, root: Path, args: list[str]) -> str | None:
        """Git call with argument list (no shell quoting, messages with
        spaces/quotes stay intact)."""
        proc = await self._run_exec(["git", *args], root)
        if proc is None:
            return None
        if proc.returncode != 0:
            return None
        return proc.stdout or ""

    async def _run_cmd(
        self, root: Path, cmd: str
    ) -> _CmdResult | None:
        """Execute shell command (LC_ALL=C, timeout).

        Returns: ``_CmdResult`` with decoded stdout or None on
        timeout/error.
        """
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                cwd=str(root),
                env={**os.environ, "LC_ALL": "C"},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await asyncio.wait_for(
                proc.communicate(), timeout=_TEST_TIMEOUT
            )
            return _CmdResult(
                returncode=proc.returncode or 0,
                stdout=out.decode(errors="replace"),
            )
        except (asyncio.TimeoutError, OSError):
            return None

    async def _run_exec(
        self, args: list[str], root: Path
    ) -> _CmdResult | None:
        """Execute argument list directly (no shell, no quoting)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(root),
                env={**os.environ, "LC_ALL": "C"},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await asyncio.wait_for(
                proc.communicate(), timeout=_TEST_TIMEOUT
            )
            return _CmdResult(
                returncode=proc.returncode or 0,
                stdout=out.decode(errors="replace"),
            )
        except (asyncio.TimeoutError, OSError):
            return None
