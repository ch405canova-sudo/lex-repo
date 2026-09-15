"""Patch tool: precise line replacement by line number.

Complements ``apply_diff`` for the most common case — changing a SINGLE
line — without the grep → sed -n → build SEARCH block ritual:

    patch(path='src/foo.py', line=42, replace='x = 1')

The tool reads the file, replaces exactly that one line (1-based),
and reports old + new line with context (±1 lines), so the model
can immediately see whether the change landed in the right place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..tools import Tool, ToolResult
from .base import resolve_in_roots

_MAX_FILE_SIZE = 1_000_000  # 1 MB (wie bei ReadFileTool)


class PatchLineTool(Tool):
    name = "patch"
    description = (
        "Replaces a line (or consecutive lines) of a file "
        "by line number (1-based) — without the grep/sed/apply_diff ritual.\n"
        "Arguments: path, line, replace (new content, incl. indentation), "
        "context (optional: expected old content for verification), "
        "count (optional: number of consecutive lines, default 1).\n"
        "Reports old + new line with ±1 context lines. "
        "For larger blocks: apply_diff; for new files: write_file."
    )
    args_schema: dict[str, Any] = {
        "path": "str — relative path to the file",
        "line": "int — 1-based line number to replace",
        "replace": "str — new content (incl. indentation); for count>1: multi-line string",
        "context": "str (optional) — expected content of the old line (verification)",
        "count": "int (optional) — number of consecutive lines to replace (default 1)",
    }

    def __init__(self, base_dir: str = ".", extra_roots: list[Path] | None = None) -> None:
        self._original_roots = [Path(base_dir).resolve()]
        if extra_roots:
            self._original_roots += [Path(r).resolve() for r in extra_roots]
        self._live_cwd: Path | None = None

    def set_live_cwd(self, cwd: str) -> None:
        """Updates the live CWD (used as preferred root)."""
        self._live_cwd = Path(cwd).resolve()

    @property
    def _effective_roots(self) -> list[Path]:
        """Roots with live CWD first (if set and != original root)."""
        if self._live_cwd and self._live_cwd != self._original_roots[0]:
            return [self._live_cwd] + self._original_roots
        return self._original_roots

    def _safe(self, rel: str) -> Path:
        return resolve_in_roots(
            rel, self._effective_roots, prefer_existing=True, prefer_existing_dir=True
        )

    async def run(self, **kwargs: Any) -> ToolResult:
        rel = kwargs.get("path", "")
        line = kwargs.get("line")
        replace = kwargs.get("replace", "")
        if not rel:
            return ToolResult(ok=False, output="", error="No 'path' specified.")
        try:
            line = int(line)
        except (TypeError, ValueError):
            return ToolResult(
                ok=False, output="",
                error=f"'line' must be a number, got: {line!r}.",
            )
        if line < 1:
            return ToolResult(ok=False, output="", error="'line' must be >= 1.")
        try:
            count = int(kwargs.get("count") or 1)
        except (TypeError, ValueError):
            return ToolResult(ok=False, output="", error=f"'count' must be a number, got: {kwargs.get('count')!r}.")
        if count < 1:
            return ToolResult(ok=False, output="", error="'count' must be >= 1.")

        try:
            path = self._safe(rel)
        except (ValueError, OSError) as e:
            return ToolResult(ok=False, output="", error=str(e))

        if not path.is_file():
            return ToolResult(
                ok=False, output="",
                error=f"File not found: {rel} (use write_file for new files).",
            )
        if path.stat().st_size > _MAX_FILE_SIZE:
            return ToolResult(
                ok=False, output="",
                error=f"File too large ({path.stat().st_size} bytes), limit: {_MAX_FILE_SIZE}.",
            )

        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            return ToolResult(ok=False, output="", error=str(e))

        lines = content.splitlines(keepends=True)
        if line > len(lines) or line + count - 1 > len(lines):
            return ToolResult(
                ok=False, output="",
                error=(
                    f"Line {line} (to {line + count - 1}) out of range — "
                    f"file has only {len(lines)} lines. "
                    f"Check the number (grep -n '<pattern>' {rel})."
                ),
            )

        old = lines[line - 1]
        context = (kwargs.get("context") or "").strip()
        if context and context not in old.rstrip("\r\n"):
            return ToolResult(
                ok=False, output="",
                error=(
                    f"Context check failed: line {line} is "
                    f"{old.rstrip()!r}, expected '{context}'. Wrong line number?"
                ),
            )

        old_text = "".join(lines[line - 1 : line + count - 1]).rstrip("\r\n")
        if old_text == replace.rstrip("\r\n"):
            return ToolResult(
                ok=False, output="",
                error=(
                    f"Line {line} is already exactly: {old_text!r} — "
                    "nothing to change."
                ),
            )

        new_lines = [l if l.endswith("\n") else l + "\n" for l in replace.splitlines()]
        if count > 1 and len(new_lines) != count:
            return ToolResult(
                ok=False, output="",
                error=f"'replace' has {len(new_lines)} lines, but count={count}.",
            )
        lines[line - 1 : line + count - 1] = new_lines
        new_content = "".join(lines)

        try:
            path.write_text(new_content, encoding="utf-8")
        except Exception as e:
            return ToolResult(ok=False, output="", error=str(e))

        # Verifikations-Kontext: ±1 Zeilen um die geänderte Stelle(n).
        ctx: list[str] = []
        for i in range(max(0, line - 2), min(len(lines), line + count - 1 + 1)):
            mark = "→" if line - 1 <= i < line + count - 1 else " "
            ctx.append(f"{mark} {i + 1:5d} | {lines[i].rstrip()}")
        return ToolResult(
            ok=True,
            output=(
                f"✓ Line {line}"
                + (f"-{line + count - 1}" if count > 1 else "")
                + f" in {rel} replaced.\n"
                + "\n".join(ctx)
            ),
        )
