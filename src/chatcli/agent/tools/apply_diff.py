"""Apply-Diff tool: precise search/replace editing instead of full-text rewrite.

The tool accepts one or more search/replace blocks:

    <<<<<<< SEARCH
    old code (must exist exactly)
    =======
    new code
    >>>>>>> REPLACE

Each block is applied ONCE to the file; blocks are processed in the order
given (i.e. a block can build on the result of the previous block). This
saves enormous token counts compared to ``write_file`` because only the
changed lines need to be transmitted.

Additional mode ``replace_all=true``: replaces EVERY occurrence of a search
text (good for constant renames).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..tools import Tool, ToolResult
from .base import resolve_in_roots

_MAX_FILE_SIZE = 1_000_000  # 1 MB (same as ReadFileTool)

_BLOCK_HEADER = "<<<<<<< SEARCH"
_BLOCK_MIDDLE = "======="
_BLOCK_FOOTER = ">>>>>>> REPLACE"


def _parse_blocks(diff: str) -> tuple[list[tuple[str, str]], str]:
    """Split the diff text into (search, replace) pairs.

    Returns: (blocks, error). ``error`` is empty on success; for a
    malformed block (missing '=======' or missing/truncated footer) it
    contains a clear error message — the caller must NOT write anything
    to the file in that case (otherwise the entire rest of the diff text
    would land literally as a replacement in the file).
    """
    blocks: list[tuple[str, str]] = []
    lines = diff.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == _BLOCK_HEADER:
            start_line = i + 1  # 1-based for the error message
            search_lines: list[str] = []
            i += 1
            middle = -1
            while i < len(lines):
                if lines[i].strip() == _BLOCK_MIDDLE:
                    middle = i
                    break
                search_lines.append(lines[i])
                i += 1
            if middle == -1:
                return blocks, (
                    f"Malformed block at line {start_line}: "
                    "'=======' is missing. Expected format:\n"
                    "  <<<<<<< SEARCH\n  <old text>\n  =======\n"
                    "  <new text>\n  >>>>>>> REPLACE"
                )
            replace_lines: list[str] = []
            i = middle + 1
            found_footer = False
            while i < len(lines):
                if lines[i].strip() == _BLOCK_FOOTER:
                    found_footer = True
                    i += 1
                    break
                replace_lines.append(lines[i])
                i += 1
            if not found_footer:
                return blocks, (
                    f"Malformed block at line {start_line}: "
                    "'>>>>>>> REPLACE' is missing or truncated (it must be "
                    "EXACTLY 7x '>'). Expected format:\n"
                    "  <<<<<<< SEARCH\n  <old text>\n  =======\n"
                    "  <new text>\n  >>>>>>> REPLACE"
                )
            blocks.append(
                ("\n".join(search_lines), "\n".join(replace_lines))
            )
        else:
            i += 1
    return blocks, ""


def _apply_one(content: str, search: str, replace: str) -> tuple[str, str]:
    """Replace the FIRST exact occurrence of ``search`` in ``content``.

    Returns: (new_content, status) with status in {"ok", "not_found",
    "ambiguous"}.
    """
    count = content.count(search)
    if count == 0:
        return content, "not_found"
    if count > 1:
        return content, "ambiguous"
    return content.replace(search, replace, 1), "ok"


def _apply_all(content: str, search: str, replace: str) -> tuple[str, int]:
    """Replace ALL occurrences. Returns: (new_content, count)."""
    count = content.count(search)
    return content.replace(search, replace), count


class ApplyDiffTool(Tool):
    name = "apply_diff"
    description = (
        "Performs precise search/replace edits on a file "
        "(multiple blocks per call). Format per block:\n"
        "  <<<<<<< SEARCH\n  <exact old text>\n  =======\n"
        "  <new text>\n  >>>>>>> REPLACE\n"
        "IMPORTANT: Write markers EXACTLY — 7x '<' + 'SEARCH', 7x '=', "
        "7x '>' + 'REPLACE' (no more, no less). "
        "The search text must occur EXACTLY (incl. whitespace/indentation) in the "
        "file — always use read_file first to see the exact text. With replace_all=true, "
        "EVERY occurrence is replaced. For new files: use write_file."
    )
    args_schema: dict[str, Any] = {
        "path": "str — relative path to the file",
        "diff": "str — one or more SEARCH/REPLACE blocks",
        "replace_all": "bool (optional, default false) — replace all occurrences",
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
        # prefer_existing: file must exist; prefer_existing_dir:
        # if the same file exists under multiple roots (e.g.
        # wiki/index.md in CWD AND in Memory-Root), the root with
        # the already-existing top-level directory wins.
        return resolve_in_roots(
            rel, self._effective_roots, prefer_existing=True, prefer_existing_dir=True
        )

    async def run(self, **kwargs: Any) -> ToolResult:
        rel = kwargs.get("path", "")
        diff = kwargs.get("diff", "")
        replace_all = bool(kwargs.get("replace_all", False))
        if not rel:
            return ToolResult(ok=False, output="", error="No 'path' specified.")
        if not diff:
            return ToolResult(ok=False, output="", error="No 'diff' specified.")
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

        blocks, parse_error = _parse_blocks(diff)
        if parse_error:
            return ToolResult(ok=False, output="", error=parse_error)
        if not blocks:
            return ToolResult(
                ok=False, output="",
                error=(
                    "No valid block found. Expected format:\n"
                    "  <<<<<<< SEARCH\n  <old text>\n  =======\n"
                    "  <new text>\n  >>>>>>> REPLACE"
                ),
            )

        original = content
        applied = 0
        errors: list[str] = []
        for idx, (search, replace) in enumerate(blocks, 1):
            if replace_all:
                new_content, n = _apply_all(content, search, replace)
                if n == 0:
                    errors.append(f"Block {idx}: search text not found.")
                    continue
                content = new_content
                applied += n
            else:
                new_content, status = _apply_one(content, search, replace)
                if status == "not_found":
                    errors.append(f"Block {idx}: search text not found.")
                elif status == "ambiguous":
                    errors.append(
                        f"Block {idx}: search text occurs multiple times — "
                        "provide more specific context or use replace_all=true."
                    )
                else:
                    content = new_content
                    applied += 1

        if content == original:
            return ToolResult(
                ok=False, output="",
                error="No change applied. " + " | ".join(errors),
            )

        try:
            path.write_text(content, encoding="utf-8")
        except Exception as e:
            return ToolResult(ok=False, output="", error=str(e))

        summary = f"✓ {applied} change(s) applied to {rel}."
        if errors:
            summary += f"  [Warning: {len(errors)} block(s) failed: {'; '.join(errors)}]"
        return ToolResult(ok=True, output=summary)
