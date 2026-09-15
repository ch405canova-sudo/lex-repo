"""File operations: read, list, write, regex search."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from ..tools import Tool, ToolResult
from .base import resolve_in_roots

_MAX_FILE_SIZE = 1_000_000  # 1 MB


class ReadFileTool(Tool):
    name = "read_file"
    description = "Reads a text file and returns its content (max 1 MB)."
    args_schema = {"path": "str — relative path to the file"}

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

    def _safe(self, rel: str) -> tuple[Path, list[str]]:
        result = resolve_in_roots(
            rel, self._effective_roots, prefer_existing=True, alternatives=True
        )
        # alternatives=True garantiert ein Tuple (mypy kann den Union nicht eingrenzen)
        path, alts = result  # type: ignore[misc]
        return path, alts

    async def run(self, **kwargs: Any) -> ToolResult:
        rel = kwargs.get("path", "")
        try:
            path, alts = self._safe(rel)
        except (ValueError, OSError) as e:
            return ToolResult(ok=False, output="", error=str(e))

        if not path.is_file():
            # Hilfreicher Fuzzy-Hinweis: Zeige ähnliche Dateien, damit das
            # Modell nicht blind denselben falschen Pfad wiederholt.
            hint = ""
            parent = path.parent
            filename = path.name
            target_stem = Path(filename).stem
            candidates: list[str] = []

            def _fuzzy_scan(directory: Path) -> None:
                """Scans a directory for .md files whose stem contains
                target_stem as a substring (date-prefix tolerant)."""
                try:
                    for f in sorted(directory.iterdir()):
                        if f.is_file() and f.suffix == ".md":
                            stem = f.stem
                            # Substring-Match: "file-tool-cwd-vs-repo-pfad" findet
                            # "2026-09-12-file-tool-cwd-vs-repo-pfad"
                            if target_stem in stem or stem in target_stem:
                                candidates.append(str(f.relative_to(directory)))
                except (OSError, PermissionError):
                    pass

            # 1) Parent-Verzeichnis des aufgelösten Pfads
            if parent.is_dir():
                _fuzzy_scan(parent)

            # 2) Falls parent nicht existiert: in allen Roots shallow-suchen
            if not candidates:
                for root in self._effective_roots:
                    try:
                        for f in sorted(root.iterdir()):
                            if f.is_file() and f.suffix == ".md":
                                stem = f.stem
                                if target_stem in stem or stem in target_stem:
                                    candidates.append(str(f.relative_to(root)))
                    except (OSError, PermissionError):
                        continue

            # Dedup (gleiche Datei kann unter mehreren Roots auftauchen)
            seen: set[str] = set()
            unique: list[str] = []
            for c in candidates:
                if c not in seen:
                    seen.add(c)
                    unique.append(c)

            if unique:
                hint = f"\nDid you mean: {', '.join(unique[:5])}"
            # CWD-Hinweis: zeigt dem Modell, gegen welches Root der Pfad
            # aufgelöst wurde — hilft bei CWD-Mixups.
            roots_hint = "\n".join(f"  - {r}" for r in self._effective_roots)
            return ToolResult(
                ok=False, output="",
                error=(
                    f"File not found: {rel}{hint}\n"
                    f"Path resolved against roots:\n{roots_hint}\n"
                    f"Hint: Use list_dir('.') to see the structure, "
                    f"or adjust the path (relative to CWD or memory root)."
                ),
            )

        size = path.stat().st_size
        if size > _MAX_FILE_SIZE:
            return ToolResult(
                ok=False, output="",
                error=f"File too large ({size} bytes), limit: {_MAX_FILE_SIZE}.",
            )

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return ToolResult(ok=False, output="", error=str(e))

        if alts:
            alt_hint = "\n[Ambiguity] Path also exists at: " + ", ".join(alts)
        else:
            alt_hint = ""
        return ToolResult(ok=True, output=content + alt_hint)


class ListDirTool(Tool):
    name = "list_dir"
    description = "Lists files/directories in a directory (non-recursive)."
    args_schema = {"path": "str (optional, default '.') — relative directory path"}

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

    def _safe(self, rel: str) -> tuple[Path, list[str]]:
        # prefer_existing_dir: list_dir("wiki") muss das Memory-Root
        # zeigen, nicht ein (leeres) wiki/ im CWD.
        result = resolve_in_roots(
            rel, self._effective_roots, prefer_existing_dir=True, alternatives=True
        )
        path, alts = result  # type: ignore[misc]
        return path, alts

    async def run(self, **kwargs: Any) -> ToolResult:
        rel = kwargs.get("path", ".")
        try:
            d, alts = self._safe(rel)
        except (ValueError, OSError) as e:
            return ToolResult(ok=False, output="", error=str(e))

        if not d.is_dir():
            # Hilfreicher Hinweis: Zeige das CWD und Memory-Roots, damit
            # das Modell den Pfad korrigieren kann.
            roots_hint = "\n".join(f"  - {r}" for r in self._effective_roots)
            return ToolResult(
                ok=False, output="",
                error=(
                    f"Not a directory: {rel}\n"
                    f"Path resolved against roots:\n{roots_hint}\n"
                    f"Hint: Use list_dir('.') to see the structure."
                ),
            )

        entries = []
        for entry in sorted(d.iterdir(), key=lambda x: (x.is_file(), x.name)):
            try:
                if entry.is_dir():
                    entries.append(f"  {entry.name}/")
                else:
                    size = entry.stat().st_size
                    entries.append(f"  {entry.name}  ({size} B)")
            except PermissionError:
                entries.append(f"  {entry.name}  [no permission]")

        if alts:
            alt_hint = "\n[Ambiguity] Path also exists at: " + ", ".join(alts)
        else:
            alt_hint = ""
        return ToolResult(ok=True, output=("\n".join(entries) or "(empty)") + alt_hint)


class WriteFileTool(Tool):
    name = "write_file"
    description = "Writes/overwrites a text file (sandbox only)."
    args_schema = {
        "path": "str — relative target path",
        "content": "str — file content (full text)",
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
        # prefer_existing_dir: Schreibaktionen landen im Root, unter dem
        # das oberste Verzeichnis (z. B. wiki/, raw/) schon existiert —
        # sonst würde wiki/... ins Projekt-CWD geschrieben (Sandbox-Root
        # 1 gewinnt blind).
        return resolve_in_roots(rel, self._effective_roots, prefer_existing_dir=True)

    async def run(self, **kwargs: Any) -> ToolResult:
        rel = kwargs.get("path", "")
        content = kwargs.get("content", "")
        if not rel:
            return ToolResult(ok=False, output="", error="No 'path' specified.")
        try:
            path = self._safe(rel)
        except (ValueError, OSError) as e:
            return ToolResult(ok=False, output="", error=str(e))

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except Exception as e:
            return ToolResult(ok=False, output="", error=str(e))

        return ToolResult(ok=True, output=f"✓ {path} ({len(content)} chars) written.")


class SearchRegexTool(Tool):
    name = "search"
    description = "Searches a regex in files under a directory (recursive, max 50 matches)."
    args_schema = {
        "pattern": "str — regex pattern (Python syntax)",
        "path": "str (optional, default '.') — start directory (relative)",
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
        return resolve_in_roots(rel, self._effective_roots, prefer_existing=True)

    async def run(self, **kwargs: Any) -> ToolResult:
        pattern = kwargs.get("pattern", "")
        rel = kwargs.get("path", ".")
        if not pattern:
            return ToolResult(ok=False, output="", error="No 'pattern' specified.")
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return ToolResult(ok=False, output="", error=f"Regex error: {e}")

        try:
            root = self._safe(rel)
        except (ValueError, OSError) as e:
            return ToolResult(ok=False, output="", error=str(e))

        if not root.is_dir():
            return ToolResult(ok=False, output="", error=f"Not a directory: {rel}")

        matches: list[str] = []
        skipped: list[str] = []
        _SKIP_DIRS = {".venv", ".git", "node_modules", "__pycache__", ".hg", ".svn"}
        for fpath in sorted(root.rglob("*")):
            # Verzeichnisse wie .venv/.git/node_modules aussortieren,
            # sonst crawlt rglob tausende irrelevante Dateien.
            # (Zählen NICHT als übersprungene Dateien — sie sind
            # Ausschluss-Verzeichnisse, keine übersprungenen Dateien.)
            if any(part in _SKIP_DIRS for part in fpath.relative_to(root).parts):
                continue
            if fpath.is_file():
                # Größenlimit wie bei ReadFileTool — sonst OOM bei großen Dateien
                try:
                    if fpath.stat().st_size > _MAX_FILE_SIZE:
                        skipped.append(str(fpath.relative_to(root)))
                        continue
                except OSError:
                    continue
                try:
                    text = fpath.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                for i, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        matches.append(f"{fpath}:{i}:{line.strip()[:200]}")
                        if len(matches) >= 50:
                            break
            if len(matches) >= 50:
                break

        if skipped:
            skipped_note = f"\n[Skipped: {len(skipped)} files (larger than {_MAX_FILE_SIZE} B or in excluded directories)]"
        else:
            skipped_note = ""
        output = ("\n".join(matches) if matches else "No matches.") + skipped_note
        return ToolResult(ok=True, output=output)
