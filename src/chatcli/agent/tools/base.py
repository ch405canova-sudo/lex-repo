"""Tool-Basisklasse, ToolRegistry und Factory."""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

# Zentrale Regex für Exit-Code-Extraktion aus Tool-Output.
# Verhindert Duplikate in repl.py / loop.py und garantiert, dass
# negative Codes (-1) überall korrekt gematcht werden.
EXIT_CODE_RE = re.compile(r"\[exit_code: (-?\d+)\]")


def _under(path: Path, root: Path) -> bool:
    """True, wenn ``path`` unterhalb von ``root`` liegt."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_in_roots(
    rel: str,
    roots: list[Path],
    prefer_existing: bool = False,
    prefer_existing_dir: bool = False,
    alternatives: bool = False,
) -> tuple[Path, list[str]] | Path:
    """Löst einen relativen Pfad gegen MEHRERE Sandbox-Roots auf.

    Die Sandbox erlaubt mehrere Basisverzeichnisse (z. B. das Projekt-CWD
    UND das Memory-/Wiki-Verzeichnis). ``rel`` wird gegen jeden Root
    geprüft; der erste Root, unter dem der Pfad liegt, gewinnt.

    ``prefer_existing=True`` (nur für Lese-Tools): wenn der Pfad unter
    mehreren Roots liegen KÖNNTE, wird der Root bevorzugt, unter dem die
    Datei tatsächlich existiert — so findet ``read_file`` eine Wiki-Datei,
    auch wenn ein gleichnamiger Pfad im CWD existieren könnte.

    ``prefer_existing_dir=True`` (für Schreib-Tools): der Root gewinnt,
    unter dem das oberste Verzeichnis des Pfads bereits existiert (z. B.
    ``wiki/`` oder ``raw/`` nur im Memory-Root). So landen Wiki-/Raw-
    Schreibaktionen im Memory-Root, auch wenn das CWD denselben Pfad
    aufnehmen könnte.

    ``alternatives=True``: liefert ``(path, alternatives)`` — die Liste
    enthält alle ANDEREN Roots, unter denen derselbe relative Pfad AUCH
    existiert (z. B. ``wiki/index.md`` sowohl im CWD als auch im Memory-Root).
    Die Tools geben diese Liste als Ambiguitäts-Warnung an das Modell aus.

    Wirft ``ValueError``, wenn der Pfad unter keinem Root liegt
    (Sandbox-Verletzung).
    """
    resolved = _resolve_path(rel, roots, prefer_existing, prefer_existing_dir)
    if not alternatives:
        return resolved
    alts: list[str] = []
    if len(roots) > 1 and rel not in (".", ""):
        for root in roots:
            cand = (root / rel).resolve()
            if cand != resolved and cand.exists() and _under(cand, root):
                alts.append(str(cand))
    return resolved, alts


def _resolve_path(
    rel: str,
    roots: list[Path],
    prefer_existing: bool = False,
    prefer_existing_dir: bool = False,
) -> Path:
    """Core of path resolution (without ambiguity tracking)."""
    # Absolute paths: only allowed if they actually lie under a
    # sandbox root (the prompt uses absolute memory paths like
    # /home/user/Ai/lex/memory/wiki/...). Previously lstrip("/") was
    # applied blindly -> "/home/..." became "home/..." relative to
    # the CWD -> "file not found" despite a correct prompt.
    if rel.startswith("/"):
        abs_path = Path(rel).resolve()
        for root in roots:
            if _under(abs_path, root):
                return abs_path
        raise ValueError(
            f"'{rel}' is outside the allowed directories (sandbox)."
        )
    if not rel:
        raise ValueError("leerer Pfad")
    if prefer_existing:
        for root in roots:
            cand = (root / rel).resolve()
            if cand.is_file() and _under(cand, root):
                return cand
    if prefer_existing_dir:
        first = rel.split("/", 1)[0]
        for root in roots:
            if (root / first).is_dir():
                cand = (root / rel).resolve()
                if _under(cand, root):
                    return cand
    for root in roots:
        cand = (root / rel).resolve()
        if _under(cand, root):
            return cand
    raise ValueError(
        f"'{rel}' is outside the allowed directories (sandbox)."
    )

@dataclass
class ToolResult:
    """Ergebnis eines Tool-Aufrufs."""
    ok: bool
    output: str
    error: str = ""


class Tool(abc.ABC):
    """Abstraktes Tool, das vom Agent aufgerufen werden kann."""

    name: str = ""
    description: str = ""
    # WICHTIG: Tool ist KEIN Dataclass — ein dataclasses.field() hier würde
    # eine Field-Instanz (kein dict) als Klassenattribut erzeugen und
    # schema()/openai_schema() für Tools ohne eigenes args_schema brechen.
    args_schema: dict[str, Any] = {}

    @abc.abstractmethod
    async def run(self, **kwargs: Any) -> ToolResult:
        """Führe das Tool aus und liefere ein ToolResult."""
        ...

    def schema(self) -> dict[str, Any]:
        """Liefere ein JSON-serialisierbares Schema für das Tool."""
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.args_schema or {},
        }

    def openai_schema(self) -> dict[str, Any]:
        """OpenAI-kompatible Function-Definition (native Tool-Calls).

        Der Typ wird aus dem Typpräfix der args_schema-Beschreibung abgeleitet
        (``str`` → string, ``int`` → integer, ``list[...]`` → array); Argumente
        mit ``optional`` in der Beschreibung landen nicht in ``required``.
        """
        properties: dict[str, Any] = {}
        required: list[str] = []
        for arg_name, arg_desc in (self.args_schema or {}).items():
            desc = str(arg_desc)
            token = desc.split(" ", 1)[0] if desc else "str"
            if token.startswith("int"):
                type_: str = "integer"
            elif token.startswith("list"):
                type_ = "array"
            else:
                type_ = "string"
            prop: dict[str, Any] = {"type": type_}
            if type_ == "array" and token.startswith("list[str]"):
                prop["items"] = {"type": "string"}
            short = desc.split("—", 1)[-1].strip() if "—" in desc else ""
            prop["description"] = short or desc
            properties[arg_name] = prop
            if "optional" not in desc:
                required.append(arg_name)
        params: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            params["required"] = required
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }


class ToolRegistry:
    """Zentrale Sammlung verfügbarer Tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        """Liste aller registrierten Tool-Namen."""
        return list(self._tools.keys())

    def descriptions(self) -> str:
        """Kurzbeschreibung aller Tools (für den System-Prompt).

        Das JSON-Format und die Protokollregeln stehen bereits im
        system_prompt (config.py) – hier wird nur die Tool-Liste geliefert.
        """
        if not self._tools:
            return ""
        lines = ["Available tools:"]
        for t in self._tools.values():
            lines.append(f"  - {t.name}: {t.description}")
            if t.args_schema:
                for arg_name, arg_info in t.args_schema.items():
                    lines.append(f"      * {arg_name}: {arg_info}")
        return "\n".join(lines)

    def openai_schemas(self) -> list[dict[str, Any]]:
        """OpenAI-Tool-Definitionen aller registrierten Tools (Payload für
        ``chat``/``chat_stream`` mit ``tools=...``)."""
        return [t.openai_schema() for t in self._tools.values()]

    async def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Ruft ein Tool asynchron auf (nur für Tests, sonst tool.run() nutzen)."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, output="", error=f"Unknown tool: '{name}'")
        return await tool.run(**args)

    def set_live_cwd(self, cwd: str) -> None:
        """Setzt das Live-CWD auf alle Tools, die es unterstützen.

        Wird nach jeder Shell-CWD-Änderung aufgerufen, damit File-Tools
        relative Pfade gegen das aktuelle Arbeitsverzeichnis auflösen
        (statt nur gegen den statischen Startup-Root).
        """
        for tool in self._tools.values():
            if hasattr(tool, "set_live_cwd"):
                tool.set_live_cwd(cwd)


def build_default_registry(
    enable_shell: bool = True,
    skill_dir: str | None = None,
    enable_file_ops: bool = True,
    enable_planner: bool = True,
    enable_scrape: bool = True,
    enable_web_search: bool = True,
    enable_semantic_search: bool = True,
    enable_script: bool = True,
    enable_self_improve: bool = True,
    enable_system: bool = True,
    enable_code_analysis: bool = True,
    shell_cwd: str = ".",
    wiki_dir: str | None = None,
    sudo_ask: "Callable[[str], Optional[str]] | None" = None,
    confirm_ask: "Callable[[str, str], Any] | None" = None,
    shell_on_line: "Callable[[str], None] | None" = None,
) -> ToolRegistry:
    """Baut die Standard-Tool-Registry basierend auf der Config.

    Wichtig: Die Imports sind hier bewusst lokal (lazy), um zirkuläre
    Imports zwischen base.py und den Implementierungsmodulen zu vermeiden.
    """
    registry = ToolRegistry()

    if enable_shell:
        from .shell import ShellTool
        registry.register(
            ShellTool(
                cwd=shell_cwd,
                sudo_ask=sudo_ask,
                confirm_ask=confirm_ask,
                on_line=shell_on_line,
            )
        )

    if enable_script:
        from .script import ScriptTool
        registry.register(ScriptTool(cwd=shell_cwd))

    if enable_scrape:
        from .scrape import ScrapeTool
        registry.register(ScrapeTool(wiki_dir=wiki_dir))

    if enable_web_search:
        from .search import WebSearchTool
        registry.register(WebSearchTool())

    if enable_semantic_search:
        from .semantic_search import SemanticSearchTool
        registry.register(SemanticSearchTool())

    if enable_file_ops:
        from .file_ops import ReadFileTool, ListDirTool, WriteFileTool, SearchRegexTool
        from .apply_diff import ApplyDiffTool
        base = shell_cwd or "."
        # Die Datei-Tools dürfen AUF zwei Sandbox-Roots zugreifen:
        # das Projekt-CWD und das Memory-/Wiki-Verzeichnis (raw/ + wiki/).
        # So kann Lex Lessons/Wiki-Seiten direkt mit write_file/apply_diff
        # schreiben, ohne über Shell+Python-Heredyoc zu laufen.
        # Zwei extra Roots: Memory-Root (raw/+wiki/) UND Skill-Verzeichnis
        # (SKILL.md, references/, scripts/) — SKILL.md lag vorher außerhalb
        # der Sandbox, Lex konnte es nur per Shell caten (Steps verbrannt).
        extra_roots = []
        if wiki_dir:
            extra_roots.append(Path(wiki_dir).expanduser().resolve())
        # Git-Repo-Root als zusätzlicher Root: ermöglicht Pfade wie
        # "lex/chatcli/src/..." relativ zur Repo-Root (nicht nur zum CWD).
        # Erkennt den Git-Root via .git-Verzeichnis-Suche nach oben.
        git_root = Path(base).resolve()
        while git_root != git_root.parent:
            if (git_root / ".git").exists():
                break
            git_root = git_root.parent
        if git_root not in extra_roots and git_root != Path(base).resolve():
            extra_roots.append(git_root)
        if skill_dir:
            sk = Path(skill_dir).expanduser().resolve()
            if sk.is_dir():
                extra_roots.append(sk)
        elif wiki_dir:
            sk = Path(wiki_dir).expanduser().resolve().parent / "wiki-skill"
            if sk.is_dir():
                extra_roots.append(sk)
        registry.register(ReadFileTool(base_dir=base, extra_roots=extra_roots))
        registry.register(ListDirTool(base_dir=base, extra_roots=extra_roots))
        registry.register(WriteFileTool(base_dir=base, extra_roots=extra_roots))
        registry.register(SearchRegexTool(base_dir=base, extra_roots=extra_roots))
        registry.register(ApplyDiffTool(base_dir=base, extra_roots=extra_roots))
        from .patch import PatchLineTool
        registry.register(PatchLineTool(base_dir=base, extra_roots=extra_roots))

    if enable_planner:
        from .planner import PlannerTool
        registry.register(PlannerTool())

    if enable_system:
        from .system import SystemTool
        from .jobs import JobsTool
        registry.register(SystemTool())
        registry.register(JobsTool(on_line=shell_on_line))

    if enable_self_improve:
        from .self_improve import SelfImproveTool
        registry.register(
            SelfImproveTool(
                wiki_dir=wiki_dir or "", shell_cwd=shell_cwd or "."
            )
        )

    if enable_code_analysis:
        from .code_analysis import CodeAnalysisTool
        # Projekt-Root = das Verzeichnis, in dem pyproject.toml liegt.
        # shell_cwd ist relativ; wir brauchen den absoluten Pfad zum chatcli-Paket.
        project_root = Path(shell_cwd or ".").resolve()
        # Fallback: wenn CWD nicht der chatcli-Root ist, nach pyproject.toml suchen.
        if not (project_root / "pyproject.toml").exists():
            p = project_root
            for _ in range(5):
                if (p / "pyproject.toml").exists():
                    break
                if p.parent == p:
                    break
                p = p.parent
            project_root = p
        registry.register(CodeAnalysisTool(project_root=str(project_root)))

    return registry
