"""Rich output helpers."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
console = Console()


def banner() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]lex[/] — Terminal AI Agent",
            subtitle="local LLM via llama-server",
        )
    )


def user_msg(text: str) -> None:
    console.print(f"[bold blue]You:[/] [white]{text}[/]")


def error_msg(text: str) -> None:
    console.print(f"[bold red]Error:[/] {text}")


def info_msg(text: str) -> None:
    console.print(f"[dim]{text}[/]")


def slash_help() -> None:
    console.print(
        Panel(
            "\n".join([
                "[bold]/help[/]        — show this help",
                "[bold]/clear[/]       — clear chat history",
                "[bold]/tools[/]       — list available tools",
                "[bold]/history[/]      — show message history",
                "[bold]/save[/] <path>  — save history to file (JSON)",
                "[bold]/system[/] <text>— set system prompt (loop restart required)",
                "[bold]/tokens[/]       — show token counter",
                "[bold]/context[/]      — show server context usage",
                "[bold]/learn[/] <topic>— research a topic on the web and store it in the wiki",
                "[bold]/ingest[/] <text> — save text/URL into the wiki (Ingest)",
                "[bold]/exit[/]         — exit",
            ])
        )
    )
