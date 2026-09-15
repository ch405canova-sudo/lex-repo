"""Tool-Subsystem: Basisklassen, Registry und konkrete Tools.

Public API of the package — all names here are the only ones
that consumers (loop, repl, agent) need to import.
"""

from .base import Tool, ToolResult, ToolRegistry, build_default_registry

__all__ = [
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "build_default_registry",
]
