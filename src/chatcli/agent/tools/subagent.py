"""Sub-agent isolation (Phase 3): delegates subtasks to an isolated
AgentLoop with fresh context and reduced tools.

Benefits:
- "Large research" or "check 10 files" without burdening the main context
  (context overflow protection).
- The sub-agent works in its own context; only the condensed summary
  comes back to the parent loop.

Design:
- The tool starts a new AgentLoop with:
  - Fresh history (only system prompt + task)
  - Reduced tools (shell, read_file, write_file, search_regex, semantic_search)
  - Limited max_steps (config.subagent_max_steps, default 10)
- The sub-agent's final answer is returned as a ToolResult.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .base import Tool, ToolResult

log = logging.getLogger(__name__)

# Tools the sub-agent does NOT get (avoiding recursion
# and context overload):
_EXCLUDED_TOOLS: frozenset[str] = frozenset({
    "subagent",       # no recursion
    "plan",           # planner belongs to parent
    "self_improve",   # parent only
    "scrape",         # too context-heavy for sub-agent
    "web_search",     # same
})


class SubAgentTool(Tool):
    """Starts an isolated sub-agent for subtasks.

    Arguments:
        task (str): The specific subtask the sub-agent should execute.
                    Should be formulated precisely and self-contained.
        context (str, optional): Additional context (e.g. relevant paths,
                    prior findings) passed to the sub-agent.
    """

    name = "subagent"
    args_schema: dict[str, Any] = {
        "task": "str — The specific subtask for the sub-agent. Formulated self-contained.",
        "context": "str (optional) — Additional context: relevant paths, prior findings, constraints.",
    }

    def __init__(
        self,
        client: Any,
        config: Any,
        parent_registry: Any,
    ) -> None:
        """
        Args:
            client: LlamaClient (same as in the parent loop).
            config: Config instance.
            parent_registry: The parent loop's ToolRegistry (will be reduced).
        """
        self._client = client
        self._config = config
        self._parent_registry = parent_registry

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                "Delegates a subtask to an isolated sub-agent with fresh context. "
                "The sub-agent only receives core tools "
                "(shell, read_file, write_file, search_regex, semantic_search) "
                "and works in its own context. Use it for large research, "
                "checking multiple files or complex analyses that should not "
                "burden the main context. Provide a precise, self-contained subtask."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "The specific subtask for the sub-agent. "
                            "Formulated self-contained, without reference to "
                            "the main context."
                        ),
                    },
                    "context": {
                        "type": "string",
                        "description": (
                            "Optional additional context: relevant paths, "
                            "prior findings, constraints. Helps the "
                            "sub-agent without knowing the parent context."
                        ),
                    },
                },
                "required": ["task"],
            },
        }

    async def run(self, **kwargs: Any) -> ToolResult:
        task: str = kwargs.get("task", "").strip()
        context: str = kwargs.get("context", "").strip()

        if not task:
            return ToolResult(ok=False, output="", error="task must not be empty.")

        # Max steps from config (default 10)
        max_steps = getattr(self._config, "subagent_max_steps", 10)

        # Build reduced registry: only core tools, without subagent/plan/etc.
        from .base import ToolRegistry
        sub_registry = ToolRegistry()
        for tool_name in self._parent_registry.names():
            if tool_name in _EXCLUDED_TOOLS:
                continue
            tool = self._parent_registry.get(tool_name)
            if tool is not None:
                sub_registry.register(tool)

        # System prompt for the sub-agent (compact, no lessons/style).
        cwd = getattr(self._config, "shell_cwd", ".") or "."
        system_prompt = (
            f"You are a sub-agent executing a subtask. "
            f"Work efficiently and return a condensed summary "
            f"(max. ~200 words) of your results at the end.\n\n"
            f"## Context\nWorking directory: {cwd}\n\n"
            f"## Rules\n"
            f"- Use tools to solve the task.\n"
            f"- At the end, give ONE clear, complete summary as a text answer.\n"
            f"- Keep the answer compact (max. ~200 words).\n"
        )

        # Task + optional context as user message.
        user_msg = f"Task: {task}"
        if context:
            user_msg += f"\n\nAdditional context:\n{context}"

        # Start sub-agent loop (fresh, without parent history).
        try:
            from ..loop import AgentLoop
            sub_loop = AgentLoop(
                client=self._client,
                registry=sub_registry,
                config=self._config,
                on_event=None,  # keine Events an den Parent-UI
            )
            # Override system prompt (compact instead of full prompt).
            sub_loop._history[0]["content"] = system_prompt

            # Execute sub-agent: collect all yielded chunks.
            chunks: list[str] = []
            async for chunk in sub_loop.ask(user_msg):
                chunks.append(chunk)

            result_text = "".join(chunks).strip()

            if not result_text:
                return ToolResult(
                    ok=False, output="",
                    error="Sub-agent returned no answer.",
                )

            # Return condensed summary.
            summary = (
                f"[Sub-Agent Result]\n"
                f"Task: {task}\n"
                f"Summary:\n{result_text}"
            )
            return ToolResult(ok=True, output=summary)

        except Exception as e:
            log.warning("Sub-agent failed: %s", e)
            return ToolResult(
                ok=False, output="",
                error=f"Sub-agent error: {e}",
            )
