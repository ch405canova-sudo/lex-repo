"""Web search tool: search results via the local SearXNG metasearch server.

Only backend: SearXNG (JSON API, port 8888, self-hosted, no API key,
stable JSON response).
Dependency-free (urllib + json), so no venv/pip needed.
Result URLs can then be ingested into the wiki via the scrape tool.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.parse
import urllib.request
from typing import Any

from .base import Tool, ToolResult

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# SearXNG instance (local). Overridable via SEARXNG_URL.
_SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://127.0.0.1:8888")


def _searxng_search(query: str, max_results: int, timeout: float) -> list[tuple[str, str, str]]:
    """Queries the local SearXNG JSON API and returns (title, URL, snippet)."""
    url = (
        _SEARXNG_URL
        + "/search?q="
        + urllib.parse.quote_plus(query)
        + "&format=json"
        + "&language=auto&pageno=1"
    )
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="replace"))

    results: list[tuple[str, str, str]] = []
    for item in data.get("results", [])[:max_results]:
        title = str(item.get("title", "")).strip()
        target = str(item.get("url", "")).strip()
        snippet = str(item.get("content", "")).strip()
        if not target.startswith("http"):
            continue
        results.append((title, target, snippet))
    return results


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Web search via the local SearXNG server. Returns title, URL and "
        "snippet of top results. Load good results with the scrape tool."
    )
    args_schema: dict[str, Any] = {
        "query": "str — search query",
        "max_results": "int (optional, default 5) — number of results",
    }

    async def run(self, **kwargs: Any) -> ToolResult:
        query: str = str(kwargs.get("query", "")).strip()
        max_results: int = int(kwargs.get("max_results", 5))

        if not query:
            return ToolResult(ok=False, output="", error="No 'query' specified.")

        try:
            results = await asyncio.to_thread(_searxng_search, query, max_results, 25.0)
        except Exception as e:
            return ToolResult(
                ok=False, output="",
                error=f"SearXNG search failed: {e} "
                      f"(Is the server running? URL: {_SEARXNG_URL})",
            )

        if not results:
            return ToolResult(
                ok=False, output="",
                error="No results (SearXNG empty). "
                      "Try a different search term or load the URL directly with scrape.",
            )

        lines = []
        for i, (title, url, snippet) in enumerate(results, 1):
            line = f"{i}. {title}\n   {url}"
            if snippet:
                line += f"\n   {snippet}"
            lines.append(line)

        return ToolResult(ok=True, output="\n".join(lines))
