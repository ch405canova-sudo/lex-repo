"""Scrape tool: load a webpage and extract text content.

Two-step process:
1. HTML is loaded via urllib (browser user-agent, follows redirects).
2. Extraction runs through the trafilatura Python API (``trafilatura.extract``),
   since the trafilatura CLI in the current version (2.2.0) reliably produces
   "empty HTML tree" / empty output with ``-u`` and ``-i``, even though the
   page loads fine via curl.

The trafilatura Python API is reached through the uv tool isolation
(``uv tool install trafilatura``): The tool's Python interpreter is located
(uv default path or shebang of the ``trafilatura`` binary) and fed with a
small script that receives HTML on stdin and outputs Markdown on stdout.

Fallback (only when trafilatura yields too little): Crawl4AI with headless
Chrome (``uv tool install crawl4ai``). This renders JavaScript-heavy pages/SPAs
that trafilatura returns empty. Crawl4AI is slower (browser), so it's only
the fallback path.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

from .base import Tool, ToolResult

# Max characters returned to context (context protection).
DEFAULT_MAX_CHARS = 12000

# For save_to_raw: max characters that still go back to context.
# The source is in raw/ — the agent doesn't need to read it again
# to "remember" it.
_RAW_ECHO_MAX = 2000

# Browser user-agent: Some servers deliver different content to bots
# (e.g. "trafilatura/x.y.z") or block them; a browser UA is the most
# reliable default for static pages.
_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# Mini-script for the trafilatura subprocess: HTML from stdin, Markdown to stdout.
_EXTRACT_SCRIPT = (
    "import sys, trafilatura;\n"
    "html = sys.stdin.buffer.read().decode('utf-8', 'replace');\n"
    "text = trafilatura.extract(html, output_format='markdown');\n"
    "sys.stdout.buffer.write((text or '').encode('utf-8'))\n"
)


def _find_trafilatura_python() -> str | None:
    """Finds the Python interpreter of the trafilatura uv tool installation.

    Order:
    1. uv default path ``~/.local/share/uv/tools/trafilatura/bin/python*``
    2. Shebang of the ``trafilatura`` binary (``~/.local/bin/trafilatura``)
    """
    # 1) uv default path
    uv_dir = Path.home() / ".local" / "share" / "uv" / "tools" / "trafilatura" / "bin"
    if uv_dir.is_dir():
        for candidate in sorted(uv_dir.glob("python*")):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)

    # 2) Shebang of the CLI binary
    binary = shutil.which("trafilatura")
    if binary:
        try:
            first_line = Path(binary).read_text(encoding="utf-8", errors="replace").splitlines()[0]
        except (OSError, IndexError):
            first_line = ""
        if first_line.startswith("#!") and "python" in first_line:
            parts = first_line[2:].strip().split()
            if parts:
                candidate = parts[0]
                if candidate and Path(candidate).is_file():
                    return candidate
    return None


def _download(url: str, timeout: float) -> tuple[bytes, str]:
    """Loads the URL and returns (bytes, content-type). Redirects are followed."""
    req = urllib.request.Request(url, headers={"User-Agent": _BROWSER_UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        ctype = resp.headers.get("Content-Type", "")
        return resp.read(), ctype


def _slugify(text: str, max_len: int = 60) -> str:
    """Builds a kebab-case slug from title/URL (max max_len chars)."""
    text = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    text = re.sub(r"[\s_]+", "-", text).strip("-")
    return text[:max_len].strip("-")


# Mini-script for the Crawl4AI subprocess: URL on stdin, Markdown to stdout.
# Uses system Chrome (headless) — no separate Playwright browser needed.
_CRAWL4AI_SCRIPT = (
    "import sys, asyncio\n"
    "from crawl4ai import AsyncWebCrawler\n"
    "from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy\n"
    "from crawl4ai.async_configs import BrowserConfig\n"
    "async def main():\n"
    "    url = sys.stdin.buffer.read().decode('utf-8').strip()\n"
    "    cfg = BrowserConfig(headless=True, browser_type='chromium', chrome_channel='chrome')\n"
    "    strategy = AsyncPlaywrightCrawlerStrategy(browser_config=cfg)\n"
    "    crawler = AsyncWebCrawler(crawler_strategy=strategy)\n"
    "    result = await crawler.arun(url=url)\n"
    "    md = (result.markdown or '').strip() if result.success else ''\n"
    "    sys.stdout.buffer.write(md.encode('utf-8'))\n"
    "    await crawler.close()\n"
    "asyncio.run(main())\n"
)


def _find_crawl4ai_python() -> str | None:
    """Finds the Python interpreter of the crawl4ai uv tool installation.

    Order:
    1. uv default path ``~/.local/share/uv/tools/crawl4ai/bin/python*``
    2. Shebang of the ``crwl`` binary (``~/.local/bin/crwl``)
    """
    uv_dir = Path.home() / ".local" / "share" / "uv" / "tools" / "crawl4ai" / "bin"
    if uv_dir.is_dir():
        for candidate in sorted(uv_dir.glob("python*")):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)

    binary = shutil.which("crwl")
    if binary:
        try:
            first_line = Path(binary).read_text(encoding="utf-8", errors="replace").splitlines()[0]
        except (OSError, IndexError):
            first_line = ""
        if first_line.startswith("#!") and "python" in first_line:
            parts = first_line[2:].strip().split()
            if parts:
                candidate = parts[0]
                if candidate and Path(candidate).is_file():
                    return candidate
    return None


async def _crawl4ai_extract(url: str, timeout: float) -> str:
    """Renders the URL with Crawl4AI (headless Chrome) and returns Markdown.

    Only as fallback when trafilatura yields too little (JS-heavy pages).
    """
    python = _find_crawl4ai_python()
    if not python:
        raise FileNotFoundError("crawl4ai uv tool not installed")

    proc = await asyncio.create_subprocess_exec(
        python, "-c", _CRAWL4AI_SCRIPT,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b = await asyncio.wait_for(
        proc.communicate(input=url.encode("utf-8")), timeout=timeout
    )
    return stdout_b.decode(errors="replace").strip()


class ScrapeTool(Tool):
    name = "scrape"
    description = (
        "Loads a URL and extracts the main text content as Markdown "
        "(trafilatura). With save_to_raw=true the tool writes the source "
        "DIRECTLY to raw/<topic>/YYYY-MM-DD-slug.md (with metadata header) "
        "and returns only a short echo — saves context and steps "
        "during ingest.\n"
        "Examples:\n"
        "  scrape(url='https://example.com/doc')\n"
        "  scrape(url='https://example.com/doc', save_to_raw=true, topic='scraping')"
    )
    args_schema: dict[str, Any] = {
        "url": "str — the URL to load (http/https)",
        "max_chars": "int (optional, default 12000) — max characters of output",
        "timeout": "int (optional, default 30) — timeout in seconds",
        "save_to_raw": "bool (optional, default false) — write source directly to raw/ (ingest)",
        "topic": "str (optional) — topic directory for save_to_raw (default: derived from URL)",
        "slug": "str (optional) — filename without date (kebab-case, default: derived from URL)",
    }

    def __init__(
        self,
        python: str | None = None,
        wiki_dir: str | None = None,
    ) -> None:
        self._python = python or _find_trafilatura_python()
        # wiki_dir: root of the memory (raw/ + wiki/). Without wiki_dir,
        # save_to_raw is disabled (tool stays read-only).
        self._wiki_dir = Path(wiki_dir).expanduser().resolve() if wiki_dir else None

    async def run(self, **kwargs: Any) -> ToolResult:
        url: str = str(kwargs.get("url", "")).strip()
        if not url:
            return ToolResult(ok=False, output="", error="No 'url' specified.")
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        try:
            max_chars: int = int(kwargs.get("max_chars", DEFAULT_MAX_CHARS))
            timeout: int = int(kwargs.get("timeout", 30))
        except (ValueError, TypeError):
            return ToolResult(
                ok=False, output="",
                error="Invalid value for 'max_chars' or 'timeout' (must be a number).",
            )

        # save_to_raw option: write source directly to raw/ (ingest flow).
        save_to_raw = bool(kwargs.get("save_to_raw", False))
        topic = str(kwargs.get("topic", "")).strip() or None
        slug = str(kwargs.get("slug", "")).strip() or None
        if save_to_raw and self._wiki_dir is None:
            return ToolResult(
                ok=False, output="",
                error=(
                    "save_to_raw=true, but the scrape tool was registered without "
                    "wiki_dir — no raw/ known."
                ),
            )

        if self._python is None:
            return ToolResult(
                ok=False, output="",
                error=(
                    "trafilatura Python API not found. Install: "
                    "`uv tool install trafilatura`."
                ),
            )

        # 1) Load HTML (blocking → thread)
        try:
            html_bytes, ctype = await asyncio.wait_for(
                asyncio.to_thread(_download, url, float(timeout)), timeout=timeout
            )
        except asyncio.TimeoutError:
            return ToolResult(
                ok=False, output="",
                error=f"Timeout after {timeout}s while loading {url}.",
            )
        except urllib.error.HTTPError as e:
            return ToolResult(
                ok=False, output="",
                error=f"HTTP {e.code} at {url} (page does not exist or is blocked).",
            )
        except urllib.error.URLError as e:
            return ToolResult(
                ok=False, output="",
                error=f"Network error at {url}: {e.reason}",
            )
        except Exception as e:
            return ToolResult(ok=False, output="", error=str(e))

        # Bail early on non-HTML content (PDFs, JSON, …)
        if ctype and "html" not in ctype.lower() and "text" not in ctype.lower():
            return ToolResult(
                ok=False, output="",
                error=f"Not HTML (Content-Type: {ctype}). Load directly via curl: {url}",
            )

        # 2) Extract text (trafilatura subprocess)
        try:
            proc = await asyncio.create_subprocess_exec(
                self._python, "-c", _EXTRACT_SCRIPT,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(input=html_bytes), timeout=timeout
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await proc.wait()
            except Exception:
                pass
            return ToolResult(
                ok=False, output="",
                error=f"Timeout after {timeout}s during extraction of {url}.",
            )
        except FileNotFoundError:
            return ToolResult(
                ok=False, output="",
                error="trafilatura Python interpreter not found.",
            )
        except Exception as e:
            return ToolResult(ok=False, output="", error=str(e))

        text = stdout_b.decode(errors="replace").strip()
        stderr = stderr_b.decode(errors="replace").strip()

        # Fallback: Crawl4AI (headless Chrome) renders JavaScript-heavy
        # pages that trafilatura returns empty or only with boilerplate.
        # Threshold: < 200 chars = practically empty (JS-SPA, bot block, …).
        if len(text) < 200:
            try:
                text = await asyncio.wait_for(
                    _crawl4ai_extract(url, float(timeout)), timeout=timeout
                )
            except FileNotFoundError:
                pass  # crawl4ai not installed → old error message below
            except (asyncio.TimeoutError, Exception):
                pass  # Crawl4AI failed → old error message below

        if not text:
            hint = ""
            if "html" in (html_bytes[:200].decode(errors="replace").lower()):
                hint = (
                    " (JavaScript-heavy page? Crawl4AI fallback also yielded "
                    "nothing — load __NEXT_DATA__/JSON via curl "
                    "or choose a different source.)"
                )
            detail = stderr[:300] if stderr else ""
            return ToolResult(
                ok=False, output="",
                error=f"No text extractable from {url}{hint}. {detail}",
            )

        # save_to_raw: write source with metadata header directly to raw/.
        # Context gets only a short echo — the file IS the source.
        if save_to_raw:
            return self._save_to_raw(url, text, topic, slug)

        truncated = False
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars]
            truncated = True

        if truncated:
            text += f"\n[… truncated to {max_chars} chars — increase max_chars for more content]"

        return ToolResult(ok=True, output=text)

    # ------------------------------------------------------------------

    def _save_to_raw(self, url: str, text: str, topic: str | None, slug: str | None) -> ToolResult:
        """Writes the extracted source to raw/<topic>/YYYY-MM-DD-slug.md.

        Filename: kebab-case slug from URL path (or explicitly passed),
        date = today. On name conflict a numeric suffix is appended
        (convention from SKILL.md → Fetch).
        """
        from urllib.parse import urlparse

        host = urlparse(url).netloc.split(":")[0] or "source"
        slug = slug or _slugify(urlparse(url).path.strip("/")) or _slugify(host)
        topic = topic or _slugify(host, 40) or "misc"

        today = date.today().isoformat()
        raw_dir = self._wiki_dir / "raw" / topic  # type: ignore[operator]
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Name conflict → numeric suffix (convention: -2, -3, …)
        base = f"{today}-{slug}"
        candidate = raw_dir / f"{base}.md"
        n = 2
        while candidate.exists():
            candidate = raw_dir / f"{base}-{n}.md"
            n += 1

        # Title = first Markdown heading line, otherwise host
        title = host
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        else:
            title = host

        content = (
            f"# {title}\n"
            "\n"
            f"> Source: {url}\n"
            f"> Collected: {today}\n"
            f"> Published: Unknown\n"
            "\n"
            f"{text}\n"
        )
        candidate.write_text(content, encoding="utf-8")

        # Short echo instead of full text: the file IS the context.
        preview = text[:_RAW_ECHO_MAX]
        if len(text) > _RAW_ECHO_MAX:
            preview += f"\n[… {len(text) - _RAW_ECHO_MAX} more chars — full text in the file]"
        return ToolResult(
            ok=True,
            output=(
                f"✓ Source saved: {candidate} ({len(text)} chars extracted)\n"
                f"  Topic: {topic} | Title: {title}\n"
                f"  Next steps: Triage (search wiki) → Compile → "
                f"update index.md + log.md (format: SKILL.md → Ingest).\n"
                f"--- Preview ---\n{preview}"
            ),
        )
