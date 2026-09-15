#!/usr/bin/env python3
"""Launcher for the Lex terminal agent. Runs without venv / pip install.

Usage:
    python3 lex.py                        # interactive chat REPL
    python3 lex.py --prompt "..."        # single-shot (for pipes)
    python3 lex.py scrape <url>           # load a web page as Markdown
    python3 lex.py --help                # CLI help

How it works:
    - Adds the package directory (src/) directly to sys.path, so
      `import chatcli` works without installation.
    - All CLI arguments are forwarded to the real CLI (chatcli.__main__.main).
    - If prompt_toolkit is missing, a simple input() fallback is used.
    - `lex.py scrape <url>` calls the trafilatura CLI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add the package directory to sys.path (no pip install needed).
SRC = Path(__file__).resolve().parent / "src"
if not SRC.exists():
    sys.exit(f"Error: package directory not found: {SRC}")
sys.path.insert(0, str(SRC))

# Set CWD to the project root so shell tools and self_improve
# always resolve relative to the repo root.
_PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(_PROJECT_ROOT)

from chatcli.__main__ import main  # noqa: E402


def _scrape(url: str, max_chars: int = 0) -> int:
    """Load a URL and print the extracted Markdown text.

    Uses the trafilatura CLI (uv tool install trafilatura).
    max_chars=0 means no truncation.
    """
    import subprocess
    from pathlib import Path

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    candidates = [
        "trafilatura",
        str(Path.home() / ".local" / "bin" / "trafilatura"),
        str(Path.home() / ".local" / "share" / "uv" / "tools" / "trafilatura" / "bin" / "trafilatura"),
    ]
    import shutil as _sh
    binary = next((c for c in candidates if _sh.which(c)), None)
    if binary is None:
        sys.exit(
            "Error: trafilatura not found.\n"
            "  → Install (without venv):  uv tool install trafilatura"
        )

    cmd = [binary, "-u", url, "--output-format", "markdown"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        sys.exit(f"Error: timeout after 60s loading {url}")

    if proc.returncode != 0 or not proc.stdout.strip():
        sys.exit(f"Scrape error ({url}):\n{proc.stderr.strip() or proc.stdout.strip()}")

    text = proc.stdout.strip()
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars] + f"\n[… truncated to {max_chars} chars]"
    print(text)
    return 0


if __name__ == "__main__":
    # lex.py scrape <url> [max_chars] → direct trafilatura, no LLM needed
    if len(sys.argv) >= 2 and sys.argv[1] == "scrape":
        if len(sys.argv) < 3:
            sys.exit("Usage: lex.py scrape <url> [max_chars]")
        _scrape(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
        sys.exit(0)

    main()
