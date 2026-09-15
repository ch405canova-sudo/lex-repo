"""chatcli — entry point.

Usage:
  python -m chatcli                 # interactive mode
  python -m chatcli --prompt "..."  # single-shot (for pipes)
  python -m chatcli --help          # CLI help
"""

from __future__ import annotations

import argparse
import asyncio
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="chatcli",
        description="chatcli — terminal chat CLI & agent for llama.cpp",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name (default: local)",
    )
    parser.add_argument(
        "--host", default=None,
        help="llama-server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="llama-server port (default: 8080)",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Max response tokens (default: 4096)",
    )
    parser.add_argument(
        "--system", default=None,
        help="System prompt (overrides config)",
    )
    parser.add_argument(
        "--prompt", "-p", default=None,
        help="Single-shot mode: ask a question and exit (no REPL)",
    )
    parser.add_argument(
        "--non-interactive", action="store_true",
        help="With --prompt: no history, stdout only",
    )
    parser.add_argument(
        "--no-shell", action="store_true",
        help="Disable the shell tool",
    )
    parser.add_argument(
        "--no-files", action="store_true",
        help="Disable file tools",
    )
    parser.add_argument(
        "--no-scrape", action="store_true",
        help="Disable the scrape tool (trafilatura)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Max agent steps (default: 8)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Logging einrichten
    import logging
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    # Config laden
    from chatcli.config import Config
    config = Config.load()

    # CLI-Args overrideen
    if args.host:
        config.host = args.host
    if args.port is not None:
        config.port = args.port
    if args.model:
        config.model = args.model
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.max_tokens is not None:
        config.max_tokens = args.max_tokens
    if args.system:
        config.system_prompt = args.system
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.prompt:
        config.prompt_text = args.prompt
        config.non_interactive = True
    if args.non_interactive:
        config.non_interactive = True
    if args.no_shell:
        config.enable_shell = False
    if args.no_files:
        config.enable_file_ops = False
    if args.no_scrape:
        config.enable_scrape = False

    # REPL starten
    from chatcli.repl import ChatCLI

    cli = ChatCLI(config)
    try:
        asyncio.run(cli.start())
    except KeyboardInterrupt:
        print("\n[Interrupted]")
        sys.exit(0)
    except BrokenPipeError:
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
