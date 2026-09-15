# Lex

A terminal AI agent for Linux — coded for a local LLM via [llama.cpp](https://github.com/ggml-org/llama.cpp).

## What it does

- Interactive REPL with an autonomous agent loop (plan → act → observe)
- Tools: shell, file ops, search, plan, self-improve, sub-agent, semantic search, code analysis
- Works with any local LLM served via `llama-server` (OpenAI-compatible HTTP API)
- No cloud, no API keys — fully local

## Quick Start

1. Build `llama.cpp` and start a server on port 8080:
   ```bash
   llama-server -m your-model.gguf --port 8080
   ```
2. Install dependencies:
   ```bash
   pip install httpx rich prompt_toolkit pyyaml numpy
   ```
3. Run:
   ```bash
   python3 lex.py
   ```

### Single-shot mode

```bash
python3 lex.py --prompt "What files are in the current directory?"
```

### Options

```bash
python3 lex.py \
  --model local \
  --temperature 0.3 \
  --max-tokens 8192 \
  --system "You are an expert in Linux administration."
```

## Configuration

Priority: **CLI arguments** → **environment variables** → **YAML file** → **defaults**

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHATCLI_HOST` | `127.0.0.1` | llama-server host |
| `CHATCLI_PORT` | `8080` | llama-server port |
| `CHATCLI_MODEL` | `local` | Model name |
| `CHATCLI_MAX_TOKENS` | `98304` | Max response tokens |
| `CHATCLI_TEMPERATURE` | `0.6` | Sampling temperature |
| `CHATCLI_MAX_STEPS` | `100` | Max agent steps per question |
| `CHATCLI_TIMEOUT` | `300` | HTTP timeout (seconds) |
| `CHATCLI_NO_SHELL` | `0` | `1` = disable shell tool |
| `CHATCLI_NO_FILES` | `0` | `1` = disable file tools |

### YAML config file

```yaml
# ~/.config/chatcli/config.yaml
host: 127.0.0.1
port: 8080
model: local
max_tokens: 98304
temperature: 0.6
max_steps: 100
shell_cwd: .
```

## Tools

| Tool | Description |
|------|-------------|
| `shell` | Execute a shell command (stateful, stdout/stderr/exit_code) |
| `read_file` | Read a file (max 1 MB) |
| `write_file` | Write a file (sandboxed) |
| `apply_diff` | Apply search/replace blocks to a file |
| `list_dir` | List directory contents |
| `search` | Regex search in files (recursive, max 50 hits) |
| `semantic_search` | Semantic search by meaning (not text match) |
| `plan` | Create a step-by-step plan |
| `scrape` | Load a web page as Markdown |
| `script` | Execute a code snippet (bash/python) |
| `subagent` | Spawn a sub-agent for a subtask |
| `self_improve` | Self-improvement: lessons + code changes |
| `code_analysis` | Analyze code structure |
| `system` | System state snapshot (load, ram, disk, ports) |
| `jobs` | Manage background jobs |

All file tools run in a sandbox: `..` and absolute paths are blocked.

## Project Structure

```
lex/
├── lex.py                 # Launcher (no venv needed)
├── pyproject.toml         # Package config
├── embed_index.py         # Semantic code/doc index (SQLite + embeddings)
├── src/chatcli/
│   ├── __init__.py
│   ├── __main__.py        # Entry point + CLI
│   ├── config.py          # Configuration (env + YAML)
│   ├── llama_client.py    # HTTP client for llama-server
│   ├── memory.py          # Memory / lessons injection
│   ├── output.py          # Rich terminal output
│   ├── repl.py            # Interactive REPL
│   └── agent/
│       ├── __init__.py
│       ├── loop.py        # Agent loop (plan → act → observe)
│       ├── parser.py      # Tool-call parsing
│       └── tools/
│           ├── base.py    # Tool base class & registry
│           ├── shell.py   # Shell commands
│           ├── file_ops.py# File read/write/search
│           ├── apply_diff.py
│           ├── patch.py
│           ├── planner.py
│           ├── search.py
│           ├── semantic_search.py
│           ├── scrape.py
│           ├── script.py
│           ├── system.py
│           ├── jobs.py
│           ├── subagent.py
│           ├── self_improve.py
│           └── code_analysis.py
├── tests/                 # Test suite
└── .github/workflows/     # CI (pytest on push)
```

## Tests

```bash
pytest tests/ -q
```

## License

[MIT](LICENSE)
