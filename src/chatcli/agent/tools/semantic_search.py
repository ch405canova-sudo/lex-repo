"""Semantic search via the local embedding index (embed_index.py).

Complements the regex-based ``search`` tool: finds code/docs by
MEANING instead of text match. Uses the local embedding server
(port 8081) and the SQLite index (lex/.embed_index.db).

The index covers the workspace + memory (raw/ + wiki/). After an
ingest (new raw/ file), incrementally update with ``reindex=true``.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import struct
import urllib.error
import urllib.request
from typing import Any

try:
    import numpy as np
except ImportError:
    # lex.py runs with system Python (no venv) — numpy is missing there.
    # Then the pure-Python fallback (_top_k_pure) is used for scoring.
    np = None  # type: ignore[assignment]

from .base import Tool, ToolResult

EMBED_URL = os.environ.get(
    "CHATCLI_EMBED_URL", "http://127.0.0.1:8081/embedding"
)


def _embed(text: str) -> list[float] | None:
    """Embedding of a single text (None on error)."""
    try:
        req = urllib.request.Request(
            EMBED_URL,
            data=json.dumps({"input": [text]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
        items = resp if isinstance(resp, list) else [resp]
        e = items[0]["embedding"]
        if isinstance(e, list) and e and isinstance(e[0], list):
            e = e[0]
        return e
    except (urllib.error.URLError, OSError, IndexError, KeyError):
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class SemanticSearchTool(Tool):
    name = "semantic_search"
    description = (
        "Semantic search via the local embedding index (meaning, "
        "not text match). Finds code/docs/wiki by meaning. "
        "Use for 'where is X documented' instead of regex search. "
        "After an ingest, incrementally update with reindex=true."
    )
    args_schema: dict[str, Any] = {
        "query": "str — search query (meaning, e.g. 'path problems sandbox')",
        "max_results": "int (optional, default 5) — number of results",
        "reindex": "bool (optional, default false) — incrementally update index (after ingest)",
        "scope": "str (optional, default 'all') — filter: 'wiki' (only wiki/), 'raw' (only raw/), 'code' (only code), 'all' (everything)",
    }

    def __init__(self, db_path: str | None = None) -> None:
        # DB path: embed_index.py is in the repo root, which is
        # 4 levels above this module (tools/agent/chatcli/src).
        repo_root = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "..", "..",
        ))
        self._db = db_path or os.path.join(repo_root, ".embed_index.db")
        self._workspace = repo_root
        # Embedding array cache: (db_mtime, (rows, embeddings array)).
        # The index only changes on reindex → per DB mtime we load the
        # vectors only ONCE, after which a query is just an
        # embedding HTTP call + one matrix multiplication.
        self._emb_cache: tuple[float, tuple[list, "np.ndarray"]] | None = None

    async def run(self, **kwargs: Any) -> ToolResult:
        query = str(kwargs.get("query", "")).strip()
        max_results = int(kwargs.get("max_results", 5))
        reindex = bool(kwargs.get("reindex", False))
        scope = str(kwargs.get("scope", "all")).strip().lower()

        if not query:
            return ToolResult(ok=False, output="", error="No 'query' specified.")

        # Optional incremental reindex (after ingest)
        if reindex:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "python3", "embed_index.py", "reindex",
                    cwd=self._workspace,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                out, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
                if proc.returncode != 0:
                    return ToolResult(
                        ok=False, output="",
                        error=f"Reindex failed: {out.decode(errors='replace')[:500]}"
                    )
            except (OSError, asyncio.TimeoutError) as e:
                return ToolResult(ok=False, output="", error=f"Reindex error: {e}")

        # Embedding of the query
        qvec = await asyncio.to_thread(_embed, query)
        if qvec is None:
            return ToolResult(
                ok=False, output="",
                error=f"Embedding server not reachable at {EMBED_URL} "
                      f"(is it running?)."
            )

        # Read DB + score
        if not os.path.isfile(self._db):
            return ToolResult(
                ok=False, output="",
                error="Index not found. First: python3 embed_index.py build"
            )

        try:
            rows, embs = await asyncio.to_thread(self._load_embeddings)
        except (sqlite3.Error, OSError) as e:
            return ToolResult(ok=False, output="", error=f"Index error: {e}")

        if not rows:
            return ToolResult(
                ok=False, output="",
                error="Index empty. First: python3 lex/embed_index.py build"
            )

        qnorm = sum(x * x for x in qvec) ** 0.5
        if qnorm == 0.0:
            return ToolResult(ok=False, output="", error="Query embedding is empty.")

        # Cosine score: with numpy as ONE matrix multiplication
        # (1241×384-FP32 flash in ~1 ms); without numpy (system Python via
        # lex.py) pure-Python fallback — same order of magnitude per query.
        if np is not None:
            q = np.asarray(qvec, dtype=np.float32)
            norms = np.linalg.norm(embs, axis=1)
            scores = embs @ q / (norms * qnorm)
            top = [(int(i), float(scores[i]))
                   for i in np.argsort(scores)[::-1][:max_results]]
        else:
            top = _top_k_pure(embs, qvec, max_results)

        # Scope filter: only paths in the desired area
        if scope in ("wiki", "raw", "code"):
            prefix = {
                "wiki": "lex/memory/wiki/",
                "raw": "lex/memory/raw/",
                "code": "",  # alles außer memory/
            }[scope]
            if scope == "code":
                top = [(i, s) for i, s in top
                       if not rows[i][0].startswith("lex/memory/")]
            else:
                top = [(i, s) for i, s in top
                       if rows[i][0].startswith(prefix)]
        lines = []
        for i, score in top:
            path, ln, text = rows[i][0], rows[i][1], rows[i][2]
            preview = " ".join(text.split())[:150]
            lines.append(f"{score:.4f}  {path}:{ln}  [{preview}]")
        return ToolResult(ok=True, output="\n".join(lines))

    def _load_embeddings(self) -> tuple[list, list]:
        """Loads (rows, embedding array) — cached per DB mtime.

        ``rows`` = [(path, line_no, text), …] in DB order; the
        array is (n_chunks, dim) float32. With numpy an ``ndarray``,
        without numpy (system Python) a list of float lists. On
        changed DB (reindex) the cache is discarded and reloaded.
        """
        mtime = os.path.getmtime(self._db)
        if self._emb_cache is not None and self._emb_cache[0] == mtime:
            return self._emb_cache[1]
        db = sqlite3.connect(self._db)
        rows_raw = db.execute(
            "SELECT path, line_no, text, emb FROM chunks").fetchall()
        db.close()
        rows = [(p, ln, t) for p, ln, t, _ in rows_raw]
        if rows_raw:
            dim = len(rows_raw[0][3]) // 4
            buf = b"".join(r[3] for r in rows_raw)
            if np is not None:
                embs = np.frombuffer(buf, dtype=np.float32).reshape(
                    len(rows_raw), dim)
            else:
                flat = struct.unpack(f"{len(buf) // 4}f", buf)
                embs = [list(flat[j * dim:(j + 1) * dim])
                        for j in range(len(rows_raw))]
        else:
            embs = (np.zeros((0, 0), dtype=np.float32) if np is not None
                    else [])
        self._emb_cache = (mtime, (rows, embs))
        return rows, embs


def _top_k_pure(embs: list[list[float]], qvec: list[float],
                max_results: int) -> list[tuple[int, float]]:
    """Top-k by cosine score WITHOUT numpy (brute-force computation, ~0.5M flops).

    Only relevant when numpy is missing (system Python via lex.py). Returns
    ``[(row_index, score), …]`` sorted descending.
    """
    qnorm = sum(x * x for x in qvec) ** 0.5
    if qnorm == 0.0:
        return []
    scored = []
    for i, v in enumerate(embs):
        dot = sum(a * b for a, b in zip(v, qvec))
        nrm = sum(x * x for x in v) ** 0.5
        scored.append((i, dot / (nrm * qnorm) if nrm else 0.0))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:max_results]

