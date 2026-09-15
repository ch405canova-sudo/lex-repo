"""Memory access: inject style rules + lessons from the wiki at startup.

The memory itself (raw/ + wiki/) lives under config.wiki_dir.
Two topics are automatically injected into the system prompt at EVERY start:
- Style topic (config.style_topic): behavior rules (e.g. humor/emojis)
- Lessons topic (config.lessons_topic): lessons from resolved errors,
  so Lex doesn't repeat the same mistakes (compounding intelligence)

The rest of the wiki stays lazy (only read by Lex via tools on query/ingest).
"""

from __future__ import annotations

import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from .config import Config

# Embedding server for selective lesson injection (same server
# as the semantic_search tool).
_EMBED_URL = os.environ.get(
    "CHATCLI_EMBED_URL", "http://127.0.0.1:8081/embedding"
)
# Cache: normalized lesson body -> embedding. Lessons are rare;
# per session each lesson is embedded only ONCE. If the body text
# changes, the key changes -> automatically re-embedded.
_lesson_emb_cache: dict[str, list[float]] = {}


def wiki_root(config: Config) -> Path:
    """Absolute path to the memory directory (raw/ + wiki/)."""
    return config.wiki_root


def skill_dir(config: Config) -> Path:
    """Absolute path to the wiki skill (SKILL.md, references/, scripts/)."""
    return config.skill_dir


def _topic_bodies(config: Config, topic: str) -> list[str]:
    """All articles of a topic as body (without metadata header), sorted."""
    topic_dir = wiki_root(config) / "wiki" / topic
    if not topic_dir.is_dir():
        return []
    parts: list[str] = []
    for f in sorted(topic_dir.glob("*.md")):
        try:
            text = f.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        # Drop metadata lines (> Sources/Raw/Updated/Archived),
        # they are irrelevant for the prompt.
        lines = [l for l in text.splitlines() if not l.lstrip().startswith(">")]
        body = "\n".join(lines).strip()
        if body:
            parts.append(body)
    return parts


def style_block(config: Config) -> str:
    """All style rules of the topic as prompt text ("" if empty)."""
    return "\n\n".join(_topic_bodies(config, config.style_topic))


def _lesson_entries(config: Config, topic: str) -> list[tuple[str, str]]:
    """Lessons as (collected_date, body), archived + duplicates removed.

    Lifecycle:
    - `> Archived:` lessons are NOT injected (only in the wiki).
    - Identical bodies (after normalization) count only once.
    - Sorting: newest collected date first.
    """
    topic_dir = wiki_root(config) / "wiki" / topic
    if not topic_dir.is_dir():
        return []
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for f in sorted(topic_dir.glob("*.md")):
        try:
            text = f.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if re.search(r"^>\s*Archived:", text, re.MULTILINE):
            continue  # archived -> no longer in the prompt
        m = re.search(r"^>\s*Collected:\s*(\d{4}-\d{2}-\d{2})", text, re.MULTILINE)
        date = m.group(1) if m else "0000-00-00"
        lines = [l for l in text.splitlines() if not l.lstrip().startswith(">")]
        body = "\n".join(lines).strip()
        if not body:
            continue
        key = " ".join(body.split()).lower()  # dedup: normalized body
        if key in seen:
            continue
        seen.add(key)
        entries.append((date, body))
    entries.sort(key=lambda e: e[0], reverse=True)
    return entries


def _embed_texts(texts: list[str]) -> list[list[float]] | None:
    """Embeddings of a text list (None on server error)."""
    if not texts:
        return []
    try:
        import json
        req = urllib.request.Request(
            _EMBED_URL,
            data=json.dumps({"input": texts}).encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
        items = resp if isinstance(resp, list) else [resp]
        out: list[list[float]] = []
        for item in items:
            e = item["embedding"]
            if isinstance(e, list) and e and isinstance(e[0], list):
                e = e[0]
            out.append(e)
        return out
    except (urllib.error.URLError, OSError, KeyError, ValueError):
        return None


# Embedding length per lesson: only the first ~480 chars (title +
# error/cause). Longer bodies would break the 511-token limit of the
# embedding server and crash the whole batch.
_EMBED_CHARS = 480


def _lesson_vectors(
   entries: list[tuple[str, str]]
) -> list[list[float]] | None:
    """Embeddings of lesson bodies (with cache). None = server down.

    Per lesson only the first `_EMBED_CHARS` chars are embedded —
    this keeps each lesson under the 511-token limit of the server and
    doesn't break the batch.
    """
    keys = [
        " ".join(body[:_EMBED_CHARS].split()).lower()
        for _, body in entries
    ]
    vecs: list[list[float] | None] = []
    for key in keys:
        vecs.append(_lesson_emb_cache.get(key))
    if all(v is not None for v in vecs):
        return [v for v in vecs if v is not None]
    missing = [b[:_EMBED_CHARS] for (_, b), k in zip(entries, keys)
               if k not in _lesson_emb_cache]
    fresh = _embed_texts(missing)
    if fresh is None:
        return None
    it = iter(fresh)
    out: list[list[float]] = []
    for (_, body), k in zip(entries, keys):
        if k not in _lesson_emb_cache:
            e = next(it)
            _lesson_emb_cache[k] = e
        out.append(_lesson_emb_cache[k])
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def select_lesson_bodies(
   config: Config, query: str, count: int
) -> list[str]:
    """Selective lesson selection: the `count` lessons that best match
    the current task (query) (semantic, not "newest N").

    Falls back to "newest N" if the embedding server is unreachable —
    injection must never block startup.
    """
    entries = _lesson_entries(config, config.lessons_topic)
    if not query or not entries:
        return [b for _, b in entries[:count]]
    qvec = (_embed_texts([query]) or [None])[0]
    if qvec is None:
        return [b for _, b in entries[:count]]
    vecs = _lesson_vectors(entries)
    if vecs is None:
        return [b for _, b in entries[:count]]
    scored = sorted(
        zip(entries, vecs), key=lambda p: _cosine(qvec, p[1]), reverse=True
    )
    return [body for ((_, body), _) in scored[:count]]


def lessons_block(config: Config, query: str | None = None) -> str:
    """Lessons as prompt text ("" if empty).

    Without `query`: the `max_injected_lessons` NEWEST (startup behavior).
    With `query`: the `selective_lesson_count` MOST RELEVANT (semantic
    selection, less context noise per task).
    """
    if query:
        bodies = select_lesson_bodies(
            config, query, config.selective_lesson_count
        )
    else:
        entries = _lesson_entries(config, config.lessons_topic)
        bodies = [b for _, b in entries[: config.max_injected_lessons]]
    return "\n\n".join(bodies)


def style_injection(config: Config, query: str | None = None) -> str:
    """Complete injection block: notice + style rules (if present).

    `query` (current user task): if set AND
    `enable_selective_lessons` active, the semantically RELEVANT lessons
    are injected instead of the newest N (less context noise).
    """
    root = wiki_root(config)
    skill = skill_dir(config)
    if not root.joinpath("wiki").is_dir():
        return ""

    # Style rules first (stable across turns -> prefix-cache friendly),
    # lessons last (can change per turn).
    style = style_block(config)
    style_block_text = ""
    if style:
        style_block_text = (
            f"\n## Style Rules (from your memory, topic '{config.style_topic}')\n"
            "These rules are in effect NOW from this moment:\n\n"
            + style
        )

    block = (
        "\n## Active Memory\n"
        f"Your wiki is under {root}. Full schema: "
        f"{skill}/SKILL.md (read on ingest/lint).\n"
        "- **Ingest** (user says 'remember this', 'learn <topic>', or provides source/URL):\n"
        f"  Source to {root}/raw/<topic>/YYYY-MM-DD-slug.md (immutable) —\n"
        "  use scrape with save_to_raw (writes the source directly to raw/),\n"
        f"  then create/update wiki page in {root}/wiki/<topic>/\n"
        f"  (templates: {skill}/references/), then update index.md and log.md.\n"
        "  For topics use web_search + scrape.\n"
        "- **Query** ('what do I know about X'): first read "
        f"{root}/wiki/index.md, then search the wiki, answer\n"
        "  with links to the wiki pages. Wiki content takes priority over training.\n"
        "- **Lint** ('check wiki'): verify structure/links and run\n"
        f"  python3 {skill}/scripts/check_evidence.py {root}.\n"
        "- Never invent facts: only what is in raw/ gets cited.\n"
        "  Mark contradictions as 'Status: Disputed', never silently overwrite.\n"
        "\n"
        f"## Lessons (your error memory, topic '{config.lessons_topic}')\n"
        "The injected lessons below are lessons from errors you have already\n"
        "made and resolved — they are in effect from this moment.\n"
        "- Do not repeat any error for which a lesson exists.\n"
        "- When you resolve an error whose solution is not yet a lesson:\n"
        f"  write ONE short lesson (5-15 lines: Error / Cause / Solution)\n"
        f"  as a new wiki page in {root}/wiki/{config.lessons_topic}/ and register it\n"
        "  in index.md + log.md (format: SKILL.md -> Lessons).\n"
        "- Success lessons: If an UNCONVENTIONAL approach worked\n"
        "  (not an error, but a trick/workaround that saved you time),\n"
        "  document it as a short success lesson (max 5 lines: Situation /\n"
        "  Trick / Result) in the same topic — only if it is reusable.\n"
        "- Outdated lessons: If an injected lesson is obsolete (e.g. due to\n"
        "  a code fix), add the line '> Archived: <date>' to the\n"
        "  file — it will then no longer be injected, but remains in the wiki.\n"
    )

    block += style_block_text

    # Lessons ONLY on query (per turn via _refresh_system_memory).
    # At startup (query=None) NO lessons are injected — only style rules.
    # This saves ~30k chars of context at startup; the selective injection
    # from turn 1 delivers the relevant lessons per task.
    if query and config.enable_selective_lessons:
        lessons = lessons_block(config, query=query)
        if lessons:
            block += (
                f"\n## Injected Lessons (from your memory, topic '{config.lessons_topic}')\n"
                "These lessons are in effect NOW from this moment:\n\n"
                + lessons
            )

    return block


def learn_prompt(config: Config, topic: str) -> str:
    """Ready-made prompt for 'learn <topic>' (web-based ingest)."""
    root = wiki_root(config)
    skill = skill_dir(config)
    return (
        f"LEARN THE TOPIC: {topic}\n"
        f"\n"
        f"1. Use web_search (2-3 different search terms, incl. synonyms)\n"
        f"   and scrape to find 2-4 good sources.\n"
        f"2. Save each source under {root}/raw/<topic>/YYYY-MM-DD-slug.md\n"
        f"   (template: {skill}/references/raw-template.md) — easiest with\n"
        f"   scrape(save_to_raw=true, topic='...') writing directly to raw/.\n"
        f"3. Compile 1-3 wiki pages under {root}/wiki/<topic>/ "
        f"(article-template.md), with raw links.\n"
        f"4. Update {root}/wiki/index.md and {root}/wiki/log.md.\n"
        f"5. At the end: short summary of what you learned (with emojis, "
        f"if the style rules allow)."
    )
