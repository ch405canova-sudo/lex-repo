"""Parser für Tool-Call-Antworten des Modells.

Unterstützt:
- Reines JSON: {"tool": "shell", "args": {"command": "ls"}}
- JSON in Markdown-Codeblock: ```json ... ```
- Mehrere Tools hintereinander (Array)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    tool: str
    args: dict[str, Any]


def parse_tool_calls(text: str) -> list[ToolCall] | None:
    """
    Versuche, aus dem Modell-Output Tool-Calls zu extrahieren.

    Rückgabe:
    - list[ToolCall] — wenn valide Tool-Calls gefunden
    - None — wenn kein Tool-Call erkannt wird (normale Antwort)
    """
    text = text.strip()

    # 1) JSON-Codeblock extrahieren, falls vorhanden
    json_str = _extract_json_block(text)

    # 2) Direkter JSON-Versuch
    candidates = []
    if json_str:
        candidates.append(json_str)
    candidates.append(text)

    for raw in candidates:
        result = _try_parse(raw)
        if result is not None:
            return result

    return None


def _extract_json_block(text: str) -> str | None:
    r"""Extrahiere den KOMPLETTEN Inhalt zwischen ```-Markern.

    Marker-basiert statt Regex-basiert: Der alte nicht-gierige Regex-Ansatz
    (\{.*?\} / \[.*?\]) brach beim ersten geschlossenen } bzw. ] ab und
    zerlegte verschachteltes JSON (z. B. {"tool": "x", "args": {"a": "b"}})
    in Trümmer. Hier wird der Block bis zum nächsten ```-Marker komplett
    übernommen und von _try_parse() ausgewertet.
    """
    m = re.search(r"```[ \t]*(?:json)?[ \t]*\n?", text)
    if not m:
        return None
    start = m.end()
    end = text.find("```", start)
    if end == -1:
        return None
    return text[start:end]


def _data_to_calls(data: Any) -> list[ToolCall] | None:
    """Konvertiere ein geparstes JSON-Objekt/-Array in ToolCalls."""
    if isinstance(data, dict):
        if "tool" in data and isinstance(data["tool"], str):
            args = data.get("args", {})
            if not isinstance(args, dict):
                args = {}
            return [ToolCall(tool=data["tool"], args=args)]
        return None
    if isinstance(data, list) and data and all(
        isinstance(x, dict) and "tool" in x for x in data
    ):
        return [
            ToolCall(tool=x["tool"], args=x.get("args", {}) or {}) for x in data
        ]
    return None


def _try_parse(raw: str) -> list[ToolCall] | None:
    """Versuche, aus raw ein einzelnes Tool-Call-Objekt oder eine Liste zu parsen.

    Toleriert auch mehrere aufeinanderfolgende JSON-Objekte ohne Array-Wrapper
    (z.B. eines pro Zeile) — das Modell macht das regelmäßig.
    """
    raw = raw.strip()
    if not raw:
        return None

    # Zum ersten '{' oder '[' springen (Präfix-Text ignorieren)
    start = 0
    if not raw.startswith(("{", "[")):
        for i, c in enumerate(raw):
            if c in "{[":
                start = i
                break
        else:
            return None
        raw = raw[start:]

    # 1) Gesamter Text ist ein einziges gültiges JSON (Objekt oder Array)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None
    if data is not None:
        parsed = _data_to_calls(data)
        if parsed is not None:
            return parsed

    # 2) Fallback: nacheinander stehende JSON-Objekte (raw_decode)
    calls: list[ToolCall] = []
    dec = json.JSONDecoder()
    idx = 0
    while idx < len(raw):
        if raw[idx] not in "{[":
            idx += 1
            continue
        try:
            obj, end = dec.raw_decode(raw, idx)
        except json.JSONDecodeError:
            break
        idx = end
        if isinstance(obj, dict) and "tool" in obj and isinstance(obj["tool"], str):
            calls.append(
                ToolCall(tool=obj["tool"], args=obj.get("args", {}) or {})
            )
    return calls or None
