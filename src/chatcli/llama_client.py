"""HTTP-Client für llama-server (OpenAI-kompatible API).

Unterstützt native Tool-Calls (``tools=...`` im Request, ``tool_calls`` im
Delta). Ob der Server das kann, wird lazily erkannt: Bei HTTP 400 mit
Tool-Hinweis im Body wird automatisch ohne Tools neu versucht und der
Fallback (Text-Protokoll) dauerhaft aktiviert.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import httpx

from .config import Config

log = logging.getLogger(__name__)


@dataclass
class StreamChunk:
    """Ein einzelnes Delta aus einem streamenden Request.

    Tool-Call-Felder: ``tool_call_index`` ist ``None``, wenn der Chunk nur
    Prosa/Reasoning enthält; ansonsten beschreiben die Felder das Delta des
    Tool-Calls mit dieser Index-Position (``id``/``name`` kommen meist im
    ersten Chunk, ``args`` wird per Delta angestückt).
    """
    content: str = ""
    reasoning: str = ""
    finish_reason: str = ""
    tool_call_index: Optional[int] = None
    tool_call_id: str = ""
    tool_call_name: str = ""
    tool_call_args_delta: str = ""


class LlamaClient:
    """Async-Client für den lokalen llama-server."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(config.timeout, connect=10.0),
        )
        # None = unbekannt (Server wird mit Tools angesprochen),
        # False  = Server lehnt Tools ab → dauerhaft Text-Protokoll.
        self._supports_tools: Optional[bool] = None

    @property
    def supports_tools(self) -> bool:
        """True, bis der Server Tools explizit abgelehnt hat."""
        return self._supports_tools is not False

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        """Prüfe ob der Server erreichbar ist."""
        try:
            r = await self._client.get("/health", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def server_params(self) -> Optional[dict]:
        """Lade die Server-Sampling-Parameter des llama-server (``GET /props``).

        Der Server (gestartet über ai.sh) ist die Single Source of Truth für
        Sampling: ``default_generation_settings.params`` enthält die
        effektiv gültigen ``temperature``, ``top_k``, ``top_p``, ``min_p``,
        ``repeat_penalty`` usw. ``n_ctx`` (Kontextgröße) sitzt eine Ebene
        höher und wird hier mit in das Ergebnis-Dictionary aufgenommen.
        Rückgabe ``None``, wenn der Endpunkt keine Props liefert.
        """
        try:
            r = await self._client.get("/props", timeout=5.0)
            r.raise_for_status()
            props = r.json()
        except (httpx.HTTPError, json.JSONDecodeError, ValueError):
            return None
        if not isinstance(props, dict):
            return None
        dgs = props.get("default_generation_settings")
        if not isinstance(dgs, dict):
            return None
        params = dgs.get("params")
        if not isinstance(params, dict):
            return None
        result = dict(params)
        # n_ctx liegt in default_generation_settings, nicht in params
        if isinstance(dgs.get("n_ctx"), int):
            result["n_ctx"] = dgs["n_ctx"]
        return result

    async def context_usage(self) -> Optional[dict]:
        """Frage die Kontext-Nutzung des llama-server ab (``GET /slots``).

        Rückgabe: ``{"n_ctx": <Gesamt>, "n_ctx_used": <belegt>}`` — oder
        ``None``, wenn der Server keine Slot-Infos liefert.
        """
        try:
            r = await self._client.get("/slots", timeout=5.0)
            r.raise_for_status()
            slots = r.json()
        except (httpx.HTTPError, json.JSONDecodeError, ValueError):
            return None
        if not isinstance(slots, list) or not slots:
            return None
        slot = slots[0]
        if not isinstance(slot, dict):
            return None
        n_ctx = slot.get("n_ctx")
        # Belegte Tokens = Prompt-Tokens des aktiven Slots (KV-Cache).
        # Fallback: n_prompt_tokens_processed, sonst 0.
        n_used = slot.get("n_prompt_tokens")
        if not isinstance(n_used, int):
            n_used = slot.get("n_prompt_tokens_processed", 0)
        if not isinstance(n_ctx, int):
            return None
        return {"n_ctx": n_ctx, "n_ctx_used": n_used}

    # ------------------------------------------------------------------

    def _build_payload(
        self,
        messages: list[dict],
        *,
        stream: bool,
        tools: Optional[list[dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        payload: dict = {
            "model": self._config.model,
            "messages": messages,
            "stream": stream,
            "temperature": self._config.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self._config.max_tokens,
            "top_p": self._config.top_p,
            "top_k": self._config.top_k,
            "min_p": self._config.min_p,
            "repeat_penalty": self._config.repeat_penalty,
        }
        if tools:
            payload["tools"] = tools
        return payload

    @staticmethod
    def _parse_tool_call_delta(delta: dict) -> list[
        tuple[Optional[int], str, str, str]
    ]:
        """Extrahiere ALLE Tool-Call-Deltas aus einem Delta.

        Ein Stream-Delta kann mehrere ``tool_calls``-Einträge enthalten
        (z. B. wenn das Modell parallel mehrere Tools aufruft). Jeder
        Eintrag wird als eigener Tuple (index, id, name, args) zurückgegeben.
        """
        tcs = delta.get("tool_calls")
        if not tcs:
            return []
        results: list[tuple[Optional[int], str, str, str]] = []
        for tc in tcs:
            idx = tc.get("index")
            fn = tc.get("function") or {}
            results.append((
                idx,
                tc.get("id") or "",
                fn.get("name") or "",
                fn.get("arguments") or "",
            ))
        return results

    async def chat_stream(
        self,
        messages: list[dict],
        *,
        tools: Optional[list[dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Streamende Chat-Anfrage — liefert StreamChunk by StreamChunk.

        ``tools``: OpenAI-Tool-Definitionen (native Tool-Calls). Wird der
        Server mit Tools nicht akzeptiert (HTTP 400), wird automatisch
        ohne Tools neu gestreamt und der Text-Protokoll-Fallback gesetzt.
        Der finale Chunk enthält finish_reason ("stop"/"length").
        """
        kwargs = dict(stream=True, temperature=temperature, max_tokens=max_tokens)
        payload = self._build_payload(
            messages,
            tools=None if (tools and self._supports_tools is False) else tools,
            **kwargs,
        )

        # #5 Error-Classification + Retry: Transiente HTTP-Fehler (429, 5xx)
        # werden mit exponentiellem Backoff retryed (max 3×).
        # Fatale Fehler (401, 403, 422) werfen sofort.
        _RETRYABLE_CODES = {429, 500, 502, 503, 504}
        _FATAL_CODES = {401, 403, 422}
        _MAX_RETRIES = 3

        attempt = 0
        while True:
            attempt += 1
            try:
                async with self._client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json=payload,
                ) as resp:
                    # Fatale Fehler: sofort abbrechen.
                    if resp.status_code in _FATAL_CODES:
                        raise httpx.HTTPStatusError(
                            f"Fatal HTTP error {resp.status_code}",
                            request=resp.request,
                            response=resp,
                        )
                    # Transiente Fehler: retryen mit Backoff.
                    if resp.status_code in _RETRYABLE_CODES:
                        if attempt > _MAX_RETRIES:
                            raise httpx.HTTPStatusError(
                                f"Server error {resp.status_code} after "
                                f"{_MAX_RETRIES} retries — server possibly down.",
                                request=resp.request,
                                response=resp,
                            )
                        delay = min(2 ** attempt, 10.0)
                        log.warning(
                            "Transient error %d (attempt %d/%d) — "
                            "retrying in %.1fs",
                            resp.status_code, attempt, _MAX_RETRIES, delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    if resp.status_code == 400 and tools and self._supports_tools is not False:
                        # Capability-Check: Server lehnt 'tools' ab (entweder
                        # erstes Mal oder nach Server-Neustart/Config-Change)
                        # → Fallback auf das Text-Protokoll, dauerhaft.
                        body = (await resp.aread()).decode(errors="replace")
                        if "tool" in body.lower():
                            self._supports_tools = False
                            log.info(
                                "Server does not support 'tools' — "
                                "falling back to JSON text protocol."
                            )
                            tools = None
                            payload = self._build_payload(
                                messages, tools=None, **kwargs
                            )
                            continue
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:]  # strip "data:"
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(data_str)
                        except json.JSONDecodeError:
                            log.warning(
                                "Stream: ignoring broken JSON line: %s", data_str[:200]
                            )
                            continue
                        if isinstance(chunk_data.get("error"), (dict, str)):
                            log.warning("Stream error from server: %s", chunk_data["error"])
                            continue
                        choices = chunk_data.get("choices")
                        if not isinstance(choices, list) or not choices:
                            continue
                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        tc_deltas = self._parse_tool_call_delta(delta)
                        content = delta.get("content") or ""
                        reasoning = delta.get("reasoning_content") or ""
                        finish_reason = choice.get("finish_reason") or ""
                        if tc_deltas:
                            # Ein Delta kann mehrere tool_calls-Einträge enthalten —
                            # jeder wird als eigener Chunk yieldet, sonst gingen
                            # alle außer dem ersten verloren.
                            for i, (tc_idx, tc_id, tc_name, tc_args) in enumerate(tc_deltas):
                                yield StreamChunk(
                                    content=content if i == 0 else "",
                                    reasoning=reasoning if i == 0 else "",
                                    finish_reason=finish_reason if i == len(tc_deltas) - 1 else "",
                                    tool_call_index=tc_idx,
                                    tool_call_id=tc_id,
                                    tool_call_name=tc_name,
                                    tool_call_args_delta=tc_args,
                                )
                        else:
                            yield StreamChunk(
                                content=content,
                                reasoning=reasoning,
                                finish_reason=finish_reason,
                            )
                return
            except httpx.HTTPStatusError as e:
                # Transiente Fehler, die nicht durch den Retry-Loop gefangen
                # wurden (z. B. ConnectionError mit 5xx), hier abfangen.
                if e.response is not None and e.response.status_code in _RETRYABLE_CODES:
                    if attempt > _MAX_RETRIES:
                        raise
                    delay = min(2 ** attempt, 10.0)
                    log.warning(
                        "HTTP error %d (attempt %d/%d) — retrying in %.1fs",
                        e.response.status_code, attempt, _MAX_RETRIES, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
