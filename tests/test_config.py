"""Unit-Tests für chatcli.config.Config + Server-Parameter-Sync."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chatcli.config import Config
from chatcli.llama_client import LlamaClient


def test_defaults():
    # Ensure no env overrides
    for var in ("CHATCLI_HOST", "CHATCLI_PORT", "CHATCLI_MODEL",
                "CHATCLI_MAX_TOKENS", "CHATCLI_TEMPERATURE",
                "CHATCLI_MAX_STEPS", "CHATCLI_TIMEOUT",
                "CHATCLI_NO_SHELL", "CHATCLI_NO_FILES", "CHATCLI_NO_PLANNER"):
        os.environ.pop(var, None)
    c = Config()
    assert c.host == "127.0.0.1"
    assert c.port == 8080
    assert c.max_steps == 100
    assert c.base_url == "http://127.0.0.1:8080"


def test_env_override(monkeypatch):
    monkeypatch.setenv("CHATCLI_PORT", "9999")
    monkeypatch.setenv("CHATCLI_MODEL", "my-model")
    c = Config()
    assert c.port == 9999
    assert c.model == "my-model"


def test_chat_url():
    c = Config(host="192.168.1.1", port=7777)
    assert c.chat_url == "http://192.168.1.1:7777/v1/chat/completions"


def test_no_shell_flag():
    os.environ["CHATCLI_NO_SHELL"] = "1"
    try:
        c = Config()
        assert c.enable_shell is False
    finally:
        os.environ.pop("CHATCLI_NO_SHELL", None)


def test_sync_sampling_from_server():
    """Server-Parameter (GET /props) werden übernommen, wenn nicht explizit gesetzt."""
    c = Config()
    c.sync_sampling_from_server({
        "temperature": 1.0,
        "top_k": 20,
        "top_p": 0.95,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
        "n_ctx": 163840,
    })
    assert c.temperature == 1.0
    assert c.top_k == 20
    assert c.top_p == 0.95
    assert c.min_p == 0.0
    assert c.repeat_penalty == 1.0


def test_sync_sampling_respects_env_override(monkeypatch):
    """CHATCLI_TEMPERATURE gilt als explizit → Serverwert wird NICHT gesetzt."""
    monkeypatch.setenv("CHATCLI_TEMPERATURE", "0.3")
    c = Config()
    c.sync_sampling_from_server({"temperature": 1.0, "top_k": 5})
    assert c.temperature == 0.3
    # Nicht-overrideene Parameter fließen trotzdem durch
    assert c.top_k == 5


def test_sync_sampling_ignores_bad_values():
    """Ungültige/fehlende Werte dürfen nicht crashen — Defaults bleiben."""
    c = Config()
    c.sync_sampling_from_server({
        "temperature": "garbage",
        "top_k": None,
        "top_p": True,  # Bool gilt NICHT als Zahl
    })
    assert c.temperature == 0.6  # Default unverändert
    assert c.top_k == 20
    assert c.top_p == 0.95


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeHttpClient:
    """Mock für httpx.AsyncClient: liefert /props mit Payload."""

    def __init__(self, props_payload=None, status_error=False):
        self._props = props_payload
        self._status_error = status_error

    async def aclose(self):
        pass

    async def get(self, path, timeout=None):
        if path == "/props" and self._props is not None:
            return _FakeResp(self._props)
        import httpx
        raise httpx.HTTPError("no route")


def _client_with(props_payload):
    c = LlamaClient(Config())
    c._client = _FakeHttpClient(props_payload)
    return c


def test_server_params_parses_props():
    """GET /props → default_generation_settings.params extrahieren."""
    c = _client_with({
        "default_generation_settings": {
            "n_ctx": 163840,
            "params": {
                "temperature": 1.0,
                "top_k": 20,
                "top_p": 0.95,
                "min_p": 0.0,
            }
        }
    })
    import asyncio
    params = asyncio.run(c.server_params())
    assert params is not None
    assert params["temperature"] == 1.0
    assert params["top_k"] == 20
    assert params["n_ctx"] == 163840


def test_server_params_none_on_error():
    """Fehlende Props → None (Server ohne /props-Endpunkt)."""
    c = _client_with(None)
    import asyncio
    assert asyncio.run(c.server_params()) is None


def test_server_params_ignores_malformed():
    """Kaputtes JSON-Struktur → None statt Exception."""
    c = _client_with({"default_generation_settings": "kaputt"})
    import asyncio
    assert asyncio.run(c.server_params()) is None


# ─── _detect_project_root ───────────────────────────────────────────────

def test_detect_project_root_marker_in_cwd(tmp_path):
    """pyproject.toml im CWD → wird als Root erkannt."""
    from chatcli.config import _detect_project_root
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    result = _detect_project_root(tmp_path)
    assert result == tmp_path


def test_detect_project_root_marker_in_parent(tmp_path):
    """CWD ist Unterverzeichnis, Marker liegt im Parent → Parent wird erkannt."""
    from chatcli.config import _detect_project_root
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    sub = tmp_path / "src" / "chatcli"
    sub.mkdir(parents=True)
    result = _detect_project_root(sub)
    assert result == tmp_path


def test_detect_project_root_subdir_fallback(tmp_path):
    """Kein Marker in CWD/Eltern, aber Unterverzeichnis hat Marker → Fallback."""
    from chatcli.config import _detect_project_root
    # tmp_path hat KEINEN Marker (simuliert ~/)
    project = tmp_path / "Ai"
    project.mkdir()
    (project / ".git").mkdir()
    result = _detect_project_root(tmp_path)
    assert result == project


def test_detect_project_root_no_marker_returns_start(tmp_path):
    """Kein Marker überall → start-Pfad wird zurückgegeben."""
    from chatcli.config import _detect_project_root
    result = _detect_project_root(tmp_path)
    assert result == tmp_path
