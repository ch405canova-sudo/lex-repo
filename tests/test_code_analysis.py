"""Unit-Tests für das Code-Analysis-Tool (mypy/bandit/pylint/radon/vulture)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Awaitable

import pytest

from chatcli.agent.tools.code_analysis import CodeAnalysisTool, _build_cmd


def test_build_cmd_all_tools():
    assert _build_cmd("mypy", ".") == ["uv", "run", "mypy", "--strict", "."]
    assert _build_cmd("bandit", "src") == ["uv", "run", "bandit", "-r", "src", "-f", "txt"]
    assert _build_cmd("pylint", ".") == ["uv", "run", "pylint", "--score=y", "."]
    assert _build_cmd("radon", ".") == ["uv", "run", "radon", "cc", "-a", "-s", "."]
    assert _build_cmd("vulture", ".") == [
        "uv", "run", "vulture", "--min-confidence", "60", ".",
    ]


def test_build_cmd_unknown_tool():
    with pytest.raises(ValueError, match="Unknown tool"):
        _build_cmd("flake8", ".")


@pytest.mark.anyio
async def test_missing_tool_param(tmp_path: Path):
    tool = CodeAnalysisTool(project_root=str(tmp_path))
    res = await tool.run()
    assert not res.ok
    assert "tool parameter" in res.error


@pytest.mark.anyio
async def test_unknown_tool_rejected(tmp_path: Path):
    tool = CodeAnalysisTool(project_root=str(tmp_path))
    res = await tool.run(tool="flake8")
    assert not res.ok
    assert "Unknown tool" in res.error


@pytest.mark.anyio
async def test_missing_path_rejected(tmp_path: Path):
    tool = CodeAnalysisTool(project_root=str(tmp_path))
    res = await tool.run(tool="mypy", path="does/not/exist")
    assert not res.ok
    assert "does not exist" in res.error


class _FakeProc:
    """Simuliert einen asyncio-Subprocess (stdout als Bytes, Returncode)."""

    def __init__(self, out: bytes, code: int) -> None:
        self._out = out
        self.returncode = code

    async def communicate(self) -> tuple[bytes, Any]:
        return self._out, None


def _patch_exec(monkeypatch, out: bytes, code: int) -> None:
    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        return _FakeProc(out, code)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)


@pytest.mark.anyio
async def test_run_success(monkeypatch, tmp_path: Path):
    (tmp_path / "src" / "chatcli").mkdir(parents=True)
    _patch_exec(monkeypatch, b"[mypy] exit_code=0\nSuccess: no issues found", 0)

    tool = CodeAnalysisTool(project_root=str(tmp_path))
    res = await tool.run(tool="mypy")
    assert res.ok
    assert "[mypy]" in res.output
    assert "Success" in res.output


@pytest.mark.anyio
async def test_run_nonzero_exit(monkeypatch, tmp_path: Path):
    (tmp_path / "src" / "chatcli").mkdir(parents=True)
    _patch_exec(monkeypatch, b"[mypy] exit_code=1\n2 errors found", 1)

    tool = CodeAnalysisTool(project_root=str(tmp_path))
    res = await tool.run(tool="mypy")
    assert not res.ok
    assert "exit_code=1" in res.output


@pytest.mark.anyio
async def test_run_output_truncated(monkeypatch, tmp_path: Path):
    (tmp_path / "src" / "chatcli").mkdir(parents=True)
    _patch_exec(monkeypatch, b"x" * 20_000, 0)

    tool = CodeAnalysisTool(project_root=str(tmp_path))
    res = await tool.run(tool="mypy")
    assert "cut off" in res.output


@pytest.mark.anyio
async def test_run_timeout(monkeypatch, tmp_path: Path):
    (tmp_path / "src" / "chatcli").mkdir(parents=True)

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        raise asyncio.TimeoutError()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    tool = CodeAnalysisTool(project_root=str(tmp_path))
    res = await tool.run(tool="mypy")
    assert not res.ok
    assert "timeout" in res.error.lower()


def test_registry_registers_code_analysis():
    from chatcli.agent.tools.base import build_default_registry

    reg = build_default_registry(
        enable_shell=False,
        enable_file_ops=False,
        enable_planner=False,
        enable_scrape=False,
        enable_web_search=False,
        enable_semantic_search=False,
        enable_script=False,
        enable_self_improve=False,
        enable_system=False,
    )
    assert "code_analysis" in reg.names()
