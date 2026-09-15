"""Unit-Tests für SystemTool (Snapshot) und JobsTool (Background-Prozesse)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chatcli.agent.tools.system import SystemTool
from chatcli.agent.tools.jobs import JobsTool
from chatcli.agent.tools.base import build_default_registry


# ── SystemTool ─────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_system_snapshot_contains_sections():
    """Snapshot liefert Last + Disk + Top-Prozesse (Kern-Sektionen)."""
    t = SystemTool()
    r = await t.run()
    assert r.ok, r.error
    assert "System-Snapshot" in r.output
    assert "Uptime/Last:" in r.output
    assert "Disk:" in r.output
    assert "TopProc:" in r.output


@pytest.mark.anyio
async def test_system_filter_to_ram_only():
    """sections=['ram'] liefert nur die RAM-Sektion."""
    t = SystemTool()
    r = await t.run(sections=["ram"])
    assert r.ok, r.error
    assert "RAM:" in r.output
    # Die Disk-Sektion darf dabei NICHT mitschleppen
    assert "Disk:" not in r.output


@pytest.mark.anyio
async def test_system_output_capped():
    """Snapshot wird nie länger als _MAX_CHARS (Kontext-Schutz)."""
    t = SystemTool()
    r = await t.run()
    assert len(r.output) <= SystemTool._MAX_CHARS + 30


# ── JobsTool ───────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_jobs_lifecycle():
    """start → status → read → (beendet) → kill ist no-op."""
    t = JobsTool()
    start = await t.run(action="start", command="echo job-output; echo done")
    assert start.ok, start.error
    job_id = start.output.splitlines()[0].split()[0]

    status = await t.run(action="status")
    assert status.ok
    assert job_id in status.output

    # Warten bis der Job fertig ist (max. 5s)
    for _ in range(50):
        await asyncio.sleep(0.1)
        s = await t.run(action="status")
        if "beendet" in s.output:
            break

    read = await t.run(action="read", job=job_id)
    assert read.ok
    assert "job-output" in read.output
    assert "done" in read.output

    kill = await t.run(action="kill", job=job_id)
    assert kill.ok
    assert "already finished" in kill.output


@pytest.mark.anyio
async def test_jobs_kill_running_process():
    """kill beendet einen laufenden Prozess tatsächlich."""
    t = JobsTool()
    start = await t.run(action="start", command="sleep 30")
    assert start.ok, start.error
    job_id = start.output.splitlines()[0].split()[0]

    kill = await t.run(action="kill", job=job_id)
    assert kill.ok, kill.error
    assert "terminated" in kill.output


@pytest.mark.anyio
async def test_jobs_read_streams_to_on_line():
    """on_line bekommt job-Zeilen für die Live-Box."""
    seen: list[str] = []
    t = JobsTool(on_line=seen.append)
    start = await t.run(action="start", command="echo live-zeile")
    assert start.ok
    job_id = start.output.splitlines()[0].split()[0]
    for _ in range(50):
        await asyncio.sleep(0.1)
        if any(job_id in line for line in seen):
            break
    assert any("live-zeile" in line for line in seen)


@pytest.mark.anyio
async def test_jobs_start_requires_command():
    t = JobsTool()
    r = await t.run(action="start")
    assert not r.ok
    assert "command" in r.error


@pytest.mark.anyio
async def test_jobs_unknown_action_rejected():
    t = JobsTool()
    r = await t.run(action="explode")
    assert not r.ok


# ── Registry-Wiring ────────────────────────────────────────────────


def test_registry_registers_system_and_jobs():
    reg = build_default_registry(enable_system=True)
    assert reg.get("system") is not None
    assert reg.get("jobs") is not None


def test_registry_can_disable_system():
    reg = build_default_registry(enable_system=False)
    assert reg.get("system") is None
    assert reg.get("jobs") is None
