"""Tests für die Guardrails (harte Gates + Confirm-Kategorie)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chatcli.agent.tools.guardrails import check_command


# ── Hard-Block ─────────────────────────────────────────────────────

@pytest.mark.parametrize("cmd", [
    "mkfs.ext4 /dev/sda1",
    "sudo mkfs /dev/sdb",
    "dd if=/dev/zero of=/dev/sda bs=1M",
    "rm -rf /",
    "rm -rf ~",
    "rm -rf $HOME",
    ":(){ :|:& };:",
    "shutdown -h now",
    "reboot",
    "echo x > /dev/sda",
    "chmod -R 777 /",
    "userdel chaos",
    "crontab -r",
    "history -c",
])
def test_hard_block(cmd):
    hit = check_command(cmd)
    assert hit is not None
    assert hit[0] == "hard"


# ── Confirm-Kategorie ─────────────────────────────────────────────

@pytest.mark.parametrize("cmd", [
    "rm -rf ./build",
    "rm -r target/",
    "rm -f file.txt",
    "find /tmp -name '*.log' -delete",
    "dd if=disk.img of=backup.img",
    "chmod -R 755 /home/chaos/proj",
    "chown -R user:group /srv/app",
    "mv /etc/passwd /dev/null",
    "killall chrome",
    "pkill -f llama-server",
    "systemctl restart nginx",
    "iptables -F",
    "git reset --hard HEAD~1",
    "git clean -fd",
])
def test_confirm_category(cmd):
    hit = check_command(cmd)
    assert hit is not None
    assert hit[0] == "confirm"


# ── Erlaubt (kein False-Positive) ─────────────────────────────────

@pytest.mark.parametrize("cmd", [
    "ls -la",
    "echo hello",
    "grep -rn TODO src/",
    "cat file.txt",
    "rm file.txt",            # einzelnes File ohne -r/-f
    "python3 script.py",
    "curl -s https://example.com",
    "nmap -sT 192.168.1.1",
    "systemctl status nginx",
    "systemctl list-units",
    "dd help",                # 'dd' als Argument, kein dd-Befehl
])
def test_allowed(cmd):
    assert check_command(cmd) is None


def test_empty_command_allowed():
    assert check_command("") is None


# ── Heredoc-Strip (False-Positive-Schutz) ───────────────────────────

@pytest.mark.parametrize("cmd", [
    # "shutdown" im Heredoc-Inhalt darf NICHT blockieren:
    "cat > /tmp/file.py << 'EOF'\n#!/usr/bin/env python3\n# clean shutdown handler\ndef main(): pass\nEOF",
    "echo hello << 'EOF'\nreboot is a system command\nEOF",
    "cat << EOF\npoweroff and halt are dangerous\nEOF",
])
def test_heredoc_content_not_blocked(cmd):
    """Heredoc-Inhalt ist Daten, kein Shell-Code → keine Blockierung."""
    assert check_command(cmd) is None


# Aber: "shutdown" im eigentlichen Befehlsteil (vor <<) WIRD blockiert:
def test_shutdown_before_heredoc_still_blocked():
    assert check_command("shutdown -h now << /dev/null") is not None


# Einfacher Fall ohne Heredoc bleibt unverändert:
def test_plain_shutdown_still_blocked():
    hit = check_command("shutdown -h now")
    assert hit is not None and hit[0] == "hard"
