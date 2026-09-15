"""Guardrails: hard security gates for shell commands (code, not prompt).

Two categories:
- **HARD-BLOCK**: Commands that are always aborted (no ``confirm``
  helps). Examples: disk formatting, ``rm -rf /``, fork bombs.
- **CONFIRM**: Destructive, but possible in legitimate scenarios (e.g.
  ``rm -rf ./build``). These are blocked UNTIL the agent has explicitly
  asked the user and calls the tool again with ``confirm=True``.

Design principle: prompt rules are soft (the model can ignore them),
code gates are hard. The gate returns an error message that instructs
the model to ask the user — so the confirmation stays visible in the
chat history and traceable.
"""

from __future__ import annotations

import re

# (pattern, short reason). Patterns are checked against the COMPLETE command
# (incl. pipes/semicolons) — deliberately coarse, since false positives
# (wrongly blocked) are less bad than a false negative.
_HARD_BLOCK: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bmkfs(\.\w+)?\b"), "filesystem formatting (mkfs)"),
    (re.compile(r"\bdd\b[^|;&]*\bof=/dev/"), "dd directly on a block device"),
    (re.compile(r"\b(fdisk|parted|wipefs)\b\s+(/dev/|-\s*/dev/)"), "partitioning/deletion on /dev/*"),
    (re.compile(r"\brm\s+(-[a-zA-Z]*[rR][a-zA-Z]*\s+)+(/|~|\$HOME)(/|\s|$)"), "rm -rf on /, ~ or $HOME"),
    (re.compile(r":\(\)\s*\{"), "fork bomb pattern (:(){...})"),
    (re.compile(r"\b(shutdown|reboot|poweroff|halt)\b"), "system shutdown/restart"),
    (re.compile(r">\s*/dev/(sd|nvme|hd|vd|mmcblk)"), "writing directly to a block device"),
    (re.compile(r"\bchmod\s+(-R\s+)?0?777\s+/"), "chmod 777 on /"),
    (re.compile(r"\buserdel\b"), "user deletion"),
    (re.compile(r"\bcrontab\s+-r\b"), "cron job deletion"),
    (re.compile(r"\bhistory\s+-c\b"), "shell history clearing (forensic trace)"),
    (re.compile(r"\bshred\b\s+(-[a-zA-Z]+\s+)*(/dev/|~|\$HOME)"), "shred on /dev/*, ~ or $HOME"),
]

_CONFIRM: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\brm\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*[rR]"), "rm with -r/-R (recursive deletion)"),
    (re.compile(r"\brm\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*[fF]"), "rm with -f"),
    (re.compile(r"\bfind\b[^|;&]*\s-delete\b"), "find -delete"),
    (re.compile(r"\btruncate\s+(-[a-zA-Z]+\s+)*-s\b"), "truncate (cutting files)"),
    (re.compile(r"\bdd\b\s*(if=|of=|bs=|count=|status=|conv=|seek=|skip=|iflag=|oflag=)"), "dd (block-level copy)"),
    (re.compile(r"\bchmod\s+-R\b"), "chmod -R (recursive permission change)"),
    (re.compile(r"\bchown\s+-R\b"), "chown -R (recursive ownership change)"),
    (re.compile(r"\bmv\b[^|;&]*\s*/dev/null"), "mv to /dev/null (irreversible)"),
    (re.compile(r"\bkillall\b|\bpkill\b"), "process kill (killall/pkill)"),
    (re.compile(r"\bsystemctl\s+(stop|restart|disable|mask)\b"), "systemctl stop/restart/disable"),
    (re.compile(r"\biptables\s+-F\b"), "iptables flush (delete all rules)"),
    (re.compile(r"\bdropdb\b|\bDROP\s+(TABLE|DATABASE)\b"), "database object deletion"),
    (re.compile(r"\bgit\s+reset\s+--hard\b"), "git reset --hard (uncommitted changes gone)"),
    (re.compile(r"\bgit\s+clean\b"), "git clean (untracked files gone)"),
]


def _strip_heredoc(cmd: str) -> str:
    """Removes heredoc content (after ``<<``) — that is data, not shell code.

    Example: ``cat > file << 'EOF'\\nshutdown code here\\nEOF``
    → only ``cat > file`` is checked.
    """
    idx = cmd.find("<<")
    if idx == -1:
        return cmd
    return cmd[:idx].strip()


def check_command(command: str) -> tuple[str, str] | None:
    """Checks a shell command against the guardrail rules.

    Returns: ``None`` if the command is allowed, otherwise
    ``(category, reason)`` with ``category`` ∈ {"hard", "confirm"}.

    Heredoc content (after ``<<``) is ignored since it is data,
    not executable shell code.
    """
    cmd = command.strip()
    if not cmd:
        return None
    check_target = _strip_heredoc(cmd)
    for pattern, reason in _HARD_BLOCK:
        if pattern.search(check_target):
            return ("hard", reason)
    for pattern, reason in _CONFIRM:
        if pattern.search(check_target):
            return ("confirm", reason)
    return None
