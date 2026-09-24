"""Shell tool: executes commands in the sandbox directory.

Stateful Shell
--------------
Since the stateful update, every command runs in ONE persistent bash
process (``bash --noprofile --norc``, stdin/stdout via pipes). This way
``cd``, ``export``/env vars, ``source`` and active aliases survive
across all tool calls of a session.

Mechanics: Per command, ``cmd </dev/null; echo __MARKER__<n> $?`` is
written. The marker echo (unique sequence number) marks the end of
output — so the agent can reliably read the output, even if the command
itself prints lines with ``[exit_code:``.

``reset=true`` kills and restarts the shell (e.g. after a poisoned
environment variable). Sudo commands still run in a separate process
(Askpass mechanism, see below).

Sudo Handling
-------------
Commands containing ``sudo`` are detected. If the process is not running
as root, the password is requested via callback (``sudo_ask``) —
once per session, then cached in RAM only. The password is never written
to logs, history, or tool output.

IMPORTANT: ``sudo_ask`` is an **async** callback (``async def cb(command)
-> Optional[str]``). The agent loop runs in an active asyncio event
loop — a synchronous prompt_toolkit ``prompt()`` call inside it would
crash (``asyncio.run()`` in running loop) or hang. Therefore the
callback is called with ``await``; the REPL uses
``prompt_toolkit``'s ``prompt_async()`` for this.

Mechanics: ``sudo -A`` + Askpass script. The script reads the password
from a 0600 file in a 0700 tempdir (no plaintext in the script itself,
no shell-quoting issues). Wrong password → clear cache, ask callback
again, max 3 attempts.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from ..tools import Tool, ToolResult
from .guardrails import check_command

log = logging.getLogger(__name__)

_SUDO_MAX_ATTEMPTS = 3
# Passwort-Abfrage darf so lange auf den Nutzer warten — danach sauber
# abbrechen (sonst hängt der Agent ewig am Spinner).
_SUDO_ASK_TIMEOUT = 60.0
# Bestätigungs-Abfrage darf so lange auf den Nutzer warten — danach wird
# der Befehl blockiert (sicherer Default).
_CONFIRM_ASK_TIMEOUT = 30.0
# ulimit-Werte für den persistenten bash-Prozess (Resource-Limits gegen
# runaway-Prozesse: 600s CPU, 2 GB RAM, 1 GB Datei, 1024 FDs).
_ULIMITS = "-t 600 -v 2097152 -f 1048576 -n 1024"
# bwrap-Sandbox: aktiv, wenn bwrap installiert ist. Bindet die
# Sandbox-Roots + Systemverzeichnisse (read-only) und tmpfs-t /tmp.
_HAS_BWRAP = shutil.which("bwrap") is not None
# Marker-Präfix; die Sequenznummer macht jeden Marker eindeutig,
# damit keine fremde Ausgabe versehentlich als Ende erkannt wird.
_MARKER = "___CHATCLI_MARKER___"


def _needs_sudo(command: str) -> bool:
    """Erkennt, ob irgendeine Segment des Befehls mit sudo beginnt.

    Segmente = Aufteilung an ``;``, ``&&``, ``||``, ``|``. Env-Zuweisungen
    vor dem Befehl (z. B. ``FOO=bar sudo …``) werden übersprungen.
    """
    for segment in command.replace("&&", ";").replace("||", ";").split(";"):
        # In jedem Pipeline-Abschnitt prüfen (sudo kann in jeder Stufe stehen)
        for part in segment.split("|"):
            part = part.strip()
            if not part:
                continue
            tokens = part.split()
            for tok in tokens:
                if tok == "sudo":
                    return True
                if "=" in tok and not tok.startswith("-"):
                    continue  # Env-Variablen-Zuweisung
                break  # erstes nicht-Env-Token = der eigentliche Befehl
    return False


def _sudo_force_prompt(command: str) -> bool:
    """True, wenn der Aufruf das Prompt-Verhalten explizit erzwingt.

    Wichtig: ``sudo -k`` invalidiert den sudo-Timestamp und setzt dadurch
    sogar unter root wieder einen Passwort-Dialog durch — genau der von den
    Regressionstests abgedeckte Fall.
    """
    for segment in command.replace("&&", ";").replace("||", ";").split(";"):
        for part in segment.split("|"):
            part = part.strip()
            if not part:
                continue
            try:
                tokens = shlex.split(part)
            except ValueError:
                tokens = part.split()
            for i, tok in enumerate(tokens):
                if tok == "sudo":
                    if any(opt in ("-k", "-S", "-s") for opt in tokens[i + 1 : i + 6]):
                        return True
                    return False
                if "=" in tok and not tok.startswith("-"):
                    continue
                break
    return False


class ShellTool(Tool):
    name = "shell"
    description = (
        "Executes a shell command and returns stdout/stderr/exit_code. "
        "The shell is STATEFUL: cd, export and env vars persist across "
        "multiple calls. reset=true starts a fresh shell. "
        "Commands with 'sudo' are automatically executed with password prompt. "
        "Destructive commands (rm -r, dd, systemctl stop ...) are blocked "
        "until you have explicitly asked the user — then call again with confirm=true.\n"
        "Examples:\n"
        "  shell(command='ls -la')\n"
        "  shell(command='grep -rn TODO src/', timeout=15)\n"
        "  shell(command='rm -rf ./build', confirm=true)   # ONLY after user confirmation"
    )
    args_schema: dict[str, Any] = {
        "command": "str — the command to execute (e.g. 'ls -la')",
        "cwd": "str (optional) — working directory (relative to project root)",
        "timeout": "int (optional, default 30) — timeout in seconds",
        "reset": "bool (optional, default false) — restart shell session",
        "confirm": "bool (optional, default false) — true ONLY if the user has just explicitly confirmed the destructive command",
    }

    def __init__(
        self,
        cwd: str = ".",
        sudo_ask: Optional[Callable[[str], Any]] = None,
        confirm_ask: Optional[Callable[[str, str], Any]] = None,
        persistent: bool = True,
        sandbox: bool = True,
        sudo_ask_timeout: float = _SUDO_ASK_TIMEOUT,
        confirm_ask_timeout: float = _CONFIRM_ASK_TIMEOUT,
        on_line: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._base = Path(cwd).resolve()
        self._sudo_ask = sudo_ask
        self._confirm_ask = confirm_ask
        self._sudo_ask_timeout = sudo_ask_timeout
        self._confirm_ask_timeout = confirm_ask_timeout
        self._sudo_pw: Optional[str] = None
        self._sudo_ask_timed_out = False
        self._sudo_ask_error: Optional[str] = None
        self._persistent = persistent
        self._sandbox = sandbox and _HAS_BWRAP
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._lock = asyncio.Lock()
        self._seq = 0
        # CWD-Tracking: nach jedem Befehl wird das aktuelle pwd aus der
        # Marker-Zeile gelesen. Damit weiß der Loop, wo die Shell steht.
        self._tracked_cwd: Optional[str] = None
        # Live-Streaming: jede ausgegebene Zeile wird (synchro, fire-and-forget)
        # an on_line durchgereicht — die REPL rendert sie live in einer Box.
        self._on_line = on_line

    @property
    def current_cwd(self) -> Optional[str]:
        """Current CWD of the persistent shell (after last command)."""
        if self._tracked_cwd is not None:
            return self._tracked_cwd
        return str(self._base)

    def _emit_line(self, line: str) -> None:
        """Eine Ausgabe-Zeile live an den Callback geben (darf nicht werfen)."""
        cb = self._on_line
        if cb is None:
            return
        try:
            cb(line)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Persistenter bash-Prozess
    # ------------------------------------------------------------------

    async def _probe_bwrap(self) -> None:
        """Einmaliger Test, ob bwrap hier überhaupt läuft.

        In Containern/VMs ist das User-Namespace oft blockiert
        ("setting up uid map: Permission denied") — dann fällt die
        Sandbox für die Session still auf 'ohne bwrap' zurück.
        """
        if not self._sandbox:
            return
        bwrap = shutil.which("bwrap")
        try:
            proc = await asyncio.create_subprocess_exec(
                bwrap, "--ro-bind", "/", "/", "true",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=10)
            if proc.returncode != 0:
                self._sandbox = False
        except Exception:
            self._sandbox = False

    async def _ensure_proc(self, target: Path) -> asyncio.subprocess.Process:
        """Startet den persistenten bash-Prozess (falls nicht vorhanden).

        Mit Sandbox (bwrap): bash läuft in einem Mount-Namespace, das nur
        die Sandbox-Roots (les-/schreibbar), Systemverzeichnisse (read-only)
        und ein frisches /tmp sieht. Ohne bwrap (oder wenn bwrap hier
        nicht laufen kann): wie bisher.
        """
        if self._proc is not None and self._proc.returncode is None:
            return self._proc
        if self._sandbox:
            await self._probe_bwrap()
        await self._kill_proc()
        # LC_ALL=C: unterdrückt die bash-Locale-Warnung (fehlende
        # en_US.UTF-8-Locale) in der Prozess-Umgebung — der Rest der
        # Umgebung bleibt erhalten.
        env = {**os.environ, "LC_ALL": "C"}
        if self._sandbox:
            # bwrap: Mount-Namespace mit bind-mounted Roots.
            # - Sandbox-Root: les-/schreibbar
            # - /usr /bin /lib* /etc /opt /srv: read-only
            # - /tmp: frisches tmpfs, /dev: minimales dev
            # - die-with-parent: Kind-Prozesse sterben mit der Shell
            # - ulimit: Resource-Limits (CPU/RAM/Datei/FDs)
            bwrap = shutil.which("bwrap")
            argv = [
                bwrap,
                "--bind", str(self._base), str(self._base),
            ]
            for ro in ("/usr", "/bin", "/lib", "/lib64", "/etc", "/opt", "/srv", "/var"):
                if Path(ro).is_dir():
                    argv += ["--ro-bind", ro, ro]
            # /proc (read-only): ohne diesen Bind sind ps/pgrep/pkill/fuser
            # in der Sandbox blind ("mount -t proc proc /proc") — der Agent
            # kann dann keine Host-Prozesse identifizieren oder räumen.
            if Path("/proc").is_dir():
                argv += ["--ro-bind", "/proc", "/proc"]
            argv += [
                "--tmpfs", "/tmp",
                "--dev", "/dev",
                "--die-with-parent",
                # bash -c 'ulimit …; exec bash …': die Limits gelten für
                # die ganze Session, der innere bash liest stdin weiter.
                "bash", "--noprofile", "--norc", "-c",
                f"ulimit {_ULIMITS} 2>/dev/null; exec bash --noprofile --norc",
            ]
        else:
            argv = ["bash", "--noprofile", "--norc"]
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(target),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self._proc = proc
        return proc

    async def _kill_proc(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            await proc.wait()
        except Exception:
            pass
        # Pipes explizit schließen, damit der BaseSubprocessTransport
        # NICHT erst per GC nach Loop-Ende aufgeräumt wird — sonst
        # wirft __del__ RuntimeError("Event loop is closed").
        # WICHTIG: stdout muss auch geschlossen werden (stdin allein reicht
        # nicht — die stdout-Pipe hält den Transport am Leben).
        for pipe in (proc.stdin, proc.stdout):
            if pipe is not None:
                try:
                    pipe.close()
                except Exception:
                    pass

    async def _run_persistent(
        self, command: str, target: Path, timeout: int
    ) -> ToolResult:
        """Executes the command in the persistent bash process.

        For each command, a unique marker + exit code is written after the
        output; the stdin-EOF protection redirect (``</dev/null``) prevents
        interactive commands (cat, read) from hanging on the next input line.
        """
        async with self._lock:
            self._seq += 1
            marker = f"{_MARKER}{self._seq}"
            try:
                proc = await self._ensure_proc(target)
                # </dev/null: nur für diesen einen Befehl, nicht für die
                # Shell selbst (sonst würde cat & Co. die nächste
                # Befehlszeile fressen). Die Marker-Zeile wird ECHOt,
                # sonst würde bash sie als (nicht vorhandenes) Kommando
                # ausführen.
                # WICHTIG: abschließendes \n — ohne Zeilenumbruch würde
                # bash die letzte Zeile nicht ausführen.
                # pwd in der Marker-Zeile: ermöglicht CWD-Tracking ohne
                # extra Round-Trip (cd bleibt in der stateful Shell erhalten).
                payload = f"{command} </dev/null\necho {marker} $? $(pwd)\n"
                if proc.stdin is None or proc.stdin.is_closing():
                    return ToolResult(
                        ok=False, output="",
                        error="Shell process has crashed — please try again.",
                    )
                proc.stdin.write(payload.encode("utf-8", "replace"))
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError, OSError):
                await self._kill_proc()
                return ToolResult(
                    ok=False, output="",
                    error="Shell process has crashed — please try again.",
                )

            out_lines: list[str] = []
            code: Optional[int] = None
            try:
                async def _read_until_marker() -> None:
                    nonlocal code
                    assert proc.stdout is not None  # PIPE garantiert StreamReader
                    while True:
                        line = await proc.stdout.readline()
                        if not line:
                            # EOF: Prozess ist weg.
                            await proc.wait()
                            return
                        text = line.decode("utf-8", "replace").rstrip("\n")
                        if text == marker:
                            return
                        if text.startswith(marker + " "):
                            rest = text[len(marker) + 1:].strip()
                            parts = rest.split(" ", 1)
                            try:
                                code = int(parts[0])
                            except (ValueError, IndexError):
                                code = -1
                            # CWD aus der Marker-Zeile (zweite Komponente)
                            if len(parts) > 1 and parts[1]:
                                self._tracked_cwd = parts[1]
                            return
                        out_lines.append(text)
                        # Live: Zeile sofort an die REPL durchreichen.
                        self._emit_line(text)

                await asyncio.wait_for(_read_until_marker(), timeout=timeout)
            except asyncio.TimeoutError:
                await self._kill_proc()
                return ToolResult(
                    ok=False, output="",
                    error=f"Timeout after {timeout}s. Command was aborted.",
                )

            if code is None:
                # EOF ohne Marker (Prozess ist zwischenzeitlich gestorben)
                await self._kill_proc()
                rc = proc.returncode if proc.returncode is not None else -1
                return ToolResult(
                    ok=False,
                    output=_format_output("\n".join(out_lines), "", rc, reason="EOF without marker — process died"),
                    error="Shell process terminated during execution.",
                )
            return ToolResult(
                ok=code == 0,
                output=_format_output("\n".join(out_lines), "", code),
            )

    # ------------------------------------------------------------------
    # Sudo-Hilfen
    # ------------------------------------------------------------------

    async def _ask_password(self, hint: str) -> Optional[str]:
        """Passwort aus RAM-Cache holen oder per (async) Callback abfragen.

        Mit Timeout: wartet der Nutzer zu lange, bricht die Abfrage
        sauber ab (``_sudo_ask_timed_out``), statt den Agent ewig zu
        blockieren.
        """
        if self._sudo_pw is not None:
            return self._sudo_pw
        if self._sudo_ask is None:
            return None
        try:
            pw = await asyncio.wait_for(
                self._sudo_ask(hint), timeout=self._sudo_ask_timeout
            )
        except asyncio.TimeoutError:
            self._sudo_ask_timed_out = True
            return None
        except (EOFError, KeyboardInterrupt):
            return None
        except Exception as exc:
            log.warning(
                "sudo-Callback-Fehler (nicht Timeout/EOF): %s", exc,
            )
            self._sudo_ask_error = str(exc)
            return None
        if not pw:
            return None
        self._sudo_pw = pw
        return pw

    @staticmethod
    def _make_askpass(pw_file: str) -> str:
        """Askpass-Skript, das das Passwort aus der 0600-Datei liest."""
        return f"#!/bin/sh\ncat '{pw_file}'\n"

    async def _sudo_available_without_pw(self) -> bool:
        """sudo -n true: geht sudo ohne Passwort (NOPASSWD / frischer Cache)?"""
        try:
            proc = await asyncio.create_subprocess_exec(
                "sudo", "-n", "true",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            return proc.returncode == 0
        except Exception:
            return False

    async def _run_sudo(self, command: str, target: Path, timeout: int, force_prompt: bool = False) -> ToolResult:
        # 1) Passwortfrei möglich? (NOPASSWD oder noch gültiger sudo-Cache)
        if not force_prompt and await self._sudo_available_without_pw():
            return await self._run_plain(command, target, timeout)

        last_err = ""
        for attempt in range(_SUDO_MAX_ATTEMPTS):
            pw = await self._ask_password(command)
            if pw is None:
                break

            tmpdir = tempfile.mkdtemp(prefix="chatcli-sudo-")
            pw_file = os.path.join(tmpdir, "pw")
            askpass = os.path.join(tmpdir, "askpass")
            try:
                with open(pw_file, "w", encoding="utf-8") as f:
                    f.write(pw + "\n")
                os.chmod(pw_file, stat.S_IRUSR | stat.S_IWUSR)
                with open(askpass, "w", encoding="utf-8") as f:
                    f.write(self._make_askpass(pw_file))
                os.chmod(askpass, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
                os.chmod(tmpdir, stat.S_IRWXU)

                env = {**os.environ, "SUDO_ASKPASS": askpass}
                # Das erste 'sudo'-Token aus dem Befehl entfernen und mit
                # 'sudo -A' (Askpass) neu prefixen — sonst doppeltes sudo.
                try:
                    proc = await asyncio.create_subprocess_shell(
                        "sudo -A -p '' " + _strip_sudo(command),
                        cwd=str(target),
                        env=env,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                except (OSError, ValueError) as exc:
                    last_err = str(exc)
                    break
                try:
                    out_b, err_b = await asyncio.wait_for(
                        proc.communicate(), timeout=timeout
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    try:
                        await proc.wait()
                    except Exception:
                        pass
                    return ToolResult(
                        ok=False, output="",
                        error=f"Timeout after {timeout}s. Command was aborted.",
                    )
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

            stdout = out_b.decode(errors="replace")
            stderr = err_b.decode(errors="replace")

            if proc.returncode == 0:
                # Passwort war korrekt → bleibt im RAM-Cache,
                # der sudo-Timestamp ist jetzt warm.
                return ToolResult(
                    ok=True,
                    output=_format_output(stdout, stderr, proc.returncode),
                )

            last_err = stderr.strip()
            # Das (vermutlich falsche) Passwort darf nach einem Fehlschlag
            # NIEMALS im Cache bleiben — sonst würde es blind wiederverwendet.
            self._sudo_pw = None
            if "password" in stderr.lower() or "try again" in stderr.lower():
                continue
            break

        if not last_err:
            if self._sudo_ask_timed_out:
                return ToolResult(
                    ok=False, output="",
                    error=(
                        f"⏰ sudo password prompt timed out after "
                        f"{int(self._sudo_ask_timeout)}s (no input). "
                        "Command not executed — please try again "
                        "when you are at the terminal."
                    ),
                )
            if self._sudo_ask_error:
                return ToolResult(
                    ok=False, output="",
                    error=(
                        f"sudo password prompt failed: "
                        f"{self._sudo_ask_error[:200]} "
                        "(Command not executed.)"
                    ),
                )
            return ToolResult(
                ok=False, output="",
                error=(
                    "Could not prompt for sudo password "
                    "(no interactive mode). Command not executed."
                ),
            )
        return ToolResult(
            ok=False,
            output=_format_output("", last_err, -1, reason="sudo error (password rejected or unavailable)"),
            error=(
                f"sudo failed: {last_err[:300]}"
                + _missing_service_hint(last_err)
            ),
        )

    # ------------------------------------------------------------------
    # Confirm-Abfrage (Destructive-Befehle)
    # ------------------------------------------------------------------

    # Antworten, die als Zustimmung gewertet werden (String-Normalisierung).
    _CONFIRM_YES = {"j", "y", "ja", "yes", "true", "1", "ok"}

    async def _ask_confirm(self, command: str, reason: str) -> Optional[bool]:
        """Frage den Nutzer direkt per Callback, ob ein destruktiver Befehl
        ausgeführt werden darf.

        Rückgabe: ``True`` = zustimmt, ``False`` = explizit abgelehnt,
        ``None`` = Abfrage nicht möglich (kein Callback, Timeout, Fehler).
        String-Antworten werden normalisiert — ``bool("nein")`` wäre
        sonst (fälschlich) True.
        """
        if self._confirm_ask is None:
            return None
        try:
            result = await asyncio.wait_for(
                self._confirm_ask(command, reason),
                timeout=self._confirm_ask_timeout,
            )
        except asyncio.TimeoutError:
            log.warning("confirm-Abfrage: Timeout nach %ss", int(self._confirm_ask_timeout))
            return None
        except (EOFError, KeyboardInterrupt):
            return None
        except Exception as exc:
            log.warning("confirm-Callback-Fehler: %s", exc)
            return None
        if isinstance(result, str):
            return result.strip().lower() in self._CONFIRM_YES
        return bool(result)

    # ------------------------------------------------------------------

    async def _run_plain(self, command: str, target: Path, timeout: int) -> ToolResult:
        """Normaler (nicht-sudo-)Pfad, inkl. timeout + Kill."""
        proc = None
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(target),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_chunks: list[bytes] = []
            stderr_b = b""
            async def _pump() -> None:
                nonlocal stderr_b
                assert proc is not None
                assert proc.stdout is not None  # PIPE garantiert StreamReader
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    stdout_chunks.append(line)
                    self._emit_line(line.decode("utf-8", "replace").rstrip("\n"))
                assert proc.stderr is not None
                stderr_b = await proc.stderr.read()
                # wait() NUR wenn der Transport den Child noch nicht
                # reaped hat. Normalerweise ist nach EOF auf beiden Pipes
                # der Child längst weg und returncode gesetzt — ein
                # zweites wait() liest den Exit-Status doppelt und
                # triggert "exit status already read" (asyncio-WARNING).
                if proc.returncode is None:
                    await proc.wait()
            await asyncio.wait_for(_pump(), timeout=timeout)
            stdout_b = b"".join(stdout_chunks)
        except asyncio.TimeoutError:
            # Subprozess explizit beenden — wait_for cancelt nur die
            # communicate()-Task, der Child würde sonst endlos weiterlaufen.
            if proc:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    await proc.wait()
                except Exception:
                    pass
            return ToolResult(
                ok=False, output="",
                error=f"Timeout after {timeout}s. Command was aborted.",
            )
        except Exception as e:
            return ToolResult(ok=False, output="", error=str(e))

        stdout = stdout_b.decode(errors="replace")
        stderr = stderr_b.decode(errors="replace")
        return ToolResult(
            ok=proc.returncode == 0,
            output=_format_output(stdout, stderr, proc.returncode),
        )

    async def run(self, **kwargs: Any) -> ToolResult:
        command: str = kwargs.get("command", "")
        rel_cwd: str = kwargs.get("cwd", ".")
        if not command:
            return ToolResult(ok=False, output="", error="No 'command' specified.")

        # Guardrails: harte Gates (Code, nicht Prompt).
        hit = check_command(command)
        if hit:
            category, reason = hit
            if category == "hard":
                return ToolResult(
                    ok=False, output="",
                    error=(
                        f"BLOCKED (Guardrail): {reason}. "
                        "This command will never be executed."
                    ),
                )
            if not kwargs.get("confirm"):
                # Direkter Code-Gate-Pfad: Nutzer wird per Callback gefragt.
                confirmed = await self._ask_confirm(command, reason)
                if confirmed is None:
                    return ToolResult(
                        ok=False, output="",
                        error=(
                            f"Confirmation required (Guardrail): {reason}. "
                            "Confirmation prompt not possible (no callback/"
                            "timeout). Command NOT executed."
                        ),
                    )
                if not confirmed:
                    return ToolResult(
                        ok=False, output="",
                        error=(
                            f"REJECTED (Guardrail): {reason}. "
                            "The user rejected execution. "
                            "Command NOT executed."
                        ),
                    )
                # Bestätigt → weiter (confirm=True ist implizit gesetzt).
        try:
            timeout: int = int(kwargs.get("timeout", 30))
        except (ValueError, TypeError):
            return ToolResult(
                ok=False, output="",
                error="Invalid value for 'timeout' (must be a number).",
            )
        reset = bool(kwargs.get("reset", False))

        # Sandbox: nur unter _base erlauben
        target = (self._base / rel_cwd).resolve()
        try:
            target.relative_to(self._base)
        except ValueError:
            return ToolResult(
                ok=False, output="",
                error=f"Access denied: '{rel_cwd}' is outside the sandbox.",
            )

        # --- Sudo-Pfad (eigener Prozess, Askpass-Mechanik) -------------
        if _needs_sudo(command) and (os.geteuid() != 0 or _sudo_force_prompt(command)):
            return await self._run_sudo(command, target, timeout, force_prompt=_sudo_force_prompt(command))

        if not self._persistent:
            return await self._run_plain(command, target, timeout)

        if reset:
            async with self._lock:
                await self._kill_proc()
            return await self._run_persistent(command, target, timeout)

        return await self._run_persistent(command, target, timeout)

    async def close(self) -> None:
        """Beendet den persistenten bash-Prozess (beim CLI-Ausgang)."""
        await self._kill_proc()


_MISSING_UNIT_RE = re.compile(
    r"unit\s+\S+.*could not be found|unit not found|"
    r"no such file|not found",
    re.IGNORECASE,
)


def _missing_service_hint(stderr: str) -> str:
    """Detects 'Unit not found' / 'No such file' in sudo stderr.

    Returns a learning hint (install path) so the agent doesn't
    abort but instead fixes the prerequisite (missing package).
    """
    if not _MISSING_UNIT_RE.search(stderr):
        return ""
    return (
        "\n[Learning hint: 'not found' = service/package missing. "
        "Pre-check: `command -v <binary>` / `systemctl list-unit-files | grep <name>`, "
        "then `sudo apt install -y <package>`, then restart + verify "
        "(check port/status)."
    )


def _strip_sudo(command: str) -> str:
    """Entfernt das erste 'sudo'-Token (sonst würde sudo -A sudo … laufen)."""
    tokens = command.split()
    for i, tok in enumerate(tokens):
        if tok == "sudo":
            tokens.pop(i)
            break
    return " ".join(tokens)


def _format_output(stdout: str, stderr: str, code: int, reason: str = "") -> str:
    parts = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"[stderr]\n{stderr}")
    suffix = f"[exit_code: {code}]" + (f" ({reason})" if reason else "")
    parts.append(suffix)
    return "\n".join(parts)
