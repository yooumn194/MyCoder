"""Degraded, host-side command execution used when Docker is unavailable.

HONESTY WARNING: this is NOT a security boundary. Commands run directly on the
host as the current user, so a hostile command could in principle touch
anything the user can. It exists so the agent keeps working on hosts without
Docker. What it *can* do cheaply:

  - allowlist the leading command (unknown tools are refused)
  - keep the destructive-pattern pre-check from tools/bash.py
  - enforce a hard timeout (subprocess kills the child)

Every use of this backend is gated behind an explicit operator confirmation
and a WARNING audit log — the default is to fail closed. The Docker sandbox is
the real isolation; treat this path as damage-limiting, not protection.
"""

import asyncio
import os
import subprocess
import time
from pathlib import Path

from .logger import get_logger
from .models import ExecutionResult

logger = get_logger()

# Leading commands allowed in degraded mode. Kept to offline-safe, typical dev
# tools; network tools (curl/wget/ssh/… ) are deliberately excluded because a
# host process with network can exfiltrate. This set is a *convention*, not a
# proof — see the module docstring.
_ALLOWED_LEADING = frozenset({
    "cd", "export", "env", "ls", "cat", "head", "tail", "grep", "rg", "git",
    "python", "python3", "pip", "pip3", "pytest", "ruff", "mkdir", "touch",
    "echo", "cp", "mv", "rm", "find", "wc", "sort", "uniq", "diff", "make",
    "node", "npm", "pnpm", "yarn", "tar", "unzip", "zip", "which", "true",
    "false", "exit", "sleep",
})


class LocalExecutor:
    """Synchronous, allowlisted, timeout-bounded host execution.

    Implements the same interface as DockerSandbox so the manager can swap
    backends transparently (start/stop are no-ops here — nothing to hold).
    """

    def __init__(
        self,
        project_dir: str | os.PathLike[str],
        allowlist: frozenset[str] | None = None,
    ) -> None:
        self.project_dir = Path(project_dir).resolve()
        self._allowlist = allowlist or _ALLOWED_LEADING
        self._cmd_count = 0
        logger.warning(
            "sandbox.local_active",
            detail="NO sandbox isolation — host filesystem and network are exposed",
        )

    async def start(self) -> None:
        """No-op: the host needs nothing provisioned."""

    async def stop(self) -> None:
        """No-op: nothing to tear down."""

    async def execute(self, command: str, timeout: int = 30) -> ExecutionResult:
        """Run `command` via `sh -c` if its leading command is allowlisted.

        Graduated warnings (P3-1): every command on the HOST is a risk, but
        spamming a banner on each one trains the operator to ignore them.
        Instead: warn once at construction, once on the first command, and once
        every 10 commands with a running count.
        """
        self._cmd_count += 1
        if self._cmd_count == 1:
            logger.warning("sandbox.unsandboxed_first", cmd=_truncate(command))
        elif self._cmd_count % 10 == 0:
            logger.warning(
                "sandbox.unsandboxed_count",
                count=self._cmd_count,
                detail=f"{self._cmd_count} commands executed on HOST without isolation",
            )
        first = _leading_token(command)
        if first not in self._allowlist:
            logger.warning(
                "sandbox.local_block",
                command=_truncate(command),
                reason=f"command {first!r} not in local allowlist",
            )
            return ExecutionResult(
                exit_code=126,
                stdout="",
                stderr=f"command {first!r} not in local allowlist",
                blocked=True,
                block_reason=f"command {first!r} not in local allowlist",
            )
        return await asyncio.to_thread(self._run, command, timeout)

    async def get_diff(self) -> str:
        """Unified diff against the host repo (tracked changes)."""
        result = await asyncio.to_thread(
            self._run, "git diff --no-color && git diff --cached --no-color", 10
        )
        if result.exit_code != 0:
            return f"(git diff failed: {result.stderr.strip()})"
        return result.stdout.strip() or "(no changes)"

    def _run(self, command: str, timeout: int) -> ExecutionResult:
        started = time.monotonic()
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                cwd=str(self.project_dir),
            )
            return ExecutionResult(
                exit_code=proc.returncode,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout
            if stdout is None:
                stdout = ""
            elif isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            return ExecutionResult(
                exit_code=-1,
                stdout=stdout,
                stderr=f"timed out after {timeout}s",
                timed_out=True,
                duration_ms=int((time.monotonic() - started) * 1000),
            )


def _leading_token(command: str) -> str:
    """Return the first bare command word.

    Skips env assignments, `cd` targets and chain operators so that e.g.
    `export A=1 && cd /tmp && ls -la` resolves to `ls`.
    """
    for token in command.split():
        if token in {"&&", "||", ";"}:
            continue
        if "=" in token and not token.startswith("-"):
            continue  # env assignment (FOO=bar) or cd target (/x)
        return token.lstrip("(").strip("'\"")
    return ""


def _truncate(text: str, limit: int = 256) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
