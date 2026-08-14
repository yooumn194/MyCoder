"""execute_in_sandbox: shell execution behind an isolated sandbox.

Replaces the old `bash` tool (tools/bash.py). The regex gate there was a
*blacklist* — trivially bypassable, the shell has infinitely many ways to be
dangerous. The execution pipeline now has three layers:

    1. hard pre-check   (_check_dangerous) — blocks catastrophic commands
    2. confirmation     (ConfirmPolicy)    — asks before risky-but-legal
       commands, mirroring Claude Code's permission system
    3. sandbox backend  (Docker, else the degraded local executor)

Layers 1 and 2 guard the host; layer 3 is the real containment. After a
successful run the tool reports which /workspace files changed (via docker
diff) instead of copying them out wholesale; sync_workspace() is the explicit
pull-back step, so npm-install-sized change sets never flood the host.
"""

import re

from ..sandbox import (
    ALLOW_RISKY_ENV,
    ConfirmPolicy,
    ExecutionResult,
    SandboxManager,
    run_async,
)
from ..sandbox.executor import set_active_manager
from ..sandbox.logger import get_logger
from ..sandbox.policy import ALTERNATIVE_HINTS
from .base import Tool
from .bash import _check_dangerous

logger = get_logger()

_MAX_OUTPUT = 15_000
_manager: SandboxManager | None = None


def _get_manager() -> SandboxManager:
    global _manager
    if _manager is None:
        _manager = SandboxManager()
    set_active_manager(_manager)
    return _manager


def _get_policy() -> ConfirmPolicy:
    """The confirmation policy lives on the manager (= one per session), so a
    fresh SandboxManager starts with an empty approval cache."""
    return _get_manager().policy

# Deletion-class commands get an extra hint in the tool output: the host won't
# mirror the deletion unless the agent explicitly calls sync_workspace(clean=True).
_DELETE_COMMAND_RE = re.compile(r"\b(rm\s|git\s+clean|rmdir\s)")


class ExecuteInSandboxTool(Tool):
    name = "execute_in_sandbox"
    idempotent = False  # arbitrary commands may have non-idempotent side effects
    description = (
        "Run a shell command in an isolated Docker sandbox and return stdout, "
        "stderr, exit code. Isolation: no network, read-only root, non-root, "
        "zero capabilities, memory/CPU/pids limits, hard timeout. Writes land "
        "in /workspace — call sync_workspace() to pull them back. Risky "
        "commands (network, installs, git push, recursive rm) ask for "
        "confirmation first. Use for tests, git, scripts."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to run",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 30)",
                "minimum": 1,
                "maximum": 600,
            },
        },
        "required": ["command"],
    }

    def execute(self, command: str, timeout: int = 30) -> str:
        timeout = min(max(int(timeout), 1), 600)

        # Cheap first-line gate, defense in depth: catch obvious self-destruct
        # commands before they cost a container cycle. The sandbox contains
        # whatever slips past. Note this also protects the degraded local path,
        # where a regex is all the protection we have.
        warning = _check_dangerous(command)
        if warning:
            logger.warning(
                "sandbox.block",
                reason=warning,
                command=_truncate(command),
            )
            return f"⚠ Blocked: {warning}\nCommand: {command}"

        # Permission-style gate: risky-but-legal commands ask the operator
        # first (session-cached, env-overridable, fail-closed). Applied here —
        # at the tool boundary — so it guards BOTH backends: the Docker sandbox
        # and, more importantly, the degraded local executor that can reach the
        # host network.
        allowed, rule = run_async(_get_policy().decide(command))
        if not allowed:
            hint = ALTERNATIVE_HINTS.get(rule.category, "请调整命令")
            return (
                f"⚠ Cancelled: {rule.reason}\n"
                f"Command: {command}\n"
                f"替代方案: {hint}\n"
                f"不要重试相同命令。若确需执行，请设置 {ALLOW_RISKY_ENV}=1 或调整命令。"
            )

        try:
            result = run_async(_get_manager().execute(command, timeout))
        except Exception as e:  # backend failure surfaces as a plain error
            return f"Error executing in sandbox: {e}"
        out = _format(result, command)
        if result.ok:
            out += _changed_files_suffix(command)
        return out


def _changed_files_suffix(command: str) -> str:
    """Which /workspace files changed (from docker diff), appended to output.

    Never copies files out — that is sync_workspace()'s job. The list is
    truncated to 50 entries with a total count so an npm-install-sized change
    set doesn't flood the context.
    """
    sync = _get_manager().get_sync()
    suffix = ""
    if sync is not None:
        try:
            changed, truncated, total = run_async(sync.diff_changed_files())
        except Exception:
            changed, truncated, total = [], False, 0
        if changed:
            suffix += "\n[changed files: " + ", ".join(changed) + "]"
            if truncated:
                suffix += (
                    f"\n[total {total} files changed; list truncated. "
                    f"Call sync_workspace() to sync all of them.]"
                )
    if _DELETE_COMMAND_RE.search(command):
        suffix += (
            "\n[files deleted in sandbox. To mirror deletions on the host, "
            "call sync_workspace(clean=True).]"
        )
    return suffix


def _format(result: ExecutionResult, command: str) -> str:
    if result.blocked:
        return (
            f"⚠ Blocked: {result.block_reason or 'unknown reason'}\n"
            f"Command: {command}"
        )
    out = result.stdout
    if result.stderr:
        out += f"\n[stderr]\n{result.stderr}"
    if result.timed_out:
        out += "\n[timed out]"
    elif result.exit_code != 0:
        out += f"\n[exit code: {result.exit_code}]"
    if len(out) > _MAX_OUTPUT:
        out = (
            out[:6000]
            + f"\n\n... truncated ({len(out)} chars total) ...\n\n"
            + out[-3000:]
        )
    return out.strip() or "(no output)"


def _truncate(text: str, limit: int = 256) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
