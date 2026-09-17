"""execute_in_sandbox: shell execution behind an isolated sandbox.

Replaces the old `bash` tool (tools/bash.py). The regex gate there was a
*blacklist* — trivially bypassable, the shell has infinitely many ways to be
dangerous. The execution pipeline now has three layers:

    1. hard pre-check   (_check_dangerous) — blocks catastrophic commands
    2. confirmation     (ConfirmPolicy)    — asks before risky-but-legal
       commands, mirroring Claude Code's permission system
    3. sandbox backend  (Docker, else the degraded local executor)

Layers 1 and 2 guard the host; layer 3 is the real containment. Containment is
expressed as permissions, not as a second copy of the tree: the container
bind-mounts the host project directory read-write at /workspace, so the shell
and the file tools act on the same files. After a successful run the tool
reports which files changed, purely as information.
"""

import re

from ..patch_policy import patch_scope_violation
from ..sandbox import (
    ALLOW_RISKY_ENV,
    ConfirmPolicy,
    ExecutionResult,
    RestorePoint,
    SandboxManager,
    run_async,
)
from ..sandbox.executor import set_active_manager
from ..sandbox.logger import get_logger
from ..sandbox.policy import ALTERNATIVE_HINTS
from .base import Tool, ToolResult
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


def capture_session_restore_point() -> RestorePoint | None:
    """Record where the project stood before the first command of a session.

    P0-1 mounts the project directory itself into the container, so a
    destructive command reaches the real checkout — callers (the CLI, the API)
    invoke this at session start so recovery does not depend on noticing the
    damage in time.
    """
    return _get_manager().capture_restore_point()

# Deletion-class commands get an extra hint in the tool output: the deletion is
# already real on the host (one filesystem), so it shows up in `git status`
# immediately and can only be undone from the session's restore point.
_DELETE_COMMAND_RE = re.compile(
    r"(?:\brm\s|\bgit\s+clean|\brmdir\s|\bfind\b[^|;]*\s-delete\b|\bshred\b|"
    r"\bxargs\b[^|;]*\brm\b|"
    r"\b(?:shutil\.rmtree|os\.removedirs|os\.remove|os\.unlink)\s*\()"
)
# Categories whose denial still deserves a recovery hint: the model that just
# failed to destroy something may reach for a workaround next.
_DESTRUCTIVE_CATEGORIES = frozenset(
    {
        "recursive_delete",
        "git_rewrite",
        "workspace_overwrite",
        "workspace_overwrite_tracked",
    }
)


class ExecuteInSandboxTool(Tool):
    name = "execute_in_sandbox"
    idempotent = False  # arbitrary commands may have non-idempotent side effects
    description = (
        "Run a shell command in an isolated Docker sandbox and return stdout, "
        "stderr, exit code. Isolation: no network, read-only root, non-root, "
        "zero capabilities, memory/CPU/pids limits, hard timeout. The sandbox "
        "mounts the project directory itself at /workspace, so files written "
        "here are visible to read_file/edit_file immediately and vice versa "
        "(one filesystem, nothing to sync). Risky commands (network, installs, "
        "git push, recursive rm) ask for confirmation first. Use for tests, "
        "git, scripts."
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

    def __init__(self, manager: SandboxManager | None = None) -> None:
        self.manager = manager

    def _manager(self) -> SandboxManager:
        # Instance-bound managers are used by the API for tenant/workspace
        # isolation. CLI callers retain the historical process-global manager.
        return self.manager or _get_manager()

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
        allowed, rule = run_async(self._manager().policy.decide(command))
        if not allowed:
            hint = ALTERNATIVE_HINTS.get(rule.category, "请调整命令")
            recovery = (
                self._manager().recovery.hint()
                if rule.category in _DESTRUCTIVE_CATEGORIES
                else ""
            )
            return (
                f"⚠ Cancelled: {rule.reason}\n"
                f"Command: {command}\n"
                f"替代方案: {hint}\n"
                f"不要重试相同命令。若确需执行，请设置 {ALLOW_RISKY_ENV}=1 或调整命令。"
                + (f"\n{recovery}" if recovery else "")
            )

        manager = self._manager()
        try:
            result = run_async(manager.execute(command, timeout))
        except Exception as e:  # backend failure surfaces as a plain error
            return ToolResult(f"Error executing in sandbox: {e}", status="error")
        out = _format(result, command)
        # Report how the process ended as DATA, not as prose: the harness
        # decides "did the check pass?" from this field instead of guessing
        # from the command string (P0-4).
        status = "success"
        if result.blocked:
            status = "error"
        elif out.startswith("Error"):
            status = "error"
        if result.ok:
            if manager.benchmark_mode:
                try:
                    diff = run_async(manager.get_diff())
                    violation = patch_scope_violation(
                        diff,
                        project_root=manager.project_dir,
                        protect_benchmark_files=True,
                    )
                except Exception as exc:
                    return ToolResult(
                        f"Error: benchmark patch safety check failed closed: {exc}",
                        status="error",
                        exit_code=result.exit_code,
                    )
                if violation is not None:
                    return ToolResult(
                        f"Error: command left an unsafe benchmark patch: {violation}. "
                        "Undo the unintended file change before continuing.",
                        status="error",
                        exit_code=result.exit_code,
                    )
            out += _changed_files_suffix(command, manager=manager)
        return ToolResult(out, status=status, exit_code=result.exit_code)


def _changed_files_suffix(
    command: str, manager: SandboxManager | None = None
) -> str:
    """Which files changed, appended to the output.

    Purely informational — the changes are already on disk in the shared tree.
    The list is truncated to 50 entries with a total count so an
    npm-install-sized change set doesn't flood the context.
    """
    sync = (manager or _get_manager()).get_sync()
    suffix = ""
    if sync is not None:
        try:
            changed, truncated, total = run_async(sync.diff_changed_files())
        except Exception:
            changed, truncated, total = [], False, 0
        if changed:
            suffix += "\n[changed files: " + ", ".join(changed) + "]"
            if truncated:
                suffix += f"\n[total {total} files changed; list truncated.]"
    if _DELETE_COMMAND_RE.search(command):
        suffix += (
            "\n[files deleted. The deletion is already reflected in the "
            "working tree and in git status.]"
        )
        recovery = (manager or _get_manager()).recovery.hint()
        if recovery:
            suffix += f"\n{recovery}"
    else:
        # A successful overwrite/reset is just as capable of erasing the
        # pre-run state as rm.  Surface the session-scoped restore command in
        # the tool result so a model does not have to infer it from git status.
        active_manager = manager or _get_manager()
        policy = getattr(active_manager, "policy", None)
        rule = policy.check(command) if policy is not None else None
        if rule is not None and rule.category in _DESTRUCTIVE_CATEGORIES:
            recovery = active_manager.recovery.hint()
            if recovery:
                suffix += f"\n{recovery}"
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
