"""sync_workspace: pull /workspace changes back to the host project directory.

The sandbox's /workspace volume is invisible to the host-side file tools.
execute_in_sandbox only *reports* changed files; this tool is the explicit
pull-back step the agent calls when it needs the sandbox's output on the host.
"""

from ..sandbox import run_async
from ..sandbox.logger import get_logger
from .base import Tool
from .sandbox_tool import _get_manager

logger = get_logger()


class ToolResult(str):
    """A tool result that is a string (what the LLM sees) and carries the
    structured fields code needs (synced_files, status) — a reconciliation
    between the spec's ToolResult fields and the project's string tool
    contract (tools/base.py: Tool.execute -> str)."""

    def __new__(cls, text: str, *, status: str = "ok", synced_files=None):
        obj = super().__new__(cls, text)
        obj.status = status
        obj.synced_files = list(synced_files or [])
        return obj


class SyncWorkspaceTool(Tool):
    name = "sync_workspace"
    description = (
        "Sync files the sandbox changed back from /workspace to the host "
        "project directory. Call it after execute_in_sandbox creates or "
        "modifies files you need to read or edit on the host. Pass "
        "clean=True to also delete host files that were deleted inside the "
        "sandbox (rsync --delete semantics)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "clean": {
                "type": "boolean",
                "description": (
                    "Also delete host files that no longer exist in the "
                    "sandbox workspace (default false)"
                ),
            },
        },
        "required": [],
    }

    def execute(self, clean: bool = False) -> ToolResult:
        sync = _get_manager().get_sync()
        if sync is None:
            # fail-closed: without a live Docker sandbox there is nothing to
            # sync, and we say so rather than silently succeeding.
            return ToolResult(
                "Error: no active Docker sandbox to sync. Sandboxed execution "
                "is unavailable (or running in degraded local mode).",
                status="error",
            )
        try:
            files = run_async(sync.copy_out(clean=bool(clean)))
        except Exception as e:
            return ToolResult(
                f"Error syncing workspace: {e}",
                status="error",
            )
        if clean:
            logger.info("sandbox.sync_clean", files_synced=len(files))
        summary = ", ".join(files) if files else "(no changes)"
        return ToolResult(
            f"Synced {len(files)} file(s): {summary}",
            status="ok",
            synced_files=files,
        )
