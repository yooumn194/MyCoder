"""Tool registry."""

from pathlib import Path

from ..sandbox import SandboxManager

from .read_file import ReadFileTool
from .file_state import FileStateTracker
from .write import WriteFileTool
from .edit import EditFileTool
from .glob_tool import GlobTool
from .grep import GrepTool
from .agent import AgentTool
from .fetch import FetchUrlTool
from .grep_search import GrepSearchTool
from .list_files import ListFilesTool
from .sandbox_tool import ExecuteInSandboxTool
from .subagent_tools import SpawnSubagentTool
from .todo_tools import TodoUpdateTool, TodoWriteTool
from .memory_tools import (
    MemoryConfirmTool,
    MemoryCorrectTool,
    MemoryForgetTool,
    MemoryListTool,
    MemorySaveTool,
    MemorySearchTool,
    MemoryStatsTool,
)


ALL_TOOLS = [
    # The old `bash` tool (regex-blacklist gated, runs on the host) is replaced
    # by the sandboxed executor: same contract, isolated by Docker, with a
    # user-confirmed local fallback. The sandbox bind-mounts the host project
    # directory at /workspace, so file tools and shell commands share ONE tree
    # and need no reconciliation step. grep_search / list_files are the
    # Phase 2 agentic-search tools (path-guarded, rg-first). todo_write /
    # todo_update are the Phase 3 planning tools. mycoder/agent.py only
    # gains a tiny guard + correction hook in _exec_tool (Phase 3 spec).
    ExecuteInSandboxTool(),
    GrepSearchTool(),
    ListFilesTool(),
    ReadFileTool(),
    WriteFileTool(),
    EditFileTool(),
    GlobTool(),
    GrepTool(),
    AgentTool(),
    FetchUrlTool(),
    TodoWriteTool(),
    TodoUpdateTool(),
    SpawnSubagentTool(),
    # Phase 5 memory tools (lazy store — no side effects at import time)
    MemorySaveTool(),
    MemorySearchTool(),
    MemoryListTool(),
    MemoryForgetTool(),
    MemoryConfirmTool(),
    MemoryCorrectTool(),
    MemoryStatsTool(),
]


def get_tool(name: str):
    """Look up a tool by name."""
    for t in ALL_TOOLS:
        if t.name == name:
            return t
    return None


def build_scoped_tools(
    project_root: str | Path,
    session_id: str,
    *,
    sandbox_policy: str = "interactive",
    sandbox_image: str | None = None,
    sandbox_user: str | None = None,
) -> tuple[list, SandboxManager]:
    """Build a fresh, workspace-bound registry for one API run.

    File tools share a canonical root and sandbox/sync share one manager, so
    neither mutable tool state nor filesystem authority crosses sessions.
    """
    root = Path(project_root).resolve()
    manager = SandboxManager(
        project_dir=root,
        session_id=session_id,
        benchmark_mode=sandbox_policy == "benchmark",
        image=sandbox_image,
        user=sandbox_user,
    )
    file_state = FileStateTracker()
    tools = [
        ExecuteInSandboxTool(manager),
        GrepSearchTool(project_root=root),
        ListFilesTool(project_root=root),
        ReadFileTool(project_root=root, file_state=file_state),
        WriteFileTool(
            project_root=root,
            protect_benchmark_files=sandbox_policy == "benchmark",
        ),
        EditFileTool(
            project_root=root,
            protect_benchmark_files=sandbox_policy == "benchmark",
            file_state=file_state,
        ),
        GlobTool(project_root=root),
        GrepTool(project_root=root),
        AgentTool(),
        FetchUrlTool(),
        TodoWriteTool(),
        TodoUpdateTool(),
        SpawnSubagentTool(),
        MemorySaveTool(),
        MemorySearchTool(),
        MemoryListTool(),
        MemoryForgetTool(),
        MemoryConfirmTool(),
        MemoryCorrectTool(),
        MemoryStatsTool(),
    ]
    return tools, manager
