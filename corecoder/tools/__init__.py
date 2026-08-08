"""Tool registry."""

from .read_file import ReadFileTool
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
from .sync_tool import SyncWorkspaceTool
from .todo_tools import TodoUpdateTool, TodoWriteTool


ALL_TOOLS = [
    # The old `bash` tool (regex-blacklist gated, runs on the host) is replaced
    # by the sandboxed executor: same contract, isolated by Docker, with a
    # user-confirmed local fallback. sync_workspace pulls the sandbox's
    # /workspace changes back to the host. grep_search / list_files are the
    # Phase 2 agentic-search tools (path-guarded, rg-first). todo_write /
    # todo_update are the Phase 3 planning tools. corecoder/agent.py only
    # gains a tiny guard + correction hook in _exec_tool (Phase 3 spec).
    ExecuteInSandboxTool(),
    SyncWorkspaceTool(),
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
]


def get_tool(name: str):
    """Look up a tool by name."""
    for t in ALL_TOOLS:
        if t.name == name:
            return t
    return None
