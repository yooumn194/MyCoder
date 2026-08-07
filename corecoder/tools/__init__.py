"""Tool registry."""

from .read import ReadFileTool
from .write import WriteFileTool
from .edit import EditFileTool
from .glob_tool import GlobTool
from .grep import GrepTool
from .agent import AgentTool
from .fetch import FetchUrlTool
from .sandbox_tool import ExecuteInSandboxTool
from .sync_tool import SyncWorkspaceTool


ALL_TOOLS = [
    # The old `bash` tool (regex-blacklist gated, runs on the host) is replaced
    # by the sandboxed executor: same contract, isolated by Docker, with a
    # user-confirmed local fallback. sync_workspace pulls the sandbox's
    # /workspace changes back to the host. corecoder/agent.py needs no changes.
    ExecuteInSandboxTool(),
    SyncWorkspaceTool(),
    ReadFileTool(),
    WriteFileTool(),
    EditFileTool(),
    GlobTool(),
    GrepTool(),
    AgentTool(),
    FetchUrlTool(),
]


def get_tool(name: str):
    """Look up a tool by name."""
    for t in ALL_TOOLS:
        if t.name == name:
            return t
    return None
