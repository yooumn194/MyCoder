"""File reading with line numbers."""

from .base import Tool
from .workspace_path import resolve_workspace_path, try_on_demand_sync


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read a file's contents with line numbers. "
        "Always read a file before editing it."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file",
            },
            "offset": {
                "type": "integer",
                "description": "Start line (1-based). Default 1.",
            },
            "limit": {
                "type": "integer",
                "description": "Max lines to read. Default 2000.",
            },
        },
        "required": ["file_path"],
    }

    def execute(self, file_path: str, offset: int = 1, limit: int = 2000) -> str:
        try:
            p = resolve_workspace_path(file_path)
            if not p.exists():
                # The file may live in the sandbox workspace but not yet on the
                # host — pull just that one file on demand before giving up.
                if file_path.startswith("/workspace"):
                    try_on_demand_sync(file_path)
                if not p.exists():
                    if file_path.startswith("/workspace"):
                        return (
                            f"Error: {file_path} does not exist in the sandbox "
                            "workspace. Run the command that creates it first, "
                            "or call sync_workspace() to sync it to the host."
                        )
                    return f"Error: {file_path} not found"
            if not p.is_file():
                return f"Error: {file_path} is a directory, not a file"

            text = p.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            total = len(lines)

            start = max(0, offset - 1)
            chunk = lines[start : start + limit]
            numbered = [f"{start + i + 1}\t{ln}" for i, ln in enumerate(chunk)]
            result = "\n".join(numbered)

            if total > start + limit:
                result += f"\n... ({total} lines total, showing {start+1}-{start+len(chunk)})"
            return result or "(empty file)"
        except Exception as e:
            return f"Error: {e}"
