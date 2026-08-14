"""File creation / overwrite."""

from .base import Tool
from .batch_diagnostics import BatchDiagnostics
from .edit import _changed_files
from .workspace_path import resolve_workspace_path


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Create a new file or completely overwrite an existing one. "
        "For small edits to existing files, prefer edit_file instead."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path for the file",
            },
            "content": {
                "type": "string",
                "description": "Full file content to write",
            },
        },
        "required": ["file_path", "content"],
    }

    def __init__(self, batch_diagnostics: BatchDiagnostics | None = None) -> None:
        # P1-2: coalesce repeated writes into one LSP diagnostics request
        self.diagnostics = batch_diagnostics or BatchDiagnostics()

    def execute(self, file_path: str, content: str) -> str:
        try:
            p = resolve_workspace_path(file_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            _changed_files.add(str(p))
            # queue this file for diagnostics (batch at the threshold)
            self.diagnostics.add(file_path)
            n_lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            return f"Wrote {n_lines} lines to {file_path}"
        except Exception as e:
            return f"Error: {e}"
