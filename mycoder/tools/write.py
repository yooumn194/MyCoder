"""File creation / overwrite."""

from pathlib import Path

from .base import Tool
from .batch_diagnostics import BatchDiagnostics
from .edit import _changed_files
from .path_guard import PathGuard
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

    def __init__(
        self,
        batch_diagnostics: BatchDiagnostics | None = None,
        *,
        project_root: Path | str | None = None,
    ) -> None:
        # P1-2: coalesce repeated writes into one LSP diagnostics request
        self.diagnostics = batch_diagnostics or BatchDiagnostics()
        self._project_root = Path(project_root).resolve() if project_root else None

    def execute(self, file_path: str, content: str) -> str:
        try:
            if self._project_root is None:
                p = resolve_workspace_path(file_path)
            else:
                raw = Path(file_path)
                if file_path == "/workspace" or file_path.startswith("/workspace/"):
                    raw = self._project_root / raw.relative_to("/workspace")
                elif not raw.is_absolute():
                    raw = self._project_root / raw
                p = PathGuard(self._project_root).resolve(str(raw))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            _changed_files.add(str(p))
            # queue this file for diagnostics (batch at the threshold)
            self.diagnostics.add(file_path)
            n_lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            return f"Wrote {n_lines} lines to {file_path}"
        except Exception as e:
            return f"Error: {e}"
