"""Safe file creation.

Existing files are intentionally owned by ``edit_file``.  Treating create and
replace as the same operation lets a partial model response silently truncate a
large source file, so ``write_file`` fails closed when the target already
exists.
"""

from pathlib import Path

from .base import Tool
from .batch_diagnostics import BatchDiagnostics
from .edit import _changed_files
from .path_guard import PathGuard
from .workspace_path import resolve_workspace_path
from ..patch_policy import is_protected_benchmark_path


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Create a new file. Refuses to overwrite an existing path; use edit_file "
        "with an exact, unique old_string when modifying existing source."
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
        protect_benchmark_files: bool = False,
    ) -> None:
        # P1-2: coalesce repeated writes into one LSP diagnostics request
        self.diagnostics = batch_diagnostics or BatchDiagnostics()
        self._project_root = Path(project_root).resolve() if project_root else None
        self._protect_benchmark_files = protect_benchmark_files

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
            if self._is_protected(p):
                return (
                    f"Error: benchmark policy protects test/config file {file_path}; "
                    "inspect it but modify production code only"
                )
            shadow = self._shadow_copy_target(p, content)
            if shadow is not None:
                return (
                    f"Error: refusing likely shadow copy {file_path}; existing "
                    f"module with near-identical content: {shadow}. Edit that file instead."
                )
            p.parent.mkdir(parents=True, exist_ok=True)
            try:
                # Exclusive creation makes the create-only contract atomic;
                # a concurrent creator cannot be overwritten between an
                # exists() check and write_text().
                with p.open("x", encoding="utf-8") as handle:
                    handle.write(content)
            except FileExistsError:
                return (
                    f"Error: refusing to overwrite existing file {file_path}; "
                    "use edit_file with an exact old_string"
                )
            # No push into the sandbox: /workspace IS this path (single
            # bind-mounted filesystem), so the write is already visible to the
            # next `execute_in_sandbox` call.
            _changed_files.add(str(p))
            # queue this file for diagnostics (batch at the threshold)
            self.diagnostics.add(file_path)
            n_lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            return f"Wrote {n_lines} lines to {file_path}"
        except Exception as e:
            return f"Error: {e}"

    def _is_protected(self, path: Path) -> bool:
        if not self._protect_benchmark_files:
            return False
        relative = (
            path.relative_to(self._project_root)
            if self._project_root is not None
            else path
        )
        return is_protected_benchmark_path(relative)

    def _shadow_copy_target(self, path: Path, content: str) -> Path | None:
        """Detect a wrong-path whole-module copy before it reaches the repo."""
        if self._project_root is None or len(content) < 4096:
            return None
        new_lines = {line.strip() for line in content.splitlines() if line.strip()}
        if len(new_lines) < 100:
            return None
        checked = 0
        for candidate in self._project_root.rglob(path.name):
            if checked >= 64:
                break
            if candidate == path or not candidate.is_file() or ".git" in candidate.parts:
                continue
            checked += 1
            try:
                existing = candidate.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if not (len(content) / 2 <= len(existing) <= len(content) * 2):
                continue
            existing_lines = {
                line.strip() for line in existing.splitlines() if line.strip()
            }
            overlap = len(new_lines & existing_lines) / len(new_lines)
            if overlap >= 0.9:
                return candidate.relative_to(self._project_root)
        return None
