"""Search-and-replace file editing (Claude Code's key innovation).

The core idea: instead of sending whole-file rewrites or line-number patches,
the LLM specifies an *exact* substring to find and its replacement. The
substring must appear exactly once in the file, which eliminates ambiguity
and makes edits safe and reviewable.
"""

import difflib
from pathlib import Path

from .base import Tool
from .file_state import FileStateTracker
from .path_guard import PathGuard
from .workspace_path import resolve_workspace_path
from ..patch_policy import is_protected_benchmark_path

# track files changed this session for /diff
_changed_files: set[str] = set()


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "Edit a file by replacing an exact string match. "
        "Read the file in this agent session before editing; stale reads are rejected. "
        "old_string must be non-empty and appear exactly once in the file for safety; "
        "old_string and new_string must differ. "
        "Strip read_file's line-number prefix before copying old_string. "
        "Include enough surrounding context to ensure uniqueness, or set replace_all=true."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to edit",
            },
            "old_string": {
                "type": "string",
                "description": "Exact text to find (must be unique in file)",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace every exact occurrence (default false)",
            },
        },
        "required": ["file_path", "old_string", "new_string"],
    }

    def __init__(
        self,
        *,
        project_root: Path | str | None = None,
        protect_benchmark_files: bool = False,
        file_state: FileStateTracker | None = None,
    ) -> None:
        self._project_root = Path(project_root).resolve() if project_root else None
        self._protect_benchmark_files = protect_benchmark_files
        self._file_state = file_state

    def execute(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        try:
            if self._project_root is None:
                p = Path(resolve_workspace_path(file_path))
            else:
                raw = Path(file_path)
                if file_path == "/workspace" or file_path.startswith("/workspace/"):
                    raw = self._project_root / raw.relative_to("/workspace")
                elif not raw.is_absolute():
                    raw = self._project_root / raw
                p = PathGuard(self._project_root).resolve(str(raw))
            relative = (
                p.relative_to(self._project_root)
                if self._project_root is not None
                else p
            )
            if self._protect_benchmark_files and is_protected_benchmark_path(relative):
                return (
                    f"Error: benchmark policy protects test/config file {file_path}; "
                    "inspect it but modify production code only"
                )
            if not p.exists():
                return f"Error: {file_path} not found"

            try:
                content = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return f"Error: {file_path} is not a UTF-8 text file (edit_file only edits text files)"
            if self._file_state is not None:
                state_error = self._file_state.validate(p, content)
                if state_error == "FILE_NOT_READ":
                    return (
                        f"Error [FILE_NOT_READ]: {file_path} must be read in this agent "
                        "session before it can be edited"
                    )
                if state_error == "STALE_FILE":
                    return (
                        f"Error [STALE_FILE]: {file_path} changed after it was read; "
                        "read it again before editing"
                    )
            if not old_string:
                return (
                    "Error [EMPTY_OLD_STRING]: old_string must be non-empty; "
                    "read a concrete context before editing"
                )
            if old_string == new_string:
                return (
                    "Error [NO_OP_EDIT]: old_string and new_string are identical; "
                    "provide an actual replacement or do not call edit_file"
                )
            occurrences = content.count(old_string)

            if occurrences == 0:
                preview = content[:500] + ("..." if len(content) > 500 else "")
                return f"Error: old_string not found in {file_path}.\nFile starts with:\n{preview}"
            if occurrences > 1 and not replace_all:
                return (
                    f"Error: old_string appears {occurrences} times in {file_path}. "
                    "Include more surrounding lines to make it unique or set replace_all=true."
                )

            new_content = content.replace(old_string, new_string, -1 if replace_all else 1)
            if new_content == content:
                # Defensive invariant: a successful mutation must change the
                # bytes on disk, even if a future replacement implementation
                # gains normalization or fuzzy matching.
                return "Error [NO_OP_EDIT]: replacement produced no file change"
            p.write_text(new_content, encoding="utf-8")
            if self._file_state is not None:
                self._file_state.observe(p, new_content)
            # No push into the sandbox: /workspace IS this path (single
            # bind-mounted filesystem), so the write is already visible to the
            # next `execute_in_sandbox` call.
            _changed_files.add(str(p))

            # generate a unified diff so the user/LLM can see exactly what changed
            diff = _unified_diff(content, new_content, str(p))
            return f"Edited {file_path}\n{diff}"
        except Exception as e:
            return f"Error: {e}"


def _unified_diff(old: str, new: str, filename: str, context: int = 3) -> str:
    """Generate a compact unified diff between old and new file content."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        n=context,
    )
    result = "".join(diff)
    # truncate enormous diffs
    if len(result) > 3000:
        result = result[:2500] + "\n... (diff truncated)\n"
    return result
