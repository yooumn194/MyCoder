"""grep_search: regex search across the project, ripgrep-first.

Zero index, zero embedding — a plain search over the project directory. Runs
on the host (the project dir Phase 1 mounts). Uses ripgrep when available,
falls back to a pure-Python walk (os.walk + re) so it works on hosts without
rg. Both paths share one output formatter, so results are identical.

Bounded in three ways: max_results caps what is returned, the rg subprocess
has a 30s timeout, and the fallback walk stops after PY_MAX_FILES files.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

from .base import Tool
from .path_guard import PathGuard, PathTraversalError
from .workspace_path import get_project_root

RG_TIMEOUT = 30          # seconds; on expiry return a graceful timeout note
PY_MAX_FILES = 50_000    # fallback walk cap
DEFAULT_MAX_RESULTS = 50

# directories the fallback prunes during the walk (rg skips .git by default)
_PRUNED_DIRS = {".git", "node_modules", "__pycache__"}


def _is_text_head(head: bytes) -> bool:
    """A NUL byte in the first chunk marks a binary file (same heuristic rg)."""
    return b"\x00" not in head


class _SearchTimeout(Exception):
    pass


class _SearchError(Exception):
    pass


class GrepSearchTool(Tool):
    name = "grep_search"
    description = (
        "Regex-search the project for a pattern and return matches grouped by "
        "file with line numbers (e.g. '## src/auth/handler.py' then 'L23: "
        "def authenticate_user(...)'). Uses ripgrep when available and a pure-"
        "Python fallback otherwise. Restrict with file_types (comma-separated "
        "extensions like 'py,js') to keep results small. Prefer this over the "
        "legacy `grep` tool: it caps results and filters by file type."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex to search for. Zero-width lookahead/lookbehind are not supported.",
            },
            "path": {
                "type": "string",
                "description": "Directory to search under (default '.' = project root)",
            },
            "file_types": {
                "type": "string",
                "description": "Comma-separated extensions to restrict to, e.g. 'py,js' (default: all files)",
            },
            "max_results": {
                "type": "integer",
                "description": "Max matches to return (default 50)",
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "Case-sensitive matching (default true)",
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, *, project_root=None, rg_path="_default"):
        self._project_root = project_root
        # rg_path is injectable so tests can force either backend. Default:
        # whatever `rg` binary is on PATH (None means the Python fallback).
        self._rg_path = shutil.which("rg") if rg_path == "_default" else rg_path

    def execute(
        self,
        pattern: str,
        path: str = ".",
        file_types: str | None = None,
        max_results: int = DEFAULT_MAX_RESULTS,
        case_sensitive: bool = True,
    ) -> str:
        # resolve() once so relpath() between the walked search root and the
        # project root agree — a raw path with a symlinked ancestor (macOS
        # /var) would otherwise produce garbage ../../ relative paths.
        root = (Path(self._project_root) if self._project_root else get_project_root()).resolve()
        guard = PathGuard(project_root=root)
        try:
            search_root = guard.resolve(path)
        except PathTraversalError as e:
            return f"[错误：{e}]"
        max_results = max(1, int(max_results))
        types = [t.strip().lstrip(".") for t in (file_types or "").split(",") if t.strip()]

        try:
            if self._rg_path:
                matches, total, truncated = self._rg_search(
                    pattern, search_root, root, types, max_results, case_sensitive
                )
            else:
                matches, total, truncated = self._py_search(
                    pattern, search_root, root, types, max_results, case_sensitive
                )
        except _SearchTimeout:
            return "[搜索超时，请缩小范围]"
        except _SearchError as e:
            return f"[搜索失败：{e}]"
        return self._format(matches, total, truncated)

    # ---------------------------------------------------------------- backends

    def _rg_search(self, pattern, search_root, project_root, types, max_results, case_sensitive):
        """Run ripgrep and parse `path:linenum:content` lines.

        cwd is the PROJECT root and the search argument is the search root's
        path relative to it, so rg prints PROJECT-relative paths — the same
        ones the Python fallback produces, and the same ones the other tools
        (read_file / list_files) accept back.
        """
        cmd = [
            self._rg_path,
            "--no-heading",
            "--line-number",
            "--with-filename",
            "--color",
            "never",
        ]
        if not case_sensitive:
            cmd.append("-i")
        for ext in types:
            cmd.extend(["-g", f"*.{ext}"])
        cmd.extend([pattern, os.path.relpath(search_root, project_root) or "."])
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=RG_TIMEOUT,
                cwd=str(project_root),
            )
        except subprocess.TimeoutExpired:
            raise _SearchTimeout
        except OSError:
            raise _SearchError("rg 无法执行")
        if proc.returncode == 2:  # rg exits 2 on a bad pattern / invocation
            raise _SearchError(proc.stderr.strip() or "无效的正则表达式")

        matches, total = [], 0
        for line in proc.stdout.splitlines():
            first = line.find(":")
            if first < 0:
                continue
            second = line.find(":", first + 1)
            if second < 0:
                continue
            try:
                lineno = int(line[first + 1 : second])
            except ValueError:
                continue
            total += 1
            if len(matches) < max_results:
                matches.append((line[:first], lineno, line[second + 1 :]))
        return matches, total, total > max_results

    def _py_search(self, pattern, search_root, project_root, types, max_results, case_sensitive):
        """Pure-Python fallback: bounded walk + per-line regex search."""
        try:
            regex = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
        except re.error as e:
            raise _SearchError(str(e))

        matches, total, files_seen = [], 0, 0
        for dirpath, dirnames, filenames in os.walk(search_root):
            dirnames[:] = [d for d in dirnames if d not in _PRUNED_DIRS]
            for fname in filenames:
                if types and Path(fname).suffix.lstrip(".") not in types:
                    continue
                files_seen += 1
                if files_seen > PY_MAX_FILES:
                    break
                fpath = os.path.join(dirpath, fname)
                try:
                    # binary detection mirrors rg's default (which skips binary
                    # files): a file whose head is not valid UTF-8 is skipped.
                    with open(fpath, "rb") as fh:
                        if not _is_text_head(fh.read(8192)):
                            continue
                    with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                        for lineno, line in enumerate(fh, 1):
                            if regex.search(line):
                                rel = os.path.relpath(fpath, project_root)
                                total += 1
                                if len(matches) < max_results:
                                    matches.append((rel, lineno, line.rstrip("\n")))
                except (OSError, UnicodeDecodeError):
                    continue  # unreadable/binary file — skip, don't abort the search
        return matches, total, total > max_results

    # ----------------------------------------------------------------- format

    @staticmethod
    def _format(matches, total, truncated) -> str:
        """Shared formatter — guarantees rg and Python backends render alike."""
        if total == 0:
            return "(no matches)"
        groups: dict[str, list[tuple[int, str]]] = {}
        for rel, lineno, content in matches:
            groups.setdefault(rel, []).append((lineno, content))
        shown = len(matches)
        noun_m = "match" if shown == 1 else "matches"
        noun_f = "file" if len(groups) == 1 else "files"
        lines = [f"Found {shown} {noun_m} in {len(groups)} {noun_f}:"]
        for rel, hits in groups.items():
            lines.append("")
            lines.append(f"## {rel}")
            for lineno, content in hits:
                lines.append(f"L{lineno}: {content}")
        if truncated:
            lines.append("")
            lines.append(f"... (结果已截断，共 {total} 条匹配，请缩小搜索范围)")
        return "\n".join(lines)
