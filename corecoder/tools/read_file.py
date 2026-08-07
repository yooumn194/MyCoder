"""File reading with line numbers, ranges, truncation and path guarding.

Phase 2 spec: reads at most 300 lines per call, formats line numbers 4-wide
right-aligned (" 42 | line"), detects binary files, and every path must pass
PathGuard (stay inside the project root). The Phase 1 /workspace mapping is
preserved, so read_file can read files the sandbox produced.
"""

from pathlib import Path

from .base import Tool
from .path_guard import PathGuard, PathTraversalError
from .workspace_path import (
    get_project_root,
    resolve_workspace_path,
    try_on_demand_sync,
)

MAX_LINES = 300


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read a file with line numbers (format: ' 42 | line'). At most 300 "
        "lines per call — use start_line/end_line to page through long files. "
        "Binary files are rejected. /workspace/... paths are mapped onto the "
        "host project automatically. Always read a file before editing it."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file",
            },
            "start_line": {
                "type": "integer",
                "description": "Start line (1-based). Default 1.",
            },
            "end_line": {
                "type": "integer",
                "description": "End line, inclusive. Default: up to 300 lines from start_line.",
            },
        },
        "required": ["file_path"],
    }

    def __init__(self, *, project_root: Path | str | None = None) -> None:
        self._project_root = project_root

    def execute(
        self,
        file_path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        try:
            root = Path(self._project_root) if self._project_root else get_project_root()
            guard = PathGuard(project_root=root)
            # 1. resolve the user path: relative paths are project-root-based;
            #    absolute /workspace/... paths are mapped onto the host project
            p = Path(file_path)
            if p.is_absolute():
                p = Path(resolve_workspace_path(str(p)))
            else:
                p = root / p
            # 2. path guard: the canonical path must stay inside the project
            try:
                p = guard.resolve(str(p))
            except PathTraversalError as e:
                return f"[错误：{e}]"

            if not p.exists():
                # a /workspace file may exist only in the sandbox — pull it on demand
                if file_path.startswith("/workspace"):
                    try_on_demand_sync(file_path)
                if not p.exists():
                    if file_path.startswith("/workspace"):
                        return (
                            f"[错误：{file_path} 在沙箱工作区中不存在。请先执行相关命令生成该文件，"
                            "或调用 sync_workspace() 同步。]"
                        )
                    return f"[错误：文件不存在：{file_path}]"
            if not p.is_file():
                return f"[错误：{file_path} 是目录，不是文件]"

            # 3. binary detection (NUL byte in the head, same heuristic as rg)
            with open(p, "rb") as fh:
                if b"\x00" in fh.read(8192):
                    return "[二进制文件，无法读取]"

            # 4. read text
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"[错误：{e}]"

        lines = text.splitlines()
        total = len(lines)
        if total == 0:
            return "(empty file)"

        start = max(1, int(start_line or 1))
        if end_line is not None:
            end = int(end_line)
        else:
            end = min(total, start + MAX_LINES - 1)
        if end - start + 1 > MAX_LINES:
            end = start + MAX_LINES - 1  # hard 300-line cap even on explicit ranges
        if end < start:
            return "[错误：end_line 小于 start_line]"

        chunk = lines[start - 1 : end]
        numbered = "\n".join(f"{start + i:4} | {ln}" for i, ln in enumerate(chunk))

        if total > end:  # there is more file below what we showed
            if start == 1:
                numbered += (
                    f"\n[文件共 {total} 行，仅显示前 {end} 行。"
                    "请使用 start_line/end_line 参数读取特定范围]"
                )
            else:
                numbered += (
                    f"\n[文件共 {total} 行，仅显示第 {start}-{end} 行。"
                    "请使用 start_line/end_line 参数读取特定范围]"
                )
        return numbered or "(empty file)"
