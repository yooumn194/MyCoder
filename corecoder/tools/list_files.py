"""list_files: glob-based file lookup within the project root.

Every returned path is PathGuard-validated (must stay inside the project
root), default-excludes junk directories, and never follows symlinks.

Security policy on symlinks: we deliberately do NOT follow symlinked
directories or report symlinked files. Following a symlink would let a glob
like `**/*.py` reach files outside the project through a link inside it — the
same escape PathGuard blocks for explicit paths. Dropping symlinks entirely
(the `find -L` / glob-follow behavior is the dangerous one) is the fail-closed
choice: a symlinked project never leaks, and an internal symlink's target can
be listed by globbing the target's real path instead.
"""

from pathlib import Path

from .base import Tool
from .path_guard import PathGuard, PathTraversalError
from .workspace_path import get_project_root

_DEFAULT_EXCLUDED = {"node_modules", ".git", "__pycache__"}


class ListFilesTool(Tool):
    name = "list_files"
    description = (
        "List files matching a glob pattern (e.g. '**/*.py'), one path per "
        "line, sorted alphabetically. Default-excludes node_modules, .git, "
        "__pycache__ and *.pyc, and never follows symlinks. Prefer this over "
        "the legacy `glob` tool: it is path-guarded and symlink-safe."
    )
    parameters = {
        "type": "object",
        "properties": {
            "glob_pattern": {
                "type": "string",
                "description": "Glob pattern to match, e.g. '**/*.py'",
            },
            "path": {
                "type": "string",
                "description": "Base directory (default '.' = project root)",
            },
        },
        "required": ["glob_pattern"],
    }

    def __init__(self, *, project_root=None):
        self._project_root = project_root

    def execute(self, glob_pattern: str, path: str = ".") -> str:
        root = Path(self._project_root) if self._project_root else get_project_root()
        guard = PathGuard(project_root=root)
        try:
            base = guard.resolve(path)
        except PathTraversalError as e:
            return f"[错误：{e}]"

        results: list[Path] = []
        try:
            for p in base.glob(glob_pattern):
                # every result must stay inside the root (blocks glob patterns
                # that escape via '..' or a symlinked directory)
                try:
                    guard.resolve(str(p))
                except PathTraversalError:
                    continue
                if not self._keep(p, base):
                    continue
                results.append(p)
        except Exception as e:
            return f"[错误：{e}]"

        lines = sorted(str(p.relative_to(root)) for p in results)
        return "\n".join(lines) or "(no matches)"

    @staticmethod
    def _keep(p: Path, base: Path) -> bool:
        if p.is_symlink():
            return False  # 不跟踪符号链接（叶子）
        if _has_symlink_ancestor(p, base):
            return False  # 经符号链接目录到达的路径也不返回
        if any(part in _DEFAULT_EXCLUDED for part in p.parts):
            return False
        if p.suffix == ".pyc":
            return False
        return True


def _has_symlink_ancestor(p: Path, base: Path) -> bool:
    """True when any parent between `base` and `p` is a symlink."""
    for parent in p.parents:
        if parent == base:
            break
        if parent.is_symlink():
            return True
    return False
