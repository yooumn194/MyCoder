"""File pattern matching."""

from pathlib import Path
from .base import Tool
from .path_guard import PathGuard, PathTraversalError

# TODO(Phase 4): once list_files (path-guarded, symlink-safe, default-excludes)
# fully replaces this legacy tool in the agent's workflow, remove GlobTool and
# its tests. Keeping both registered confuses the agent about which lookup tool
# to use; the search-strategy prompt already steers it to list_files.


class GlobTool(Tool):
    predictive_safe = True
    name = "glob"
    description = (
        "Find files matching a glob pattern. "
        "Supports ** for recursive matching (e.g. '**/*.py')."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern, e.g. '**/*.py' or 'src/**/*.ts'",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (default: cwd)",
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, *, project_root=None) -> None:
        self._project_root = Path(project_root).resolve() if project_root else None

    def execute(self, pattern: str, path: str = ".") -> str:
        try:
            guard = PathGuard(self._project_root) if self._project_root else None
            base = (
                guard.resolve(path)
                if guard is not None
                else Path(path).expanduser().resolve()
            )
            if not base.is_dir():
                return f"Error: {path} is not a directory"

            hits = []
            for hit in base.glob(pattern):
                if guard is None:
                    hits.append(hit)
                else:
                    try:
                        hits.append(guard.resolve(str(hit)))
                    except PathTraversalError:
                        continue
            # sort by mtime, newest first
            hits.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

            total = len(hits)
            shown = hits[:100]
            lines = [str(h) for h in shown]
            result = "\n".join(lines)

            if total > 100:
                result += f"\n... ({total} matches, showing first 100)"
            return result or "No files matched."
        except Exception as e:
            return f"Error: {e}"
