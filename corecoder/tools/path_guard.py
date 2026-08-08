"""Path safety validation shared by every host-side tool.

Defends the project directory against two escape vectors:

  * path traversal — `../etc/passwd` or an absolute path outside the root;
  * symlink escape — a symlink sitting inside the project but pointing
    somewhere outside it (e.g. `config -> /etc`).

Every tool that takes a user-supplied path runs it through
PathGuard.resolve(); anything that escapes the root is rejected with a clear
error and no I/O ever touches the outside path.
"""

from pathlib import Path


class PathTraversalError(Exception):
    """Raised when a path escapes the project root."""


class PathGuard:
    """Validates that user-supplied paths stay inside a project root."""

    def __init__(self, project_root: Path | str) -> None:
        self.project_root = Path(project_root).resolve()

    def resolve(self, user_path: str | Path) -> Path:
        """Resolve a user path and reject it if it escapes the project root.

        Three checks:

        1. join against the root and fully resolve — `.resolve()` follows every
           symlink component, so the result is the canonical real path;
        2. the canonical path must stay inside the root — this is the check
           that actually stops symlink escape, because the escape target lands
           outside and relative_to() fails;
        3. report symlink escapes explicitly (a clearer error than the generic
           range error in step 2) by inspecting the unresolved components.
        """
        if isinstance(user_path, Path):
            user_path = str(user_path)
        candidate = self.project_root / user_path

        resolved = candidate.resolve()  # follows symlinks -> canonical target
        try:
            resolved.relative_to(self.project_root)
        except ValueError:
            raise PathTraversalError(f"路径超出项目范围: {user_path}")

        self._check_symlink_components(candidate, user_path)
        return resolved

    def _check_symlink_components(self, candidate: Path, user_path: str) -> None:
        """A symlink inside the root pointing outside is caught by step 2; this
        gives a specific, actionable message for that case.

        Only the user-controlled portion BELOW the project root is inspected.
        Walking the whole absolute path would flag system symlinks (e.g. macOS
        /var -> /private/var) as escapes even though the final resolved path is
        validly inside the project — a real false-positive that surfaced on
        paths under /var/folders.
        """
        try:
            rel = candidate.relative_to(self.project_root)
        except ValueError:
            return  # absolute path outside the root is rejected by resolve()
        for rel_part in reversed(rel.parents):
            part = self.project_root / rel_part
            if not part.exists():
                continue
            if part.is_symlink():
                target = part.resolve()
                try:
                    target.relative_to(self.project_root)
                except ValueError:
                    raise PathTraversalError(
                        f"符号链接指向项目外: {user_path} → {target}"
                    )
