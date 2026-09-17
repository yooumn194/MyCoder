"""Single-filesystem workspace mapping for the sandbox.

Historically this module reconciled TWO copies of the repository: the host
checkout (edited by read_file / write_file / edit_file) and a private Docker
volume at /workspace (edited by execute_in_sandbox). That split-brain design is
gone — see ``docker_executor`` — and with it every failure mode the
reconciliation needed:

    * a host edit invisible to ``pytest`` inside the container
    * a container edit invisible to ``git diff`` on the host
    * a best-effort ``copy_out()`` that silently dropped work whenever the
      container was reaped, restarted, or failed mid-copy

The container now bind-mounts the host project directory read-write at
/workspace, so both actors share one tree. Everything left here is therefore
pure bookkeeping over that single tree:

    resolve_path()   /workspace/foo.py -> {host_project_dir}/foo.py (a pure
                     string mapping — the two names denote the SAME file)
    changed_files()  which paths the last command changed, for tool output

No bytes are ever copied anywhere.
"""

from __future__ import annotations

from pathlib import Path

from .logger import get_logger

logger = get_logger()

_WORKSPACE = "/workspace"
_WORKSPACE_PREFIX = "/workspace/"
_EXCLUDE_DIRS = {".git", "node_modules", "__pycache__"}
_EXCLUDE_SUFFIXES = (".pyc",)
# changed_files lists are truncated before being returned to a tool.
_MAX_CHANGED_FILES = 50


class WorkspaceSync:
    """Path mapping + change listing for one bind-mounted workspace.

    The name is kept for call-site compatibility (``SandboxManager.get_sync``,
    ``tools/workspace_path.py``); there is nothing left to synchronize.
    """

    def __init__(self, host_project_dir, backend=None) -> None:
        self.host_project_dir = Path(host_project_dir).resolve()
        # Optional: only the Docker backend can list changes by running `git`
        # inside the container. LocalExecutor already works on the host tree.
        self.backend = backend

    # ------------------------------------------------------------- path map

    def resolve_path(self, path: str) -> str:
        """Map /workspace/foo.py -> {host_project_dir}/foo.py.

        Unconditional and free of Docker I/O: with a bind mount the mapping is
        a fact about the mount table, not about a container's liveness. It stays
        valid before start(), after stop(), and after an idle reap.
        """
        if path == _WORKSPACE:
            return str(self.host_project_dir)
        if path.startswith(_WORKSPACE_PREFIX):
            rel = path[len(_WORKSPACE_PREFIX):]
            return str(self.host_project_dir / rel)
        return path

    # ------------------------------------------------------- change listing

    async def diff_changed_files(self) -> tuple[list[str], bool, int]:
        """(changed_rel_paths, truncated, total_count) for tool output.

        Purely informational: the files are already in place on the host, so
        there is nothing for the agent to pull back.
        """
        all_paths = [rel for rel, _ in await self._raw_changes()]
        total = len(all_paths)
        return all_paths[:_MAX_CHANGED_FILES], total > _MAX_CHANGED_FILES, total

    async def _raw_changes(self) -> list[tuple[str, int]]:
        """(relative_path, Kind) pairs under /workspace, excludes applied.

        Kind: 0 modified, 1 added, 2 deleted (docker API change types), kept
        only so callers that branch on it keep working.
        """
        if self.backend is None:
            return []
        backend = self.backend
        restart = getattr(backend, "ensure_started", None)
        if restart is not None:
            await restart()
        result = await backend._exec(
            [
                "/bin/sh",
                "-c",
                "git -C /workspace status --porcelain=v1 --untracked-files=normal",
            ]
        )
        if result.exit_code != 0:  # not a git repo; nothing to report
            return []
        return [
            (rel, kind)
            for rel, kind in _git_status_changes(result.stdout)
            if not self._excluded(rel)
        ]

    @staticmethod
    def _excluded(rel: str) -> bool:
        parts = rel.split("/")
        if any(part in _EXCLUDE_DIRS for part in parts):
            return True
        return rel.endswith(_EXCLUDE_SUFFIXES)


def _git_status_changes(porcelain: str) -> list[tuple[str, int]]:
    """Parse `git status --porcelain=v1` into (rel_path, kind) pairs.

    Each line is `XY path`; a rename/copy is `XY old -> new`. Untracked files
    appear as `?? path`. Deletions (D in X or Y) map to Kind 2 so callers can
    tell them apart; everything else maps to Kind 1 (present).
    """
    out: list[tuple[str, int]] = []
    for line in porcelain.splitlines():
        if len(line) < 4:
            continue
        code, path = line[:2], line[3:]
        if " -> " in path:  # rename/copy: keep the destination
            path = path.split(" -> ", 1)[1]
        path = path.strip().strip('"')
        if not path:
            continue
        out.append((path, 2 if "D" in code else 1))
    return out
