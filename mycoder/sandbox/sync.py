"""Sandbox /workspace <-> host project directory incremental sync.

execute_in_sandbox writes to the container's /workspace volume, which
read_file / write_file on the host cannot see. WorkspaceSync reconciles the
two views without ever scanning the full tree:

    copy_out()      `docker diff` lists exactly what changed under /workspace;
                    each changed file is copied back via get_archive.
    resolve_path()  /workspace/foo.py -> {host_project_dir}/foo.py, gated on
                    the VOLUME still existing (not the container: after
                    sandbox.stop() the container is gone but the volume — and
                    therefore the mapping — survives).

Deletion handling (v2.1): a deletion in the sandbox is never silently
ignored. clean=True removes the host file (rsync --delete semantics, audit
sandbox.sync_clean); clean=False renames it to {rel}.container-deleted and
marks the return entry "(pending)" (audit sandbox.pending_deletion) so the
operator decides.
"""

import asyncio
import io
import shutil
import tarfile
from pathlib import Path

from .logger import get_logger

logger = get_logger()

_WORKSPACE = "/workspace"
_WORKSPACE_PREFIX = "/workspace/"
_EXCLUDE_DIRS = {".git", "node_modules", "__pycache__"}
_EXCLUDE_SUFFIXES = (".pyc",)
# v2.1: changed_files lists are truncated before being returned to a tool.
_MAX_CHANGED_FILES = 50


class WorkspaceSync:
    """Incremental /workspace <-> host sync bound to a DockerSandbox."""

    def __init__(self, host_project_dir, backend) -> None:
        self.host_project_dir = Path(host_project_dir).resolve()
        self.backend = backend  # DockerSandbox

    # ------------------------------------------------------------- path map

    async def _ensure_started(self) -> None:
        """Restart the container when it isn't running (e.g. after an idle
        reap), so diff/copy resume on the surviving volume instead of erroring.
        Defensive getattr keeps fake backends (tests) working unchanged."""
        ensure = getattr(self.backend, "ensure_started", None)
        if ensure is not None:
            await ensure()

    def volume_exists(self) -> bool:
        """True when the workspace VOLUME still exists.

        Keyed on the volume name, never on container liveness (v2.1): the
        mapping stays valid after the container is stopped/removed. docker
        volume inspect is a cheap call; failures are treated as "gone" so the
        caller falls back to the unmapped path (fail-closed, no hallucination).
        """
        try:
            self.backend._client().volumes.get(self.backend._volume_name)
            return True
        except Exception:
            return False

    def resolve_path(self, path: str) -> str:
        """Map /workspace/foo.py -> {host_project_dir}/foo.py.

        Gated on the volume, not the container (v2.1): the mapping stays valid
        after sandbox.stop(). Paths whose volume no longer exists pass through
        unchanged. Non-workspace paths never consult docker at all.
        """
        if path == _WORKSPACE:
            return str(self.host_project_dir) if self.volume_exists() else path
        if path.startswith(_WORKSPACE_PREFIX):
            if not self.volume_exists():
                return path
            rel = path[len(_WORKSPACE_PREFIX):]
            return str(self.host_project_dir / rel)
        return path

    # ------------------------------------------------------- change listing

    async def diff_changed_files(self) -> tuple[list[str], bool, int]:
        """(changed_rel_paths, truncated, total_count) for tool output.

        Returns only the paths, never copies — the sandbox tool appends them
        to its output and the agent decides when to call sync_workspace().
        """
        all_paths = [rel for rel, _ in await self._raw_changes()]
        total = len(all_paths)
        return all_paths[:_MAX_CHANGED_FILES], total > _MAX_CHANGED_FILES, total

    async def _raw_changes(self) -> list[tuple[str, int]]:
        """(relative_path, Kind) pairs under /workspace, excludes applied.

        INTERVIEW_NOTE: the change detector is `git status` inside the
        workspace, not `docker diff`. The workspace is seeded with `git clone
        /src`, so docker diff would report the ENTIRE clone as "added" and
        could not tell the agent's edits apart from the provisioning baseline.
        git reports exactly the agent's changes. For a non-git /src (the
        cp -a provisioning path) we fall back to docker diff, minus a baseline
        snapshot taken right after provisioning.

        Kind: 0 modified, 1 added, 2 deleted (docker API change types).
        """
        await self._ensure_started()
        result = await self.backend._exec([
            "/bin/sh",
            "-c",
            "git -C /workspace status --porcelain=v1 --untracked-files=normal",
        ])
        if result.exit_code == 0:
            changes = _git_status_changes(result.stdout)
        else:  # not a git repo (cp -a provisioning)
            changes = await self._docker_diff_changes()
        return [(rel, kind) for rel, kind in changes if not self._excluded(rel)]

    async def _docker_diff_changes(self) -> list[tuple[str, int]]:
        """docker-diff fallback for non-git workspaces, minus the provisioning baseline."""
        if self.backend._container is None:
            return []
        try:
            # container.diff() is None (JSON null) when nothing changed.
            raw = (await asyncio.to_thread(self.backend._container.diff)) or []
        except Exception:
            return []
        baseline = set(getattr(self.backend, "_fs_baseline", None) or ())
        out: list[tuple[str, int]] = []
        for change in raw:
            path = change.get("Path", "")
            if not path.startswith(_WORKSPACE_PREFIX):
                continue
            rel = path[len(_WORKSPACE_PREFIX):]
            if rel in baseline:
                continue
            out.append((rel, change.get("Kind", 0)))
        return out

    @staticmethod
    def _excluded(rel: str) -> bool:
        parts = rel.split("/")
        if any(part in _EXCLUDE_DIRS for part in parts):
            return True
        return rel.endswith(_EXCLUDE_SUFFIXES)

    # ---------------------------------------------------------- copy out

    async def copy_out(self, clean: bool = False) -> list[str]:
        """Copy changed /workspace files back to the host project dir.

        :return: list of changed relative paths; deletions are returned with a
                 "(pending)" marker when clean=False.

        The container is (re)started first: an idle-reap stopped it but the
        volume survives, so syncing after an idle gap must resume, not error.
        """
        await self._ensure_started()
        changed: list[str] = []
        for rel, kind in await self._raw_changes():
            host_path = self.host_project_dir / rel
            if kind == 2:  # deleted
                changed.append(self._handle_deletion(rel, host_path, clean))
            else:
                await self._copy_out_one(rel, host_path)
                changed.append(rel)
        return changed

    async def copy_out_files(self, file_paths: list[str]) -> None:
        """Sync only the given /workspace/... paths (on-demand, e.g. read_file)."""
        await self._ensure_started()
        for path in file_paths:
            if not path.startswith(_WORKSPACE_PREFIX):
                continue
            rel = path[len(_WORKSPACE_PREFIX):]
            await self._copy_out_one(rel, self.host_project_dir / rel)

    # ------------------------------------------------------------- helpers

    def _handle_deletion(self, rel: str, host_path: Path, clean: bool) -> str:
        if not host_path.exists():
            return f"{rel} (deleted in sandbox; nothing on host)"
        if clean:
            _remove_host_path(host_path)
            logger.info("sandbox.sync_clean", path=rel)
            return rel
        pending = Path(str(host_path) + ".container-deleted")
        host_path.rename(pending)
        logger.warning("sandbox.pending_deletion", path=rel, renamed=str(pending))
        return f"{rel} (pending)"

    async def _copy_out_one(self, rel: str, host_path: Path) -> None:
        try:
            stream, _stat = await asyncio.to_thread(
                self.backend._container.get_archive, f"{_WORKSPACE_PREFIX}{rel}"
            )
        except Exception as exc:
            logger.warning("sandbox.copy_error", path=rel, error=str(exc))
            return
        _extract_archive(stream, self.host_project_dir)


def _extract_archive(stream, dest_dir: Path) -> None:
    """Write a get_archive tar stream to dest_dir, stripping the workspace/ prefix.

    docker's get_archive('/workspace/foo.py') yields a tar whose members are
    root-relative ('workspace/foo.py'); we map them back to host-relative.
    """
    data = b"".join(stream)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
        for member in tar.getmembers():
            rel = _strip_workspace(member.name)
            if not rel:
                continue
            out = dest_dir / rel
            if member.isdir():
                out.mkdir(parents=True, exist_ok=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            with open(out, "wb") as fh:
                fh.write(src.read())


def _strip_workspace(name: str) -> str:
    name = name.lstrip("./")
    if name == "workspace" or name.startswith("workspace/"):
        return name[len("workspace"):].lstrip("/")
    return name


def _git_status_changes(porcelain: str) -> list[tuple[str, int]]:
    """Parse `git status --porcelain=v1` into (rel_path, kind) pairs.

    Each line is `XY path`; a rename/copy is `XY old -> new`. Untracked files
    appear as `?? path`. Deletions (D in X or Y) map to Kind 2 so copy_out can
    handle them; everything else maps to Kind 1 (copy back).
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


def _remove_host_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)
