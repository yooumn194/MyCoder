"""A restore point for the working tree, because the sandbox mounts it directly.

P0-1 removed the container-side copy of the project: the container bind-mounts
the host checkout read-write at ``/workspace``, so the shell and the file tools
act on one filesystem. That is what the other harnesses do, and it removes a
whole class of "my edit vanished" bugs — but it also means a destructive command
now has the blast radius of the *real* repository. Tearing the container down no
longer undoes anything.

The permission layer asks before risky commands, and the hard pre-check blocks
the catastrophic ones, but neither claims to be exhaustive: ``git clean -fdx``,
``shutil.rmtree('/workspace')`` and a bare ``> src/main.py`` all reach the
mount, and "ask" is never a guarantee — an unattended run fails closed, while a
human can approve by mistake.

So every session records where the tree stood before the first command runs:

    HEAD                    the committed state
    a snapshot commit       a real commit object holding the ENTIRE working
                            tree — tracked edits *and* untracked files —
                            built through a throwaway index
                            (``GIT_INDEX_FILE``), so neither the real index nor
                            the worktree nor the stash list is touched
    the untracked list      listed as well, because it is the part a
                            tracked-only stash cannot bring back

The snapshot is pinned under ``refs/mycoder/sessions/<session>``. It used to be
one process-wide ``refs/mycoder/restore-point``, which meant a second session in
the same repository silently clobbered the first session's only recovery point.
That historical name is still what a caller *without* a session id gets, which
is why the two live in sibling namespaces: nesting them would make git reject
the second form outright (see RESTORE_REF_PREFIX).

Recovery is then one command, printed next to every dangerous command's output
instead of leaving the model (or the operator) to improvise::

    git restore --source=<ref> --worktree -- .
    git stash apply <stash>          # tracked-only fallback, same output

Boundary, stated plainly: a restore point lives inside the repository, so it
cannot survive the repository's own removal. ``rm -rf .`` (which takes ``.git``
with it) stays a hard pre-check block, and the confirmation rules cover
``git clean``, ``git update-ref``, interpreter-side deletion and
``git checkout <ref> -- .`` — the shapes that destroy work *without* destroying
the object store. Anything that removes ``.git`` is unrecoverable by
construction and is treated as such.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .logger import get_logger

logger = get_logger()

# A ref under refs/ makes the object durable: an unreferenced commit can be
# pruned by `git gc --prune`, and a restore point that expires is not a restore
# point. It does not appear in `git branch` output and does not affect diffs.
#
# Two namespaces, both siblings under refs/mycoder/:
#
#   refs/mycoder/sessions/<slug>    one per session (what a managed session uses)
#   refs/mycoder/restore-point      the historical no-session name
#
# They MUST NOT be nested inside each other. Git's ref store cannot hold
# `refs/mycoder/restore-point` and `refs/mycoder/restore-point/<slug>` at the
# same time (a directory/file conflict), so the first capture of one form made
# every later capture of the other fail with `update-ref failed` — and since
# that failure only drops the ref, the snapshot was left unreferenced and
# prunable by `git gc --prune`. `refs/mycoder/commits/<sha>`, used by the
# SWE-bench adapter for base-commit caches, is a third sibling.
RESTORE_REF_PREFIX = "refs/mycoder/sessions"
# The no-session ref, kept at its historical name so existing
# `git stash apply refs/mycoder/restore-point` workflows keep working.
RESTORE_REF = "refs/mycoder/restore-point"

_GIT_TIMEOUT_SECONDS = 60
_SNAPSHOT_TIMEOUT_SECONDS = 180
# Plumbing identity: a snapshot commit is an internal object, and it must not
# fail (or be attributed to the user) when the repository has no user.name /
# user.email configured — a fresh CI checkout often does not.
_IDENTITY_ENV = {
    "GIT_AUTHOR_NAME": "MyCoder restore point",
    "GIT_AUTHOR_EMAIL": "restore-point@mycoder.invalid",
    "GIT_COMMITTER_NAME": "MyCoder restore point",
    "GIT_COMMITTER_EMAIL": "restore-point@mycoder.invalid",
}

_UNSAFE_REF_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def session_ref(session_id: str | None) -> str:
    """The ref that pins this snapshot.

    Git ref names reject whitespace, `~^:?*[\\` and a trailing `/`, so a
    session id is slugged rather than trusted. An empty/unknown session falls
    back to the historical generic name — the same ref a direct caller without
    a session id has always used — which lives in its own namespace rather than
    as a parent of the session refs (see RESTORE_REF_PREFIX).
    """
    if session_id is None:
        return RESTORE_REF
    slug = _UNSAFE_REF_CHARS.sub("-", str(session_id).strip()).strip("-")
    if not slug:
        slug = "default"
    return f"{RESTORE_REF_PREFIX}/{slug[:120]}"


@dataclass(frozen=True)
class RestorePoint:
    """Where the working tree stood when the session started."""

    head: str | None
    stash: str | None
    dirty: bool
    untracked: tuple[str, ...]
    created_at: str
    # Complete worktree snapshot (tracked + untracked). None when it could not
    # be built; `stash` then still covers the tracked files.
    snapshot: str | None = None
    ref: str | None = None
    # Why the complete snapshot is missing, when it is. Surfaced so a run that
    # only has the degraded (tracked-only) restore point says so.
    snapshot_error: str | None = None

    @property
    def available(self) -> bool:
        return self.head is not None or self.stash is not None or self.snapshot is not None

    def as_metadata(self) -> dict[str, object]:
        """Compact, JSON-serializable form for run metadata and session state."""
        return {
            "head": self.head,
            "stash": self.stash,
            "snapshot": self.snapshot,
            # Older injected restore-point providers did not populate ``ref``;
            # expose the stable legacy name in that case when a stash exists.
            "ref": self.ref or (RESTORE_REF if self.stash else None),
            "complete": self.snapshot is not None,
            "dirty_at_start": self.dirty,
            "untracked_count": len(self.untracked),
            "created_at": self.created_at,
        }


def _git(
    project_dir: Path,
    *args: str,
    timeout: int = _GIT_TIMEOUT_SECONDS,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Run one git command, returning (exit_code, stdout-or-error-text)."""
    try:
        proc = subprocess.run(
            ["git", "--no-pager", "-C", str(project_dir), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    output = proc.stdout if proc.returncode == 0 else (proc.stderr or proc.stdout)
    return proc.returncode, output.strip()


def _snapshot_worktree(
    root: Path,
    *,
    head: str | None,
    timeout: int,
) -> tuple[str | None, str | None]:
    """Commit the entire working tree (tracked + untracked) to a fresh object.

    Returns ``(sha, error)``. Builds its own index through ``GIT_INDEX_FILE``
    so nothing the user can see changes: the real index, the worktree and the
    stash list are all untouched. Ignored paths stay ignored — build output
    does not belong in a restore point.
    """
    with tempfile.TemporaryDirectory(prefix="mycoder-restore-") as tmp:
        env = {**os.environ, **_IDENTITY_ENV, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        # `git add -A .` against an empty index yields a tree equal to the
        # working tree, which is exactly the restore target. read-tree first
        # would instead make deleted-but-tracked files look restored.
        code, out = _git(root, "add", "-A", "--", ".", timeout=timeout, env=env)
        if code != 0:
            return None, f"git add -A failed: {out}"
        code, tree = _git(root, "write-tree", timeout=timeout, env=env)
        if code != 0:
            return None, f"git write-tree failed: {tree}"
        args = ["commit-tree", tree, "-m", "MyCoder restore point"]
        if head:
            args += ["-p", head]
        code, commit = _git(root, *args, timeout=timeout, env=env)
        if code != 0:
            return None, f"git commit-tree failed: {commit}"
        return commit.splitlines()[-1].strip() or None, None


def capture_restore_point(
    project_dir: str | Path,
    *,
    session_id: str | None = None,
    timeout: int = _GIT_TIMEOUT_SECONDS,
    snapshot_timeout: int = _SNAPSHOT_TIMEOUT_SECONDS,
) -> RestorePoint | None:
    """Record a restore point, or None when the directory is not a checkout.

    Never raises: a missing restore point weakens recovery, it must not break
    the session that was going to be protected by it.
    """
    root = Path(project_dir)
    code, head = _git(root, "rev-parse", "HEAD", timeout=timeout)
    if code != 0:
        return None
    code, status = _git(root, "status", "--porcelain", "-uall", timeout=timeout)
    if code != 0:
        status = ""
    untracked = tuple(sorted(line[3:] for line in status.splitlines() if line.startswith("?? ")))
    dirty = any(not line.startswith("?? ") for line in status.splitlines())

    ref: str | None = None
    stash: str | None = None
    snapshot: str | None = None
    snapshot_error: str | None = None

    if dirty or untracked:
        # Keep a normal stash object for backwards-compatible recovery of
        # tracked edits.  The complete snapshot below additionally preserves
        # untracked files and deleted paths, but a stash is still useful to
        # callers that already know ``git stash apply <ref>``.
        if dirty:
            code, out = _git(root, "stash", "create", timeout=timeout)
            if code == 0 and out:
                stash = out.splitlines()[-1].strip() or None
        snapshot, snapshot_error = _snapshot_worktree(
            root, head=head or None, timeout=snapshot_timeout
        )
        if snapshot is None:
            # Degraded but not useless: a stash object holds the tracked edits.
            # Untracked files are only listed in this path, which is why the
            # hint says so out loud.
            logger.warning("sandbox.restore_snapshot_failed", error=str(snapshot_error))
        if snapshot or stash:
            # One ref per session, so concurrent sessions in the same checkout
            # cannot clobber each other's recovery point.
            ref = session_ref(session_id)
            # The legacy no-session ref points at a real stash so existing
            # ``git stash apply refs/mycoder/restore-point`` workflows remain
            # valid. Session refs prefer the complete snapshot; when no
            # snapshot exists they fall back to the tracked stash object.
            target = stash if session_id is None and stash else (snapshot or stash)
            code, out = _git(root, "update-ref", ref, target or "", timeout=timeout)
            if code != 0:
                # The object still exists; only durability is lost.
                logger.warning("sandbox.restore_ref_failed", ref=ref, error=out)
                ref = None

    return RestorePoint(
        head=head or None,
        stash=stash,
        dirty=dirty,
        untracked=untracked,
        created_at=datetime.now(timezone.utc).isoformat(),
        snapshot=snapshot,
        ref=ref,
        snapshot_error=snapshot_error,
    )


def restore_hint(point: RestorePoint | None, *, limit: int = 5) -> str:
    """Recovery instructions, or "" when there is nothing to say."""
    if point is None or not point.available:
        return ""
    lines = ["[restore point recorded at session start]"]
    if point.head:
        lines.append(f"committed state: HEAD={point.head[:12]}")
    if point.snapshot:
        lines.append(
            f"complete working tree: git restore --source={point.snapshot} --worktree -- . "
            "(tracked edits and untracked files alike)"
        )
    elif point.ref and point.stash:
        lines.append(f"uncommitted work: git stash apply {point.ref} (or {point.stash[:12]})")
    if point.untracked:
        preview = ", ".join(point.untracked[:limit])
        if len(point.untracked) > limit:
            preview += f", … (+{len(point.untracked) - limit} more)"
        if point.snapshot:
            lines.append(f"untracked files (stored in complete snapshot): {preview}")
        else:
            lines.append(f"untracked files (not stored by git, listed only): {preview}")
    return "\n".join(lines)


class WorktreeRecovery:
    """Session-scoped restore point, captured once and then reused.

    Captured lazily on first use so constructing a SandboxManager stays
    side-effect free, and cached so the cost is one short git call per session
    rather than one per command. Thread-safe: the agent runs tools on a pool.
    """

    def __init__(
        self,
        project_dir: str | Path,
        *,
        session_id: str | None = None,
        timeout: int = _GIT_TIMEOUT_SECONDS,
        snapshot_timeout: int = _SNAPSHOT_TIMEOUT_SECONDS,
    ) -> None:
        self.project_dir = Path(project_dir)
        self.session_id = session_id
        self._timeout = timeout
        self._snapshot_timeout = snapshot_timeout
        self._point: RestorePoint | None = None
        self._captured = False
        self._lock = threading.Lock()

    @property
    def point(self) -> RestorePoint | None:
        return self._point

    def capture(self) -> RestorePoint | None:
        with self._lock:
            if self._captured:
                return self._point
            self._captured = True
            try:
                try:
                    self._point = capture_restore_point(
                        self.project_dir,
                        session_id=self.session_id,
                        timeout=self._timeout,
                        snapshot_timeout=self._snapshot_timeout,
                    )
                except TypeError as exc:
                    # Keep compatibility with injected capture functions from
                    # embedders/tests that implement the original signature.
                    if "session_id" not in str(exc):
                        raise
                    self._point = capture_restore_point(
                        self.project_dir,
                        timeout=self._timeout,
                    )
            except Exception as exc:  # noqa: BLE001 - recovery is best-effort
                logger.warning("sandbox.restore_point_failed", error=str(exc))
                self._point = None
            if self._point is None:
                logger.warning(
                    "sandbox.restore_point_unavailable",
                    project_dir=str(self.project_dir),
                )
            else:
                logger.info(
                    "sandbox.restore_point",
                    head=(self._point.head or "")[:12],
                    snapshot=(self._point.snapshot or "")[:12] or None,
                    complete=self._point.snapshot is not None,
                    ref=self._point.ref,
                    dirty_at_start=self._point.dirty,
                    untracked=len(self._point.untracked),
                )
            return self._point

    def hint(self) -> str:
        """Recovery instructions, capturing the point on first call."""
        self.capture()
        return restore_hint(self._point)
