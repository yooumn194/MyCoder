"""Workspace-aware path resolution for the host-side file tools.

The agent may read or write /workspace/... paths, the name the project tree
carries INSIDE the sandbox container. Because the container bind-mounts the
host project directory at /workspace, both names denote the same inode, so the
mapping below is a pure string translation — no I/O, no Docker call, valid
before start(), after stop(), and after an idle reap.
"""

import os
from pathlib import Path

from ..sandbox.executor import get_active_manager, get_active_sync


def get_project_root() -> Path:
    """The project root the host-side tools operate on.

    When a sandbox session is active this is the host project directory it
    mounts (Phase 1's project dir); otherwise it falls back to the current
    working directory. PathGuard validates every user-supplied path against
    this root.
    """
    manager = get_active_manager()
    if manager is not None:
        return manager.project_dir
    return Path(os.getcwd())


def resolve_workspace_path(file_path: str) -> Path:
    """Map /workspace/foo.py to {host_project_dir}/foo.py when applicable.

    With a single bind-mounted filesystem the file is already there; nothing
    is copied and no container needs to be running.
    """
    sync = get_active_sync()
    if sync is not None and file_path.startswith("/workspace"):
        return Path(sync.resolve_path(file_path)).resolve()
    return Path(file_path).expanduser().resolve()
