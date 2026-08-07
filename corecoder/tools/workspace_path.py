"""Workspace-aware path resolution for the host-side file tools.

The agent may read or write /workspace/... paths that live in the sandbox
volume. When a Docker sandbox is active and its volume still exists, these are
mapped onto the host project directory so the host file tools operate on the
same files the sandbox produced. The mapping is gated on the VOLUME, not the
container: after sandbox.stop() the container is gone but the volume — and
therefore the mapping — stays valid.
"""

from pathlib import Path

from ..sandbox import run_async
from ..sandbox.executor import get_active_sync


def resolve_workspace_path(file_path: str) -> Path:
    """Map /workspace/foo.py to {host_project_dir}/foo.py when applicable.

    resolve_path() itself gates on the volume existing (and only touches docker
    for /workspace paths), so plain host paths never incur a docker call here.
    """
    sync = get_active_sync()
    if sync is not None and file_path.startswith("/workspace"):
        return Path(sync.resolve_path(file_path)).resolve()
    return Path(file_path).expanduser().resolve()


def try_on_demand_sync(file_path: str) -> None:
    """Best-effort targeted copy_out for one /workspace path.

    Lets read_file see a file the sandbox just created without the agent
    having to call sync_workspace() first. Failure is fine — the caller still
    reports the missing file with guidance.
    """
    sync = get_active_sync()
    if sync is None or not sync.volume_exists():
        return
    try:
        run_async(sync.copy_out_files([file_path]))
    except Exception:
        pass
