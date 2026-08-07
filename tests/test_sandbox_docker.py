"""Integration tests for DockerSandbox against a real Docker daemon.

Skipped entirely when the docker SDK or daemon is unavailable, or when the
sandbox image has not been built:

    docker build -t corecoder-sandbox:3.12 -f sandbox/Dockerfile sandbox/

Each test verifies a concrete security property of the container (non-root,
read-only root, no network, resource limits, timeout self-heal, clean
teardown) — the properties that a regex blacklist could never provide.
"""

import asyncio
import subprocess
from pathlib import Path

import pytest

from corecoder.sandbox import (
    ConfirmPolicy,
    DockerSandbox,
    SandboxManager,
    SandboxResourceExhausted,
)
from corecoder.tools import get_tool

IMAGE = "corecoder-sandbox:3.12"


def _docker_usable() -> bool:
    try:
        import docker
    except ImportError:
        return False
    try:
        docker.from_env().ping()
        return True
    except Exception:
        return False


def _image_built() -> bool:
    try:
        import docker
    except ImportError:
        return False
    try:
        docker.from_env().images.get(IMAGE)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _docker_usable(),
    reason="docker SDK or daemon not available",
)


def _git(repo, args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _make_repo(tmp_path_factory) -> str:
    repo = tmp_path_factory.mktemp("proj")
    _git(repo, ["init", "-q"])
    (repo / "tracked.txt").write_text("line1\n")
    _git(repo, ["add", "."])
    _git(repo, ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"])
    return str(repo)


@pytest.fixture(scope="module")
async def sandbox(tmp_path_factory):
    """A started DockerSandbox on a fresh git repo; torn down afterwards.

    The shared sandbox is used only by tests that leave the container healthy.
    Destructive tests (OOM, fork bomb) get their own sandbox so they can't
    poison shared state — that separation is itself the lesson: a container
    whose pids or memory you exhaust is not safe to share.
    """
    if not _image_built():
        pytest.skip(
            f"image {IMAGE!r} not built; run: "
            "docker build -t corecoder-sandbox:3.12 -f sandbox/Dockerfile sandbox/"
        )
    sbx = DockerSandbox(project_dir=_make_repo(tmp_path_factory))
    await sbx.start()
    try:
        yield sbx
    finally:
        await sbx.stop()


# --- happy path ------------------------------------------------------------

async def test_execute_basic(sandbox):
    r = await sandbox.execute("echo hello from sandbox")
    assert r.ok
    assert "hello from sandbox" in r.stdout


async def test_lazy_start_on_first_execute(tmp_path_factory):
    """A fresh sandbox must start itself on the first execute.

    Regression: execute() used to call start() while already holding the
    sandbox lock, and the non-reentrant mutex deadlocked on this exact path
    (the explicit start() in the shared fixture hid it from the suite).
    """
    sbx = DockerSandbox(project_dir=_make_repo(tmp_path_factory))
    try:
        r = await asyncio.wait_for(sbx.execute("echo lazy-ok", timeout=10), timeout=30)
        assert r.ok
        assert "lazy-ok" in r.stdout
    finally:
        await sbx.stop()


async def test_workspace_seeded_from_readonly_src(sandbox):
    """The workspace is a git clone of the ro-mounted /src baseline."""
    r = await sandbox.execute("cat tracked.txt")
    assert r.ok
    assert "line1" in r.stdout


async def test_git_available_in_sandbox(sandbox):
    r = await sandbox.execute("git --version")
    assert r.exit_code == 0
    assert "git version" in r.stdout


async def test_rsync_available_in_sandbox(sandbox):
    r = await sandbox.execute("rsync --version")
    assert r.exit_code == 0


# --- security properties ---------------------------------------------------

async def test_runs_as_non_root_user(sandbox):
    r = await sandbox.execute("python3 -c 'import os; print(os.getuid())'")
    assert r.ok
    assert r.stdout.strip() == "1000"  # the sandbox user


async def test_root_fs_is_readonly(sandbox):
    r = await sandbox.execute("touch /usr/bin/evil")
    assert not r.ok
    assert "Read-only file system" in r.stderr + r.stdout


async def test_no_network(sandbox):
    r = await sandbox.execute(
        "python3 -c 'import socket; socket.create_connection((\"1.1.1.1\", 80), 2)'",
        timeout=5,
    )
    assert not r.ok  # no route out of the container


async def test_timeout_kills_and_self_heals(sandbox):
    """A runaway command is killed and the container is recreated seamlessly."""
    r = await sandbox.execute('python3 -c "import time; time.sleep(30)"', timeout=2)
    assert r.timed_out
    r2 = await sandbox.execute("echo still alive")
    assert r2.ok
    assert "still alive" in r2.stdout


# --- diff ------------------------------------------------------------------

async def test_get_diff_returns_unified_diff(sandbox):
    await sandbox.execute("echo line2 >> tracked.txt")
    d = await sandbox.get_diff()
    assert "diff --git" in d
    assert "+line2" in d


# --- destructive limits (own sandbox: must not poison the shared one) ------

async def test_memory_limit_bounds_allocation(tmp_path_factory):
    """The memory cap is enforced — either the exec is OOM-killed (exit != 0)
    or, when the OOM killer takes the container's PID1, the circuit breaker
    fires with SandboxResourceExhausted after MAX_OOM_RETRIES. Both prove the
    512m limit bites."""
    sbx = DockerSandbox(project_dir=_make_repo(tmp_path_factory))
    await sbx.start()
    try:
        try:
            r = await sbx.execute(
                'python3 -c "x = bytearray(700 * 1024 * 1024)"',  # 700MB > 512m+0swap
                timeout=20,
            )
            assert not r.ok  # OOM-killed (137) or MemoryError
        except SandboxResourceExhausted:
            pass  # container kept OOM-killing -> circuit breaker stopped retries
    finally:
        await sbx.stop()


async def test_pids_limit_stops_fork_bomb(tmp_path_factory):
    sbx = DockerSandbox(project_dir=_make_repo(tmp_path_factory))
    await sbx.start()
    try:
        r = await sbx.execute(
            "python3 -c 'import os; [os.fork() for _ in range(500)]'",
            timeout=15,
        )
        assert not r.ok  # fork() fails past the pids limit
    finally:
        await sbx.stop()


# --- P0-1: workspace view unification (end to end) -------------------------

async def test_sandbox_exec_sync_then_host_read(tmp_path_factory):
    """P0-1 acceptance: sandbox touch -> sync_workspace() -> host read_file sees it.

    Also proves execute_in_sandbox reports changed files instead of copying
    them out, and that read_file maps /workspace/... onto the host dir.
    """
    import corecoder.tools.sandbox_tool as st

    repo = _make_repo(tmp_path_factory)
    manager = SandboxManager(
        project_dir=repo,
        policy=ConfirmPolicy(confirmer=lambda cmd, reason: "approved"),
    )
    st._manager = manager
    try:
        tool = get_tool("execute_in_sandbox")
        r = tool.execute(command="echo hello > /workspace/hello.txt", timeout=30)
        assert "[changed files:" in r  # changed_files reported, not copied
        assert "hello.txt" in r

        # the host does NOT have the file yet
        assert not (Path(repo) / "hello.txt").exists()

        sync_tool = get_tool("sync_workspace")
        out = sync_tool.execute(clean=False)
        assert "hello.txt" in out
        assert (Path(repo) / "hello.txt").read_text().strip() == "hello"

        # read_file now maps /workspace/hello.txt onto the host copy
        content = get_tool("read_file").execute(file_path="/workspace/hello.txt")
        assert "hello" in content
    finally:
        await manager.stop()
        st._manager = None


async def test_execute_reports_changed_files_and_delete_hint(tmp_path_factory):
    """A deletion-class command carries a clean=True hint in the output."""
    import corecoder.tools.sandbox_tool as st

    repo = _make_repo(tmp_path_factory)
    manager = SandboxManager(
        project_dir=repo,
        policy=ConfirmPolicy(confirmer=lambda cmd, reason: "approved"),
    )
    st._manager = manager
    try:
        tool = get_tool("execute_in_sandbox")
        r = tool.execute(command="echo x > /workspace/del.txt && rm /workspace/del.txt", timeout=30)
        assert "sync_workspace(clean=True)" in r  # deletion hint
    finally:
        await manager.stop()
        st._manager = None


# --- teardown --------------------------------------------------------------

async def test_stop_removes_container_and_volume(tmp_path_factory):
    import docker

    sbx = DockerSandbox(project_dir=_make_repo(tmp_path_factory))
    await sbx.start()
    cid = sbx._container.id
    volume = sbx._volume_name
    await sbx.stop()

    client = docker.from_env()
    try:
        client.containers.get(cid)
        raise AssertionError("container still present after stop()")
    except docker.errors.NotFound:
        pass
    try:
        client.volumes.get(volume)
        raise AssertionError("workspace volume still present after stop()")
    except docker.errors.NotFound:
        pass
    client.close()
