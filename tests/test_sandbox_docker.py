"""Integration tests for DockerSandbox against a real Docker daemon.

Skipped entirely when the docker SDK or daemon is unavailable, or when the
sandbox image has not been built:

    docker build -t mycoder-sandbox:3.12 -f sandbox/Dockerfile sandbox/

Each test verifies a concrete security property of the container (non-root,
read-only root, no network, resource limits, timeout self-heal, clean
teardown) — the properties that a regex blacklist could never provide.
"""

import asyncio
import os
import subprocess
import time
from pathlib import Path

import pytest

from mycoder.sandbox import (
    ConfirmPolicy,
    DockerSandbox,
    SandboxManager,
    SandboxResourceExhausted,
)
from mycoder.tools import get_tool
from mycoder.tools.sandbox_tool import ExecuteInSandboxTool

IMAGE = "mycoder-sandbox:3.12"


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
            "docker build -t mycoder-sandbox:3.12 -f sandbox/Dockerfile sandbox/"
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


async def test_workspace_is_the_host_checkout(sandbox):
    """One filesystem: /workspace sees the host tree, including .git."""
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
    host_uid = getattr(os, "getuid", lambda: 1000)()
    expected_uid = host_uid if host_uid > 0 and host_uid != 1000 else 1000
    assert r.stdout.strip() == str(expected_uid)
    assert expected_uid != 0


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


async def test_benchmark_tool_rejects_untracked_shadow_in_real_docker(
    tmp_path_factory,
):
    repo = Path(_make_repo(tmp_path_factory))
    source = repo / "package" / "module.py"
    source.parent.mkdir()
    source.write_text(
        "\n".join(f"def function_{index}(): return {index}" for index in range(250)),
        encoding="utf-8",
    )
    _git(repo, ["add", "."])
    _git(
        repo,
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "module"],
    )
    manager = SandboxManager(project_dir=repo, benchmark_mode=True)
    tool = ExecuteInSandboxTool(manager)
    try:
        result = await asyncio.to_thread(
            tool.execute, "cp package/module.py module.py"
        )
        diff = await manager.get_diff()
    finally:
        await manager.stop()

    assert "unsafe benchmark patch" in result
    assert "shadows repository file package/module.py" in result
    assert "new file mode" in diff


async def test_shell_edit_survives_host_read(tmp_path_factory):
    """P0-1: a shell edit lands at its repository path, no copy involved."""
    repo = Path(_make_repo(tmp_path_factory))
    nested = repo / "package" / "module.py"
    nested.parent.mkdir()
    nested.write_text("before\n", encoding="utf-8")
    _git(repo, ["add", "."])
    _git(
        repo,
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "nested"],
    )
    sandbox = DockerSandbox(project_dir=repo)
    await sandbox.start()
    try:
        result = await sandbox.execute("printf 'after\\n' >> package/module.py")
        assert result.ok
    finally:
        await sandbox.stop()

    assert nested.read_text(encoding="utf-8") == "before\nafter\n"
    assert not (repo / "module.py").exists()


async def test_get_diff_includes_untracked_files(sandbox):
    await sandbox.execute("printf 'VALUE = 1\\n' > new_module.py")

    diff = await sandbox.get_diff()

    assert "diff --git" in diff
    assert "new_module.py" in diff
    assert "+VALUE = 1" in diff


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

async def test_shell_writes_are_immediately_visible_to_host_tools(tmp_path_factory):
    """P0-1 acceptance: ONE filesystem, so there is nothing to synchronize.

    A file created by `execute_in_sandbox` is readable by the host file tools
    and present in `git status` in the very same round — the split-brain
    "touch in the container, sync, then read on the host" dance is gone.
    """
    import mycoder.tools.sandbox_tool as st

    repo = Path(_make_repo(tmp_path_factory))
    manager = SandboxManager(
        project_dir=repo,
        policy=ConfirmPolicy(confirmer=lambda cmd, reason: "approved"),
    )
    st._manager = manager
    try:
        tool = get_tool("execute_in_sandbox")
        r = tool.execute(command="echo hello > /workspace/hello.txt", timeout=30)
        assert "[changed files:" in r  # reported, not copied
        assert "hello.txt" in r

        # no sync step: the host already sees it
        assert (repo / "hello.txt").read_text().strip() == "hello"
        content = get_tool("read_file").execute(file_path="/workspace/hello.txt")
        assert "hello" in content

        # ...and the edit is already in the repository diff
        diff = await manager.get_diff()
        assert "hello.txt" in diff
    finally:
        await manager.stop()
        st._manager = None


async def test_host_edit_is_immediately_visible_to_the_shell(tmp_path_factory):
    """P0-1, the other direction: an edit_file write is seen by the next
    command without any push into the container."""
    import mycoder.tools.sandbox_tool as st

    repo = Path(_make_repo(tmp_path_factory))
    manager = SandboxManager(
        project_dir=repo,
        policy=ConfirmPolicy(confirmer=lambda cmd, reason: "approved"),
    )
    st._manager = manager
    try:
        (repo / "shared.txt").write_text("VALUE = 2\n", encoding="utf-8")
        tool = get_tool("execute_in_sandbox")
        r = tool.execute(command="cat /workspace/shared.txt", timeout=30)
        assert "VALUE = 2" in r
    finally:
        await manager.stop()
        st._manager = None


async def test_execute_reports_deletion_in_the_shared_tree(tmp_path_factory):
    """A deletion is already real on the host; the output says so."""
    import mycoder.tools.sandbox_tool as st

    repo = _make_repo(tmp_path_factory)
    manager = SandboxManager(
        project_dir=repo,
        policy=ConfirmPolicy(confirmer=lambda cmd, reason: "approved"),
    )
    st._manager = manager
    try:
        tool = get_tool("execute_in_sandbox")
        r = tool.execute(command="echo x > /workspace/del.txt && rm /workspace/del.txt", timeout=30)
        assert "files deleted" in r
        assert not (Path(repo) / "del.txt").exists()
    finally:
        await manager.stop()
        st._manager = None


# --- teardown --------------------------------------------------------------

async def test_stop_removes_container(tmp_path_factory):
    import docker

    sbx = DockerSandbox(project_dir=_make_repo(tmp_path_factory))
    await sbx.start()
    cid = sbx._container.id
    await sbx.stop()

    client = docker.from_env()
    try:
        client.containers.get(cid)
        raise AssertionError("container still present after stop()")
    except docker.errors.NotFound:
        pass
    client.close()


# --- idle auto-reaping -----------------------------------------------------

async def test_idle_timeout_reaps_container_and_restarts(tmp_path_factory):
    """A real container stops itself after idle_timeout; the next execute()
    restarts it on the SAME volume — agent work written before the idle gap
    survives (auto-close must not discard unsynced changes)."""
    if not _image_built():
        pytest.skip(
            f"image {IMAGE!r} not built; run: "
            "docker build -t mycoder-sandbox:3.12 -f sandbox/Dockerfile sandbox/"
        )
    sbx = DockerSandbox(project_dir=_make_repo(tmp_path_factory), idle_timeout=2)
    await sbx.start()
    try:
        await sbx.execute("echo keep-me > /workspace/keep.txt")
        first_cid = sbx._container.id

        # go idle long enough to be reaped; poll so timing jitter is absorbed
        deadline = time.monotonic() + 8
        while sbx._container is not None and time.monotonic() < deadline:
            await asyncio.sleep(0.3)
        assert sbx._container is None, "container was not reaped within 8s"
        assert sbx._started is False

        # next execute restarts transparently over the same host tree
        r = await sbx.execute("cat /workspace/keep.txt")
        assert r.ok
        assert "keep-me" in r.stdout
        assert sbx._container.id != first_cid  # fresh container, same files
    finally:
        await sbx.stop()
