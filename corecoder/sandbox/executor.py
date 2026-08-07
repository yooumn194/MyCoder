"""Sandbox backend selection and graceful degradation.

The manager picks a backend exactly once per process:

    Docker  -> DockerSandbox        (real isolation; the intended path)
    no Docker -> LocalExecutor      (degraded, host-side, ONLY on operator
                                     confirmation, otherwise fail closed)

The choice is cached so the agent loop doesn't re-probe Docker on every tool
call, and it is serialized with a threading lock so parallel tool calls (the
agent executes tools on a thread pool) can't race to create two backends.
"""

import asyncio
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Protocol

from structlog.contextvars import bind_contextvars

from .docker_executor import DockerSandbox
from .local_executor import LocalExecutor
from .logger import get_logger
from .models import ExecutionResult
from .policy import ConfirmPolicy
from .sync import WorkspaceSync

logger = get_logger()

# Env var that opts into host-side degraded execution without a TTY prompt
# (CI, daemons, unattended containers). Fail-closed unless set.
_ALLOW_LOCAL_EXEC = "CORECODER_ALLOW_LOCAL_EXEC"


class SandboxBackend(Protocol):
    """Common interface implemented by DockerSandbox and LocalExecutor."""

    async def start(self) -> None: ...
    async def execute(self, command: str, timeout: int = 30) -> ExecutionResult: ...
    async def get_diff(self) -> str: ...
    async def stop(self) -> None: ...


class SandboxManager:
    """Chooses Docker when available, else the degraded local executor."""

    def __init__(
        self,
        *,
        project_dir: str | os.PathLike[str] | None = None,
        confirm=None,
        docker_available_check=None,
        session_id: str | None = None,
        policy: ConfirmPolicy | None = None,
    ) -> None:
        self.project_dir = Path(project_dir or os.getcwd()).resolve()
        self._confirm = confirm or self._default_confirm
        # Injectable so tests can force either path deterministically.
        self._docker_check = docker_available_check or self._docker_available
        self._backend: SandboxBackend | None = None
        self._selection_lock = threading.Lock()
        # Every audit event for this sandbox carries one session_id, bound into
        # the structlog contextvars so all downstream loggers pick it up via
        # the merge_contextvars processor (see sandbox/logger.py).
        self.session_id = session_id or str(uuid.uuid4())
        bind_contextvars(session_id=self.session_id)
        # The confirmation policy is per-manager (= per-session): a fresh
        # SandboxManager gets an empty approval cache, so approvals never leak
        # across sessions (P2-1).
        self.policy = policy or ConfirmPolicy()

    async def get(self) -> SandboxBackend | None:
        """Return the chosen backend, or None if degraded mode is refused."""
        if self._backend is not None:
            return self._backend
        # Lock is held across backend creation; to_thread keeps it off-loop.
        if not await asyncio.to_thread(self._selection_lock.acquire):
            return None
        try:
            if self._backend is not None:
                return self._backend  # another call won the race
            if await self._docker_check():
                self._backend = DockerSandbox(project_dir=self.project_dir)
                logger.info("sandbox.backend", backend="docker")
            else:
                confirmed = await asyncio.to_thread(self._confirm)
                if not confirmed:
                    logger.warning(
                        "sandbox.backend_refused",
                        backend="local",
                        reason="not confirmed",
                    )
                    return None  # fail closed: never run host commands silently
                self._backend = LocalExecutor(project_dir=self.project_dir)
                logger.warning(
                    "sandbox.backend",
                    backend="local",
                    reason="docker_unavailable",
                    confirmed=True,
                )
            return self._backend
        finally:
            self._selection_lock.release()

    async def execute(self, command: str, timeout: int = 30) -> ExecutionResult:
        backend = await self.get()
        if backend is None:
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=(
                    "sandbox unavailable: Docker is not reachable and local "
                    "execution was not authorized"
                ),
                blocked=True,
                block_reason="sandbox unavailable",
            )
        return await backend.execute(command, timeout)

    async def get_diff(self) -> str:
        backend = await self.get()
        if backend is None:
            return "(sandbox unavailable)"
        return await backend.get_diff()

    async def stop(self) -> None:
        if self._backend is not None:
            await self._backend.stop()

    def get_sync(self) -> WorkspaceSync | None:
        """WorkspaceSync bound to the active Docker backend.

        Returns None when the backend is the degraded LocalExecutor — its
        "workspace" already IS the host directory, so there is nothing to sync.
        """
        if isinstance(self._backend, DockerSandbox):
            return WorkspaceSync(
                host_project_dir=self.project_dir,
                backend=self._backend,
            )
        return None

    # --------------------------------------------------------------- helpers

    @staticmethod
    async def _docker_available() -> bool:
        """True when the Docker SDK is installed and the daemon responds."""
        try:
            import docker  # lazy: keeps import working without the SDK
        except ImportError:
            return False
        try:
            client = docker.from_env()
            try:
                await asyncio.to_thread(client.ping)
                return True
            finally:
                client.close()
        except Exception:
            return False

    @staticmethod
    def _default_confirm() -> bool:
        """Interactive confirmation; fail-closed when unattended."""
        # Explicit, unattended opt-in.
        if os.getenv(_ALLOW_LOCAL_EXEC, "").strip().lower() in {
            "1", "true", "yes", "on",
        }:
            return True
        # Interactive prompt when attached to a terminal.
        if sys.stdin.isatty():
            try:
                answer = input(
                    "⚠  Docker is unavailable. Run commands on the HOST with "
                    "restricted permissions? This is NOT a security sandbox. "
                    "[y/N] "
                )
                return answer.strip().lower() in {"y", "yes"}
            except EOFError:
                return False
        # Unattended and no explicit opt-in -> fail closed.
        return False


_active_manager: SandboxManager | None = None


def set_active_manager(manager: SandboxManager | None) -> None:
    """Record which manager the sandbox tools are using (for file-tool sync)."""
    global _active_manager
    _active_manager = manager


def get_active_sync() -> WorkspaceSync | None:
    """The current manager's WorkspaceSync, or None (no Docker sandbox).

    read_file / write_file consult this to map /workspace/... paths onto the
    host project directory.
    """
    if _active_manager is None:
        return None
    return _active_manager.get_sync()


def run_async(coro):
    """Bridge sync -> async safely whether or not a loop is already running.

    The agent calls tools synchronously on a thread pool, so normally there is
    no running loop here and asyncio.run() is right. But pytest-asyncio tests
    and library embeddings may already be inside a loop, where asyncio.run()
    would raise RuntimeError; in that case we run the coroutine on a fresh
    thread's own loop instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
