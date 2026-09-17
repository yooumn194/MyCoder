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
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from structlog.contextvars import bind_contextvars

from .docker_executor import (
    DEFAULT_IMAGE,
    DockerSandbox,
    _docker_ping_timeout,
    _run_daemon_thread,
    _sandbox_network,
)
from .local_executor import LocalExecutor
from .logger import get_logger
from .models import ExecutionResult
from .policy import ConfirmPolicy, is_unresolvable_target
from .recovery import RestorePoint, WorktreeRecovery
from .sync import WorkspaceSync

logger = get_logger()

# Env var that opts into host-side degraded execution without a TTY prompt
# (CI, daemons, unattended containers). Fail-closed unless set.
_ALLOW_LOCAL_EXEC = "MYCODER_ALLOW_LOCAL_EXEC"

# Seconds a Docker sandbox may sit idle before its container is auto-stopped
# and restarted transparently on the next call. Nothing is lost by the recycle:
# the workspace is the host project directory, not a container-side copy.
# Override with MYCODER_SANDBOX_IDLE_TIMEOUT; 0 disables reaping.
_IDLE_TIMEOUT_ENV = "MYCODER_SANDBOX_IDLE_TIMEOUT"
_IDLE_TIMEOUT_DEFAULT = 600.0

# P0-4: a verification command the HARNESS owns. When set, a benchmark run is
# judged by this command's exit code instead of by whether the model happened to
# run a test itself. "Did the tests pass?" is a fact about the repository, not
# something to infer from the transcript.
_BENCHMARK_VERIFY_CMD_ENV = "MYCODER_BENCHMARK_VERIFY_CMD"
_BENCHMARK_VERIFY_TIMEOUT_ENV = "MYCODER_BENCHMARK_VERIFY_TIMEOUT"

# Public aliases: the API layer names these variables in validation errors and
# warnings, and duplicating the literal there would let the two drift apart.
BENCHMARK_VERIFY_CMD_ENV = _BENCHMARK_VERIFY_CMD_ENV
BENCHMARK_VERIFY_TIMEOUT_ENV = _BENCHMARK_VERIFY_TIMEOUT_ENV

# A verification command can be *configured* and still not be *runnable*.
# SWE-bench's official images ship the repository and its conda env but no test
# runner: pytest is installed by the official harness at evaluation time, and
# this sandbox runs with network_mode=none. `python -m pytest` there exits
# non-zero for a reason that says nothing about the patch, so treating that exit
# code as "the model failed" is a category error. The bundled catch-all probes
# first and, when nothing is runnable, reports this marker with exit code 127;
# `verify()` turns that into a third verdict instead of a red one.
VERIFY_UNAVAILABLE_EXIT_CODE = 127
VERIFY_UNAVAILABLE_MARKER = "MYCODER_VERIFY_UNAVAILABLE"

VERIFY_PASSED = "passed"
VERIFY_FAILED = "failed"
VERIFY_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class VerifyOutcome:
    """The harness's verdict on a benchmark run — three states, not two.

    "could not be run" is a different fact from "ran and failed", and
    collapsing them made every instance whose image lacks a test runtime look
    like a wrong answer in `harness_verification_summary`.
    """

    status: str
    evidence: str
    command: str | None = None

    @property
    def passed(self) -> bool | None:
        """True/False for a run that actually happened, None when it could not.

        Returning False here would re-collapse the third state at the exact
        place callers are most likely to consume it, so the property keeps the
        distinction and `as_dict` simply forwards it.
        """
        if not self.available:
            return None
        return self.status == VERIFY_PASSED

    @property
    def available(self) -> bool:
        return self.status != VERIFY_UNAVAILABLE

    def as_dict(self) -> dict:
        return {
            "command": self.command,
            "status": self.status,
            # None — not False — when unavailable. Downstream code counts these
            # states by name precisely so "never ran" cannot be counted as a
            # failure.
            "passed": self.passed,
            "evidence": self.evidence,
        }


# Which user the container runs as. The image ships a non-root `sandbox` user
# at uid 1000. DockerSandbox remaps that identity to the current host's
# non-root uid when a read-write bind mount would otherwise be unwritable (for
# example on Linux CI runners). The API can request `root` per run for upstream
# SWE-bench images that have no `sandbox` user; this variable is the same switch
# for CLI/deployment use.
SANDBOX_USER_ENV = "MYCODER_SANDBOX_USER"
SANDBOX_USERS = ("sandbox", "root")
_SANDBOX_IMAGE_UID = 1000

# One `git ls-files` per overwrite target, on the confirmation path only. It
# decides whether a command would clobber repository content or a scratch file,
# so it has to answer, but it must not be able to hang a tool call either.
_TRACKED_PROBE_TIMEOUT_SECONDS = 10


def sandbox_user_from_env(default: str = "sandbox") -> str:
    """Container user override, validated; an unknown value falls back."""
    raw = os.getenv(SANDBOX_USER_ENV, "").strip().lower()
    if not raw:
        return default
    if raw not in SANDBOX_USERS:
        logger.warning(
            "sandbox.user_invalid",
            value=raw,
            hint=f"choose one of {', '.join(SANDBOX_USERS)}",
        )
        return default
    return raw


def benchmark_verify_command(override: str | None = None) -> str | None:
    """The harness-owned verification command, or None when not configured.

    ``override`` carries a per-run command (an API request field) and wins over
    the process-wide ``MYCODER_BENCHMARK_VERIFY_CMD``; the environment variable
    stays useful as the default for a whole benchmark batch.
    """
    if override is not None and override.strip():
        return override.strip()
    command = os.getenv(_BENCHMARK_VERIFY_CMD_ENV, "").strip()
    return command or None


def benchmark_verify_timeout(override: int | None = None) -> int:
    """Deadline for the harness-owned verification command.

    A per-run ``override`` wins over ``MYCODER_BENCHMARK_VERIFY_TIMEOUT``.
    """
    if override is not None:
        try:
            return max(1, int(override))
        except (TypeError, ValueError):
            logger.warning("sandbox.verify_timeout_invalid", value=str(override))
    raw = os.getenv(_BENCHMARK_VERIFY_TIMEOUT_ENV, "600").strip()
    try:
        return max(1, int(float(raw)))
    except ValueError:
        return 600


def _idle_timeout_from_env() -> float:
    """Idle timeout for the Docker backend, from env or the default."""
    raw = os.getenv(_IDLE_TIMEOUT_ENV, "").strip()
    if not raw:
        return _IDLE_TIMEOUT_DEFAULT
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning("sandbox.idle_timeout_invalid", value=raw)
        return _IDLE_TIMEOUT_DEFAULT


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
        idle_timeout: float | None = None,
        benchmark_mode: bool = False,
        image: str | None = None,
        user: str = "sandbox",
    ) -> None:
        self.project_dir = Path(project_dir or os.getcwd()).resolve()
        self._confirm = confirm or self._default_confirm
        # Injectable so tests can force either path deterministically.
        self._docker_check = docker_available_check or self._docker_available
        self._backend: SandboxBackend | None = None
        self._selection_lock = threading.Lock()
        # Idle reaping for the Docker backend; None -> MYCODER_SANDBOX_IDLE_TIMEOUT
        # (default 600s, 0 disables). The degraded LocalExecutor has nothing to
        # reap, so this only ever reaches DockerSandbox.
        self._idle_timeout = _idle_timeout_from_env() if idle_timeout is None else max(0.0, float(idle_timeout))
        # Every audit event for this sandbox carries one session_id, bound into
        # the structlog contextvars so all downstream loggers pick it up via
        # the merge_contextvars processor (see sandbox/logger.py).
        self.session_id = session_id or str(uuid.uuid4())
        self._benchmark_mode = benchmark_mode
        self.image = image or None
        # Precedence: an explicit per-run request, then MYCODER_SANDBOX_USER,
        # then the image default.
        self.user = str(user).strip().lower() if user else sandbox_user_from_env()
        if self.user not in SANDBOX_USERS:
            raise ValueError(f"sandbox user must be one of {SANDBOX_USERS}")
        bind_contextvars(session_id=self.session_id)
        # The confirmation policy is per-manager (= per-session): a fresh
        # SandboxManager gets an empty approval cache, so approvals never leak
        # across sessions (P2-1).
        self.policy = policy or ConfirmPolicy(
            # A benchmark workspace is a one-off checkout of a held-out
            # instance: rewriting a SCRATCH file in it (a repro script, a
            # `pytest > build.log` redirect) costs nothing beyond this run, and
            # those shapes are everywhere. An interactive run edits the
            # operator's real project, so there the same command is worth a
            # question.
            #
            # `workspace_overwrite_tracked` is deliberately NOT in this set:
            # that rule fires only when the command would clobber a file the
            # repository tracks — i.e. the very work the run exists to produce.
            # Auto-approving it would make the probe the only protection, and a
            # probe is a guess about a path, not an answer from an operator. So
            # `> src/main.py` still has to be answered, while `> repro.py`
            # does not.
            auto_approve_categories=self._scoped_approvals(),
            # The two questions the overwrite rules ask about a target: does it
            # exist (the noise filter), and is it part of the repository (the
            # severity split). Both resolve against the same directory the
            # command runs in (/workspace == project_dir).
            target_probe=self._workspace_target_exists,
            tracked_probe=self._workspace_target_tracked,
        )
        # The container mounts this very directory read-write, so "undo" is no
        # longer free. A restore point is captured on first use and reported
        # next to dangerous commands (see sandbox/recovery.py).
        self.recovery = WorktreeRecovery(self.project_dir, session_id=self.session_id)
        self.restore_point: RestorePoint | None = None

    def _scoped_approvals(self) -> set[str] | None:
        """Categories this manager may run without asking, or None if interactive.

        `install` is in the set only when the container actually has egress.
        With the shipped ``network_mode=none`` an auto-approved `pip install`
        cannot succeed, so "auto-approved" would mean "silently guaranteed to
        fail"; a denial instead returns the structured hint telling the model to
        use the environment the image already ships.
        """
        if not self._benchmark_mode:
            return None
        approvals = {"workspace_overwrite"}
        if _sandbox_network() != "none":
            approvals.add("install")
        return approvals

    def timeout_scale(self) -> float:
        """Multiplier for caller-owned timeouts, from the selected backend.

        Only the Docker backend can report emulation (an amd64-only image on an
        arm64 host runs under QEMU). The local executor is native, and before a
        backend exists there is nothing to scale.
        """
        scale = getattr(self._backend, "timeout_scale", None)
        return float(scale()) if callable(scale) else 1.0

    @property
    def emulated(self) -> bool:
        """Whether the selected container is running under emulation."""
        return bool(getattr(self._backend, "emulated", False))

    def _workspace_target_exists(self, target: str) -> bool:
        """Does `target` already exist under the workspace root?

        Commands run with cwd=/workspace, which is this project directory, so a
        relative path resolves here. Anything that cannot be resolved is
        reported as existing so the confirmation layer fails closed.
        """
        try:
            path = Path(target)
            if not path.is_absolute():
                path = self.project_dir / path
            return path.exists()
        except OSError:
            return True

    def _workspace_target_tracked(self, target: str) -> bool:
        """Is `target` a path the repository tracks, i.e. part of the patch?

        The severity question, asked only after the policy has established that
        a command would replace an existing file. Rewriting an untracked path
        (a repro script, build output) is throwaway work; rewriting a TRACKED
        file destroys repository content.

        Fails closed on everything it cannot resolve — a glob, a variable, a
        path outside the workspace, a workspace that is not a checkout — because
        "I could not tell" must not read as "safe to overwrite".
        """
        if is_unresolvable_target(target):
            return True
        try:
            path = Path(target)
            if not path.is_absolute():
                path = self.project_dir / path
            relative = path.resolve().relative_to(self.project_dir)
        except (OSError, ValueError):
            return True
        if not (self.project_dir / ".git").exists():
            return True
        try:
            completed = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.project_dir),
                    "ls-files",
                    "--error-unmatch",
                    "--",
                    str(relative),
                ],
                capture_output=True,
                text=True,
                timeout=_TRACKED_PROBE_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError):
            return True
        return completed.returncode == 0

    def _warn_on_uid_mismatch(self) -> None:
        """Record a host-UID remap needed for the bind-mounted workspace.

        The bundled image uses uid 1000.  Linux bind mounts preserve host
        ownership, so a CI runner (or a developer whose uid is not 1000) can
        otherwise get a confusing permission error halfway through a run.
        This is advisory: callers can explicitly choose ``user="root"`` for a
        disposable benchmark checkout, while interactive runs retain the
        least-privilege default.
        """
        if self.user != "sandbox" or not hasattr(os, "getuid"):
            return
        try:
            host_uid = os.getuid()
        except OSError:
            return
        if host_uid != _SANDBOX_IMAGE_UID:
            logger.info(
                "sandbox.uid_remap",
                host_uid=host_uid,
                container_uid=_SANDBOX_IMAGE_UID,
                effective_uid=host_uid,
                hint="bind-mounted workspace will run as the current non-root host uid",
            )

    def capture_restore_point(self) -> RestorePoint | None:
        """Record (once) where the working tree stood when the session started."""
        self.restore_point = self.recovery.capture()
        return self.restore_point

    @property
    def benchmark_mode(self) -> bool:
        """Whether this manager enforces fail-closed benchmark semantics."""
        return self._benchmark_mode

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
                self._backend = DockerSandbox(
                    project_dir=self.project_dir,
                    image=self.image or DEFAULT_IMAGE,
                    user=self.user,
                    idle_timeout=self._idle_timeout,
                    benchmark_limits=self._benchmark_mode,
                )
                self._warn_on_uid_mismatch()
                logger.info(
                    "sandbox.backend",
                    backend="docker",
                    idle_timeout=self._idle_timeout,
                )
            else:
                if self._benchmark_mode:
                    logger.warning(
                        "sandbox.backend_refused",
                        backend="local",
                        reason="docker_required_for_benchmark",
                    )
                    return None
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
                    "sandbox unavailable: Docker is required for benchmark mode"
                    if self._benchmark_mode
                    else "sandbox unavailable: Docker is not reachable and local execution was not authorized"
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

    async def verify(self, command: str, timeout: int = 600) -> VerifyOutcome:
        """Run a harness-owned verification command and classify the verdict.

        P0-4: the harness, not the model, decides what "verified" means and
        reads the verdict from the process exit code. The returned evidence
        string is bounded so it can be embedded in an API error.

        One deliberate exception to "the exit code decides": a command that
        could not be run at all. The bundled catch-all reports that with
        ``VERIFY_UNAVAILABLE_MARKER`` and exit 127, and this returns
        ``unavailable`` — an image that ships no test runner then produces a
        "cannot tell" signal instead of a fake red.
        """
        result = await self.execute(command, timeout)
        tail = (result.stderr or result.stdout or "").strip()[-1200:]
        output = f"{result.stdout or ''}{result.stderr or ''}"
        unavailable = VERIFY_UNAVAILABLE_MARKER in output or (
            # 127 is also a shell's "command not found": both mean the check
            # never actually ran.
            result.exit_code == VERIFY_UNAVAILABLE_EXIT_CODE
            and not result.timed_out
            and not result.blocked
        )
        if unavailable:
            status = VERIFY_UNAVAILABLE
        elif result.exit_code == 0 and not result.timed_out and not result.blocked:
            status = VERIFY_PASSED
        else:
            status = VERIFY_FAILED
        evidence = (
            f"status={status} exit_code={result.exit_code}"
            + (" timed_out=True" if result.timed_out else "")
            + (f" blocked={result.block_reason}" if result.blocked else "")
        )
        if tail:
            evidence += f"\n{tail}"
        return VerifyOutcome(status=status, evidence=evidence, command=command)

    async def stop(self) -> None:
        if self._backend is not None:
            await self._backend.stop()

    def stop_sync(self) -> None:
        """Synchronous teardown for process-exit cleanup (atexit / signals).

        During interpreter shutdown asyncio.run() can no longer create the
        executor threads that the async stop() needs, so exit hooks call this
        instead; backends implement stop_sync() without touching an event loop.
        """
        backend = self._backend
        if backend is None:
            return
        stop_sync = getattr(backend, "stop_sync", None)
        if stop_sync is not None:
            stop_sync()
        else:
            run_async(backend.stop())

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
                # A daemon socket can remain half-open while Docker Desktop
                # starts or loses its VM. ``to_thread`` alone would pin the
                # event-loop executor forever, preventing shutdown and making
                # every new agent run appear hung.
                await asyncio.wait_for(
                    _run_daemon_thread(client.ping),
                    timeout=_docker_ping_timeout(),
                )
                return True
            except TimeoutError:
                logger.warning(
                    "sandbox.docker_ping_timeout",
                    timeout=_docker_ping_timeout(),
                )
                return False
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    # ``close`` normally just releases a socket, but a broken
                    # Unix socket can make urllib3 drain synchronously. Keep
                    # health probing from blocking the event loop on cleanup.
                    try:
                        await _run_daemon_thread(close)
                    except Exception:
                        pass
        except Exception:
            return False

    @staticmethod
    def _default_confirm() -> bool:
        """Interactive confirmation; fail-closed when unattended."""
        # Explicit, unattended opt-in.
        if os.getenv(_ALLOW_LOCAL_EXEC, "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
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


def get_active_manager() -> SandboxManager | None:
    """The manager the sandbox tools are currently using, if any."""
    return _active_manager


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
