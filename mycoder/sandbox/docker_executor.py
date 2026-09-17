"""Docker-backed command execution sandbox.

Security model
--------------
The old `bash` tool used a regex *blacklist*: it pattern-matched a handful of
known-destructive commands and let everything else through. Blacklists are
brittle — the space of "dangerous things a shell can do" is unbounded, so a
clever command always slips past. Phase 1 replaces that with *containment*:
whatever the command tries, Docker and the kernel confine it. The blacklist
survives only as a cheap pre-check (tools/sandbox_tool.py) to avoid burning a
container cycle on an obvious self-destruct command.

Per-launch hardening, all set in `_create_container` and all audited:

    read_only            root filesystem is read-only
    tmpfs /tmp           the one writable scratch space, RAM-backed, 64m cap
    network_mode=none    no ingress, no egress — exfiltration has no channel
    user=sandbox         never root (image also pins USER sandbox)
    no-new-privileges    setuid and friends can never escalate
    cap_drop=ALL         the container holds zero Linux capabilities
    mem_limit            512m interactive / 2g benchmark (MYCODER_SANDBOX_MEM)
    memswap_limit        == RAM, so a big allocation can't spill to disk
    cpu_quota            0.5 core interactive / 2 cores benchmark (_CPU)
    pids_limit=128       a fork bomb dies instead of thrashing the host

The wider benchmark profile exists because a benchmark run replays a vendor
project's own test suite inside a vendor image; the tighter interactive profile
turned ordinary test runs into OOM kills that then read as model failures.
`network_mode` is configurable (MYCODER_SANDBOX_NETWORK) and the benchmark
policy only auto-approves package installs when it is not `none` — an approved
install that cannot reach the network is just a guaranteed failure.

Filesystem
----------
    /workspace  the host project directory, bind-mounted READ-WRITE

There is exactly ONE copy of the repository: the host checkout itself. Every
      actor in the loop — the host file tools (read_file / edit_file / write_file)
      and `execute_in_sandbox` — reads and writes that same tree, so an
      observation is always taken on the state the next action will mutate.

      An earlier design bind-mounted the project read-only at /src and gave the
      container a second, cloned copy in its own volume at /workspace. That
      "two filesystem" split was the single largest source of benchmark
      instability: a host edit was invisible to `pytest` inside the container, a
      container edit was invisible to `get_diff()` on the host, and the two
      copies had to be reconciled by a best-effort copy-back that silently lost
      changes whenever a container was reaped, restarted, or failed mid-copy.
      Containment is now expressed purely as *permissions* (non-root user, zero
      capabilities, read-only rootfs, no network), not as a second copy of the
      tree — which is also how Codex / Claude Code / OpenCode model it.

Timeouts
--------
`docker exec` has no native kill. A runaway command is handled by killing and
recreating the whole container (self-heal) over the same bind-mounted tree, so
no work is lost. Killing the container is the only *certain* way to stop an
arbitrary process: process-level SIGKILL needs a pid manager we don't have, and
cancelling a blocked syscall from a timeout isn't reliable.

NOTE: `docker` is imported lazily so that a host without the Docker SDK can
still import this module and fall back to `local_executor`.
"""

import asyncio
import os
import platform as host_platform
import threading
import time
from pathlib import Path

from .locking import AsyncMutex
from .logger import get_logger
from .models import ExecutionResult

logger = get_logger()

DEFAULT_IMAGE = "mycoder-sandbox:3.12"
_STOP_TIMEOUT = 2  # seconds before docker escalates stop() to SIGKILL
_REMOVE_WAIT_TIMEOUT = 3.0  # auto_remove may still be completing after stop()
_REMOVE_POLL_INTERVAL = 0.05
_CPU_PERIOD = 100000  # cgroup period: cpu_quota=50000 => 0.5 core

# OOM circuit breaker: how many consecutive OOM-kills before we stop retrying.
MAX_OOM_RETRIES = 2
# Container-death heals: how many times we rebuild the container after it dies
# under a command for a NON-OOM reason (pids exhausted, external kill, …).
MAX_HEAL_RETRIES = 3
_MAX_DIFF_FILES = 256
_MAX_DIFF_BYTES = 2 * 1024 * 1024
_IMAGE_PULL_TIMEOUT_DEFAULT = 120.0
_DOCKER_PING_TIMEOUT_DEFAULT = 5.0
# Deadline for one container lifecycle call (create / start / stop+remove).
# These used to be bare `asyncio.to_thread(...)` awaits. The docker SDK does
# set a 60s per-socket timeout, but that is not an overall deadline: a daemon
# that trickles bytes (paused VM, full disk, wedged containerd) resets it on
# every read, so the call — and the API worker awaiting it — can outlive any
# sane budget with no log line and no way out. Generous by default because
# starting a large SWE-bench image can legitimately take a while.
_CONTAINER_OP_TIMEOUT_ENV = "MYCODER_SANDBOX_CONTAINER_OP_TIMEOUT"
_CONTAINER_OP_TIMEOUT_LEGACY_ENV = "MYCODER_SANDBOX_START_TIMEOUT"
_CONTAINER_OP_TIMEOUT_DEFAULT = 180.0

# Deadline for one `docker exec` request. A command deadline (`execute`) bounds
# the *command*; this bounds the *Docker call*, and the two are not the same
# thing: `get_diff()` and `WorkspaceSync._raw_changes()` run git through exec
# with no command budget at all, and in benchmark mode the change listing runs
# after every successful command. Docker exec has no native timeout and the SDK
# resets its socket timeout on every read, so a daemon that trickles bytes
# (paused VM, wedged containerd, full disk) used to leave the API worker
# awaiting forever with nothing in the log. Generous by default because a
# legitimate `pytest` run inside a SWE-bench image is slow.
_EXEC_TIMEOUT_ENV = "MYCODER_SANDBOX_EXEC_TIMEOUT"
_EXEC_TIMEOUT_DEFAULT = 300.0
# `execute()` owns the user-facing command deadline and turns its expiry into a
# timed-out ExecutionResult plus a container restart. It therefore gives the
# inner exec a slightly longer deadline, so that path — not this one — is the
# one that fires for a slow command. Only a daemon that has stopped answering
# altogether lives long enough to hit the inner deadline first.
_EXEC_DEADLINE_GRACE_SECONDS = 30.0

# Resource defaults, split by intent. Interactive work edits the operator's own
# repository one command at a time; a benchmark run replays a real project's
# test suite inside a vendor image that assumes it owns the machine. Both stay
# overridable through MYCODER_SANDBOX_MEM / MYCODER_SANDBOX_CPU.
_SANDBOX_MEM_DEFAULT = "512m"
_BENCHMARK_MEM_DEFAULT = "2g"
_SANDBOX_CPU_DEFAULT = "0.5"
_BENCHMARK_CPU_DEFAULT = "2"
_EMULATION_SLOWDOWN_ENV = "MYCODER_SANDBOX_EMULATION_SLOWDOWN"
_EMULATION_SLOWDOWN_DEFAULT = 2.0


def _mem_limit(benchmark: bool = False) -> str:
    """Container memory cap: MYCODER_SANDBOX_MEM wins, then the mode default.

    Interactive work edits the operator's own repository one command at a time,
    so 512m is the right default. A benchmark run replays a real project's own
    test suite (django / sympy / matplotlib), and 512m turned ordinary test
    runs into OOM kills — which then looked like the model failing the task.
    """
    return os.getenv("MYCODER_SANDBOX_MEM") or (_BENCHMARK_MEM_DEFAULT if benchmark else _SANDBOX_MEM_DEFAULT)


def _cpu_quota(benchmark: bool = False) -> int:
    """CPU quota for a configurable core budget (MYCODER_SANDBOX_CPU)."""
    raw = os.getenv("MYCODER_SANDBOX_CPU") or (_BENCHMARK_CPU_DEFAULT if benchmark else _SANDBOX_CPU_DEFAULT)
    return int(float(raw) * _CPU_PERIOD)


def _sandbox_network() -> str:
    """Container network mode (MYCODER_SANDBOX_NETWORK); default ``none``.

    ``none`` is the safe default and the shipped one. It is also the reason the
    benchmark policy must not auto-approve package installs: an approved
    ``pip install`` cannot succeed without egress, so "auto-approved" would
    only mean "silently guaranteed to fail".
    """
    return os.getenv("MYCODER_SANDBOX_NETWORK", "none").strip().lower() or "none"


def _emulation_slowdown() -> float:
    """Timeout multiplier applied while a container runs under emulation.

    Official SWE-bench images are published for amd64 only, so an arm64 host
    runs them through QEMU. Everything is roughly 2-5x slower, and a verify
    timeout sized for native execution expires on runs that were doing fine.
    """
    try:
        return max(1.0, float(os.getenv(_EMULATION_SLOWDOWN_ENV, _EMULATION_SLOWDOWN_DEFAULT)))
    except ValueError:
        return _EMULATION_SLOWDOWN_DEFAULT


_ARCH_ALIASES = {
    "x86_64": "amd64",
    "amd64": "amd64",
    "aarch64": "arm64",
    "arm64": "arm64",
}


def _host_arch() -> str | None:
    """Host architecture as this process sees it, normalised to Docker's names.

    Only a fallback: on macOS this is unreliable. A Python built for Intel
    reports ``x86_64`` even when it is running under Rosetta on Apple Silicon,
    while the Docker daemon is running arm64 containers. Ask the daemon
    (``_docker_host_arch``) whenever one is available.
    """
    return _ARCH_ALIASES.get((host_platform.machine() or "").strip().lower())


def _docker_host_arch(client: object) -> str | None:
    """The daemon's architecture, or None when it cannot be asked."""
    try:
        info = client.info()
    except Exception:  # noqa: BLE001 - a probe must never break a start
        return None
    raw = str((info or {}).get("Architecture") or "").strip().lower()
    return _ARCH_ALIASES.get(raw)


def _is_emulated(target: str | None, host_arch: str | None = None) -> bool:
    """Would `target` (e.g. ``linux/amd64``) have to be emulated on this host?"""
    if not target or "/" not in target:
        return False
    guest = _ARCH_ALIASES.get(target.rsplit("/", 1)[1].strip().lower())
    host = host_arch or _host_arch()
    if guest is None or host is None:
        return False
    return guest != host


def _pids_limit() -> int:
    """Max processes in the container (MYCODER_SANDBOX_PIDS)."""
    return int(os.getenv("MYCODER_SANDBOX_PIDS", "128"))


def _docker_platform(image: str | None = None) -> str | None:
    """Return an optional target platform for pull/create on mixed-arch hosts.

    Docker CLI honors ``DOCKER_DEFAULT_PLATFORM`` but docker-py's pull/create
    calls do not consistently inherit it.  Keep an explicit MyCoder override
    for API deployments and fall back to Docker's conventional environment
    variable so SWE-bench's amd64 images work on Apple Silicon.
    """
    value = os.getenv("MYCODER_DOCKER_PLATFORM") or os.getenv("DOCKER_DEFAULT_PLATFORM")
    value = value.strip() if value else ""
    if value:
        return value
    # SWE-bench publishes several legacy images as amd64-only.  The adapter
    # already identifies them by the official image prefix, so automatically
    # select emulation on Apple Silicon instead of requiring every API
    # deployment to export a host-specific Docker variable.
    if image and image.lower().startswith("swebench/sweb.eval.") and (
        host_platform.system() == "Darwin"
        or host_platform.machine().lower() in {"arm64", "aarch64"}
    ):
        return "linux/amd64"
    return None


def _image_pull_timeout() -> float:
    """Maximum time allowed for one registry pull.

    Docker's daemon can keep a registry request open indefinitely when the
    host network or registry is unhealthy.  A sandbox start must fail closed
    instead of holding an API worker forever, so this is deliberately a
    bounded, environment-configurable deadline.
    """
    try:
        value = float(os.getenv("MYCODER_SANDBOX_IMAGE_PULL_TIMEOUT", str(_IMAGE_PULL_TIMEOUT_DEFAULT)))
    except (TypeError, ValueError):
        value = _IMAGE_PULL_TIMEOUT_DEFAULT
    return min(3600.0, max(1.0, value))


def _container_op_timeout() -> float:
    """Deadline for one container lifecycle call (create / start / teardown)."""
    raw = os.getenv(_CONTAINER_OP_TIMEOUT_ENV)
    if raw is None:
        # ``MYCODER_SANDBOX_START_TIMEOUT`` shipped in the first lifecycle
        # timeout patch; keep it as a compatibility alias while exposing the
        # more accurate name for create/start/teardown operations.
        raw = os.getenv(_CONTAINER_OP_TIMEOUT_LEGACY_ENV, str(_CONTAINER_OP_TIMEOUT_DEFAULT))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = _CONTAINER_OP_TIMEOUT_DEFAULT
    return min(3600.0, max(1.0, value))


def _exec_timeout() -> float:
    """Deadline for one docker exec request (default 300s, env-configurable)."""
    raw = os.getenv(_EXEC_TIMEOUT_ENV, str(_EXEC_TIMEOUT_DEFAULT))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = _EXEC_TIMEOUT_DEFAULT
    return min(3600.0, max(1.0, value))


def _docker_ping_timeout() -> float:
    """Return the bounded Docker-daemon health-check deadline."""
    raw = os.getenv("MYCODER_DOCKER_PING_TIMEOUT", str(_DOCKER_PING_TIMEOUT_DEFAULT))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = _DOCKER_PING_TIMEOUT_DEFAULT
    return min(60.0, max(0.5, value))


async def _run_daemon_thread(func, *args, **kwargs):
    """Run a blocking Docker SDK call without pinning the event-loop executor.

    ``asyncio.to_thread`` cannot be cancelled while the SDK is blocked in a
    socket read; the default executor then keeps the process alive during API
    shutdown.  A short-lived daemon thread lets the caller enforce a real
    deadline.  The caller is responsible for closing the SDK client on
    timeout, which normally interrupts the underlying request as well.
    """
    loop = asyncio.get_running_loop()
    result: asyncio.Future = loop.create_future()

    def finish(callback, value):
        if loop.is_closed():
            return

        def deliver():
            if not result.done():
                callback(value)

        loop.call_soon_threadsafe(deliver)

    def worker():
        try:
            finish(result.set_result, func(*args, **kwargs))
        except BaseException as exc:  # propagate SDK errors to the coroutine
            finish(result.set_exception, exc)

    threading.Thread(
        target=worker,
        name="mycoder-docker-call",
        daemon=True,
    ).start()
    return await result


# INTERVIEW_NOTE: TaskGroup vs asyncio.gather.
# asyncio.gather does not cancel siblings when one task fails, and it hands you
# "done" futures for cancelled tasks that you must remember to await, so
# exceptions can silently drop. TaskGroup cancels every remaining child the
# moment one fails and *always* waits for all of them before the block exits —
# which is exactly the "both must finish, neither may silently fail" contract
# get_diff() wants. We use it there for the two independent git queries.
#
# For the *single* command deadline (execute) we use asyncio.wait_for instead.
# Its early-return-on-success semantics are what a timeout wants, whereas
# TaskGroup's "wait for every child" contract would make a fast command hang
# until the timeout expired. Teardown (stop) is sequential on purpose: an
# earlier design removed a private workspace volume here, and Docker 409s such a
# removal while the container still references it — interleaving that teardown
# leaked volumes, it was not an optimization. The volume is gone now (one
# filesystem), so the ordering is simply "container first, nothing second".
# TaskGroup belongs only where work is genuinely independent.


class SandboxError(RuntimeError):
    """Raised when the Docker sandbox cannot be provisioned or used."""


class SandboxExecTimeout(SandboxError):
    """One `docker exec` call did not return inside its own deadline.

    Distinct from a command hitting its budget: this says the daemon or the
    container stopped answering mid-request. The generic exec deadline below is
    what makes it reachable, and `_exec_resilient` treats it as a poisoned
    container (teardown + retry on a fresh one) rather than as a command
    failure.
    """


class SandboxResourceExhausted(SandboxError):
    """The container keeps OOM-killing; retrying is pointless. Fail closed."""


class DockerSandbox:
    """A lazily-started, self-healing Docker sandbox.

    Lifecycle:

        await sbx.start()                  # once
        result = await sbx.execute("...")  # any number of times
        diff   = await sbx.get_diff()
        await sbx.stop()                   # graceful teardown + cleanup

    With an idle_timeout set, the container also stops itself after that many
    seconds without execute()/get_diff() and is transparently restarted on the
    next call. Nothing is lost by that recycle: the workspace IS the host
    project directory (bind-mounted read-write), not a copy inside the
    container — see _watchdog_loop.
    """

    def __init__(
        self,
        project_dir: str | os.PathLike[str],
        *,
        image: str = DEFAULT_IMAGE,
        user: str = "sandbox",
        docker_client: object | None = None,
        idle_timeout: float = 0,
        benchmark_limits: bool = False,
    ) -> None:
        self.project_dir = Path(project_dir).resolve()
        self._image = image
        if user not in {"sandbox", "root"}:
            raise ValueError("sandbox user must be 'sandbox' or 'root'")
        self._user = user
        # Which resource profile to size the container with. Benchmark runs
        # replay a vendor project's own test suite and need the wider budget;
        # MYCODER_SANDBOX_MEM / _CPU still override either profile.
        self._benchmark_limits = bool(benchmark_limits)
        # Filled in once the daemon can be asked (see `_docker_host_arch`).
        # None means "fall back to this process's own architecture".
        self._host_arch_cache: str | None = None
        # docker_client is injectable so tests can fake it; when None we lazily
        # build a real client with docker.from_env() and own its lifecycle.
        self._docker = docker_client
        self._owns_client = docker_client is None
        self._container = None
        self._started = False
        # Idle reaping: after `idle_timeout` seconds without execute()/get_diff()
        # the container is stopped by a daemon watchdog thread and restarted on
        # demand. Nothing on disk is affected — the workspace is the host
        # directory itself. 0 disables it. SandboxManager feeds this from
        # MYCODER_SANDBOX_IDLE_TIMEOUT.
        self._idle_timeout = max(0.0, float(idle_timeout))
        self._last_activity = 0.0
        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_stop_evt: threading.Event | None = None
        # Cross-loop mutex: the agent runs tools on a thread pool, each call
        # spinning up its own event loop, so an asyncio.Lock would break.
        self._lock = AsyncMutex()

    # ----------------------------------------------------------- properties

    @property
    def platform(self) -> str | None:
        """Explicit pull/create platform for this image, or None for the host's."""
        return _docker_platform(self._image)

    def _daemon_arch(self) -> str | None:
        """The daemon's architecture, asked once and remembered.

        Deliberately resolved on demand rather than only inside `_start_locked`.
        `SandboxManager.get()` hands this object back *before* the container
        exists, and callers legitimately size their budgets off
        `timeout_scale()` at that point — the API does exactly that when it
        computes the verification timeout. Reading a cold cache there fell back
        to `host_platform.machine()`, which reports x86_64 under Rosetta on
        Apple Silicon, so an emulated run silently reported "native" and got no
        timeout slack: precisely the failure this flag exists to prevent.
        """
        if self._host_arch_cache is None:
            try:
                self._host_arch_cache = _docker_host_arch(self._client())
            except Exception:  # noqa: BLE001 - a probe must never break a call
                # No daemon to ask (or no docker at all): stay unknown and let
                # the process-local fallback apply.
                return None
        return self._host_arch_cache

    @property
    def emulated(self) -> bool:
        """True when this container has to run under QEMU on this host.

        Official SWE-bench images are amd64-only; on Apple Silicon every command
        in them is emulated and several times slower. That is a fact the caller
        should be able to act on (budgets, timeouts, expectations), not a silent
        constant.
        """
        return _is_emulated(self.platform, self._daemon_arch())

    @property
    def mem_limit(self) -> str:
        return _mem_limit(self._benchmark_limits)

    @property
    def cpu_quota(self) -> int:
        return _cpu_quota(self._benchmark_limits)

    def timeout_scale(self) -> float:
        """Multiplier for caller-owned timeouts under emulation."""
        return _emulation_slowdown() if self.emulated else 1.0

    # ---------------------------------------------------------------- public

    async def start(self) -> None:
        """Provision the container (idempotent; safe to call repeatedly)."""
        async with self._lock:
            await self._start_locked()

    async def _start_locked(self) -> None:
        """Start the container, assuming `self._lock` is already held.

        Split from `start()` because `execute()` / `get_diff()` lazily start the
        sandbox while *already holding* the lock — a naive `await self.start()`
        there would re-acquire the non-reentrant AsyncMutex and deadlock (this
        was a real bug: a fresh sandbox's first `execute()` hung forever).
        """
        if self._started:
            return
        client = self._client()
        try:
            await self._ensure_image(client)
            container = await self._container_call(
                "container create", self._create_container, client
            )
            self._container = container
            await self._container_call("container start", container.start)
            self._started = True
            self._last_activity = time.monotonic()
            self._start_watchdog()
            # Ask the daemon, not this process: a Rosetta Python reports
            # x86_64 on Apple Silicon, which would hide the emulation. Cold by
            # this point in the normal flow, but `_daemon_arch` also fills it on
            # demand for callers that never start a container.
            self._daemon_arch()
            logger.info(
                "sandbox.start",
                container_id=_short_id(container.id),
                image=self._image,
                project=str(self.project_dir),
                idle_timeout=self._idle_timeout,
                mem_limit=self.mem_limit,
                cpu=self.cpu_quota / _CPU_PERIOD,
                network=_sandbox_network(),
                platform=self.platform,
                emulated=self.emulated,
            )
            if self.emulated:
                # Not fatal, but it changes what "slow" means for every command
                # in this run: timeouts sized for native execution expire on
                # work that was progressing normally. Say so once, up front.
                logger.warning(
                    "sandbox.emulated",
                    platform=self.platform,
                    host=self._daemon_arch() or host_platform.machine(),
                    timeout_scale=self.timeout_scale(),
                    hint=(
                        "official SWE-bench images are amd64-only; expect "
                        "several times the native runtime"
                    ),
                )
        except BaseException:
            await self._rollback_failed_start()
            raise

    async def _rollback_failed_start(self) -> None:
        """Remove the container created by an incomplete initial start.

        The workspace is the host project directory, so there is no separate
        resource to retain: agent edits already live in the real checkout and
        survive a failed start by construction.
        """
        self._started = False
        container = self._container
        if container is not None:
            if await self._teardown_bounded(container):
                self._container = None
        if self._owns_client:
            # Closing a Docker SDK client is normally tiny, but it may block
            # while a half-open HTTP connection is being torn down. Keep the
            # rollback path from pinning asyncio's shared executor.
            await _run_daemon_thread(self._close_owned_client)
        logger.warning("sandbox.start_rolled_back")

    async def _container_call(self, action: str, func, *args, **kwargs):
        """Run one blocking container lifecycle call under its own deadline.

        ``_run_daemon_thread`` + ``wait_for``, the same pair ``_ensure_image``
        uses: a half-open daemon socket must never hold an API worker. Unlike a
        registry request, a timed-out create/start cannot be retried blindly —
        the worker thread may still finish in the background — so the caller is
        expected to tear the half-built container down and fail closed.
        """
        deadline = _container_op_timeout()
        try:
            return await asyncio.wait_for(
                _run_daemon_thread(func, *args, **kwargs),
                timeout=deadline,
            )
        except TimeoutError as exc:
            logger.warning("sandbox.container_op_timeout", action=action, timeout=deadline)
            raise SandboxError(
                f"Docker {action} did not return within {deadline:g}s; the daemon "
                f"looks wedged. Check `docker version` / the Docker VM and retry, or "
                f"raise {_CONTAINER_OP_TIMEOUT_ENV}."
            ) from exc

    async def _teardown_bounded(self, container) -> bool:
        """Teardown with a deadline; a hung daemon means 'not removed'."""
        try:
            return bool(
                await self._container_call(
                    "container remove", self._teardown_container, container
                )
            )
        except SandboxError as exc:
            logger.warning(
                "sandbox.container_teardown_timeout",
                container_id=_short_id(getattr(container, "id", None)),
                error=str(exc),
            )
            return False

    async def execute(self, command: str, timeout: int = 30) -> ExecutionResult:
        """Run `command` via `sh -c` in /workspace under a hard timeout."""
        if timeout <= 0:
            timeout = 30
        async with self._lock:
            if not self._started:
                await self._start_locked()
            self._last_activity = time.monotonic()
            try:
                # wait_for is the right primitive for a single deadline: on
                # success it returns immediately; on expiry it cancels the
                # wrapped task. The to_thread worker can't actually be killed,
                # so we self-heal the container — see module docstring.
                #
                # The inner exec deadline is this budget plus a grace, so a slow
                # command still ends here (timed-out result + restart) instead of
                # inside _exec, whose failure means "the daemon stopped
                # answering" and reads very differently to a caller.
                return await asyncio.wait_for(
                    self._exec_resilient(
                        ["/bin/sh", "-c", command],
                        timeout=timeout + _EXEC_DEADLINE_GRACE_SECONDS,
                    ),
                    timeout=timeout,
                )
            except TimeoutError:
                logger.warning(
                    "sandbox.timeout",
                    command=_truncate(command),
                    timeout=timeout,
                )
                await self._restart()
                return ExecutionResult(
                    exit_code=-1,
                    stdout="",
                    stderr=f"timed out after {timeout}s",
                    timed_out=True,
                )
            except SandboxExecTimeout as exc:
                # Only reachable when the grace above was not enough (repeated
                # heals). Same user-visible outcome as a command timeout: the
                # healer already rebuilt the container, so there is nothing left
                # to reclaim here.
                logger.warning(
                    "sandbox.timeout",
                    command=_truncate(command),
                    timeout=timeout,
                    error=str(exc),
                )
                return ExecutionResult(
                    exit_code=-1,
                    stdout="",
                    stderr=str(exc),
                    timed_out=True,
                )
            except SandboxError:
                raise

    async def get_diff(self) -> str:
        """Unified diff of all workspace changes, including untracked files.

        The git queries are independent read-only operations, so they run
        concurrently inside a TaskGroup: both must finish before get_diff()
        returns, and a failure in either propagates instead of being dropped.

        The probe up front uses _exec_resilient so a container that died since
        the last command (OOM, pids limit) is self-healed before we diff.
        """
        async with self._lock:
            if not self._started:
                await self._start_locked()
            self._last_activity = time.monotonic()
            probe = await self._exec_resilient(
                ["/bin/sh", "-c", "[ -d /workspace/.git ] && echo yes"]
            )
            if probe.stdout.strip() != "yes":
                return "(workspace is not a git repository; diff unavailable)"
            async with asyncio.TaskGroup() as tg:
                tracked = tg.create_task(
                    self._exec(["/bin/sh", "-c", "git -C /workspace diff --no-color"])
                )
                staged = tg.create_task(
                    self._exec(["/bin/sh", "-c", "git -C /workspace diff --cached --no-color"])
                )
                untracked = tg.create_task(
                    self._exec(
                        [
                            "git",
                            "-C",
                            "/workspace",
                            "ls-files",
                            "--others",
                            "--exclude-standard",
                            "-z",
                        ]
                    )
                )
            parts = [
                tracked.result().stdout.strip(),
                staged.result().stdout.strip(),
            ]
            paths = [
                path
                for path in untracked.result().stdout.split("\0")
                if path
            ]
            if len(paths) > _MAX_DIFF_FILES:
                raise SandboxError(
                    f"diff contains too many untracked files ({len(paths)} > {_MAX_DIFF_FILES})"
                )
            for path in paths:
                result = await self._exec(
                    [
                        "git",
                        "diff",
                        "--no-index",
                        "--no-color",
                        "--binary",
                        "--",
                        "/dev/null",
                        path,
                    ],
                    workdir="/workspace",
                )
                if result.exit_code not in (0, 1):
                    raise SandboxError(
                        f"failed to diff untracked file {path!r}: {result.stderr.strip()}"
                    )
                parts.append(result.stdout.strip())
                if sum(len(part.encode()) for part in parts) > _MAX_DIFF_BYTES:
                    raise SandboxError(
                        f"workspace diff exceeds {_MAX_DIFF_BYTES} bytes"
                    )
            return "\n".join(p for p in parts if p).strip() or "(no changes)"

    async def ensure_started(self) -> None:
        """Make sure the container is running (lazy start; safe anytime).

        Diff/execute call this after an idle-reap stopped the container. The
        workspace is the host checkout and therefore survives reaping by
        construction — a restart must not error out.
        """
        async with self._lock:
            if not self._started:
                await self._start_locked()

    async def stop(self) -> None:
        """Gracefully stop the container and clean up its resources.

        With a single filesystem there is no volume to remove: the workspace is
        the host project directory and outlives the container by design, so
        teardown is just "remove the container" (which is also auto-removed).
        """
        async with self._lock:
            await self._stop_async_locked()

    async def _stop_async_locked(self) -> None:
        """Stop the sandbox without an unbounded executor worker.

        ``_stop_locked`` remains the synchronous atexit path.  API requests
        use this async variant so Docker stop/remove and client close each run
        through the same daemon-thread deadline as startup.  A hung daemon can
        therefore return a bounded error instead of holding the request (or
        the process) indefinitely.
        """
        self._stop_watchdog()
        self._started = False
        container = self._container
        container_removed = True
        if container is not None:
            container_removed = await self._teardown_bounded(container)
            if container_removed:
                self._container = None
        if self._owns_client:
            await _run_daemon_thread(self._close_owned_client)
        logger.info(
            "sandbox.stop",
            container_id=_short_id(getattr(container, "id", None)),
            container_removed=container_removed,
        )

    def stop_sync(self) -> None:
        """Blocking teardown for process-exit cleanup (no event loop needed).

        atexit / signal handlers run while the interpreter is shutting down,
        where asyncio.run() can no longer spawn the executor threads that the
        async stop() depends on (run_in_executor raises "no running event
        loop" / interpreter-shutdown errors). Docker's API is synchronous
        anyway — stop() only used to_thread to keep the loop unblocked — so we
        call the same helpers directly. Takes the same AsyncMutex (synchronously)
        to serialize against an in-flight execute().
        """
        with self._lock.underlying:
            self._stop_locked()

    def _stop_locked(self) -> None:
        """Final resource cleanup; caller must hold ``self._lock``."""
        self._stop_watchdog()
        self._started = False
        container = self._container
        container_removed = True
        if container is not None:
            container_removed = self._teardown_container(container)
            if container_removed:
                self._container = None
        if self._owns_client:
            self._close_owned_client()
        logger.info(
            "sandbox.stop",
            container_id=_short_id(getattr(container, "id", None)),
            container_removed=container_removed,
        )

    # --------------------------------------------------- idle auto-reaping

    def _start_watchdog(self) -> None:
        """Arm the idle-reaper daemon thread (no-op when disabled/armed)."""
        if self._idle_timeout <= 0 or self._watchdog_thread is not None:
            return
        stop_evt = threading.Event()
        self._watchdog_stop_evt = stop_evt
        thread = threading.Thread(
            target=self._watchdog_loop,
            args=(stop_evt,),
            daemon=True,
            name="mycoder-sandbox-idle",
        )
        self._watchdog_thread = thread
        thread.start()

    def _stop_watchdog(self, join_timeout: float = 2.0) -> None:
        """Signal the reaper to stop and wait for it to finish.

        Safe from the reaper thread itself (it triggers _shutdown_idle ->
        stop): joining your own thread would deadlock, so a reaper only clears
        the refs and lets the loop exit on its own.
        """
        evt, self._watchdog_stop_evt = self._watchdog_stop_evt, None
        thread, self._watchdog_thread = self._watchdog_thread, None
        if evt is not None:
            evt.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(join_timeout)

    def _watchdog_loop(self, stop_evt: threading.Event) -> None:
        """Sleep in idle_timeout slices; reap the container when idle.

        The agent runs tools on a thread pool and each call spins up its own
        event loop (run_async), so a background asyncio task would be bound to
        a loop that is gone by the time it fires — hence a plain daemon
        thread. stop()/_shutdown_idle() are serialized with the same
        AsyncMutex, so racing a tool call is safe: the reaper blocks on the
        lock until the in-flight command finishes.
        """
        while not stop_evt.wait(self._idle_timeout):
            if not self._started:
                return  # already stopped; nothing left to reap
            if time.monotonic() - self._last_activity >= self._idle_timeout:
                logger.info(
                    "sandbox.idle_shutdown",
                    idle_timeout=self._idle_timeout,
                    idle_for=int(time.monotonic() - self._last_activity),
                )
                try:
                    asyncio.run(self._shutdown_idle())
                except Exception as exc:  # a reaper must never crash
                    logger.warning("sandbox.idle_shutdown_error", error=str(exc))
                return

    async def _shutdown_idle(self) -> None:
        """Stop the container but keep every byte of work (idle reaping).

        The workspace is the host project directory, so an idle reap is now
        lossless by construction: the next execute() simply starts a fresh
        container over the same tree. (Under the old two-copy design this had
        to carefully preserve the container's private volume, and any copy-back
        that had not happened yet was lost.)
        """
        async with self._lock:
            self._stop_watchdog()  # reaper thread: clears refs, no join
            if self._container is None:
                return
            container = self._container
            self._started = False
            container_removed = await self._teardown_bounded(container)
            if container_removed:
                self._container = None
            else:
                # Keep using the existing handle so execute() can either reach
                # a still-running container or self-heal it. Marking it stopped
                # would overwrite the handle on the next lazy start.
                self._started = True
            if self._owns_client:
                await _run_daemon_thread(self._close_owned_client)
            logger.info(
                "sandbox.idle_stopped",
                container_id=_short_id(container.id),
                container_removed=container_removed,
            )

    # ------------------------------------------------------------- internals

    def _client(self):
        if self._docker is None:
            import docker  # lazy: see module docstring

            self._docker = docker.from_env()
        return self._docker

    async def _ensure_image(self, client) -> None:
        try:
            # Image lookup can block on a half-open daemon socket just like a
            # health ping. Keep it off the event loop and share the short
            # daemon deadline so a missing/unreachable Docker VM fails fast.
            await asyncio.wait_for(
                _run_daemon_thread(client.images.get, self._image),
                timeout=_docker_ping_timeout(),
            )
            return
        except TimeoutError as exc:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - cleanup is best-effort
                    pass
            timeout = _docker_ping_timeout()
            raise SandboxError(
                f"Docker image lookup timed out after {timeout:g}s for {self._image!r}. "
                "Check the Docker daemon and retry."
            ) from exc
        except Exception:
            pass  # missing or unreachable -> attempt a pull below
        logger.info("sandbox.image_pull", image=self._image)
        try:
            platform = _docker_platform(self._image)
            pull_kwargs = {"platform": platform} if platform else {}
            await asyncio.wait_for(
                _run_daemon_thread(client.images.pull, self._image, **pull_kwargs),
                timeout=_image_pull_timeout(),
            )
        except TimeoutError as exc:
            # Closing the client aborts the SDK's socket and prevents a
            # cancelled registry request from lingering in the process.  Do
            # not assume injected test clients implement close().
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - cleanup is best-effort
                    pass
            timeout = _image_pull_timeout()
            raise SandboxError(
                f"Sandbox image {self._image!r} pull timed out after {timeout:g}s. "
                "Check registry connectivity or pre-pull the image; set "
                "MYCODER_SANDBOX_IMAGE_PULL_TIMEOUT to adjust the deadline."
            ) from exc
        except Exception as exc:
            raise SandboxError(
                f"Sandbox image {self._image!r} is not present and could not "
                f"be pulled ({exc}). Build it once with:\n"
                f"  docker build -t {self._image} -f sandbox/Dockerfile sandbox/"
            ) from exc

    def _create_container(self, client):
        """Create (not start) the hardened container. Blocking -> to_thread."""
        runtime_user = self._runtime_user()
        environment = {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "safe.directory",
            "GIT_CONFIG_VALUE_0": "*",
        }
        if runtime_user.isdigit():
            # Numeric host-uid remapping has no passwd entry in the bundled
            # image.  Keep tools that write caches/configs pointed at the
            # writable tmpfs instead of /home/sandbox (uid 1000).
            environment.update({"HOME": "/tmp", "USER": runtime_user})
        create_kwargs = {
            "image": self._image,
            "command": ["sleep", "infinity"],
            "detach": True,
            "user": runtime_user,
            "working_dir": "/workspace",
            # ONE filesystem: the host project tree itself, read-write. The
            # container and the host file tools therefore observe and mutate
            # the same bytes (see the module docstring).
            "volumes": {
                str(self.project_dir): {"bind": "/workspace", "mode": "rw"},
            },
            # The checkout is owned by the host user, not by the in-container
            # `sandbox` uid, so git refuses to operate on it ("dubious
            # ownership"). get_diff() and every `git` command the agent runs
            # need this; it is scoped to the container's environment only.
            "environment": environment,
            "read_only": True,
            "tmpfs": {"/tmp": "size=64m"},
            "network_mode": _sandbox_network(),
            "mem_limit": self.mem_limit,
            # No swap: Docker Desktop defaults to unlimited swap, which would
            # silently let a big allocation "fit" by spilling to disk. capping
            # swap == RAM makes the memory limit actually bite.
            "memswap_limit": self.mem_limit,
            "cpu_quota": self.cpu_quota,
            "cpu_period": _CPU_PERIOD,
            "pids_limit": _pids_limit(),
            "security_opt": ["no-new-privileges:true"],
            "cap_drop": ["ALL"],
            "hostname": "sandbox",
            "auto_remove": True,
            "labels": {"mycoder.sandbox": "1"},
        }
        platform = _docker_platform(self._image)
        if platform:
            create_kwargs["platform"] = platform
        return client.containers.create(**create_kwargs)

    def _runtime_user(self) -> str:
        """Return a writable, non-root uid for a read-write bind mount.

        The image's ``sandbox`` user is fixed at uid 1000, while GitHub and
        many Linux hosts use uid 1001 (or another value).  Docker preserves
        host ownership on bind mounts, so keeping uid 1000 would make every
        edit fail with ``Permission denied``.  Running as the current host uid
        preserves least privilege and is still constrained to the mounted
        project; explicit ``user="root"`` remains the only way to request
        root.  uid 0 deliberately keeps the image user rather than silently
        escalating a root-owned host process.
        """
        if self._user != "sandbox" or not hasattr(os, "getuid"):
            return self._user
        try:
            host_uid = os.getuid()
        except OSError:
            return self._user
        if host_uid > 0 and host_uid != 1000:
            return str(host_uid)
        return self._user

    async def _exec_resilient(
        self, argv: list[str], *, timeout: float | None = None
    ) -> ExecutionResult:
        """Run argv, self-healing a dead container and breaking on repeated OOM.

        A command can take the container down in two distinct ways:

          * the container dies outright (docker exec raises) — rebuild it over
            the same host directory and retry, up to MAX_HEAL_RETRIES;
          * the container is OOM-killed — retry up to MAX_OOM_RETRIES, then
            raise SandboxResourceExhausted instead of looping forever.

        A bare exit code 137 is deliberately NOT treated as OOM: `docker kill`,
        a `timeout` wrapper or a SIGKILL from a script all produce 137 while
        State.OOMKilled stays False. Only the kernel's OOM killer sets it.

        An exec that outlives its own deadline is the third way: the request is
        gone, the container is suspect, so it heals exactly like a dead one.
        """
        oom_count = 0
        heal_count = 0
        while True:
            try:
                result = await self._exec(argv, timeout=timeout)
            except SandboxExecTimeout as exc:
                heal_count += 1
                if heal_count > MAX_HEAL_RETRIES:
                    raise
                logger.warning(
                    "sandbox.exec_timeout_recover",
                    error=str(exc),
                    heal_attempt=heal_count,
                )
                await self._restart()
                continue
            except SandboxError:
                raise
            except Exception as exc:
                heal_count += 1
                if heal_count > MAX_HEAL_RETRIES:
                    raise SandboxError(
                        f"container kept dying; giving up after {heal_count} heal attempts"
                    ) from exc
                logger.warning(
                    "sandbox.exec_error",
                    error=str(exc),
                    heal_attempt=heal_count,
                )
                await self._restart()
                continue

            if (
                result.exit_code == 137
                and result.container_id
                and self._is_oom_killed(result.container_id)
            ):
                oom_count += 1
                if oom_count > MAX_OOM_RETRIES:
                    logger.error(
                        "sandbox.oom_circuit_break",
                        oom_count=oom_count,
                        command=_truncate(" ".join(argv)),
                    )
                    raise SandboxResourceExhausted(
                        f"container was OOM-killed {oom_count} consecutive times. "
                        f"Raise MYCODER_SANDBOX_MEM (currently {self.mem_limit}) "
                        f"or reduce concurrency."
                    )
                logger.warning("sandbox.oom_retry", attempt=oom_count)
                await self._restart()
                continue

            return result

    def _is_oom_killed(self, container_id: str) -> bool:
        """True only when docker reports the container was OOM-killed.

        exit_code 137 is ambiguous on its own (docker kill, `timeout`, scripts
        sending SIGKILL); the container's State.OOMKilled flag is the
        authoritative signal. Returns False on any inspect failure — a 137 we
        cannot verify is treated as non-OOM and never trips the breaker.
        """
        try:
            container = self._client().containers.get(container_id)
            return bool(container.attrs.get("State", {}).get("OOMKilled", False))
        except Exception:
            return False

    async def _exec(
        self,
        argv: list[str],
        *,
        workdir: str = "/workspace",
        timeout: float | None = None,
    ) -> ExecutionResult:
        """Run argv inside the container; only docker I/O leaves the loop.

        Bounded by its own deadline because docker exec has none: without it,
        any caller that is not `execute()` — and `get_diff()` and
        `WorkspaceSync._raw_changes()` are both in the hot path — could await a
        wedged daemon forever. `execute()` passes its command budget (plus a
        grace) so a slow command still reports as a timeout rather than as a
        sandbox failure.
        """
        if self._container is None:
            raise SandboxError("sandbox container is not running")
        deadline = _exec_timeout() if timeout is None else timeout
        started = time.monotonic()
        # A cancelled ``to_thread`` coroutine still waits for its worker during
        # interpreter shutdown.  Docker exec has no native timeout, so the
        # caller cancels this daemon thread on a command deadline and then
        # recreates the container; keeping the worker daemonised makes that
        # deadline real for the API process as well.
        try:
            exit_code, output = await asyncio.wait_for(
                _run_daemon_thread(
                    self._container.exec_run,
                    argv,
                    workdir=workdir,
                    demux=True,
                ),
                timeout=deadline,
            )
        except TimeoutError as exc:
            logger.warning(
                "sandbox.exec_timeout",
                argv=_truncate(" ".join(argv)),
                timeout=deadline,
                container_id=_short_id(self._container.id),
            )
            raise SandboxExecTimeout(
                f"docker exec did not return within {deadline:g}s; the container "
                f"stopped answering. Raise {_EXEC_TIMEOUT_ENV} for a genuinely "
                "slower command."
            ) from exc
        if isinstance(output, tuple):
            stdout_b, stderr_b = output
        else:
            stdout_b, stderr_b = output, None
        return ExecutionResult(
            exit_code=exit_code,
            stdout=_decode(stdout_b),
            stderr=_decode(stderr_b),
            duration_ms=int((time.monotonic() - started) * 1000),
            container_id=self._container.id,
        )

    async def _restart(self) -> None:
        """Kill the poisoned container and recreate it.

        The workspace is a bind mount of the host project directory, so a
        restart is bit-for-bit lossless: the new container sees exactly the
        tree the old one was editing.
        """
        client = self._client()
        old, self._container = self._container, None
        if old is not None:
            removed = await self._teardown_bounded(old)
            if not removed:
                self._container = old
                raise SandboxError(
                    "failed to remove poisoned sandbox container; restart aborted"
                )
        container = await self._container_call(
            "container create", self._create_container, client
        )
        self._container = container
        await self._container_call("container start", container.start)
        logger.warning("sandbox.restart", container_id=_short_id(container.id))

    @staticmethod
    def _teardown_container(container) -> bool:
        try:
            container.stop(timeout=_STOP_TIMEOUT)
        except Exception as exc:
            if not _is_not_found(exc):
                logger.warning(
                    "sandbox.container_stop_failed",
                    container_id=_short_id(getattr(container, "id", None)),
                    error=str(exc),
                )
        try:
            container.remove(force=True)
            return True
        except Exception as exc:
            # auto_remove commonly wins the race after stop(); a 404 means the
            # desired final state was reached and is therefore a success.
            if _is_not_found(exc):
                return True
            if _is_removal_in_progress(exc):
                deadline = time.monotonic() + _REMOVE_WAIT_TIMEOUT
                while time.monotonic() < deadline:
                    time.sleep(_REMOVE_POLL_INTERVAL)
                    try:
                        container.reload()
                    except Exception as reload_exc:
                        if _is_not_found(reload_exc):
                            return True
                        logger.warning(
                            "sandbox.container_remove_check_failed",
                            container_id=_short_id(getattr(container, "id", None)),
                            error=str(reload_exc),
                        )
                        return False
            logger.warning(
                "sandbox.container_remove_failed",
                container_id=_short_id(getattr(container, "id", None)),
                error=str(exc),
            )
            return False

    def _close_owned_client(self) -> None:
        """Close an owned Docker client and make the next use create a new one."""
        if not self._owns_client or self._docker is None:
            return
        client = self._docker
        self._docker = None
        try:
            client.close()
        except Exception as exc:
            logger.warning("sandbox.client_close_failed", error=str(exc))


def _short_id(container_id: str | None) -> str:
    return (container_id or "")[:12] or "unknown"


def _is_not_found(exc: Exception) -> bool:
    """Recognize Docker's NotFound without importing the SDK at module load."""
    response = getattr(exc, "response", None)
    return (
        exc.__class__.__name__ == "NotFound"
        or getattr(exc, "status_code", None) == 404
        or getattr(response, "status_code", None) == 404
    )


def _is_removal_in_progress(exc: Exception) -> bool:
    """True for Docker's transient auto-remove 409 response."""
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None) or getattr(
        response, "status_code", None
    )
    return status == 409 and "removal" in str(exc).lower() and "progress" in str(exc).lower()


def _decode(raw: bytes | None) -> str:
    return (raw or b"").decode("utf-8", errors="replace")


def _truncate(text: str, limit: int = 256) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
