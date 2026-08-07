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
    mem_limit=512m       the OOM killer bounds memory abuse
    memswap_limit=512m   no swap — a big allocation can't spill to disk
    cpu_quota=50000      (50ms per 100ms period => 0.5 core) bounds cpu abuse
    pids_limit=128       a fork bomb dies instead of thrashing the host

Filesystem split
----------------
    /src        host project, bind-mounted READ-ONLY (the safe reading window)
    /workspace  a fresh Docker volume, seeded with `git clone /src` and owned
                by the sandbox user — the only writable place

So a compromise can read the project but cannot mutate it, and can never reach
anything outside the container.

Timeouts
--------
`docker exec` has no native kill. A runaway command is handled by killing and
recreating the whole container (self-heal) on the same workspace volume, so no
work is lost. Killing the container is the only *certain* way to stop an
arbitrary process: process-level SIGKILL needs a pid manager we don't have, and
cancelling a blocked syscall from a timeout isn't reliable.

NOTE: `docker` is imported lazily so that a host without the Docker SDK can
still import this module and fall back to `local_executor`.
"""

import asyncio
import os
import time
import uuid
from pathlib import Path

from .locking import AsyncMutex
from .logger import get_logger
from .models import ExecutionResult

logger = get_logger()

DEFAULT_IMAGE = "corecoder-sandbox:3.12"
_STOP_TIMEOUT = 2  # seconds before docker escalates stop() to SIGKILL
_CPU_PERIOD = 100000  # cgroup period: cpu_quota=50000 => 0.5 core

# OOM circuit breaker: how many consecutive OOM-kills before we stop retrying.
MAX_OOM_RETRIES = 2
# Container-death heals: how many times we rebuild the container after it dies
# under a command for a NON-OOM reason (pids exhausted, external kill, …).
MAX_HEAL_RETRIES = 3


def _mem_limit() -> str:
    """Container memory cap, env-configurable (CORECODER_SANDBOX_MEM)."""
    return os.getenv("CORECODER_SANDBOX_MEM", "512m")


def _cpu_quota() -> int:
    """CPU quota for a configurable core budget (CORECODER_SANDBOX_CPU)."""
    return int(float(os.getenv("CORECODER_SANDBOX_CPU", "0.5")) * _CPU_PERIOD)


def _pids_limit() -> int:
    """Max processes in the container (CORECODER_SANDBOX_PIDS)."""
    return int(os.getenv("CORECODER_SANDBOX_PIDS", "128"))

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
# until the timeout expired. And teardown (stop) is sequential on purpose:
# Docker 409s a volume removal while a container still references it, so the
# container must be gone before its volume is — "concurrent" there was a bug
# that leaked volumes, not an optimization. TaskGroup belongs only where work
# is genuinely independent.


class SandboxError(RuntimeError):
    """Raised when the Docker sandbox cannot be provisioned or used."""


class SandboxResourceExhausted(SandboxError):
    """The container keeps OOM-killing; retrying is pointless. Fail closed."""


class DockerSandbox:
    """A lazily-started, self-healing Docker sandbox.

    Lifecycle:

        await sbx.start()                  # once
        result = await sbx.execute("...")  # any number of times
        diff   = await sbx.get_diff()
        await sbx.stop()                   # graceful teardown + cleanup
    """

    def __init__(
        self,
        project_dir: str | os.PathLike[str],
        *,
        image: str = DEFAULT_IMAGE,
        docker_client: object | None = None,
    ) -> None:
        self.project_dir = Path(project_dir).resolve()
        self._image = image
        # docker_client is injectable so tests can fake it; when None we lazily
        # build a real client with docker.from_env() and own its lifecycle.
        self._docker = docker_client
        self._owns_client = docker_client is None
        self._container = None
        self._volume_name = f"corecoder-ws-{uuid.uuid4().hex[:12]}"
        self._started = False
        # Snapshot of /workspace paths present right after provisioning. The
        # sync layer subtracts it when it must fall back to `docker diff` (a
        # non-git /src), so the provisioning copy isn't reported as "changes".
        self._fs_baseline: set[str] = set()
        # Cross-loop mutex: the agent runs tools on a thread pool, each call
        # spinning up its own event loop, so an asyncio.Lock would break.
        self._lock = AsyncMutex()

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
        await self._ensure_image(client)
        container = await asyncio.to_thread(self._create_container, client)
        self._container = container
        await asyncio.to_thread(container.start)
        await self._provision_workspace()
        self._fs_baseline = await self._snapshot_workspace_paths()
        self._started = True
        logger.info(
            "sandbox.start",
            container_id=_short_id(container.id),
            image=self._image,
            project=str(self.project_dir),
        )

    async def execute(self, command: str, timeout: int = 30) -> ExecutionResult:
        """Run `command` via `sh -c` in /workspace under a hard timeout."""
        if timeout <= 0:
            timeout = 30
        async with self._lock:
            if not self._started:
                await self._start_locked()
            try:
                # wait_for is the right primitive for a single deadline: on
                # success it returns immediately; on expiry it cancels the
                # wrapped task. The to_thread worker can't actually be killed,
                # so we self-heal the container — see module docstring.
                return await asyncio.wait_for(
                    self._exec_resilient(["/bin/sh", "-c", command]),
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
            except SandboxError:
                raise

    async def get_diff(self) -> str:
        """Unified diff of the workspace working tree (tracked changes).

        The two git queries are independent read-only operations, so they run
        concurrently inside a TaskGroup: both must finish before get_diff()
        returns, and a failure in either propagates instead of being dropped.

        The probe up front uses _exec_resilient so a container that died since
        the last command (OOM, pids limit) is self-healed before we diff.
        """
        async with self._lock:
            if not self._started:
                await self._start_locked()
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
            parts = [
                tracked.result().stdout.strip(),
                staged.result().stdout.strip(),
            ]
            return "\n".join(p for p in parts if p).strip() or "(no changes)"

    async def stop(self) -> None:
        """Gracefully stop the container and clean up its resources.

        Teardown is deliberately SEQUENTIAL, not concurrent: Docker refuses to
        remove a volume while any container still references it (409 "volume is
        in use"), even with force=True — an earlier attempt that fanned these
        two steps out with a TaskGroup leaked the workspace volume on every
        stop. Remove the container first, then the volume.
        """
        async with self._lock:
            if self._container is None:
                return
            container, self._container = self._container, None
            self._started = False
            await asyncio.to_thread(self._teardown_container, container)
            await asyncio.to_thread(self._remove_workspace_volume)
            if self._owns_client:
                await asyncio.to_thread(self._client().close)
            logger.info("sandbox.stop", container_id=_short_id(container.id))

    # ------------------------------------------------------------- internals

    def _client(self):
        if self._docker is None:
            import docker  # lazy: see module docstring

            self._docker = docker.from_env()
        return self._docker

    async def _ensure_image(self, client) -> None:
        try:
            client.images.get(self._image)
            return
        except Exception:
            pass  # missing or unreachable -> attempt a pull below
        logger.info("sandbox.image_pull", image=self._image)
        try:
            await asyncio.to_thread(client.images.pull, self._image)
        except Exception as exc:
            raise SandboxError(
                f"Sandbox image {self._image!r} is not present and could not "
                f"be pulled ({exc}). Build it once with:\n"
                f"  docker build -t {self._image} -f sandbox/Dockerfile sandbox/"
            ) from exc

    def _create_container(self, client):
        """Create (not start) the hardened container. Blocking -> to_thread."""
        return client.containers.create(
            self._image,
            command=["sleep", "infinity"],
            detach=True,
            user="sandbox",
            working_dir="/workspace",
            volumes={
                str(self.project_dir): {"bind": "/src", "mode": "ro"},
                self._volume_name: {"bind": "/workspace", "mode": "rw"},
            },
            read_only=True,
            tmpfs={"/tmp": "size=64m"},
            network_mode="none",
            mem_limit=_mem_limit(),
            # No swap: Docker Desktop defaults to unlimited swap, which would
            # silently let a big allocation "fit" by spilling to disk. capping
            # swap == RAM makes the memory limit actually bite.
            memswap_limit=_mem_limit(),
            cpu_quota=_cpu_quota(),
            cpu_period=_CPU_PERIOD,
            pids_limit=_pids_limit(),
            security_opt=["no-new-privileges:true"],
            cap_drop=["ALL"],
            hostname="sandbox",
            auto_remove=True,
            labels={"corecoder.sandbox": "1"},
        )

    async def _exec_resilient(self, argv: list[str]) -> ExecutionResult:
        """Run argv, self-healing a dead container and breaking on repeated OOM.

        A command can take the container down in two distinct ways:

          * the container dies outright (docker exec raises) — rebuild it on
            the same workspace volume and retry, up to MAX_HEAL_RETRIES;
          * the container is OOM-killed — retry up to MAX_OOM_RETRIES, then
            raise SandboxResourceExhausted instead of looping forever.

        A bare exit code 137 is deliberately NOT treated as OOM: `docker kill`,
        a `timeout` wrapper or a SIGKILL from a script all produce 137 while
        State.OOMKilled stays False. Only the kernel's OOM killer sets it.
        """
        oom_count = 0
        heal_count = 0
        while True:
            try:
                result = await self._exec(argv)
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
                        f"Raise CORECODER_SANDBOX_MEM (currently {_mem_limit()}) "
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

    async def _snapshot_workspace_paths(self) -> set[str]:
        """All /workspace paths present right after provisioning (for sync)."""
        try:
            # container.diff() is None (JSON null) when nothing changed — treat
            # as an empty list.
            changes = (await asyncio.to_thread(self._container.diff)) or []
        except Exception:
            return set()
        return {
            c["Path"][len("/workspace/"):]
            for c in changes
            if c.get("Path", "").startswith("/workspace/")
        }

    async def _provision_workspace(self) -> None:
        """Seed the workspace volume from the read-only /src baseline.

        Runs as the `sandbox` user (no root needed): a fresh Docker volume
        inherits the ownership of the image's /workspace, which the Dockerfile
        chowns to sandbox. If the volume already has content — a self-heal
        restart reused it — the workspace is left as-is.
        """
        probe = await self._exec(["/bin/sh", "-c", "ls -A /workspace | wc -l"])
        if (
            probe.exit_code == 0
            and probe.stdout.strip().isdigit()
            and int(probe.stdout.strip()) > 0
        ):
            return
        seed = (
            "if [ -d /src/.git ]; then git clone --quiet /src /workspace; "
            "else cp -a /src/. /workspace/; fi"
        )
        result = await self._exec(["/bin/sh", "-c", seed])
        if result.exit_code != 0:
            raise SandboxError(
                "workspace provisioning failed: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )

    async def _exec(
        self,
        argv: list[str],
        *,
        workdir: str = "/workspace",
    ) -> ExecutionResult:
        """Run argv inside the container; only docker I/O leaves the loop."""
        if self._container is None:
            raise SandboxError("sandbox container is not running")
        started = time.monotonic()
        exit_code, output = await asyncio.to_thread(
            self._container.exec_run,
            argv,
            workdir=workdir,
            demux=True,
        )
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
        """Kill the poisoned container and recreate it on the same volume."""
        client = self._client()
        old, self._container = self._container, None
        if old is not None:
            await asyncio.to_thread(self._teardown_container, old)
        container = await asyncio.to_thread(self._create_container, client)
        self._container = container
        await asyncio.to_thread(container.start)
        logger.warning("sandbox.restart", container_id=_short_id(container.id))

    @staticmethod
    def _teardown_container(container) -> None:
        try:
            container.stop(timeout=_STOP_TIMEOUT)
        except Exception:
            pass
        try:
            container.remove(force=True)
        except Exception:
            pass  # auto_remove may already have cleaned it

    def _remove_workspace_volume(self) -> None:
        try:
            volume = self._client().volumes.get(self._volume_name)
            volume.remove(force=True)
        except Exception:
            pass


def _short_id(container_id: str | None) -> str:
    return (container_id or "")[:12] or "unknown"


def _decode(raw: bytes | None) -> str:
    return (raw or b"").decode("utf-8", errors="replace")


def _truncate(text: str, limit: int = 256) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
