"""Sandboxed command execution with graceful degradation.

Public surface:

    SandboxManager   picks Docker when available, else a user-confirmed
                     degraded local executor (fail-closed)
    DockerSandbox    the hardened Docker backend (real isolation)
    LocalExecutor    the allowlisted host fallback (degraded mode)
    ExecutionResult  structured outcome of one command

The container image itself is built from the repo-root `sandbox/Dockerfile`
(a build artifact, not Python); the executor modules live here so they ship
inside the `mycoder` wheel.
"""

from .docker_executor import (
    DockerSandbox,
    SandboxError,
    SandboxExecTimeout,
    SandboxResourceExhausted,
)
from .executor import (
    BENCHMARK_VERIFY_CMD_ENV,
    BENCHMARK_VERIFY_TIMEOUT_ENV,
    SandboxManager,
    benchmark_verify_command,
    benchmark_verify_timeout,
    get_active_sync,
    run_async,
)
from .local_executor import LocalExecutor
from .models import ExecutionResult
from .policy import ALLOW_RISKY_ENV, ConfirmPolicy
from .recovery import (
    RESTORE_REF,
    RestorePoint,
    WorktreeRecovery,
    capture_restore_point,
    restore_hint,
)
from .sync import WorkspaceSync

__all__ = [
    "ALLOW_RISKY_ENV",
    "BENCHMARK_VERIFY_CMD_ENV",
    "BENCHMARK_VERIFY_TIMEOUT_ENV",
    "RESTORE_REF",
    "ConfirmPolicy",
    "DockerSandbox",
    "ExecutionResult",
    "LocalExecutor",
    "RestorePoint",
    "SandboxError",
    "SandboxExecTimeout",
    "SandboxResourceExhausted",
    "SandboxManager",
    "WorktreeRecovery",
    "WorkspaceSync",
    "benchmark_verify_command",
    "benchmark_verify_timeout",
    "capture_restore_point",
    "get_active_sync",
    "restore_hint",
    "run_async",
]
