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

from .docker_executor import DockerSandbox, SandboxError, SandboxResourceExhausted
from .executor import SandboxManager, get_active_sync, run_async
from .local_executor import LocalExecutor
from .models import ExecutionResult
from .policy import ALLOW_RISKY_ENV, ConfirmPolicy
from .sync import WorkspaceSync

__all__ = [
    "ALLOW_RISKY_ENV",
    "ConfirmPolicy",
    "DockerSandbox",
    "ExecutionResult",
    "LocalExecutor",
    "SandboxError",
    "SandboxResourceExhausted",
    "SandboxManager",
    "WorkspaceSync",
    "get_active_sync",
    "run_async",
]
