"""Orchestrator — the main agent's orchestration engine.

Executes a task across subagents under a chosen strategy, injects per-instance
UUIDs (v1.0.1 idempotency), enforces a circuit breaker (3 consecutive failures
on a subagent -> it is skipped), and always returns v1.0.1 envelopes.
"""

import asyncio
import datetime
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ..contracts.envelope import ErrorObject, Meta, SubagentResultEnvelope
from .blackboard import Blackboard
from .definition import BUILTIN_SUBAGENTS
from .runner import SubagentRunner

CIRCUIT_BREAKER_THRESHOLD = 3


class OrchestrationStrategy(Enum):
    SEQUENTIAL = "sequential"    # A -> B -> C
    PARALLEL = "parallel"        # A + B + C simultaneously
    CONDITIONAL = "conditional"  # if A success then B else C
    AUTO = "auto"                # auto-select


@dataclass
class OrchestrationResult:
    success: bool
    results: dict[str, SubagentResultEnvelope]
    summary: str
    tokens_used: int
    elapsed_seconds: float


class Orchestrator:
    def __init__(
        self,
        blackboard: Blackboard,
        plan_store=None,
        model_router=None,
        llm=None,
        tools=None,
    ) -> None:
        self.blackboard = blackboard
        self.plan_store = plan_store
        self.model_router = model_router
        self.llm = llm
        self.tools = tools or []
        self._circuit_breaker: dict[str, int] = {}

    # ---------------------------------------------------------------- public

    async def spawn_subagent(
        self,
        subagent_type: str,
        task: str,
        blocking: bool = True,
        parent_context: Optional[dict] = None,
        executor=None,
    ) -> SubagentResultEnvelope:
        """Run one subagent (the SpawnSubagentTool path)."""
        definition = BUILTIN_SUBAGENTS.get(subagent_type)
        if definition is None:
            return self._error_envelope(
                subagent_type,
                ErrorObject(
                    code="UNKNOWN_SUBAGENT",
                    category="permanent",
                    retryable=False,
                    message=f"未知 Subagent 类型: {subagent_type}",
                ),
                parent_context,
            )
        parent_context = parent_context or {"task_id": str(uuid.uuid4())}
        runner = SubagentRunner(
            definition=definition,
            task=task,
            orchestrator=self,
            parent_context=parent_context,
            instance_id=str(uuid.uuid4()),
            executor=executor,
        )
        if not blocking:
            asyncio.create_task(runner.run())
            return self._error_envelope(
                subagent_type,
                ErrorObject(
                    code="SPAWNED",
                    category="transient",
                    retryable=False,
                    message="Subagent 已在后台启动（非阻塞模式）",
                ),
                parent_context,
            )
        result = await runner.run()
        self._record_status(subagent_type, result)
        return result

    async def orchestrate(
        self,
        task: str,
        strategy: OrchestrationStrategy = OrchestrationStrategy.AUTO,
        parent_context: Optional[dict] = None,
        subtasks: Optional[list[dict]] = None,
    ) -> OrchestrationResult:
        """Orchestration main entry."""
        start_time = time.monotonic()
        parent_context = parent_context or {"task_id": str(uuid.uuid4())}

        assignments = self._decompose(task, parent_context, subtasks)
        results = await self._execute(assignments, strategy, parent_context)
        return self._synthesize(results, start_time, parent_context)

    # ------------------------------------------------------------- internals

    def _decompose(self, task: str, parent_context: dict, subtasks: Optional[list[dict]]) -> list[dict]:
        """Turn the task into subagent assignments.

        Subtasks may be provided explicitly (tests / planner output); otherwise
        a single default explorer assignment is made.
        """
        if subtasks:
            return subtasks
        return [{"subagent_name": "explorer", "task": task}]

    async def _execute(
        self,
        assignments: list[dict],
        strategy: OrchestrationStrategy,
        parent_context: dict,
    ) -> dict[str, SubagentResultEnvelope]:
        results: dict[str, SubagentResultEnvelope] = {}
        filtered: list[dict] = []
        for assign in assignments:
            name = assign["subagent_name"]
            if self._circuit_breaker.get(name, 0) >= CIRCUIT_BREAKER_THRESHOLD:
                results[name] = self._build_circuit_break_envelope(name)
                continue
            filtered.append(assign)

        if strategy == OrchestrationStrategy.PARALLEL:
            return await self._execute_parallel(filtered, parent_context, results)
        if strategy == OrchestrationStrategy.CONDITIONAL:
            return await self._execute_conditional(filtered, parent_context, results)
        return await self._execute_sequential(filtered, parent_context, results)

    async def _execute_sequential(
        self, assignments: list[dict], parent_context: dict, results: dict
    ) -> dict[str, SubagentResultEnvelope]:
        for assign in assignments:
            name = assign["subagent_name"]
            envelope = await self._run_one(assign, parent_context)
            results[name] = envelope
            self._record_status(name, envelope)
        return results

    async def _execute_parallel(
        self, assignments: list[dict], parent_context: dict, results: dict
    ) -> dict[str, SubagentResultEnvelope]:
        outcomes = await asyncio.gather(
            *[self._run_one(assign, parent_context) for assign in assignments],
            return_exceptions=True,
        )
        for assign, outcome in zip(assignments, outcomes):
            name = assign["subagent_name"]
            if isinstance(outcome, Exception):
                results[name] = self._error_envelope(
                    name,
                    ErrorObject(
                        code="SUBAGENT_ERROR",
                        category="transient",
                        retryable=True,
                        message=str(outcome),
                    ),
                    parent_context,
                )
            else:
                results[name] = outcome
            self._record_status(name, results[name])
        return results

    async def _execute_conditional(
        self, assignments: list[dict], parent_context: dict, results: dict
    ) -> dict[str, SubagentResultEnvelope]:
        for i, assign in enumerate(assignments):
            name = assign["subagent_name"]
            # if a previous assignment failed, stop the chain
            if any(r.status in ("failed", "cancelled") for r in results.values()):
                break
            envelope = await self._run_one(assign, parent_context)
            results[name] = envelope
            self._record_status(name, envelope)
        return results

    async def _run_one(self, assign: dict, parent_context: dict) -> SubagentResultEnvelope:
        definition = BUILTIN_SUBAGENTS.get(assign["subagent_name"])
        if definition is None:
            return self._error_envelope(
                assign["subagent_name"],
                ErrorObject(
                    code="UNKNOWN_SUBAGENT",
                    category="permanent",
                    retryable=False,
                    message=f"未知 Subagent 类型: {assign['subagent_name']}",
                ),
                parent_context,
            )
        runner = SubagentRunner(
            definition=definition,
            task=assign["task"],
            orchestrator=self,
            parent_context=parent_context,
            instance_id=str(uuid.uuid4()),
            executor=assign.get("executor"),
        )
        return await runner.run()

    def _record_status(self, name: str, envelope: SubagentResultEnvelope) -> None:
        if envelope.status in ("success", "partial"):
            self._circuit_breaker[name] = 0
        else:
            self._circuit_breaker[name] = self._circuit_breaker.get(name, 0) + 1

    def _build_circuit_break_envelope(self, name: str) -> SubagentResultEnvelope:
        return SubagentResultEnvelope(
            schema_version="1.0.1",
            status="failed",
            summary=f"Subagent {name} 已被熔断（连续失败 {CIRCUIT_BREAKER_THRESHOLD} 次），本次跳过",
            confidence="low",
            error=ErrorObject(
                code="CIRCUIT_BREAKER_OPEN",
                category="system_constraint",
                retryable=False,
                message=f"Subagent {name} 连续失败 {CIRCUIT_BREAKER_THRESHOLD} 次，已熔断",
            ),
            meta=self._meta(name, {}),
        )

    def _error_envelope(self, name: str, error: ErrorObject, parent_context: dict) -> SubagentResultEnvelope:
        return SubagentResultEnvelope(
            schema_version="1.0.1",
            status="failed",
            summary=error.message[:500],
            confidence="low",
            error=error,
            meta=self._meta(name, parent_context),
        )

    def _meta(self, name: str, parent_context: dict) -> Meta:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return Meta(
            task_id=parent_context.get("task_id", "unknown"),
            subagent_name=name,
            subagent_instance_id=str(uuid.uuid4()),
            started_at=now,
            finished_at=now,
            duration_ms=0,
        )

    def _synthesize(
        self, results: dict, start_time: float, parent_context: dict
    ) -> OrchestrationResult:
        summaries = "\n".join(
            f"[{name}] {env.status}: {env.summary}" for name, env in results.items()
        )
        success = all(env.status in ("success", "partial") for env in results.values())
        tokens = sum(int((env.usage or {}).get("total_tokens", 0)) for env in results.values())
        return OrchestrationResult(
            success=success,
            results=results,
            summary=summaries or "(no subagents ran)",
            tokens_used=tokens,
            elapsed_seconds=time.monotonic() - start_time,
        )
