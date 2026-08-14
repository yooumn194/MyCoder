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
from ..observability.budget import TokenBudgetExceeded, TokenBudgetGuard
from .blackboard import Blackboard
from .definition import BUILTIN_SUBAGENTS
from .planner import TaskPlanner
from .replan import DeviationDetector, DeviationType
from .runner import SubagentRunner

CIRCUIT_BREAKER_THRESHOLD = 3
# Seconds a tripped subagent stays open before ONE probe call is allowed
# (half-open). A probe that succeeds closes the breaker; one that fails
# re-opens for another full cooldown. Without this, an open breaker never
# self-heals — a transiently-failing subagent that recovered is never retried.
CIRCUIT_BREAKER_COOLDOWN = 30.0


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
        planner: TaskPlanner | None = None,
        budget_guard: TokenBudgetGuard | None = None,
        model_factory=None,
        deviation_detector=None,
        experience_store=None,
        max_replan_rounds: int = 3,
    ) -> None:
        self.blackboard = blackboard
        self.plan_store = plan_store
        self.model_router = model_router
        self.llm = llm
        self.tools = tools or []
        # P2 model-tier routing: callable tier -> LLM. When set, each sub-agent
        # gets a tier-appropriate model (cheaper for simple roles). None uses
        # the shared orchestrator.llm (backward compatible).
        self.model_factory = model_factory
        # Phase 6: LLM-driven task decomposition. When None, _decompose keeps the
        # original hardcoded single-explorer fallback (backward compatible).
        self._planner = planner
        # P1 dynamic re-planning: deviation detection + graded recovery during
        # sequential execution. Detector default = envelope-grades only (no LLM);
        # experience_store is an optional callable(record: dict) that persists
        # every (deviation, strategy, recovered) for reuse (e.g. memory DB).
        self._deviation = deviation_detector or DeviationDetector()
        self._experience_store = experience_store
        self._max_replan_rounds = max(0, int(max_replan_rounds))
        # Token-budget enforcement (optional; None = no-op, backward compatible).
        self._budget_guard = budget_guard
        self._budget_blown = False
        self._circuit_breaker: dict[str, int] = {}
        # when a subagent tripped the breaker (monotonic timestamp) — for the
        # half-open probe after the cooldown
        self._circuit_open_since: dict[str, float] = {}

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
            budget_guard=self._budget_guard,
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

        assignments = await self._decompose(task, parent_context, subtasks)
        results = await self._execute(assignments, strategy, parent_context)
        return self._synthesize(results, start_time, parent_context)

    # ------------------------------------------------------------- internals

    async def _decompose(
        self, task: str, parent_context: dict, subtasks: Optional[list[dict]] = None
    ) -> list[dict]:
        """Turn the task into subagent assignments.

        Explicit subtasks are passed through untouched (backward compatible).
        With a TaskPlanner injected, the task is decomposed into a dependency
        DAG by the LLM; the plan is also published to the shared blackboard
        (key `{task_id}:plan`) so subagents can read the whole plan. Without a
        planner, the original single-explorer default is kept.
        """
        if subtasks:
            return subtasks
        if self._planner is not None:
            try:
                plan = await self._planner.decompose(task, parent_context)
                assignments = [self._subtask_to_assignment(st) for st in plan]
                task_id = (parent_context or {}).get("task_id")
                if task_id and self.blackboard is not None:
                    try:
                        await self.blackboard.put(task_id, "plan", assignments)
                    except Exception:  # noqa: BLE001 - plan publish is best-effort
                        pass
                return assignments
            except Exception:  # noqa: BLE001 - planner never raises, but be safe
                return [{"subagent_name": "explorer", "task": task}]
        return [{"subagent_name": "explorer", "task": task}]

    @staticmethod
    def _subtask_to_assignment(st) -> dict:
        """SubTask (planner) -> assignment dict (execution queue)."""
        return {
            "subagent_name": st.subagent_name,
            "task": st.instruction,
            "id": st.id,
            "depends_on": st.depends_on,
            "estimated_tokens": st.estimated_tokens,
        }

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
            if self._is_open(name):
                results[name] = self._build_circuit_break_envelope(name)
                continue
            if self._budget_blown:
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
        """Sequential execution with P1 dynamic re-planning.

        Each node is executed; if its envelope deviates, a graded recovery is
        applied (retry retryable failure once / insert a compensation node /
        re-plan the remaining queue), bounded by ``max_replan_rounds`` so a
        stubbornly-deviating plan can't loop forever.
        """
        queue = list(assignments)
        replan_left = self._max_replan_rounds
        while queue:
            if self._budget_blown:
                break
            assign = queue.pop(0)
            name = assign["subagent_name"]
            envelope = await self._run_one(assign, parent_context)
            results[name] = envelope
            self._record_status(name, envelope)

            deviation = self._deviation.check(envelope)
            if deviation is None or replan_left <= 0:
                continue
            replan_left -= 1
            action = await self._apply_recovery(
                deviation, assign, queue, parent_context, results
            )
            self._record_experience(deviation, action, name)
        return results

    async def _apply_recovery(
        self,
        deviation,
        assign: dict,
        queue: list[dict],
        parent_context: dict,
        results: dict,
    ) -> str:
        """Apply the graded recovery for one deviation; return the strategy
        name (recorded into the experience store)."""
        name = assign["subagent_name"]
        if deviation.type == DeviationType.HARD_FAIL:
            # permanent errors skip immediately; retryable ones get ONE retry
            if not deviation.detail.get("retryable"):
                return "hard_fail_skip"
            retry = await self._run_one(assign, parent_context)
            if retry.status in ("success", "partial"):
                results[name] = retry
                self._record_status(name, retry)
                return "hard_fail_retry_recovered"
            self._record_status(name, retry)
            return "hard_fail_retry_failed"
        if deviation.type == DeviationType.SOFT_DRIFT:
            detail = deviation.detail
            completeness = (
                f", completeness={detail.get('completeness')}"
                if detail.get("completeness") is not None
                else ""
            )
            queue.append(
                {
                    "subagent_name": "implementer",
                    "task": (
                        f"上一节点 {name} 结果不完整 "
                        f"(confidence={detail.get('confidence')}{completeness})，"
                        "请基于已有产物修正/补齐并验证。"
                    ),
                    "id": f"fix-{name}",
                    "depends_on": [],
                    "estimated_tokens": 0,
                    # inherit the compensation node's executor so a test/local
                    # orchestrator without an LLM can still run the fix
                    "executor": assign.get("executor"),
                }
            )
            return "soft_drift_insert_fix"
        if deviation.type == DeviationType.GOAL_DRIFT:
            remaining = "；".join(a["task"] for a in queue if a.get("task"))
            if self._planner is not None and remaining:
                try:
                    plan = await self._planner.decompose(
                        f"重新规划剩余任务（上一节点 {name} 偏离目标）：{remaining}",
                        parent_context,
                    )
                    queue[:] = [self._subtask_to_assignment(st) for st in plan]
                    return "goal_drift_replanned"
                except Exception:  # noqa: BLE001 - keep the remaining queue
                    return "goal_drift_replan_failed"
            return "goal_drift_no_planner"
        return "noop"

    def _record_experience(self, deviation, action: str, name: str) -> None:
        """Persist one (deviation, strategy, recovered) record so a memory-backed
        store can reuse the playbook next time (对标 Hermes 经验沉淀). Best-effort."""
        if self._experience_store is None:
            return
        try:
            self._experience_store(
                {
                    "subagent": name,
                    "deviation": deviation.type.value,
                    "detail": deviation.detail,
                    "strategy": action,
                    "recovered": action.endswith("recovered"),
                }
            )
        except Exception:  # noqa: BLE001 - experience write never breaks the run
            pass

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
            # if a previous assignment failed or the budget tripped, stop the chain
            if self._budget_blown or any(
                r.status in ("failed", "cancelled") for r in results.values()
            ):
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
            budget_guard=self._budget_guard,
        )
        return await runner.run()

    def _is_open(self, name: str) -> bool:
        """True when the subagent must be skipped (during the open cooldown).

        After the cooldown elapses the breaker transitions to half-open: the
        next call is allowed through as a probe. The probe either closes the
        breaker (via _record_status on success) or re-opens it for another full
        cooldown (via _record_status on failure).
        """
        if self._circuit_breaker.get(name, 0) < CIRCUIT_BREAKER_THRESHOLD:
            return False
        opened_at = self._circuit_open_since.get(name)
        if opened_at is None:
            self._circuit_open_since[name] = time.monotonic()
            return True
        return time.monotonic() - opened_at < CIRCUIT_BREAKER_COOLDOWN

    def _record_status(self, name: str, envelope: SubagentResultEnvelope) -> None:
        if envelope.status in ("success", "partial"):
            self._circuit_breaker[name] = 0
            self._circuit_open_since.pop(name, None)
        else:
            self._circuit_breaker[name] = self._circuit_breaker.get(name, 0) + 1
            if self._circuit_breaker[name] >= CIRCUIT_BREAKER_THRESHOLD:
                self._circuit_open_since[name] = time.monotonic()
        # Token budget: checked after every subtask; a trip flips _budget_blown
        # so the sequential/conditional loops stop scheduling further work.
        if self._budget_guard is not None:
            session_id = self._session_id_of(envelope)
            try:
                self._budget_guard.check_and_enforce(session_id)
            except TokenBudgetExceeded:
                self._budget_blown = True

    @staticmethod
    def _session_id_of(envelope: SubagentResultEnvelope) -> str:
        meta = getattr(envelope, "meta", None)
        if meta is not None:
            return getattr(meta, "session_id", None) or getattr(meta, "task_id", "unknown")
        return "unknown"

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
            session_id=parent_context.get("session_id"),
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
