"""SubagentRunner — executes one subagent in isolation, returning a valid
RFC v1.0.1 envelope.

Every exit path (success, timeout, cancel, exception) returns a validated
SubagentResultEnvelope. The subagent's own output is expected to be a full
envelope (the contract prompt asks for it); if the subagent emits only inner
result data, it is wrapped into an envelope.
"""

import asyncio
import datetime
import time
import uuid
from typing import Any, Callable, Literal, Optional

from structlog.contextvars import bind_contextvars

from ..contracts.envelope import (
    ErrorObject,
    Meta,
    SubagentResultEnvelope,
    parse_envelope,
)
from ..convergence import ConvergenceLimits
from ..observability.budget import TokenBudgetExceeded, TokenBudgetGuard
from ..patch_policy import patch_scope_violation
from .definition import SubagentDefinition


class SubagentRunner:
    def __init__(
        self,
        definition: SubagentDefinition,
        task: str,
        orchestrator: Any,
        parent_context: dict,
        instance_id: Optional[str] = None,
        executor: Optional[Callable[[str, str], Any]] = None,
        budget_guard: Optional[TokenBudgetGuard] = None,
        max_turn_tokens: int | None = None,
        reserved_tokens: int = 0,
        prior_tool_events: list[dict] | None = None,
    ) -> None:
        self.definition = definition
        self.task = task
        self.orchestrator = orchestrator
        self.parent_context = parent_context
        self.instance_id = instance_id or str(uuid.uuid4())
        # executor(task, system_prompt) -> dict; injectable for tests so the
        # Runner needs no real LLM. Default spawns a sub-Agent.
        self._executor = executor
        # Token-budget enforcement (optional; None = no-op, backward compatible).
        self._budget_guard = budget_guard
        self._max_turn_tokens = max_turn_tokens
        self._reserved_tokens = max(0, int(reserved_tokens))
        self._verification_evidence: list[dict] = []
        self._tool_events: list[dict] = []
        self._prior_tool_events = [dict(event) for event in (prior_tool_events or [])]
        self._session_id = (parent_context or {}).get("session_id") or (parent_context or {}).get("task_id", "unknown")
        self._start_time: Optional[float] = None

    async def run(self) -> SubagentResultEnvelope:
        self._start_time = time.monotonic()
        try:
            async with asyncio.timeout(self.definition.timeout_seconds):
                result_data = await self._run_loop()
            envelope = self._build_success_envelope(result_data)
            patch_error = await self._validate_required_patch(envelope)
            if patch_error is not None:
                return patch_error
            verification_error = self._validate_required_verification(envelope)
            return verification_error or envelope
        except TimeoutError:
            return self._build_error_envelope(
                "failed",
                ErrorObject(
                    code="SUBAGENT_TIMEOUT",
                    category="system_constraint",
                    retryable=True,
                    message=f"Subagent 执行超时（{self.definition.timeout_seconds}s）",
                ),
            )
        except TokenBudgetExceeded as exc:
            return self._build_error_envelope(
                "failed",
                ErrorObject(
                    code="TOKEN_BUDGET_EXCEEDED",
                    category="system_constraint",
                    retryable=False,
                    message=str(exc),
                ),
            )
        except asyncio.CancelledError:
            return self._build_error_envelope(
                "cancelled",
                ErrorObject(
                    code="TASK_CANCELLED",
                    category="system_constraint",
                    retryable=False,
                    message="任务被用户取消",
                ),
            )
        except Exception as exc:  # noqa: BLE001 - any failure becomes an envelope
            return self._build_error_envelope(
                "failed",
                ErrorObject(
                    code="SUBAGENT_ERROR",
                    category="transient",
                    retryable=True,
                    message=str(exc),
                ),
            )

    # -------------------------------------------------------------- internals

    def _check_budget(self) -> None:
        """Raise TokenBudgetExceeded before an LLM call when the session budget
        is exhausted (optional guard; no-op when not injected)."""
        if self._budget_guard is not None:
            self._budget_guard.check_and_enforce(self._session_id)

    async def _validate_required_patch(self, envelope: SubagentResultEnvelope) -> SubagentResultEnvelope | None:
        """Reject a benchmark implementer that reports success without a diff.

        The sandbox diff is the source of truth.  This turns an otherwise
        silent analysis-only completion into a retryable orchestration
        deviation before the verifier is scheduled.
        """
        if (
            self.definition.name != "implementer"
            or not self.parent_context.get("require_patch")
            or envelope.status not in ("success", "partial")
        ):
            return None
        manager = getattr(self.orchestrator, "_sandbox_manager", None)
        if manager is None:
            return None
        diff = await manager.get_diff()
        if str(diff or "").lstrip().startswith("diff --git "):
            violation = patch_scope_violation(
                str(diff),
                project_root=getattr(manager, "project_dir", None),
                protect_benchmark_files=True,
            )
            if violation is None:
                return None
            return self._build_error_envelope(
                "failed",
                ErrorObject(
                    code="PATCH_SCOPE_VIOLATION",
                    category="system_constraint",
                    retryable=True,
                    message=f"补丁范围安全检查失败：{violation}",
                ),
            )
        return self._build_error_envelope(
            "failed",
            ErrorObject(
                code="PATCH_REQUIRED",
                category="system_constraint",
                retryable=True,
                message=(
                    "实现节点结束但仓库没有任何 diff；必须实际修改生产代码并确认 git diff 非空。"
                    f"动作证据：{self._patch_failure_evidence()}"
                ),
            ),
        )

    def _patch_failure_evidence(self) -> str:
        """Summarize action-level evidence without exposing arguments/content."""
        events = self.action_evidence
        calls = len(events)
        if calls == 0:
            return "tool_calls=0，模型未产生可解析的工具调用"
        inspections = sum(
            event.get("name") in {"read_file", "grep_search", "list_files"} and bool(event.get("succeeded"))
            for event in events
        )
        mutations = [event for event in events if event.get("name") in {"edit_file", "write_file"}]
        failed = [event for event in mutations if not event.get("mutation")]
        parts = [
            f"tool_calls={calls}",
            f"successful_inspections={inspections}",
            f"mutation_attempts={len(mutations)}",
            f"failed_mutations={len(failed)}",
        ]
        if failed:
            last = failed[-1]
            parts.append(f"last_mutation={last.get('name', 'unknown')}:{last.get('status', 'error')}")
        elif not mutations:
            parts.append("failure=no_mutation_tool_call")
        return "，".join(parts)

    @property
    def action_evidence(self) -> list[dict]:
        """Tool evidence accumulated across retries of the same assignment."""
        return [dict(event) for event in (*self._prior_tool_events, *self._tool_events)]

    def _verification_is_agent_owned(self) -> bool:
        """Must THIS verifier subagent prove it ran a check itself?

        True for a patch pipeline that has no harness-side check. Once the
        harness owns the verdict (``verification_owned_by == "harness"``, i.e.
        a benchmark verification command is configured), the answer is no: the
        command's exit code decides the run, so forcing the subagent to run a
        second check only spends tokens and can fail a run the harness passes.
        Note this deliberately overrides the ``require_patch`` fallback — that
        fallback exists so a patch pipeline is always verified *somewhere*, and
        with harness verification it is.
        """
        if self.definition.name != "verifier":
            return False
        if self.parent_context.get("verification_owned_by") == "harness":
            return False
        return bool(
            self.parent_context.get("require_verification")
            or self.parent_context.get("require_patch")
        )

    def _validate_required_verification(self, envelope: SubagentResultEnvelope) -> SubagentResultEnvelope | None:
        """A real benchmark verifier cannot succeed without an executed check."""
        if (
            self._executor is not None
            or self.definition.name != "verifier"
            or not self.parent_context.get("require_patch")
            or self.parent_context.get("verification_owned_by") == "harness"
            or envelope.status not in ("success", "partial")
        ):
            return None
        if self._verification_evidence:
            # A verifier may run a focused probe first and then try a broader
            # suite that is unavailable in the image.  Keep the successful
            # probe as valid evidence; requiring the *last* command to pass
            # turned correct patches into false failures for that case.
            if any(event.get("succeeded") for event in self._verification_evidence):
                return None
            latest = self._verification_evidence[-1]
            command = str(latest.get("arguments", {}).get("command", "check"))
            return self._build_error_envelope(
                "failed",
                ErrorObject(
                    code="VERIFICATION_FAILED",
                    category="permanent",
                    retryable=False,
                    message=f"验证命令失败，不能宣告补丁成功：{command[:300]}",
                ),
            )
        return self._build_error_envelope(
            "failed",
            ErrorObject(
                code="VERIFICATION_REQUIRED",
                category="system_constraint",
                retryable=True,
                message=("验证节点没有成功执行测试、编译器或静态检查；仅查看 diff 或输出结论不能作为验证证据"),
            ),
        )

    async def _run_loop(self) -> dict:
        # P2 token budget (#10): bind the subagent session id so LLM traces
        # (LLMTracer keys by the session_id contextvar) aggregate under it —
        # the budget guard can then enforce per-subagent usage instead of
        # always reading zero. asyncio.to_thread propagates this context.
        bind_contextvars(session_id=self._session_id)
        self._check_budget()
        system_prompt = self._build_system_prompt()
        if self._executor is not None:
            return await self._executor(self.task, system_prompt)
        return await self._run_sub_agent(system_prompt)

    async def _run_sub_agent(self, system_prompt: str) -> dict:
        """Default: spawn a real sub-Agent with the definition's tool whitelist."""
        llm = getattr(self.orchestrator, "llm", None)
        # P2 model-tier routing (cost): when the orchestrator exposes a
        # model_factory(tier) -> LLM, build a tier-appropriate model for this
        # sub-agent (explorer=fast, implementer=standard, ...). Falls back to
        # the orchestrator's shared LLM when no factory / no model for the tier.
        factory = getattr(self.orchestrator, "model_factory", None)
        if factory is not None:
            llm = factory(self.definition.model_tier) or llm
        if llm is None:
            raise RuntimeError("orchestrator has no llm; inject an executor instead")
        allowed = self.definition.allowed_tools
        tools = [t for t in getattr(self.orchestrator, "tools", []) if t.name in allowed]
        agent_factory = getattr(self.orchestrator, "agent_factory", None)
        # Multi-agent runs use role-scoped token limits. Reusing the API's
        # session-level soft ratio here would apply it twice (for example,
        # 20k/35k over an implementer's 20k allowance => an accidental 11.4k
        # cutoff) and starve useful work.
        convergence_limits = ConvergenceLimits.from_env(self.definition.max_turns)
        if agent_factory is not None:
            sub = agent_factory.build(
                llm=llm,
                tools=tools,
                max_rounds=self.definition.max_turns,
                max_context_tokens=self.definition.max_context_tokens,
                reasoning_strategy=getattr(self.orchestrator, "reasoning_strategy", None),
                budget_guard=self._budget_guard,
                convergence_limits=convergence_limits,
                require_mutation=(self.definition.name == "implementer" and bool(self.parent_context.get("require_patch"))),
                require_verification=self._verification_is_agent_owned(),
                strict_tool_choice=bool(self.parent_context.get("strict_tool_choice")),
                max_turn_tokens=self._max_turn_tokens or self.definition.max_tokens,
                reserved_tokens=self._reserved_tokens,
                mutation_reserved_tokens=self.parent_context.get("mutation_reserved_tokens"),
                verification_reserved_tokens=self.parent_context.get("verification_reserved_tokens"),
                tool_tracer=getattr(self.orchestrator, "tool_tracer", None),
                trace_context={
                    "subagent_name": self.definition.name,
                    "subagent_instance_id": self.instance_id,
                },
            )
        else:
            from ..agent import Agent

            sub = Agent(
                llm=llm,
                tools=tools,
                max_rounds=self.definition.max_turns,
                max_context_tokens=self.definition.max_context_tokens,
                reasoning_strategy=getattr(self.orchestrator, "reasoning_strategy", None),
                budget_guard=self._budget_guard,
                convergence_limits=convergence_limits,
                require_mutation=(self.definition.name == "implementer" and bool(self.parent_context.get("require_patch"))),
                require_verification=self._verification_is_agent_owned(),
                strict_tool_choice=bool(self.parent_context.get("strict_tool_choice")),
                max_turn_tokens=self._max_turn_tokens or self.definition.max_tokens,
                reserved_tokens=self._reserved_tokens,
                mutation_reserved_tokens=self.parent_context.get("mutation_reserved_tokens"),
                verification_reserved_tokens=self.parent_context.get("verification_reserved_tokens"),
                tool_tracer=getattr(self.orchestrator, "tool_tracer", None),
                trace_context={
                    "subagent_name": self.definition.name,
                    "subagent_instance_id": self.instance_id,
                },
            )
        raw = await asyncio.to_thread(sub.chat, f"{system_prompt}\n\nTask: {self.task}")
        self._verification_evidence = list(getattr(sub, "verification_evidence", []))
        self._tool_events = list(getattr(sub, "tool_events", []))
        return await self._ensure_envelope(raw, llm)

    async def _ensure_envelope(self, raw: str, llm) -> dict:
        """Return a parsed envelope dict, repairing prose output with one
        json_object-mode call if the sub-agent didn't emit valid JSON.

        Sub-agents commonly end their tool loop with prose; the RFC envelope
        must still be produced. A single no-tools `response_format=json_object`
        call converts the draft into the envelope (ADR-001: semantics are then
        Pydantic-validated downstream).
        """
        from ..contracts import parse_result

        try:
            data = parse_result(raw)
            parse_envelope(data)  # quick validity probe
            return data
        except Exception:
            pass
        # Patch runs are judged by the repository diff, not by an LLM-authored
        # envelope. Deterministic wrapping saves a redundant repair call and
        # leaves budget for the independent verifier.
        if self.parent_context.get("require_patch"):
            return {
                "summary": raw[:500] or "Subagent completed without prose output.",
                "output": raw,
            }
        # The repair asks for LOOSE structured findings, not the strict nested
        # envelope — the envelope is constructed deterministically from them.
        self._check_budget()
        repair = await asyncio.to_thread(
            llm.chat,
            [
                {
                    "role": "system",
                    "content": (
                        "Reply with a SINGLE JSON object: "
                        '{"summary": "<your findings in <=500 chars>", '
                        '"findings": <your structured findings>}. Example: '
                        '{"summary": "found runner.py; it orchestrates '
                        'subagents", "findings": '
                        '{"files": ["mycoder/agents/runner.py"], '
                        '"notes": "spawns and validates envelopes"}}'
                        "No prose, no markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": (f"Convert your answer into that JSON. Your previous answer:\n\n{raw[:4000]}"),
                },
            ],
            response_format={"type": "json_object"},
        )
        return parse_result(repair.content)

    def _build_system_prompt(self) -> str:
        if self.parent_context.get("require_patch"):
            # A patch run must enter the tool loop before it can satisfy the
            # envelope contract. Putting "output JSON" first makes thinking
            # models terminate with prose/JSON and emit zero tool calls, which
            # is indistinguishable from an analysis-only implementer. The
            # runtime validates/wraps the final envelope after the diff gate.
            contract_instruction = (
                "Use the available tools first: inspect the repository, make the "
                "required production-code change, and verify it when appropriate. "
                "Do not finish with a contract JSON object before the required "
                "tool actions have completed. After the tool loop, a concise "
                "prose summary is acceptable; the runtime will construct and "
                "validate the result contract."
            )
        else:
            contract_instruction = (
                "Output MUST be a single RFC v1.0.1 Subagent Result Contract JSON "
                "object. Include meta.task_id, meta.subagent_name, "
                "meta.subagent_instance_id (echo the provided values), "
                "meta.started_at/finished_at/duration_ms, status, summary (<=500 "
                "chars), confidence, and the appropriate result.<type> payload."
            )
        return (
            f"{self.definition.system_prompt}\n\n"
            f"{contract_instruction}"
        )

    def _build_success_envelope(self, result_data: dict) -> SubagentResultEnvelope:
        """Validate the subagent's emitted envelope.

        If the output looks like an envelope attempt (has status + meta) it
        must pass strict validation — a violation raises so `run()` converts it
        to an error envelope (no silent wrapping). If the output is inner
        result data (no status), it is wrapped into a success envelope.
        """
        if "status" in result_data and "meta" in result_data:
            # the orchestrator owns ids + timing — inject BEFORE validation so
            # a subagent that omitted them still produces a valid envelope
            meta = result_data.setdefault("meta", {})
            meta["task_id"] = self.parent_context.get("task_id", "unknown")
            meta["session_id"] = self.parent_context.get("session_id")
            meta["subagent_instance_id"] = self.instance_id
            if self._start_time is not None:
                now = datetime.datetime.now(datetime.timezone.utc).isoformat()
                meta.setdefault("started_at", now)
                meta.setdefault("finished_at", now)
                meta.setdefault("duration_ms", int((time.monotonic() - self._start_time) * 1000))
            return parse_envelope(result_data)
        # inner-data path: infer a payload and wrap
        return SubagentResultEnvelope(
            schema_version="1.0.1",
            status="success",
            summary=self._summarize(result_data),
            confidence="high",
            result=self._infer_result_payload(result_data),
            meta=self._build_meta(),
            artifacts=self._extract_artifacts(result_data),
        )

    def _build_error_envelope(
        self,
        status: Literal["failed", "cancelled"],
        error: ErrorObject,
    ) -> SubagentResultEnvelope:
        return SubagentResultEnvelope(
            schema_version="1.0.1",
            status=status,
            summary=error.message[:500],
            confidence="low",
            error=error,
            meta=self._build_meta(),
        )

    def _build_meta(self) -> Meta:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return Meta(
            task_id=self.parent_context.get("task_id", "unknown"),
            subagent_name=self.definition.name,
            subagent_instance_id=self.instance_id,
            session_id=self.parent_context.get("session_id"),
            started_at=now,
            finished_at=now,
            duration_ms=int((time.monotonic() - self._start_time) * 1000) if self._start_time else 0,
            parent_tool_use_id=self.parent_context.get("tool_use_id"),
        )

    # ------------------------------------------------------ result inference

    @staticmethod
    def _infer_result_payload(data: dict):
        """Wrap loose sub-agent findings into a GeneralResult (always validates).

        The strict per-type payloads (ExplorationResult etc.) are only produced
        when the sub-agent emits a correctly-typed `result`; otherwise the
        findings ride in structured_output, which pydantic accepts as-is.
        """
        from ..contracts.envelope import GeneralResult

        inner = data.get("result")
        if isinstance(inner, dict) and inner.get("type"):
            try:
                # pydantic Union picks the right model from the literal type
                return inner
            except Exception:  # noqa: BLE001 - fall back to GeneralResult
                pass
        structured = data.get("findings") if "findings" in data else (inner if isinstance(inner, dict) else None)
        return GeneralResult(
            type="general",
            output=SubagentRunner._summarize(data),
            structured_output=structured,
        )

    @staticmethod
    def _summarize(data: dict) -> str:
        summary = data.get("summary")
        if isinstance(summary, str) and summary:
            return summary[:500]
        return "Subagent completed."

    @staticmethod
    def _extract_artifacts(data: dict):
        from ..contracts.envelope import Artifact

        artifacts = data.get("artifacts")
        if not isinstance(artifacts, list):
            return None
        return [Artifact(**a) for a in artifacts[:100]]

    @staticmethod
    def _build_usage(data: dict) -> dict | None:
        usage = data.get("usage")
        return usage if isinstance(usage, dict) else None
