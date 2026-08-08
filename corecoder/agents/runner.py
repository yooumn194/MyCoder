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

from ..contracts.envelope import (
    ErrorObject,
    Meta,
    SubagentResultEnvelope,
    parse_envelope,
)
from .definition import SubagentDefinition
from .tool_validator import ToolOutputValidator


class SubagentRunner:
    def __init__(
        self,
        definition: SubagentDefinition,
        task: str,
        orchestrator: Any,
        parent_context: dict,
        instance_id: Optional[str] = None,
        executor: Optional[Callable[[str, str], Any]] = None,
        tool_validator: Optional[ToolOutputValidator] = None,
    ) -> None:
        self.definition = definition
        self.task = task
        self.orchestrator = orchestrator
        self.parent_context = parent_context
        self.instance_id = instance_id or str(uuid.uuid4())
        # executor(task, system_prompt) -> dict; injectable for tests so the
        # Runner needs no real LLM. Default spawns a sub-Agent.
        self._executor = executor
        # P0-1: tool-output contract validation (subagent's inner tool calls)
        self.tool_validator = tool_validator or ToolOutputValidator()
        self._start_time: Optional[float] = None

    async def run(self) -> SubagentResultEnvelope:
        self._start_time = time.monotonic()
        try:
            async with asyncio.timeout(self.definition.timeout_seconds):
                result_data = await self._run_loop()
            return self._build_success_envelope(result_data)
        except TimeoutError:
            return self._build_error_envelope(
                "failed",
                ErrorObject(
                    code="SUBAGENT_TIMEOUT",
                    category="system_constraint",
                    retryable=False,
                    message=f"Subagent 执行超时（{self.definition.timeout_seconds}s）",
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

    async def _run_loop(self) -> dict:
        system_prompt = self._build_system_prompt()
        if self._executor is not None:
            return await self._executor(self.task, system_prompt)
        return await self._run_sub_agent(system_prompt)

    async def _run_sub_agent(self, system_prompt: str) -> dict:
        """Default: spawn a real sub-Agent with the definition's tool whitelist."""
        from ..agent import Agent

        llm = getattr(self.orchestrator, "llm", None)
        if llm is None:
            raise RuntimeError("orchestrator has no llm; inject an executor instead")
        allowed = self.definition.allowed_tools
        tools = [t for t in getattr(self.orchestrator, "tools", []) if t.name in allowed]
        sub = Agent(llm=llm, tools=tools, max_rounds=self.definition.max_turns)
        raw = await asyncio.to_thread(
            sub.chat, f"{system_prompt}\n\nTask: {self.task}"
        )
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
        # The repair asks for LOOSE structured findings, not the strict nested
        # envelope — the envelope is constructed deterministically from them.
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
                        '{"files": ["corecoder/agents/runner.py"], '
                        '"notes": "spawns and validates envelopes"}}'
                        "No prose, no markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Convert your answer into that JSON. Your previous "
                        f"answer:\n\n{raw[:4000]}"
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
        return parse_result(repair.content)

    def _build_system_prompt(self) -> str:
        return (
            f"{self.definition.system_prompt}\n\n"
            "Output MUST be a single RFC v1.0.1 Subagent Result Contract JSON "
            "object. Include meta.task_id, meta.subagent_name, "
            "meta.subagent_instance_id (echo the provided values), "
            "meta.started_at/finished_at/duration_ms, status, summary (<=500 "
            "chars), confidence, and the appropriate result.<type> payload."
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
            meta["subagent_instance_id"] = self.instance_id
            if self._start_time is not None:
                now = datetime.datetime.now(datetime.timezone.utc).isoformat()
                meta.setdefault("started_at", now)
                meta.setdefault("finished_at", now)
                meta.setdefault(
                    "duration_ms", int((time.monotonic() - self._start_time) * 1000)
                )
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
            started_at=now,
            finished_at=now,
            duration_ms=int((time.monotonic() - self._start_time) * 1000)
            if self._start_time
            else 0,
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
