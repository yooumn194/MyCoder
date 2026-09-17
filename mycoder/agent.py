"""Core agent loop.

This is the heart of MyCoder.  The pattern is simple:

    user message -> LLM (with tools) -> tool calls? -> execute -> loop
                                      -> text reply? -> return to user

It normally stops when the LLM responds with plain text (no tool calls).
Runtime convergence limits also stop repeated, stagnant, overlong, or
budget-exhausting loops and reserve one tool-free final response when possible.
"""

from __future__ import annotations

import concurrent.futures
import inspect
import json
import os
import re
import time

from .convergence import (
    ConvergenceController,
    ConvergenceLimits,
    ConvergencePhase,
    ToolAdmission,
    ToolObservation,
)
from .context import ContextManager, estimate_tokens
from .llm import LLM
from .memory.integration import MemoryIntegration
from .planner import planning_guard
from .tools import ALL_TOOLS
from .tools.base import Tool, ToolResult
from .tools.agent import AgentTool
from .tools.correction import run_with_correction
from .tools.idempotency import IdempotencyStore
from .tools.subagent_tools import SpawnSubagentTool
from .prompt import system_prompt
from .prompts.reasoning import (
    REASONING_STRATEGIES,
    resolve_strategy,
    resolve_strategy_by_task,
)
from .sandbox.logger import get_logger
from .tools.security import redact_output

logger = get_logger("mycoder.agent")

_INSPECTION_TOOLS = frozenset({"read", "read_file", "grep_search", "list_files"})
_MUTATION_TOOLS = frozenset({"edit_file", "write_file"})


class _ToolExecutionResult(ToolResult):
    """String-compatible tool result carrying trace metadata internally."""

    def __new__(
        cls,
        value,
        *,
        status: str,
        exit_code: int | None = None,
        duration_ms: float = 0.0,
        retry_count: int = 0,
        cache_hit: bool = False,
    ):
        return super().__new__(
            cls,
            value,
            status=status,
            exit_code=exit_code,
            duration_ms=duration_ms,
            retry_count=retry_count,
            cache_hit=cache_hit,
        )


def _injection_guard_enabled() -> bool:
    """True unless the model-layer injection guard is explicitly disabled."""
    return os.getenv("MYCODER_INJECTION_GUARD", "on").strip().lower() != "off"


def _injection_classifier_enabled() -> bool:
    """Allow regex-only defense for reproducible trusted benchmark inputs."""
    return os.getenv("MYCODER_INJECTION_CLASSIFIER", "on").strip().lower() != "off"


def _nonnegative_env(name: str, default: int) -> int:
    """Read a bounded integer knob without letting bad env input break startup."""
    try:
        return max(0, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return max(0, int(default))


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool] | None = None,
        max_context_tokens: int = 128_000,
        max_rounds: int = 50,
        memory: MemoryIntegration | None = None,
        tool_selector=None,
        reasoning_strategy: str | None = None,
        injection_detector=None,
        memory_compressor=None,
        experience_store=None,
        budget_guard=None,
        convergence_limits: ConvergenceLimits | None = None,
        require_mutation: bool = False,
        require_verification: bool = False,
        strict_tool_choice: bool = False,
        max_turn_tokens: int | None = None,
        reserved_tokens: int = 0,
        max_mutation_extension_rounds: int | None = None,
        max_mutation_feedback_rounds: int | None = None,
        max_inspection_rounds_before_action: int | None = None,
        mutation_reserved_tokens: int | None = None,
        verification_reserved_tokens: int | None = None,
        tool_tracer=None,
        trace_context: dict | None = None,
    ):
        self.llm = llm
        self.tools = tools if tools is not None else ALL_TOOLS
        self._tool_by_name = {t.name: t for t in self.tools}
        self.messages: list[dict] = []
        self.context = ContextManager(max_tokens=max_context_tokens)
        # P1 memory closure: when a MemoryCompressor is wired, context
        # compression demotes the compressed turns into the long-term memory DB
        # (extract_facts) instead of dropping them.
        if memory_compressor is not None:
            self.context.on_compressed = memory_compressor.on_compressed
        self.max_rounds = max_rounds
        self.budget_guard = budget_guard
        self.require_mutation = bool(require_mutation)
        self.require_verification = bool(require_verification)
        # Recorded only, not enforced: the agent never sends a provider
        # tool_choice (see the stable-catalog comment in the loop), so a
        # benchmark run cannot be told apart from an interactive one by this
        # flag. It stays in the contract for tracing and for callers that log
        # which postconditions applied.
        self.strict_tool_choice = bool(strict_tool_choice)
        self.max_turn_tokens = max(1, int(max_turn_tokens)) if max_turn_tokens is not None else None
        self.reserved_tokens = max(0, int(reserved_tokens))
        if max_mutation_extension_rounds is None:
            try:
                max_mutation_extension_rounds = int(os.getenv("MYCODER_MUTATION_EXTENSION_ROUNDS", "2"))
            except ValueError:
                max_mutation_extension_rounds = 2
        self.max_mutation_extension_rounds = max(0, int(max_mutation_extension_rounds))
        if max_mutation_feedback_rounds is None:
            try:
                max_mutation_feedback_rounds = int(os.getenv("MYCODER_MUTATION_FEEDBACK_ROUNDS", "3"))
            except ValueError:
                max_mutation_feedback_rounds = 3
        self.max_mutation_feedback_rounds = max(1, int(max_mutation_feedback_rounds))
        if max_inspection_rounds_before_action is None:
            try:
                max_inspection_rounds_before_action = int(
                    os.getenv("MYCODER_INSPECTION_ROUNDS_BEFORE_ACTION", "2")
                )
            except ValueError:
                max_inspection_rounds_before_action = 2
        self.max_inspection_rounds_before_action = max(
            1,
            int(max_inspection_rounds_before_action),
        )
        if mutation_reserved_tokens is None:
            mutation_reserved_tokens = _nonnegative_env(
                "MYCODER_MUTATION_RESERVED_TOKENS", 2048
            )
        if verification_reserved_tokens is None:
            verification_reserved_tokens = _nonnegative_env(
                "MYCODER_VERIFICATION_RESERVED_TOKENS", 4096
            )
        self.mutation_reserved_tokens = max(0, int(mutation_reserved_tokens))
        self.verification_reserved_tokens = max(0, int(verification_reserved_tokens))
        self.tool_tracer = tool_tracer
        self.trace_context = dict(trace_context or {})
        self.convergence_limits = convergence_limits or ConvergenceLimits.from_env(max_rounds)
        # P1 (prompts/reasoning.py): reasoning strategy — ReAct / Plan-and-
        # Execute / Reflection. A manual override (explicit arg or the
        # PLANNING_STRATEGY env) fixes one strategy for the session; otherwise
        # AUTO mode routes each task to its fitting strategy.
        self._strategy_mode = "auto"
        if reasoning_strategy is not None or os.getenv("PLANNING_STRATEGY"):
            self._strategy_mode = "manual"
        self.reasoning_strategy = resolve_strategy(reasoning_strategy)
        self._system = system_prompt(self.tools, reasoning_strategy=self.reasoning_strategy)
        # Phase 5: optional memory integration (CLI wires one; idempotent).
        self.memory = memory
        if memory is not None:
            memory.install()
        # P1 model-layer injection defense: regex fast-scan on user input AND
        # tool results, plus an LLM classifier on user input (cheapest tier),
        # and <tool_output> role isolation declared in the system prompt.
        # Enabled by default; MYCODER_INJECTION_GUARD=off disables it.
        self._injection = injection_detector
        if self._injection is None and _injection_guard_enabled():
            from .tools.security import InjectionDetector, build_injection_classifier

            classifier = build_injection_classifier(llm) if _injection_classifier_enabled() else None
            self._injection = InjectionDetector(classifier=classifier)
        # P0 (tools/selector.py): when set, only the tools most relevant to the
        # current user message are injected into the LLM call, cutting token
        # cost and sharpening tool choice. None = inject all (backward compat).
        self.tool_selector = tool_selector
        # P0 (tools/idempotency.py): opt-in cache for stable tool results.
        # Mutable workspace observations and file mutations never use it.
        self._idem = IdempotencyStore()
        # Tool-call metrics: success / failure / retry counters — the material
        # for "成功率 / 工具失败率 / 重试率" answers (see _tool_metrics()).
        self._tool_calls = 0
        self._tool_success = 0
        self._tool_failure = 0
        self._tool_retries = 0
        self._tool_durations: list[float] = []  # ms per real execution
        self._last_convergence: dict = {}
        self._turn_start_used: int | None = None
        self._turn_tool_events: list[dict] = []
        self._active_mutation_done = False
        self._active_verification_done = False
        self._last_phase_trace: dict = {"phase": ConvergencePhase.EXPLORE.value, "phase_transitions": []}
        self._active_convergence: ConvergenceController | None = None

        # wire up sub-agent capability
        for t in self.tools:
            if isinstance(t, AgentTool):
                t._parent_agent = self
            elif isinstance(t, SpawnSubagentTool):
                t._parent_agent = self
                # P1 re-planning experience hook (persists deviation playbooks)
                if experience_store is not None:
                    t._experience_store = experience_store
                # P2 token budget: sub-agents get per-session budget protection
                if budget_guard is not None:
                    t._budget_guard = budget_guard

    def _full_messages(self) -> list[dict]:
        return [{"role": "system", "content": self._system}] + self.messages

    def _select_tools(self, query: str | None = None) -> list[Tool]:
        if self.tool_selector is not None and query and query.strip():
            return self.tool_selector.select(query, self.tools)
        return self.tools

    def _tool_schemas(self, query: str | None = None) -> list[dict]:
        return [t.schema() for t in self._select_tools(query)]


    def _selection_query(self) -> str:
        """Build a tool-selection relevance query from the RECENT conversation.

        Selection is evaluated once per TURN (see chat()), from the messages
        available at that moment, so the injected tool set GROWS as the session
        progresses instead of being frozen by the first message forever (#13).
        Within a turn the catalog does NOT move: recomputing it per round is the
        P0-2 regression, where the same instance saw a different capability set
        every round and produced different outcomes for identical input. Grow
        across turns, hold steady within one — that is the deliberate balance.
        """
        parts: list[str] = []
        for m in self.messages[-6:]:
            content = m.get("content")
            if isinstance(content, str) and content:
                parts.append(content[:200])
            for tc in m.get("tool_calls") or []:
                if isinstance(tc, dict) and tc.get("name"):
                    parts.append(str(tc["name"]))
        text = "\n".join(parts)[:1500]
        return text or "task"

    def _action_budget_prompt(self, phase: str, *, allow_targeted_read: bool = False) -> str:
        """Transient rollout guidance once inspection has consumed its budget."""
        if phase == "force_edit":
            if allow_targeted_read:
                next_action = (
                    "The previous edit failed because the target file was not read. "
                    "Call read_file once for the exact production file, then call "
                    "edit_file with the smallest safe change. Do not broaden the search."
                )
            else:
                next_action = (
                    "Call edit_file now using the repository evidence already collected. "
                    "The action deadline has been reached: do not call another inspection "
                    "tool even if the provider still exposes one. Infer the smallest safe "
                    "change from the code already in context. Use write_file only when the "
                    "task genuinely requires a new file."
                )
        else:
            next_action = (
                "Make the smallest safe production-code edit now. If one exact context "
                "detail is still missing, perform at most one targeted inspection first."
            )
        return (
            "<action_budget>\n"
            "The bounded exploration phase is complete. Do not broaden the search or "
            "repeat prior reads. "
            f"{next_action}\n"
            "</action_budget>"
        )

    @staticmethod
    def _mutation_needs_read_recovery(failure: str | None) -> bool:
        """Whether an edit failure proves the model must read before retrying."""
        text = str(failure or "").upper()
        return any(
            marker in text
            for marker in (
                "FILE_NOT_READ",
                "MUST BE READ",
                "READ BEFORE IT CAN BE EDITED",
            )
        )

    def _mutation_feedback(
        self,
        *,
        attempt: int,
        phase: str,
        reason: str,
        last_failure: str | None = None,
    ) -> str:
        payload = {
            "code": "MUTATION_REQUIRED",
            "attempt": attempt,
            "max_attempts": self.max_mutation_feedback_rounds,
            "phase": phase,
            "observed": reason[:500],
            "next_action": (
                "Call read_file/grep_search/list_files now and inspect the exact production file."
                if phase == "inspect"
                else "Call edit_file for an existing production file or write_file only for a genuinely new file now."
            ),
        }
        if last_failure:
            payload["last_failure"] = last_failure[:500]
        return (
            "<required_action_feedback>\n"
            f"{json.dumps(payload, ensure_ascii=False)}\n"
            "Do not answer with analysis or a proposed diff. Execute the required tool call.\n"
            "</required_action_feedback>"
        )

    def _record_requirement_feedback(
        self,
        feedback: str,
        *,
        attempt: int,
        phase: str,
    ) -> None:
        if self.tool_tracer is None:
            return
        try:
            self.tool_tracer.record_requirement_feedback(
                self._session_id() or "unknown",
                feedback,
                attempt=attempt,
                phase=phase,
                subagent_name=self.trace_context.get("subagent_name", "main"),
                subagent_instance_id=self.trace_context.get("subagent_instance_id"),
            )
        except Exception as exc:  # noqa: BLE001 - tracing cannot break execution
            logger.warning("tool_requirement_trace_failed", error_msg=str(exc))

    def _record_tool_call(
        self,
        tc,
        result,
        *,
        mutation: bool,
        verification: bool,
        admission: ToolAdmission,
        phase: str | None = None,
    ) -> None:
        if self.tool_tracer is None:
            return
        status = getattr(result, "status", None) or getattr(result, "trace_status", None)
        if not admission.allowed:
            status = "blocked"
        if status is None:
            status = "success" if self._tool_result_succeeded(result) else "error"
        try:
            self.tool_tracer.record(
                self._session_id() or "unknown",
                tc.id,
                tc.name,
                dict(tc.arguments),
                result,
                status=status,
                duration_ms=getattr(result, "duration_ms", 0.0),
                retry_count=getattr(result, "retry_count", 0),
                cache_hit=getattr(result, "cache_hit", False),
                mutation=mutation,
                verification=verification,
                phase=phase,
                subagent_name=self.trace_context.get("subagent_name", "main"),
                subagent_instance_id=self.trace_context.get("subagent_instance_id"),
            )
        except Exception as exc:  # noqa: BLE001 - tracing cannot break execution
            logger.warning("tool_call_trace_failed", error_msg=str(exc))

    def chat(self, user_input: str, on_token=None, on_tool=None) -> str:
        """Process one user message. May involve multiple LLM/tool rounds."""
        self._turn_start_used = self._current_budget_used()
        self._turn_tool_events = []
        # Pre-LLM injection defense: regex always runs; the semantic classifier
        # runs only when the request contains a suspicious cue. Calling a
        # second LLM for every ordinary coding prompt doubles latency/cost and
        # introduces an unnecessary false-positive failure point.
        if self._injection is not None:
            blocked, reason = self._injection.defend(
                user_input,
                classifier_on_suspicious_only=True,
            )
            if blocked:
                return f"⚠ 检测到可能的指令注入，已阻止处理：{reason}"
        # P1 auto mode: route THIS task to its fitting strategy (plan for
        # architecture, reflection for bug-fix, react otherwise) and rebuild the
        # system prompt, so a long session uses the best reasoning mode per task.
        if self._strategy_mode == "auto":
            strategy = resolve_strategy_by_task(user_input)
            if strategy != self.reasoning_strategy:
                self._apply_strategy(strategy)
        self.messages.append({"role": "user", "content": user_input})
        self.context.maybe_compress(self.messages, self.llm)

        # P0-2: the tool catalog for this turn is decided ONCE, here, and never
        # reshaped mid-loop (see the assignment inside the round loop). A
        # requirement-bearing run always advertises the full role-scoped set so
        # no required capability can be selected away; other runs may use the
        # relevance selector, but it is also evaluated only once.
        if self.tool_selector is not None and not (
            self.require_mutation or self.require_verification
        ):
            turn_tool_schemas = self._tool_schemas(self._selection_query())
        else:
            turn_tool_schemas = [tool.schema() for tool in self.tools]

        convergence = ConvergenceController(self.convergence_limits)
        self._active_convergence = convergence
        required_mutation_done = False
        required_verification_done = False
        self._active_mutation_done = False
        self._active_verification_done = False
        mutation_extension_rounds = 0
        mutation_feedback_rounds = 0
        inspection_rounds_without_mutation = 0
        inspection_done = False
        action_context_compacted = False
        forced_requirement_phase: str | None = None
        last_mutation_failure: str | None = None
        # A failed edit may be recovered by one targeted read, but that read
        # must be consumed.  Otherwise providers that ignore the mutation
        # contract can keep replaying the same inspection forever.
        recovery_read_used = False
        for _ in range(self.max_rounds + 1):
            budget_requirement_prompt: str | None = None
            self._active_mutation_done = required_mutation_done
            self._active_verification_done = required_verification_done
            budget_ratio = self._budget_ratio()
            if (
                self._missing_requirement(required_mutation_done, required_verification_done)
                and not self._can_afford_requirement_round()
            ):
                return self._requirement_failure(
                    required_mutation_done,
                    required_verification_done,
                    "reserved budget reached",
                )
            if (
                self._missing_requirement(required_mutation_done, required_verification_done)
                and budget_ratio is not None
                and budget_ratio >= self.convergence_limits.soft_budget_ratio
            ):
                if mutation_extension_rounds >= self.max_mutation_extension_rounds:
                    return self._requirement_failure(
                        required_mutation_done,
                        required_verification_done,
                        "evidence-preserving extension exhausted",
                    )
                if not self._can_afford_requirement_round():
                    return self._requirement_failure(
                        required_mutation_done,
                        required_verification_done,
                        "reserved budget reached",
                    )
                mutation_extension_rounds += 1
                budget_ratio = None
                budget_requirement_prompt = self._requirement_prompt(
                    required_mutation_done,
                    required_verification_done,
                )
            action_budget_phase: str | None = None
            if self.require_mutation and not required_mutation_done:
                if inspection_rounds_without_mutation > self.max_inspection_rounds_before_action:
                    action_budget_phase = "force_edit"
                elif inspection_rounds_without_mutation >= self.max_inspection_rounds_before_action:
                    action_budget_phase = "mutate"
                phase_hint = action_budget_phase or forced_requirement_phase
                convergence.transition(
                    ConvergencePhase.MUTATE
                    if phase_hint in {"mutate", "force_edit"}
                    else ConvergencePhase.EXPLORE,
                    reason=phase_hint or "mutation pending",
                )
            elif self.require_verification and not required_verification_done:
                convergence.transition(ConvergencePhase.VERIFY, reason="verification pending")
            else:
                convergence.transition(ConvergencePhase.FINALIZE, reason="requirements complete")
            stop_reason = convergence.before_round(budget_ratio)
            if stop_reason:
                if self._missing_requirement(required_mutation_done, required_verification_done):
                    return self._requirement_failure(
                        required_mutation_done,
                        required_verification_done,
                        stop_reason,
                    )
                return self._finalize_after_convergence(stop_reason, on_token=on_token, controller=convergence)
            # Predictive execution: the LLM streams tool calls; a call whose
            # arguments complete early is handed to a pool and runs WHILE the
            # stream keeps generating, so its result is ready when generation
            # finishes — one serial RTT saved per round (StreamingToolExecutor).
            predicted: dict[str, concurrent.futures.Future] = {}
            admissions: dict[str, ToolAdmission] = {}
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
            try:
                # A forced requirement must not be defeated by the relevance
                # selector omitting edit_file after an analysis-only response.
                # Use the Agent's already role-scoped tools, then narrow to the
                # action family for this phase.
                # The rollout budget is monotonic: a terminal-stop retry must
                # never weaken a stricter force-edit phase after a stop retry.
                effective_requirement_phase = action_budget_phase or forced_requirement_phase
                allow_recovery_read = (
                    effective_requirement_phase == "force_edit"
                    and self._mutation_needs_read_recovery(last_mutation_failure)
                    and not recovery_read_used
                )
                # P0-2: ONE stable catalog for the whole turn.
                #
                # The catalog used to be reshaped every round — mutation-only
                # here, verification-only there, plus a per-round relevance
                # selection and a forced ``tool_choice``. That is a control
                # loop built by mutating the model's world: the same instance
                # could see a different capability set on every round, so its
                # output distribution drifted and the run became
                # irreproducible (the mechanism behind "the same SWE-bench
                # instance produces six different outcomes").
                #
                # Instead the catalog is computed ONCE and then held constant,
                # exactly like Codex / Claude Code / OpenCode. Requirements are
                # enforced where enforcement belongs: at the pre-tool gate
                # below (reject the disallowed call) plus structured
                # ``_mutation_feedback`` / ``_requirement_prompt`` corrections
                # in the transcript. No provider-dependent tool_choice is used.
                tool_schemas = turn_tool_schemas
                allowed_tool_names = {
                    str(schema.get("function", {}).get("name")) for schema in tool_schemas if schema.get("type") == "function"
                }

                def _admit(tc):
                    # Never execute a role-valid function that was omitted from
                    # this round by the selector or forced-action policy.
                    if tc.name not in allowed_tool_names:
                        return ToolAdmission(
                            signature=f"schema:{tc.name}",
                            novel=False,
                            blocked_reason=(
                                f"CONVERGENCE_BLOCKED: tool '{tc.name}' was not "
                                "offered in this round; call one of the declared tools."
                            ),
                        )
                    if (
                        effective_requirement_phase == "force_edit"
                        and tc.name in _INSPECTION_TOOLS
                        and not allow_recovery_read
                    ):
                        return ToolAdmission(
                            signature=f"action-budget:{tc.name}",
                            novel=False,
                            blocked_reason=(
                                "ACTION_REQUIRED: bounded inspection is complete. "
                                "Use Edit/edit_file with the file content already read; "
                                "do not call another read or search tool."
                            ),
                        )
                    return convergence.admit_tool(tc.name, tc.arguments)

                def _predict(tc):
                    admission = _admit(tc)
                    admissions[tc.id] = admission
                    tool = self._tool_by_name.get(tc.name)
                    # Speculation is deny-by-default. Idempotent writes are not
                    # enough: they may still expose partial model arguments.
                    if admission.allowed and tool is not None and getattr(tool, "predictive_safe", False):
                        predicted[tc.id] = pool.submit(self._exec_tool, tc)

                if (
                    action_budget_phase in {"mutate", "force_edit"}
                    and not action_context_compacted
                ):
                    self.context.compact_for_action(self.messages, keep_recent=4)
                    action_context_compacted = True
                request_messages = self._full_messages()
                request_controls = [
                    control
                    for control in (
                        self._action_budget_prompt(
                            action_budget_phase,
                            allow_targeted_read=allow_recovery_read,
                        )
                        if action_budget_phase is not None
                        else None,
                        budget_requirement_prompt,
                    )
                    if control
                ]
                if request_controls:
                    control_text = "\n\n".join(request_controls)
                    # Keep rollout control provider-compatible and transient:
                    # augment the leading system message for this request only,
                    # instead of polluting the durable conversation with a
                    # synthetic user turn after every inspection/budget gate.
                    request_messages[0] = {
                        "role": "system",
                        "content": (
                            f"{self._system}\n\n"
                            f"{control_text}"
                        ),
                    }
                chat_kwargs = {
                    "messages": request_messages,
                    "tools": tool_schemas,
                    "on_token": on_token,
                    "predictive_executor": _predict,
                }
                # No tool_choice is sent, ever: with a catalog that is constant
                # for the whole turn there is nothing left to force, and forcing
                # it was itself provider-dependent (DeepSeek's thinking endpoint
                # rejects named choices). `self.strict_tool_choice` now only
                # records the contract for tracing.
                # Propagate the runtime phase to the single LLM trace span.
                # This is a contextvar rather than a prompt field, so provider
                # requests stay byte-for-byte compatible and the value also
                # survives asyncio.to_thread in subagent runners.
                from structlog.contextvars import bound_contextvars

                with bound_contextvars(agent_phase=convergence.phase.value):
                    resp = self.llm.chat(**chat_kwargs)
            finally:
                pool.shutdown(wait=True)

            # No tool calls means the model is asking to end the loop. Treat
            # mutation/verification requirements as a terminal stop gate (as
            # Codex stop hooks do), rather than interrupting every successful
            # inspection round with a synthetic user message.
            if not resp.tool_calls:
                self.messages.append(resp.message)
                if self._missing_requirement(required_mutation_done, required_verification_done):
                    if self.require_mutation and not required_mutation_done:
                        mutation_feedback_rounds += 1
                        phase = "mutate" if inspection_done else "inspect"
                        feedback = self._mutation_feedback(
                            attempt=mutation_feedback_rounds,
                            phase=phase,
                            reason="model attempted to finish without a successful repository mutation",
                            last_failure=last_mutation_failure,
                        )
                        self._record_requirement_feedback(
                            feedback,
                            attempt=mutation_feedback_rounds,
                            phase=phase,
                        )
                        if mutation_feedback_rounds >= self.max_mutation_feedback_rounds:
                            return self._requirement_failure(
                                required_mutation_done,
                                required_verification_done,
                                (
                                    "mutation feedback limit reached "
                                    f"({mutation_feedback_rounds}); model repeatedly attempted to finish without mutation"
                                ),
                            )
                        forced_requirement_phase = phase
                    else:
                        feedback = self._requirement_prompt(required_mutation_done, required_verification_done)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": feedback,
                        }
                    )
                    stop_reason = convergence.finish_round([])
                    if stop_reason:
                        return self._requirement_failure(
                            required_mutation_done,
                            required_verification_done,
                            stop_reason,
                        )
                    continue
                self._last_convergence = convergence.snapshot("model_completed")
                self._last_phase_trace = convergence.phase_snapshot()
                # Post-LLM: redact secret shapes before the reply reaches the user.
                return redact_output(resp.content)

            # tool calls -> execute. A call already predicted ran concurrently
            # during generation; the rest run now. Results are guarded (injection
            # scan) and wrapped in <tool_output> role tags before回灌.
            self.messages.append(resp.message)

            try:
                results: list[str] = []
                for tc in resp.tool_calls:
                    if on_tool:
                        on_tool(tc.name, tc.arguments)
                    admission = admissions.get(tc.id)
                    if admission is None:
                        admission = _admit(tc)
                        admissions[tc.id] = admission
                    if not admission.allowed:
                        results.append(admission.blocked_reason or "CONVERGENCE_BLOCKED")
                    else:
                        fut = predicted.get(tc.id)
                        results.append(fut.result() if fut is not None else self._exec_tool(tc))
                observations: list[ToolObservation] = []
                round_inspected = False
                round_mutated = False
                for tc, result in zip(resp.tool_calls, results):
                    result = self._guard_tool_result(tc.name, result)
                    mutation = self._is_required_mutation(tc.name, result)
                    succeeded = self._tool_result_succeeded(result)
                    if mutation:
                        required_mutation_done = True
                        # Keep the convergence-visible copy in step with the
                        # local one: finalization may be reached later in this
                        # same round and reads `self._active_*`.
                        self._active_mutation_done = True
                        forced_requirement_phase = None
                        round_mutated = True
                    if tc.name in _INSPECTION_TOOLS and succeeded:
                        inspection_done = True
                        round_inspected = True
                    if tc.name in _MUTATION_TOOLS and not mutation:
                        last_mutation_failure = str(result)[:500]
                        # A new FILE_NOT_READ failure starts a fresh, bounded
                        # recovery window.  Other mutation failures should not
                        # reopen inspection because the target is already
                        # known.
                        recovery_read_used = False if self._mutation_needs_read_recovery(last_mutation_failure) else True
                    elif allow_recovery_read and tc.name in _INSPECTION_TOOLS:
                        recovery_read_used = True
                    verified = self._is_verification_evidence(tc.name, tc.arguments, result)
                    # A failed check is useful diagnostic evidence for a
        # normal verification-only request, but benchmark patch
                    # runs must keep the postcondition open until a check
                    # succeeds.  This gives the model a chance to switch from
                    # an unavailable behavioral test to a focused/static check
                    # instead of silently finalizing a failing patch.
                    # P0-4: a patch run keeps the postcondition open until a
                    # check EXITS 0 (`_check_passed`). A verification-only run
                    # may legitimately report a failing check — that is the
                    # finding, not a reason to loop forever. In both cases the
                    # verdict comes from the process exit code, never from a
                    # regex over the command line.
                    if verified and (self._check_passed(result) or not self.require_mutation):
                        required_verification_done = True
                        self._active_verification_done = True
                    self._record_tool_call(
                        tc,
                        result,
                        mutation=mutation,
                        verification=verified,
                        admission=admissions[tc.id],
                        phase=convergence.phase.value,
                    )
                    self._turn_tool_events.append(
                        {
                            "tool_call_id": tc.id,
                            "name": tc.name,
                            "arguments": dict(tc.arguments),
                            "succeeded": succeeded,
                            "verification": verified,
                            "mutation": mutation,
                            "status": (
                                "blocked"
                                if not admissions[tc.id].allowed
                                else getattr(result, "status", None)
                                or getattr(result, "trace_status", None)
                                or ("success" if succeeded else "error")
                            ),
                            "duration_ms": getattr(result, "duration_ms", 0.0),
                            "retry_count": getattr(result, "retry_count", 0),
                            "phase": convergence.phase.value,
                        }
                    )
                    observations.append(ToolObservation(admissions[tc.id], tc.name, result))
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": self._wrap_tool_output(tc.name, result),
                        }
                    )
                if round_mutated:
                    inspection_rounds_without_mutation = 0
                elif round_inspected:
                    inspection_rounds_without_mutation += 1

                # Providers that do not implement named/required tool choice
                # can still emit an inspection call at the terminal mutation
                # boundary, even though only edit/write schemas were sent.
                # Treat that as an actionable protocol miss: give the model a
                # user-role correction and keep the same mutation phase alive
                # for a bounded retry.  Calling ``finish_round`` here would
                # mark the controller STOP on the blocked observation, making
                # the next retry look like a phase regression and causing the
                # whole assignment to be retried from scratch.
                blocked_required_action = (
                    self.require_mutation
                    and not required_mutation_done
                    and effective_requirement_phase == "force_edit"
                    and any(not observation.admission.allowed for observation in observations)
                )
            except KeyboardInterrupt:
                # Ctrl+C mid-execution would leave the assistant tool_calls
                # message without replies, poisoning the next request; backfill
                self._answer_pending_tool_calls(resp.tool_calls)
                raise

            # compress if tool outputs are big
            self.context.maybe_compress(self.messages, self.llm)

            # Required postconditions are the action loop's completion
            # contract. Once all of them hold, do not offer another tool round:
            # switch to one tool-free summary call. This prevents a successful
            # implementer from drifting back into repeated inspection until the
            # generic round limit is reached.
            if not self._missing_requirement(
                required_mutation_done,
                required_verification_done,
            ) and (self.require_mutation or self.require_verification):
                return self._finalize_after_convergence(
                    "required postconditions completed",
                    on_token=on_token,
                    controller=convergence,
                )

            # A stop-gate retry that first requested inspection advances to the
            # mutation phase without adding another user message. Tool outputs
            # (including edit failures) are already the canonical feedback for
            # the next sample, matching Codex/OpenCode's loop contract.
            if forced_requirement_phase == "inspect" and inspection_done:
                forced_requirement_phase = "mutate"

            if blocked_required_action:
                mutation_feedback_rounds += 1
                feedback = self._mutation_feedback(
                    attempt=mutation_feedback_rounds,
                    phase="mutate",
                    reason=(
                        "provider emitted an inspection tool at the terminal edit boundary; "
                        "the call was rejected because only mutation tools were offered"
                    ),
                    last_failure=last_mutation_failure,
                )
                self._record_requirement_feedback(
                    feedback,
                    attempt=mutation_feedback_rounds,
                    phase="mutate",
                )
                if mutation_feedback_rounds >= self.max_mutation_feedback_rounds:
                    return self._requirement_failure(
                        required_mutation_done,
                        required_verification_done,
                        (
                            "mutation feedback limit reached "
                            f"({mutation_feedback_rounds}); provider repeatedly ignored the mutation-only tool catalog"
                        ),
                    )
                self.messages.append({"role": "user", "content": feedback})
                # Skip the stagnation stop gate for this one bounded recovery
                # round.  No state change occurred, but the correction itself
                # is new actionable context for the provider.
                continue

            stop_reason = convergence.finish_round(observations)
            if stop_reason:
                if self._missing_requirement(required_mutation_done, required_verification_done):
                    return self._requirement_failure(
                        required_mutation_done,
                        required_verification_done,
                        stop_reason,
                    )
                return self._finalize_after_convergence(stop_reason, on_token=on_token, controller=convergence)

        return self._finalize_after_convergence(
            f"round limit reached ({self.max_rounds})",
            on_token=on_token,
            controller=convergence,
        )

    @staticmethod
    def _session_id() -> str | None:
        try:
            from structlog.contextvars import get_contextvars

            value = get_contextvars().get("session_id")
            return str(value) if value else None
        except Exception:  # noqa: BLE001 - context is optional for local callers
            return None

    def _current_budget_used(self) -> int | None:
        if self.budget_guard is None:
            return None
        session_id = self._session_id()
        if not session_id:
            return None
        getter = getattr(self.budget_guard, "get_used", None)
        if getter is not None:
            return max(0, int(getter(session_id)))
        maximum = max(1, int(self.budget_guard.max_tokens_per_session))
        return max(0, maximum - int(self.budget_guard.get_remaining(session_id)))

    def _budget_ratio(self) -> float | None:
        if self.budget_guard is None:
            return None
        session_id = self._session_id()
        if not session_id:
            return None
        maximum = max(1, int(self.budget_guard.max_tokens_per_session))
        used = self._current_budget_used() or 0
        start = self._turn_start_used if self._turn_start_used is not None else used
        local_used = max(0, used - start)
        global_turn_capacity = max(1, maximum - start - self.reserved_tokens)
        turn_capacity = min(
            global_turn_capacity,
            self.max_turn_tokens or global_turn_capacity,
        )
        ratio = local_used / max(1, turn_capacity)
        local_remaining = max(0, turn_capacity - local_used)
        # Each sub-agent is measured from its own starting point. This prevents
        # a verifier from inheriting the implementer's 70% usage while still
        # respecting the shared hard cap and any downstream reservation.
        if local_remaining <= self._finalization_reserve():
            return max(ratio, self.convergence_limits.soft_budget_ratio)
        return ratio

    @staticmethod
    def _tool_result_succeeded(result: str) -> bool:
        if getattr(result, "status", None) == "error":
            return False
        text = str(result).lstrip()
        return (
            not text.startswith(
                (
                    "Error",
                    "⚠ Blocked",
                    "⚠ Cancelled",
                    "CONVERGENCE_BLOCKED",
                    "ACTION_REQUIRED",
                )
            )
            and "[timed out]" not in text
            and "[exit code:" not in text
        )

    # A command that can exit 0 while saying nothing about correctness. This is
    # a short DENY list, not the old allow list of known test runners: the old
    # pattern list made `./run_tests.sh`, `make check` and every project-local
    # script invisible, so a verifier that had run a real check was told it
    # hadn't. Anything not listed here that exits 0 is accepted as a check.
    _NON_VERIFICATION_COMMAND = re.compile(
        r"^\s*(?:sudo\s+)?"
        # `cd somewhere && …` is a prefix, not the command under test
        r"(?:cd\s+[^\s;&|]+\s*(?:&&|;)\s*)*"
        r"(?:env(?:\s+\w+=\S+)*\s+)?"
        r"(?:"
        r"ls|cat|head|tail|sed|awk|echo|printf|pwd|which|whereis|find|wc|tree|"
        r"rg|grep|egrep|fgrep|less|more|file|stat|du|df|env|printenv|touch|mkdir|"
        r"cp|mv|tar|unzip|curl|wget|"
        r"git\s+(?:status|diff|log|show|branch|ls-files|rev-parse|config|stash|add|commit|checkout|apply|clone)|"
        r"(?:pip|pip3|python3?\s+-m\s+pip)\s+install|npm\s+(?:install|ci)|"
        r"yarn\s+(?:install|add)|pnpm\s+install|apt-get|apt|apk|brew"
        r")\b"
    )

    # A second, narrower deny list: commands that exit 0 while asserting
    # nothing. They satisfy "a process ran and reported an exit code", so the
    # list above cannot catch them — `true`, `:`, `sleep 0` and `bash -c true`
    # would otherwise let a run satisfy require_verification without running
    # any check, which inflates the success rate MyCoder reports for itself.
    _COMMAND_PREFIX = re.compile(
        r"""^\s*
            (?:sudo\s+)?
            (?:env\s+)?
            (?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*
            (?:cd\s+[^\s;&|]+\s*(?:&&|;)\s*)*
        """,
        re.X,
    )
    _SHELL_WRAPPER = re.compile(
        r"^(?:ba|z|d|k)?sh\s+-c\s+(?P<quote>['\"])(?P<body>.*)(?P=quote)\s*$",
        re.X | re.S,
    )
    # Same wrapper without quotes: `bash -c true`. Tried after the quoted form
    # so `bash -c "pytest -q"` keeps its body intact.
    _SHELL_WRAPPER_BARE = re.compile(
        r"^(?:ba|z|d|k)?sh\s+-c\s+(?P<body>\S.*)$",
        re.X | re.S,
    )
    _NO_OP_BODY = re.compile(
        r"""^(?::|true|false|sleep\s+[\d.]+|wait|exit(?:\s+\d+)?)\s*$""",
        re.X,
    )
    # `python -c '<body>'` is a real check as soon as the body does anything, so
    # only bodies that cannot fail are treated as no-ops.
    _NO_OP_PYTHON_BODY = frozenset(
        {"", "pass", "...", "0", "1", "None", "True", "False", "exit()", "quit()"}
    )
    _PYTHON_INLINE = re.compile(r"^python[23]?\s+-c\s+(?P<body>.*)$", re.X | re.S)

    @classmethod
    def _is_no_op_command(cls, command: str, _depth: int = 0) -> bool:
        """Whether the command provably asserts nothing about the code.

        Shell wrappers are unwrapped first (`bash -c "true"` is a no-op,
        `bash -c "pytest -q"` is not), so the check stays conservative: anything
        that could conceivably run a test is still accepted as evidence.
        """
        if _depth > 4:
            return False
        stripped = cls._COMMAND_PREFIX.sub("", command.strip()).strip()
        if not stripped.strip(";&| \t\r\n"):
            return True
        wrapper = cls._SHELL_WRAPPER.match(stripped) or cls._SHELL_WRAPPER_BARE.match(stripped)
        if wrapper:
            return cls._is_no_op_command(wrapper.group("body"), _depth + 1)
        inline = cls._PYTHON_INLINE.match(stripped)
        if inline:
            body = inline.group("body").strip().strip("'\"").strip()
            return body in cls._NO_OP_PYTHON_BODY
        return bool(cls._NO_OP_BODY.match(stripped))

    @classmethod
    def _is_verification_evidence(cls, tool_name: str, arguments: dict, result: str) -> bool:
        """Whether a check command actually RAN — whatever its verdict.

        The previous implementation regex-matched the command string against a
        list of known runners. That inverted the burden of proof: a project's
        own ``./run_tests.sh`` or a bare ``make check`` counted as "not a
        check", while a ``grep`` whose path merely contained ``/testing/`` did.
        The test now is "did a process run and report an exit code", which is a
        fact the sandbox owns. Whether the check PASSED is a separate question
        (see the caller: a patch run needs a green check, a verification-only
        run may legitimately report a red one).
        """
        if tool_name != "execute_in_sandbox":
            return False
        text = str(result).lstrip()
        if text.startswith(("Error", "⚠ Blocked", "⚠ Cancelled")):
            return False
        if "[timed out]" in text:
            return False
        exit_code = getattr(result, "exit_code", None)
        if exit_code is None:
            # Back-compat for results produced outside ExecuteInSandboxTool:
            # a non-zero exit is still rendered as "[exit code: N]".
            match = re.search(r"\[exit code:\s*(-?\d+)\]", text)
            if match is None:
                return False  # cannot prove a process ran and reported
            exit_code = int(match.group(1))
        command = str(arguments.get("command", "")).strip()
        if not command:
            return False
        if cls._is_no_op_command(command):
            return False
        return not cls._NON_VERIFICATION_COMMAND.match(command)

    @staticmethod
    def _check_passed(result: str) -> bool:
        """Whether a sandbox check exited 0 (as opposed to merely running)."""
        exit_code = getattr(result, "exit_code", None)
        if exit_code is None:
            match = re.search(r"\[exit code:\s*(-?\d+)\]", str(result))
            if match is None:
                return False
            exit_code = int(match.group(1))
        return exit_code == 0

    def _missing_requirement(self, mutation_done: bool, verification_done: bool) -> bool:
        return (self.require_mutation and not mutation_done) or (self.require_verification and not verification_done)

    def _requirement_prompt(self, mutation_done: bool, verification_done: bool) -> str:
        if self.require_mutation and not mutation_done:
            return (
                "A repository patch is required, but no successful mutation has occurred. "
                "Keep read/search tools available: locate and read the exact target first, "
                "then use edit_file for an existing file or write_file only for a new file. "
                "Do not overwrite an existing file or return analysis only."
            )
        if self.require_verification and not verification_done:
            return (
                "Verification evidence is required before completion. Run a focused test, "
                "compiler, or static checker with execute_in_sandbox. Inspecting git diff or "
                "stating that the change looks correct is not verification."
            )
        return "Complete the task using the available evidence."

    def _requirement_failure(self, mutation_done: bool, verification_done: bool, reason: str) -> str:
        if self._active_convergence is not None:
            self._active_convergence.transition(ConvergencePhase.STOP, reason=reason)
            self._last_convergence = self._active_convergence.snapshot(reason)
            self._last_phase_trace = self._active_convergence.phase_snapshot()
        requirement = "repository mutation" if self.require_mutation and not mutation_done else "verification evidence"
        return f"(required {requirement} not completed: {reason})"

    def _spendable_remaining(self) -> int | None:
        if self.budget_guard is None:
            return None
        session_id = self._session_id()
        if not session_id:
            return None
        return max(
            0,
            int(self.budget_guard.get_remaining(session_id)) - self.reserved_tokens,
        )

    def _can_afford_requirement_round(self) -> bool:
        remaining = self._spendable_remaining()
        return remaining is None or remaining > self._requirement_round_reserve()

    def _requirement_round_reserve(self) -> int:
        """Minimum cost of one action call, without a prose finalization.

        Patch-mode runners validate the real diff and wrap the result
        deterministically. ``reserved_tokens`` already protects downstream
        subagents (for example, the verifier share); adding the current
        requirement reserves again here double-counts that capacity and can
        reject the verifier before it executes its first check. The LLM tracer
        remains the authoritative prompt+output admission guard.
        """
        prompt_tokens = estimate_tokens(self._full_messages())
        try:
            configured_output = int(getattr(self.llm, "max_tokens", 1024) or 1024)
        except (TypeError, ValueError):
            configured_output = 1024
        return prompt_tokens + min(1024, max(256, configured_output)) + 256

    def _active_requirement_reserve(self) -> int:
        """Reserve independent capacity for the pending action phase."""
        reserve = 0
        if self.require_mutation and not self._active_mutation_done:
            reserve += self.mutation_reserved_tokens
        if self.require_verification and not self._active_verification_done:
            reserve += self.verification_reserved_tokens
        return reserve

    @staticmethod
    def _is_required_mutation(tool_name: str, result: str) -> bool:
        """Whether a tool result is evidence of a repository mutation."""
        if getattr(result, "status", None) == "error":
            return False
        text = str(result).lstrip()
        if text.startswith(("Error", "⚠ Blocked", "CONVERGENCE_BLOCKED")):
            return False
        if tool_name in {"write_file", "edit_file"}:
            return True
        return False

    def _finalization_reserve(self) -> int:
        prompt_tokens = estimate_tokens(self._full_messages())
        try:
            configured_output = int(getattr(self.llm, "max_tokens", 4096) or 4096)
        except (TypeError, ValueError):
            configured_output = 4096
        output_cap = min(
            4096,
            max(256, configured_output),
        )
        return prompt_tokens + output_cap + 512

    def _can_afford_finalization(self) -> bool:
        """Reserve enough budget for one tool-free final response."""
        if self.budget_guard is None:
            return True
        session_id = self._session_id()
        if not session_id:
            return True
        remaining = self._spendable_remaining()
        if remaining is None:
            return True
        return remaining > self._finalization_reserve()

    @property
    def verification_evidence(self) -> list[dict]:
        """Postcondition checks performed during the current turn, pass or fail."""
        return [event for event in self._turn_tool_events if event["verification"]]

    @property
    def tool_events(self) -> list[dict]:
        """Bounded action evidence for the runner's deterministic postconditions."""
        return [dict(event) for event in self._turn_tool_events]

    def _finalize_after_convergence(
        self,
        reason: str,
        on_token=None,
        controller: ConvergenceController | None = None,
    ) -> str:
        """Ask once for a tool-free summary if the hard budget can afford it."""
        # P0-3b: never spend the tool-free summary turn while a required
        # postcondition is still open.  The old code did, and the summary turn
        # is exactly where a model that "says" its verification command loses
        # it: with tools=[] the DSML fallback is skipped, so the emitted
        # execute_in_sandbox call is parsed as nothing, executed as nothing,
        # and the run is reported as "completed without a successful
        # verification command" even though the command was requested.
        if self._missing_requirement(
            bool(self._active_mutation_done),
            bool(self._active_verification_done),
        ):
            return self._requirement_failure(
                self._active_mutation_done,
                self._active_verification_done,
                reason,
            )
        if controller is not None:
            controller.transition(ConvergencePhase.FINALIZE, reason="final response")
            self._last_convergence = controller.snapshot(reason)
            self._last_phase_trace = controller.phase_snapshot()
        logger.info(
            "agent_convergence_stop",
            reason=reason,
            **self._last_convergence,
            **self._last_phase_trace,
        )
        if not self._can_afford_finalization():
            return f"(stopped by convergence control: {reason}; final-call budget reserved)"
        notice = (
            "[Convergence control] Stop taking actions now. "
            f"Reason: {reason}. Summarize completed work, verification, and any "
            "remaining limitation concisely. Do not request or call tools."
        )
        final_messages = self._full_messages()
        # Keep the provider-compatible invariant that the system message is
        # first instead of inserting a second system role midway through the
        # tool conversation.
        final_messages[0] = {
            "role": "system",
            "content": f"{self._system}\n\n{notice}",
        }
        from structlog.contextvars import bound_contextvars

        with bound_contextvars(agent_phase=ConvergencePhase.FINALIZE.value):
            response = self.llm.chat(
                messages=final_messages,
                tools=[],
                on_token=on_token,
            )
        if response.tool_calls:
            self.messages.append(response.message)
            self._answer_pending_tool_calls(response.tool_calls)
            return f"(stopped by convergence control: {reason})"
        self.messages.append(response.message)
        return redact_output(response.content)

    def _guard_tool_result(self, name: str, result: str) -> str:
        """Fast-scan a tool result for injection; a hit replaces it with a
        notice so the malicious text never reaches the model. A broad cue scan
        gates the semantic classifier, catching indirect attacks without adding
        an LLM call for every ordinary tool result.
        """
        if self._injection is None:
            return result
        blocked, reason = self._injection.defend(
            result,
            use_classifier=True,
            classifier_on_suspicious_only=True,
        )
        if blocked:
            return f"⚠ 工具 {name} 输出疑似包含指令注入，已隔离：{reason}"
        return result

    def _wrap_tool_output(self, name: str, result: str) -> str:
        """<tool_output> role isolation: mark tool results as untrusted data so
        the system prompt's isolation rule applies to every result."""
        protocol = getattr(self.llm, "tool_protocol", None)
        display_name = protocol.to_wire_name(name) if protocol is not None else name
        return f'<tool_output tool="{display_name}">\n{result}\n</tool_output>'

    def _exec_tool(self, tc) -> str:
        """Execute a single tool call, returning the result string."""
        if getattr(tc, "parse_error", None):
            return _ToolExecutionResult(
                f"Error [INVALID_TOOL_INPUT]: {tc.parse_error}; regenerate the complete arguments",
                status="error",
            )
        tool = self._tool_by_name.get(tc.name)
        if tool is None:
            return _ToolExecutionResult(
                f"Error: unknown tool '{tc.name}'",
                status="error",
            )
        # Phase 3 planning guard: soft/hard interception at the dispatch layer
        # (see mycoder/planner.py). Soft mode preserves open-ended editing;
        # MYCODER_ENFORCE_PLANNING=1 hard-blocks mutation without a plan.
        guard_msg = planning_guard(tc.name, query=tc.arguments.get("task_goal"))
        if guard_msg:
            return _ToolExecutionResult(guard_msg, status="blocked")
        # validate arguments first so a TypeError raised *inside* the tool isn't
        # mislabelled as a bad-arguments error from the caller
        try:
            inspect.signature(tool.execute).bind(**tc.arguments)
        except TypeError as e:
            return _ToolExecutionResult(
                f"Error: bad arguments for {tc.name}: {e}",
                status="error",
            )

        # Retry safety and result memoization are separate contracts. A read is
        # safe to retry, but caching it across rounds would return stale
        # workspace state after an edit. Only explicitly cacheable,
        # state-independent tool results use this store.
        idempotent = bool(getattr(tool, "idempotent", True))
        cacheable = bool(getattr(tool, "cacheable", False))
        idem_key = self._idem.key(tool.name, tc.arguments)
        if cacheable:
            cached = self._idem.get(idem_key)
            if cached is not None:
                return _ToolExecutionResult(
                    cached,
                    status="cache_hit",
                    cache_hit=True,
                )
        # A real execution (cache miss) — counted for the tool metrics.
        self._tool_calls += 1
        _started = time.monotonic()
        retry_count = 0

        def _count_retry(_strategy):
            nonlocal retry_count
            retry_count += 1
            self._tool_retries += 1

        try:
            # Phase 3 self-correction: deterministic retry strategies on
            # transient failures (retry_same / retry_modified with timeout
            # extension); everything else surfaces for the agent to reflect.
            # Non-idempotent tools pass retry_safe=False so a side effect that
            # already happened before a failure is never double-applied.
            result = run_with_correction(
                tool.execute,
                retry_safe=idempotent,
                on_retry=_count_retry,
                **tc.arguments,
            )
        except Exception as e:
            self._tool_failure += 1
            duration_ms = (time.monotonic() - _started) * 1000
            self._tool_durations.append(duration_ms)
            return _ToolExecutionResult(
                f"Error executing {tc.name}: {e}",
                status="error",
                duration_ms=duration_ms,
                retry_count=retry_count,
            )
        failed = not self._tool_result_succeeded(result)
        # Never memoize a failed execution. Otherwise a required-mutation retry
        # with corrected context but identical arguments can be served the old
        # error forever and the action loop cannot recover.
        if cacheable and not failed:
            self._idem.put(idem_key, result)
        if failed:
            self._tool_failure += 1
        else:
            self._tool_success += 1
        duration_ms = (time.monotonic() - _started) * 1000
        self._tool_durations.append(duration_ms)
        return _ToolExecutionResult(
            result,
            status="error" if failed else "success",
            exit_code=getattr(result, "exit_code", None),
            duration_ms=duration_ms,
            retry_count=retry_count,
        )

    def _exec_tools_parallel(self, tool_calls, on_tool=None) -> list[str]:
        """Run multiple tool calls concurrently using threads.

        This is inspired by Claude Code's StreamingToolExecutor which starts
        executing tools while the model is still generating.  We simplify to:
        when the model returns N tool calls at once, run them in parallel.

        Duplicate retry-safe calls (same tool + args) within one assistant
        response are coalesced: they observe the same pre-execution round and
        need not run in parallel twice. Cross-round result caching remains a
        separate, explicit ``cacheable`` contract in ``_exec_tool``.
        """
        from .tools.idempotency import _fingerprint

        for tc in tool_calls:
            if on_tool:
                on_tool(tc.name, tc.arguments)

        # tc -> key. Idempotent duplicates share a key (dedup: one run, all
        # reuse the result); non-idempotent calls get a unique key each.
        tc_key: dict[str, tuple[str, str]] = {}
        order: list[tuple[tuple[str, str], object]] = []
        seen: set[tuple[str, str]] = set()
        for i, tc in enumerate(tool_calls):
            tool = self._tool_by_name.get(tc.name)
            idempotent = bool(getattr(tool, "idempotent", True)) if tool else True
            if idempotent:
                key = (tc.name, _fingerprint(tc.arguments))
                if key not in seen:
                    seen.add(key)
                    order.append((key, tc))
            else:
                key = ("__run__", i)
                order.append((key, tc))
            tc_key[tc.id] = key

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {key: pool.submit(self._exec_tool, tc) for key, tc in order}
            results = {key: futures[key].result() for key, _ in order}

        return [results[tc_key[tc.id]] for tc in tool_calls]

    def _answer_pending_tool_calls(self, tool_calls):
        """Backfill a tool reply for every call that didn't get one.

        OpenAI-compatible APIs reject a request where an assistant message has
        tool_calls without a matching tool reply for each id, so this keeps the
        history valid when execution is interrupted partway through.
        """
        answered = {m.get("tool_call_id") for m in self.messages if m.get("role") == "tool"}
        for tc in tool_calls:
            if tc.id not in answered:
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "[interrupted]",
                    }
                )

    def _apply_strategy(self, strategy: str) -> None:
        """Rebuild the system prompt for a (new) reasoning strategy."""
        self.reasoning_strategy = strategy
        self._system = system_prompt(self.tools, reasoning_strategy=strategy)

    def set_strategy(self, name: str) -> str:
        """Switch the reasoning strategy at runtime.

        'auto'  -> per-task auto routing;
        a strategy name -> fixed manual mode (until 'auto' again).
        Returns a human-readable confirmation message.
        """
        name = (name or "").strip().lower()
        if name == "auto":
            self._strategy_mode = "auto"
            return "推理策略：自动（按任务切换）"
        if name in REASONING_STRATEGIES:
            self._strategy_mode = "manual"
            self._apply_strategy(name)
            return f"推理策略：{name}（手动固定；/strategy auto 切回自动）"
        return f"未知策略 {name!r}；可用: auto|{'|'.join(REASONING_STRATEGIES)}"

    def _tool_metrics(self) -> dict:
        """Tool-call success / failure / retry metrics (面经「成功率、重试率、
        工具失败率」的实测来源). Only real executions count — idempotency-cache
        hits don't re-execute, so they're excluded."""
        total = self._tool_calls
        durations = sorted(self._tool_durations)
        avg_d = sum(durations) / len(durations) if durations else 0.0
        p95_d = durations[min(len(durations) - 1, int(len(durations) * 0.95) - 1)] if durations else 0.0
        return {
            "calls": total,
            "successes": self._tool_success,
            "failures": self._tool_failure,
            "retries": self._tool_retries,
            "success_rate": round(self._tool_success / total, 4) if total else 0.0,
            "failure_rate": round(self._tool_failure / total, 4) if total else 0.0,
            "retry_rate": round(self._tool_retries / total, 4) if total else 0.0,
            "avg_duration_ms": round(avg_d, 2),
            "p95_duration_ms": round(p95_d, 2),
            "convergence": dict(self._last_convergence),
            "convergence_phase": self._last_phase_trace.get("phase"),
            "convergence_transitions": list(self._last_phase_trace.get("phase_transitions", [])),
        }

    def reset(self):
        """Clear conversation history."""
        self.messages.clear()
