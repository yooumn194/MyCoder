"""Trace replay — re-execute the deterministic half of a recorded run.

A recorded run has two halves:

* the model's decisions — non-deterministic, already paid for, and
  fundamentally unreproducible (a provider is not a function);
* everything downstream of them — the tool layer, the convergence controller,
  the requirement gates, context compression, the message the loop actually
  sends — which *is* deterministic given those decisions.

``ReplayLLM`` serves the recorded decisions back to the **real** agent loop, so
the second half runs for real: offline, without a provider, without spending
tokens. Anything that changed since the recording — a tool that now returns
something else, a prompt that now renders differently, a policy that now admits
a different call — shows up as a difference, pinned to the exchange or the tool
call where it happened. That is the property a benchmark run needs when it says
"run 3 resolved this instance and run 5 did not": replay tells you whether the
difference is *your* change (the action layer now behaves differently) or the
model's variance (the action layer is identical, the sample was not).

Two independent checks, because they fail differently:

* **request fingerprints** — every LLM call is compared against the recorded
  message digests, so a divergence anywhere in the conversation is caught even
  if no tool was responsible (compression, control text, prompt assembly);
* **tool observations** — each replayed tool result is compared byte-for-byte
  against the recorded one, which says *what* changed and *where*.

Scope: single-agent runs (``execution_mode=single``, or the CLI). Delegated
runs are detected and refused rather than approximated — sub-agent instance ids
and parallel scheduling are not reproducible yet, and a wrong answer there
would be worse than no answer. See ``ReplayReport.notes`` for the evidence.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from mycoder.observability.run_log import (
    RunLog,
    RunLogError,
    RunLogRecorder,
    digest_messages,
    digest_tools,
)
from mycoder.sandbox.logger import get_logger

logger = get_logger("mycoder.replay")

_PREVIEW_CHARS = 400
_DEFAULT_MAX_DIFFERENCES = 25

#: Statuses reported by ``replay`` (also the CLI exit codes).
STATUS_MATCH = "match"
STATUS_DIVERGED = "diverged"
STATUS_UNSUPPORTED = "unsupported"
STATUS_INCOMPLETE = "incomplete"
STATUS_ERROR = "error"

_EXIT_CODES = {
    STATUS_MATCH: 0,
    STATUS_DIVERGED: 1,
    STATUS_UNSUPPORTED: 2,
    STATUS_INCOMPLETE: 2,
    STATUS_ERROR: 2,
}


class ReplayError(RuntimeError):
    """Replay could not be carried out (bad log, unusable environment)."""


class ReplayDivergence(ReplayError):
    """The replayed loop asked the model something the recording never saw."""

    def __init__(self, message: str, *, index: int = -1, expected: str = "", actual: str = "") -> None:
        super().__init__(message)
        self.index = index
        self.expected = expected
        self.actual = actual


class ReplayedProviderError(ReplayError):
    """A recorded exchange was a provider failure; replay raises it again."""


# --------------------------------------------------------------------------
# Report primitives
# --------------------------------------------------------------------------


def _preview(text: str, limit: int = _PREVIEW_CHARS) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… (+{len(text) - limit} chars)"


def _line_difference(expected: str, actual: str) -> tuple[int, str, str] | None:
    """First differing line of two texts, 0-based, or None when identical."""
    expected_lines = expected.splitlines()
    actual_lines = actual.splitlines()
    for index, (left, right) in enumerate(zip(expected_lines, actual_lines)):
        if left != right:
            return index, left, right
    if len(expected_lines) != len(actual_lines):
        index = min(len(expected_lines), len(actual_lines))
        left = expected_lines[index] if index < len(expected_lines) else "<no further line>"
        right = actual_lines[index] if index < len(actual_lines) else "<no further line>"
        return index, left, right
    return None


@dataclass(frozen=True)
class Difference:
    """One place where the replay stopped agreeing with the recording."""

    kind: str
    summary: str
    expected: str = ""
    actual: str = ""
    index: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "summary": self.summary,
            "expected": _preview(self.expected),
            "actual": _preview(self.actual),
            "index": self.index,
        }

    def render(self, position: int | None = None) -> str:
        head = f"  - [{self.kind}] {self.summary}"
        if position is not None:
            head = f"  {position}. [{self.kind}] {self.summary}"
        lines = [head]
        if self.expected or self.actual:
            lines.append(f"      recorded: {_preview(self.expected, 200)!r}")
            lines.append(f"      replayed: {_preview(self.actual, 200)!r}")
        return "\n".join(lines)


@dataclass
class ReplayReport:
    """What replay found. ``status == "match"`` means the action layer is stable."""

    log_path: str
    run_index: int = 0
    mode: str = "single"
    status: str = STATUS_MATCH
    recorded_status: str = ""
    model: str = ""
    answer: str = ""
    replay_answer: str = ""
    differences: list[Difference] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    exchanges_recorded: int = 0
    exchanges_served: int = 0
    tool_calls_recorded: int = 0
    tool_calls_replayed: int = 0
    controls_recorded: int = 0
    controls_replayed: int = 0
    tool_calls_matched: int = 0
    controls_matched: int = 0
    output_log: str | None = None
    error: str | None = None
    max_differences: int = _DEFAULT_MAX_DIFFERENCES
    #: False when the log was analysed without re-running anything. ``status``
    #: stays MATCH (nothing was found wrong), but the report must not claim the
    #: action layer was *verified* — that would be self-certification.
    executed: bool = True
    _dropped: int = field(default=0, repr=False)

    @property
    def matched(self) -> bool:
        return self.status == STATUS_MATCH

    @property
    def first_difference(self) -> Difference | None:
        return self.differences[0] if self.differences else None

    def add(self, difference: Difference) -> None:
        if len(self.differences) < self.max_differences:
            self.differences.append(difference)
        else:
            self._dropped += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "log": self.log_path,
            "run_index": self.run_index,
            "mode": self.mode,
            "status": self.status,
            "recorded_status": self.recorded_status,
            "model": self.model,
            "matched": self.matched,
            "executed": self.executed,
            "exchanges": {"recorded": self.exchanges_recorded, "served": self.exchanges_served},
            "tool_calls": {
                "recorded": self.tool_calls_recorded,
                "replayed": self.tool_calls_replayed,
                "matched": self.tool_calls_matched,
            },
            "controls": {
                "recorded": self.controls_recorded,
                "replayed": self.controls_replayed,
                "matched": self.controls_matched,
            },
            "first_difference": self.first_difference.to_dict() if self.first_difference else None,
            "differences": [item.to_dict() for item in self.differences],
            "differences_dropped": self._dropped,
            "notes": list(self.notes),
            "answer": self.answer,
            "replay_answer": self.replay_answer,
            "output_log": self.output_log,
            "error": self.error,
        }

    def render_markdown(self) -> str:
        if not self.executed and self.matched:
            headline = "ANALYSED (not re-run)"
        elif self.matched:
            headline = "MATCH"
        else:
            headline = self.status.upper()
        lines = [
            f"# Trace replay — {headline}",
            "",
            f"- log: `{self.log_path}` (run #{self.run_index}, mode={self.mode})",
            f"- model: `{self.model}`  recorded status: `{self.recorded_status}`",
            (
                f"- exchanges: {self.exchanges_served}/{self.exchanges_recorded} served  |  "
                f"tool calls: {self.tool_calls_matched}/{self.tool_calls_recorded} identical "
                f"({self.tool_calls_replayed} replayed)  |  "
                f"controls: {self.controls_matched}/{self.controls_recorded} identical"
            ),
        ]
        if self.output_log:
            lines.append(f"- replay log: `{self.output_log}`")
        if self.error:
            lines.append(f"- error: `{self.error}`")
        if self.notes:
            lines.append("")
            lines.append("Notes:")
            lines.extend(f"  - {note}" for note in self.notes)
        if self.differences:
            lines.append("")
            lines.append(f"Differences ({len(self.differences)} shown, {self._dropped} more dropped):")
            lines.extend(item.render(position) for position, item in enumerate(self.differences, start=1))
        elif self.matched and self.executed:
            lines.append("")
            lines.append("The action layer reproduced the recording exactly.")
        lines.append("")
        lines.append("Recorded answer:")
        lines.append(f"```\n{_preview(self.answer, 1500)}\n```")
        if self.replay_answer and self.replay_answer != self.answer:
            lines.append("Replay answer:")
            lines.append(f"```\n{_preview(self.replay_answer, 1500)}\n```")
        return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# ReplayLLM
# --------------------------------------------------------------------------


class ReplayLLM:
    """Serve recorded LLM responses to the real loop, in order.

    Structurally compatible with ``mycoder.llm.LLM`` for everything the loop
    reads (``model``, ``provider``, ``tool_protocol``, token counters) and
    deliberately strict about everything else: an unrecognised request is a
    divergence, never a guess. Serving the wrong response would produce a
    plausible-looking report about a run that never happened.
    """

    #: Replay serves every model tier from the recording: letting the tier
    #: factory build a real client would spend provider tokens mid-replay.
    builds_tier_clients = False

    def __init__(
        self,
        log: RunLog,
        *,
        budget_guard: Any | None = None,
        session_id: str | None = None,
    ) -> None:
        self.log = log
        self.model = log.model or "replay"
        self.provider = log.provider or "unknown"
        self.tool_dialect = log.tool_dialect
        self.extra: dict[str, Any] = {}
        self.caller = "replay"
        self.session_id = session_id or log.session_id or "replay"
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_reasoning_tokens = 0
        self._cursor = 0
        self._tool_protocol: Any | None = None
        self._budget_guard = budget_guard

    # ------------------------------------------------------------- interface
    @property
    def tool_protocol(self):
        """Rebuilt from the recorded model so ``<tool_output>`` framing matches."""
        if self._tool_protocol is None:
            from mycoder.tool_protocol import ToolProtocolAdapter

            self._tool_protocol = ToolProtocolAdapter.for_model(
                self.model,
                self.tool_dialect,
                self.provider,
            )
        return self._tool_protocol

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def remaining(self) -> int:
        return max(0, len(self.log.exchanges) - self._cursor)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token=None,
        response_format: dict | None = None,
        predictive_executor=None,
        tool_choice=None,
        strict_tool_choice: bool = False,
        timeout_seconds: float | None = None,
        request_max_retries: int | None = None,
    ):
        """Return the next recorded response, or raise ``ReplayDivergence``.

        ``predictive_executor`` is intentionally not invoked: speculation is
        what made the original ordering provider-timing dependent, and a replay
        must execute tool calls in recorded order to compare them one by one.
        """
        index = self._match(
            messages,
            tools,
            response_format=response_format,
            tool_choice=tool_choice,
            strict_tool_choice=strict_tool_choice,
        )
        if index is None:
            raise self._divergence(
                messages,
                tools,
                response_format=response_format,
                tool_choice=tool_choice,
                strict_tool_choice=strict_tool_choice,
            )
        exchange = self.log.exchanges[index]
        self._cursor = index + 1
        if exchange.error:
            raise ReplayedProviderError(exchange.error)
        response = _to_response(exchange)
        self.total_prompt_tokens += response.prompt_tokens
        self.total_completion_tokens += response.completion_tokens
        self.total_reasoning_tokens += response.reasoning_tokens
        self._feed_budget(response)
        if on_token is not None and response.content:
            on_token(response.content)
        logger.debug(
            "replay_exchange",
            index=index,
            caller=exchange.caller,
            phase=exchange.phase,
            tool_calls=len(response.tool_calls),
        )
        return response

    # -------------------------------------------------------------- internals
    def _feed_budget(self, response) -> None:
        """Replay the recorded token usage into the budget guard.

        The recorded loop decided its rollout budget from cumulative usage; a
        replay that starts from zero usage would take different control-flow
        branches (extension rounds, requirement reserves) and report a
        divergence that is really a missing input.
        """
        if self._budget_guard is None:
            return
        try:
            self._budget_guard.add_usage(
                self.session_id,
                response.prompt_tokens + response.completion_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - budget bookkeeping is best-effort
            logger.warning("replay_budget_feed_failed", error_msg=str(exc))

    def _match(
        self,
        messages: Sequence[Any],
        tools: Sequence[Any] | None,
        *,
        response_format: Mapping[str, Any] | None = None,
        tool_choice: Any = None,
        strict_tool_choice: bool = False,
    ) -> int | None:
        if self._cursor >= len(self.log.exchanges):
            return None
        if self.log.exchanges[self._cursor].matches(
            messages,
            tools,
            response_format=response_format,
            tool_choice=tool_choice,
            strict_tool_choice=strict_tool_choice,
        ):
            return self._cursor
        return None

    def _divergence(
        self,
        messages: Sequence[Any],
        tools: Sequence[Any] | None,
        *,
        response_format: Mapping[str, Any] | None = None,
        tool_choice: Any = None,
        strict_tool_choice: bool = False,
    ) -> ReplayDivergence:
        if self._cursor >= len(self.log.exchanges):
            return ReplayDivergence(
                (
                    f"the replay made LLM call #{self._cursor + 1}, but the recording only has "
                    f"{len(self.log.exchanges)} — the loop is now running past the recording "
                    "(an extra round, retry, or compaction call)"
                ),
                index=self._cursor,
            )
        expected = self.log.exchanges[self._cursor]
        actual_digests = digest_messages(messages)
        position = expected.locate(messages)
        parts = [
            f"exchange {self._cursor} (caller={expected.caller}, phase={expected.phase}) does not match the recorded request"
        ]
        if expected.message_count != len(actual_digests):
            parts.append(f"message count {expected.message_count} recorded vs {len(actual_digests)} replayed")
        if position is not None:
            label = _message_label(messages, position)
            parts.append(f"first differing message: index {position} ({label})")
            recorded = _recorded_message_text(self.log, expected, position)
            if recorded:
                parts.append(f"recorded text: {_preview(recorded, 200)!r}")
                if position < len(messages):
                    parts.append(f"replayed text: {_preview(str(messages[position].get('content')), 200)!r}")
        if expected.tools_digest != digest_tools(tools):
            parts.append(f"tool catalog changed ({expected.tools_count} recorded vs {len(tools or ())} replayed)")
        format_matches = (
            expected.response_format == bool(response_format)
            if isinstance(expected.response_format, bool)
            else _canonical_value(expected.response_format) == _canonical_value(response_format)
        )
        if not format_matches:
            parts.append(
                f"response_format changed ({expected.response_format!r} recorded vs {response_format!r} replayed)"
            )
        if _canonical_value(expected.tool_choice) != _canonical_value(tool_choice):
            parts.append("tool_choice changed")
        if expected.strict_tool_choice != bool(strict_tool_choice):
            parts.append(
                f"strict_tool_choice changed ({expected.strict_tool_choice} recorded vs {bool(strict_tool_choice)} replayed)"
            )
        return ReplayDivergence(
            "; ".join(parts),
            index=self._cursor,
            expected=_recorded_message_text(self.log, expected, position) if position is not None else "",
            actual=str(messages[position].get("content")) if position is not None and position < len(messages) else "",
        )


def _canonical_value(value: Any) -> str:
    """Canonicalize request options without importing recorder internals."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))


def _response_from_payload(payload: Mapping[str, Any]) -> Any:
    """Rebuild an ``LLMResponse`` from its recorded form."""
    from mycoder.llm import LLMResponse, ToolCall

    tool_calls = [
        ToolCall(
            id=str(item.get("id") or ""),
            name=str(item.get("name") or ""),
            arguments=dict(item.get("arguments") or {}),
            parse_error=item.get("parse_error"),
        )
        for item in payload.get("tool_calls") or []
    ]
    return LLMResponse(
        content=str(payload.get("content") or ""),
        reasoning_content=str(payload.get("reasoning_content") or ""),
        tool_calls=tool_calls,
        prompt_tokens=int(payload.get("prompt_tokens") or 0),
        completion_tokens=int(payload.get("completion_tokens") or 0),
        cached_tokens=int(payload.get("cached_tokens") or 0),
        reasoning_tokens=int(payload.get("reasoning_tokens") or 0),
    )


def _to_response(exchange) -> Any:
    """Rebuild an ``LLMResponse`` from the recorded one."""
    return _response_from_payload(exchange.response or {})


def _message_label(messages: Sequence[Any], position: int) -> str:
    if position >= len(messages):
        return "absent (replayed conversation is shorter)"
    message = messages[position]
    if not isinstance(message, Mapping):
        return type(message).__name__
    role = str(message.get("role") or "?")
    label = f"role={role}"
    for key in ("tool_call_id", "name"):
        if message.get(key):
            label += f", {key}={message[key]}"
    return label


def _reconstruct_messages(log: RunLog, exchange) -> list[dict] | None:
    """Rebuild the recorded conversation as it stood for one exchange.

    Replay stores message *digests* per exchange rather than a conversation
    snapshot per call (that would be quadratic). The conversation is still fully
    recoverable from the append-only stream — prompt, then each recorded
    assistant message, tool observation and injected control in sequence — and
    the reconstruction is verified against the recorded digests before it is
    quoted, so a compaction pass that rewrote earlier messages cannot be
    presented as "what the model saw".

    The system message is the one exception: only its digest is recorded, so it
    is replaced by a placeholder. This transcript is for *quoting* a divergence,
    not for replaying one — feed the live request to ``ReplayLLM`` instead.
    """
    messages: list[dict] = [
        {"role": "system", "content": "<system prompt: recorded only as a digest>"},
        {"role": "user", "content": log.prompt},
    ]
    for event in log.events:
        if int(event.get("seq", 0)) >= exchange.seq:
            break
        kind = event.get("type")
        if kind == "llm_call" and event.get("response") is not None:
            # LLMResponse.message is the single source of truth for how an
            # assistant turn is serialised into the transcript.
            messages.append(_response_from_payload(event["response"]).message)
        elif kind == "tool_result":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": event.get("tool_call_id"),
                    "content": event.get("content") or "",
                }
            )
        elif kind == "control":
            messages.append({"role": "user", "content": event.get("content") or ""})
    if exchange.index == 0:
        # The first request is the one that carried this prompt; nothing else
        # has been appended yet, so its length must already agree.
        if len(messages) != exchange.message_count:
            return None
        return messages
    recovered = digest_messages(messages)
    # Position 0 is the system prompt (recorded as a digest only); the carried
    # conversation must otherwise reproduce the recorded fingerprints exactly.
    if recovered[1:] != list(exchange.digests[1:]):
        return None
    return messages


def _recorded_message_text(log: RunLog, exchange, position: int) -> str:
    """The recorded text of one request message, or '' when unrecoverable."""
    if position < 0 or position >= exchange.message_count:
        return ""
    messages = _reconstruct_messages(log, exchange)
    if messages is None or position >= len(messages):
        return ""
    content = messages[position].get("content")
    return str(content) if content is not None else ""


# --------------------------------------------------------------------------
# Comparison bookkeeping
# --------------------------------------------------------------------------


class _Tracker:
    """Ordered comparison of what the replay produced against the recording."""

    def __init__(self, log: RunLog) -> None:
        self.log = log
        self._tool_cursor = 0
        self._control_cursor = 0

    # ------------------------------------------------------------ tool calls
    def observe_tool_result(self, payload: Mapping[str, Any]) -> list[Difference]:
        differences: list[Difference] = []
        if self._tool_cursor >= len(self.log.tool_results):
            return [
                Difference(
                    kind="extra_tool_call",
                    summary=(
                        f"tool `{payload.get('name')}` ran at position {self._tool_cursor}, "
                        "but the recording has no further tool calls"
                    ),
                    actual=str(payload.get("content") or ""),
                    index=self._tool_cursor,
                )
            ]
        expected = self.log.tool_results[self._tool_cursor]
        position = self._tool_cursor
        self._tool_cursor += 1
        if str(payload.get("name")) != expected.name:
            differences.append(
                Difference(
                    kind="tool_sequence",
                    summary=(
                        f"tool call {position} is `{payload.get('name')}` in the replay but `{expected.name}` in the recording"
                    ),
                    expected=expected.name,
                    actual=str(payload.get("name")),
                    index=position,
                )
            )
            return differences
        content = str(payload.get("content") or "")
        if content != expected.content:
            line = _line_difference(expected.content, content)
            detail = ""
            if line is not None:
                number, recorded_line, replayed_line = line
                detail = f" (first differing line {number + 1})"
            differences.append(
                Difference(
                    kind="tool_result",
                    summary=(f"`{expected.name}` (call {position}, {expected.tool_call_id}) returned different output{detail}"),
                    expected=expected.content,
                    actual=content,
                    index=position,
                )
            )
        if str(payload.get("status")) != expected.status:
            differences.append(
                Difference(
                    kind="tool_status",
                    summary=(
                        f"`{expected.name}` (call {position}) finished `{payload.get('status')}` "
                        f"in the replay but `{expected.status}` in the recording"
                    ),
                    expected=expected.status,
                    actual=str(payload.get("status")),
                    index=position,
                )
            )
        for field_name in ("mutation", "verification", "blocked"):
            if bool(payload.get(field_name)) != bool(getattr(expected, field_name)):
                differences.append(
                    Difference(
                        kind="tool_flags",
                        summary=(
                            f"`{expected.name}` (call {position}) classified "
                            f"{field_name}={bool(payload.get(field_name))} in the replay but "
                            f"{bool(getattr(expected, field_name))} in the recording"
                        ),
                        expected=str(getattr(expected, field_name)),
                        actual=str(bool(payload.get(field_name))),
                        index=position,
                    )
                )
        return differences

    # -------------------------------------------------------------- controls
    def observe_control(self, payload: Mapping[str, Any]) -> list[Difference]:
        differences: list[Difference] = []
        if self._control_cursor >= len(self.log.controls):
            return [
                Difference(
                    kind="extra_control",
                    summary=(
                        f"the loop injected a `{payload.get('kind')}` message, but the recording has no further control messages"
                    ),
                    actual=str(payload.get("content") or ""),
                    index=self._control_cursor,
                )
            ]
        expected = self.log.controls[self._control_cursor]
        position = self._control_cursor
        self._control_cursor += 1
        if str(payload.get("content") or "") != str(expected.get("content") or ""):
            differences.append(
                Difference(
                    kind="control",
                    summary=(
                        f"control message {position} (`{expected.get('kind')}`) differs — the loop's injected feedback changed"
                    ),
                    expected=str(expected.get("content") or ""),
                    actual=str(payload.get("content") or ""),
                    index=position,
                )
            )
        return differences

    def trailing(self) -> list[Difference]:
        differences: list[Difference] = []
        for position in range(self._tool_cursor, len(self.log.tool_results)):
            expected = self.log.tool_results[position]
            differences.append(
                Difference(
                    kind="missing_tool_call",
                    summary=(f"the recording has tool call {position} (`{expected.name}`) that the replay never made"),
                    expected=expected.content,
                    index=position,
                )
            )
        return differences

    @property
    def tools_compared(self) -> int:
        return min(self._tool_cursor, len(self.log.tool_results))

    @property
    def controls_compared(self) -> int:
        return min(self._control_cursor, len(self.log.controls))


class _ReplayRecorder(RunLogRecorder):
    """A run recorder that also compares every event against the recording.

    Writing the replay's own log is what makes the result auditable: the two
    files can be diffed directly, and the comparison is a side effect of the
    same call rather than a second, drift-prone code path.
    """

    def __init__(self, target: str | Path, tracker: _Tracker, *, write: bool = True) -> None:
        super().__init__(target, run_id="replay")
        self._tracker = tracker
        self._write = write
        self.differences: list[Difference] = []

    def record_tool_result(self, **payload):  # type: ignore[override]
        self.differences.extend(self._tracker.observe_tool_result(payload))
        return super().record_tool_result(**payload) if self._write else None

    def record_control(self, **payload):  # type: ignore[override]
        self.differences.extend(self._tracker.observe_control(payload))
        return super().record_control(**payload) if self._write else None


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def _convergence_limits(flags: Mapping[str, Any], max_rounds: int):
    from mycoder.convergence import ConvergenceLimits

    limits = ConvergenceLimits.from_env(max_rounds)
    updates = {}
    for key, field_name in (
        ("convergence_max_rounds", "max_rounds"),
        ("max_tool_calls", "max_tool_calls"),
        ("max_identical_tool_calls", "max_identical_tool_calls"),
        ("max_stagnant_rounds", "max_stagnant_rounds"),
        ("soft_budget_ratio", "soft_budget_ratio"),
    ):
        if flags.get(key) is not None:
            updates[field_name] = flags[key]
    if not updates:
        return limits
    from dataclasses import replace

    return replace(limits, **updates)


def _build_tools(
    window: RunLog,
    tools: Sequence[Any] | None,
    workspace: str | Path | None,
    sandbox_policy: str,
    sandbox_image: str | None,
    sandbox_user: str | None,
):
    if tools is not None:
        # Embedding callers (and tests) supply the exact registry; no sandbox
        # manager is created, so nothing has to be torn down.
        return list(tools), None
    if workspace is None:
        from mycoder.tools import ALL_TOOLS

        return list(ALL_TOOLS), None
    from mycoder.tools import build_scoped_tools

    return build_scoped_tools(
        workspace,
        window.session_id or f"replay-{window.path.stem}",
        sandbox_policy=sandbox_policy,
        sandbox_image=sandbox_image,
        sandbox_user=sandbox_user,
    )


def replay(
    path: str | Path,
    *,
    run_index: int = 0,
    mode: str = "auto",
    workspace: str | Path | None = None,
    tools: Sequence[Any] | None = None,
    sandbox_policy: str = "interactive",
    sandbox_image: str | None = None,
    sandbox_user: str | None = None,
    output: str | Path | None = None,
    write_log: bool = True,
    max_differences: int = _DEFAULT_MAX_DIFFERENCES,
    max_rounds: int | None = None,
    execute: bool = True,
) -> ReplayReport:
    """Re-execute the recorded run's deterministic half and compare.

    ``execute=False`` analyses the log without running anything (useful on a
    huge or foreign log). ``workspace`` selects the repository root and sandbox
    wiring; without it the process-wide tools are used, exactly like the
    interactive CLI. ``tools`` overrides both, for embedding callers that own
    the registry.
    """
    log = RunLog.load(path)
    window = log.window(run_index)
    report = ReplayReport(
        log_path=str(Path(path).expanduser()),
        run_index=run_index,
        recorded_status=window.status,
        model=window.model,
        answer=window.answer,
        exchanges_recorded=len(window.exchanges),
        tool_calls_recorded=len(window.tool_results),
        controls_recorded=len(window.controls),
        max_differences=max(1, int(max_differences)),
        executed=bool(execute),
    )
    if mode == "auto":
        mode = "multi" if window.multi_agent else "single"
    report.mode = mode
    if mode != "single":
        report.status = STATUS_UNSUPPORTED
        report.notes.append(
            "delegated (multi-agent) runs are not replayable yet: sub-agent instance ids and "
            "parallel scheduling are not reproducible, so a replay would report divergence that "
            "belongs to the wiring rather than to the change under test."
        )
        report.notes.append(
            f"evidence of delegation: {len(window.nested_starts)} nested run_start event(s), "
            f"{sum(1 for item in window.tool_results if item.subagent not in {'', 'main'})} "
            "sub-agent tool call(s)"
        )
        return report
    incomplete_recording = not window.complete
    if incomplete_recording:
        report.status = STATUS_INCOMPLETE
        report.notes.append(
            f"the recording has no run_end event (tail truncated: {window.partial_tail}); replaying what was recorded."
        )
    if not execute:
        report.notes.append("execute=False: the log was analysed but nothing was re-run.")
        return report

    flags = window.flags
    rounds = int(max_rounds or flags.get("max_rounds") or 50)
    limits = _convergence_limits(flags, rounds)
    leader = None
    try:
        tools, leader = _build_tools(window, tools, workspace, sandbox_policy, sandbox_image, sandbox_user)
    except Exception as exc:  # noqa: BLE001 - a broken environment is a reportable outcome
        report.status = STATUS_ERROR
        report.error = f"could not build the tool set: {exc}"
        return report

    guard = _build_guard(flags)
    tracker = _Tracker(window)
    recorder = _ReplayRecorder(
        output or _default_output_path(window.path),
        tracker,
        write=write_log,
    )
    report.output_log = str(recorder.path)
    llm = ReplayLLM(window, budget_guard=guard, session_id=window.session_id)
    report.notes.extend(_environment_notes(window, workspace, sandbox_policy, flags, leader))

    agent = _build_agent(window, llm, tools, guard, limits, recorder, rounds)
    from structlog.contextvars import bound_contextvars

    answer = ""
    # Preserve the evidence boundary: an exact replay of a log prefix is not
    # a complete verification.  A later divergence/error may upgrade this to
    # a stronger failure status, but a clean prefix must stay INCOMPLETE.
    status = STATUS_INCOMPLETE if incomplete_recording else STATUS_MATCH
    llm_difference: Difference | None = None
    try:
        with bound_contextvars(session_id=llm.session_id):
            answer = agent.chat(window.prompt)
    except ReplayDivergence as exc:
        status = STATUS_DIVERGED
        llm_difference = Difference(
            kind="llm_request",
            summary=str(exc),
            expected=exc.expected,
            actual=exc.actual,
            index=exc.index,
        )
    except ReplayedProviderError as exc:
        recorded_error = (window.run_end or {}).get("error") or ""
        if recorded_error and str(exc) == str(recorded_error):
            report.notes.append("the recorded provider failure was reproduced verbatim (error paths replay too)")
        else:
            status = STATUS_ERROR
            report.error = str(exc)
    except Exception as exc:  # noqa: BLE001 - any crash is part of the verdict
        status = STATUS_ERROR
        report.error = f"{type(exc).__name__}: {exc}"
    finally:
        recorder.close()
        _close_manager(leader)

    # All differences in the order they were discovered: the tool/control layer
    # first (they happened while the loop ran), then the request that could not
    # be served, then anything the recording had that the replay never reached.
    ordered = [*recorder.differences, *tracker.trailing()]
    if llm_difference is not None:
        ordered.append(llm_difference)
    for difference in ordered:
        report.add(difference)
    report.tool_calls_replayed = tracker.tools_compared
    report.tool_calls_matched = _matched_count(ordered, tracker.tools_compared, _TOOL_KINDS)
    report.controls_replayed = tracker.controls_compared
    report.controls_matched = _matched_count(ordered, tracker.controls_compared, _CONTROL_KINDS)
    report.exchanges_served = min(llm.cursor, len(window.exchanges))
    report.replay_answer = answer
    if status in {STATUS_MATCH, STATUS_INCOMPLETE} and ordered:
        status = STATUS_DIVERGED
    report.status = status
    logger.info(
        "replay_finished",
        log=report.log_path,
        status=report.status,
        differences=len(report.differences),
    )
    return report


_TOOL_KINDS = frozenset({"tool_result", "tool_status", "tool_flags", "tool_sequence", "extra_tool_call", "missing_tool_call"})
_CONTROL_KINDS = frozenset({"control", "extra_control"})


def _matched_count(differences: Sequence[Difference], compared: int, kinds: frozenset[str]) -> int:
    failing = {item.index for item in differences if item.kind in kinds}
    return max(0, compared - len(failing))


def _build_guard(flags: Mapping[str, Any]):
    maximum = flags.get("budget_max_tokens")
    if maximum is None:
        return None
    from mycoder.observability.budget import TokenBudgetGuard

    return TokenBudgetGuard(max_tokens_per_session=int(maximum))


def _build_agent(window: RunLog, llm, tools, guard, limits, recorder, rounds: int):
    from mycoder.agent import Agent

    flags = window.flags
    strategy_mode = str(flags.get("strategy_mode") or "auto")
    reasoning_strategy = None if strategy_mode == "auto" else flags.get("reasoning_strategy")
    return Agent(
        llm=llm,
        tools=tools,
        max_context_tokens=int(flags.get("max_context_tokens") or 128_000),
        max_rounds=rounds,
        reasoning_strategy=reasoning_strategy,
        budget_guard=guard,
        convergence_limits=limits,
        require_mutation=bool(flags.get("require_mutation")),
        require_verification=bool(flags.get("require_verification")),
        strict_tool_choice=bool(flags.get("strict_tool_choice")),
        max_turn_tokens=flags.get("max_turn_tokens"),
        reserved_tokens=int(flags.get("reserved_tokens") or 0),
        mutation_reserved_tokens=flags.get("mutation_reserved_tokens"),
        verification_reserved_tokens=flags.get("verification_reserved_tokens"),
        run_recorder=recorder,
    )


def _default_output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.replay.jsonl")


def _close_manager(manager) -> None:
    """Tear down a sandbox manager this replay created (never a caller's).

    ``SandboxManager`` exposes ``stop`` as a coroutine and ``stop_sync`` as its
    blocking twin; calling the coroutine without awaiting it would silently
    leave the container running, so only blocking closers are used.
    """
    if manager is None:
        return
    for name in ("stop_sync", "cleanup", "close", "stop"):
        closer = getattr(manager, name, None)
        if not callable(closer) or inspect.iscoroutinefunction(closer):
            continue
        try:
            closer()
        except Exception as exc:  # noqa: BLE001 - teardown must not mask the verdict
            logger.warning("replay_sandbox_teardown_failed", error_msg=str(exc))
        return
    logger.debug("replay_sandbox_teardown_skipped", manager=type(manager).__name__)


def _environment_notes(window: RunLog, workspace, sandbox_policy, flags, leader) -> list[str]:
    """Say out loud when the replay does not run where the recording did."""
    notes: list[str] = []
    recorded_root = str(flags.get("project_root") or "") or None
    if recorded_root:
        current_root = str(getattr(leader, "project_dir", "") or (workspace or "") or os.getcwd())
        if os.path.abspath(current_root) != os.path.abspath(recorded_root):
            notes.append(
                "workspace differs: recorded at "
                f"`{recorded_root}`, replayed at `{current_root}` — tool output that depends on "
                "absolute paths or repository state will differ"
            )
    elif window.cwd and os.path.abspath(window.cwd) != os.path.abspath(os.getcwd()):
        notes.append(
            f"working directory differs: recorded in `{window.cwd}`, replayed in `{os.getcwd()}` "
            "— host tools resolve relative paths against the current directory"
        )
    recorded_policy = flags.get("sandbox_policy")
    if recorded_policy and str(recorded_policy) != str(sandbox_policy):
        notes.append(
            f"sandbox policy differs: recorded `{recorded_policy}`, replayed `{sandbox_policy}` "
            "(pass --sandbox-policy/--sandbox-image/--sandbox-user to match)"
        )
    if flags.get("memory"):
        notes.append(
            "memory was enabled during recording but is not installed for a replay: retrieval is "
            "not reproducible (embeddings + store contents), so planner prompts that consumed a "
            "memory section will be reported as a divergence"
        )
    return notes


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mycoder.replay",
        description="Re-run a recorded run's tool/loop layer without a provider and report where it diverges.",
    )
    parser.add_argument("log", type=Path, help="run log written by RunLogRecorder (events.jsonl)")
    parser.add_argument("--run-index", type=int, default=0, help="which top-level run in the file (default 0)")
    parser.add_argument("--list-runs", action="store_true", help="list the runs in the file and exit")
    parser.add_argument("--mode", choices=["auto", "single", "multi"], default="auto")
    parser.add_argument("--workspace", type=Path, default=None, help="repository root to replay against")
    parser.add_argument("--sandbox-policy", default="interactive", choices=["interactive", "benchmark"])
    parser.add_argument("--sandbox-image", default=None)
    parser.add_argument("--sandbox-user", default=None)
    parser.add_argument("--out", type=Path, default=None, help="where to write the replay's own run log")
    parser.add_argument("--json", type=Path, default=None, help="write the report as JSON")
    parser.add_argument("--max-rounds", type=int, default=None, help="override the recorded round limit")
    parser.add_argument("--max-differences", type=int, default=_DEFAULT_MAX_DIFFERENCES)
    parser.add_argument("--no-execute", action="store_true", help="analyse the log without re-running")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.list_runs:
        try:
            log = RunLog.load(args.log)
        except RunLogError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return _EXIT_CODES[STATUS_ERROR]
        for index, (start, end) in enumerate(log.window_bounds()):
            window = log.window(index)
            print(
                f"#{index}  prompt={_preview(window.prompt, 80)!r}  "
                f"status={window.status}  exchanges={len(window.exchanges)}  tools={len(window.tool_results)}  "
                f"mode={'multi' if window.multi_agent else 'single'}"
            )
        return _EXIT_CODES[STATUS_MATCH]

    try:
        report = replay(
            args.log,
            run_index=args.run_index,
            mode=args.mode,
            workspace=args.workspace,
            sandbox_policy=args.sandbox_policy,
            sandbox_image=args.sandbox_image,
            sandbox_user=args.sandbox_user,
            output=args.out,
            max_differences=args.max_differences,
            max_rounds=args.max_rounds,
            execute=not args.no_execute,
        )
    except RunLogError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_CODES[STATUS_ERROR]
    print(report.render_markdown())
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report written to {args.json}")
    return _EXIT_CODES.get(report.status, 1)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
