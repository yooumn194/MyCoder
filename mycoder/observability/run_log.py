"""Append-only run logs — the event stream that trace replay consumes.

`ObservabilityStore` is built for *aggregation*: redacted previews, TTL expiry,
counters. Reproducing a run needs the opposite properties, so the run log is a
separate artifact rather than another trace type:

* **complete** — every LLM exchange (request fingerprint + the full response)
  and every tool observation, in the order they happened;
* **append-only** — one JSON object per line, never rewritten, so a crashed or
  killed run still leaves a valid prefix that can be replayed;
* **faithful** — the exact text the model saw, not a truncated preview. The
  digests replay compares are computed over that text, so a lossy log cannot
  round-trip.

One file per run, written next to the run it describes (an eval run directory,
a debugging session). `mycoder.replay` reads it back and re-executes the
deterministic half of the run without a provider.

Security: a run log reproduces conversation text verbatim — that is precisely
what makes it replayable, and it is the same text `~/.mycoder/sessions/*.json`
already stores. Enable recording for trusted workloads (benchmark instances,
local debugging) and treat the resulting files as sensitive. `MYCODER_RUN_LOG_MAX_CHARS`
bounds any single recorded field when a log would otherwise be too large to
keep; a truncated log is still replayable, it just loses byte-fidelity past
the cap and says so.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mycoder.sandbox.logger import get_logger

logger = get_logger("mycoder.run_log")

SCHEMA_VERSION = 1
_DIGEST_CHARS = 16
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

RUN_START = "run_start"
LLM_CALL = "llm_call"
TOOL_RESULT = "tool_result"
CONTROL = "control"
RUN_END = "run_end"


class RunLogError(RuntimeError):
    """The run log is missing, malformed, or written by a newer schema."""


# --------------------------------------------------------------------------
# Fingerprints
#
# Replay compares requests, not responses: a request is the entire conversation
# the loop has built so far, so one digest list pinpoints *where* two runs
# stopped agreeing without storing the conversation twice.
# --------------------------------------------------------------------------


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )


def digest(value: Any) -> str:
    """Stable short digest of any JSON-serialisable value."""
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()[:_DIGEST_CHARS]


def message_digest(message: Any) -> str:
    """Digest one request message including every field the provider sees."""
    if isinstance(message, Mapping):
        return digest({str(key): message[key] for key in sorted(message)})
    return digest(message)


def digest_messages(messages: Sequence[Any] | None) -> list[str]:
    return [message_digest(message) for message in messages or []]


def digest_tools(tools: Sequence[Any] | None) -> str | None:
    if not tools:
        return None
    return digest([_tool_signature(tool) for tool in tools])


def _tool_signature(tool: Any) -> Any:
    """Reduce a tool schema to what actually reaches the provider.

    Internal callers pass either a wire schema dict or a ``Tool`` object; both
    must fingerprint identically for the same catalog.
    """
    if isinstance(tool, Mapping):
        return {str(key): tool[key] for key in sorted(tool)}
    schema = getattr(tool, "schema", None)
    if callable(schema):
        try:
            return _tool_signature(schema())
        except Exception:  # noqa: BLE001 - a broken schema must not break logging
            return str(tool)
    return str(tool)


def _context() -> dict[str, Any]:
    """Best-effort structlog contextvars snapshot (never raises)."""
    try:
        from structlog.contextvars import get_contextvars

        values = get_contextvars()
    except Exception:  # noqa: BLE001 - context is optional
        return {}
    return {key: values.get(key) for key in ("session_id", "agent_phase", "subagent_name") if values.get(key)}


def _bounded(value: str, limit: int) -> tuple[str, bool]:
    if limit and len(value) > limit:
        return value[:limit], True
    return value, False


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False, default=str)
        return value
    except (TypeError, ValueError):
        return str(value)


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------


class RunLogRecorder:
    """Append-only JSONL writer. One instance per run.

    Every method is best-effort: a recorder failure logs a warning and disables
    itself rather than breaking the run it observes. Writes are serialised so
    concurrent tool calls (predictive execution, parallel sub-agents) cannot
    interleave a partial line.
    """

    def __init__(
        self,
        target: str | Path,
        *,
        run_id: str | None = None,
        max_field_chars: int | None = None,
        run_context: Mapping[str, Any] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._handle = None
        self._broken = False
        self.path = _resolve_path(target, run_id)
        limit = max_field_chars
        if limit is None:
            try:
                limit = int(os.getenv("MYCODER_RUN_LOG_MAX_CHARS", "0"))
            except ValueError:
                limit = 0
        self.max_field_chars = max(0, int(limit or 0))
        self.run_id = run_id or self.path.stem
        #: Facts the *caller* knows and the agent does not (execution mode,
        #: sandbox policy, tenant/workspace). Merged into the run_start flags so
        #: the log stays self-describing for replay.
        self.run_context = dict(run_context or {})

    # ------------------------------------------------------------- lifecycle
    def _ensure_handle(self):
        if self._handle is not None:
            return self._handle
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        return self._handle

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                try:
                    self._handle.close()
                finally:
                    self._handle = None

    def __enter__(self) -> "RunLogRecorder":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def write(self, event: Mapping[str, Any]) -> dict | None:
        """Append one event. Returns the stored record, or None if disabled."""
        with self._lock:
            if self._broken:
                return None
            record = {"seq": self._seq, "ts": time.time(), **event}
            self._seq += 1
            line = json.dumps(record, ensure_ascii=False, default=str)
            try:
                handle = self._ensure_handle()
                handle.write(line + "\n")
                handle.flush()
            except Exception as exc:  # noqa: BLE001 - recording never breaks a run
                logger.warning("run_log_write_failed", path=str(self.path), error_msg=str(exc))
                self._broken = True
                return None
            return record

    # ---------------------------------------------------------------- events
    def record_run_start(
        self,
        *,
        prompt: str,
        model: str,
        provider: str = "unknown",
        cwd: str | None = None,
        tools: Sequence[str] | None = None,
        flags: Mapping[str, Any] | None = None,
        tool_dialect: str | None = None,
        session_id: str | None = None,
        version: str | None = None,
        nested: bool = False,
    ) -> dict | None:
        """Run metadata: everything replay needs to rebuild an equivalent agent.

        The flags are not decoration — rebuilding an agent without
        ``require_mutation`` / ``soft_budget_ratio`` / ``max_rounds`` produces a
        *different* loop, and the divergence would be blamed on the tools.
        """
        merged: dict[str, Any] = {str(key): _jsonable(value) for key, value in self.run_context.items()}
        merged.update({str(key): _jsonable(value) for key, value in (flags or {}).items()})
        return self.write(
            {
                "type": RUN_START,
                "schema": SCHEMA_VERSION,
                "run_id": self.run_id,
                "session_id": session_id,
                "model": str(model),
                "provider": str(provider),
                "tool_dialect": tool_dialect,
                "cwd": cwd or os.getcwd(),
                "prompt": prompt,
                "tools": [str(name) for name in (tools or [])],
                "flags": merged,
                "version": version,
                "nested": bool(nested),
            }
        )

    def record_llm_call(
        self,
        *,
        messages: Sequence[Any] | None,
        tools: Sequence[Any] | None = None,
        response: Any | None = None,
        error: str | None = None,
        duration_ms: float = 0.0,
        caller: str | None = None,
        phase: str | None = None,
        response_format: Any = None,
        tool_choice: Any = None,
        strict_tool_choice: bool = False,
        streamed: bool = False,
    ) -> dict | None:
        """One LLM exchange: the request fingerprint plus the full response."""
        context = _context()
        digest_list = digest_messages(messages)
        content, content_capped = _bounded(str(getattr(response, "content", "") or ""), self.max_field_chars)
        reasoning, reasoning_capped = _bounded(str(getattr(response, "reasoning_content", "") or ""), self.max_field_chars)
        return self.write(
            {
                "type": LLM_CALL,
                "caller": caller or context.get("subagent_name") or "unknown",
                "phase": phase or context.get("agent_phase"),
                "session_id": context.get("session_id"),
                "request": {
                    "messages": len(digest_list),
                    "digests": digest_list,
                    "tools": digest_tools(tools),
                    "tools_count": len(tools or ()),
                    # Keep the provider's actual schema, not just a truthy
                    # bit: two different JSON schemas both evaluate to True
                    # but can change the model response materially.
                    "response_format": _jsonable(response_format) if response_format is not None else None,
                    "tool_choice": _jsonable(tool_choice) if tool_choice is not None else None,
                    "strict_tool_choice": bool(strict_tool_choice),
                    "streamed": bool(streamed),
                },
                "response": None
                if response is None
                else {
                    "content": content,
                    "content_truncated": content_capped,
                    "reasoning_content": reasoning,
                    "reasoning_truncated": reasoning_capped,
                    "tool_calls": [_serialize_tool_call(call) for call in getattr(response, "tool_calls", None) or []],
                    "prompt_tokens": int(getattr(response, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(response, "completion_tokens", 0) or 0),
                    "cached_tokens": int(getattr(response, "cached_tokens", 0) or 0),
                    "reasoning_tokens": int(getattr(response, "reasoning_tokens", 0) or 0),
                },
                "error": error,
                "duration_ms": round(float(duration_ms), 2),
            }
        )

    def record_tool_result(
        self,
        *,
        name: str,
        tool_call_id: str,
        arguments: Mapping[str, Any] | None,
        content: str,
        status: str,
        round_index: int | None = None,
        phase: str | None = None,
        subagent: str = "main",
        mutation: bool = False,
        verification: bool = False,
        blocked: bool = False,
        duration_ms: float = 0.0,
        retry_count: int = 0,
        cache_hit: bool = False,
    ) -> dict | None:
        """One tool observation, recorded as it entered the transcript.

        ``content`` is post-guard and post-wrapping: the exact string the model
        saw. Replay compares against that, so anything the loop normally
        transforms (injection isolation, ``<tool_output>`` framing) is already
        applied on both sides.
        """
        stored, capped = _bounded(str(content or ""), self.max_field_chars)
        return self.write(
            {
                "type": TOOL_RESULT,
                "round": round_index,
                "phase": phase or _context().get("agent_phase"),
                "subagent": subagent,
                "tool_call_id": str(tool_call_id),
                "name": str(name),
                "arguments": _jsonable(dict(arguments or {})),
                "status": str(status),
                "mutation": bool(mutation),
                "verification": bool(verification),
                "blocked": bool(blocked),
                "duration_ms": round(float(duration_ms), 2),
                "retry_count": int(retry_count),
                "cache_hit": bool(cache_hit),
                "content": stored,
                "content_truncated": capped,
            }
        )

    def record_control(
        self,
        *,
        kind: str,
        content: str,
        round_index: int | None = None,
        attempt: int | None = None,
        phase: str | None = None,
    ) -> dict | None:
        """A message the *loop* injected (mutation feedback, requirement prompt).

        These are derived, not sampled: replay recomputes them and diffs the
        text, which is how a changed control rule becomes visible.
        """
        stored, capped = _bounded(str(content or ""), self.max_field_chars)
        return self.write(
            {
                "type": CONTROL,
                "round": round_index,
                "kind": str(kind),
                "attempt": attempt,
                "phase": phase,
                "content": stored,
                "content_truncated": capped,
            }
        )

    def record_run_end(
        self,
        *,
        status: str,
        answer: str = "",
        error: str | None = None,
        rounds: int | None = None,
        nested: bool = False,
    ) -> dict | None:
        stored, capped = _bounded(str(answer or ""), self.max_field_chars)
        return self.write(
            {
                "type": RUN_END,
                "run_id": self.run_id,
                "status": str(status),
                "answer": stored,
                "answer_truncated": capped,
                "error": error,
                "rounds": rounds,
                "nested": bool(nested),
            }
        )


def _resolve_path(target: str | Path, run_id: str | None) -> Path:
    path = Path(target).expanduser()
    if path.suffix == ".jsonl":
        return path.resolve()
    name = _SAFE_NAME_RE.sub("-", str(run_id or "run")).strip(".-_") or "run"
    return (path / f"{name}.jsonl").resolve()


def _serialize_tool_call(call: Any) -> dict[str, Any]:
    arguments = getattr(call, "arguments", None)
    if arguments is None and isinstance(call, Mapping):
        arguments = call.get("arguments")
    return {
        "id": str(getattr(call, "id", None) or (call.get("id") if isinstance(call, Mapping) else "") or ""),
        "name": str(getattr(call, "name", None) or (call.get("name") if isinstance(call, Mapping) else "") or ""),
        "arguments": _jsonable(arguments if isinstance(arguments, dict) else {}),
        "parse_error": getattr(call, "parse_error", None),
    }


def llm_core(llm: Any) -> Any:
    """The provider client behind any transparent LLM decorator."""
    return getattr(llm, "wrapped_llm", llm)


def llm_like(template: Any, llm: Any) -> Any:
    """Rewrap a freshly built LLM the way ``template`` was wrapped.

    Used by ``model_router.build_model_factory`` when it constructs a second
    client for another tier: the new client inherits the template's decorators
    instead of dropping out of the recording.
    """
    recorder = getattr(template, "run_log_recorder", None)
    if recorder is None or isinstance(llm, RecordingLLM):
        return llm
    return RecordingLLM(llm, recorder)


def recording_model_factory(factory, recorder: RunLogRecorder):
    """Wrap a model-tier factory so tier sub-agents are recorded too.

    Without this, ``build_model_factory`` hands a sub-agent a *freshly built*
    LLM for its tier, whose calls never reach the recorder — the log would look
    complete while silently missing those exchanges, which is worse than no log
    at all.
    """
    if factory is None or recorder is None:
        return factory

    def wrapped(tier: str | None):
        llm = factory(tier)
        if llm is None or isinstance(llm, RecordingLLM):
            return llm
        return RecordingLLM(llm, recorder)

    return wrapped


class RecordingLLM:
    """Transparent LLM decorator that appends every exchange to a run log.

    Unknown attributes are delegated to the wrapped instance, so callers that
    read ``model`` / ``provider`` / ``client`` / ``extra`` (including the
    model-tier factory) cannot tell the difference.
    """

    #: This wrapper forwards provider request options (``timeout_seconds`` /
    #: ``request_max_retries``), so capability branches that would prefer a real
    #: provider client should treat it as one. Recording must be observation-only:
    #: silently downgrading a caller to a different code path would make the
    #: recorded run differ from the unrecorded one it claims to describe.
    supports_request_options = True

    def __init__(self, llm: Any, recorder: RunLogRecorder | None) -> None:
        object.__setattr__(self, "_llm", llm)
        object.__setattr__(self, "_recorder", recorder)

    def __getattr__(self, name: str) -> Any:
        try:
            inner = object.__getattribute__(self, "_llm")
        except AttributeError:  # pragma: no cover - only during partial init
            raise AttributeError(name) from None
        return getattr(inner, name)

    # ``model`` is the one attribute callers *write* (the CLI switches model on
    # resume and via /model). Plain delegation would set it on the wrapper and
    # leave the provider request on the old model, so it gets a real forwarder.
    @property
    def model(self) -> Any:
        return object.__getattribute__(self, "_llm").model

    @model.setter
    def model(self, value: Any) -> None:
        object.__getattribute__(self, "_llm").model = value

    @property
    def run_log_path(self) -> Path | None:
        recorder = object.__getattribute__(self, "_recorder")
        return recorder.path if recorder is not None else None

    # ------------------------------------------------- decorator propagation
    # Anything that *builds* a new LLM (the model-tier factory, the injection
    # classifier's fast-tier resolution) reads these to rewrap its output, so a
    # call cannot escape the recording just because it went through a second
    # client. Without them the log would look complete while silently missing
    # exactly the calls that are hardest to reproduce.
    @property
    def wrapped_llm(self) -> Any:
        return object.__getattribute__(self, "_llm")

    @property
    def run_log_recorder(self) -> RunLogRecorder | None:
        return object.__getattribute__(self, "_recorder")

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
        recorder = object.__getattribute__(self, "_recorder")
        inner = object.__getattribute__(self, "_llm")
        if recorder is None:
            return inner.chat(
                messages=messages,
                tools=tools,
                on_token=on_token,
                response_format=response_format,
                predictive_executor=predictive_executor,
                tool_choice=tool_choice,
                strict_tool_choice=strict_tool_choice,
                timeout_seconds=timeout_seconds,
                request_max_retries=request_max_retries,
            )
        started = time.monotonic()
        try:
            response = inner.chat(
                messages=messages,
                tools=tools,
                on_token=on_token,
                response_format=response_format,
                predictive_executor=predictive_executor,
                tool_choice=tool_choice,
                strict_tool_choice=strict_tool_choice,
                timeout_seconds=timeout_seconds,
                request_max_retries=request_max_retries,
            )
        except BaseException as exc:  # noqa: BLE001 - record, then re-raise
            recorder.record_llm_call(
                messages=messages,
                tools=tools,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=(time.monotonic() - started) * 1000,
                caller=getattr(inner, "caller", None),
                response_format=response_format,
                tool_choice=tool_choice,
                strict_tool_choice=strict_tool_choice,
                streamed=on_token is not None,
            )
            raise
        recorder.record_llm_call(
            messages=messages,
            tools=tools,
            response=response,
            duration_ms=(time.monotonic() - started) * 1000,
            caller=getattr(inner, "caller", None),
            response_format=response_format,
            tool_choice=tool_choice,
            strict_tool_choice=strict_tool_choice,
            streamed=on_token is not None,
        )
        return response


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Exchange:
    """One recorded LLM call: the request fingerprint and the full response."""

    index: int
    seq: int
    caller: str
    phase: str | None
    digests: tuple[str, ...]
    message_count: int
    tools_digest: str | None
    tools_count: int
    response_format: Any
    tool_choice: Any
    strict_tool_choice: bool
    response: Mapping[str, Any] | None
    error: str | None
    duration_ms: float
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def tool_calls(self) -> list[Mapping[str, Any]]:
        if not self.response:
            return []
        return list(self.response.get("tool_calls") or [])

    def matches(
        self,
        messages: Sequence[Any] | None,
        tools: Sequence[Any] | None,
        *,
        response_format: Mapping[str, Any] | None = None,
        tool_choice: Any = None,
        strict_tool_choice: bool = False,
    ) -> bool:
        return (
            self.digests == tuple(digest_messages(messages))
            and self.tools_digest == digest_tools(tools)
            and (
                self.response_format == bool(response_format)
                if isinstance(self.response_format, bool)
                else _canonical(self.response_format) == _canonical(response_format)
            )
            and _canonical(self.tool_choice) == _canonical(tool_choice)
            and self.strict_tool_choice == bool(strict_tool_choice)
        )

    def request_matches(self, messages: Sequence[Any] | None) -> bool:
        return self.digests == tuple(digest_messages(messages))

    def locate(self, messages: Sequence[Any] | None) -> int | None:
        """Index of the first message that differs from the recording."""
        actual = digest_messages(messages)
        for position, (expected, got) in enumerate(zip(self.digests, actual)):
            if expected != got:
                return position
        if len(actual) != len(self.digests):
            return min(len(actual), len(self.digests))
        return None


@dataclass(frozen=True)
class ToolObservation:
    """One recorded tool call, as the model saw it."""

    index: int
    seq: int
    round_index: int | None
    name: str
    tool_call_id: str
    arguments: Mapping[str, Any]
    content: str
    status: str
    mutation: bool
    verification: bool
    blocked: bool
    duration_ms: float
    subagent: str
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class RunLog:
    """A parsed run log."""

    path: Path
    schema: int
    run_start: Mapping[str, Any]
    exchanges: list[Exchange]
    tool_results: list[ToolObservation]
    controls: list[Mapping[str, Any]]
    run_end: Mapping[str, Any] | None
    events: list[Mapping[str, Any]]
    partial_tail: bool = False
    #: run_start events for nested runs (sub-agents), in order.
    nested_starts: list[Mapping[str, Any]] = field(default_factory=list)

    # -------------------------------------------------------------- metadata
    @property
    def prompt(self) -> str:
        return str(self.run_start.get("prompt") or "")

    @property
    def model(self) -> str:
        return str(self.run_start.get("model") or "")

    @property
    def provider(self) -> str:
        return str(self.run_start.get("provider") or "unknown")

    @property
    def tool_dialect(self) -> str | None:
        value = self.run_start.get("tool_dialect")
        return str(value) if value else None

    @property
    def session_id(self) -> str | None:
        value = self.run_start.get("session_id")
        return str(value) if value else None

    @property
    def cwd(self) -> str | None:
        value = self.run_start.get("cwd")
        return str(value) if value else None

    @property
    def tool_names(self) -> list[str]:
        return [str(name) for name in self.run_start.get("tools") or []]

    @property
    def flags(self) -> dict[str, Any]:
        return dict(self.run_start.get("flags") or {})

    @property
    def answer(self) -> str:
        return str((self.run_end or {}).get("answer") or "")

    @property
    def status(self) -> str:
        return str((self.run_end or {}).get("status") or "unterminated")

    @property
    def complete(self) -> bool:
        """True when the run reached its terminal event."""
        return self.run_end is not None and not self.partial_tail

    def exchange_for(self, messages: Sequence[Any] | None, tools: Sequence[Any] | None, start: int) -> int | None:
        """Index of the first exchange at/after ``start`` whose request matches."""
        for index in range(max(0, start), len(self.exchanges)):
            if self.exchanges[index].matches(messages, tools):
                return index
        return None

    def window_bounds(self) -> list[tuple[int, int]]:
        """Event-index spans of the top-level runs in this file, in order.

        A REPL session appends one ``run_start``/``run_end`` pair per turn, so a
        single file can hold several independently replayable runs. Nested
        ``run_start`` events (sub-agents) stay inside their parent's span.
        """
        starts = [index for index, event in enumerate(self.events) if event.get("type") == RUN_START and not event.get("nested")]
        bounds: list[tuple[int, int]] = []
        for position, start in enumerate(starts):
            end = starts[position + 1] if position + 1 < len(starts) else len(self.events)
            bounds.append((start, end))
        return bounds

    def window(self, index: int = 0) -> "RunLog":
        """A view restricted to one top-level run (default: the first)."""
        bounds = self.window_bounds()
        if not bounds:
            raise RunLogError(f"{self.path}: no top-level run_start event")
        if index not in range(-len(bounds), len(bounds)):
            raise RunLogError(f"{self.path}: run index {index} out of range (0..{len(bounds) - 1})")
        start, end = bounds[index]
        events = self.events[start:end]
        # A torn line belongs to the final top-level window.  Do not mark an
        # earlier, already-terminated turn incomplete merely because a later
        # turn was interrupted while being appended.
        window_partial = self.partial_tail and not any(event.get("type") == RUN_END for event in events)
        return _build(self.path, events, partial_tail=window_partial)

    @property
    def run_count(self) -> int:
        return max(1, len(self.window_bounds()))

    @property
    def multi_agent(self) -> bool:
        """True when the recording contains delegation.

        Used to pick the replay entry point automatically: re-driving a
        delegated run through a single agent cannot reproduce it, and guessing
        wrong would blame the tools for a wiring difference.
        """
        if str(self.flags.get("execution_mode") or "") == "multi":
            return True
        return bool(self.nested_starts) or any(observation.subagent not in {"", "main"} for observation in self.tool_results)

    @staticmethod
    def load(path: str | Path) -> "RunLog":
        return _parse(Path(path).expanduser())


def _parse(path: Path) -> RunLog:
    if not path.exists():
        raise RunLogError(f"run log not found: {path}")
    events: list[Mapping[str, Any]] = []
    partial_tail = False
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")
    for position, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if position == len(lines) - 1 or not any(rest.strip() for rest in lines[position + 1 :]):
                # An interrupted write leaves one torn line at the tail; the
                # prefix stays replayable, which is the whole point of append-only.
                partial_tail = True
                break
            raise RunLogError(f"{path}:{position + 1}: malformed run log line") from None
        if not isinstance(record, Mapping):
            raise RunLogError(f"{path}:{position + 1}: run log line is not an object")
        events.append(record)
    return _build(path, events, partial_tail=partial_tail)


def _build(path: Path, events: Sequence[Mapping[str, Any]], *, partial_tail: bool) -> RunLog:
    if not events:
        raise RunLogError(f"run log is empty: {path}")

    starts = [event for event in events if event.get("type") == RUN_START]
    if not starts:
        raise RunLogError(f"{path}: no run_start event")
    schema = int(starts[0].get("schema") or 0)
    if schema > SCHEMA_VERSION:
        raise RunLogError(f"{path}: run log schema {schema} is newer than {SCHEMA_VERSION}")
    if schema < 1:
        raise RunLogError(f"{path}: unsupported run log schema {schema}")

    exchanges: list[Exchange] = []
    tool_results: list[ToolObservation] = []
    controls: list[Mapping[str, Any]] = []
    run_end: Mapping[str, Any] | None = None
    for event in events:
        kind = event.get("type")
        if kind == LLM_CALL:
            request = event.get("request") or {}
            exchanges.append(
                Exchange(
                    index=len(exchanges),
                    seq=int(event.get("seq", len(exchanges))),
                    caller=str(event.get("caller") or "unknown"),
                    phase=event.get("phase"),
                    digests=tuple(str(item) for item in request.get("digests") or ()),
                    message_count=int(request.get("messages") or 0),
                    tools_digest=request.get("tools"),
                    tools_count=int(request.get("tools_count") or 0),
                    response_format=request.get("response_format"),
                    tool_choice=request.get("tool_choice"),
                    strict_tool_choice=bool(request.get("strict_tool_choice")),
                    response=event.get("response"),
                    error=event.get("error"),
                    duration_ms=float(event.get("duration_ms") or 0.0),
                    raw=event,
                )
            )
        elif kind == TOOL_RESULT:
            tool_results.append(
                ToolObservation(
                    index=len(tool_results),
                    seq=int(event.get("seq", len(tool_results))),
                    round_index=event.get("round"),
                    name=str(event.get("name") or "unknown"),
                    tool_call_id=str(event.get("tool_call_id") or ""),
                    arguments=dict(event.get("arguments") or {}),
                    content=str(event.get("content") or ""),
                    status=str(event.get("status") or "unknown"),
                    mutation=bool(event.get("mutation")),
                    verification=bool(event.get("verification")),
                    blocked=bool(event.get("blocked")),
                    duration_ms=float(event.get("duration_ms") or 0.0),
                    subagent=str(event.get("subagent") or "main"),
                    raw=event,
                )
            )
        elif kind == CONTROL:
            controls.append(event)
        elif kind == RUN_END and not event.get("nested") and run_end is None:
            run_end = event

    return RunLog(
        path=path,
        schema=schema,
        run_start=starts[0],
        exchanges=exchanges,
        tool_results=tool_results,
        controls=controls,
        run_end=run_end,
        events=list(events),
        partial_tail=partial_tail,
        nested_starts=[event for event in starts if event.get("nested")],
    )


def iter_events(path: str | Path) -> Iterable[Mapping[str, Any]]:
    """Stream events from a (possibly very large) run log without parsing it all."""
    with Path(path).expanduser().open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                return
