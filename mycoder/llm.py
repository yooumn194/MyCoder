"""LLM provider layer - thin wrapper over OpenAI-compatible APIs.

Since most providers (DeepSeek, Qwen, Kimi, GLM, Ollama, etc.) expose an
OpenAI-compatible endpoint, we just use the openai SDK directly.  Switch
provider by changing OPENAI_BASE_URL + OPENAI_API_KEY. That's it.

For providers that are NOT OpenAI-compatible (AWS Bedrock, Google Vertex,
etc.), use the LiteLLM backend which routes to 100+ providers through a
single unified interface. Set MYCODER_PROVIDER=litellm.
"""

import copy
import functools
import json
import os
import re
import time
from dataclasses import dataclass, field

from openai import OpenAI, APIError, BadRequestError, RateLimitError, APITimeoutError, APIConnectionError

from .observability.trace import LLMTracer, estimate_tokens
from .sandbox.logger import get_logger
from .tool_protocol import ToolProtocolAdapter

logger = get_logger("mycoder.llm")


class ToolChoiceCapabilityError(RuntimeError):
    """The provider cannot honor a requirement-critical tool choice."""


def _request_timeout_seconds() -> float:
    """Bound provider calls so a broken stream cannot occupy a worker forever."""
    try:
        value = float(os.getenv("MYCODER_LLM_TIMEOUT_SECONDS", "180"))
    except ValueError:
        value = 180.0
    return min(600.0, max(10.0, value))


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    parse_error: str | None = None


@dataclass
class LLMResponse:
    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0  # prompt tokens served from the provider's prefix cache
    reasoning_tokens: int = 0  # subset of completion tokens used for reasoning

    @property
    def message(self) -> dict:
        """Convert to OpenAI message format for appending to history."""
        # A thinking provider can spend its whole completion in the private
        # reasoning channel and return neither visible content nor tool calls.
        # Persisting that as ``content=None`` creates an invalid assistant
        # message and poisons the next OpenAI-compatible request.  Keep a small
        # visible placeholder in history; never copy private reasoning into the
        # public content field.
        content = self.content or (None if self.tool_calls else "No visible response; continue.")
        msg: dict = {"role": "assistant", "content": content}
        if self.reasoning_content:
            # DeepSeek thinking models require the original reasoning payload
            # on the assistant tool-call message in the next request.
            msg["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]
        return msg


_DSML_INVOKE = re.compile(
    r'<\|\|DSML\|\|\s+invoke\s+name="([^"]+)">(.*?)</\|\|DSML\|\|\s+invoke>',
    re.DOTALL,
)
_DSML_PARAMETER = re.compile(
    r'<\|\|DSML\|\|\s+parameter\s+name="([^"]+)"\s+string="(true|false)">'
    r"(.*?)</\|\|DSML\|\|\s+parameter>",
    re.DOTALL,
)
# A stream can be cut off at any token. These three match the *unterminated*
# shapes so a half-written invocation can still be recovered instead of being
# reported as "the model produced no tool call".
_DSML_INVOKE_OPEN = re.compile(r'<\|\|DSML\|\|\s+invoke\s+name="([^"]+)">', re.DOTALL)
_DSML_INVOKE_CLOSE = re.compile(r"</\|\|DSML\|\|\s+invoke>")
_DSML_PARAMETER_OPEN = re.compile(
    r'<\|\|DSML\|\|\s+parameter\s+name="([^"]+)"\s+string="(true|false)">', re.DOTALL
)


def _dsml_arguments(body: str) -> tuple[dict, bool, bool]:
    """(arguments, malformed_value, saw_any_parameter) for one invoke body."""
    arguments: dict = {}
    malformed = False
    for parameter in _DSML_PARAMETER.finditer(body):
        key, is_string, raw_value = parameter.groups()
        if is_string == "true":
            value = raw_value
        else:
            try:
                value = json.loads(raw_value.strip())
            except json.JSONDecodeError:
                malformed = True
                break
        arguments[key] = value
    return arguments, malformed, bool(_DSML_PARAMETER_OPEN.search(body))


def _dsml_partial_invoke(normalized: str):
    """The trailing invocation whose close tag never arrived, if any.

    DeepSeek Flash emits DSML as plain text, so hitting ``max_tokens`` (or a
    stream error) mid-invocation leaves a well-formed ``invoke`` header whose
    ``</｜｜DSML｜｜ invoke>`` never comes. The old parser required the closing
    tag, so those rounds produced ``tool_calls=0`` and the benchmark reported
    "模型未产生可解析的工具调用" — the model *had* acted, the harness just
    threw it away.
    """
    opens = list(_DSML_INVOKE_OPEN.finditer(normalized))
    if not opens:
        return None
    start = opens[-1]
    body = normalized[start.end():]
    if _DSML_INVOKE_CLOSE.search(body):
        return None  # terminated; the closed-parse above already handled it
    return start.group(1), body


def _parse_dsml_tool_calls(content: str, allowed_names: set[str]) -> list[ToolCall]:
    """Decode DeepSeek's textual DSML fallback into validated tool calls."""
    normalized = content.replace("｜", "|")
    parsed: list[ToolCall] = []
    for index, invoke in enumerate(_DSML_INVOKE.finditer(normalized)):
        name, body = invoke.groups()
        if name not in allowed_names:
            continue
        arguments, malformed, _ = _dsml_arguments(body)
        if not malformed:
            parsed.append(ToolCall(id=f"dsml_call_{index}", name=name, arguments=arguments))

    # Flush a truncated tail invocation. Emitting it — even with a
    # ``parse_error`` when a parameter itself was cut in half — turns a silent
    # "no tool call" into actionable feedback: the tool layer answers with
    # ``INVALID_TOOL_INPUT`` and the model re-emits the call in full.
    partial = _dsml_partial_invoke(normalized)
    if partial is not None:
        name, body = partial
        if name in allowed_names:
            arguments, malformed, _ = _dsml_arguments(body)
            # An unclosed invocation with no parameter is necessarily a
            # truncated stream.  A genuinely zero-argument call is handled by
            # the closed-invocation parser above and never reaches this tail
            # branch, so requiring at least one parameter here avoids silently
            # treating a header-only response as a successful tool call.
            complete = (
                not malformed
                and bool(_DSML_PARAMETER_OPEN.search(body))
                and len(_DSML_PARAMETER.findall(body))
                == len(_DSML_PARAMETER_OPEN.findall(body))
            )
            if complete:
                parse_error = None
            elif arguments:
                parse_error = (
                    "the DSML invocation was truncated before it completed; "
                    "re-emit the whole call with every parameter"
                )
            else:
                # The stream stopped right after `<|DSML| invoke name="...">`,
                # before the first parameter closed. Returning nothing here is
                # indistinguishable from the model deliberately not calling a
                # tool, and the turn ends with tool_calls=0 — the failure this
                # whole path exists to remove. Emit it with no arguments and
                # let the tool layer answer INVALID_TOOL_INPUT.
                parse_error = (
                    "the DSML invocation was cut off immediately after the tool "
                    "name, so no parameters arrived; re-emit the whole call"
                )
            parsed.append(
                ToolCall(
                    id=f"dsml_call_partial_{len(parsed)}",
                    name=name,
                    arguments=arguments,
                    parse_error=parse_error,
                )
            )
    return parsed


# pricing per million tokens: (input, output)
# sources: openai.com/api/pricing, api-docs.deepseek.com, platform.claude.com,
#          platform.moonshot.ai, alibabacloud.com/help/en/model-studio
_PRICING = {
    # OpenAI - current flagships
    "gpt-5.5": (5, 30),
    "gpt-5.4": (2.5, 15),
    "gpt-5.4-mini": (0.75, 4.5),
    "gpt-5.4-nano": (0.2, 1.25),
    "o4-mini": (1.1, 4.4),
    # OpenAI - previous gen (still widely used)
    "gpt-4.1": (2, 8),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.1, 0.4),
    "gpt-4o": (2.5, 10),
    "gpt-4o-mini": (0.15, 0.6),
    # DeepSeek
    "deepseek-flash": (0.27, 1.10),
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    # Anthropic Claude
    "claude-opus-4-6": (5, 25),
    "claude-sonnet-4-6": (3, 15),
    "claude-haiku-4-5": (1, 5),
    # Alibaba Qwen
    "qwen3-max": (0.78, 3.9),
    "qwen3-plus": (0.26, 0.78),
    "qwen-max": (0.78, 3.9),
    # Moonshot Kimi
    "kimi-k2.5": (0.6, 3),
}


def _messages_text(messages: list) -> str:
    """Flatten an OpenAI messages list into plain text for token estimation."""
    parts = []
    for m in messages or []:
        if isinstance(m, dict):
            content = m.get("content")
            if content:
                parts.append(str(content))
            reasoning_content = m.get("reasoning_content")
            if reasoning_content:
                parts.append(str(reasoning_content))
    return "\n".join(parts)


def _extract_cached_tokens(usage) -> int:
    """Provider prompt-cache hit tokens.

    OpenAI reports them in usage.prompt_tokens_details.cached_tokens; DeepSeek
    in usage.prompt_cache_hit_tokens. 0 when the provider doesn't report it.
    """
    try:
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
            if cached:
                return int(cached)
    except Exception:  # noqa: BLE001 - caching stats are best-effort
        pass
    try:
        return int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _value(obj, *names, default=0):
    """Read snake_case/camelCase fields from SDK objects or plain dicts."""
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if value is not None:
            return value
    return default


def _extract_reasoning_tokens(usage) -> int:
    """Read OpenRouter/OpenAI reasoning usage from the final stream chunk."""
    details = _value(
        usage,
        "completion_tokens_details",
        "completionTokensDetails",
        default=None,
    )
    if details is None:
        return 0
    try:
        return int(_value(details, "reasoning_tokens", "reasoningTokens") or 0)
    except (TypeError, ValueError):
        return 0


def _extract_reasoning_content(delta) -> str:
    """Read provider-specific streamed reasoning text without exposing it."""
    value = _value(delta, "reasoning_content", "reasoningContent", default=None)
    if value is None:
        extra = getattr(delta, "model_extra", None) or {}
        value = _value(extra, "reasoning_content", "reasoningContent", default="")
    return str(value or "")


def _context_value(name: str):
    """Read a structlog contextvar (e.g. session_id) if one is bound."""
    try:
        from structlog.contextvars import get_contextvars

        return get_contextvars().get(name)
    except Exception:  # noqa: BLE001 - contextvars are best-effort
        return None


def _projected_request_tokens(llm, messages: list[dict], tools: list[dict] | None) -> int:
    """Conservative prompt+completion estimate used for hard-budget admission."""
    prompt_text = _messages_text(messages)
    if tools:
        prompt_text += "\n" + json.dumps(tools, ensure_ascii=False, default=str)
    prompt_tokens = estimate_tokens(prompt_text)
    if prompt_tokens is None:
        prompt_tokens = max(1, len(prompt_text) // 3)
    try:
        output_cap = int(getattr(llm, "extra", {}).get("max_tokens", 4096))
    except (AttributeError, TypeError, ValueError):
        output_cap = 4096
    # A provider response rarely consumes its full output cap for a tool
    # decision, but reserving the full 4k on every call starves downstream
    # verifier subagents under a shared session budget.  Keep a bounded safety
    # margin for admission and let the authoritative post-call usage check
    # enforce the hard cap.  Deployments with unusually verbose tool calls can
    # raise this without changing model configuration.
    try:
        projection_cap = int(os.getenv("MYCODER_BUDGET_PROJECTION_OUTPUT_TOKENS", "1024"))
    except (TypeError, ValueError):
        projection_cap = 1024
    projection_cap = min(8192, max(256, projection_cap))
    return max(1, int(prompt_tokens)) + min(projection_cap, max(256, output_cap))


def _traced(method):
    """Wrap LLM.chat / LiteLLM.chat with an optional LLMTracer.

    Zero-cost when the instance has no tracer (`_tracer` is None). Token usage
    is read from the returned LLMResponse; when the provider omitted usage, it
    is estimated via tiktoken (optional) and recorded as -1 if tiktoken is
    missing. Exceptions are recorded (error/timeout) and re-raised.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        tracer = getattr(self, "_tracer", None)
        if tracer is None:
            return method(self, *args, **kwargs)
        messages = args[0] if args else kwargs.get("messages", [])
        tools = args[1] if len(args) > 1 else kwargs.get("tools")
        session_id = _context_value("session_id") or "unknown"
        caller = getattr(self, "caller", None) or "llm"
        phase = _context_value("agent_phase")
        protocol = getattr(self, "tool_protocol", None)
        requested_tool_choice = kwargs.get("tool_choice")
        try:
            wire_tool_names = [
                str(item.get("function", {}).get("name"))
                for item in (protocol.tools_to_wire(tools or []) if protocol else (tools or []))
                if isinstance(item, dict)
            ]
        except Exception:  # noqa: BLE001 - tracing must never affect a provider call
            wire_tool_names = []
        try:
            wire_tool_choice = (
                protocol.tool_choice_to_wire(requested_tool_choice)
                if protocol is not None and requested_tool_choice is not None
                else requested_tool_choice
            )
        except Exception:  # noqa: BLE001
            wire_tool_choice = requested_tool_choice
        fallback_before = int(getattr(self, "tool_choice_fallbacks", 0) or 0)

        # TTFT (time-to-first-token): wrap on_token so the FIRST streamed token
        # is timed from request start — a key "卡顿感知" metric.
        started = 0.0
        ttft_ms: float | None = None
        on_token = kwargs.get("on_token")
        if on_token is not None:
            _first = [True]

            def _ttft_token(tok):
                nonlocal ttft_ms
                if _first[0]:
                    _first[0] = False
                    ttft_ms = (time.monotonic() - started) * 1000
                on_token(tok)

            kwargs["on_token"] = _ttft_token

        with tracer.trace(
            session_id=session_id,
            caller=caller,
            model=getattr(self, "model", "unknown"),
            provider=getattr(self, "provider", "unknown"),
            tool_dialect=getattr(protocol, "dialect", None),
            wire_tool_names=wire_tool_names,
            tool_choice_requested=requested_tool_choice,
            tool_choice_wire=wire_tool_choice,
            strict_tool_choice=bool(kwargs.get("strict_tool_choice", False)),
            phase=phase,
            projected_tokens=_projected_request_tokens(
                self,
                messages,
                tools,
            ),
        ) as ctx:
            # Start TTFT after budget admission, immediately before the actual
            # provider method. The trace context starts slightly earlier, so
            # full duration is guaranteed to include TTFT.
            started = time.monotonic()
            try:
                resp = method(self, *args, **kwargs)
            except Exception:
                ctx["tool_choice_degraded"] = int(
                    getattr(self, "tool_choice_fallbacks", 0) or 0
                ) > fallback_before
                raise
            ctx["tool_choice_degraded"] = int(
                getattr(self, "tool_choice_fallbacks", 0) or 0
            ) > fallback_before
            prompt = int(getattr(resp, "prompt_tokens", 0) or 0)
            completion = int(getattr(resp, "completion_tokens", 0) or 0)
            if prompt == 0 and completion == 0:
                prompt_est = estimate_tokens(_messages_text(messages))
                completion_est = estimate_tokens(getattr(resp, "content", "") or "")
                prompt = prompt_est if prompt_est is not None else -1
                completion = completion_est if completion_est is not None else -1
            ctx["prompt_tokens"] = prompt
            ctx["completion_tokens"] = completion
            ctx["cached_tokens"] = getattr(resp, "cached_tokens", 0) or 0
            ctx["reasoning_tokens"] = getattr(resp, "reasoning_tokens", 0) or 0
            ctx["ttft_ms"] = ttft_ms
        return resp

    return wrapper


class LLM:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        *,
        provider: str = "openai",
        tracer: LLMTracer | None = None,
        caller: str = "llm",
        tool_dialect: str | None = None,
        **kwargs,
    ):
        self.model = model
        self.provider = provider.strip().lower()
        self.api_key = api_key
        self.base_url = base_url
        self.tool_protocol = ToolProtocolAdapter.for_model(model, tool_dialect, self.provider)
        self.tool_dialect = tool_dialect if tool_dialect is not None else os.getenv(
            "MYCODER_TOOL_DIALECT", "auto"
        )
        client_kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": _request_timeout_seconds(),
        }
        if self.provider == "openrouter":
            headers = {}
            if site_url := os.getenv("OPENROUTER_SITE_URL"):
                headers["HTTP-Referer"] = site_url
            if app_name := os.getenv("OPENROUTER_APP_NAME"):
                headers["X-OpenRouter-Title"] = app_name
            if headers:
                client_kwargs["default_headers"] = headers
        self.client = OpenAI(**client_kwargs)
        self.extra = kwargs  # temperature, max_tokens, etc.
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_reasoning_tokens = 0
        # Observability: optional call tracer; None = no tracing (backward compat).
        self._tracer = tracer
        self.caller = caller
        self.tool_choice_fallbacks = 0

    @property
    def estimated_cost(self) -> float | None:
        """Rough cost estimate in USD. Returns None if model not in pricing table."""
        pricing = _PRICING.get(self.model)
        if not pricing:
            return None
        input_rate, output_rate = pricing
        return self.total_prompt_tokens * input_rate / 1_000_000 + self.total_completion_tokens * output_rate / 1_000_000

    @_traced
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
    ) -> LLMResponse:
        """Send messages, stream back response, handle tool calls.

        `response_format` (e.g. {"type": "json_object"}) forces structured
        output — used by the sub-agent envelope repair pass (Phase 4). Note it
        generally cannot be combined with `tools` on the same call.

        `predictive_executor` (optional) is called with a ToolCall the moment
        its streamed arguments parse as complete JSON — so the caller can start
        executing it WHILE the stream is still generating (predictive / stream
        execution, saving one serial round-trip per tool round).
        """
        params: dict = {
            "model": self.model,
            "messages": self.tool_protocol.messages_to_wire(messages),
            "stream": True,
            **self.extra,
        }
        if response_format:
            params["response_format"] = response_format
        if timeout_seconds is not None:
            # A timeout around asyncio.to_thread only abandons the waiter; the
            # synchronous provider stream keeps running and can still consume
            # money and session budget. Put the deadline on the HTTP request.
            params["timeout"] = max(0.1, float(timeout_seconds))
        if tools:
            if strict_tool_choice and tool_choice is not None:
                # DeepSeek's thinking endpoint rejects ``tool_choice`` while
                # thinking is enabled, but supports named choices when
                # thinking is explicitly disabled.  Keep the strict contract
                # intact and let the bounded BadRequest fallback below retry
                # with that provider-compatible request shape.
                deepseek_thinking_compat = (
                    self.provider == "deepseek"
                    and not self.tool_protocol.supports_tool_choice
                )
                if not self.tool_protocol.supports_tool_choice and not deepseek_thinking_compat:
                    raise ToolChoiceCapabilityError(
                        f"provider {self.provider!r} does not support tool_choice for "
                        f"model {self.model!r}"
                    )
                if (
                    isinstance(tool_choice, dict)
                    and not self.tool_protocol.supports_named_tool_choice
                    and not deepseek_thinking_compat
                ):
                    raise ToolChoiceCapabilityError(
                        f"provider {self.provider!r} does not support named tool_choice "
                        f"for model {self.model!r}"
                    )
            params["tools"] = self.tool_protocol.tools_to_wire(tools)
            if tool_choice is not None:
                params["tool_choice"] = self.tool_protocol.tool_choice_to_wire(tool_choice)
        # stream_options is an OpenAI extension; fall back only when the provider
        # rejects the param (400 BadRequest), not on transient errors that
        # _call_with_retry already exhausted - otherwise we'd double the retries
        params["stream_options"] = {"include_usage": True}
        def _start_stream():
            if request_max_retries is None:
                return self._call_with_retry(params)
            return self._call_with_retry(
                params,
                max_retries=max(1, int(request_max_retries)),
            )

        try:
            stream = _start_stream()
        except BadRequestError:
            params.pop("stream_options", None)
            try:
                stream = _start_stream()
            except BadRequestError:
                # Some OpenAI-compatible providers implement tools but reject
                # tool_choice="required". Degrade to auto only after proving
                # the provider cannot honor the stronger contract.
                if "tool_choice" not in params:
                    raise
                # DeepSeek Flash's thinking mode is the exception: the same
                # named choice succeeds when private reasoning is disabled.
                # Retry once with only this request adjusted, preserving the
                # strict mutation contract instead of silently falling back to
                # auto (which can produce another inspection call).
                compatibility_adjusted = False
                if (
                    strict_tool_choice
                    and self.provider == "deepseek"
                    and not self.tool_protocol.supports_tool_choice
                ):
                    compatible = copy.deepcopy(params)
                    extra_body = compatible.get("extra_body")
                    if not isinstance(extra_body, dict):
                        extra_body = {}
                    else:
                        extra_body = copy.deepcopy(extra_body)
                    thinking = extra_body.get("thinking")
                    if not isinstance(thinking, dict) or thinking.get("type") != "disabled":
                        extra_body["thinking"] = {"type": "disabled"}
                    compatible["extra_body"] = extra_body
                    try:
                        stream = self._call_with_retry(compatible)
                    except BadRequestError as exc:
                        raise ToolChoiceCapabilityError(
                            f"provider {self.provider!r} rejected strict tool_choice "
                            f"even with thinking disabled for model {self.model!r}"
                        ) from exc
                    else:
                        # Keep tracing/fallback accounting aligned with the
                        # request actually sent; downstream parsing is
                        # unchanged because tool names remain protocol-adapted.
                        params = compatible
                        self.tool_choice_fallbacks += 1
                        logger.info(
                            "llm.tool_choice_compatibility_adjusted",
                            provider=self.provider,
                            model=self.model,
                            adjustment="deepseek_thinking_disabled",
                        )
                        compatibility_adjusted = True
                if not compatibility_adjusted:
                    if strict_tool_choice:
                        raise ToolChoiceCapabilityError(
                            f"provider {self.provider!r} rejected tool_choice for model "
                            f"{self.model!r}; strict tool contract cannot be satisfied"
                        )
                    self.tool_choice_fallbacks += 1
                    logger.warning(
                        "llm.tool_choice_degraded",
                        provider=self.provider,
                        model=self.model,
                        reason="provider_rejected_tool_choice",
                    )
                    params.pop("tool_choice")
                    stream = _start_stream()

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tc_map: dict[int, dict] = {}  # index -> {id, name, arguments_str}
        predicted_idx: set[int] = set()  # tool-call indices already handed off
        prompt_tok = 0
        completion_tok = 0
        cached_tok = 0
        reasoning_tok = 0

        for chunk in stream:
            # usage info comes in the final chunk
            if chunk.usage:
                # some providers send usage with null fields; coerce to 0 so the
                # running totals below don't blow up on int + None
                prompt_tok = chunk.usage.prompt_tokens or 0
                completion_tok = chunk.usage.completion_tokens or 0
                cached_tok = _extract_cached_tokens(chunk.usage)
                reasoning_tok = _extract_reasoning_tokens(chunk.usage)

            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if reasoning := _extract_reasoning_content(delta):
                reasoning_parts.append(reasoning)

            # accumulate text
            if delta.content:
                content_parts.append(delta.content)
                if on_token:
                    on_token(delta.content)

            # accumulate tool calls across chunks
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tc_map:
                        tc_map[idx] = {"id": "", "name": "", "args": ""}
                    if tc_delta.id:
                        tc_map[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc_map[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc_map[idx]["args"] += tc_delta.function.arguments
                    # Predictive execution: once this call's streamed arguments
                    # parse as complete JSON, hand it to the executor NOW so it
                    # runs while the stream keeps generating (saves one RTT).
                    if (
                        predictive_executor is not None
                        and idx not in predicted_idx
                        and tc_map[idx]["name"]
                        and tc_map[idx]["args"]
                    ):
                        try:
                            args = json.loads(tc_map[idx]["args"])
                        except json.JSONDecodeError:
                            continue  # args still partial; keep accumulating
                        predicted_idx.add(idx)
                        predictive_executor(
                            ToolCall(
                                id=tc_map[idx]["id"],
                                name=self.tool_protocol.from_wire_name(tc_map[idx]["name"]),
                                arguments=self.tool_protocol.from_wire_arguments(
                                    tc_map[idx]["name"], args
                                ),
                            )
                        )

        # parse accumulated tool calls
        parsed: list[ToolCall] = []
        for idx in sorted(tc_map):
            raw = tc_map[idx]
            parse_error = None
            try:
                args = json.loads(raw["args"])
            except (json.JSONDecodeError, KeyError):
                args = {}
                parse_error = "tool arguments were truncated or were not valid JSON"
            parsed.append(
                ToolCall(
                    id=raw["id"],
                    name=self.tool_protocol.from_wire_name(raw["name"]),
                    arguments=self.tool_protocol.from_wire_arguments(raw["name"], args),
                    parse_error=parse_error,
                )
            )

        if not parsed and tools:
            allowed_names = {
                str(tool.get("function", {}).get("name"))
                for tool in (params.get("tools") or [])
                if tool.get("type") == "function" and tool.get("function", {}).get("name")
            }
            # Accept the internal aliases as a backward-compatible textual
            # fallback.  The advertised schema still uses exactly one stable
            # wire name, so this does not add duplicate tools to the request.
            allowed_names.update(
                str(tool.get("function", {}).get("name"))
                for tool in tools
                if tool.get("type") == "function" and tool.get("function", {}).get("name")
            )
            # DeepSeek Flash may emit its textual DSML invocation in the
            # provider-specific reasoning channel while leaving only a short
            # visible answer. Both channels are model output; tool names still
            # pass through the declared-tool allowlist and argument parser.
            wire_calls = _parse_dsml_tool_calls(
                "\n".join(("".join(reasoning_parts), "".join(content_parts))),
                allowed_names,
            )
            parsed = [
                ToolCall(
                    id=call.id,
                    name=self.tool_protocol.from_wire_name(call.name),
                    arguments=self.tool_protocol.from_wire_arguments(call.name, call.arguments),
                    # The tolerant parser flags a truncated parameter so the
                    # agent can send the call back for a retry. Dropping the
                    # flag here would silently downgrade that recovery path to
                    # a generic "missing required argument" error.
                    parse_error=call.parse_error,
                )
                for call in wire_calls
            ]

        self.total_prompt_tokens += prompt_tok
        self.total_completion_tokens += completion_tok
        self.total_reasoning_tokens += reasoning_tok

        return LLMResponse(
            content="".join(content_parts),
            reasoning_content="".join(reasoning_parts),
            tool_calls=parsed,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
            cached_tokens=cached_tok,
            reasoning_tokens=reasoning_tok,
        )

    def _call_with_retry(self, params: dict, max_retries: int = 3):
        """Retry on transient errors with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return self.client.chat.completions.create(**params)
            except (RateLimitError, APITimeoutError, APIConnectionError):
                if attempt == max_retries - 1:
                    raise
                wait = 2**attempt
                time.sleep(wait)
            except APIError as e:
                # retry 5xx server errors but not 4xx; base APIError has no status_code so read it defensively
                status_code = getattr(e, "status_code", None)
                if status_code and status_code >= 500 and attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise


class LiteLLM(LLM):
    """LLM backend via LiteLLM, supporting 100+ providers.

    Use this when your target provider is NOT OpenAI-compatible
    (AWS Bedrock, Google Vertex, Cohere, etc.) or when you want
    a single interface to switch between any provider by changing
    the model string.

    Set MYCODER_PROVIDER=litellm and use LiteLLM model strings
    like ``anthropic/claude-3-haiku``, ``bedrock/anthropic.claude-v2``,
    ``vertex_ai/gemini-pro``, etc.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        provider: str = "litellm",
        tracer: LLMTracer | None = None,
        caller: str = "llm",
        tool_dialect: str | None = None,
        **kwargs,
    ):
        # skip LLM.__init__ which creates an OpenAI client
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.provider = provider.strip().lower()
        self.tool_protocol = ToolProtocolAdapter.for_model(model, tool_dialect, self.provider)
        self.tool_dialect = tool_dialect if tool_dialect is not None else os.getenv(
            "MYCODER_TOOL_DIALECT", "auto"
        )
        self.extra = kwargs
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_reasoning_tokens = 0
        # Observability: optional call tracer; None = no tracing (backward compat).
        self._tracer = tracer
        self.caller = caller
        self.tool_choice_fallbacks = 0

    @_traced
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
    ) -> LLMResponse:
        """Send messages via litellm, stream back response, handle tool calls.

        predictive_executor is accepted for interface parity with LLM.chat but
        not used (predictive execution is implemented on the LLM backend)."""
        params: dict = {
            "model": self.model,
            "messages": self.tool_protocol.messages_to_wire(messages),
            "stream": True,
            **self.extra,
        }
        if response_format:
            params["response_format"] = response_format
        if timeout_seconds is not None:
            params["timeout"] = max(0.1, float(timeout_seconds))
        if tools:
            if strict_tool_choice and tool_choice is not None:
                if not self.tool_protocol.supports_tool_choice:
                    raise ToolChoiceCapabilityError(
                        f"provider {self.provider!r} does not support tool_choice for "
                        f"model {self.model!r}"
                    )
                if (
                    isinstance(tool_choice, dict)
                    and not self.tool_protocol.supports_named_tool_choice
                ):
                    raise ToolChoiceCapabilityError(
                        f"provider {self.provider!r} does not support named tool_choice "
                        f"for model {self.model!r}"
                    )
            params["tools"] = self.tool_protocol.tools_to_wire(tools)
            if tool_choice is not None:
                params["tool_choice"] = self.tool_protocol.tool_choice_to_wire(tool_choice)
        if strict_tool_choice:
            # LiteLLM otherwise drops unsupported fields and makes a
            # requirement-critical request look successful.
            params["drop_params"] = False

        # ask for usage stats in the final chunk; litellm drops this for providers
        # that don't support it (drop_params), so it's safe to always request
        params["stream_options"] = {"include_usage": True}
        stream = (
            self._call_with_retry(params)
            if request_max_retries is None
            else self._call_with_retry(
                params,
                max_retries=max(1, int(request_max_retries)),
            )
        )

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tc_map: dict[int, dict] = {}
        prompt_tok = 0
        completion_tok = 0
        reasoning_tok = 0

        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage:
                prompt_tok = getattr(usage, "prompt_tokens", 0) or 0
                completion_tok = getattr(usage, "completion_tokens", 0) or 0
                reasoning_tok = _extract_reasoning_tokens(usage)

            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta

            if reasoning := _extract_reasoning_content(delta):
                reasoning_parts.append(reasoning)

            if getattr(delta, "content", None):
                content_parts.append(delta.content)
                if on_token:
                    on_token(delta.content)

            if getattr(delta, "tool_calls", None):
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tc_map:
                        tc_map[idx] = {"id": "", "name": "", "args": ""}
                    if tc_delta.id:
                        tc_map[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc_map[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc_map[idx]["args"] += tc_delta.function.arguments

        parsed: list[ToolCall] = []
        for idx in sorted(tc_map):
            raw = tc_map[idx]
            parse_error = None
            try:
                args = json.loads(raw["args"])
            except (json.JSONDecodeError, KeyError):
                args = {}
                parse_error = "tool arguments were truncated or were not valid JSON"
            parsed.append(
                ToolCall(
                    id=raw["id"],
                    name=self.tool_protocol.from_wire_name(raw["name"]),
                    arguments=self.tool_protocol.from_wire_arguments(raw["name"], args),
                    parse_error=parse_error,
                )
            )

        self.total_prompt_tokens += prompt_tok
        self.total_completion_tokens += completion_tok
        self.total_reasoning_tokens += reasoning_tok

        return LLMResponse(
            content="".join(content_parts),
            reasoning_content="".join(reasoning_parts),
            tool_calls=parsed,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
            reasoning_tokens=reasoning_tok,
        )

    def _call_with_retry(self, params: dict, max_retries: int = 3):
        """Retry on transient errors with exponential backoff via litellm."""
        import litellm

        params.setdefault("drop_params", True)
        if self.api_key:
            params["api_key"] = self.api_key
        if self.base_url:
            params["api_base"] = self.base_url

        for attempt in range(max_retries):
            try:
                return litellm.completion(**params)
            except Exception as e:
                err = str(e).lower()
                is_transient = any(kw in err for kw in ["rate_limit", "timeout", "connection", "502", "503", "529"])
                is_server = any(kw in err for kw in ["500", "502", "503", "504"])
                if (is_transient or is_server) and attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise
