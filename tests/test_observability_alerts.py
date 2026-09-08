"""P2 SLO alerts (observability/alerts.py + LLMTracer wiring)."""

import pytest

from mycoder.observability.alerts import AlertManager, AlertRule, default_rules
from mycoder.observability.budget import TokenBudgetExceeded, TokenBudgetGuard
from mycoder.observability.trace import LLMTracer


def test_rule_breach_and_operators():
    r = AlertRule("sla", "success_rate", 0.9, op="<", severity="critical")
    assert r.breached(0.8) is True
    assert r.breached(0.95) is False
    assert AlertRule("b", "v", 5, op=">=").breached(5.0) is True
    assert AlertRule("b", "v", 5, op="<=").breached(5.0) is True


def test_manager_fires_and_debounces_by_cooldown():
    m = AlertManager(rules=[AlertRule("low_success_rate", "success_rate", 0.9, op="<")])
    assert len(m.evaluate("s1", {"success_rate": 0.5})) == 1
    # cooldown: an immediate re-evaluation does not re-fire for the same session
    assert m.evaluate("s1", {"success_rate": 0.4}) == []
    # a different session can still fire
    assert len(m.evaluate("s2", {"success_rate": 0.4})) == 1
    m.reset()
    assert len(m.evaluate("s1", {"success_rate": 0.4})) == 1  # reset clears the debounce


def test_manager_no_breach():
    m = AlertManager(rules=default_rules())
    assert (
        m.evaluate("s1", {"success_rate": 1.0, "p95_duration_ms": 100, "budget_ratio": 0.1})
        == []
    )


def test_tracer_fires_alert_after_failed_calls():
    m = AlertManager(rules=[AlertRule("low_success_rate", "success_rate", 0.9, op="<")])
    tracer = LLMTracer()
    tracer.attach_alert_manager(m)

    with tracer.trace("s1", "t", "m"):
        pass
    try:
        with tracer.trace("s1", "t", "m"):
            raise TimeoutError("boom")
    except TimeoutError:
        pass
    # 1 success + 1 error -> success_rate 0.5 < 0.9 -> alert debounced/fired
    assert ("s1", "low_success_rate") in m._last_fired  # noqa: SLF001


def test_ttft_recorded_from_first_on_token():
    """TTFT: the first streamed token is timed from request start and lands in
    the session summary (avg_ttft_ms / p95_ttft_ms)."""
    import time

    from mycoder.llm import LLMResponse, _traced

    tracer = LLMTracer()
    received: list[str] = []

    class _Fake:
        def __init__(self):
            self._tracer = tracer
            self.caller = "llm"
            self.model = "fake-model"

        @_traced
        def chat(self, messages, on_token=None):
            time.sleep(0.01)  # simulate network latency before first token
            on_token("你")
            on_token("好")
            return LLMResponse(content="你好", prompt_tokens=5, completion_tokens=2)

    _Fake().chat([{"role": "user", "content": "hi"}], on_token=received.append)

    assert received == ["你", "好"]  # original callback still streamed through
    s = tracer.get_session_summary("unknown")  # no session contextvar -> "unknown"
    assert s["total_calls"] == 1
    assert s["avg_ttft_ms"] > 0.0  # first-token latency was captured
    assert s["p95_ttft_ms"] == s["avg_ttft_ms"]  # single call
    assert s["avg_duration_ms"] >= s["avg_ttft_ms"]  # ttft <= full duration


def test_reasoning_tokens_are_traced_but_not_double_counted():
    from mycoder.llm import LLMResponse, _traced

    tracer = LLMTracer()

    class _Fake:
        _tracer = tracer
        caller = "llm"
        model = "reasoning-model"

        @_traced
        def chat(self, messages):
            return LLMResponse(
                content="ok",
                prompt_tokens=5,
                completion_tokens=8,
                reasoning_tokens=3,
            )

    _Fake().chat([{"role": "user", "content": "hi"}])
    summary = tracer.get_session_summary("unknown")

    assert summary["reasoning_tokens"] == 3
    assert summary["total_tokens"] == 13


def test_session_budget_is_enforced_after_each_llm_call():
    tracer = LLMTracer()
    guard = TokenBudgetGuard(max_tokens_per_session=10, tracer=tracer)
    tracer.register_budget_guard("budgeted", guard)

    with tracer.trace("budgeted", "api", "model") as ctx:
        ctx["prompt_tokens"] = 6
        ctx["completion_tokens"] = 3

    with pytest.raises(TokenBudgetExceeded) as exc_info:
        with tracer.trace("budgeted", "api", "model") as ctx:
            ctx["prompt_tokens"] = 1
            ctx["completion_tokens"] = 1

    assert exc_info.value.used_tokens == 11
    assert tracer.get_session_summary("budgeted")["total_calls"] == 2

    # Once exhausted, the next provider call is rejected before entering it.
    entered = False
    with pytest.raises(TokenBudgetExceeded):
        with tracer.trace("budgeted", "api", "model"):
            entered = True
    assert entered is False
    assert tracer.get_session_summary("budgeted")["total_calls"] == 2

    tracer.unregister_budget_guard("budgeted", guard)
