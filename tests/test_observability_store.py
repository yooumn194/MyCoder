"""Process-independent observability state: trace/cost/rate-limit/alerts."""

from concurrent.futures import ThreadPoolExecutor

from mycoder.observability.alerts import AlertManager, AlertRule
from mycoder.observability.ratelimit import RateLimiter
from mycoder.observability.store import (
    RedisObservabilityStore,
    SQLiteObservabilityStore,
)
from mycoder.observability.trace import LLMTracer
from mycoder.observability.tool_trace import ToolTracer


class _Log:
    def __init__(self):
        self.events: list[dict] = []

    def error(self, _event, **fields):
        self.events.append(fields)


def test_trace_and_cost_survive_tracer_reconstruction(tmp_path):
    path = tmp_path / "api_state.db"
    writer = LLMTracer(store=SQLiteObservabilityStore(path))
    with writer.trace("session-a", "api", "model-a") as ctx:
        ctx["prompt_tokens"] = 1000
        ctx["completion_tokens"] = 500
        ctx["reasoning_tokens"] = 100

    # A new store + tracer simulates another worker or a process restart.
    reader = LLMTracer(store=SQLiteObservabilityStore(path))
    summary = reader.get_session_summary("session-a")
    cost = reader.get_cost_estimate(
        "session-a",
        price_per_1k={"model-a": {"input": 0.001, "output": 0.002}},
    )

    assert summary["total_calls"] == 1
    assert summary["total_tokens"] == 1500
    assert summary["reasoning_tokens"] == 100
    assert cost["total_cost_usd"] == 0.002
    assert reader.list_sessions() == ["session-a"]


def test_tool_trace_survives_reconstruction_and_stays_out_of_llm_metrics(tmp_path):
    path = tmp_path / "api_state.db"
    writer = ToolTracer(store=SQLiteObservabilityStore(path))
    writer.record(
        "session-tools",
        "call-1",
        "edit_file",
        {"file_path": "app.py", "api_key": "sk-secret"},
        "Edited app.py",
        status="success",
        duration_ms=12.5,
        retry_count=1,
        mutation=True,
        subagent_name="implementer",
    )
    writer.record_requirement_feedback(
        "session-tools",
        "mutation required",
        attempt=1,
        phase="mutate",
        subagent_name="implementer",
    )

    reader = ToolTracer(store=SQLiteObservabilityStore(path))
    summary = reader.get_session_summary("session-tools")
    traces = reader.list_traces("session-tools")

    assert summary["calls"] == 1
    assert summary["mutations"] == 1
    assert summary["retries"] == 1
    assert summary["requirement_misses"] == 1
    assert traces[0]["arguments"]["api_key"] == "[REDACTED]"
    assert "sk-secret" not in str(traces)
    assert LLMTracer(store=SQLiteObservabilityStore(path)).get_session_summary("session-tools")["total_calls"] == 0


def test_monitor_report_can_aggregate_only_tenant_allowed_traces(tmp_path):
    from mycoder.observability.report import build_monitor_report

    tracer = LLMTracer(store=SQLiteObservabilityStore(tmp_path / "api_state.db"))
    with tracer.trace("tenant-a-session", "api", "model") as ctx:
        ctx["prompt_tokens"] = 10
    with tracer.trace("tenant-b-session", "api", "model") as ctx:
        ctx["prompt_tokens"] = 20

    report = build_monitor_report(
        tracer,
        price_per_1k={"default": 0.001},
        trace_session_ids=["tenant-a-session"],
    )

    assert report["llm"]["calls"] == 1
    assert report["llm"]["tokens"] == 10
    assert report["llm"]["sessions"] == 1
    assert set(report["per_session"]) == {"tenant-a-session"}


def test_monitor_report_aggregates_tool_only_sessions(tmp_path):
    from mycoder.observability.report import build_monitor_report

    store = SQLiteObservabilityStore(tmp_path / "api_state.db")
    llm_tracer = LLMTracer(store=store)
    tool_tracer = ToolTracer(store=store)
    tool_tracer.record(
        "tool-only",
        "call-1",
        "edit_file",
        {},
        "Edited app.py",
        status="success",
        mutation=True,
    )

    report = build_monitor_report(llm_tracer, tool_tracer=tool_tracer)

    assert report["llm"]["calls"] == 0
    assert report["tools"]["calls"] == 1
    assert report["tools"]["mutations"] == 1
    assert report["per_session"]["tool-only"]["tools"]["calls"] == 1


def test_rate_limit_is_shared_across_store_instances(tmp_path):
    path = tmp_path / "api_state.db"
    first = RateLimiter(1, store=SQLiteObservabilityStore(path))
    second = RateLimiter(1, store=SQLiteObservabilityStore(path))

    assert first.allow("tenant:client") is True
    assert second.allow("tenant:client") is False
    assert second.allow("other-tenant:client") is True


def test_rate_limit_admission_is_atomic_across_threads(tmp_path):
    path = tmp_path / "api_state.db"
    limiters = [RateLimiter(3, store=SQLiteObservabilityStore(path)) for _ in range(12)]
    with ThreadPoolExecutor(max_workers=12) as pool:
        decisions = list(pool.map(lambda limiter: limiter.allow("shared"), limiters))

    assert sum(decisions) == 3


def test_alert_cooldown_and_history_survive_manager_reconstruction(tmp_path):
    path = tmp_path / "api_state.db"
    rule = AlertRule("low_success", "success_rate", 0.9, op="<", cooldown_seconds=60)
    first_log = _Log()
    second_log = _Log()
    first = AlertManager(rules=[rule], log=first_log, store=SQLiteObservabilityStore(path))
    second = AlertManager(rules=[rule], log=second_log, store=SQLiteObservabilityStore(path))

    assert len(first.evaluate("session-a", {"success_rate": 0.5})) == 1
    assert second.evaluate("session-a", {"success_rate": 0.4}) == []
    assert len(second.evaluate("session-b", {"success_rate": 0.4})) == 1
    history = second.list_alerts()
    assert len(history) == 2
    assert {item["session_id"] for item in history} == {"session-a", "session-b"}

    second.reset()
    assert len(first.evaluate("session-a", {"success_rate": 0.4})) == 1


def test_sqlite_retention_prunes_expired_traces(tmp_path):
    import time

    store = SQLiteObservabilityStore(tmp_path / "api_state.db", ttl_seconds=60)
    base = {
        "session_id": "session-a",
        "caller": "api",
        "model": "model",
        "prompt_tokens": 1,
        "completion_tokens": 0,
        "total_tokens": 1,
        "duration_ms": 1.0,
        "status": "success",
    }
    store.append_trace({**base, "call_id": "expired", "timestamp": time.time() - 61})
    store.append_trace({**base, "call_id": "current", "timestamp": time.time()})

    assert [trace["call_id"] for trace in store.list_traces()] == ["current"]


def test_sqlite_alert_history_is_bounded(tmp_path):
    store = SQLiteObservabilityStore(tmp_path / "api_state.db", ttl_seconds=60, max_alerts=1)
    first = {
        "session_id": "session-a",
        "rule": "latency",
        "metric": "p95_duration_ms",
        "value": 6000,
    }
    second = {**first, "session_id": "session-b"}

    assert store.claim_alert("session-a", "latency", 60, first)
    assert store.claim_alert("session-b", "latency", 60, second)

    history = store.list_alerts()
    assert len(history) == 1
    assert history[0]["session_id"] == "session-b"


class _FakeRedis:
    def __init__(self):
        self.calls: list[tuple] = []

    def eval(self, *args):
        self.calls.append(args)
        return 1


def _redis_store(fake: _FakeRedis) -> RedisObservabilityStore:
    store = RedisObservabilityStore.__new__(RedisObservabilityStore)
    store._redis = fake  # noqa: SLF001
    store._prefix = "mycoder:observability"  # noqa: SLF001
    store.ttl_seconds = 3600
    store.max_alerts = 100
    return store


def test_redis_rate_limit_uses_atomic_server_time_lua():
    redis = _FakeRedis()
    store = _redis_store(redis)

    assert store.rate_limit_allow("tenant:client", 5) is True
    script, key_count, redis_key, window_ms, limit, member = redis.calls[0]
    assert "redis.call('TIME')" in script
    assert "ZREMRANGEBYSCORE" in script and "ZADD" in script
    assert key_count == 1
    assert redis_key.startswith("mycoder:observability:rate:")
    assert "tenant" not in redis_key
    assert window_ms == 60_000 and limit == 5 and member


def test_redis_alert_claim_is_atomic_and_persists_history():
    redis = _FakeRedis()
    store = _redis_store(redis)
    alert = {
        "session_id": "s",
        "rule": "latency",
        "metric": "p95_duration_ms",
        "value": 9000,
    }

    assert store.claim_alert("s", "latency", 60, alert) is True
    script, key_count, cooldown_key, history_key, cooldown_ms, payload, *_ = redis.calls[0]
    assert "SET" in script and "NX" in script and "LPUSH" in script
    assert key_count == 2
    assert cooldown_key.startswith("mycoder:observability:alert-cooldown:")
    assert history_key == "mycoder:observability:alerts"
    assert cooldown_ms == 60_000
    assert '"rule": "latency"' in payload
