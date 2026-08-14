"""API service layer tests — deterministic (no real LLM).

Covers the FastAPI layer in isolation: state backends, /health, /status
(404 + perf), /run (503 without a key; 202 -> background completion via a fake
orchestrator), /cost, session sanitization, the 429 token-budget handler and
the optional rate limiter. The tracer / dependency overrides are reset per test.
"""

import json

import pytest

import mycoder.config as cfg

cfg._load_dotenv = lambda: None  # noqa: SLF001 - keep tests off the dev .env

from fastapi.testclient import TestClient  # noqa: E402

from api import server  # noqa: E402
from api.server import app, sanitize_session_id  # noqa: E402
from api.state_backend import (  # noqa: E402
    LocalStateBackend,
    create_state_backend,
)


# ------------------------------------------------------------ state backend
class _DictBackend:
    """Thread-safe in-memory StateBackend for TestClient (sqlite connections are
    thread-bound, so the real LocalStateBackend can't cross TestClient's worker
    thread)."""

    def __init__(self):
        self.sessions: dict = {}
        self.blackboards: dict = {}

    async def get_session(self, sid):
        return self.sessions.get(sid)

    async def save_session(self, sid, data):
        self.sessions[sid] = data

    async def get_blackboard(self, sid):
        return self.blackboards.get(sid)

    async def save_blackboard(self, sid, data):
        self.blackboards[sid] = data

    async def list_sessions(self):
        return [{"session_id": sid, **data} for sid, data in self.sessions.items()]


def test_local_backend_session_and_blackboard_crud(tmp_path):
    import asyncio

    backend = LocalStateBackend(project_dir=tmp_path / "proj")
    async def _run():
        assert await backend.get_session("s1") is None
        await backend.save_session("s1", {"status": "running"})
        await backend.save_session("s1", {"status": "success", "perf": {"llm_calls": 3}})
        assert (await backend.get_session("s1"))["status"] == "success"
        assert (await backend.get_session("s1"))["perf"]["llm_calls"] == 3
        await backend.save_blackboard("s1", {"task:plan": {"n": 1}})
        assert (await backend.get_blackboard("s1"))["task:plan"]["n"] == 1

    asyncio.run(_run())


def test_create_state_backend_factory(tmp_path):
    assert isinstance(create_state_backend("local"), LocalStateBackend)


def test_sanitize_session_id():
    assert sanitize_session_id("ab-cd_12") == "ab-cd_12"
    assert sanitize_session_id("a/b..c") == "abc"  # traversal scrubbed
    assert sanitize_session_id("a b*c;rm") == "abcrm"
    assert len(sanitize_session_id("x" * 200)) <= 64
    assert len(sanitize_session_id("")) == 16  # fresh id


# ------------------------------------------------------------ HTTP endpoints
@pytest.fixture
def client():
    backend = _DictBackend()
    app.dependency_overrides[server.get_state_backend] = lambda: backend
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["redis"] in ("connected", "disconnected")
    assert body["version"]


def test_status_404_for_unknown_session(client):
    assert client.get("/v1/agent/status/nope").status_code == 404


def test_status_returns_perf():
    backend = _DictBackend()
    backend.sessions["s1"] = {
        "status": "success",
        "perf": {"llm_calls": 10, "total_tokens": 269273, "cost_usd": 0.35},
    }
    app.dependency_overrides[server.get_state_backend] = lambda: backend
    try:
        r = TestClient(app).get("/v1/agent/status/s1")
        assert r.status_code == 200
        assert r.json()["perf"]["total_tokens"] == 269273
    finally:
        app.dependency_overrides.clear()


def test_run_requires_llm_key(client, monkeypatch):
    monkeypatch.setattr(server, "get_default_llm", lambda: None)
    r = client.post("/v1/agent/run", json={"task": "hi"})
    assert r.status_code == 503


class _FakeResult:
    success = True
    tokens_used = 0
    summary = "ok"
    results = {}


class _FakeOrchestrator:
    async def orchestrate(self, **kwargs):
        return _FakeResult()


def test_run_schedules_and_background_completes(client, monkeypatch):
    monkeypatch.setattr(server, "get_default_llm", lambda: object())
    monkeypatch.setattr(server, "get_orchestrator", lambda sb: (lambda sid, **kw: _FakeOrchestrator()))

    r = client.post("/v1/agent/run", json={"task": "写个函数", "session_id": "s-run"})
    assert r.status_code == 202
    assert r.json()["status"] == "running"

    # TestClient completes background tasks before returning, so the worker's
    # fake orchestrate() has already written a terminal state.
    st = client.get("/v1/agent/status/s-run")
    assert st.status_code == 200
    assert st.json()["status"] == "success"


def test_cost_endpoint(client):
    from api.dependencies import get_tracer

    tracer = get_tracer()
    with tracer.trace("s-cost", "test", "glm-5.2") as ctx:
        ctx["prompt_tokens"] = 1000
        ctx["completion_tokens"] = 500
    r = client.get("/v1/agent/cost/s-cost")
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["total_calls"] == 1
    assert body["cost"]["total_cost_usd"] > 0
    assert client.get("/v1/agent/cost/never").status_code == 404


def test_token_budget_handler_returns_429():
    import asyncio

    from mycoder.observability.budget import TokenBudgetExceeded

    exc = TokenBudgetExceeded(session_id="s", used_tokens=100, max_tokens=50)
    resp = asyncio.run(server._on_token_budget(None, exc))  # noqa: SLF001
    assert resp.status_code == 429
    assert json.loads(resp.body)["code"] == "TOKEN_BUDGET_EXCEEDED"


def test_rate_limiter_429_on_breach(client, monkeypatch):
    from mycoder.observability.ratelimit import RateLimiter

    monkeypatch.setattr(server, "get_default_llm", lambda: object())
    monkeypatch.setattr(server, "get_orchestrator", lambda sb: (lambda sid, **kw: _FakeOrchestrator()))
    monkeypatch.setattr(server, "RATE_LIMITER", RateLimiter(requests_per_minute=1))

    assert client.post("/v1/agent/run", json={"task": "t1"}).status_code == 202
    assert client.post("/v1/agent/run", json={"task": "t2"}).status_code == 429


def test_lifespan_shutdown_stops_sandbox_manager(monkeypatch):
    """FastAPI shutdown tears down the process-global sandbox (auto-close).

    The implementer subagent can create a DockerSandbox; the lifespan shutdown
    must stop it so the container + volume don't outlive the server process.
    """
    from unittest import mock

    import mycoder.sandbox.executor as ex_mod

    manager = mock.AsyncMock()
    monkeypatch.setattr(ex_mod, "get_active_manager", lambda: manager)
    with TestClient(app):
        pass  # lifespan startup + shutdown run around this block
    manager.stop.assert_awaited_once()


def test_lifespan_closes_state_backend(monkeypatch):
    """#14: FastAPI shutdown closes the state backend (Redis connection too)."""
    from unittest import mock

    import api.dependencies as deps

    backend = mock.AsyncMock()
    backend.close = mock.AsyncMock()
    monkeypatch.setattr(deps, "get_state_backend", lambda: backend)
    with TestClient(app):
        pass
    backend.close.assert_awaited_once()


def test_metrics_aggregates_success_rate(client):
    """GET /v1/agent/metrics aggregates production run statuses (P2)."""
    backend = app.dependency_overrides[server.get_state_backend]()
    backend.sessions["a"] = {"status": "success"}
    backend.sessions["b"] = {"status": "failed", "error": {"code": "SUBAGENT_ERROR"}}
    backend.sessions["c"] = {"status": "running"}

    r = client.get("/v1/agent/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["total_runs"] == 3
    assert body["completed"] == 2
    assert body["running"] == 1
    assert body["success"] == 1 and body["failed"] == 1
    assert body["success_rate"] == 0.5
    assert body["failure_distribution"] == {"SUBAGENT_ERROR": 1}


def test_monitor_report_aggregates_llm_and_runs(client):
    """GET /v1/agent/report is the one-snapshot monitor endpoint (P3)."""
    backend = app.dependency_overrides[server.get_state_backend]()
    backend.sessions["a"] = {"status": "success"}

    r = client.get("/v1/agent/report")
    assert r.status_code == 200
    body = r.json()
    assert "llm" in body and "per_session" in body
    assert "generated_at" in body
    assert body["production_runs"]["success_rate"] == 1.0
