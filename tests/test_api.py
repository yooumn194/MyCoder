"""API service layer tests — deterministic (no real LLM).

Covers the FastAPI layer in isolation: state backends, /health, /status
(404 + perf), /run (503 without a key; 202 -> background completion via a fake
orchestrator), /cost, session sanitization, the 429 token-budget handler and
the optional rate limiter. The tracer / dependency overrides are reset per test.
"""

import json

import pytest

import mycoder.config as cfg
from mycoder.agents.checkpoint import CheckpointStore

cfg._load_dotenv = lambda: None  # noqa: SLF001 - keep tests off the dev .env

from fastapi.testclient import TestClient  # noqa: E402

from api import server  # noqa: E402
from api.server import app, sanitize_session_id  # noqa: E402
from api.state_backend import (  # noqa: E402
    LocalStateBackend,
    create_state_backend,
)
from api.auth import scope_session_id  # noqa: E402


# ------------------------------------------------------------ state backend
class _DictBackend:
    """Thread-safe in-memory StateBackend for TestClient (sqlite connections are
    thread-bound, so the real LocalStateBackend can't cross TestClient's worker
    thread)."""

    def __init__(self):
        self.sessions: dict = {}
        self.blackboards: dict = {}
        self.jobs: dict = {}
        self.leases: dict = {}
        self.dead_jobs: list[dict] = []

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

    async def enqueue_job(self, job_id, payload):
        if job_id in self.jobs:
            return False
        self.jobs[job_id] = {"payload": payload, "owner": None, "attempts": 0}
        return True

    async def claim_job(self, worker_id, lease_seconds, job_id=None):
        candidates = [job_id] if job_id else list(self.jobs)
        for candidate in candidates:
            job = self.jobs.get(candidate)
            if job and job["owner"] is None:
                job["owner"] = worker_id
                job["attempts"] += 1
                return {"job_id": candidate, "payload": job["payload"], "attempts": job["attempts"]}
        return None

    async def renew_job(self, job_id, worker_id, lease_seconds):
        return bool(self.jobs.get(job_id, {}).get("owner") == worker_id)

    async def complete_job(self, job_id, worker_id):
        if self.jobs.get(job_id, {}).get("owner") != worker_id:
            return False
        del self.jobs[job_id]
        return True

    async def retry_job(self, job_id, worker_id, delay_seconds=0):
        if self.jobs.get(job_id, {}).get("owner") != worker_id:
            return False
        self.jobs[job_id]["owner"] = None
        return True

    async def dead_letter_job(self, job_id, worker_id, reason):
        job = self.jobs.get(job_id)
        if not job or job.get("owner") != worker_id:
            return False
        self.dead_jobs.append(
            {
                "job_id": job_id,
                "payload": job["payload"],
                "attempts": job["attempts"],
                "reason": reason,
                "failed_at": 1.0,
            }
        )
        del self.jobs[job_id]
        return True

    async def list_dead_jobs(self, limit=100):
        return list(reversed(self.dead_jobs))[:limit]

    async def acquire_lease(self, name, owner, lease_seconds):
        if name in self.leases:
            return False
        self.leases[name] = owner
        return True

    async def renew_lease(self, name, owner, lease_seconds):
        return self.leases.get(name) == owner

    async def release_lease(self, name, owner):
        if self.leases.get(name) != owner:
            return False
        del self.leases[name]
        return True


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


def test_durable_queue_dual_worker_contention_and_crash_recovery(tmp_path):
    """Only one worker wins; an expired claim is recovered after a crash."""
    import asyncio
    import time

    project = tmp_path / "shared"
    first = LocalStateBackend(project_dir=project)
    second = LocalStateBackend(project_dir=project)

    async def _run():
        assert await first.enqueue_job("job-1", {"task": "x"}) is True
        assert await second.enqueue_job("job-1", {"task": "x"}) is False
        claims = await asyncio.gather(
            first.claim_job("worker-a", 60),
            second.claim_job("worker-b", 60),
        )
        winners = [claim for claim in claims if claim is not None]
        assert len(winners) == 1
        assert winners[0]["attempts"] == 1
        owner = "worker-a" if claims[0] else "worker-b"
        loser_backend = second if owner == "worker-a" else first
        # Simulate SIGKILL: no ACK/retry, only the lease expires.
        conn = first._connect()  # noqa: SLF001 - deterministic crash simulation
        conn.execute("UPDATE jobs SET lease_until=? WHERE job_id='job-1'", (time.time() - 1,))
        conn.commit()
        recovered = await loser_backend.claim_job("worker-restarted", 60)
        assert recovered is not None
        assert recovered["attempts"] == 2
        assert await loser_backend.complete_job("job-1", "worker-restarted") is True

    asyncio.run(_run())


def test_local_dead_letter_atomically_removes_active_job(tmp_path):
    import asyncio

    backend = LocalStateBackend(project_dir=tmp_path)

    async def _run():
        assert await backend.enqueue_job("bad", {"task": "x"})
        claim = await backend.claim_job("worker", 60)
        assert claim and claim["attempts"] == 1
        assert await backend.dead_letter_job("bad", "worker", "boom")
        assert await backend.claim_job("other", 60) is None
        dead = await backend.list_dead_jobs()
        assert dead[0]["job_id"] == "bad"
        assert dead[0]["payload"] == {"task": "x"}
        assert dead[0]["attempts"] == 1
        assert dead[0]["reason"] == "boom"

    asyncio.run(_run())


def test_local_named_lease_is_owner_checked(tmp_path):
    import asyncio

    backend = LocalStateBackend(project_dir=tmp_path)

    async def _run():
        assert await backend.acquire_lease("session:s1", "a", 60) is True
        assert await backend.acquire_lease("session:s1", "b", 60) is False
        assert await backend.renew_lease("session:s1", "b", 60) is False
        assert await backend.release_lease("session:s1", "b") is False
        assert await backend.release_lease("session:s1", "a") is True

    asyncio.run(_run())


def test_sanitize_session_id():
    assert sanitize_session_id("ab-cd_12") == "ab-cd_12"
    assert sanitize_session_id("a/b..c") == "abc"  # traversal scrubbed
    assert sanitize_session_id("a b*c;rm") == "abcrm"
    assert len(sanitize_session_id("x" * 200)) <= 64
    assert len(sanitize_session_id("")) == 16  # fresh id


# ------------------------------------------------------------ HTTP endpoints
@pytest.fixture
def client(tmp_path, monkeypatch):
    import api.dependencies as dependencies

    monkeypatch.setenv("MYCODER_OBSERVABILITY_PATH", str(tmp_path / "api_state.db"))
    dependencies.reset_observability_runtime()
    backend = _DictBackend()
    checkpoint_store = CheckpointStore(tmp_path / "checkpoints")
    app.dependency_overrides[server.get_state_backend] = lambda: backend
    app.dependency_overrides[server.get_checkpoint_store] = lambda: checkpoint_store
    yield TestClient(app)
    dependencies.reset_observability_runtime()
    app.dependency_overrides.clear()
    server._ACTIVE_SESSIONS.clear()  # noqa: SLF001


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["redis"] in ("connected", "disconnected")
    assert body["version"]


def test_status_404_for_unknown_session(client):
    assert client.get("/v1/agent/status/nope").status_code == 404


def test_api_auth_is_required_by_default(client, monkeypatch):
    monkeypatch.delenv("MYCODER_REQUIRE_AUTH", raising=False)
    monkeypatch.delenv("MYCODER_API_KEYS", raising=False)

    response = client.get("/v1/agent/status/nope")

    assert response.status_code == 503
    assert "authentication is required" in response.json()["detail"]


def test_api_key_auth_and_cross_tenant_session_isolation(client, monkeypatch):
    monkeypatch.setenv("MYCODER_API_KEYS", '{"tenant-a":"secret-a","tenant-b":"secret-b"}')
    backend = app.dependency_overrides[server.get_state_backend]()
    public_id = "same-id"
    backend.sessions[scope_session_id("tenant-a", public_id)] = {
        "tenant_id": "tenant-a", "public_session_id": public_id, "status": "success"
    }
    backend.sessions[scope_session_id("tenant-b", public_id)] = {
        "tenant_id": "tenant-b", "public_session_id": public_id, "status": "failed"
    }

    assert client.get(f"/v1/agent/status/{public_id}").status_code == 401
    a = client.get(f"/v1/agent/status/{public_id}", headers={"X-API-Key": "secret-a"})
    b = client.get(
        f"/v1/agent/status/{public_id}",
        headers={"Authorization": "Bearer secret-b"},
    )
    assert a.json()["status"] == "success"
    assert b.json()["status"] == "failed"
    assert client.get(
        f"/v1/agent/status/{public_id}", headers={"X-API-Key": "wrong"}
    ).status_code == 401


def test_metrics_are_tenant_scoped(client, monkeypatch):
    monkeypatch.setenv("MYCODER_API_KEYS", '{"a":"ka","b":"kb"}')
    backend = app.dependency_overrides[server.get_state_backend]()
    backend.sessions["a-1"] = {"tenant_id": "a", "status": "success"}
    backend.sessions["b-1"] = {"tenant_id": "b", "status": "failed"}
    response = client.get("/v1/agent/metrics", headers={"X-API-Key": "ka"})
    assert response.status_code == 200
    assert response.json()["total_runs"] == 1
    assert response.json()["success_rate"] == 1.0


def test_status_returns_perf(tmp_path):
    backend = _DictBackend()
    checkpoint_store = CheckpointStore(tmp_path / "checkpoints")
    backend.sessions["s1"] = {
        "status": "success",
        "perf": {"llm_calls": 10, "total_tokens": 269273, "cost_usd": 0.35},
    }
    app.dependency_overrides[server.get_state_backend] = lambda: backend
    app.dependency_overrides[server.get_checkpoint_store] = lambda: checkpoint_store
    try:
        r = TestClient(app).get("/v1/agent/status/s1")
        assert r.status_code == 200
        assert r.json()["perf"]["total_tokens"] == 269273
    finally:
        app.dependency_overrides.clear()


def test_sse_returns_structured_terminal_snapshot(client):
    backend = app.dependency_overrides[server.get_state_backend]()
    backend.sessions["streamed"] = {
        "status": "success",
        "current_step": "done",
        "token_usage": 42,
        "output": "answer",
        "error": None,
    }

    response = client.get("/v1/agent/events/streamed")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: completed" in response.text
    data_line = next(
        line for line in response.text.splitlines() if line.startswith("data: ")
    )
    payload = json.loads(data_line.removeprefix("data: "))
    assert payload["output"] == "answer"
    assert payload["token_usage"] == 42


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
    def __init__(self, result=None):
        self.result = result or _FakeResult()
        self.kwargs = None

    async def orchestrate(self, **kwargs):
        self.kwargs = kwargs
        return self.result


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
    assert st.json()["output"] == "ok"


def test_run_exposes_single_reasoning_ablation_mode(client, monkeypatch):
    import mycoder.agent as agent_module

    captured = {}

    class _SingleAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def chat(self, task):
            captured["task"] = task
            return "done"

    orchestrator = _FakeOrchestrator()
    orchestrator.llm = object()
    orchestrator.tools = []
    monkeypatch.setattr(agent_module, "Agent", _SingleAgent)
    monkeypatch.setattr(server, "get_default_llm", lambda: object())
    monkeypatch.setattr(
        server, "get_orchestrator", lambda sb: (lambda sid, **kw: orchestrator)
    )

    response = client.post(
        "/v1/agent/run",
        json={
            "task": "ablation",
            "session_id": "single-react",
            "execution_mode": "single",
            "reasoning_strategy": "react",
        },
    )
    assert response.status_code == 202
    assert captured["reasoning_strategy"] == "react"
    assert captured["task"] == "ablation"


def test_run_passes_explicit_multi_orchestration_strategy(client, monkeypatch):
    orchestrator = _FakeOrchestrator()
    monkeypatch.setattr(server, "get_default_llm", lambda: object())
    monkeypatch.setattr(
        server, "get_orchestrator", lambda sb: (lambda sid, **kw: orchestrator)
    )
    response = client.post(
        "/v1/agent/run",
        json={
            "task": "multi",
            "session_id": "multi-parallel",
            "execution_mode": "multi",
            "orchestration_strategy": "parallel",
        },
    )
    assert response.status_code == 202
    assert orchestrator.kwargs["strategy"] == server.OrchestrationStrategy.PARALLEL


def test_resume_reuses_checkpoint_plan_and_marks_status(client, monkeypatch):
    backend = app.dependency_overrides[server.get_state_backend]()
    store = app.dependency_overrides[server.get_checkpoint_store]()
    store.save_plan(
        "s-resume",
        "继续实现",
        [{"id": "a", "subagent_name": "explorer", "task": "inspect"}],
    )
    orchestrator = _FakeOrchestrator()
    monkeypatch.setattr(server, "get_default_llm", lambda: object())
    monkeypatch.setattr(
        server, "get_orchestrator", lambda sb: (lambda sid, **kw: orchestrator)
    )

    response = client.post(
        "/v1/agent/run",
        json={"task": "继续实现", "session_id": "s-resume", "resume": True},
    )

    assert response.status_code == 202
    assert orchestrator.kwargs["resume"] is True
    assert orchestrator.kwargs["subtasks"] is None
    assert backend.sessions["s-resume"]["resumed"] is True
    assert client.get("/v1/agent/status/s-resume").json()["resumed"] is True
    assert store.load("s-resume") is None  # successful run cleans up


def test_resume_requires_existing_matching_checkpoint(client, monkeypatch):
    store = app.dependency_overrides[server.get_checkpoint_store]()
    monkeypatch.setattr(server, "get_default_llm", lambda: object())

    missing = client.post(
        "/v1/agent/run",
        json={"task": "x", "session_id": "missing", "resume": True},
    )
    assert missing.status_code == 404

    store.save_plan(
        "s-mismatch",
        "original",
        [{"id": "a", "subagent_name": "explorer", "task": "inspect"}],
    )
    mismatch = client.post(
        "/v1/agent/run",
        json={"task": "different", "session_id": "s-mismatch", "resume": True},
    )
    assert mismatch.status_code == 409


def test_failed_resume_keeps_checkpoint_progress(client, monkeypatch):
    store = app.dependency_overrides[server.get_checkpoint_store]()
    store.save_plan(
        "s-failed",
        "task",
        [{"subagent_name": "explorer", "task": "inspect"}],
    )

    class _FailedResult:
        success = False
        tokens_used = 0
        summary = "failed"
        results = {}

    monkeypatch.setattr(server, "get_default_llm", lambda: object())
    monkeypatch.setattr(
        server,
        "get_orchestrator",
        lambda sb: (lambda sid, **kw: _FakeOrchestrator(_FailedResult())),
    )

    response = client.post(
        "/v1/agent/run",
        json={"task": "task", "session_id": "s-failed", "resume": True},
    )
    status = client.get("/v1/agent/status/s-failed").json()

    assert response.status_code == 202
    assert status["status"] == "failed"
    assert status["checkpoint"] == {
        "task": "task",
        "assignment_count": 1,
        "completed_count": 0,
        "completed_steps": [],
        "remaining_steps": ["explorer-1"],
    }


def test_active_session_cannot_be_started_twice(client, monkeypatch):
    monkeypatch.setattr(server, "get_default_llm", lambda: object())
    server._ACTIVE_SESSIONS.add("s-active")  # noqa: SLF001
    try:
        response = client.post(
            "/v1/agent/run", json={"task": "x", "session_id": "s-active"}
        )
    finally:
        server._ACTIVE_SESSIONS.discard("s-active")  # noqa: SLF001
    assert response.status_code == 409


def test_worker_dead_letters_after_max_attempts(tmp_path, monkeypatch):
    import asyncio

    backend = _DictBackend()
    store = CheckpointStore(tmp_path / "checkpoints")
    payload = {
        "session_id": "poison",
        "public_session_id": "poison",
        "tenant_id": "local",
        "workspace_id": "default",
        "task": "fail",
    }

    async def explode(*_args, **_kwargs):
        raise RuntimeError("infrastructure down")

    monkeypatch.setattr(server, "_run_agent", explode)
    monkeypatch.setattr(server, "_JOB_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(server, "_job_retry_delay", lambda _attempts: 0)

    async def _run():
        await backend.enqueue_job("poison", payload)
        await backend.save_session("poison", {"status": "running"})
        with pytest.raises(RuntimeError, match="infrastructure down"):
            await server._process_job_id(backend, store, "poison", "w1")
        with pytest.raises(RuntimeError, match="infrastructure down"):
            await server._process_job_id(backend, store, "poison", "w2")
        assert "poison" not in backend.jobs
        assert backend.dead_jobs[0]["attempts"] == 2
        assert "RuntimeError" in backend.dead_jobs[0]["reason"]
        assert backend.sessions["poison"]["error"]["code"] == "JOB_MAX_ATTEMPTS"

    asyncio.run(_run())


def test_dead_letter_endpoint_is_observable(client):
    backend = app.dependency_overrides[server.get_state_backend]()
    backend.dead_jobs.append(
        {"job_id": "d1", "payload": {"tenant_id": None}, "attempts": 3}
    )
    response = client.get("/v1/agent/dead-letter")
    assert response.status_code == 200
    assert response.json()["jobs"][0]["job_id"] == "d1"


def test_cost_endpoint(client):
    import api.dependencies as dependencies
    from api.dependencies import get_tracer

    tracer = get_tracer()
    with tracer.trace("s-cost", "test", "glm-5.2") as ctx:
        ctx["prompt_tokens"] = 1000
        ctx["completion_tokens"] = 500
    # Simulate a worker/process reconstruction: endpoint must read persisted
    # traces rather than the old Python object.
    dependencies.reset_observability_runtime()
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


def test_alert_history_endpoint_reads_persisted_store(client):
    import api.dependencies as dependencies
    from mycoder.observability.alerts import AlertRule

    manager = dependencies.get_alert_manager()
    manager.rules = [
        AlertRule("latency", "p95_duration_ms", 100, op=">", cooldown_seconds=60)
    ]
    assert manager.evaluate("s-alert", {"p95_duration_ms": 200})

    # Rebuild every process-local wrapper; persisted history must remain.
    dependencies.reset_observability_runtime()
    response = client.get("/v1/agent/alerts")

    assert response.status_code == 200
    alerts = response.json()["alerts"]
    assert alerts[0]["session_id"] == "s-alert"
    assert alerts[0]["rule"] == "latency"


def test_alert_history_is_tenant_scoped(client, monkeypatch):
    import api.dependencies as dependencies
    from mycoder.observability.alerts import AlertRule

    monkeypatch.setenv("MYCODER_API_KEYS", '{"tenant-a":"ka","tenant-b":"kb"}')
    backend = app.dependency_overrides[server.get_state_backend]()
    session_a = scope_session_id("tenant-a", "shared")
    session_b = scope_session_id("tenant-b", "shared")
    backend.sessions[session_a] = {"tenant_id": "tenant-a", "status": "success"}
    backend.sessions[session_b] = {"tenant_id": "tenant-b", "status": "success"}

    manager = dependencies.get_alert_manager()
    manager.rules = [
        AlertRule("latency", "p95_duration_ms", 100, op=">", cooldown_seconds=60)
    ]
    assert manager.evaluate(session_a, {"p95_duration_ms": 200})
    assert manager.evaluate(session_b, {"p95_duration_ms": 200})

    response = client.get("/v1/agent/alerts", headers={"X-API-Key": "ka"})

    assert response.status_code == 200
    alerts = response.json()["alerts"]
    assert len(alerts) == 1
    assert alerts[0]["session_id"] == session_a


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
