"""Tests for Phase 4 multi-agent orchestration (Module A)."""

import asyncio


from mycoder.agents import (
    Blackboard,
    BUILTIN_SUBAGENTS,
    OrchestrationStrategy,
    Orchestrator,
    SubagentRunner,
)
from mycoder.contracts.envelope import (
    SubagentResultEnvelope,
)
from mycoder.tools.subagent_tools import SpawnSubagentTool

INSTANCE = "11111111-2222-3333-4444-555555555555"


def _meta(**kw):
    base = {
        "task_id": "t",
        "subagent_name": "x",
        "subagent_instance_id": INSTANCE,
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:01Z",
        "duration_ms": 1,
    }
    base.update(kw)
    return base


def _envelope(status="success", **kw):
    base = {
        "meta": _meta(),
        "status": status,
        "summary": "done",
        "confidence": "low" if status in ("failed", "cancelled") else "high",
        "result": {"type": "general", "output": "ok"} if status in ("success", "partial") else None,
        "error": None if status == "success" else {"code": "E", "category": "permanent", "retryable": False, "message": "err"},
    }
    base.update(kw)
    return base


def _success_envelope() -> dict:
    return _envelope("success")


def _ctx():
    return {"task_id": "task-1"}


def _executor_returning(data):
    async def _exec(task, system_prompt):
        return data
    return _exec


# ---------------------------------------------------------------------------
# SubagentDefinition
# ---------------------------------------------------------------------------

def test_subagent_readonly_enforced():
    """Read-only subagents (explorer/planner/reviewer) must not own mutation tools."""
    mutation = {"write_file", "edit_file", "execute_in_sandbox"}
    for name in ("explorer", "planner", "reviewer"):
        assert BUILTIN_SUBAGENTS[name].read_only is True
        assert not (set(BUILTIN_SUBAGENTS[name].allowed_tools) & mutation)
    assert BUILTIN_SUBAGENTS["implementer"].read_only is False
    assert "write_file" in BUILTIN_SUBAGENTS["implementer"].allowed_tools


# ---------------------------------------------------------------------------
# SubagentRunner
# ---------------------------------------------------------------------------

async def test_subagent_envelope_validation():
    runner = SubagentRunner(
        BUILTIN_SUBAGENTS["explorer"], "search foo", orchestrator=None,
        parent_context=_ctx(), instance_id=INSTANCE,
        executor=_executor_returning(_success_envelope()),
    )
    env = await runner.run()
    assert isinstance(env, SubagentResultEnvelope)
    assert env.status == "success"
    assert env.meta.subagent_instance_id == INSTANCE


async def test_subagent_envelope_partial_validation():
    data = _envelope("partial", confidence="medium", completeness_ratio=0.5,
                     error={"code": "P", "category": "transient", "retryable": True, "message": "half done"})
    runner = SubagentRunner(
        BUILTIN_SUBAGENTS["implementer"], "t", orchestrator=None,
        parent_context=_ctx(), instance_id=INSTANCE, executor=_executor_returning(data),
    )
    env = await runner.run()
    assert env.status == "partial"
    assert env.completeness_ratio == 0.5


async def test_subagent_invalid_partial_ratio_becomes_error():
    """ratio=0.0 must hard-fail -> error envelope, never silently accepted."""
    data = _envelope("partial", confidence="medium", completeness_ratio=0.0,
                     error={"code": "P", "category": "transient", "retryable": True, "message": "x"})
    runner = SubagentRunner(
        BUILTIN_SUBAGENTS["implementer"], "t", orchestrator=None,
        parent_context=_ctx(), instance_id=INSTANCE, executor=_executor_returning(data),
    )
    env = await runner.run()
    assert env.status == "failed"  # contract violation surfaced, not wrapped


async def test_subagent_envelope_failed_no_result():
    data = _envelope("failed")
    runner = SubagentRunner(
        BUILTIN_SUBAGENTS["reviewer"], "t", orchestrator=None,
        parent_context=_ctx(), instance_id=INSTANCE, executor=_executor_returning(data),
    )
    env = await runner.run()
    assert env.status == "failed"
    assert env.error is not None
    assert env.result is None


async def test_subagent_timeout():
    async def _slow(task, system_prompt):
        await asyncio.sleep(5)
        return _success_envelope()

    from dataclasses import replace

    # copy the definition with a tiny timeout — do NOT mutate the global catalog
    definition = replace(BUILTIN_SUBAGENTS["explorer"], timeout_seconds=0.1)
    runner = SubagentRunner(definition, "t", orchestrator=None,
                            parent_context=_ctx(), instance_id=INSTANCE, executor=_slow)
    env = await runner.run()
    assert env.status == "failed"
    assert env.error.code == "SUBAGENT_TIMEOUT"


async def test_subagent_inner_data_wrapped():
    runner = SubagentRunner(
        BUILTIN_SUBAGENTS["explorer"], "t", orchestrator=None,
        parent_context=_ctx(), instance_id=INSTANCE,
        executor=_executor_returning({"files_found": [], "patterns_searched": [], "total_matches": 0}),
    )
    env = await runner.run()
    assert env.status == "success"  # inner data wrapped as success
    assert env.result.type == "general"


# ---------------------------------------------------------------------------
# Blackboard
# ---------------------------------------------------------------------------

async def test_blackboard_put_get():
    bb = Blackboard(ttl_seconds=60)
    await bb.put("task-1", "discovery:found", {"n": 1})
    assert await bb.get("task-1", "discovery:found") == {"n": 1}
    assert await bb.get("task-1", "discovery:missing") is None


async def test_blackboard_ttl_expiry():
    bb = Blackboard(ttl_seconds=0)
    await bb.put("task-1", "k", "v")
    assert await bb.get("task-1", "k") is None  # already expired


async def test_blackboard_asyncio_lock_concurrent():
    bb = Blackboard()
    async def _writer(i):
        for _ in range(20):
            await bb.put("task-1", "count", i)
    await asyncio.gather(*[_writer(i) for i in range(5)])
    assert await bb.get("task-1", "count") in {0, 1, 2, 3, 4}


async def test_blackboard_query_prefix():
    bb = Blackboard()
    await bb.put("task-1", "discovery:a", 1)
    await bb.put("task-1", "discovery:b", 2)
    await bb.put("task-1", "other:c", 3)
    found = await bb.query("task-1", "discovery:")
    assert len(found) == 2


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _orchestrator():
    return Orchestrator(blackboard=Blackboard(), llm=None)


async def test_orchestrator_sequential_order():
    order = []

    async def _exec_a(task, system_prompt):
        order.append("a")
        return _success_envelope()

    async def _exec_b(task, system_prompt):
        order.append("b")
        return _success_envelope()

    orch = _orchestrator()
    result = await orch.orchestrate(
        "t", OrchestrationStrategy.SEQUENTIAL, parent_context=_ctx(),
        subtasks=[
            {"subagent_name": "explorer", "task": "a", "executor": _exec_a},
            {"subagent_name": "implementer", "task": "b", "executor": _exec_b},
        ],
    )
    assert order == ["a", "b"]  # sequential order preserved
    assert result.success


async def test_parallel_subagents():
    orch = _orchestrator()
    result = await orch.orchestrate(
        "t", OrchestrationStrategy.PARALLEL, parent_context=_ctx(),
        subtasks=[
            {"subagent_name": "explorer", "task": "a", "executor": _executor_returning(_success_envelope())},
            {"subagent_name": "reviewer", "task": "b", "executor": _executor_returning(_success_envelope())},
        ],
    )
    assert {"explorer", "reviewer"} <= set(result.results)
    assert result.results["explorer"].status == "success"


async def test_circuit_breaker_after_3_failures():
    async def _fail(task, system_prompt):
        return _envelope("failed")

    orch = _orchestrator()
    # first call: 3 failures trip the breaker (count reaches the threshold)
    await orch.orchestrate(
        "t", OrchestrationStrategy.SEQUENTIAL, parent_context=_ctx(),
        subtasks=[
            {"subagent_name": "reviewer", "task": f"f{i}", "executor": _fail} for i in range(3)
        ],
    )
    # next call (within the cooldown): the same subagent is skipped before running
    result = await orch.orchestrate(
        "t", OrchestrationStrategy.SEQUENTIAL, parent_context=_ctx(),
        subtasks=[{"subagent_name": "reviewer", "task": "f4", "executor": _fail}],
    )
    env = result.results["reviewer"]
    assert env.error.code == "CIRCUIT_BREAKER_OPEN"


async def test_circuit_breaker_self_heals_after_cooldown():
    """A probe after the cooldown runs; on success the breaker closes."""
    from mycoder.agents.orchestrator import CIRCUIT_BREAKER_COOLDOWN

    calls = {"n": 0}

    async def _exec(task, system_prompt):
        calls["n"] += 1
        return _envelope("failed" if calls["n"] <= 3 else "success")

    orch = _orchestrator()
    await orch.orchestrate(
        "t", OrchestrationStrategy.SEQUENTIAL, parent_context=_ctx(),
        subtasks=[{"subagent_name": "reviewer", "task": f"f{i}", "executor": _exec} for i in range(3)],
    )
    # simulate the cooldown elapsing -> next call is a half-open probe
    orch._circuit_open_since["reviewer"] -= CIRCUIT_BREAKER_COOLDOWN + 1
    result = await orch.orchestrate(
        "t", OrchestrationStrategy.SEQUENTIAL, parent_context=_ctx(),
        subtasks=[{"subagent_name": "reviewer", "task": "probe", "executor": _exec}],
    )
    env = result.results["reviewer"]
    assert env.status == "success"  # probe succeeded -> closed
    assert orch._circuit_breaker["reviewer"] == 0


async def test_circuit_breaker_probe_failure_reopens():
    """A failing probe re-opens the breaker for another full cooldown."""
    from mycoder.agents.orchestrator import CIRCUIT_BREAKER_COOLDOWN

    async def _always_fail(task, system_prompt):
        return _envelope("failed")

    orch = _orchestrator()
    await orch.orchestrate(
        "t", OrchestrationStrategy.SEQUENTIAL, parent_context=_ctx(),
        subtasks=[{"subagent_name": "reviewer", "task": f"f{i}", "executor": _always_fail} for i in range(3)],
    )
    # after cooldown, the probe RUNS (and fails)
    orch._circuit_open_since["reviewer"] -= CIRCUIT_BREAKER_COOLDOWN + 1
    r1 = await orch.orchestrate(
        "t", OrchestrationStrategy.SEQUENTIAL, parent_context=_ctx(),
        subtasks=[{"subagent_name": "reviewer", "task": "probe1", "executor": _always_fail}],
    )
    assert r1.results["reviewer"].status == "failed"  # probe ran, failed
    # re-opened: immediately after, it is skipped again (fresh cooldown)
    r2 = await orch.orchestrate(
        "t", OrchestrationStrategy.SEQUENTIAL, parent_context=_ctx(),
        subtasks=[{"subagent_name": "reviewer", "task": "probe2", "executor": _always_fail}],
    )
    assert r2.results["reviewer"].error.code == "CIRCUIT_BREAKER_OPEN"


# ---------------------------------------------------------------------------
# spawn_subagent tool
# ---------------------------------------------------------------------------

def test_spawn_subagent_tool_formats_envelope():
    async def _exec(task, system_prompt):
        return _success_envelope()

    class _FakeOrch:
        async def spawn_subagent(self, subagent_type, task, blocking):
            runner = SubagentRunner(
                BUILTIN_SUBAGENTS[subagent_type], task, orchestrator=None,
                parent_context=_ctx(), instance_id=INSTANCE, executor=_exec,
            )
            return await runner.run()

    tool = SpawnSubagentTool(orchestrator=_FakeOrch())
    out = tool.execute(subagent_type="explorer", task="search")
    assert "Sub-agent success" in out
