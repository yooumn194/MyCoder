"""Checkpoint / resume (断点续跑) — orchestrator mid-plan interruption recovery."""

import asyncio

from mycoder.agents import Blackboard, Orchestrator
from mycoder.agents.checkpoint import CheckpointStore
from mycoder.contracts.envelope import SubagentResultEnvelope


def _env(status="success", **kw):
    base = {
        "meta": {
            "task_id": "t",
            "subagent_name": "x",
            "subagent_instance_id": "11111111-2222-3333-4444-555555555555",
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:00:01Z",
            "duration_ms": 1,
        },
        "status": status,
        "summary": "done",
        "confidence": "low" if status in ("failed", "cancelled") else "high",
        "result": {"type": "general", "output": "ok"} if status in ("success", "partial") else None,
        "error": None if status == "success" else {"code": "E", "category": "transient", "retryable": True, "message": "err"},
    }
    base.update(kw)
    return SubagentResultEnvelope.model_validate(base)


def _make_exec(sequence):
    calls = []

    async def exec_fn(task, system_prompt):
        calls.append(task)
        return sequence.pop(0).model_dump()

    exec_fn.calls = calls
    return exec_fn


def test_checkpoint_store_roundtrip(tmp_path):
    store = CheckpointStore(base_dir=tmp_path)
    assignments = [
        {"id": "a", "subagent_name": "explorer", "task": "t1"},
        {"id": "b", "subagent_name": "explorer", "task": "t2"},
    ]
    store.save_plan("s1", "the task", assignments)
    store.record_result("s1", "a", _env("success").model_dump())

    cp = store.load("s1")
    assert cp is not None
    assert cp["task"] == "the task"
    assert cp["assignments"] == assignments
    assert cp["results"]["a"].status == "success"
    assert store.summary("s1") == {
        "task": "the task",
        "assignment_count": 2,
        "completed_count": 1,
        "completed_steps": ["a"],
        "remaining_steps": ["b"],
    }

    assert store.load("missing") is None
    store.clear("s1")
    assert store.load("s1") is None


def test_save_plan_drops_executor_field(tmp_path):
    """Runtime-injected executor callables are not persisted."""
    store = CheckpointStore(base_dir=tmp_path)
    store.save_plan(
        "s1",
        "t",
        [{"id": "a", "subagent_name": "explorer", "task": "t1", "executor": lambda: None}],
    )
    cp = store.load("s1")
    assert "executor" not in cp["assignments"][0]


def test_orchestrate_resume_reuses_plan_and_skips_completed(tmp_path):
    """After a completed run's checkpoints, resume skips every step (executor is
    not called again) — the same plan is reused, not re-decomposed."""
    store = CheckpointStore(base_dir=tmp_path)
    orch = Orchestrator(blackboard=Blackboard(), llm=None, checkpoint_store=store)
    ex = _make_exec([_env("success")] * 3)
    assignments = [
        {"id": f"t{i}", "subagent_name": "explorer", "task": f"t{i}", "executor": ex}
        for i in range(3)
    ]

    asyncio.run(
        orch.orchestrate(task="x", parent_context={"task_id": "s1"}, subtasks=assignments)
    )
    assert ex.calls == ["t0", "t1", "t2"]
    assert len(store.load("s1")["results"]) == 3  # all steps checkpointed

    # resume: same plan, all steps already succeeded -> skipped
    r2 = asyncio.run(
        orch.orchestrate(task="x", parent_context={"task_id": "s1"}, resume=True)
    )
    assert ex.calls == ["t0", "t1", "t2"]  # no new executions
    assert r2.success is True


def test_execute_sequential_skips_checkpointed_steps(tmp_path):
    """_execute_sequential skips already-successful steps and only runs the
    remaining ones (the resume-with-partial-progress path)."""
    store = CheckpointStore(base_dir=tmp_path)
    orch = Orchestrator(blackboard=Blackboard(), llm=None, checkpoint_store=store)
    orch._session_id = "s1"
    # simulate a resume: steps a & c succeeded before the interruption
    orch._checkpoint_done = {"a": _env("success"), "c": _env("success")}

    ex = _make_exec([_env("success")])  # only b executes
    assignments = [
        {"id": "a", "subagent_name": "explorer", "task": "a", "executor": ex},
        {"id": "b", "subagent_name": "explorer", "task": "b", "executor": ex},
        {"id": "c", "subagent_name": "explorer", "task": "c", "executor": ex},
    ]
    results = asyncio.run(
        orch._execute_sequential(assignments, {"task_id": "s1"}, {})
    )
    assert ex.calls == ["b"]  # a/c skipped from checkpoint
    assert results["explorer"].status == "success"  # c's checkpointed envelope


def test_execute_conditional_skips_checkpointed_steps(tmp_path):
    """Conditional strategy must honor the same resume contract."""
    store = CheckpointStore(base_dir=tmp_path)
    orch = Orchestrator(blackboard=Blackboard(), llm=None, checkpoint_store=store)
    orch._session_id = "s1"
    orch._checkpoint_done = {"a": _env("success")}
    orch._completed_by_id = {}
    orch._result_keys = {}
    ex = _make_exec([_env("success")])
    assignments = orch._normalize_assignments(
        [
            {"id": "a", "subagent_name": "explorer", "task": "a", "executor": ex},
            {"id": "b", "subagent_name": "explorer", "task": "b", "executor": ex},
        ]
    )

    results = asyncio.run(
        orch._execute_conditional(assignments, {"task_id": "s1"}, {})
    )

    assert ex.calls == ["b"]
    assert len(results) == 2
