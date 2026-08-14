"""P1 dynamic re-planning — DeviationDetector + graded recovery in the
Orchestrator's sequential execution (对标 Codex / Claude Code / Hermes)."""

import asyncio

from mycoder.agents import Blackboard, Orchestrator
from mycoder.agents.planner import SubTask
from mycoder.agents.replan import Deviation, DeviationDetector, DeviationType
from mycoder.contracts.envelope import SubagentResultEnvelope


# ------------------------------------------------------------- helpers
def _meta(**kw):
    base = {
        "task_id": "t",
        "subagent_name": "x",
        "subagent_instance_id": "11111111-2222-3333-4444-555555555555",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:01Z",
        "duration_ms": 1,
    }
    base.update(kw)
    return base


def _env(status="success", **kw):
    base = {
        "meta": _meta(),
        "status": status,
        "summary": "done",
        "confidence": "low" if status in ("failed", "cancelled") else "high",
        "result": {"type": "general", "output": "ok"} if status in ("success", "partial") else None,
        "error": None if status == "success" else {"code": "E", "category": "permanent", "retryable": False, "message": "err"},
    }
    base.update(kw)
    return SubagentResultEnvelope.model_validate(base)


def _make_exec(sequence):
    """Async executor that pops one envelope per call; records the tasks seen."""
    calls = []

    async def exec_fn(task, system_prompt):
        calls.append(task)
        return sequence.pop(0).model_dump()

    exec_fn.calls = calls
    return exec_fn


def _orchestrator(**kw):
    records = []
    orch = Orchestrator(
        blackboard=Blackboard(),
        llm=None,
        experience_store=lambda r: records.append(r),
        **kw,
    )
    orch._exp_records = records
    return orch


# ------------------------------------------------------------- detector
def test_detector_hard_fail_retryable_flag():
    det = DeviationDetector()
    d = det.check(_env("failed"))
    assert d is not None and d.type == DeviationType.HARD_FAIL
    assert d.detail["retryable"] is False

    retry = _env("failed", error={"code": "E", "category": "transient", "retryable": True, "message": "x"})
    assert det.check(retry).detail["retryable"] is True


def test_detector_soft_drift_on_partial_and_low_confidence():
    det = DeviationDetector()
    assert det.check(_env("partial", confidence="medium", completeness_ratio=0.5)).type == DeviationType.SOFT_DRIFT
    assert det.check(_env("success", confidence="low")).type == DeviationType.SOFT_DRIFT


def test_detector_clean_success_no_deviation():
    assert DeviationDetector().check(_env("success")) is None


def test_detector_goal_drift_gated_by_judge():
    det = DeviationDetector(goal_judge=lambda env, goal: goal == "off")
    env = _env("success")
    assert det.check(env, goal="off").type == DeviationType.GOAL_DRIFT
    assert det.check(env, goal="on") is None  # judge says on-track


def test_detector_goal_judge_failure_ignored():
    def bad(env, goal):
        raise RuntimeError("judge down")

    det = DeviationDetector(goal_judge=bad)
    assert det.check(_env("success"), goal="anything") is None  # falls through


# ------------------------------------------------------------- recovery
def test_retryable_hard_fail_retries_once_and_recovers():
    seq = [
        _env("failed", error={"code": "E", "category": "transient", "retryable": True, "message": "x"}),
        _env("success"),
    ]
    ex = _make_exec(seq)
    orch = _orchestrator()
    result = asyncio.run(
        orch.orchestrate(task="t", subtasks=[{"subagent_name": "reviewer", "task": "a", "executor": ex}])
    )
    assert result.results["reviewer"].status == "success"
    assert len(ex.calls) == 2  # original + one retry
    assert orch._exp_records[0]["strategy"] == "hard_fail_retry_recovered"
    assert orch._exp_records[0]["recovered"] is True


def test_retryable_hard_fail_retry_still_fails_then_skips():
    seq = [
        _env("failed", error={"code": "E", "category": "transient", "retryable": True, "message": "x"}),
        _env("failed", error={"code": "E", "category": "transient", "retryable": True, "message": "x"}),
    ]
    ex = _make_exec(seq)
    orch = _orchestrator()
    result = asyncio.run(
        orch.orchestrate(task="t", subtasks=[{"subagent_name": "reviewer", "task": "a", "executor": ex}])
    )
    assert result.results["reviewer"].status == "failed"
    assert len(ex.calls) == 2
    assert orch._exp_records[0]["strategy"] == "hard_fail_retry_failed"
    assert orch._exp_records[0]["recovered"] is False


def test_permanent_hard_fail_skips_without_retry():
    ex = _make_exec([_env("failed")])  # retryable False
    orch = _orchestrator()
    result = asyncio.run(
        orch.orchestrate(task="t", subtasks=[{"subagent_name": "reviewer", "task": "a", "executor": ex}])
    )
    assert result.results["reviewer"].status == "failed"
    assert len(ex.calls) == 1  # no pointless retry
    assert orch._exp_records[0]["strategy"] == "hard_fail_skip"


def test_soft_drift_inserts_compensation_node():
    # review returns partial -> a compensation implementer node runs next and
    # succeeds (same executor), overriding the partial result
    seq = [_env("partial", confidence="medium", completeness_ratio=0.5), _env("success")]
    ex = _make_exec(seq)
    orch = _orchestrator()
    result = asyncio.run(
        orch.orchestrate(task="t", subtasks=[{"subagent_name": "reviewer", "task": "a", "executor": ex}])
    )
    assert len(ex.calls) == 2
    assert "不完整" in ex.calls[1]  # the compensation task references the deviation
    assert result.results["implementer"].status == "success"
    assert orch._exp_records[0]["strategy"] == "soft_drift_insert_fix"


def test_max_replan_rounds_limits_deviation_handling():
    """Only max_replan_rounds deviations are acted on; the rest run as-is."""
    seq = [_env("failed")] * 3  # three permanent failures -> each a deviation
    ex = _make_exec(seq)
    orch = _orchestrator(max_replan_rounds=1)
    asyncio.run(
        orch.orchestrate(
            task="t",
            subtasks=[
                {"subagent_name": "reviewer", "task": f"f{i}", "executor": ex} for i in range(3)
            ],
        )
    )
    assert len(ex.calls) == 3  # all ran once (skip is a no-op action)
    assert len(orch._exp_records) == 1  # only the first deviation was handled


def test_goal_drift_replans_remaining_queue():
    class _Planner:
        async def decompose(self, task, context=None):
            return [SubTask(id="r1", subagent_name="reviewer", instruction="replanned", depends_on=[])]

    orch = Orchestrator(blackboard=Blackboard(), llm=None, planner=_Planner())
    queue = [{"subagent_name": "implementer", "task": "old", "id": "t2", "depends_on": [], "estimated_tokens": 0}]
    dev = Deviation(DeviationType.GOAL_DRIFT, {"goal": "x"})
    action = asyncio.run(
        orch._apply_recovery(dev, {"subagent_name": "reviewer", "task": "a"}, queue, {"task_id": "t"}, {})
    )
    assert action == "goal_drift_replanned"
    assert queue == [
        {"subagent_name": "reviewer", "task": "replanned", "id": "r1", "depends_on": [], "estimated_tokens": 0}
    ]


def test_goal_drift_without_planner_keeps_queue():
    orch = Orchestrator(blackboard=Blackboard(), llm=None, planner=None)
    queue = [{"subagent_name": "implementer", "task": "old", "id": "t2"}]
    dev = Deviation(DeviationType.GOAL_DRIFT, {"goal": "x"})
    action = asyncio.run(
        orch._apply_recovery(dev, {"subagent_name": "reviewer", "task": "a"}, queue, {"task_id": "t"}, {})
    )
    assert action == "goal_drift_no_planner"
    assert len(queue) == 1  # untouched
