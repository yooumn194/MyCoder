"""Phase 4 end-to-end: Orchestrator + Blackboard + subagents sharing state."""

from mycoder.agents import Blackboard, BUILTIN_SUBAGENTS, OrchestrationStrategy, Orchestrator, SubagentRunner


async def test_phase4_end_to_end():
    """Explore -> write to Blackboard -> implement reads it -> both succeed."""
    bb = Blackboard()
    orch = Orchestrator(blackboard=bb, llm=None)

    async def explorer_exec(task, system_prompt):
        await bb.put("task-1", "discovery:foo", {"line": 42})
        return {
            "status": "success", "summary": "found foo at line 42", "confidence": "high",
            "result": {"type": "code_exploration", "files_found": [], "patterns_searched": ["foo"], "total_matches": 1},
            "meta": {"task_id": "task-1", "subagent_name": "explorer"},
        }

    async def implementer_exec(task, system_prompt):
        found = await bb.get("task-1", "discovery:foo")
        return {
            "status": "success",
            "summary": f"implemented using discovery at line {found['line']}",
            "confidence": "high",
            "result": {"type": "code_generation", "files": [{"path": "a.py", "action": "modified"}]},
            "meta": {"task_id": "task-1", "subagent_name": "implementer"},
        }

    result = await orch.orchestrate(
        "implement foo", OrchestrationStrategy.SEQUENTIAL,
        parent_context={"task_id": "task-1"},
        subtasks=[
            {"subagent_name": "explorer", "task": "find foo", "executor": explorer_exec},
            {"subagent_name": "implementer", "task": "implement", "executor": implementer_exec},
        ],
    )
    assert result.success
    assert result.results["explorer"].status == "success"
    # the implementer read the explorer's blackboard entry
    assert "line 42" in result.results["implementer"].summary
    # instance ids injected by the orchestrator
    assert result.results["explorer"].meta.subagent_instance_id
    assert result.results["implementer"].meta.subagent_instance_id


async def test_blackboard_shares_discovery_between_runners():
    bb = Blackboard()
    orch = Orchestrator(blackboard=bb, llm=None)

    async def writer_exec(task, system_prompt):
        await bb.put("task-2", "symbols:auth", ["handler.py", "middleware.py"])
        return {
            "status": "success", "summary": "found auth symbols", "confidence": "high",
            "result": {"type": "code_exploration", "files_found": [], "patterns_searched": ["auth"], "total_matches": 2},
            "meta": {"task_id": "task-2", "subagent_name": "explorer"},
        }

    async def reader_exec(task, system_prompt):
        symbols = await bb.query("task-2", "symbols:")
        return {
            "status": "success",
            "summary": f"read {len(symbols)} symbol groups from blackboard",
            "confidence": "high",
            "result": {"type": "code_generation", "files": []},
            "meta": {"task_id": "task-2", "subagent_name": "implementer"},
        }

    result = await orch.orchestrate(
        "t", OrchestrationStrategy.SEQUENTIAL, parent_context={"task_id": "task-2"},
        subtasks=[
            {"subagent_name": "explorer", "task": "write", "executor": writer_exec},
            {"subagent_name": "implementer", "task": "read", "executor": reader_exec},
        ],
    )
    assert "1 symbol groups" in result.results["implementer"].summary


async def test_subagent_instance_id_injected_into_envelope():
    """A subagent that omits the instance id still gets a valid envelope (RFC)."""
    async def _exec(task, system_prompt):
        return {
            "status": "success", "summary": "done", "confidence": "high",
            "result": {"type": "general", "output": "ok"},
            "meta": {"task_id": "task-3", "subagent_name": "explorer"},  # no instance_id
        }

    runner = SubagentRunner(
        BUILTIN_SUBAGENTS["explorer"], "t", orchestrator=None,
        parent_context={"task_id": "task-3"}, instance_id="22222222-2222-2222-2222-222222222222",
        executor=_exec,
    )
    env = await runner.run()
    assert env.status == "success"
    assert env.meta.subagent_instance_id == "22222222-2222-2222-2222-222222222222"
