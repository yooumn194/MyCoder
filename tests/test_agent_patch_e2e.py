"""Offline proof that the production patch pipeline can read, edit and verify."""

from mycoder.agents import Blackboard, OrchestrationStrategy, Orchestrator
from mycoder.llm import LLMResponse, ToolCall
from mycoder.tools.base import Tool, ToolResult
from mycoder.tools.edit import EditFileTool
from mycoder.tools.read_file import ReadFileTool
from mycoder.tools.write import WriteFileTool


class _ExecuteCheckTool(Tool):
    name = "execute_in_sandbox"
    idempotent = False
    description = "Run a focused check"
    parameters = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }

    def execute(self, command: str) -> str:
        assert "python -c" in command
        assert "assert" in command
        # The real sandbox tool reports how the process ended; the verdict is
        # read from this field, not from the command string.
        return ToolResult("behavior assertion passed", exit_code=0)


class _PatchLLM:
    max_tokens = 256

    def __init__(self) -> None:
        self.implementer_calls = 0
        self.verifier_calls = 0

    def chat(self, *, messages, tools, **_kwargs):
        prompt = "\n".join(str(message.get("content") or "") for message in messages)
        if "You are a verifier subagent" in prompt:
            self.verifier_calls += 1
            if self.verifier_calls == 1:
                return LLMResponse(
                    tool_calls=[
                        ToolCall(
                            id="verify",
                            name="execute_in_sandbox",
                            arguments={
                                "command": (
                                    "python -c \"from module import VALUE; "
                                    "assert VALUE == 2\""
                                )
                            },
                        )
                    ]
                )
            return LLMResponse(content="focused test passed")

        self.implementer_calls += 1
        if self.implementer_calls == 1:
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="read",
                        name="read_file",
                        arguments={"file_path": "module.py"},
                    )
                ]
            )
        if self.implementer_calls == 2:
            names = {schema["function"]["name"] for schema in tools}
            assert "edit_file" in names
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="edit",
                        name="edit_file",
                        arguments={
                            "file_path": "module.py",
                            "old_string": "VALUE = 1",
                            "new_string": "VALUE = 2",
                        },
                    )
                ]
            )
        return LLMResponse(content="minimal patch applied")


class _DiffManager:
    def __init__(self, path) -> None:
        self.path = path

    async def get_diff(self) -> str:
        if self.path.read_text(encoding="utf-8") == "VALUE = 2\n":
            return "diff --git a/module.py b/module.py\n-VALUE = 1\n+VALUE = 2\n"
        return "(no changes)"


async def test_patch_pipeline_reads_atomic_edits_and_requires_verification(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MYCODER_INJECTION_GUARD", "off")
    target = tmp_path / "module.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    llm = _PatchLLM()
    orchestrator = Orchestrator(
        blackboard=Blackboard(),
        llm=llm,
        tools=[
            ReadFileTool(project_root=tmp_path),
            EditFileTool(project_root=tmp_path),
            WriteFileTool(project_root=tmp_path),
            _ExecuteCheckTool(),
        ],
    )
    orchestrator._sandbox_manager = _DiffManager(target)

    outcome = await orchestrator.orchestrate(
        "Change module.py VALUE from 1 to 2",
        OrchestrationStrategy.SEQUENTIAL,
        parent_context={
            "task_id": "offline-e2e",
            "session_id": "offline-e2e",
            "require_patch": True,
        },
        subtasks=[
            {
                "id": "implement",
                "subagent_name": "implementer",
                "task": "Change module.py VALUE from 1 to 2",
            }
        ],
    )
    assert outcome.success is True
    assert target.read_text(encoding="utf-8") == "VALUE = 2\n"
    assert outcome.results["implementer"].status == "success"
    assert outcome.results["verifier"].status == "success"
    assert llm.implementer_calls == 3
    assert llm.verifier_calls == 2
