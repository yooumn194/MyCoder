import asyncio

from mycoder.agents.planner import TaskPlanner
from mycoder.llm import LLM, LLMResponse


def test_task_planner_uses_provider_timeout_without_nested_transport_retries():
    class _ManagedLLM(LLM):
        def __init__(self):
            self.kwargs = None

        def chat(self, _messages, **kwargs):
            self.kwargs = kwargs
            return LLMResponse(
                content=(
                    '[{"id":"t1","subagent_name":"implementer",'
                    '"instruction":"fix it","depends_on":[],"estimated_tokens":1000}]'
                )
            )

    llm = _ManagedLLM()
    planner = TaskPlanner(llm=llm, timeout_seconds=7.5)

    result = asyncio.run(planner.decompose("fix it", {"task_id": "t"}))

    assert len(result) == 1
    assert llm.kwargs["timeout_seconds"] == 7.5
    assert llm.kwargs["request_max_retries"] == 1
    assert llm.kwargs["response_format"] == {"type": "json_object"}


def test_custom_planner_adapter_keeps_waiter_timeout_compatibility():
    class _SlowCustomLLM:
        @staticmethod
        def chat(_messages, **_kwargs):
            import time

            time.sleep(0.05)
            return "[]"

    planner = TaskPlanner(llm=_SlowCustomLLM(), timeout_seconds=0.005)

    result = asyncio.run(planner.decompose("fix it", {"task_id": "t"}))

    assert len(result) == 1
    assert result[0].subagent_name == "implementer"
