import json

import pytest

from mycoder.tool_protocol import (
    ToolProtocolAdapter,
    provider_tool_capabilities,
    resolve_tool_dialect,
)


def _schema(name, properties, required):
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "test",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
    ]


def test_deepseek_flash_uses_zcode_dialect_by_default(monkeypatch):
    monkeypatch.delenv("MYCODER_TOOL_DIALECT", raising=False)
    assert resolve_tool_dialect("deepseek-flash") == "zcode"
    assert resolve_tool_dialect("gpt-5.5") == "native"


def test_explicit_native_dialect_overrides_model(monkeypatch):
    monkeypatch.setenv("MYCODER_TOOL_DIALECT", "native")
    assert resolve_tool_dialect("deepseek-flash") == "native"
    assert ToolProtocolAdapter.for_model("deepseek-flash").dialect == "native"


def test_provider_capabilities_are_not_global_model_switches():
    assert provider_tool_capabilities("openrouter", "gpt-4o").dialect == "native"
    assert provider_tool_capabilities("openrouter", "deepseek-flash").dialect == "zcode"
    assert provider_tool_capabilities("deepseek", "gpt-4o").provider == "deepseek"


def test_deepseek_thinking_disables_requirement_tool_choice():
    capabilities = provider_tool_capabilities("deepseek", "deepseek-flash")
    assert capabilities.supports_tool_choice is False
    assert capabilities.supports_named_tool_choice is False


def test_invalid_tool_dialect_fails_closed():
    with pytest.raises(ValueError, match="tool dialect"):
        resolve_tool_dialect("model", "unknown")


def test_zcode_schema_and_arguments_round_trip_without_mutating_input():
    protocol = ToolProtocolAdapter("zcode")
    schemas = _schema(
        "list_files",
        {"glob_pattern": {"type": "string"}, "path": {"type": "string"}},
        ["glob_pattern"],
    )

    converted = protocol.tools_to_wire(schemas)

    assert schemas[0]["function"]["name"] == "list_files"
    assert converted[0]["function"]["name"] == "Glob"
    parameters = converted[0]["function"]["parameters"]
    assert set(parameters["properties"]) == {"pattern", "path"}
    assert parameters["required"] == ["pattern"]
    assert protocol.from_wire_name("Glob") == "list_files"
    assert protocol.from_wire_arguments("Glob", {"pattern": "**/*.py"}) == {
        "glob_pattern": "**/*.py"
    }


def test_zcode_history_is_encoded_only_at_provider_boundary():
    protocol = ToolProtocolAdapter("zcode")
    messages = [
        {"role": "system", "content": "Use read_file then edit_file."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "list_files",
                        "arguments": json.dumps({"glob_pattern": "**/*.py"}),
                    },
                }
            ],
        },
    ]

    converted = protocol.messages_to_wire(messages)

    assert converted[0]["content"] == "Use Read then Edit."
    function = converted[1]["tool_calls"][0]["function"]
    assert function["name"] == "Glob"
    assert json.loads(function["arguments"]) == {"pattern": "**/*.py"}
    assert messages[1]["tool_calls"][0]["function"]["name"] == "list_files"


def test_llm_encodes_zcode_request_and_decodes_response():
    from mycoder.llm import LLM

    class _Value:
        def __init__(self, **values):
            self.__dict__.update(values)

    llm = LLM(model="deepseek-flash", api_key="test", provider="openrouter")
    captured = {}

    def fake_stream(params):
        captured.update(params)
        yield _Value(
            usage=None,
            choices=[
                _Value(
                    delta=_Value(
                        content=None,
                        tool_calls=[
                            _Value(
                                index=0,
                                id="call-1",
                                function=_Value(
                                    name="Glob",
                                    arguments='{"pattern":"**/*.py"}',
                                ),
                            )
                        ],
                        model_extra={},
                    )
                )
            ],
        )

    llm._call_with_retry = fake_stream
    response = llm.chat(
        [{"role": "system", "content": "Use list_files."}],
        tools=_schema(
            "list_files",
            {"glob_pattern": {"type": "string"}},
            ["glob_pattern"],
        ),
    )

    assert captured["tools"][0]["function"]["name"] == "Glob"
    assert captured["messages"][0]["content"] == "Use Glob."
    assert response.tool_calls[0].name == "list_files"
    assert response.tool_calls[0].arguments == {"glob_pattern": "**/*.py"}


def test_deepseek_strict_choice_retries_with_thinking_disabled(monkeypatch):
    """Strict mutation calls stay strict even though DeepSeek thinking rejects
    the first request shape.

    The compatibility retry is deliberately tested at the provider boundary so
    a future refactor cannot silently turn this into an ``auto`` request.
    """
    from types import SimpleNamespace

    import httpx
    from openai import BadRequestError

    from mycoder.llm import LLM

    llm = LLM(model="deepseek-flash", api_key="test", provider="deepseek")
    captured = []
    bad_response = httpx.Response(400, request=httpx.Request("POST", "http://test"))

    def fake_stream(params):
        captured.append(params.copy())
        if len(captured) < 3:
            raise BadRequestError("thinking tool_choice rejected", response=bad_response, body={})
        return iter([
            SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call-1",
                                function=SimpleNamespace(
                                    name="Edit",
                                    arguments='{"file_path":"x.py","old_string":"a","new_string":"b"}',
                                ),
                            )
                        ],
                        model_extra={},
                    )
                )
            ],
            )
        ])

    llm._call_with_retry = fake_stream
    response = llm.chat(
        [{"role": "user", "content": "edit x.py"}],
        tools=_schema("edit_file", {}, []),
        tool_choice={"type": "function", "function": {"name": "edit_file"}},
        strict_tool_choice=True,
    )

    assert response.tool_calls[0].name == "edit_file"
    assert captured[-1]["extra_body"]["thinking"] == {"type": "disabled"}
    assert captured[-1]["tool_choice"]["function"]["name"] == "Edit"
    assert "tool_choice" in captured[-1]


def test_truncated_tool_arguments_are_not_silently_treated_as_empty():
    from mycoder.llm import LLM

    class _Value:
        def __init__(self, **values):
            self.__dict__.update(values)

    llm = LLM(model="deepseek-flash", api_key="test")

    def fake_stream(_params):
        yield _Value(
            usage=None,
            choices=[
                _Value(
                    delta=_Value(
                        content=None,
                        tool_calls=[
                            _Value(
                                index=0,
                                id="call-1",
                                function=_Value(name="Edit", arguments='{"file_path":'),
                            )
                        ],
                        model_extra={},
                    )
                )
            ],
        )

    llm._call_with_retry = fake_stream
    response = llm.chat(
        [{"role": "user", "content": "edit"}],
        tools=_schema("edit_file", {}, []),
    )

    assert response.tool_calls[0].name == "edit_file"
    assert response.tool_calls[0].arguments == {}
    assert "not valid JSON" in response.tool_calls[0].parse_error
