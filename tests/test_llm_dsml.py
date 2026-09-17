"""Compatibility tests for provider-specific textual tool calls."""



from mycoder.llm import LLM, _parse_dsml_tool_calls


def test_parse_dsml_tool_calls_only_allows_declared_tools():
    content = """Analysis first.
<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="edit_file">
<｜｜DSML｜｜ parameter name="file_path" string="true">module.py</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="line" string="false">12</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
<｜｜DSML｜｜ invoke name="undeclared">
<｜｜DSML｜｜ parameter name="cmd" string="true">bad</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
</｜｜DSML｜｜ calls>"""

    calls = _parse_dsml_tool_calls(content, {"edit_file"})

    assert len(calls) == 1
    assert calls[0].name == "edit_file"
    assert calls[0].arguments == {"file_path": "module.py", "line": 12}


def test_parse_dsml_rejects_invalid_non_string_value():
    content = """<｜｜DSML｜｜ invoke name="read_file">
<｜｜DSML｜｜ parameter name="start_line" string="false">not-json</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>"""

    assert _parse_dsml_tool_calls(content, {"read_file"}) == []


def test_parse_dsml_allows_declared_zero_argument_tool():
    content = '<||DSML|| invoke name="list_files"></||DSML|| invoke>'

    calls = _parse_dsml_tool_calls(content, {"list_files"})

    assert len(calls) == 1
    assert calls[0].arguments == {}


def test_parse_dsml_flushes_a_complete_invocation_cut_before_its_close_tag():
    """P0-3: hitting max_tokens after the last parameter must not look like
    "the model produced no tool call" — every argument is already there."""
    content = (
        'Analysis.\n<｜｜DSML｜｜ invoke name="edit_file">\n'
        '<｜｜DSML｜｜ parameter name="file_path" string="true">module.py</｜｜DSML｜｜ parameter>\n'
        '<｜｜DSML｜｜ parameter name="old_string" string="true">VALUE = 1</｜｜DSML｜｜ parameter>\n'
        '<｜｜DSML｜｜ parameter name="new_string" string="true">VALUE = 2</｜｜DSML｜｜ parameter>\n'
    )

    calls = _parse_dsml_tool_calls(content, {"edit_file"})

    assert len(calls) == 1
    assert calls[0].name == "edit_file"
    assert calls[0].arguments == {
        "file_path": "module.py",
        "old_string": "VALUE = 1",
        "new_string": "VALUE = 2",
    }
    assert calls[0].parse_error is None  # usable as-is


def test_parse_dsml_flushes_a_half_written_invocation_with_a_parse_error():
    """A parameter cut in half is still surfaced, flagged as incomplete, so the
    tool layer answers INVALID_TOOL_INPUT and the model re-emits it."""
    content = (
        '<｜｜DSML｜｜ invoke name="write_file">\n'
        '<｜｜DSML｜｜ parameter name="file_path" string="true">module.py</｜｜DSML｜｜ parameter>\n'
        '<｜｜DSML｜｜ parameter name="content" string="true">VALUE = 2\n'
    )

    calls = _parse_dsml_tool_calls(content, {"write_file"})

    assert len(calls) == 1
    assert calls[0].name == "write_file"
    assert calls[0].arguments == {"file_path": "module.py"}
    assert calls[0].parse_error and "truncated" in calls[0].parse_error


def test_parse_dsml_keeps_closed_calls_and_appends_the_truncated_tail():
    """A completed call plus a truncated one both survive."""
    content = (
        '<｜｜DSML｜｜ invoke name="read_file">\n'
        '<｜｜DSML｜｜ parameter name="file_path" string="true">a.py</｜｜DSML｜｜ parameter>\n'
        "</｜｜DSML｜｜ invoke>\n"
        '<｜｜DSML｜｜ invoke name="edit_file">\n'
        '<｜｜DSML｜｜ parameter name="file_path" string="true">a.py</｜｜DSML｜｜ parameter>\n'
    )

    calls = _parse_dsml_tool_calls(content, {"read_file", "edit_file"})

    assert [call.name for call in calls] == ["read_file", "edit_file"]
    assert calls[1].parse_error is None


def test_parse_dsml_ignores_undeclared_truncated_invocations():
    content = '<｜｜DSML｜｜ invoke name="undeclared">\n<｜｜DSML｜｜ parameter name="x" string="true">1</｜｜DSML｜｜ parameter>\n'

    assert _parse_dsml_tool_calls(content, {"edit_file"}) == []


def test_parse_dsml_surfaces_a_header_that_was_cut_off_before_any_parameter():
    """A stream that dies right after the tool name must not read as "no call".

    Reporting nothing here is indistinguishable from the model choosing not to
    act: the turn ends with ``tool_calls=0`` and the work never happens. The
    call comes through with empty arguments and a parse error, which the tool
    layer turns into INVALID_TOOL_INPUT so the model re-emits it in full.
    """
    content = '<｜｜DSML｜｜ invoke name="edit_file">\n'

    calls = _parse_dsml_tool_calls(content, {"edit_file"})

    assert len(calls) == 1
    assert calls[0].name == "edit_file"
    assert calls[0].arguments == {}
    assert calls[0].parse_error and "cut off" in calls[0].parse_error


def test_parse_dsml_ignores_a_bare_header_for_an_undeclared_tool():
    """The allowlist still applies to the truncated tail."""
    content = '<｜｜DSML｜｜ invoke name="undeclared">\n'

    assert _parse_dsml_tool_calls(content, {"edit_file"}) == []


def test_parse_dsml_gives_no_parse_error_to_a_closed_zero_argument_invoke():
    """A *closed* invoke with no parameters parsed cleanly, so it carries no
    parse error: the tool layer reports the missing argument on its own."""
    content = (
        '<｜｜DSML｜｜ invoke name="edit_file">\n'
        '</｜｜DSML｜｜ invoke>'
    )

    calls = _parse_dsml_tool_calls(content, {"edit_file"})

    assert len(calls) == 1
    assert calls[0].name == "edit_file"
    assert calls[0].arguments == {}
    assert calls[0].parse_error is None


def _dsml_chat(dsml: str, tool_names: list[str]):
    """Drive ``LLM.chat`` with a DeepSeek-shaped DSML reasoning stream.

    The parser tests above exercise the wire format directly.  This helper
    covers the hand-off from the parsed call to the public ``ToolCall`` object,
    which is only observable through ``chat``.
    """

    class _F:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    def fake_stream(_params):
        yield _F(
            choices=[
                _F(
                    delta=_F(
                        content="I will apply the change.",
                        tool_calls=None,
                        model_extra={"reasoning_content": dsml},
                    )
                )
            ],
            usage=None,
        )

    llm = LLM(model="deepseek-flash", api_key="k", provider="deepseek")
    llm._call_with_retry = fake_stream
    return llm.chat(
        [{"role": "user", "content": "fix it"}],
        tools=[
            {"type": "function", "function": {"name": name}} for name in tool_names
        ],
    )


def test_chat_surfaces_the_parse_error_of_a_truncated_dsml_invocation():
    """P0-3 regression: a half-written parameter is surfaced through ``chat``
    with its parse error intact, so the agent can ask the model to re-emit it.
    Losing the flag here silently downgrades the retry path to a generic
    "missing required argument" error."""
    dsml = (
        '<｜｜DSML｜｜ invoke name="write_file">\n'
        '<｜｜DSML｜｜ parameter name="file_path" string="true">module.py</｜｜DSML｜｜ parameter>\n'
        '<｜｜DSML｜｜ parameter name="content" string="true">VALUE = 2\n'
    )

    response = _dsml_chat(dsml, ["write_file"])

    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.name == "write_file"
    assert call.arguments == {"file_path": "module.py"}
    assert call.parse_error and "truncated" in call.parse_error


def test_chat_keeps_a_complete_unclosed_dsml_invocation_usable():
    """The other half of the same fix: when only the closing tag is missing the
    call is complete, so it must arrive usable, with no error flag."""
    dsml = (
        '<｜｜DSML｜｜ invoke name="edit_file">\n'
        '<｜｜DSML｜｜ parameter name="file_path" string="true">module.py</｜｜DSML｜｜ parameter>\n'
        '<｜｜DSML｜｜ parameter name="old_string" string="true">VALUE = 1</｜｜DSML｜｜ parameter>\n'
        '<｜｜DSML｜｜ parameter name="new_string" string="true">VALUE = 2</｜｜DSML｜｜ parameter>\n'
    )

    response = _dsml_chat(dsml, ["edit_file"])

    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.name == "edit_file"
    assert call.arguments == {
        "file_path": "module.py",
        "old_string": "VALUE = 1",
        "new_string": "VALUE = 2",
    }
    assert call.parse_error is None
