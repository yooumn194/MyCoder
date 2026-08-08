"""Tests for P0-1: subagent tool output contract validation."""

from corecoder.agents.tool_validator import ToolOutputValidator


def test_validator_passes_wellformed_grep():
    v = ToolOutputValidator()
    ok, err = v.validate("grep_search", {"matches": [], "total_count": 0})
    assert ok and err is None


def test_subagent_tool_validation_fails():
    v = ToolOutputValidator()
    ok, err = v.validate("grep_search", {"total_count": 3})  # missing matches
    assert not ok
    assert "matches" in err


def test_validator_requires_correct_types():
    v = ToolOutputValidator()
    assert not v.validate("grep_search", {"matches": "not-a-list", "total_count": 1})[0]
    assert not v.validate("read_file", {"content": "x", "line_count": 9999})[0]  # > 300
    assert not v.validate("list_files", {"files": ["a"] * 1001})[0]  # > 1000


def test_validator_unknown_tool_passes():
    v = ToolOutputValidator()
    assert v.validate("some_custom_tool", "anything") == (True, None)


async def test_subagent_tool_validation_retry_succeeds():
    """First call returns malformed output, the retry fixes it -> success."""
    v = ToolOutputValidator()
    calls = []

    async def _call(tool_name, params):
        calls.append(params)
        if len(calls) == 1:
            return {"total_count": 5}  # missing matches -> invalid
        return {"matches": [], "total_count": 5}

    output, success = await v.validate_and_retry("grep_search", {"pattern": "x"}, _call)
    assert success
    assert output["matches"] == []
    assert len(calls) == 2
    # the retry carried the validation error as correction context
    assert "_validation_error" in calls[1]


async def test_subagent_tool_validation_gives_up_gracefully():
    """Invalid output after max retries -> returns it, no crash."""
    v = ToolOutputValidator()

    async def _call(tool_name, params):
        return {"total_count": 5}  # always missing matches

    output, success = await v.validate_and_retry("grep_search", {"pattern": "x"}, _call)
    assert not success
    assert output == {"total_count": 5}  # gracefully returned the last output
