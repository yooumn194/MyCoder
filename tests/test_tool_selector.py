"""P0 dynamic tool selection (tools/selector.py)."""

from mycoder.tools import ALL_TOOLS
from mycoder.tools.selector import CORE_TOOLS, ToolSelector


def test_core_tools_always_included():
    sel = ToolSelector(top_n=12)
    picked = {t.name for t in sel.select("任何查询", ALL_TOOLS)}
    assert CORE_TOOLS <= picked


def test_relevant_tools_ranked_above_unrelated():
    sel = ToolSelector(top_n=10)
    names = [t.name for t in sel.select("search memory retrieve", ALL_TOOLS)]
    assert "memory_search" in names
    # memory_search surfaces for a memory query, ahead of the generic tail
    assert names.index("memory_search") < names.index("grep")


def test_top_n_caps_injected_set():
    """A tight top_n still keeps the whole core set plus reserve slots for the
    remainder — core is never starved by the budget."""
    sel = ToolSelector(top_n=8)
    picked = sel.select("edit write files", ALL_TOOLS)
    assert CORE_TOOLS <= {t.name for t in picked}
    # core (11) + reserve (4 default) — the budget can't shrink below that
    assert len(picked) <= len(CORE_TOOLS) + 4


def test_core_includes_planning_pair():
    """todo_write/todo_update must travel together — splitting them lets
    planning_guard deadlock the plan flow."""
    assert "todo_write" in CORE_TOOLS
    assert "todo_update" in CORE_TOOLS


def test_additional_include_keeps_enabled_mcp_tools():
    """MCP tools the operator enabled (additional_include) are never dropped
    by relevance ranking."""

    class _MCPTool:
        name = "mcp_github_search"
        description = "Search GitHub via MCP"
        parameters = {"type": "object", "properties": {}}

    tools = [*ALL_TOOLS, _MCPTool()]
    sel = ToolSelector(additional_include={"mcp_github_search"})
    picked = {t.name for t in sel.select("写一个普通函数", tools)}
    assert CORE_TOOLS <= picked
    assert "mcp_github_search" in picked  # always present, not ranked out


def test_selector_preserves_core_first():
    sel = ToolSelector(top_n=20)
    names = [t.name for t in sel.select("", ALL_TOOLS)]
    # empty query -> all tools, but core stays at the front, in tool order
    expected_core = [t.name for t in ALL_TOOLS if t.name in CORE_TOOLS]
    assert names[: len(expected_core)] == expected_core
