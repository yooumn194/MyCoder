"""P0 dynamic tool selection (tools/selector.py)."""

from corecoder.tools import ALL_TOOLS
from corecoder.tools.selector import CORE_TOOLS, ToolSelector


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
    sel = ToolSelector(top_n=8)
    picked = sel.select("edit write files", ALL_TOOLS)
    assert len(picked) <= 8
    # core still present even under a tight budget
    assert CORE_TOOLS <= {t.name for t in picked}


def test_selector_preserves_core_first():
    sel = ToolSelector(top_n=20)
    names = [t.name for t in sel.select("", ALL_TOOLS)]
    # empty query -> all tools, but core stays at the front, in tool order
    expected_core = [t.name for t in ALL_TOOLS if t.name in CORE_TOOLS]
    assert names[: len(expected_core)] == expected_core
