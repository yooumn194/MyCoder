"""Tests for Phase 4 LSP symbol intelligence (Module B)."""

from mycoder.mcp.adapter import MCPToolAdapter
from mycoder.mcp.lsp_compressor import LSPResultCompressor, enrich_diagnostic
from mycoder.mcp.lsp_metadata import describe_lsp_tool


def _ref(uri, line):
    return {"uri": uri, "range": {"start": {"line": line, "character": 0}}}


def _diag(uri, line, severity, message):
    return {"uri": uri, "range": {"start": {"line": line, "character": 0}}, "severity": severity, "message": message}


# ---------------------------------------------------------------------------
# compressor
# ---------------------------------------------------------------------------

def test_compress_references_groups_by_file_dedups_lines():
    refs = [
        _ref("file:///workspace/app.py", 5),
        _ref("file:///workspace/app.py", 5),  # duplicate line -> deduped
        _ref("file:///workspace/app.py", 9),
        _ref("file:///workspace/other.py", 1),
    ]
    out = LSPResultCompressor().compress_references(refs)
    assert "app.py: [6, 10]" in out
    assert "other.py: [2]" in out


def test_compress_references_truncates_at_max():
    refs = [_ref(f"file:///workspace/f{i}.py", 0) for i in range(100)]
    out = LSPResultCompressor().compress_references(refs)
    assert "截断，共 100 个引用" in out
    # MAX_REFERENCES=20 files shown
    assert out.count(".py: [1]") == 20


def test_compress_references_ranks_project_over_tests():
    refs = [
        _ref("file:///workspace/test_auth.py", 1),
        _ref("file:///workspace/auth.py", 1),
    ]
    out = LSPResultCompressor().compress_references(refs)
    # project file first, test file last
    assert out.index("auth.py") < out.index("test_auth.py")


def test_compress_diagnostics_groups_and_counts():
    diags = [
        _diag("file:///workspace/a.py", 0, "error", "bad"),
        _diag("file:///workspace/a.py", 1, "warning", "warn"),
        _diag("file:///workspace/b.py", 0, "error", "boom"),
    ]
    out = LSPResultCompressor().compress_diagnostics(diags)
    assert "## file:///workspace/a.py (1 error, 1 warning)" in out
    assert "L1 [error] bad" in out
    assert "boom" in out


def test_compress_diagnostics_caps_per_file():
    diags = [_diag("file:///workspace/a.py", i, "warning", f"w{i}") for i in range(30)]
    out = LSPResultCompressor().compress_diagnostics(diags)
    assert "and 20 more" in out


# ---------------------------------------------------------------------------
# intent metadata
# ---------------------------------------------------------------------------

def test_tool_description_contains_intent():
    desc = describe_lsp_tool("references", "Find references to a symbol.")
    assert "✅ 适用场景" in desc
    assert "重命名前评估影响面" in desc
    assert "❌ 不适用场景" in desc
    assert "比 grep_search 慢但精确" in desc


def test_unknown_tool_unchanged():
    assert describe_lsp_tool("unknown_tool", "base desc") == "base desc"


def test_rename_requires_approval_marked():
    desc = describe_lsp_tool("rename", "Rename a symbol.")
    assert "审批" in desc


def test_adapter_injects_lsp_metadata():
    class _FakeClient:
        async def call_tool(self, name, arguments):
            return {"content": []}

    adapter = MCPToolAdapter(
        _FakeClient(), "lsp",
        {"name": "references", "description": "Find references.", "inputSchema": {"type": "object", "properties": {}}},
    )
    assert "✅ 适用场景" in adapter.description
    assert adapter.name == "mcp_lsp_references"
    # non-lsp servers are untouched
    adapter2 = MCPToolAdapter(
        _FakeClient(), "filesystem",
        {"name": "read_file", "description": "Read a file.", "inputSchema": {"type": "object", "properties": {}}},
    )
    assert adapter2.description == "Read a file."


def test_lsp_server_config_present():
    from mycoder.mcp.config import load_mcp_config

    config = load_mcp_config()
    assert "lsp" in config["servers"]
    assert config["servers"]["lsp"]["enabled"] is False  # opt-in
    assert "lsp" in config["security"]["allowed_tools"]


# ---------------------------------------------------------------------------
# P0-2: diagnostic enrichment
# ---------------------------------------------------------------------------

def test_diagnostic_enrich_python():
    diag = enrich_diagnostic({"message": "undefined name 'foo'"}, "Python")
    assert diag["action_type"] == "import_or_define"
    assert "导入" in diag["fix_suggestion"]


def test_diagnostic_enrich_typescript():
    diag = enrich_diagnostic(
        {"message": "Type 'string' is not assignable to type 'number'"}, "TypeScript"
    )
    assert diag["action_type"] == "type_fix"
    assert "类型转换" in diag["fix_suggestion"]
    # go + unused variable
    go = enrich_diagnostic({"message": "x declared and not used"}, "Go")
    assert go["action_type"] == "remove_or_ignore"


def test_diagnostic_fallback_suggestion():
    diag = enrich_diagnostic({"message": "some weird custom error"}, "Python")
    assert diag["action_type"] == "manual"
    assert "请根据错误消息" in diag["fix_suggestion"]


def test_compress_diagnostics_includes_fix():
    out = LSPResultCompressor().compress_diagnostics(
        [{"uri": "file:///workspace/a.ts", "range": {"start": {"line": 41}}, "severity": "error",
          "message": "Type 'string' is not assignable to type 'number'"}],
        language="TypeScript",
    )
    assert "修复建议" in out
    assert "type_fix" in out
