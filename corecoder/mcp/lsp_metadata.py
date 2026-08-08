"""Intent-aware metadata for LSP tools (cognitive alignment).

The core LLM pain the LSP integration solves is "does the model know whether
to reach for grep or for the symbol-aware LSP tool?". A description that says
"return symbol references" does not help; one that says "use BEFORE renaming
to assess the blast radius" does. This metadata is injected into each lsp_*
tool's description by MCPToolAdapter.
"""

# tool name (without the `mcp_lsp_` prefix) -> intent metadata
LSP_TOOL_METADATA: dict[str, dict] = {
    "definition": {
        "when_to_use": [
            "精确跳转到符号定义",
            "理解类型来源",
            "查看接口实现",
        ],
        "when_not_to_use": [
            "搜索普通字符串",
            "查找文件路径",
            "搜索配置项",
        ],
        "note": "返回精确位置（文件+行号），比 grep 更可靠",
        "example": "definition(symbol='authenticate_user')",
    },
    "references": {
        "when_to_use": [
            "重命名前评估影响面",
            "理解函数调用链",
            "排查未使用代码",
        ],
        "when_not_to_use": [
            "模糊搜索文件名",
            "查找配置项",
            "搜索注释内容",
        ],
        "note": "比 grep_search 慢但精确，仅在需要语义理解时使用",
        "example": "references(symbol='authenticate_user')",
    },
    "rename": {
        "when_to_use": ["安全重命名符号", "跨文件重命名"],
        "when_not_to_use": ["普通文本替换", "修改配置变量"],
        "note": "LSP 会分析所有引用并自动更新，比 sed 安全",
        "requires_approval": True,
        "example": "rename(symbol='old_name', new_name='new_name')",
    },
    "diagnostics": {
        "when_to_use": ["验证修改后代码是否引入错误", "了解当前文件问题"],
        "when_not_to_use": ["全局质量审计（用 review subagent）"],
        "note": "修改文件后调用，比重新编译更快反馈",
        "example": "diagnostics(uri='file:///workspace/app.py')",
    },
    "symbols": {
        "when_to_use": ["获取文件/workspace 的符号大纲"],
        "when_not_to_use": ["全文搜索关键词"],
        "note": "一次性给出结构概览，适合了解文件布局",
        "example": "symbols(query='auth')",
    },
}


def describe_lsp_tool(tool_name: str, base_description: str) -> str:
    """Augment an lsp_* tool's description with intent metadata.

    tool_name is the mcp tool name WITHOUT the mcp_lsp_ prefix (e.g.
    "references"). Returns the description with ✅/❌ scenario markers.
    """
    meta = LSP_TOOL_METADATA.get(tool_name)
    if meta is None:
        return base_description
    parts = [base_description.rstrip("."), ""]
    if meta.get("when_to_use"):
        parts.append("✅ 适用场景:")
        parts.extend(f"  - {s}" for s in meta["when_to_use"])
    if meta.get("when_not_to_use"):
        parts.append("❌ 不适用场景:")
        parts.extend(f"  - {s}" for s in meta["when_not_to_use"])
    if meta.get("note"):
        parts.append(f"⚠️  {meta['note']}")
    if meta.get("requires_approval"):
        parts.append("⚠️  此操作需审批（修改符号引用）")
    if meta.get("example"):
        parts.append(f"示例: {meta['example']}")
    return "\n".join(parts)
