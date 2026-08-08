"""LSPResultCompressor — turn raw LSP responses into LLM-friendly text.

Strategy: dedup -> sort -> truncate -> paginate. References are grouped by
file, deduped by line, ranked (project files first, tests last), capped at
MAX_REFERENCES. Diagnostics are grouped by file and severity, and enriched
(P0-2) with an actionable fix_suggestion + action_type so the agent doesn't
have to re-derive how to fix each error from its message alone.
"""

import re
from collections import Counter

# P0-2: diagnostic message pattern -> actionable fix (extensible; Python/TS/Go).
DIAGNOSTIC_FIX_MAP: dict[str, dict[str, dict[str, str]]] = {
    "Python": {
        "undefined name": {
            "suggestion": "检查是否已导入该模块或变量，或检查拼写错误",
            "action_type": "import_or_define",
        },
        "unused import": {
            "suggestion": "移除该 import 或添加 # noqa 注释",
            "action_type": "remove_or_ignore",
        },
        "indentation error": {
            "suggestion": "检查缩进层级是否一致，确保不要混合使用空格和 Tab",
            "action_type": "fix_indentation",
        },
        "^E[0-9]+$": {
            "suggestion": "检查语法错误，参考错误消息定位问题",
            "action_type": "syntax_fix",
        },
    },
    "TypeScript": {
        "is not assignable": {
            "suggestion": "检查类型定义，添加类型转换或修正接口声明",
            "action_type": "type_fix",
        },
        "Cannot find module": {
            "suggestion": "安装对应类型包 (@types/...) 或修正导入路径",
            "action_type": "install_types",
        },
        "no overload matches": {
            "suggestion": "检查参数类型是否匹配函数签名",
            "action_type": "param_fix",
        },
    },
    "Go": {
        "cannot use .* as .* in": {
            "suggestion": "检查类型转换，或调整函数签名以匹配",
            "action_type": "type_conversion",
        },
        "declared and not used|unused variable": {
            "suggestion": "移除未使用的变量，或使用 _ 忽略",
            "action_type": "remove_or_ignore",
        },
    },
}

_FALLBACK_FIX = {"suggestion": "请根据错误消息定位并修复问题", "action_type": "manual"}


def enrich_diagnostic(diagnostic: dict, language: str) -> dict:
    """Add fix_suggestion + action_type to a diagnostic for a language."""
    if "fix_suggestion" in diagnostic:
        return diagnostic
    message = diagnostic.get("message", "")
    fix = None
    for pattern, fix_info in DIAGNOSTIC_FIX_MAP.get(language, {}).items():
        if re.search(pattern, message, re.IGNORECASE):
            fix = fix_info
            break
    diagnostic["fix_suggestion"] = (fix or _FALLBACK_FIX)["suggestion"]
    diagnostic["action_type"] = (fix or _FALLBACK_FIX)["action_type"]
    return diagnostic


class LSPResultCompressor:
    MAX_REFERENCES = 20
    MAX_DIAGNOSTICS_PER_FILE = 10

    # ----------------------------------------------------------- references

    def compress_references(self, references: list[dict]) -> str:
        by_file: dict[str, list[int]] = {}
        for ref in references:
            uri = ref.get("uri", "unknown")
            line = ref.get("range", {}).get("start", {}).get("line", 0)
            by_file.setdefault(uri, []).append(int(line) + 1)

        ranked = self._rank_files(by_file)
        truncated = len(ranked) > self.MAX_REFERENCES
        if truncated:
            ranked = ranked[: self.MAX_REFERENCES]

        output = []
        for file_path, lines in ranked:
            unique = sorted(set(lines))
            output.append(f"{file_path}: [{', '.join(map(str, unique))}]")
        if truncated:
            output.append(f"... (截断，共 {len(references)} 个引用)")
        return "\n".join(output) if output else "(no references)"

    @staticmethod
    def _rank_files(by_file: dict[str, list[int]]) -> list[tuple[str, list[int]]]:
        """Rank: project files first, test files last, then by match count."""
        def rank(item):
            path = item[0]
            if "/test" in path or path.startswith("test"):
                return (2, -len(item[1]), path)
            if "node_modules" in path or ".venv" in path:
                return (3, -len(item[1]), path)
            return (0, -len(item[1]), path)

        return sorted(by_file.items(), key=rank)

    # ---------------------------------------------------------- diagnostics

    def compress_diagnostics(self, diagnostics: list[dict], language: str | None = None) -> str:
        by_file: dict[str, list[dict]] = {}
        for diag in diagnostics:
            uri = diag.get("uri", "unknown")
            by_file.setdefault(uri, []).append(diag)

        severity_order = {"error": 0, "warning": 1, "information": 2, "hint": 3}
        output = []
        for uri, items in sorted(by_file.items()):
            items.sort(key=lambda d: severity_order.get(d.get("severity", "hint"), 9))
            shown = items[: self.MAX_DIAGNOSTICS_PER_FILE]
            counts = Counter(d.get("severity", "hint") for d in items)
            summary = ", ".join(f"{n} {sev}" for sev, n in counts.items())
            output.append(f"## {uri} ({summary})")
            for d in shown:
                enriched = enrich_diagnostic(dict(d), language or "") if language else d
                line = enriched.get("range", {}).get("start", {}).get("line", 0) + 1
                output.append(f"  L{line} [{enriched.get('severity', 'info')}] {enriched.get('message', '')}")
                if enriched.get("fix_suggestion"):
                    output.append(f"    → 修复建议: {enriched['fix_suggestion']}")
                    output.append(f"    → 动作类型: {enriched['action_type']}")
            if len(items) > len(shown):
                output.append(f"  ... and {len(items) - len(shown)} more")
        return "\n".join(output) if output else "(no diagnostics)"
