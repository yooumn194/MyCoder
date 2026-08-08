"""LSPResultCompressor — turn raw LSP responses into LLM-friendly text.

Strategy: dedup -> sort -> truncate -> paginate. References are grouped by
file, deduped by line, ranked (project files first, tests last), capped at
MAX_REFERENCES. Diagnostics are grouped by file and severity, with actionable
messages.
"""

from collections import Counter


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

    def compress_diagnostics(self, diagnostics: list[dict]) -> str:
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
                line = d.get("range", {}).get("start", {}).get("line", 0) + 1
                output.append(f"  L{line} [{d.get('severity', 'info')}] {d.get('message', '')}")
            if len(items) > len(shown):
                output.append(f"  ... and {len(items) - len(shown)} more")
        return "\n".join(output) if output else "(no diagnostics)"
