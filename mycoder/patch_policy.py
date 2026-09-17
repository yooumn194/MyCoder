"""Deterministic safety checks for repository patches before verification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath


_IGNORED_TREE_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "build",
    "dist",
    "node_modules",
    "venv",
}
_PROTECTED_BENCHMARK_NAMES = {
    "conftest.py",
    "pyproject.toml",
    "pytest.ini",
    "setup.cfg",
    "sitecustomize.py",
    "tox.ini",
    "usercustomize.py",
}


@dataclass(frozen=True)
class PatchFile:
    path: str
    is_new: bool
    additions: int
    deletions: int


def parse_patch_files(diff: str) -> list[PatchFile]:
    """Extract per-file change shape from a git unified diff."""
    files: list[PatchFile] = []
    path: str | None = None
    is_new = False
    additions = 0
    deletions = 0

    def flush() -> None:
        nonlocal path, is_new, additions, deletions
        if path is not None:
            files.append(PatchFile(path, is_new, additions, deletions))
        path = None
        is_new = False
        additions = 0
        deletions = 0

    for line in str(diff or "").splitlines():
        if line.startswith("diff --git a/"):
            flush()
            parts = line.split(" b/", 1)
            path = parts[1] if len(parts) == 2 else parts[0][len("diff --git a/") :]
        elif path is not None and (
            line.startswith("new file mode") or line == "--- /dev/null"
        ):
            is_new = True
        elif path is not None and line.startswith("+") and not line.startswith("+++"):
            additions += 1
        elif path is not None and line.startswith("-") and not line.startswith("---"):
            deletions += 1
    flush()
    return files


def is_protected_benchmark_path(path: str | Path) -> bool:
    """Whether changing *path* could alter or replace benchmark tests."""
    pure = PurePosixPath(str(path).replace("\\", "/"))
    parts = {part.casefold() for part in pure.parts}
    name = pure.name.casefold()
    return (
        name in _PROTECTED_BENCHMARK_NAMES
        or "test" in parts
        or "tests" in parts
        or name.startswith("test_")
        or name.endswith("_test.py")
        or (pure.suffix.casefold() == ".ini" and name != "mypy.ini")
    )


def patch_scope_violation(
    diff: str,
    *,
    project_root: str | Path | None = None,
    protect_benchmark_files: bool = False,
) -> str | None:
    """Reject high-confidence destructive or shadow-copy change shapes.

    This deliberately avoids a blanket patch-size limit: legitimate broad
    changes remain possible. It blocks only a severe truncation or a large new
    file sharing a basename with another changed file, both common signatures
    of a model confusing a relative path or overwriting a file with a snippet.
    """
    files = parse_patch_files(diff)
    for item in files:
        if protect_benchmark_files and is_protected_benchmark_path(item.path):
            return f"benchmark-protected test/config file changed: {item.path}"
        if item.deletions >= 200 and item.additions * 10 < item.deletions:
            return (
                f"suspicious truncation of {item.path}: "
                f"+{item.additions}/-{item.deletions}"
            )
        if not item.is_new or item.additions < 200:
            continue
        basename = PurePosixPath(item.path).name
        duplicate = next(
            (
                other.path
                for other in files
                if other.path != item.path
                and PurePosixPath(other.path).name == basename
            ),
            None,
        )
        if duplicate is None and project_root is not None:
            root = Path(project_root).resolve()
            for candidate in root.rglob(basename):
                try:
                    relative = candidate.relative_to(root)
                except ValueError:
                    continue
                if (
                    candidate.is_file()
                    and relative.as_posix() != item.path
                    and not (_IGNORED_TREE_PARTS & set(relative.parts))
                ):
                    duplicate = relative.as_posix()
                    break
        if duplicate:
            return (
                f"large new file {item.path} shadows repository file {duplicate} "
                f"(+{item.additions} lines)"
            )
    return None
