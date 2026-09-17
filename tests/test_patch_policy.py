"""Deterministic regression tests for patch-scope safety checks."""

from mycoder.patch_policy import (
    is_protected_benchmark_path,
    parse_patch_files,
    patch_scope_violation,
)


def _large_new_file(path: str, lines: int = 250) -> str:
    body = "\n".join(f"+line {index}" for index in range(lines))
    return (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        f"@@ -0,0 +1,{lines} @@\n{body}\n"
    )


def test_parse_patch_files_counts_changes_and_new_files():
    diff = (
        _large_new_file("crypto.py")
        + "diff --git a/sympy/crypto/crypto.py b/sympy/crypto/crypto.py\n"
        + "--- a/sympy/crypto/crypto.py\n"
        + "+++ b/sympy/crypto/crypto.py\n"
        + "@@ -1,2 +1,2 @@\n-old\n+new\n unchanged\n"
    )

    files = parse_patch_files(diff)

    assert [(item.path, item.is_new) for item in files] == [
        ("crypto.py", True),
        ("sympy/crypto/crypto.py", False),
    ]
    assert (files[1].additions, files[1].deletions) == (1, 1)


def test_rejects_large_shadow_module_added_beside_real_edit():
    diff = (
        _large_new_file("crypto.py")
        + "diff --git a/sympy/crypto/crypto.py b/sympy/crypto/crypto.py\n"
        + "--- a/sympy/crypto/crypto.py\n"
        + "+++ b/sympy/crypto/crypto.py\n"
        + "@@ -1 +1 @@\n-old\n+new\n"
    )

    violation = patch_scope_violation(diff)

    assert violation is not None
    assert "shadows repository file sympy/crypto/crypto.py" in violation


def test_rejects_large_shadow_when_existing_file_is_not_in_diff(tmp_path):
    existing = tmp_path / "sympy" / "crypto" / "crypto.py"
    existing.parent.mkdir(parents=True)
    existing.write_text("existing module\n", encoding="utf-8")

    violation = patch_scope_violation(
        _large_new_file("crypto.py"), project_root=tmp_path
    )

    assert violation is not None
    assert "shadows repository file sympy/crypto/crypto.py" in violation


def test_rejects_severe_file_truncation():
    deleted = "\n".join(f"-line {index}" for index in range(250))
    diff = (
        "diff --git a/module.py b/module.py\n"
        "--- a/module.py\n"
        "+++ b/module.py\n"
        f"@@ -1,250 +1 @@\n{deleted}\n+replacement\n"
    )

    violation = patch_scope_violation(diff)

    assert violation == "suspicious truncation of module.py: +1/-250"


def test_accepts_small_focused_edit():
    diff = (
        "diff --git a/sympy/crypto/crypto.py b/sympy/crypto/crypto.py\n"
        "--- a/sympy/crypto/crypto.py\n"
        "+++ b/sympy/crypto/crypto.py\n"
        "@@ -1 +1 @@\n-\"----\": \"1\"\n+\".----\": \"1\"\n"
    )

    assert patch_scope_violation(diff) is None


def test_benchmark_protected_paths_cover_tests_and_runner_config():
    assert is_protected_benchmark_path("sympy/crypto/tests/test_crypto.py")
    assert is_protected_benchmark_path("test_crypto.py")
    assert is_protected_benchmark_path("integration/crypto_test.py")
    assert is_protected_benchmark_path("pytest.ini")
    assert not is_protected_benchmark_path("sympy/crypto/crypto.py")
    assert not is_protected_benchmark_path("mypy.ini")


def test_rejects_protected_benchmark_file_in_patch():
    diff = (
        "diff --git a/tests/test_module.py b/tests/test_module.py\n"
        "--- a/tests/test_module.py\n"
        "+++ b/tests/test_module.py\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )

    violation = patch_scope_violation(diff, protect_benchmark_files=True)

    assert violation == (
        "benchmark-protected test/config file changed: tests/test_module.py"
    )
