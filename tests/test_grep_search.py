"""Tests for grep_search — ripgrep-first regex search with a Python fallback."""

import shutil
import subprocess

import pytest

from corecoder.tools.grep_search import GrepSearchTool


def _make_proj(tmp_path):
    (tmp_path / "src" / "auth").mkdir(parents=True)
    (tmp_path / "src" / "auth" / "handler.py").write_text(
        "def authenticate_user(username, password):\n"
        "    return token\n"
        "x = authenticate_user('a', 'b')\n"
    )
    (tmp_path / "src" / "auth" / "middleware.py").write_text(
        "from .handler import authenticate_user\n"
        "authenticate_user('u', 'p')\n"
    )
    (tmp_path / "README.md").write_text("docs about authenticate_user\n")
    return tmp_path


def _py_tool(tmp_path):
    return GrepSearchTool(project_root=tmp_path, rg_path=None)


def _rg_tool(tmp_path):
    rg = shutil.which("rg")
    return GrepSearchTool(project_root=tmp_path, rg_path=rg)


def test_exact_match_function_name(tmp_path):
    _make_proj(tmp_path)
    r = _py_tool(tmp_path).execute(pattern="def authenticate_user", file_types="py")
    assert "## src/auth/handler.py" in r
    assert "L1: def authenticate_user" in r
    assert "Found 1 match in 1 file" in r


def test_regex_pattern(tmp_path):
    _make_proj(tmp_path)
    r = _py_tool(tmp_path).execute(pattern=r"\bdef\s+\w+", file_types="py")
    assert "def authenticate_user" in r


def test_no_match_returns_empty(tmp_path):
    _make_proj(tmp_path)
    assert _py_tool(tmp_path).execute(pattern="zzz_nothing") == "(no matches)"


def test_max_results_truncation(tmp_path):
    _make_proj(tmp_path)
    r = _py_tool(tmp_path).execute(
        pattern="authenticate_user", file_types="py", max_results=1
    )
    assert "Found 1 match" in r  # header shows the SHOWN count
    assert "结果已截断" in r      # the note reports the true total


def test_file_types_filter(tmp_path):
    _make_proj(tmp_path)
    tool = _py_tool(tmp_path)
    r_py = tool.execute(pattern="authenticate_user", file_types="py")
    assert "README.md" not in r_py  # .md excluded when filtering to py
    r_md = tool.execute(pattern="authenticate_user", file_types="md")
    assert "README.md" in r_md


def test_empty_dir_no_crash(tmp_path):
    assert _py_tool(tmp_path).execute(pattern="foo") == "(no matches)"


def test_binary_file_skipped(tmp_path):
    (tmp_path / "data.bin").write_bytes(b"\x00\x01\x02authenticate_user")
    assert _py_tool(tmp_path).execute(pattern="authenticate_user") == "(no matches)"


def test_special_char_escape(tmp_path):
    (tmp_path / "code.py").write_text("result = foo.bar()\n")
    r = _py_tool(tmp_path).execute(pattern=r"foo\.bar\(\)", file_types="py")
    assert "foo.bar()" in r


def test_case_sensitivity(tmp_path):
    (tmp_path / "code.py").write_text("AUTHENTICATE_USER here\n")
    tool = _py_tool(tmp_path)
    assert tool.execute(pattern="authenticate_user", case_sensitive=False) != "(no matches)"
    assert tool.execute(pattern="authenticate_user", case_sensitive=True) == "(no matches)"


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
def test_rg_backend_equivalent_to_fallback(tmp_path):
    """Same input -> rg and Python backends produce byte-identical output."""
    _make_proj(tmp_path)
    pattern = "authenticate_user"
    assert _rg_tool(tmp_path).execute(pattern=pattern, file_types="py") == _py_tool(
        tmp_path
    ).execute(pattern=pattern, file_types="py")


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
def test_rg_backend_basic(tmp_path):
    _make_proj(tmp_path)
    r = _rg_tool(tmp_path).execute(pattern="def authenticate_user", file_types="py")
    assert "## src/auth/handler.py" in r
    assert "L1: def authenticate_user" in r


def test_rg_output_parsing(monkeypatch, tmp_path):
    """The rg backend's `path:linenum:content` parser, fed canned rg output."""
    _make_proj(tmp_path)
    tool = GrepSearchTool(project_root=tmp_path, rg_path="/usr/bin/rg")

    class _FakeProc:
        returncode = 0
        stderr = ""
        stdout = (
            "src/auth/handler.py:3:def authenticate_user()\n"
            "src/auth/handler.py:5:    x = authenticate_user()\n"
        )

    monkeypatch.setattr(
        "corecoder.tools.grep_search.subprocess.run", lambda *a, **k: _FakeProc()
    )
    r = tool.execute(pattern="authenticate_user")
    assert "## src/auth/handler.py" in r
    assert "L3: def authenticate_user()" in r
    assert "L5:     x = authenticate_user()" in r
    assert "Found 2 matches in 1 file" in r


def test_timeout_returns_graceful(monkeypatch, tmp_path):
    """A slow rg must time out with a graceful note, never hang the agent."""
    tool = GrepSearchTool(project_root=tmp_path, rg_path="/usr/bin/rg")

    def _slow_run(*args, **kwargs):
        raise subprocess.TimeoutExpired("rg", 30)

    monkeypatch.setattr("corecoder.tools.grep_search.subprocess.run", _slow_run)
    r = tool.execute(pattern="foo")
    assert "搜索超时" in r


def test_path_traversal_rejected(tmp_path):
    _make_proj(tmp_path)
    r = _py_tool(tmp_path).execute(pattern="foo", path="../etc")
    assert "路径超出项目范围" in r


def test_invalid_regex_reported(tmp_path):
    _make_proj(tmp_path)
    r = _py_tool(tmp_path).execute(pattern="[invalid")
    assert "搜索失败" in r or "(no matches)" in r


def test_fallback_survives_non_utf8_file(tmp_path):
    """A GBK-encoded file (bytes invalid in UTF-8) must not crash the search.

    The fallback reads with errors="replace", so a non-UTF-8 file degrades to
    replacement characters instead of aborting the whole walk — and the valid
    UTF-8 files are still searched.
    """
    _make_proj(tmp_path)
    (tmp_path / "gbk.py").write_bytes('名称 = "你好"\n'.encode("gbk"))
    r = _py_tool(tmp_path).execute(pattern="def authenticate_user", file_types="py")
    assert "src/auth/handler.py" in r  # valid files still searched, no crash
