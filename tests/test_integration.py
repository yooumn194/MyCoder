"""End-to-end integration tests for the Phase 2 search toolset.

These exercise grep_search / list_files / read_file together the way the Agent
would: locate a symbol, inspect the file, narrow the scope, and fail safely on
a slow search or a traversal attempt.
"""

import subprocess

from corecoder.tools import grep_search as gs
from corecoder.tools.grep_search import GrepSearchTool
from corecoder.tools.list_files import ListFilesTool
from corecoder.tools.read_file import ReadFileTool


def _make_proj(tmp_path):
    (tmp_path / "src" / "auth").mkdir(parents=True)
    (tmp_path / "src" / "auth" / "handler.py").write_text(
        "import logging\n"
        "\n"
        "def authenticate_user(username, password):\n"
        "    return {'user': username}\n"
    )
    (tmp_path / "src" / "auth" / "middleware.py").write_text(
        "from .handler import authenticate_user\n"
        "result = authenticate_user('u', 'p')\n"
    )
    (tmp_path / "README.md").write_text("docs about authenticate_user\n")
    return tmp_path


def _tools(tmp_path):
    return (
        GrepSearchTool(project_root=tmp_path, rg_path=None),
        ListFilesTool(project_root=tmp_path),
        ReadFileTool(project_root=tmp_path),
    )


def test_grep_then_read_workflow(tmp_path):
    """grep_search to locate the implementation, then read_file to inspect it."""
    _make_proj(tmp_path)
    grep, _, read = _tools(tmp_path)
    r = grep.execute(pattern="def authenticate_user", file_types="py")
    assert "src/auth/handler.py" in r

    lines = read.execute(file_path="src/auth/handler.py", start_line=1, end_line=5)
    assert "import logging" in lines
    assert "def authenticate_user" in lines


def test_multi_round_narrowing(tmp_path):
    """Start broad, then narrow by file type and by path."""
    _make_proj(tmp_path)
    grep, list_files, _ = _tools(tmp_path)

    broad = grep.execute(pattern="authenticate_user")
    assert "README.md" in broad  # unfiltered search hits the .md too

    typed = grep.execute(pattern="authenticate_user", file_types="py")
    assert "README.md" not in typed

    scoped = grep.execute(pattern="authenticate_user", file_types="py", path="src/auth")
    assert "src/auth/handler.py" in scoped
    assert "middleware.py" in scoped

    files = list_files.execute(glob_pattern="**/*.py")
    assert "src/auth/handler.py" in files


def test_slow_search_times_out_gracefully(tmp_path, monkeypatch):
    """A slow search must return the timeout note, never hang the agent."""
    _make_proj(tmp_path)
    tool = GrepSearchTool(project_root=tmp_path, rg_path="/usr/bin/rg")

    def _slow_run(*args, **kwargs):
        raise subprocess.TimeoutExpired("rg", 30)

    monkeypatch.setattr(gs.subprocess, "run", _slow_run)
    out = tool.execute(pattern="authenticate_user")
    assert "搜索超时" in out


def test_search_never_leaves_project_root(tmp_path):
    """PathGuard holds end-to-end: traversal is rejected at every search tool."""
    _make_proj(tmp_path)
    grep, list_files, read = _tools(tmp_path)
    assert "路径超出项目范围" in grep.execute(pattern="x", path="../etc")
    assert "路径超出项目范围" in list_files.execute(glob_pattern="**/*.py", path="..")
    assert "路径超出项目范围" in read.execute(file_path="../etc/passwd")
