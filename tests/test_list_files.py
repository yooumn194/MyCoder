"""Tests for list_files — glob-based file lookup within the project root."""

from mycoder.tools.list_files import ListFilesTool


def _tool(tmp_path):
    return ListFilesTool(project_root=tmp_path)


def _make_proj(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x")
    (tmp_path / "src" / "utils.py").write_text("x")
    (tmp_path / "tests" / "test_app.py").mkdir(parents=True)
    (tmp_path / "tests" / "test_app.py" / "__init__.py").write_text("")
    return tmp_path


def test_glob_returns_all_python_files(tmp_path):
    _make_proj(tmp_path)
    r = _tool(tmp_path).execute(glob_pattern="**/*.py")
    lines = r.splitlines()
    assert "src/app.py" in lines
    assert "src/utils.py" in lines
    assert "tests/test_app.py/__init__.py" in lines
    # sorted alphabetically
    assert lines == sorted(lines)


def test_no_match_returns_empty(tmp_path):
    _make_proj(tmp_path)
    assert _tool(tmp_path).execute(glob_pattern="**/*.rs") == "(no matches)"


def test_default_excludes(tmp_path):
    _make_proj(tmp_path)
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text("x")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "cache.pyc").write_text("x")  # *.pyc excluded
    r = _tool(tmp_path).execute(glob_pattern="**/*")
    assert "node_modules" not in r
    assert "cache.pyc" not in r
    assert "src/app.py" in r


def test_symlink_not_followed(tmp_path):
    """A symlinked directory's contents must not appear in the results."""
    outside = tmp_path.parent / "outside_files"
    outside.mkdir()
    (outside / "leaked.py").write_text("x")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "real.py").write_text("x")
    (tmp_path / "src" / "link").symlink_to(outside, target_is_directory=True)
    r = _tool(tmp_path).execute(glob_pattern="**/*.py")
    assert "src/real.py" in r
    assert "leaked.py" not in r  # reached only through the symlink


def test_path_traversal_rejected(tmp_path):
    _make_proj(tmp_path)
    r = _tool(tmp_path).execute(glob_pattern="**/*.py", path="../etc")
    assert "路径超出项目范围" in r
