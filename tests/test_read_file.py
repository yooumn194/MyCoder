"""Tests for read_file — ranges, truncation, binary detection, path guarding."""

from mycoder.tools.read_file import MAX_LINES, ReadFileTool


def _tool(tmp_path):
    return ReadFileTool(project_root=tmp_path)


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_start_end_line_range(tmp_path):
    f = tmp_path / "sample.txt"
    _write(f, "\n".join(f"line{i}" for i in range(100)))
    r = _tool(tmp_path).execute(file_path="sample.txt", start_line=10, end_line=14)
    assert "10 | line9" in r  # line 10 (1-based) carries content "line9"
    assert "line8" not in r
    assert "line15" not in r


def test_long_file_truncates_with_hint(tmp_path):
    f = tmp_path / "big.txt"
    _write(f, "\n".join(f"line{i}" for i in range(1247)))
    r = _tool(tmp_path).execute(file_path="big.txt")
    assert "1 | line0" in r
    assert "1247 | line1246" not in r  # beyond the 300-line window
    assert "文件共 1247 行" in r
    assert "仅显示前" in r
    # exactly MAX_LINES numbered rows shown
    rows = [ln for ln in r.splitlines() if " | " in ln]
    assert len(rows) == MAX_LINES


def test_explicit_range_still_capped_at_300(tmp_path):
    f = tmp_path / "big.txt"
    _write(f, "\n".join(f"line{i}" for i in range(1000)))
    r = _tool(tmp_path).execute(file_path="big.txt", start_line=100, end_line=9999)
    rows = [ln for ln in r.splitlines() if " | " in ln]
    assert len(rows) == MAX_LINES  # hard cap on explicit ranges too


def test_file_not_found(tmp_path):
    r = _tool(tmp_path).execute(file_path="no_such_file.txt")
    assert "不存在" in r


def test_binary_file_rejected(tmp_path):
    f = tmp_path / "blob.bin"
    f.write_bytes(b"\x00\x01\x02hello")
    r = _tool(tmp_path).execute(file_path="blob.bin")
    assert "二进制文件" in r


def test_line_number_format_four_wide(tmp_path):
    f = tmp_path / "fmt.txt"
    _write(f, "\n".join(f"content{i}" for i in range(5)))
    r = _tool(tmp_path).execute(file_path="fmt.txt", start_line=5, end_line=5)
    assert r == "   5 | content4"  # 4-wide right-aligned, ' | ' separator


def test_path_traversal_rejected(tmp_path):
    _write(tmp_path / "ok.txt", "x")
    r = _tool(tmp_path).execute(file_path="../etc/passwd")
    assert "路径超出项目范围" in r


def test_empty_file(tmp_path):
    (tmp_path / "empty.txt").write_text("")
    assert _tool(tmp_path).execute(file_path="empty.txt") == "(empty file)"
