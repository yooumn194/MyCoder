from mycoder.tools import build_scoped_tools
from mycoder.tools.edit import EditFileTool
from mycoder.tools.write import WriteFileTool


def test_scoped_mutation_tools_reject_workspace_escape(tmp_path):
    root = tmp_path / "tenant" / "workspace"
    root.mkdir(parents=True)
    outside = tmp_path / "outside.txt"

    write = WriteFileTool(project_root=root)
    assert "Error:" in write.execute(str(outside), "secret")
    assert not outside.exists()

    target = root / "inside.txt"
    target.write_text("before", encoding="utf-8")
    edit = EditFileTool(project_root=root)
    assert "Error:" in edit.execute(str(outside), "x", "y")
    assert target.read_text(encoding="utf-8") == "before"


def test_scoped_tool_registries_do_not_share_sandbox_manager(tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    _, manager_a = build_scoped_tools(first, "session-a")
    _, manager_b = build_scoped_tools(second, "session-b")
    assert manager_a is not manager_b
    assert manager_a.project_dir == first.resolve()
    assert manager_b.project_dir == second.resolve()
