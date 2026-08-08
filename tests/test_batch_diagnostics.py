"""Tests for P1-2: batch diagnostics (3 writes -> 1 LSP diagnostics request)."""

from corecoder.tools.batch_diagnostics import BatchDiagnostics
from corecoder.tools.write import WriteFileTool


def test_batch_diagnostics_after_3_writes(tmp_path):
    triggered: list[list[str]] = []
    batch = BatchDiagnostics(threshold=3, trigger=triggered.append)
    write = WriteFileTool(batch_diagnostics=batch)

    for i in range(3):
        write.execute(file_path=str(tmp_path / f"f{i}.py"), content="x\n")

    # three writes -> exactly one batch trigger carrying all three files
    assert len(triggered) == 1
    assert len(triggered[0]) == 3
    assert batch.pending == []


def test_single_write_immediate_diagnostics(tmp_path):
    triggered: list[list[str]] = []
    batch = BatchDiagnostics(threshold=3, trigger=triggered.append)
    write = WriteFileTool(batch_diagnostics=batch)

    path = str(tmp_path / "a.py")
    write.execute(file_path=path, content="x\n")
    assert batch.pending == [path]  # queued, not yet flushed
    batch.flush()  # explicit flush (e.g. conversation end) -> immediate
    assert len(triggered) == 1
    assert triggered[0] == [path]
