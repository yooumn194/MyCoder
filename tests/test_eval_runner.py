"""Tests for P1-1: the evaluation runner CLI / benchmark."""

import json

from corecoder.eval.runner import main, run_benchmark


def test_run_benchmark_produces_report():
    report = run_benchmark(cases=20)
    assert report["benchmark"] == "orchestration"
    assert report["cases"] == 20
    metrics = report["metrics"]
    assert 0.0 <= metrics["delegation_accuracy"] <= 1.0
    assert metrics["speedup_ratio"] > 1.0  # synthetic parallelism pays
    assert 0.0 <= metrics["lsp_adoption_rate"] <= 1.0


def test_eval_runner_writes_json_report(tmp_path, capsys):
    report_path = tmp_path / "report.json"
    rc = main(["--benchmark", "orchestration", "--cases", "5", "--report", str(report_path)])
    assert rc == 0
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["cases"] == 5
    assert "benchmark=orchestration" in capsys.readouterr().out
