from __future__ import annotations

from eval_bench import runner


class _Response:
    status_code = 200

    def __init__(self, payload: dict):
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _Client:
    def post(self, *_args, **_kwargs):
        response = _Response({"status": "running"})
        response.status_code = 202
        return response

    def get(self, *_args, **_kwargs):
        return _Response(
            {
                "status": "failed",
                "token_usage": 12,
                "error": {"code": "AGENT_FAILED", "detail": "expected"},
            }
        )


def test_run_one_watchdog_uses_monotonic_clock(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    ticks = iter((100.0, 100.1, 101.0))
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(
        runner.time,
        "time",
        lambda: (_ for _ in ()).throw(AssertionError("wall clock used by watchdog")),
    )
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)

    problem = {
        "id": "clock-test",
        "category": "bugfix",
        "difficulty": "easy",
        "prompt": "Do nothing.",
        "context_files": {"sample.py": "VALUE = 1\n"},
        "verification": {"type": "unit_test", "test_code": ""},
        "timeout_seconds": 10,
        "max_tokens": 100,
    }
    result = runner.run_one(
        problem,
        "http://127.0.0.1:8000",
        tmp_path / "workspace",
        tmp_path / "results",
        "run",
        _Client(),
    )

    assert result["agent_status"] == "failed"
    assert result["duration_s"] == 1.0
