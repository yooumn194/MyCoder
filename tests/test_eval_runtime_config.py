import json

from eval_bench import smoke
from eval_bench.runtime_config import snapshot


def test_runtime_snapshot_is_effective_and_secret_free(monkeypatch):
    monkeypatch.setenv("MYCODER_PROFILE", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "super-secret")
    monkeypatch.setenv("MYCODER_OPENROUTER_MODEL", "minimax/minimax-m3:free")
    monkeypatch.setenv("MYCODER_OPENROUTER_BASE_URL", "https://router.example/v1?token=secret#x")
    monkeypatch.setenv("MYCODER_TOOL_DIALECT", "native")

    config = snapshot()

    assert config == {
        "profile": "openrouter",
        "provider": "openrouter",
        "model": "minimax/minimax-m3:free",
        "base_url": "https://router.example/v1",
        "tool_dialect": "native",
        "temperature": 0.0,
        "max_tokens": 4096,
        "max_context_tokens": 128000,
        "thinking": "auto",
    }
    assert "super-secret" not in json.dumps(config)


def test_smoke_is_a_single_control_preset(monkeypatch, capsys):
    observed = {}

    def fake_adapter(argv):
        observed["argv"] = argv
        return 0

    monkeypatch.setattr(smoke.adapter, "main", fake_adapter)
    assert smoke.main(["--dry-run", "--instance-id", "pytest-dev__pytest-5262"]) == 0
    assert observed["argv"] == [
        "--base-url", "http://localhost:8000",
        "--instance-id", "pytest-dev__pytest-5262",
        "--workspace", "workspaces/swe-bench/local",
        "--repo-cache", "workspaces/swe-bench-repos",
        "--results", "results/swe-bench/smoke",
        "--max-tokens", "35000",
        "--soft-budget-tokens", "20000",
        "--timeout-seconds", "1200",
        "--mode", "control",
        "--parallel", "1",
        "--execution-mode", "single",
        "--reasoning-strategy", "react",
        "--orchestration-strategy", "sequential",
        "--dry-run",
    ]
    assert "[smoke] runtime=" in capsys.readouterr().out


def test_smoke_thinking_override_is_reflected_in_runtime_snapshot(monkeypatch, capsys):
    monkeypatch.delenv("MYCODER_DEEPSEEK_THINKING", raising=False)
    monkeypatch.setenv("MYCODER_PROFILE", "deepseek")
    monkeypatch.setattr(smoke.adapter, "main", lambda _argv: 0)

    assert smoke.main(["--dry-run", "--thinking", "disabled"]) == 0

    assert "'thinking': 'disabled'" in capsys.readouterr().out
