"""Tests for the executable MyCoder -> SWE-bench adapter."""

import json
import subprocess
from pathlib import Path

import pytest

from eval_bench.swe_verified import adapter


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _case() -> dict:
    return {
        "instance_id": "owner__repo-1",
        "repo": "owner/repo",
        "base_commit": "a" * 40,
        "problem_statement": "Fix it.",
        "category": "basic_skill",
        "protocol": {
            "control_prompt_template": "${problem_statement}",
            "turns": [{"turn": 1, "user_template": "${problem_statement}", "expected": "fixed"}],
        },
    }


def test_validate_suite_rejects_gold_leakage():
    case = _case()
    case["patch"] = "secret"
    with pytest.raises(ValueError, match="evaluator-only"):
        adapter.validate_suite({"cases": [case]})


def test_control_and_derived_turn_rendering():
    case = _case()
    case["protocol"]["turns"].append({"turn": 2, "user_template": "Now test ${problem_statement}"})

    assert adapter.render_turns(case, "control") == ["Fix it."]
    assert adapter.render_turns(case, "derived") == ["Fix it.", "Now test Fix it."]


def test_prepare_and_extract_patch_includes_untracked_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-q")
    _git(source, "config", "user.email", "test@example.com")
    _git(source, "config", "user.name", "Test")
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(source, "add", ".")
    _git(source, "commit", "-qm", "base")
    commit = _git(source, "rev-parse", "HEAD")
    mirror = tmp_path / "repo.git"
    _git(tmp_path, "clone", "--mirror", str(source), str(mirror))

    workdir = adapter.prepare_workspace(tmp_path / "workspaces", "owned", mirror, commit)
    (workdir / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    (workdir / "added.py").write_text("ADDED = True\n", encoding="utf-8")

    patch, paths = adapter.extract_patch(workdir, commit)

    assert paths == ["added.py", "module.py"]
    assert "VALUE = 2" in patch
    assert "ADDED = True" in patch
    assert adapter.OWNERSHIP_MARKER not in patch

    _git(workdir, "config", "user.email", "test@example.com")
    _git(workdir, "config", "user.name", "Test")
    _git(workdir, "add", "module.py", "added.py")
    _git(workdir, "commit", "-qm", "agent committed changes")
    committed_patch, committed_paths = adapter.extract_patch(workdir, commit)
    assert committed_paths == ["added.py", "module.py"]
    assert "VALUE = 2" in committed_patch


def test_prepare_workspace_cleans_owned_directory_when_fetch_fails(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-q")
    _git(source, "config", "user.email", "test@example.com")
    _git(source, "config", "user.name", "Test")
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(source, "add", ".")
    _git(source, "commit", "-qm", "base")
    mirror = tmp_path / "repo.git"
    _git(tmp_path, "clone", "--mirror", str(source), str(mirror))
    workspaces = tmp_path / "workspaces"

    with pytest.raises(adapter.AdapterError):
        adapter.prepare_workspace(workspaces, "owned", mirror, "a" * 40)

    assert not (workspaces / "owned").exists()


def test_test_poisoning_files_are_rejected(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    commit = _git(repo, "rev-parse", "HEAD")
    (repo / "conftest.py").write_text("def pytest_collection_modifyitems(items): items.clear()\n", encoding="utf-8")

    with pytest.raises(adapter.AdapterError, match="test/config"):
        adapter.extract_patch(repo, commit)


def test_official_prediction_shape_and_harness_command(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    items = [{"instance_id": "owner__repo-1", "model_name_or_path": "MyCoder", "model_patch": "diff"}]
    adapter._write_predictions(predictions, items)

    assert json.loads(predictions.read_text()) == items[0]
    command = adapter.harness_command(
        predictions=predictions,
        source={"dataset": "princeton-nlp/SWE-bench_Verified", "split": "test"},
        instance_ids=["owner__repo-1"],
        workers=1,
        run_id="run",
    )
    assert command[:3] == ["swebench", "eval", "verified"]
    assert command[command.index("--run-id") + 1] == "run"
    assert command[-2:] == ["--instance", "owner__repo-1"]
    # No report-dir by default: the flag must not appear unless a caller asks.
    assert "--report-dir" not in command


def test_harness_command_can_anchor_the_report_directory(tmp_path):
    command = adapter.harness_command(
        predictions=tmp_path / "predictions.jsonl",
        source={"dataset": "verified", "split": "test"},
        instance_ids=["a__b-1", "c__d-2"],
        workers=2,
        run_id="mycoder-abc",
        report_dir=tmp_path / "run",
    )

    # Ahead of the --instance list, so the trailing-argument shape is preserved.
    assert command[command.index("--report-dir") + 1] == str(tmp_path / "run")
    assert command[-2:] == ["--instance", "c__d-2"]


def test_apple_silicon_harness_prep_uses_explicit_amd64(monkeypatch):
    monkeypatch.setattr(adapter.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(adapter.platform, "machine", lambda: "arm64")

    commands = adapter.harness_prep_commands(["django__django-11133"])

    assert commands == [
        [
            "docker",
            "pull",
            "--platform",
            "linux/amd64",
            "swebench/sweb.eval.x86_64.django_1776_django-11133:latest",
        ]
    ]


def test_official_agent_image_matches_harness_image_name():
    assert adapter.official_agent_image("django__django-11133") == (
        "swebench/sweb.eval.x86_64.django_1776_django-11133:latest"
    )


def test_rosetta_python_still_uses_amd64_harness_images(monkeypatch):
    monkeypatch.setattr(adapter.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(adapter.platform, "machine", lambda: "x86_64")
    monkeypatch.delenv("DOCKER_DEFAULT_PLATFORM", raising=False)

    assert adapter.harness_environment()["DOCKER_DEFAULT_PLATFORM"] == "linux/amd64"
    assert adapter.harness_prep_commands(["sympy__sympy-16886"])[0][:4] == [
        "docker",
        "pull",
        "--platform",
        "linux/amd64",
    ]


def test_local_api_bypasses_system_proxy():
    client = adapter.MyCoderAPI("http://127.0.0.1:8000")

    assert not any(
        isinstance(handler, adapter.urllib.request.ProxyHandler) and handler.proxies for handler in client.opener.handlers
    )


def test_adapter_dry_run_uses_materialized_suite(capsys):
    code = adapter.main(["--dry-run", "--limit", "1"])

    assert code == 0
    assert "suite OK: 1 case" in capsys.readouterr().out


def test_adapter_dry_run_does_not_start_verification_containers(monkeypatch, capsys):
    monkeypatch.setattr(
        adapter,
        "probe_verify_runtime",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not probe Docker images"),
    )

    assert adapter.main(["--dry-run", "--limit", "1"]) == 0
    assert "suite OK: 1 case" in capsys.readouterr().out


def test_adapter_defaults_to_bounded_benchmark_contract(tmp_path, monkeypatch):
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "source": {"dataset": "verified", "split": "test"},
                "cases": [_case()],
            }
        ),
        encoding="utf-8",
    )
    results = tmp_path / "results"

    def fake_run_case(case, **kwargs):
        assert kwargs["max_tokens"] == 35_000
        options = kwargs["request_options"]
        assert options["sandbox_policy"] == "benchmark"
        assert options["soft_budget_ratio"] == pytest.approx(20_000 / 35_000)
        return (
            {
                "instance_id": case["instance_id"],
                "model_name_or_path": "MyCoder",
                "model_patch": "diff",
            },
            {"instance_id": case["instance_id"], "patch_bytes": 4, "error": None},
        )

    monkeypatch.setattr(adapter, "run_case", fake_run_case)
    assert adapter.main(["--suite", str(suite_path), "--results", str(results)]) == 0
    contract = json.loads((results / "manifest.json").read_text())["contract"]
    assert contract["soft_budget_tokens_per_turn"] == 20_000


def test_run_case_drives_api_and_emits_official_prediction(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-q")
    _git(source, "config", "user.email", "test@example.com")
    _git(source, "config", "user.name", "Test")
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(source, "add", ".")
    _git(source, "commit", "-qm", "base")
    commit = _git(source, "rev-parse", "HEAD")
    mirror = tmp_path / "repo.git"
    _git(tmp_path, "clone", "--mirror", str(source), str(mirror))
    workspace_root = tmp_path / "workspaces"

    class Cache:
        def ensure(self, repo, base_commit):
            assert repo == "owner/repo"
            assert base_commit == commit
            return mirror

    class API:
        def run_turn(self, *, workspace_name, **kwargs):
            target = workspace_root / workspace_name / "module.py"
            target.write_text("VALUE = 2\n", encoding="utf-8")
            return {"status": "success", "output": "fixed", "token_usage": 10}

    case = _case()
    case["base_commit"] = commit
    prediction, record = adapter.run_case(
        case,
        api=API(),
        cache=Cache(),
        workspace_root=workspace_root,
        run_id="12345678abcd",
        model_name="MyCoder",
        mode="control",
        max_tokens=100,
        timeout_seconds=30,
        request_options={
            "execution_mode": "multi",
            "reasoning_strategy": "auto",
            "orchestration_strategy": "auto",
        },
        reject_test_changes=True,
        keep_workspace=False,
    )

    assert set(prediction) == {"instance_id", "model_name_or_path", "model_patch"}
    assert prediction["model_name_or_path"] == "MyCoder"
    assert "VALUE = 2" in prediction["model_patch"]
    assert record["error"] is None
    assert not (workspace_root / record["workspace_id"]).exists()


def test_resume_skips_completed_case_with_identical_contract(tmp_path, monkeypatch):
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "source": {
                    "dataset": "princeton-nlp/SWE-bench_Verified",
                    "split": "test",
                },
                "cases": [_case()],
            }
        ),
        encoding="utf-8",
    )
    results = tmp_path / "results"
    calls = []

    def fake_run_case(case, **kwargs):
        calls.append(case["instance_id"])
        prediction = {
            "instance_id": case["instance_id"],
            "model_name_or_path": "yooumn194/MyCoder",
            "model_patch": "diff",
        }
        record = {
            "instance_id": case["instance_id"],
            "model_patch": None,
            "patch_bytes": 4,
            "error": None,
        }
        return prediction, record

    monkeypatch.setattr(adapter, "run_case", fake_run_case)
    common = ["--suite", str(suite_path), "--results", str(results)]

    assert adapter.main(common) == 0
    assert calls == ["owner__repo-1"]
    assert adapter.main([*common, "--resume"]) == 0
    assert calls == ["owner__repo-1"]


def test_run_case_rejects_success_without_patch(tmp_path, monkeypatch):
    class Cache:
        def ensure(self, _repo, _commit):
            return tmp_path

    class API:
        def run_turn(self, **_kwargs):
            return {"status": "success", "output": "analysis only"}

    monkeypatch.setattr(adapter, "prepare_workspace", lambda *_args: tmp_path)
    monkeypatch.setattr(adapter, "extract_patch", lambda *_args, **_kwargs: ("", []))
    _prediction, record = adapter.run_case(
        _case(),
        api=API(),
        cache=Cache(),
        workspace_root=tmp_path,
        run_id="12345678abcd",
        model_name="MyCoder",
        mode="control",
        max_tokens=100,
        timeout_seconds=30,
        request_options={
            "execution_mode": "single",
            "reasoning_strategy": "auto",
            "orchestration_strategy": "auto",
        },
        reject_test_changes=True,
        keep_workspace=True,
    )

    assert record["error"] == "AdapterError: agent completed without repository changes"


def test_verify_command_mapping_rejects_bad_input(tmp_path):
    """P0-4: the mapping feeds a command to the server, so it is validated."""
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(adapter.AdapterError, match="JSON object"):
        adapter.load_verify_commands(bad)

    blank = tmp_path / "blank.json"
    blank.write_text(json.dumps({"owner__repo-1": "  "}), encoding="utf-8")
    with pytest.raises(adapter.AdapterError, match="non-empty string"):
        adapter.load_verify_commands(blank)

    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    with pytest.raises(adapter.AdapterError, match="empty"):
        adapter.load_verify_commands(empty)

    missing = tmp_path / "missing.json"
    with pytest.raises(adapter.AdapterError, match="not found"):
        adapter.load_verify_commands(missing)


def test_verify_command_mapping_prefers_instance_then_default(tmp_path):
    path = tmp_path / "verify.json"
    path.write_text(
        json.dumps(
            {
                "*": "python -m pytest -q",
                "owner__repo-1": "python -m pytest -q tests/test_repo.py",
            }
        ),
        encoding="utf-8",
    )

    commands = adapter.load_verify_commands(path)

    assert adapter.verify_command_for(commands, "owner__repo-1") == (
        "python -m pytest -q tests/test_repo.py"
    )
    assert adapter.verify_command_for(commands, "other__repo-2") == "python -m pytest -q"
    assert adapter.verify_command_for(None, "owner__repo-1") is None
    assert adapter.verify_command_for({"*": "x"}, "ghost") == "x"


def test_run_case_sends_the_harness_verify_command(tmp_path, monkeypatch):
    seen: list[dict] = []

    class Cache:
        def ensure(self, _repo, _commit):
            return tmp_path

    class API:
        def run_turn(self, *, request_options, **_kwargs):
            seen.append(dict(request_options))
            return {"status": "success", "output": "fixed"}

    monkeypatch.setattr(adapter, "prepare_workspace", lambda *_args: tmp_path)
    monkeypatch.setattr(adapter, "extract_patch", lambda *_args, **_kwargs: ("diff", []))
    common = {
        "api": API(),
        "cache": Cache(),
        "workspace_root": tmp_path,
        "run_id": "12345678abcd",
        "model_name": "MyCoder",
        "mode": "control",
        "max_tokens": 100,
        "timeout_seconds": 30,
        "request_options": {
            "execution_mode": "single",
            "reasoning_strategy": "auto",
            "orchestration_strategy": "auto",
        },
        "reject_test_changes": True,
        "keep_workspace": True,
    }

    _prediction, record = adapter.run_case(
        _case(), **common, verify_commands={"owner__repo-1": "./run_tests.sh"}
    )

    assert seen[0]["benchmark_verify_cmd"] == "./run_tests.sh"
    assert record["harness_verify_cmd"] == "./run_tests.sh"

    # An instance with no entry keeps the agent-reported path, and the record
    # says so instead of looking like a harness-verified run.
    seen.clear()
    _prediction, record = adapter.run_case(
        _case(), **common, verify_commands={"other__repo-2": "./run_tests.sh"}
    )

    assert "benchmark_verify_cmd" not in seen[0]
    assert record["harness_verify_cmd"] is None


def test_run_case_records_the_harness_verdict(tmp_path, monkeypatch):
    """A red harness check is a result, not a failed case: the official
    evaluator scores the patch, and the record just has to say what happened."""
    verdict = {
        "command": "./run_tests.sh",
        "passed": False,
        "evidence": "exit_code=1\n1 failed",
    }

    class Cache:
        def ensure(self, _repo, _commit):
            return tmp_path

    class API:
        def run_turn(self, **_kwargs):
            return {"status": "success", "output": "fixed", "harness_verification": verdict}

    monkeypatch.setattr(adapter, "prepare_workspace", lambda *_args: tmp_path)
    monkeypatch.setattr(adapter, "extract_patch", lambda *_args, **_kwargs: ("diff", []))

    _prediction, record = adapter.run_case(
        _case(),
        api=API(),
        cache=Cache(),
        workspace_root=tmp_path,
        run_id="12345678abcd",
        model_name="MyCoder",
        mode="control",
        max_tokens=100,
        timeout_seconds=30,
        request_options={},
        reject_test_changes=True,
        keep_workspace=True,
        verify_commands={"owner__repo-1": "./run_tests.sh"},
    )

    assert record["error"] is None
    assert record["harness_verification"] == verdict


def test_manifest_summarises_harness_verdicts(tmp_path, monkeypatch):
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps({"source": {"dataset": "verified", "split": "test"}, "cases": [_case()]}),
        encoding="utf-8",
    )

    def fake_run_case(case, **_kwargs):
        return (
            {"instance_id": case["instance_id"], "model_name_or_path": "MyCoder", "model_patch": "diff"},
            {
                "instance_id": case["instance_id"],
                "patch_bytes": 4,
                "error": None,
                "harness_verification": {"command": "./run_tests.sh", "passed": True, "evidence": "exit_code=0"},
            },
        )

    monkeypatch.setattr(adapter, "run_case", fake_run_case)
    results = tmp_path / "results"

    assert adapter.main(["--suite", str(suite_path), "--results", str(results)]) == 0
    manifest = json.loads((results / "manifest.json").read_text())
    assert manifest["harness_verification_summary"] == {
        "verified_green": 1,
        "verified_red": 0,
        "verified_unavailable": 0,
        "self_certified": 0,
    }


def test_adapter_warns_when_no_verify_command_is_configured(tmp_path, monkeypatch, capsys):
    """A silent fallback would make a self-scored run look like a verified one."""
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps({"source": {"dataset": "verified", "split": "test"}, "cases": [_case()]}),
        encoding="utf-8",
    )

    def fake_run_case(case, **kwargs):
        return (
            {"instance_id": case["instance_id"], "model_name_or_path": "MyCoder", "model_patch": "diff"},
            {"instance_id": case["instance_id"], "patch_bytes": 4, "error": None},
        )

    monkeypatch.setattr(adapter, "run_case", fake_run_case)
    results = tmp_path / "results"

    # `--no-verify-commands` is the only way to get here now: the bundled
    # mapping is otherwise picked up, so a run no longer scores itself by
    # default.
    assert (
        adapter.main(
            ["--suite", str(suite_path), "--results", str(results), "--no-verify-commands"]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "WARNING" in out and "score themselves" in out
    contract = json.loads((results / "manifest.json").read_text())["contract"]
    assert contract["verify_commands"] is None


def test_adapter_uses_the_bundled_verify_commands_by_default(tmp_path, monkeypatch, capsys):
    """The harness owns the verdict on the default path, not the model."""
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps({"source": {"dataset": "verified", "split": "test"}, "cases": [_case()]}),
        encoding="utf-8",
    )
    result = (
        {"instance_id": _case()["instance_id"], "model_name_or_path": "MyCoder", "model_patch": "diff"},
        {"instance_id": _case()["instance_id"], "patch_bytes": 4, "error": None},
    )
    monkeypatch.setattr(adapter, "run_case", lambda case, **kwargs: result)
    results = tmp_path / "results"

    assert adapter.main(["--suite", str(suite_path), "--results", str(results)]) == 0
    out = capsys.readouterr().out
    assert "WARNING" not in out
    # The catch-all is weaker evidence than a per-instance command, and the
    # run says so instead of implying the held-out tests ran.
    assert 'catch-all "*"' in out

    contract = json.loads((results / "manifest.json").read_text())["contract"]
    recorded = contract["verify_commands"]
    assert recorded["source"] == "default"
    assert recorded["path"].endswith("verify_commands.json")
    assert recorded["uncovered"] == []
    assert recorded["catch_all"]


def test_adapter_rejects_a_broken_verify_command_file(tmp_path, capsys):
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps({"source": {"dataset": "verified", "split": "test"}, "cases": [_case()]}),
        encoding="utf-8",
    )
    bad = tmp_path / "verify.json"
    bad.write_text("{not json", encoding="utf-8")

    code = adapter.main(
        [
            "--suite",
            str(suite_path),
            "--verify-commands",
            str(bad),
            "--dry-run",
        ]
    )

    assert code == 1
    assert "invalid --verify-commands" in capsys.readouterr().out


def test_adapter_can_require_explicit_per_instance_verification(tmp_path, capsys):
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "source": {"dataset": "verified", "split": "test"},
                "cases": [_case()],
            }
        ),
        encoding="utf-8",
    )

    # The bundled file intentionally has only a smoke-test catch-all. Strict
    # evaluation must not silently treat that as the held-out test command.
    code = adapter.main(
        [
            "--suite",
            str(suite_path),
            "--require-per-instance-verification",
            "--dry-run",
        ]
    )
    assert code == 1
    assert "weak verification" in capsys.readouterr().out

    mapping = tmp_path / "verify.json"
    mapping.write_text(
        json.dumps({_case()["instance_id"]: "python -m pytest -q tests/test_repo.py"}),
        encoding="utf-8",
    )
    assert (
        adapter.main(
            [
                "--suite",
                str(suite_path),
                "--verify-commands",
                str(mapping),
                "--require-per-instance-verification",
                "--dry-run",
            ]
        )
        == 0
    )


def test_the_harness_report_lands_in_the_run_directory_not_the_cwd(tmp_path, monkeypatch):
    """The harness defaults --report-dir to ".", which is how a repo root ends up
    holding one <model>.<run-id>.json per run."""
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps({"source": {"dataset": "verified", "split": "test"}, "cases": [_case()]}),
        encoding="utf-8",
    )

    def fake_run_case(case, **_kwargs):
        return (
            {"instance_id": case["instance_id"], "model_name_or_path": "MyCoder", "model_patch": "diff"},
            {"instance_id": case["instance_id"], "patch_bytes": 4, "error": None},
        )

    monkeypatch.setattr(adapter, "run_case", fake_run_case)
    results = tmp_path / "results"

    assert adapter.main(["--suite", str(suite_path), "--results", str(results)]) == 0
    manifest = json.loads((results / "manifest.json").read_text())
    command = manifest["official_harness_command"]
    assert command[command.index("--report-dir") + 1] == str(results)
    # A label, not the resolved path: a literal path would make otherwise
    # identical repeats of one variance run compare unequal.
    assert manifest["contract"]["harness_report_dir"] == "run_results"


def test_an_explicit_report_directory_is_recorded_as_such(tmp_path, monkeypatch):
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps({"source": {"dataset": "verified", "split": "test"}, "cases": [_case()]}),
        encoding="utf-8",
    )

    def fake_run_case(case, **_kwargs):
        return (
            {"instance_id": case["instance_id"], "model_name_or_path": "MyCoder", "model_patch": "diff"},
            {"instance_id": case["instance_id"], "patch_bytes": 4, "error": None},
        )

    monkeypatch.setattr(adapter, "run_case", fake_run_case)
    results = tmp_path / "results"
    elsewhere = tmp_path / "elsewhere"

    assert (
        adapter.main(
            ["--suite", str(suite_path), "--results", str(results), "--harness-report-dir", str(elsewhere)]
        )
        == 0
    )
    manifest = json.loads((results / "manifest.json").read_text())
    command = manifest["official_harness_command"]
    assert command[command.index("--report-dir") + 1] == str(elsewhere)
    assert manifest["contract"]["harness_report_dir"] == "explicit"


def test_the_repeated_run_resumes_only_when_the_report_directory_is_unchanged(tmp_path, monkeypatch):
    """A changed report directory must break resume, like any other contract field."""
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps({"source": {"dataset": "verified", "split": "test"}, "cases": [_case()]}),
        encoding="utf-8",
    )

    def fake_run_case(case, **_kwargs):
        return (
            {"instance_id": case["instance_id"], "model_name_or_path": "MyCoder", "model_patch": "diff"},
            {"instance_id": case["instance_id"], "patch_bytes": 4, "error": None},
        )

    monkeypatch.setattr(adapter, "run_case", fake_run_case)
    results = tmp_path / "results"
    assert adapter.main(["--suite", str(suite_path), "--results", str(results)]) == 0

    assert adapter.main(["--suite", str(suite_path), "--results", str(results), "--resume"]) == 0
    moved = adapter.main(
        [
            "--suite", str(suite_path), "--results", str(results),
            "--harness-report-dir", str(tmp_path / "moved"), "--resume",
        ]
    )
    assert moved == 1
