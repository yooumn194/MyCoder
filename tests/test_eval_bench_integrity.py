"""Regression tests for benchmark isolation and verifier integrity."""

import json
from pathlib import Path

import pytest

from eval_bench import runner


def _problem(*, source: str = "def answer():\n    return 0\n") -> dict:
    return {
        "id": "integrity-001",
        "category": "bugfix",
        "difficulty": "easy",
        "prompt": "Make answer return 42.",
        "context_files": {"solution.py": source},
        "verification": {
            "type": "unit_test",
            "test_code": "from solution import answer\n\ndef test_answer():\n    assert answer() == 42\n",
        },
        "timeout_seconds": 30,
        "max_tokens": 1000,
    }


@pytest.mark.parametrize(
    "name",
    [
        "../escape.py",
        "/absolute.py",
        r"nested\escape.py",
        "conftest.py",
        "nested/conftest.py",
        "pytest.ini",
        "test_verify.py",
    ],
)
def test_dataset_rejects_paths_that_escape_or_control_pytest(name):
    problem = _problem()
    problem["context_files"] = {name: "VALUE = 1\n"}

    errors = runner.validate_dataset([problem])

    assert any("unsafe context file" in error for error in errors)


def test_verifier_cannot_be_bypassed_by_agent_conftest_or_pytestaddopts(tmp_path, monkeypatch):
    problem = _problem()
    workdir = tmp_path / "agent"
    workdir.mkdir()
    (workdir / "solution.py").write_text("def answer():\n    return 0\n", encoding="utf-8")
    (workdir / "conftest.py").write_text(
        "def pytest_collection_modifyitems(items):\n    items.clear()\n",
        encoding="utf-8",
    )
    (workdir / "pytest.ini").write_text("[pytest]\naddopts = --collect-only\n", encoding="utf-8")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--collect-only")

    result = runner.verify(problem, workdir)

    assert result["exit_code"] != 0
    assert result["tests_collected"] == 1
    assert result["tests_passed"] == 0
    assert runner.verification_passed(result) is False


def test_verifier_accepts_only_a_real_complete_pass(tmp_path):
    problem = _problem(source="def answer():\n    return 42\n")
    workdir = tmp_path / "agent"
    workdir.mkdir()
    (workdir / "solution.py").write_text(problem["context_files"]["solution.py"], encoding="utf-8")

    result = runner.verify(problem, workdir)

    assert result["exit_code"] == 0
    assert result["tests_collected"] == result["tests_total"] == 1
    assert result["tests_passed"] == 1
    assert runner.verification_passed(result) is True


def test_verification_requires_exact_collection_even_with_zero_exit():
    forged = {
        "exit_code": 0,
        "tests_collected": 0,
        "tests_total": 1,
        "tests_passed": 0,
        "failed": 0,
        "errors": 0,
    }

    assert runner.verification_passed(forged) is False


def test_missing_or_symlinked_declared_output_is_integrity_failure(tmp_path):
    problem = _problem()
    workdir = tmp_path / "agent"
    workdir.mkdir()

    with pytest.raises(runner.VerificationIntegrityError, match="missing or not a regular file"):
        runner.verify(problem, workdir)

    outside = tmp_path / "outside.py"
    outside.write_text("def answer():\n    return 42\n", encoding="utf-8")
    (workdir / "solution.py").symlink_to(outside)
    with pytest.raises(runner.VerificationIntegrityError, match="missing or not a regular file"):
        runner.verify(problem, workdir)


def test_scoped_workspace_is_unique_and_stale_state_is_removed(tmp_path):
    first = runner.scoped_workspace_id("bench", "bugfix-001", "12345678")
    second = runner.scoped_workspace_id("bench", "bugfix-002", "12345678")
    assert first != second
    assert len(first) <= 64

    workdir = runner.prepare_workspace(tmp_path, first)
    stale = workdir / "conftest.py"
    stale.write_text("stale", encoding="utf-8")
    recreated = runner.prepare_workspace(tmp_path, first)

    assert recreated == workdir
    assert not stale.exists()
    assert recreated.parent == Path(tmp_path).resolve()


def test_workspace_cleanup_refuses_unowned_directory_or_symlink(tmp_path):
    foreign = tmp_path / "bench-foreign"
    foreign.mkdir()
    keep = foreign / "keep.txt"
    keep.write_text("mine", encoding="utf-8")

    with pytest.raises(runner.VerificationIntegrityError, match="unowned workspace"):
        runner.prepare_workspace(tmp_path, foreign.name)
    assert keep.read_text(encoding="utf-8") == "mine"

    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "bench-linked"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escaped"):
        runner.prepare_workspace(tmp_path, linked.name)
    assert outside.is_dir()


def test_runner_workspace_matches_local_api_resolution(tmp_path, monkeypatch):
    from api.auth import Principal
    from api.server import resolve_workspace

    api_root = tmp_path / "api-root"
    runner_parent = api_root / "local"
    workspace_id = runner.scoped_workspace_id("bench", "bugfix-001", "12345678")
    prepared = runner.prepare_workspace(runner_parent, workspace_id)
    monkeypatch.setenv("MYCODER_WORKSPACE_ROOT", str(api_root))

    resolved = resolve_workspace(Principal(tenant_id="local", key_id="local-dev"), workspace_id)

    assert resolved == prepared


def test_runner_dry_run_accepts_explicit_id_selection(capsys):
    code = runner.main(
        [
            "--dry-run",
            "--ids",
            "bugfix-001",
            "refactor-006",
            "cross_file-005",
        ]
    )

    assert code == 0
    assert "3 problems" in capsys.readouterr().out


def test_checked_in_dataset_matches_generator():
    from eval_bench import _gen_dataset

    dataset_path = Path(__file__).resolve().parents[1] / "eval_bench" / "dataset.json"

    assert runner.load_dataset(dataset_path) == _gen_dataset.build()


def test_runner_rejects_unknown_selected_id(capsys):
    code = runner.main(["--dry-run", "--ids", "does-not-exist"])

    assert code == 1
    assert "unknown problem id" in capsys.readouterr().out


def test_cross_file_verifier_accepts_canonical_cache_extraction(tmp_path):
    dataset = runner.load_dataset(Path(__file__).resolve().parents[1] / "eval_bench" / "dataset.json")
    problem = next(item for item in dataset if item["id"] == "cross_file-005")
    workdir = tmp_path / "agent"
    workdir.mkdir()
    files = {
        "fetch.py": """\
from cache import Cache

_cache = Cache()

def fetch(url):
    cached = _cache.get(url)
    if cached is not None:
        return cached
    value = f"data:{url}"
    _cache.set(url, value)
    return value

def hit_count():
    return len(_cache)
""",
        "cache.py": """\
class Cache:
    def __init__(self):
        self._data = {}

    def get(self, key):
        return self._data.get(key)

    def set(self, key, value):
        self._data[key] = value

    def __len__(self):
        return len(self._data)
""",
        "metrics.py": """\
from fetch import hit_count

def report():
    return {"hits": hit_count()}
""",
    }
    for name, content in files.items():
        (workdir / name).write_text(content, encoding="utf-8")

    result = runner.verify(problem, workdir)

    assert runner.verification_passed(result) is True
    assert result["tests_collected"] == result["tests_total"] == 1


def test_resume_rejects_results_without_integrity_manifest(tmp_path, capsys):
    results_dir = tmp_path / "legacy-results"
    results_dir.mkdir()
    (results_dir / "raw_results.json").write_text("[]", encoding="utf-8")

    code = runner.main(["--results", str(results_dir), "--resume"])

    assert code == 1
    assert "integrity-v2 manifest" in capsys.readouterr().out


def test_scorer_rejects_legacy_records_and_unpaired_comparisons(tmp_path):
    from eval_bench.scorer import validate_comparable_results, validate_result_integrity

    results_dir = tmp_path / "run"
    results_dir.mkdir()
    contract = {
        "benchmark_integrity_version": runner.BENCHMARK_INTEGRITY_VERSION,
        "task_count": 1,
    }
    (results_dir / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "contract": contract}),
        encoding="utf-8",
    )
    legacy = [{"id": "a", "agent_status": "success", "error_class": None}]

    _, errors = validate_result_integrity(results_dir, legacy)

    assert any("legacy or unverified" in error for error in errors)
    assert validate_comparable_results(
        [{"id": "a", "category": "bugfix", "difficulty": "easy"}],
        [{"id": "b", "category": "bugfix", "difficulty": "easy"}],
    )
