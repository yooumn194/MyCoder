"""Tests for repeated official-scored SWE-bench runs.

The fixtures write the real on-disk layout — ``manifest.json``,
``adapter_results.json`` and an official report — because the module's whole job
is reading that layout back. Synthesising the objects in memory would test
``aggregate`` against a shape the adapter never produces.
"""

import json
from pathlib import Path

import pytest

from eval_bench.swe_verified import adapter, variance

INSTANCES = ["django__django-11133", "pytest-dev__pytest-5262", "sympy__sympy-16886"]

CONTRACT = {
    "model_name_or_path": "yooumn194/MyCoder",
    "instance_ids": INSTANCES,
    "suite_sha256": "a" * 64,
}


def _write_repeat(
    root: Path,
    index: int,
    *,
    resolved: list[str] | None,
    contract: dict | None = None,
    total_instances: int = 3,
    errors: bool = False,
    infra_failures: int = 0,
    verification: dict | None = None,
) -> variance.Repeat:
    """One repeat directory: manifest, adapter records, and (optionally) a report.

    ``resolved=None`` reproduces a run whose generation worked but whose grading
    never happened — the state ``--no-evaluate`` leaves behind.
    """
    repeat = variance.Repeat(index=index, results=root / f"repeat-{index}")
    repeat.results.mkdir(parents=True)
    run_id = f"{index:012x}"
    (repeat.results / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "status": "complete", "contract": contract or CONTRACT}),
        encoding="utf-8",
    )
    (repeat.results / "adapter_results.json").write_text(
        json.dumps(
            [
                {
                    "instance_id": instance_id,
                    "patch_bytes": 100 * index,
                    "duration_s": 10.0 * index,
                    "error": "AdapterError: turn 1 ended with timeout" if errors else None,
                    "harness_verification": verification,
                }
                for instance_id in INSTANCES
            ]
        ),
        encoding="utf-8",
    )
    if resolved is not None:
        report = {
            "total_instances": total_instances,
            "resolved_instances": len(resolved),
            "resolved_ids": sorted(resolved),
            "unresolved_ids": sorted(set(INSTANCES) - set(resolved)),
            "infra_failure_instances": infra_failures,
            "ambiguous_failure_instances": 0,
            "error_instances": 0,
            "empty_patch_instances": 0,
        }
        # Named exactly as swebench.harness.reporting names it.
        (repeat.results / f"yooumn194__MyCoder.mycoder-{run_id}.json").write_text(json.dumps(report), encoding="utf-8")
    return repeat


def _load(*repeats: variance.Repeat) -> dict:
    return variance.aggregate([variance.load_repeat(repeat) for repeat in repeats])


# ------------------------------------------------------------------ planning
def test_plan_numbering_and_the_floor_on_repeats(tmp_path):
    plan = variance.build_plan(tmp_path, repeats=3)

    assert [repeat.label for repeat in plan] == ["repeat-1", "repeat-2", "repeat-3"]
    assert plan[0].results == tmp_path / "repeat-1"
    with pytest.raises(ValueError, match="at least 1"):
        variance.build_plan(tmp_path, repeats=0)


# --------------------------------------------------- pass@1 over the repeats
def test_pass_at_1_is_the_mean_and_spread_of_the_official_verdicts(tmp_path):
    summary = _load(
        _write_repeat(tmp_path, 1, resolved=["pytest-dev__pytest-5262", "sympy__sympy-16886"]),
        _write_repeat(tmp_path, 2, resolved=INSTANCES),
        _write_repeat(tmp_path, 3, resolved=["pytest-dev__pytest-5262"]),
    )

    # 2/3, 3/3, 1/3 -> mean 2/3, population stddev sqrt(2/27).
    assert summary["pass_at_1"] == {"mean": 0.6667, "stddev": 0.2722, "n": 3}
    assert [entry["resolved"] for entry in summary["pass_at_1_by_repeat"]] == [2, 3, 1]
    assert summary["complete"] is True
    assert summary["graded_repeats"] == 3


def test_flaky_instances_are_those_resolved_in_some_repeats_only(tmp_path):
    summary = _load(
        _write_repeat(tmp_path, 1, resolved=["pytest-dev__pytest-5262", "sympy__sympy-16886"]),
        _write_repeat(tmp_path, 2, resolved=INSTANCES),
        _write_repeat(tmp_path, 3, resolved=["pytest-dev__pytest-5262"]),
    )

    assert summary["stable_resolved_ids"] == ["pytest-dev__pytest-5262"]
    assert summary["flaky_ids"] == ["django__django-11133", "sympy__sympy-16886"]
    assert summary["never_resolved_ids"] == []
    # 3 instances resolved at least once; only 1 resolved in every repeat.
    assert summary["union_resolved"] == 3
    assert summary["per_instance"]["sympy__sympy-16886"] == {"resolved": 2, "of": 3, "rate": 0.6667}


def test_one_repeat_reports_a_mean_but_no_spread_and_no_stability_claim(tmp_path):
    """A single run cannot support "0.0 variance"; it supports nothing about variance."""
    summary = _load(_write_repeat(tmp_path, 1, resolved=["pytest-dev__pytest-5262", "sympy__sympy-16886"]))

    assert summary["pass_at_1"] == {"mean": 0.6667, "stddev": None, "n": 1}
    assert summary["instability_measurable"] is False
    assert summary["flaky_ids"] == []
    report = variance.render_markdown(summary)
    assert "spread unmeasured" in report
    assert "absence of evidence, not stability" in report


# ------------------------------------------------------------ comparability
def test_repeats_configured_differently_are_refused(tmp_path):
    """The message names the field; "contracts differ" would send the operator diffing."""
    repeats = [
        _write_repeat(tmp_path, 1, resolved=["sympy__sympy-16886"]),
        _write_repeat(tmp_path, 2, resolved=["sympy__sympy-16886"]),
        _write_repeat(
            tmp_path,
            3,
            resolved=["sympy__sympy-16886"],
            contract={**CONTRACT, "suite_sha256": "b" * 64},
        ),
    ]

    with pytest.raises(variance.VarianceError, match=r"repeat-3 .*contract\.suite_sha256"):
        variance.aggregate([variance.load_repeat(repeat) for repeat in repeats])


def test_a_report_grading_a_different_instance_set_is_refused(tmp_path):
    repeats = [
        _write_repeat(tmp_path, 1, resolved=["sympy__sympy-16886"]),
        _write_repeat(tmp_path, 2, resolved=["sympy__sympy-16886"], total_instances=5),
    ]

    with pytest.raises(variance.VarianceError, match="graded 5 instance"):
        variance.aggregate([variance.load_repeat(repeat) for repeat in repeats])


def test_verdicts_for_instances_outside_the_plan_are_refused(tmp_path):
    repeat = _write_repeat(tmp_path, 1, resolved=["sympy__sympy-16886"])
    report_path = next(repeat.results.glob("*.mycoder-*.json"))
    report = json.loads(report_path.read_text())
    report["resolved_ids"] = ["some__other-repo-1"]
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(variance.VarianceError, match="outside the plan"):
        _load(repeat)


# ------------------------------------------------------------- partial runs
def test_an_ungraded_repeat_is_reported_rather_than_averaged_away(tmp_path):
    summary = _load(
        _write_repeat(tmp_path, 1, resolved=["pytest-dev__pytest-5262", "sympy__sympy-16886"]),
        _write_repeat(tmp_path, 2, resolved=None),
        _write_repeat(tmp_path, 3, resolved=["pytest-dev__pytest-5262"]),
    )

    assert summary["graded_repeats"] == 2
    assert summary["complete"] is False
    # Averaged over the two that WERE graded, and labelled as such.
    assert summary["pass_at_1"] == {"mean": 0.5, "stddev": 0.1667, "n": 2}
    assert summary["problems"] == ["repeat-2: the official harness wrote no report here (was --evaluate used?)"]
    # The denominator follows the graded repeats, so no instance claims 3 of 3.
    assert summary["per_instance"]["pytest-dev__pytest-5262"]["of"] == 2
    assert "INCOMPLETE" in variance.render_markdown(summary)


def test_a_missing_manifest_is_not_mistaken_for_a_missing_report(tmp_path):
    repeat = variance.Repeat(index=1, results=tmp_path / "repeat-1")
    repeat.results.mkdir(parents=True)

    summary = _load(repeat)

    assert summary["problems"] == ["repeat-1: the adapter wrote no manifest.json"]
    assert summary["pass_at_1"] == {"mean": None, "stddev": None, "n": 0}
    assert summary["never_resolved_ids"] == []


# --------------------------------------------------- finding the right report
def test_the_report_is_found_under_the_name_the_harness_uses(tmp_path):
    repeat = _write_repeat(tmp_path, 1, resolved=["sympy__sympy-16886"])
    manifest = json.loads((repeat.results / "manifest.json").read_text())

    assert adapter.official_report_name(manifest) == ("yooumn194__MyCoder.mycoder-000000000001.json")
    assert adapter.find_official_report(repeat.results, manifest).name == ("yooumn194__MyCoder.mycoder-000000000001.json")


def test_a_renamed_report_is_still_found(tmp_path):
    """A grading pass repeated over a subset gets renamed; that must not hide it."""
    repeat = _write_repeat(tmp_path, 1, resolved=["sympy__sympy-16886"])
    report_path = next(repeat.results.glob("*.mycoder-*.json"))
    report_path.rename(report_path.with_name("yooumn194__MyCoder.mycoder-000000000001-partial.json"))

    assert _load(repeat)["graded_repeats"] == 1


def test_two_candidate_reports_are_refused_rather_than_guessed(tmp_path):
    """Picking one would attribute another run's verdicts to this one."""
    repeat = _write_repeat(tmp_path, 1, resolved=["sympy__sympy-16886"])
    report_path = next(repeat.results.glob("*.mycoder-*.json"))
    # Renamed away from the name the harness itself would use, so the derived
    # name finds nothing and only the fallback glob sees these two.
    report_path.rename(report_path.with_name("some-other-run.mycoder-aaa.json"))
    (repeat.results / "another-run.mycoder-bbb.json").write_text("{}", encoding="utf-8")

    summary = _load(repeat)

    assert len(summary["problems"]) == 1
    assert "more than one official report" in summary["problems"][0]


# --------------------------------------------------------------- diagnostics
def test_generation_diagnostics_survive_a_run_that_was_never_graded(tmp_path):
    """--no-evaluate still buys the operator something: the generation side."""
    summary = _load(
        _write_repeat(tmp_path, 1, resolved=None, errors=True, verification={"status": "failed"}),
        _write_repeat(tmp_path, 2, resolved=None, errors=False, verification={"status": "unavailable"}),
    )

    assert summary["pass_at_1"]["n"] == 0
    assert summary["generation"]["success_rate"]["mean"] == 0.5
    assert summary["generation"]["error_distribution"] == {"AdapterError": 3}
    assert summary["harness_verification"] == {
        "verified_green": 0,
        "verified_red": 3,
        "verified_unavailable": 3,
        "self_certified": 0,
    }


def test_the_report_says_when_the_denominator_carries_infrastructure_noise(tmp_path):
    """An infra failure is counted against Pass@1, so it must be visible."""
    summary = _load(_write_repeat(tmp_path, 1, resolved=[], infra_failures=2))
    report = variance.render_markdown(summary)

    assert summary["non_verdict_counts"]["infra_failure_instances"] == 2
    assert "not verdicts about the patch" in report
    assert "| infra_failure | 2 |" in report


# ------------------------------------------------------------------ plumbing
def test_adapter_arguments_cannot_override_the_repeat_paths():
    repeat = variance.Repeat(index=2, results=Path("/tmp/var/repeat-2"))

    argv = variance.adapter_argv(repeat, ["--parallel", "2"], evaluate=True)

    assert argv[:2] == ["--parallel", "2"]
    # Appended last, so a caller cannot displace them.
    assert argv[-5:] == [
        "--results",
        "/tmp/var/repeat-2",
        "--harness-report-dir",
        "/tmp/var/repeat-2",
        "--evaluate",
    ]
    assert variance.adapter_argv(repeat, [], evaluate=False)[-1] == "/tmp/var/repeat-2"


def test_owned_flags_are_refused_rather_than_forwarded():
    """Forwarding --harness-report-dir would scatter the reports being aggregated."""
    with pytest.raises(SystemExit):
        variance.main(["--dry-run", "--harness-report-dir", "/tmp/elsewhere"])


def test_aggregating_existing_runs_writes_the_summary_next_to_them(tmp_path, capsys):
    repeats = [
        _write_repeat(tmp_path, 1, resolved=INSTANCES),
        _write_repeat(tmp_path, 2, resolved=["sympy__sympy-16886"]),
    ]

    code = variance.main([arg for repeat in repeats for arg in ("--run-dir", str(repeat.results))])

    assert code == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    # 3/3 and 1/3 -> mean 2/3, population stddev 1/3.
    assert summary["pass_at_1"] == {"mean": 0.6667, "stddev": 0.3333, "n": 2}
    assert (tmp_path / "summary.md").is_file()
    assert "Pass@1" in capsys.readouterr().out


def test_an_incomplete_run_exits_nonzero_unless_allowed(tmp_path, capsys):
    repeat = _write_repeat(tmp_path, 1, resolved=None)

    assert variance.main(["--run-dir", str(repeat.results)]) == 1
    assert "[variance] incomplete" in capsys.readouterr().out
    assert variance.main(["--run-dir", str(repeat.results), "--allow-partial"]) == 0
