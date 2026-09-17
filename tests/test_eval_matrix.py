import json

from eval_bench._gen_dataset import build as build_dataset
from eval_bench.matrix import VARIANTS, aggregate, build_plan
from eval_bench.runner import BENCHMARK_INTEGRITY_VERSION


def test_ablation_plan_is_30_by_3_for_every_variant(tmp_path):
    plan = build_plan(tmp_path, repeats=3)
    assert len(plan) == 3 * len(VARIANTS)
    assert {item["variant"] for item in plan} == set(VARIANTS)
    assert all(item["repeat"] in (1, 2, 3) for item in plan)


def test_ablation_aggregate_reports_repeat_variance_and_failures(tmp_path):
    plan = build_plan(
        tmp_path,
        repeats=2,
        variants={"single_react": VARIANTS["single_react"]},
    )
    first = [
        {
            "id": "a",
            "agent_status": "success",
            "error_class": None,
            "duration_s": 1,
            "token_usage": 10,
            "benchmark_integrity_version": BENCHMARK_INTEGRITY_VERSION,
        },
        {
            "id": "b",
            "agent_status": "failed",
            "error_class": "TIMEOUT",
            "duration_s": 3,
            "token_usage": 30,
            "benchmark_integrity_version": BENCHMARK_INTEGRITY_VERSION,
        },
    ]
    second = [
        {
            "id": "a",
            "agent_status": "success",
            "error_class": None,
            "duration_s": 2,
            "token_usage": 20,
            "benchmark_integrity_version": BENCHMARK_INTEGRITY_VERSION,
        },
        {
            "id": "b",
            "agent_status": "success",
            "error_class": None,
            "duration_s": 2,
            "token_usage": 20,
            "benchmark_integrity_version": BENCHMARK_INTEGRITY_VERSION,
        },
    ]
    for item, records in zip(plan, (first, second)):
        item["results"].mkdir(parents=True)
        (item["results"] / "raw_results.json").write_text(json.dumps(records))

    result = aggregate(plan)["single_react"]
    assert result["samples"] == 4
    assert result["completed_repeats"] == 2
    assert result["pass_rate"] == {"mean": 0.75, "stddev": 0.25, "n": 2}
    assert result["duration_s"]["mean"] == 2.0
    assert result["failure_distribution"] == {"TIMEOUT": 1}


def test_dataset_uses_difficulty_aware_session_budgets():
    expected = {"easy": 96_000, "medium": 128_000, "hard": 160_000}
    problems = build_dataset()

    assert len(problems) == 30
    assert all(problem["max_tokens"] == expected[problem["difficulty"]] for problem in problems)
