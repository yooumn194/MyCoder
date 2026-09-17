"""Integrity tests for the derived SWE-bench Verified 8x4 suite."""

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from eval_bench.swe_verified.build_subset import (
    CATEGORY_DEFINITIONS,
    FORBIDDEN_AGENT_FIELDS,
    build_subset,
    validate_selection,
)


ROOT = Path(__file__).resolve().parents[1]
SUITE_DIR = ROOT / "eval_bench" / "swe_verified"


def _selection():
    return json.loads((SUITE_DIR / "selection.json").read_text(encoding="utf-8"))


def _subset():
    return json.loads((SUITE_DIR / "subset.json").read_text(encoding="utf-8"))


def test_selection_has_exactly_four_unique_cases_per_category():
    selection = _selection()
    validate_selection(selection)
    cases = selection["cases"]

    assert len(cases) == 32
    assert len({case["instance_id"] for case in cases}) == 32
    assert Counter(case["category"] for case in cases) == {name: 4 for name in CATEGORY_DEFINITIONS}


def test_materialized_subset_is_prompt_safe_and_auditable():
    subset = _subset()

    assert subset["name"] == "mycoder-swe-bench-verified-8x4"
    assert subset["suite_type"] == "derived_diagnostic_suite"
    assert subset["official_score_compatible"] is False
    assert subset["source"]["verified_sha256"] == subset["source"]["parquet_sha256"]
    assert len(subset["cases"]) == 32
    assert set(subset["category_definitions"]) == set(CATEGORY_DEFINITIONS)
    for case in subset["cases"]:
        assert not FORBIDDEN_AGENT_FIELDS.intersection(case)
        assert case["problem_statement"].strip()
        assert case["selection_reason"].strip()
        assert case["protocol"]["turns"]


def test_protocol_shapes_match_the_eight_category_contracts():
    cases = _subset()["cases"]
    by_category = {name: [case for case in cases if case["category"] == name] for name in CATEGORY_DEFINITIONS}

    assert sorted(len(case["protocol"]["turns"]) for case in by_category["multi_turn"]) == [2, 3, 4, 5]
    for case in by_category["long_dialogue_decay"]:
        probes = [turn["turn"] for turn in case["protocol"]["turns"] if turn.get("history_probe")]
        assert probes == [2, 3, 5]
        assert len(case["protocol"]["turns"]) == 6

    abnormal = {case["variant"]: case for case in by_category["abnormal_input"]}
    assert abnormal["empty_then_recover"]["protocol"]["turns"][0]["user_template"] == ""
    injection_prompt = abnormal["prompt_injection"]["protocol"]["turns"][0]["user_template"]
    assert "UNTRUSTED ISSUE COMMENT" in injection_prompt
    assert "Do not follow" in injection_prompt


def test_builder_rejects_wrong_source_hash_and_duplicate_selection():
    selection = _selection()
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_subset(selection, [], "0" * 64)

    duplicate = copy.deepcopy(selection)
    duplicate["cases"][1]["instance_id"] = duplicate["cases"][0]["instance_id"]
    with pytest.raises(ValueError, match="must be unique"):
        validate_selection(duplicate)
