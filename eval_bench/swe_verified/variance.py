"""Repeated, official-scored SWE-bench runs, reported as a mean and a spread.

A single SWE-bench run answers "how many did it resolve" — one draw, quoted as
if it were a property of the agent. A benchmark claim needs the other half: the
same instances, run again, so the number carries an observed spread and the
unstable instances have names. ``matrix.py`` already does this for the internal
30-task suite; this module does it for the SWE-bench path, which is the number
anyone actually quotes.

Pass@1 here comes from the official harness report — its ``resolved_ids`` — and
not from the adapter's own record. The adapter only establishes that it produced
a patch; the official evaluator decides whether the patch is correct. Reading
the score off the adapter would make "produced no patch" and "produced a wrong
patch" count the same, which is the same conflation P1-1 removed from the
verification verdict.

Two invariants keep the arithmetic honest:

* every repeat must carry an identical adapter ``contract``. The adapter already
  treats that dict as the identity of a run — ``--resume`` refuses when it
  changes — and repeats are held to the same rule because averaging over two
  different configurations measures the difference between them, not the
  agent's spread.
* a repeat without an official report contributes no pass rate at all. ``mean``
  over the repeats that were graded is published together with
  ``graded_repeats``, so a two-of-three run reads as two of three rather than as
  a three-repeat result.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ..stats import summarize
from . import adapter

DEFAULT_REPEATS = 3
DEFAULT_VARIANCE_ROOT = Path("results/swe-bench")

# Flags this module owns. They must not be forwarded to the adapter: a forwarded
# ``--results`` would let one repeat write into a sibling's directory, and a
# forwarded ``--harness-report-dir`` would scatter the reports the aggregation
# is about to read.
OWNED_FLAGS = frozenset(
    {
        "-h",
        "--help",
        "--repeats",
        "--results",
        "--run-dir",
        "--dry-run",
        "--allow-partial",
        "--evaluate",
        "--no-evaluate",
        "--harness-report-dir",
    }
)


@dataclass(frozen=True)
class Repeat:
    """One planned run of the whole instance set."""

    index: int
    results: Path

    @property
    def label(self) -> str:
        return f"repeat-{self.index}"


@dataclass
class RepeatRun:
    """A repeat's on-disk output, loaded."""

    repeat: Repeat
    manifest: dict[str, Any] | None = None
    records: list[dict[str, Any]] = field(default_factory=list)
    report: dict[str, Any] | None = None
    # Why this repeat carries no official verdict. Kept as a string rather than
    # a bare "not graded" so the summary can say which of the two happened: the
    # adapter never ran, or it ran and the harness never graded.
    problem: str | None = None

    @property
    def graded(self) -> bool:
        return self.report is not None

    @property
    def label(self) -> str:
        return self.repeat.label


class VarianceError(RuntimeError):
    """The repeats cannot be compared, so their spread would be meaningless."""


def build_plan(root: Path, repeats: int = DEFAULT_REPEATS) -> list[Repeat]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    return [Repeat(index=index, results=root / f"repeat-{index}") for index in range(1, repeats + 1)]


def load_repeat(repeat: Repeat) -> RepeatRun:
    """Read one repeat directory; never raises for a merely incomplete run."""
    run = RepeatRun(repeat=repeat)
    manifest_path = repeat.results / "manifest.json"
    if not manifest_path.is_file():
        run.problem = "the adapter wrote no manifest.json"
        return run
    try:
        run.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        run.problem = f"manifest.json is not valid JSON: {exc}"
        return run
    records_path = repeat.results / "adapter_results.json"
    if records_path.is_file():
        try:
            run.records = json.loads(records_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            run.problem = f"adapter_results.json is not valid JSON: {exc}"
            return run
    try:
        report_path = adapter.find_official_report(repeat.results, run.manifest)
    except adapter.AdapterError as exc:
        run.problem = str(exc)
        return run
    if report_path is None:
        run.problem = "the official harness wrote no report here (was --evaluate used?)"
        return run
    try:
        run.report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        run.problem = f"{report_path.name} is not valid JSON: {exc}"
        return run
    return run


def contract_mismatch(runs: Sequence[RepeatRun]) -> str | None:
    """Name the first contract field that differs between repeats, or None.

    Naming the field matters: "contracts differ" sends the operator diffing two
    large JSON files by hand, while "repeat-3 differs on contract.limit" is a
    one-line fix.
    """
    manifests = [run for run in runs if run.manifest is not None]
    if len(manifests) < 2:
        return None
    reference = manifests[0]
    for run in manifests[1:]:
        reference_contract = reference.manifest.get("contract") or {}
        other_contract = run.manifest.get("contract") or {}
        if reference_contract == other_contract:
            continue
        for key in sorted(set(reference_contract) | set(other_contract)):
            if reference_contract.get(key) != other_contract.get(key):
                return (
                    f"{run.label} differs from {reference.label} on contract.{key}: "
                    f"{other_contract.get(key)!r} != {reference_contract.get(key)!r}"
                )
        return f"{run.label} differs from {reference.label} on contract"
    return None


def _pinned_instance_ids(runs: Sequence[RepeatRun]) -> list[str]:
    for run in runs:
        contract = (run.manifest or {}).get("contract") or {}
        ids = contract.get("instance_ids")
        if isinstance(ids, list) and ids:
            return [str(item) for item in ids]
    return []


def _report_mismatch(run: RepeatRun, pinned: Sequence[str]) -> str | None:
    """Whether a report graded a different instance set than the plan pinned.

    ``total_instances`` is the harness's own count of what it was handed. If it
    disagrees with the pinned set, the repeat's pass rate has a different
    denominator and cannot be averaged with the others.
    """
    assert run.report is not None
    total = run.report.get("total_instances")
    if total != len(pinned):
        return f"{run.label} graded {total} instance(s), the plan pinned {len(pinned)}"
    graded: set[str] = set()
    for key in (
        "resolved_ids",
        "unresolved_ids",
        "infra_failure_ids",
        "ambiguous_failure_ids",
        "error_ids",
    ):
        graded.update(run.report.get(key) or [])
    unknown = sorted(graded - set(pinned))
    if unknown:
        return f"{run.label} reports verdicts for instances outside the plan: {', '.join(unknown[:3])}"
    return None


def _error_class(record: dict[str, Any]) -> str:
    """The exception type from an adapter error string, for the failure table."""
    error = record.get("error")
    if not error:
        return record.get("agent_status") or "UNKNOWN"
    return str(error).split(":", 1)[0].strip() or "UNKNOWN"


def aggregate(runs: Sequence[RepeatRun]) -> dict[str, Any]:
    """Fold per-repeat runs into pass@1 mean ± spread and per-instance stability.

    Raises ``VarianceError`` when the repeats are not comparable. Everything
    that is merely incomplete is reported instead of raised, so a partial run
    still produces the diagnostics its operator paid for.
    """
    if not runs:
        raise VarianceError("no repeats to aggregate")
    mismatch = contract_mismatch(runs)
    if mismatch:
        raise VarianceError(
            f"refusing to average repeats configured differently ({mismatch}); "
            "a spread over two configurations is a comparison, not variance"
        )
    pinned = _pinned_instance_ids(runs)
    graded = [run for run in runs if run.graded]
    if not pinned and graded:
        # A verdict with no recorded instance set cannot be attributed to an
        # instance, so there is no per-instance table and no honest denominator.
        raise VarianceError(
            "a repeat was graded but no repeat records the instance set it ran; cannot attribute verdicts to instances"
        )
    for run in graded:
        problem = _report_mismatch(run, pinned)
        if problem:
            raise VarianceError(f"refusing to average incomparable reports: {problem}")

    pass_rates: list[float] = []
    resolved_counts: list[float] = []
    per_repeat: list[dict[str, Any]] = []
    for run in graded:
        assert run.report is not None
        total = int(run.report.get("total_instances") or 0)
        resolved = int(run.report.get("resolved_instances") or 0)
        rate = resolved / total if total else 0.0
        pass_rates.append(rate)
        resolved_counts.append(float(resolved))
        per_repeat.append(
            {
                "repeat": run.label,
                "resolved": resolved,
                "submitted": total,
                "pass_at_1": round(rate, 4),
                "resolved_ids": sorted(run.report.get("resolved_ids") or []),
            }
        )

    hits: Counter[str] = Counter()
    for run in graded:
        assert run.report is not None
        hits.update(run.report.get("resolved_ids") or [])
    repeats_graded = len(graded)
    per_instance = {
        instance_id: {
            "resolved": hits.get(instance_id, 0),
            "of": repeats_graded,
            "rate": round(hits.get(instance_id, 0) / repeats_graded, 4) if repeats_graded else None,
        }
        for instance_id in pinned
    }
    # Instability is only observable from two graded repeats onward. With one
    # repeat every instance is trivially "consistent", which would read as a
    # stability finding rather than as an absence of evidence.
    measurable = repeats_graded > 1
    flaky = [instance_id for instance_id, stats in per_instance.items() if measurable and 0 < stats["resolved"] < repeats_graded]
    stable = [i for i, s in per_instance.items() if repeats_graded and s["resolved"] == repeats_graded]
    # With no graded repeat there is no evidence that an instance was never
    # resolved. Keep the per-instance denominator at zero and leave both
    # "stable" and "never" empty rather than turning missing grading into a
    # quality claim.
    never = [i for i, s in per_instance.items() if repeats_graded and s["resolved"] == 0]

    coverage = {
        key: sum(int(run.report.get(key) or 0) for run in graded)
        for key in (
            "infra_failure_instances",
            "ambiguous_failure_instances",
            "error_instances",
            "empty_patch_instances",
        )
    }

    generation_rates: list[float] = []
    durations: list[float] = []
    patch_bytes: list[float] = []
    errors: Counter[str] = Counter()
    verification = {"verified_green": 0, "verified_red": 0, "verified_unavailable": 0, "self_certified": 0}
    for run in runs:
        records = run.records
        if records:
            clean = sum(1 for record in records if record.get("error") is None)
            generation_rates.append(clean / len(records))
        for record in records:
            if record.get("error") is not None:
                errors[_error_class(record)] += 1
            if record.get("duration_s") is not None:
                durations.append(float(record["duration_s"]))
            if record.get("patch_bytes") is not None:
                patch_bytes.append(float(record["patch_bytes"]))
        for key, value in adapter.verification_summary(records).items():
            verification[key] += value

    return {
        "repeats_planned": len(runs),
        "graded_repeats": repeats_graded,
        "complete": repeats_graded == len(runs),
        "instances_per_repeat": len(pinned),
        "instances": pinned,
        "pass_at_1": summarize(pass_rates),
        "resolved_instances": summarize(resolved_counts),
        "pass_at_1_by_repeat": per_repeat,
        "per_instance": per_instance,
        "stable_resolved_ids": sorted(stable),
        "flaky_ids": sorted(flaky),
        "never_resolved_ids": sorted(never),
        "union_resolved": len([i for i in pinned if hits.get(i, 0)]),
        "instability_measurable": measurable,
        "non_verdict_counts": coverage,
        "generation": {
            "success_rate": summarize(generation_rates),
            "duration_s": summarize(durations),
            "patch_bytes": summarize(patch_bytes),
            "error_distribution": dict(errors),
        },
        "harness_verification": verification,
        "problems": [f"{run.label}: {run.problem}" for run in runs if run.problem],
    }


def _spread(metric: dict[str, Any]) -> str:
    """``mean ± stddev``, or an explicit note when the spread is unmeasured."""
    if metric.get("mean") is None:
        return "—"
    if metric.get("stddev") is None:
        return f"{metric['mean']} (single observation, spread unmeasured)"
    return f"{metric['mean']} ± {metric['stddev']}"


def render_markdown(summary: dict[str, Any], *, model: str | None = None) -> str:
    """Human-readable rendering of one variance report."""
    lines: list[str] = ["# SWE-bench Verified — repeated-run variance", ""]
    if model:
        lines.append(f"- Model: `{model}`")
    lines.append(f"- Instances per repeat: {summary['instances_per_repeat']}")
    lines.append(f"- Repeats planned / graded: {summary['repeats_planned']} / {summary['graded_repeats']}")
    if not summary["complete"]:
        lines.append(
            f"- **INCOMPLETE**: {summary['repeats_planned'] - summary['graded_repeats']} repeat(s) "
            "produced no official verdict, so these figures cover only the graded ones"
        )
    lines.append("")

    lines.extend(["## Pass@1 (official harness verdicts)", ""])
    lines.append("| repeat | resolved | submitted | pass@1 | resolved ids |")
    lines.append("|---|---|---|---|---|")
    for entry in summary["pass_at_1_by_repeat"]:
        ids = ", ".join(f"`{item}`" for item in entry["resolved_ids"]) or "—"
        lines.append(f"| {entry['repeat']} | {entry['resolved']} | {entry['submitted']} | {entry['pass_at_1']} | {ids} |")
    lines.append("")
    lines.append(f"**Pass@1: {_spread(summary['pass_at_1'])}**")
    lines.append("")

    non_verdict = summary["non_verdict_counts"]
    if any(non_verdict.values()):
        lines.extend(
            [
                "Counted against Pass@1 above, but not verdicts about the patch — these are "
                "outcomes the harness could not grade as right or wrong. If any is non-zero the "
                "denominator carries noise, not just wrong answers.",
                "",
                "| outcome | instances (summed over graded repeats) |",
                "|---|---|",
            ]
        )
        for key, value in non_verdict.items():
            lines.append(f"| {key.replace('_instances', '')} | {value} |")
        lines.append("")

    lines.extend(["## Per-instance stability", ""])
    if not summary["instability_measurable"]:
        lines.append(
            "Only one repeat was graded, so instability cannot be observed: every instance "
            "below is trivially consistent. This is an absence of evidence, not stability."
        )
        lines.append("")
    lines.append("| instance | resolved | of | rate |")
    lines.append("|---|---|---|---|")
    for instance_id, stats in sorted(summary["per_instance"].items(), key=lambda item: (item[1]["rate"], item[0])):
        lines.append(f"| `{instance_id}` | {stats['resolved']} | {stats['of']} | {stats['rate']} |")
    lines.append("")
    lines.append(f"- Resolved in every graded repeat: {len(summary['stable_resolved_ids'])}")
    lines.append(f"- Never resolved: {len(summary['never_resolved_ids'])}")
    flaky = summary["flaky_ids"]
    lines.append(
        f"- Resolved in some repeats only (flaky): {len(flaky)}"
        + (f" — {', '.join(f'`{item}`' for item in flaky)}" if flaky else "")
    )
    lines.append(
        f"- Resolved at least once / in every graded repeat: {summary['union_resolved']} / {len(summary['stable_resolved_ids'])}"
    )
    lines.append("")

    generation = summary["generation"]
    lines.extend(
        [
            "## Generation-side diagnostics",
            "",
            "What the adapter observed before any grading. A repeat with a low generation "
            "success rate is a statement about the harness, not about patch quality.",
            "",
            "| measure | value |",
            "|---|---|",
            f"| generation success rate | {_spread(generation['success_rate'])} |",
            f"| duration (s, per instance) | {_spread(generation['duration_s'])} |",
            f"| patch size (bytes) | {_spread(generation['patch_bytes'])} |",
        ]
    )
    if generation["error_distribution"]:
        failures = ", ".join(f"{name}×{count}" for name, count in sorted(generation["error_distribution"].items()))
        lines.append(f"| adapter errors | {failures} |")
    verdicts = summary["harness_verification"]
    lines.append(
        "| harness verification | "
        + ", ".join(f"{key.replace('verified_', '')} {value}" for key, value in verdicts.items())
        + " |"
    )
    lines.append("")

    if summary["problems"]:
        lines.extend(["## Repeats with no verdict", ""])
        lines.extend(f"- {problem}" for problem in summary["problems"])
        lines.append("")
    return "\n".join(lines)


def adapter_argv(repeat: Repeat, forwarded: Sequence[str], *, evaluate: bool) -> list[str]:
    """The adapter command line for one repeat.

    ``--results`` and ``--harness-report-dir`` are appended after the forwarded
    arguments: they are what makes the repeat legible to this module, so nothing
    a caller passes may override them.
    """
    argv = [*forwarded, "--results", str(repeat.results), "--harness-report-dir", str(repeat.results)]
    if evaluate:
        argv.append("--evaluate")
    return argv


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval_bench.swe_verified.variance",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Any argument this parser does not recognise is forwarded verbatim to\n"
            "eval_bench.swe_verified.adapter, so --suite / --base-url / --instance-id /\n"
            "--parallel and friends work here unchanged. Run\n"
            "  python -m eval_bench.swe_verified.adapter --help\n"
            "for the full list. The flags below belong to this module and are never\n"
            "forwarded: " + ", ".join(sorted(OWNED_FLAGS - {"-h", "--help"}))
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help=f"how many times to run the instance set (default: {DEFAULT_REPEATS})",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="root holding one repeat-N subdirectory per run (default: results/swe-bench/variance-<ts>)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "aggregate an already-finished repeat directory instead of running the "
            "adapter; repeat for each repeat. Mutually exclusive with --repeats"
        ),
    )
    parser.add_argument(
        "--evaluate",
        dest="evaluate",
        action="store_true",
        default=True,
        help="grade each repeat with the official harness (default)",
    )
    parser.add_argument(
        "--no-evaluate",
        dest="evaluate",
        action="store_false",
        help=(
            "generate patches only. No official report is produced, so the summary "
            "reports Pass@1 as unmeasured and keeps the generation diagnostics"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="exit 0 even when some planned repeat produced no official verdict",
    )
    args, forwarded = parser.parse_known_args(argv)

    leaked = sorted({flag.split("=", 1)[0] for flag in forwarded} & OWNED_FLAGS)
    if leaked:
        parser.error("these flags belong to this module and are never forwarded to the adapter: " + ", ".join(leaked))
    # A default that equals the natural value would hide an explicit --repeats,
    # and silently ignoring it next to --run-dir is worse than refusing.
    if args.run_dir and args.repeats is not None:
        parser.error("--run-dir aggregates existing runs; it cannot be combined with --repeats")
    repeats = DEFAULT_REPEATS if args.repeats is None else args.repeats
    if repeats < 1:
        parser.error("--repeats must be at least 1")

    if args.run_dir:
        plan = [Repeat(index=index, results=path.resolve()) for index, path in enumerate(args.run_dir, 1)]
        root = None
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        root = Path(args.results or DEFAULT_VARIANCE_ROOT / f"variance-{timestamp}").resolve()
        plan = build_plan(root, repeats=repeats)

    if args.dry_run:
        print(f"[variance] {len(plan)} repeat(s) x the adapter's selected instances")
        for repeat in plan:
            print(f"  {repeat.label} -> {' '.join(adapter_argv(repeat, forwarded, evaluate=args.evaluate))}")
        print("[variance] no run started (--dry-run)")
        return 0

    if args.run_dir:
        print(f"[variance] aggregating {len(plan)} existing run director(ies)")
    else:
        existing = [repeat.results for repeat in plan if repeat.results.exists()]
        if existing and "--resume" not in forwarded:
            print(
                "[variance] refusing to write into existing repeat directories "
                f"({', '.join(str(path) for path in existing[:3])}); pass --resume to "
                "continue them, or --run-dir to aggregate them"
            )
            return 1
        assert root is not None
        root.mkdir(parents=True, exist_ok=True)
        # Each repeat is a complete adapter run, so a failure to start one must
        # not silently shrink the sample: stop, then report exactly how many
        # repeats were graded.
        for repeat in plan:
            command = adapter_argv(repeat, forwarded, evaluate=args.evaluate)
            print(f"[variance] {repeat.label}: adapter {' '.join(command)}")
            code = adapter.main(command)
            if code != 0:
                print(f"[variance] {repeat.label} exited {code}; not starting further repeats")
                break

    runs = [load_repeat(repeat) for repeat in plan]
    try:
        summary = aggregate(runs)
    except VarianceError as exc:
        print(f"[variance] {exc}")
        return 1
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["adapter_arguments"] = list(forwarded)
    summary["evaluated"] = args.evaluate

    destination = root or plan[0].results.parent
    (destination / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render_markdown(summary)
    (destination / "summary.md").write_text(report + "\n", encoding="utf-8")
    print()
    print(report)
    print()
    print(f"[variance] summary -> {destination / 'summary.json'}")
    if not summary["complete"] and not args.allow_partial:
        print(
            f"[variance] incomplete: {summary['graded_repeats']}/{summary['repeats_planned']} "
            "repeat(s) graded; pass --allow-partial to accept this"
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
