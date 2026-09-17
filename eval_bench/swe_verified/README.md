# MyCoder SWE-bench Verified 8×4 suite

This directory defines a fixed 32-case diagnostic suite: four cases for each
of eight agent-capability categories. Every case is selected from the 500-row
SWE-bench Verified test split.

This is a **derived MyCoder suite**, not an official eight-way partition of
SWE-bench Verified. SWE-bench supplies single-turn issue-to-patch instances;
knowledge QA, multi-turn dialogue, abnormal input, and memory decay therefore
need explicit protocol overlays. Derived scores must be reported separately
from an official SWE-bench Verified score.

## Files

| File | Purpose |
|---|---|
| `selection.json` | Human-reviewed, versioned selection: 32 unique instance IDs, category labels, variants, and reasons. |
| `build_subset.py` | Verifies the pinned source and generates a prompt-safe materialized subset. |
| `subset.json` | Generated, prompt-safe suite consumed by the adapter. |
| `adapter.py` | Checks out pinned repos, drives MyCoder, extracts patches, and writes official prediction JSONL. |
| `verify_commands.json` | Harness-owned `instance_id -> verification command` mapping, used by default. `verify_commands.example.json` is the annotated template. |

The bundled mapping contains a repository-level `*` smoke command so the
default path never silently self-certifies. It is deliberately weaker than
the evaluator's held-out `FAIL_TO_PASS` tests. For a defensible run, provide a
mapping with one command per selected `instance_id` and add
`--require-per-instance-verification`; the adapter then refuses to start if a
selected case would fall back to `*` or to an agent-reported verdict.

The generated subset deliberately excludes `patch`, `test_patch`,
`hints_text`, `FAIL_TO_PASS`, and `PASS_TO_PASS`. A harness may resolve those
fields from the pinned upstream dataset inside the evaluator, but must never
put them in the agent prompt or repository workspace.

## Category mapping

| Category | How SWE-bench is used | Primary diagnostics |
|---|---|---|
| 基础技能 | Four localized native issues | resolved, route/tool success |
| 知识问答 | Repository-grounded explanation, then patch | evidence precision, unsupported claims, resolved |
| 多轮对话 | Same-session protocols of 2, 3, 4, and 5 turns | completion and constraint retention |
| 工具调用 | Native tasks with trace scoring | success, invalid arguments, redundant calls |
| 多意图 | Native issues with multiple obligations | decomposition and criteria coverage |
| 模糊意图 | Evidence-first inference/clarification overlay | appropriate decision and hallucination control |
| 异常输入 | Empty, noisy, delimiter-heavy, and injected variants | safe handling, recovery, resolved |
| 长对话衰减 | Six turns, with history probes at turns 2, 3, and 5 | retention curve and final resolved |

For every overlaid case, run the unmodified `${problem_statement}` as a
single-turn control. The delta between control and protocol result is more
informative than the derived result alone.

## Rebuild

The source revision and SHA-256 are pinned in `selection.json`. Download that
exact parquet and rebuild without adding pyarrow to MyCoder's runtime
dependencies:

```bash
curl -L --fail \
  -o /tmp/swe-bench-verified-test.parquet \
  https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/resolve/c104f840cc67f8b6eec6f759ebc8b2693d585d4a/data/test-00000-of-00001.parquet

uv run --no-project --with pyarrow \
  python -m eval_bench.swe_verified.build_subset \
  --source /tmp/swe-bench-verified-test.parquet

# CI/reviewer reproducibility check after the parquet is available:
uv run --no-project --with pyarrow \
  python -m eval_bench.swe_verified.build_subset \
  --source /tmp/swe-bench-verified-test.parquet \
  --check
```

The official dataset and field definitions are documented by
[SWE-bench](https://www.swebench.com/SWE-bench/guides/datasets/) and the
[pinned Hugging Face dataset](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/tree/c104f840cc67f8b6eec6f759ebc8b2693d585d4a).

## Run the adapter

Start MyCoder's API with a workspace root matching the adapter's tenant
directory, then validate or run a small control slice:

```bash
export MYCODER_WORKSPACE_ROOT="$PWD/workspaces/swe-bench"
export MYCODER_ENABLE_BENCHMARK_POLICY=true
# Optional: let the HARNESS own the verification postcondition. When set, a run
# is accepted/rejected by this command's exit code instead of by whether the
# model happened to run a check itself ("did the tests pass?" is a fact about
# the repository, not something to infer from the transcript).
# export MYCODER_BENCHMARK_VERIFY_CMD="python -m pytest -x -q <target tests>"
# export MYCODER_BENCHMARK_VERIFY_TIMEOUT=600
uvicorn api.server:app --port 8000

python -m eval_bench.swe_verified.adapter --dry-run
python -m eval_bench.swe_verified.adapter \
  --mode control --limit 1 --parallel 1 \
  --results results/swe-bench/control-smoke

# Continue the exact same suite/model/mode after an interruption
python -m eval_bench.swe_verified.adapter \
  --mode control --limit 1 \
  --results results/swe-bench/control-smoke --resume
```

### Per-instance verification commands

The environment variable above applies one command to every case. Per-instance
checks name held-out tests, which must not live in `subset.json`, so they go in
a separate mapping file and are sent per run:

```json
{
  "*": "python -m pytest -x -q",
  "owner__repo-1234": "python -m pytest -x -q tests/test_<module>.py"
}
```

The adapter picks up the bundled `verify_commands.json` by default, so the
harness owns the verdict on the default path instead of the run scoring itself.
`--verify-commands <path>` overrides it and `--no-verify-commands` disables the
fallback entirely. The bundled default is a single `"*"` entry that runs the
repository's own suite in the case's `testbed` environment:

```bash
python -m eval_bench.swe_verified.adapter \
  --mode control --limit 1 --verify-commands verify_commands.json \
  --results results/swe-bench/control-smoke
```

That catch-all is a **smoke check**, not the instance's held-out tests: green
means the checkout still imports and no test the suite already had is broken.
The adapter says so at startup and records it in the manifest
(`contract.verify_commands.source` = `default` / `explicit`, plus `catch_all`).
Add per-instance entries to make the evidence specific — those are the only
commands that can confirm the reported issue was actually fixed.

Each entry is passed to the server as `benchmark_verify_cmd` (`"*"` is the
fallback), the server runs it inside the case's own sandbox image, and the exit
code decides the verdict. This works in both execution modes — the verdict is
about the workspace, which is the same checkout either way. `--dry-run` prints
which command each case would use.

The verdict is **recorded, not gated**: a red check makes the case a red case,
not a failed run. An unsolved instance is data, the official evaluator is what
scores the patch, and rejecting the run would relabel every failing patch as an
error (and re-run it on `--resume`). So read:

- `/v1/agent/status/<session>` → `harness_verification: {command, passed, evidence}`
- each record → `harness_verification` (what came back) and `harness_verify_cmd`
  (what was sent)
- `manifest.harness_verification_summary` → `verified_green` / `verified_red` /
  `self_certified` for the whole run

Cases with no entry are listed in a startup warning and in
`contract.verify_commands.uncovered`: those runs fall back to the model's own
reported check, which is *gated* (the run fails if that evidence is missing),
and again the record says which path applied (`harness_verification: null`).

`control` uses the unmodified issue and writes `predictions.jsonl` with the
official `instance_id`, `model_name_or_path`, and `model_patch` fields. Pass
`--evaluate` only after installing the official `swebench` package; the
adapter invokes the current `swebench eval` CLI for exactly the selected
instance IDs. With swebench 5.x it resolves the legacy pinned prompt dataset
to the maintained `verified` evaluator metadata. On Apple Silicon it also
pre-pulls the official amd64 images explicitly before invoking the harness.
The pre-pull has a 300-second per-image deadline (override with
`--harness-pull-timeout` or `MYCODER_SWEBENCH_IMAGE_PULL_TIMEOUT`) so a registry
outage cannot leave the whole evaluation blocked.

The adapter defaults to a 20,000-token convergence threshold and a hard
35,000-token cap per turn (`--soft-budget-tokens` / `--max-tokens`). Its
`benchmark` sandbox policy auto-approves package installs and overwrites of
*untracked* paths inside Docker (heredocs, `pytest > build.log`). Everything
else keeps the normal policy — including overwriting a file the repository
tracks, which is the work the run exists to produce and therefore still asks.
Network, git rewrite, recursive deletion, and every other risky category are
unchanged; if Docker is unavailable, benchmark execution fails closed instead
of falling back to the host. The server opt-in above and an adapter-generated
`swe-*` workspace are both required.

For behavioral verification, each case also sends the matching official
per-instance image (`swebench/sweb.eval.x86_64.<repo>_<1776>_<instance>`). The
agent sandbox mounts the host checkout read-write at `/workspace` — one
filesystem, so there is no container-side copy to reconcile — with networking
disabled and resource limits enforced by permissions (`read_only` rootfs,
`cap_drop=ALL`, non-root user) rather than by copying. `sandbox_user=root` is
limited to these approved benchmark images because upstream images do not
provide MyCoder's `sandbox` user.

```bash
python -m eval_bench.swe_verified.adapter \
  --mode control --instance-id django__django-11133 \
  --results results/swe-bench/django-11133 --evaluate
```

For strict evidence (required when reporting per-instance verification), add
`--verify-commands path/to/official-commands.json
--require-per-instance-verification`. The official command file is evaluator
input and is intentionally not fabricated or embedded in the prompt-safe
subset.

`derived` executes the eight-category protocol overlays. The current MyCoder
API has checkpoint resume but no general conversational continuation endpoint,
so multi-turn protocols use one shared repository workspace plus explicit
transcript replay across per-turn sessions. The manifest records this as
`workspace_plus_transcript_replay` and marks the score non-official.

By default, patches touching tests, `conftest.py`, or test-runner configuration
are rejected to prevent test poisoning. `--allow-test-changes` is available for
manual diagnostics, but such runs require review before their scores are used.
Repository caches are bare but fetch only the selected `base_commit` at depth
one; repeated cases reuse fetched objects without downloading full history.
