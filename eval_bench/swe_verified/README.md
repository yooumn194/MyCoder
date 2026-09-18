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
| `variance.py` | Runs the adapter N times over one instance set and reports Pass@1 as mean ± population stddev, plus which instances were flaky. |
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
`benchmark` sandbox policy auto-approves overwrites of *untracked* paths inside
Docker (heredocs, `pytest > build.log`), and auto-approves package installs
**only when the container actually has egress**
(`MYCODER_SANDBOX_NETWORK != none`) — with networking off, approving an install
would just guarantee a silent failure. Everything
else keeps the normal policy — including overwriting a file the repository
tracks, which is the work the run exists to produce and therefore still asks.
Network, git rewrite, recursive deletion, and every other risky category are
unchanged; if Docker is unavailable, benchmark execution fails closed instead
of falling back to the host. The server opt-in above and an adapter-generated
`swe-*` workspace are both required.

`--evaluate` also pins where the harness writes its report:
`--harness-report-dir`, defaulting to this run's `--results` directory. The
harness default is the current working directory, which drops one
`<model>.<run-id>.json` into whatever directory the operator was standing in —
normally the repository root.

## Repeated runs and variance

A single run is one draw. `variance.py` runs the same instance set N times and
reports the spread, so the number is quotable as `mean ± stddev` instead of as
a single hit count:

```bash
# Plan only — prints the exact per-repeat adapter command
python -m eval_bench.swe_verified.variance --repeats 3 --dry-run

# Three identical runs, each graded by the official harness
python -m eval_bench.swe_verified.variance --repeats 3 \
  --base-url http://localhost:8000 --parallel 1 \
  --instance-id pytest-dev__pytest-5262 --instance-id sympy__sympy-16886
```

Anything the variance parser does not recognise is forwarded verbatim to
`adapter.py`, so `--suite`, `--parallel`, `--mode` and the rest work unchanged;
`--help` lists the flags it owns.

Each repeat is a full adapter run under `<results>/repeat-N`, graded
separately, with its own report next to its own predictions. `summary.json` and
`summary.md` then report:

- **Pass@1** as mean ± population standard deviation over the repeats, taken
  from the official report's `resolved_ids` — never from the adapter's own
  record, which only establishes that a patch was produced.
- **Per-instance stability**: resolved in every repeat, resolved in some
  (`flaky_ids` — the most actionable output of a repeat run), or never.
- **Generation-side diagnostics**: success rate, per-instance duration, patch
  size, adapter error classes and the harness verdict counts.

Three rules keep the arithmetic meaningful:

| Rule | Why |
|---|---|
| Every repeat must carry an identical adapter `contract` | Averaging over two configurations measures the difference between them, not the agent's spread. The message names the differing field. |
| Every repeat's official report must cover the same instance set | A different denominator cannot be averaged with the others. |
| A repeat with no report is reported as ungraded, not folded in | `graded_repeats` travels with `mean`, so a two-of-three run reads as two of three. |

`stddev` is reported as `null` (and the markdown says "spread unmeasured") for
a single repeat, and stability is explicitly labelled unmeasurable: `0.0` from
one observation would read as perfect stability rather than as no evidence.

Use `--no-evaluate` for a generation-only run — the Pass@1 axis then reports as
unmeasured while the generation diagnostics are still produced. Use
`--run-dir <dir>` (repeatable) to aggregate repeats that already exist, for
instance when they were run on different days; the summary lands in their
common parent directory.

### Diagnosing a case with trace replay

`stddev` and `flaky_ids` say *which* case moved; they do not say why. Trace
replay answers that offline — no provider, no tokens, no repeat cost.

Record first. The run log is opt-in on the server, one JSONL per session:

```bash
export MYCODER_RUN_LOG_DIR="$PWD/results/run-logs"
uvicorn api.server:app --port 8000
```

Then re-execute the *deterministic* half of one recorded run against the same
checkout. LLM responses come from the log; the tool layer, convergence
controller, requirement gates, and context compression all run for real:

```bash
python -m mycoder.replay results/run-logs/swe-<id>.jsonl --list-runs
python -m mycoder.replay results/run-logs/swe-<id>.jsonl \
  --workspace workspaces/swe-bench/local/swe-<id> \
  --sandbox-policy benchmark --sandbox-image <same image> --sandbox-user root \
  --json results/run-logs/<id>.replay.json
```

Replay needs the *post-run* checkout — reproducing the tool observations means
starting from the tree the recording ended with, not from `base_commit` — so
record with `--keep-workspaces`, then point `--workspace` at that directory.

The tool catalog is part of the compared fingerprint, so the replay must build
the same one. When it does not, the report says so with counts
(`tool catalog changed (3 recorded vs 20 replayed)`) rather than reporting an
opaque prompt mismatch — that is the signal that `--sandbox-policy`,
`--sandbox-image` or `--sandbox-user` are not the ones the run actually used.

`--list-runs` prints the run index, prompt preview, exchange/tool counts and
execution mode for every run in the file; `--run-index` then selects which one
to replay. The exit code is the verdict — `0` reproduced, `1` diverged, `2`
unsupported / incomplete / error — so it can gate a CI step. Use
`--no-execute` to analyse a log without re-running anything; its headline then
reads `ANALYSED (not re-run)` rather than `MATCH`, since nothing was compared.

A `match` means the action layer reproduced the recording exactly — every tool
observation and loop-injected control message identical in order and content —
which is what makes a `flaky_ids` entry a sampling artifact rather than a
harness bug. A divergence reports positions rather than a diff: the first place
the replay stopped agreeing (tool call *N* returned different output, at first
differing line *M*; a tool call the recording never made; a control message that
changed), and for an altered LLM request the fingerprinted request the replayed
loop asked for against the one recorded. A recorded provider failure replays as
that same failure, so error paths are exercised too. Read the report's notes
before blaming the agent — replay reports environment drift (workspace root,
cwd, sandbox policy) whenever it cannot be reproduced, and three further limits
are structural rather than bugs:

- **Delegated runs are refused**, not approximated: a run recorded in
  `execution_mode=multi` replays as `unsupported`.
- **Memory-enabled runs always report a planner divergence.** Retrieval is not
  reproducible (embeddings plus store contents), so a planner prompt that
  consumed a memory section cannot be served from the log.
- **Host-dependent tool output diverges by design** — anything reading absolute
  paths or repository state outside the replayed workspace.

Only the LLM is frozen: a replayed tool call still touches the real filesystem
and the real container, so replay mutates the checkout it is pointed at. Give
it a throwaway copy if the recorded run is what produced `predictions.jsonl`.

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
