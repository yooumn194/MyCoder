# MyCoder Eval Bench

End-to-end evaluation harness for MyCoder — a **black-box** benchmark that
drives the agent purely through the HTTP API and grades its code with pytest.

## What you get

| File | Purpose |
|---|---|
| `dataset.json` | 30 hand-written coding problems (10 bugfix / 8 refactor / 7 implement / 5 cross-file; 10 easy / 12 medium / 8 hard). Each has a self-contained English prompt, context files, and a deterministic pytest verification. Pure stdlib, no third-party deps. |
| `runner.py` | Executes the dataset: writes context files, POSTs `/v1/agent/run`, polls `/v1/agent/status` to a terminal state, then runs each problem's pytest verification. Supports `--parallel`, `--resume`, `--dry-run`. |
| `matrix.py` | Fixed 30-task × 3-repeat ablation: single ReAct / Plan-and-Execute / Reflection versus multi-agent AUTO. Writes a dataset hash/config manifest, repeat mean/stddev, and failure distributions. |
| `scorer.py` | Pass@1 statistics (overall / by category / by difficulty), failure-reason distribution, `summary.json`, `report.md`, optional matplotlib `chart.png`. |
| `_gen_dataset.py` | Generator that emits `dataset.json` (edit problems here and regenerate). |
| `p1_openrouter_perf.py` | OpenRouter online benchmark for P1-4 reasoning strategies and P1-5 polluted-memory correction. |
| `swe_verified/` | Prompt-safe 8×4 SWE-bench Verified suite plus an executable MyCoder-to-official-harness adapter. |

## P1-4 / P1-5 OpenRouter performance benchmark

The benchmark uses the project's OpenAI-compatible OpenRouter client, streams
every response, and records wall latency, TTFT, prompt/completion/reasoning
tokens and task-specific quality indicators. It never accepts the API key as a
CLI argument, so the key does not leak through shell history or the process
list.

Online runs write their JSON report to the path passed with `--report`.
Reports stay under the gitignored `results/` directory by default.

```bash
# Validate the matrix without network calls (15 planned calls by default)
python -m eval_bench.p1_openrouter_perf --dry-run

# Real OpenRouter run using minimax/minimax-m3:free
export OPENROUTER_API_KEY=sk-or-...
python -m eval_bench.p1_openrouter_perf \
  --model minimax/minimax-m3:free \
  --request-delay 1 \
  --report results/perf/p1-openrouter.json

# Run only one suite when free-tier limits are tight
python -m eval_bench.p1_openrouter_perf --suite reasoning
python -m eval_bench.p1_openrouter_perf --suite memory
```

P1-4 runs the same three-task matrix under ReAct, Plan-and-Execute and
Reflection. P1-5 creates gold-labelled conflicting memories, measures conflict
pair precision/recall, then compares OpenRouter answer accuracy before and
after the known polluted entry is deprecated. The gold label is explicit in
the report: conflict detection finds suspicious pairs; it does not pretend to
know automatically which side is true.

## How it works

### One-task control smoke

After starting the API, use the fixed smoke preset before spending budget on a
larger run. It records the effective provider/model/tool dialect and refuses to
resume if any frozen setting changes:

```bash
python -m eval_bench.smoke --dry-run
MYCODER_ENABLE_BENCHMARK_POLICY=true \
MYCODER_REQUIRE_AUTH=false \
MYCODER_WORKSPACE_ROOT="$PWD/workspaces/swe-bench" \
MYCODER_DEEPSEEK_THINKING=disabled \
python -m uvicorn api.server:app --host 127.0.0.1 --port 8000

# In another terminal, use the same thinking setting so the manifest matches.
MYCODER_DEEPSEEK_THINKING=disabled python -m eval_bench.smoke \
  --max-tokens 35000 --soft-budget-tokens 20000 \
  --results results/swe-bench/smoke \
  --thinking disabled
```

The smoke preset uses `execution-mode=single` and `react` to isolate the
provider/tool/API path. Benchmark requests enforce both a repository mutation
and a successful `execute_in_sandbox` verification command before the API
records success, so an untested or failing diff cannot be reported as a
completed agent run. Pass `--execution-mode multi` only when you explicitly
want to test orchestration as well.

For a completed generation run, reuse the result directory for official
grading without calling the model again:

```bash
MYCODER_DEEPSEEK_THINKING=disabled python -m eval_bench.smoke \
  --resume --evaluate --results results/swe-bench/smoke
```

Add `--evaluate` only after a non-empty patch is present and the official
SWE-bench package/Docker images are available.

1. **Start the API server with a dedicated evaluation root.** Never run the
   benchmark against the API's default project-root workspace: that would let
   the agent read `dataset.json` and its hidden verifier.

   ```bash
  export MYCODER_WORKSPACE_ROOT="$PWD/workspaces/eval-api"
  export MYCODER_REQUIRE_AUTH=false
  mkdir -p "$MYCODER_WORKSPACE_ROOT/local"
   export OPENAI_API_KEY=sk-...
   uvicorn api.server:app --port 8000
   ```

   With authentication enabled, replace the `local` directory below with the
   authenticated tenant id. The runner creates one opaque API workspace per
   task under that directory, so concurrent and repeated cases cannot share
   files.

2. **Run the benchmark** (in another terminal, same repo root):

   ```bash
   python -m eval_bench.runner --base-url http://localhost:8000 --parallel 3
   ```

   The default runner workspace is `workspaces/eval-api/local`, matching the
   unauthenticated local setup above. After the agent finishes, the runner
   copies only declared output files to a fresh temporary verifier directory.
   It does not execute pytest in the agent-controlled directory.

3. **Score the run:**

   ```bash
   python -m eval_bench.scorer --results results/<run-timestamp> --chart
   ```

4. **Run the full ablation matrix:**

   ```bash
   # Schema/config check only: plans 30 × 3 × 4 = 360 API runs
   python -m eval_bench.matrix --dry-run

   # Authenticated services read the key from env, never a CLI argument
   export MYCODER_BENCH_API_KEY=replace-with-a-secret
   python -m eval_bench.matrix --base-url http://localhost:8000
   ```

   `manifest.json` freezes the dataset SHA-256, model/provider/temperature,
   variants and repeat count. `summary.json` reports repeat-level pass-rate
   mean/stddev, latency/token mean/stddev, and failure-class distribution.

## CLI reference

```
runner.py
  --base-url URL     MyCoder API base (default http://localhost:8000)
  --dataset PATH     dataset.json (default eval_bench/dataset.json)
  --workspace PATH   parent of isolated per-task roots
                     (default workspaces/eval-api/local)
  --workspace-id ID  prefix for unique per-task API workspace ids
                     (default bench; this is not the API default workspace)
  --results DIR      output dir (default results/<timestamp>)
  --parallel N       concurrent problems (default 3)
  --ids ID [ID ...]  run only the listed problem ids (selection is frozen)
  --resume           skip problems already in --results/raw_results.json
  --dry-run          validate the dataset schema and exit
  --execution-mode   single | multi
  --reasoning-strategy auto | react | plan_execute | reflection
  --orchestration-strategy auto | sequential | parallel | conditional

scorer.py
  --results DIR      results/<run> directory (required)
  --chart            also render chart.png (needs matplotlib)
```

## Output layout

```
results/<timestamp>/
  raw_results.json    one record per problem: agent_status, tests_passed/total,
                      duration_s, token_usage, error_class, error_msg
  summary.json        Pass@1 overall + by category/difficulty + failure dist
  report.md           human-readable report
  logs/<id>.log       per-problem execution trace
  chart.png           optional category bar chart
```

## Status / error vocabulary

* Terminal `agent_status`: `success` | `failed` (from the API worker) and
  `timeout` (imposed by the runner's watchdog when a problem exceeds
  `timeout_seconds` — the API has no session-level timeout status).
* `error_class` on failures comes from the real envelope error codes
  (`CIRCUIT_BREAKER_OPEN`, `SUBAGENT_TIMEOUT`, `TOKEN_BUDGET_EXCEEDED`,
  `SUBAGENT_ERROR`, …) plus runner-side classes `RUN_REJECTED`, `TIMEOUT`,
  `VERIFICATION_FAILED`, `VERIFICATION_INTEGRITY`, `VERIFICATION_TIMEOUT`,
  `AGENT_FAILED`, `RUNNER_ERROR`.

## Notes

* **Benchmark isolation:** every case gets a clean API-scoped filesystem root
  containing only public task inputs. The source dataset, hidden tests,
  runner, and scorer remain outside the agent's file-tool authority.
* **Verifier integrity:** only declared outputs are copied to a fresh temporary
  directory; pytest configuration/plugin environment variables are cleared,
  plugin autoload is disabled, and a pass requires exit code 0 plus the exact
  expected number of collected and passing tests.
* **Determinism:** the API uses `temperature=0` by default, so a fixed dataset
  + fixed server should give reproducible passes. If you pass `--resume`, rerun
  results for completed problems are not overwritten (idempotent continuation).
* **Idempotence:** each problem gets a unique `session_id` per run and fresh
  context files, so re-runs never inherit stale state.
