# MyCoder Eval Bench

End-to-end evaluation harness for MyCoder — a **black-box** benchmark that
drives the agent purely through the HTTP API and grades its code with pytest.

## What you get

| File | Purpose |
|---|---|
| `dataset.json` | 30 hand-written coding problems (10 bugfix / 8 refactor / 7 implement / 5 cross-file; 10 easy / 12 medium / 8 hard). Each has a self-contained English prompt, context files, and a deterministic pytest verification. Pure stdlib, no third-party deps. |
| `runner.py` | Executes the dataset: writes context files, POSTs `/v1/agent/run`, polls `/v1/agent/status` to a terminal state, then runs each problem's pytest verification. Supports `--parallel`, `--resume`, `--dry-run`. |
| `scorer.py` | Pass@1 statistics (overall / by category / by difficulty), failure-reason distribution, `summary.json`, `report.md`, optional matplotlib `chart.png`. |
| `_gen_dataset.py` | Generator that emits `dataset.json` (edit problems here and regenerate). |

## How it works

1. **Start the API server** from the repo root (the file tools resolve paths
   against the server's cwd, so the workspace must live under the repo root):

   ```bash
   export OPENAI_API_KEY=sk-...
   uvicorn api.server:app --port 8000
   ```

2. **Run the benchmark** (in another terminal, same repo root):

   ```bash
   python -m eval_bench.runner --base-url http://localhost:8000 --parallel 3
   ```

   The runner writes each problem's context files to `eval_bench/workspace/{id}/`
   and instructs the agent to edit those files. After the agent finishes, the
   runner reads the files back and runs the problem's pytest verification.

3. **Score the run:**

   ```bash
   python -m eval_bench.scorer --results results/<run-timestamp> --chart
   ```

## CLI reference

```
runner.py
  --base-url URL     MyCoder API base (default http://localhost:8000)
  --dataset PATH     dataset.json (default eval_bench/dataset.json)
  --workspace PATH   workspace dir (default eval_bench/workspace)
  --results DIR      output dir (default results/<timestamp>)
  --parallel N       concurrent problems (default 3)
  --resume           skip problems already in --results/raw_results.json
  --dry-run          validate the dataset schema and exit

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
  `VERIFICATION_FAILED`, `VERIFICATION_TIMEOUT`, `AGENT_FAILED`, `RUNNER_ERROR`.

## Notes

* **Zero-intrusion:** nothing in `mycoder/` or `api/` is modified or imported
  by the harness; it talks to the agent only over HTTP.
* **Determinism:** the API uses `temperature=0` by default, so a fixed dataset
  + fixed server should give reproducible passes. If you pass `--resume`, rerun
  results for completed problems are not overwritten (idempotent continuation).
* **Idempotence:** each problem gets a unique `session_id` per run and fresh
  context files, so re-runs never inherit stale state.
