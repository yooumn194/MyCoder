"""Eval runner — black-box benchmark over the CoreCoder HTTP API.

For each problem in dataset.json:
  1. write context_files into {workspace}/{id}/ (the agent edits these in place
     on the server host — the server must run with cwd == the repo root);
  2. POST /v1/agent/run {task, session_id, max_tokens};
  3. poll GET /v1/agent/status/{session_id} until a terminal status
     (success | failed — see api/server.py's worker) or a local watchdog
     timeout (the API has no session-level timeout status);
  4. on success, run the problem's pytest verification against the edited files.

Usage:
    python -m eval_bench.runner --base-url http://localhost:8000 --parallel 3
    python -m eval_bench.runner --dry-run          # validate dataset only
    python -m eval_bench.runner --resume --results results/run-...  # continue
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

try:
    import httpx
except ImportError:  # pragma: no cover - fall back to urllib
    httpx = None  # type: ignore[assignment]

_TERMINAL_STATUSES = {"success", "failed"}
_POLL_INTERVAL = 2.0
_DEFAULT_RESULTS_ROOT = Path("results")

# Category/difficulty vocabularies (validated in --dry-run).
_CATEGORIES = {"bugfix", "refactor", "implement", "cross_file"}
_DIFFICULTIES = {"easy", "medium", "hard"}


def _log(log_path: Path, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now(timezone.utc).isoformat()}  {line}\n")


def load_dataset(path: Path) -> list[dict]:
    return json_load(path)


def json_load(path: Path) -> list[dict]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def validate_dataset(data: list[dict]) -> list[str]:
    """Return a list of schema violations (empty = valid)."""
    errors: list[str] = []
    seen: set[str] = set()
    for p in data:
        qid = p.get("id", "<no-id>")
        if qid in seen:
            errors.append(f"{qid}: duplicate id")
        seen.add(qid)
        for field in ("category", "difficulty", "prompt", "context_files", "verification"):
            if field not in p:
                errors.append(f"{qid}: missing '{field}'")
        if p.get("category") not in _CATEGORIES:
            errors.append(f"{qid}: bad category {p.get('category')!r}")
        if p.get("difficulty") not in _DIFFICULTIES:
            errors.append(f"{qid}: bad difficulty {p.get('difficulty')!r}")
        if not str(p.get("prompt", "")).strip():
            errors.append(f"{qid}: empty prompt")
        files = p.get("context_files")
        if not isinstance(files, dict) or not files:
            errors.append(f"{qid}: context_files must be a non-empty dict")
        else:
            for name, content in files.items():
                if not str(name).endswith(".py"):
                    errors.append(f"{qid}: context file {name} is not .py")
                if "pass" not in str(content):
                    pass  # fine — content may be any valid python
        ver = p.get("verification") or {}
        if ver.get("type") != "unit_test":
            errors.append(f"{qid}: verification.type must be 'unit_test'")
        if "def test_" not in str(ver.get("test_code", "")):
            errors.append(f"{qid}: verification.test_code has no test function")
        for field in ("timeout_seconds", "max_tokens"):
            if not isinstance(p.get(field), int) or p.get(field, 0) <= 0:
                errors.append(f"{qid}: {field} must be a positive int")
    return errors


def effective_prompt(problem: dict, workspace: Path) -> str:
    """Dataset prompt + the exact workspace-relative paths the agent must edit."""
    qid = problem["id"]
    paths = ", ".join(f"{workspace.relative_to(Path.cwd())}/{qid}/{name}"
                      for name in problem["context_files"])
    return f"{problem['prompt']}\n\n[workspace] Edit these files (paths relative to the project root): {paths}"


def _count_tests(test_code: str) -> int:
    return len(re.findall(r"^\s*def\s+test_", test_code, re.MULTILINE))


def _pytest_counts(out: str) -> tuple[int, int, int]:
    """Return (passed, failed, errors) parsed from a pytest -q summary."""
    def n(pattern: str) -> int:
        m = re.search(pattern, out)
        return int(m.group(1)) if m else 0

    return n(r"(\d+) passed"), n(r"(\d+) failed"), n(r"(\d+) error")


def verify(problem: dict, workdir: Path) -> dict:
    """Run the problem's pytest verification in workdir. Returns counts."""
    test_code = problem["verification"]["test_code"]
    testfile = workdir / "test_verify.py"
    testfile.write_text(test_code, encoding="utf-8")
    total = _count_tests(test_code)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "test_verify.py", "-q", "--no-header", "--tb=short"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=90,
    )
    out = f"{proc.stdout}\n{proc.stderr}"
    passed, failed, errors = _pytest_counts(out)
    return {
        "tests_passed": max(0, total - failed - errors),
        "tests_total": total,
        "failed": failed,
        "errors": errors,
        "exit_code": proc.returncode,
        "output": out[-2000:],
    }


def _post_run(client, base_url: str, task: str, session_id: str, max_tokens: int) -> tuple[int, dict]:
    if httpx is not None:
        resp = client.post(
            f"{base_url}/v1/agent/run",
            json={"task": task, "session_id": session_id, "max_tokens": max_tokens},
            timeout=30,
        )
        return resp.status_code, _safe_json(resp)
    import urllib.request

    body = json_dumps({"task": task, "session_id": session_id, "max_tokens": max_tokens})
    req = urllib.request.Request(
        f"{base_url}/v1/agent/run", data=body.encode(), headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, json_loads(r.read())


def _safe_json(resp) -> dict:
    try:
        return resp.json()
    except Exception:  # noqa: BLE001
        return {}


def json_dumps(data, **kw) -> str:
    import json

    return json.dumps(data, ensure_ascii=False, **kw)


def json_loads(raw) -> dict:
    import json

    return json.loads(raw)


def run_one(problem: dict, base_url: str, workspace: Path, results_dir: Path, run_id: str, client, variant: str = "default") -> dict:
    """Execute one problem end-to-end and return its result record."""
    qid = problem["id"]
    workdir = workspace / qid
    workdir.mkdir(parents=True, exist_ok=True)
    # fresh context files (clear stale state from a previous run)
    for name, content in problem["context_files"].items():
        (workdir / name).write_text(content, encoding="utf-8")
    log = results_dir / "logs" / f"{qid}.log"
    _log(log, f"start problem={qid}")

    task = effective_prompt(problem, workspace)
    session_id = f"eval-{qid}-{run_id}"
    deadline = time.time() + int(problem["timeout_seconds"])
    started = time.time()
    perf: dict | None = None

    status_code, resp = _post_run(client, base_url, task, session_id, int(problem["max_tokens"]))
    _log(log, f"POST /run -> {status_code} {resp}")
    if status_code != 202:
        return _result(problem, "failed", None, None, 0, None, "RUN_REJECTED", f"http {status_code}: {resp}", variant=variant, perf=perf)

    agent_status = None
    token_usage = None
    error: dict | None = None
    perf: dict | None = None
    while time.time() < deadline:
        if httpx is not None:
            sr = client.get(f"{base_url}/v1/agent/status/{session_id}", timeout=30)
            if sr.status_code == 200:
                data = _safe_json(sr)
                agent_status = data.get("status")
                token_usage = data.get("token_usage")
                if data.get("error"):
                    error = data["error"]
                if data.get("perf"):
                    perf = data["perf"]
                _log(log, f"poll status={agent_status} token_usage={token_usage}")
        time.sleep(_POLL_INTERVAL)
        if agent_status in _TERMINAL_STATUSES:
            break

    duration = round(time.time() - started, 2)
    if agent_status not in _TERMINAL_STATUSES:
        agent_status = "timeout"
        _log(log, "watchdog: marked timeout (agent did not reach terminal status)")
        return _result(problem, "timeout", None, None, duration, token_usage, "TIMEOUT", "exceeded timeout_seconds", variant=variant, perf=perf)

    # verification
    if agent_status == "success":
        _log(log, "agent succeeded, running verification")
        try:
            v = verify(problem, workdir)
            _log(log, f"verify tests={v['tests_passed']}/{v['tests_total']} failed={v['failed']} errors={v['errors']}")
            passed = v["tests_passed"] == v["tests_total"] and v["failed"] == 0 and v["errors"] == 0
            error_cls = None if passed else "VERIFICATION_FAILED"
            error_msg = None if passed else f"pytest {v['tests_passed']}/{v['tests_total']} passed"
            return _result(problem, "success", v["tests_passed"], v["tests_total"], duration, token_usage, error_cls, error_msg, variant=variant, perf=perf)
        except subprocess.TimeoutExpired:
            return _result(problem, "success", None, None, duration, token_usage, "VERIFICATION_TIMEOUT", "pytest timed out", variant=variant, perf=perf)

    error_cls = (error or {}).get("code") or "AGENT_FAILED"
    return _result(problem, "failed", None, None, duration, token_usage, error_cls, (error or {}).get("detail") or error_cls, variant=variant, perf=perf)


def _result(problem, agent_status, tests_passed, tests_total, duration, token_usage, error_cls, error_msg, variant="default", perf=None) -> dict:
    return {
        "id": problem["id"],
        "category": problem["category"],
        "difficulty": problem["difficulty"],
        "agent_status": agent_status,
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "duration_s": duration,
        "token_usage": token_usage,
        "error_class": error_cls,
        "error_msg": error_msg,
        "variant": variant,
        "perf": perf,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval_bench.runner", description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000", help="CoreCoder API base URL")
    parser.add_argument("--dataset", default=str(Path(__file__).parent / "dataset.json"))
    parser.add_argument("--workspace", default=str(Path(__file__).parent / "workspace"))
    parser.add_argument("--results", default=None, help="output dir (default results/<timestamp>)")
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="skip problems already recorded in --results/raw_results.json")
    parser.add_argument("--dry-run", action="store_true", help="validate the dataset schema and exit")
    parser.add_argument("--tag", default="default", help="variant label recorded on every result (for scorer --compare)")
    args = parser.parse_args(argv)

    data = load_dataset(Path(args.dataset))
    violations = validate_dataset(data)
    if violations:
        for v in violations:
            print(f"[schema] {v}")
        print(f"[dry-run] dataset invalid: {len(violations)} violation(s)")
        return 1
    if args.dry_run:
        print(f"[dry-run] dataset OK: {len(data)} problems "
              f"({sum(1 for p in data if p['category']=='bugfix')} bugfix, "
              f"{sum(1 for p in data if p['category']=='refactor')} refactor, "
              f"{sum(1 for p in data if p['category']=='implement')} implement, "
              f"{sum(1 for p in data if p['category']=='cross_file')} cross_file)")
        return 0

    results_dir = Path(args.results) if args.results else _DEFAULT_RESULTS_ROOT / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "logs").mkdir(exist_ok=True)
    workspace = Path(args.workspace)
    run_id = uuid.uuid4().hex[:8]

    done_ids: set[str] = set()
    if args.resume and (results_dir / "raw_results.json").exists():
        for r in json_load(results_dir / "raw_results.json"):
            if r.get("agent_status") in ("success", "failed", "timeout"):
                done_ids.add(r["id"])
        print(f"[resume] skipping {len(done_ids)} already-completed problem(s)")

    client = httpx.Client() if httpx is not None else None
    all_results: list[dict] = []
    todo = [p for p in data if p["id"] not in done_ids]
    print(f"[run] {len(todo)}/{len(data)} problems, parallel={args.parallel}, base_url={args.base_url}")
    print(f"[run] results -> {results_dir}")

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = {
            pool.submit(
                run_one, p, args.base_url, workspace, results_dir, run_id, client, args.tag
            ): p["id"]
            for p in todo
        }
        for fut in as_completed(futures):
            qid = futures[fut]
            try:
                result = fut.result()
                all_results.append(result)
                mark = "PASS" if result["agent_status"] == "success" and result["error_class"] is None else result["agent_status"].upper()
                print(f"[{mark}] {qid}  {result.get('tests_passed')}/{result.get('tests_total')}  {result['duration_s']}s  {result.get('error_class')}")
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] {qid}: {exc}")
                all_results.append({"id": qid, "agent_status": "failed", "error_class": "RUNNER_ERROR", "error_msg": str(exc)})

    # merge with any resumed results
    if done_ids and (results_dir / "raw_results.json").exists():
        merged = list(json_load(results_dir / "raw_results.json")) + all_results
    else:
        merged = all_results
    out = results_dir / "raw_results.json"
    out.write_text(json_dumps(merged, indent=2), encoding="utf-8")
    passed = sum(1 for r in merged if r.get("agent_status") == "success" and r.get("error_class") is None)
    print(f"[done] raw results -> {out}  ({passed}/{len(merged)} passed)")
    if client is not None:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
