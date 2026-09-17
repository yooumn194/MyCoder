"""Eval runner — black-box benchmark over the MyCoder HTTP API.

For each problem in dataset.json:
  1. create a unique API-scoped workspace containing only context_files;
  2. POST /v1/agent/run {task, session_id, max_tokens};
  3. poll GET /v1/agent/status/{session_id} until a terminal status
     (success | failed — see api/server.py's worker) or a local watchdog
     timeout (the API has no session-level timeout status);
  4. copy only declared outputs into a fresh verifier directory and run pytest
     with external configuration and plugin loading disabled.

Usage:
    python -m eval_bench.runner --base-url http://localhost:8000 --parallel 3
    python -m eval_bench.runner --dry-run          # validate dataset only
    python -m eval_bench.runner --resume --results results/run-...  # continue
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from .runtime_config import snapshot as runtime_config_snapshot

try:
    import httpx
except ImportError:  # pragma: no cover - fall back to urllib
    httpx = None  # type: ignore[assignment]

_TERMINAL_STATUSES = {"success", "failed"}
_POLL_INTERVAL = 2.0
_DEFAULT_RESULTS_ROOT = Path("results")
BENCHMARK_INTEGRITY_VERSION = 2

# How much of the agent's final answer to keep for the quality dimension. The
# judge reads this; truncation is recorded, not silent, because a score on a
# clipped answer is a score on something else.
ANSWER_MAX_CHARS = 8000

# Category/difficulty vocabularies (validated in --dry-run).
_CATEGORIES = {"bugfix", "refactor", "implement", "cross_file"}
_DIFFICULTIES = {"easy", "medium", "hard"}
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_WORKSPACE_MARKER = ".mycoder-eval-owned"
_VERIFIER_CONTROL_FILES = {
    "conftest.py",
    "pytest.ini",
    "pyproject.toml",
    "setup.cfg",
    "sitecustomize.py",
    "test_verify.py",
    "tox.ini",
    "usercustomize.py",
}


class VerificationIntegrityError(RuntimeError):
    """The agent workspace cannot be copied safely into the verifier."""


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
        if not isinstance(qid, str) or not _SAFE_ID_RE.fullmatch(qid):
            errors.append(f"{qid}: id must match {_SAFE_ID_RE.pattern}")
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
                reason = _unsafe_context_path(name)
                if reason:
                    errors.append(f"{qid}: unsafe context file {name!r}: {reason}")
                elif not str(name).endswith(".py"):
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


def _unsafe_context_path(name: object) -> str | None:
    if not isinstance(name, str) or not name:
        return "path must be a non-empty string"
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        return "path must be relative and cannot contain '.' or '..'"
    if "\\" in name:
        return "backslashes are not allowed"
    if path.name in _VERIFIER_CONTROL_FILES:
        return f"{path.name} can alter verifier behavior"
    return None


def effective_prompt(problem: dict) -> str:
    """Dataset prompt plus paths relative to the task's isolated API root."""
    paths = ", ".join(problem["context_files"])
    return (
        f"{problem['prompt']}\n\n"
        f"[isolated workspace] Edit only the requested task files. Agent-root-relative paths: {paths}"
    )


def scoped_workspace_id(prefix: str, qid: str, run_id: str) -> str:
    """Return a unique API workspace id without trusting a dataset id as a path."""
    if not _SAFE_ID_RE.fullmatch(prefix):
        raise ValueError(f"workspace id prefix must match {_SAFE_ID_RE.pattern}")
    slug = re.sub(r"[^A-Za-z0-9_.-]", "-", qid)[:24].strip("-.") or "case"
    digest = hashlib.sha256(qid.encode("utf-8")).hexdigest()[:8]
    suffix = f"-{slug}-{digest}-{run_id[:8]}"
    workspace_id = f"{prefix[: 64 - len(suffix)]}{suffix}"
    if not _SAFE_ID_RE.fullmatch(workspace_id):
        raise ValueError(f"generated invalid workspace id: {workspace_id!r}")
    return workspace_id


def prepare_workspace(parent: Path, workspace_id: str) -> Path:
    """Create a clean, exact child directory owned by this benchmark run."""
    root = parent.resolve()
    workdir = root / workspace_id
    if workdir.parent != root or workdir.is_symlink():
        raise ValueError("generated task workspace escaped its configured parent")
    if workdir.exists():
        marker = workdir / _WORKSPACE_MARKER
        if not marker.is_file() or marker.read_text(encoding="utf-8") != workspace_id:
            raise VerificationIntegrityError(f"refusing to delete unowned workspace: {workdir}")
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    (workdir / _WORKSPACE_MARKER).write_text(workspace_id, encoding="utf-8")
    return workdir


def _count_tests(test_code: str) -> int:
    return len(re.findall(r"^\s*def\s+test_", test_code, re.MULTILINE))


def _pytest_counts(out: str) -> tuple[int, int, int]:
    """Return (passed, failed, errors) parsed from a pytest -q summary."""
    def n(pattern: str) -> int:
        m = re.search(pattern, out)
        return int(m.group(1)) if m else 0

    return n(r"(\d+) passed"), n(r"(\d+) failed"), n(r"(\d+) error")


def _copy_agent_outputs(problem: dict, workdir: Path, verify_dir: Path) -> None:
    root = workdir.resolve()
    for name in problem["context_files"]:
        source = workdir / name
        if source.is_symlink() or not source.is_file():
            raise VerificationIntegrityError(f"required output is missing or not a regular file: {name}")
        try:
            source.resolve().relative_to(root)
        except ValueError as exc:
            raise VerificationIntegrityError(f"required output escapes task workspace: {name}") from exc
        target = verify_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def verify(problem: dict, workdir: Path) -> dict:
    """Verify declared outputs in a fresh, agent-unmodifiable pytest directory."""
    test_code = problem["verification"]["test_code"]
    total = _count_tests(test_code)
    if total <= 0:
        raise VerificationIntegrityError("verification contains no test functions")

    # Never execute pytest in the agent-controlled directory. In particular,
    # conftest.py, pytest.ini, or plugins could deselect tests or forge a pass.
    with tempfile.TemporaryDirectory(prefix="mycoder-eval-verify-") as temp_dir:
        verify_dir = Path(temp_dir)
        _copy_agent_outputs(problem, workdir, verify_dir)
        testfile = verify_dir / "test_verify.py"
        testfile.write_text(test_code, encoding="utf-8")
        config = verify_dir / "pytest.ini"
        config.write_text("[pytest]\naddopts =\n", encoding="utf-8")
        env = os.environ.copy()
        for variable in ("PYTHONPATH", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
            env.pop(variable, None)
        env.update({"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONDONTWRITEBYTECODE": "1"})
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "test_verify.py",
                "-q",
                "--no-header",
                "--tb=short",
                "-p",
                "no:cacheprovider",
                "-c",
                str(config),
                "--confcutdir",
                str(verify_dir),
                "--rootdir",
                str(verify_dir),
            ],
            cwd=verify_dir,
            capture_output=True,
            text=True,
            timeout=90,
            env=env,
        )
        out = f"{proc.stdout}\n{proc.stderr}"
    passed, failed, errors = _pytest_counts(out)
    return {
        "tests_passed": passed,
        "tests_total": total,
        "tests_collected": passed + failed,
        "failed": failed,
        "errors": errors,
        "exit_code": proc.returncode,
        "output": out[-2000:],
    }


def verification_passed(result: dict) -> bool:
    """Require a clean pytest exit and exact, complete test collection."""
    return (
        result["exit_code"] == 0
        and result["tests_collected"] == result["tests_total"]
        and result["tests_passed"] == result["tests_total"]
        and result["failed"] == 0
        and result["errors"] == 0
    )


def _headers() -> dict[str, str]:
    key = os.getenv("MYCODER_BENCH_API_KEY", "").strip()
    return {"X-API-Key": key} if key else {}


def _http_client(base_url: str):
    """Build a client that can reach a local API without proxy interception.

    The evaluation runner normally talks to a local uvicorn process.  Some
    managed environments transparently route Python HTTP traffic through a
    gateway that returns an empty 502 for loopback addresses; curl and the
    SWE-bench adapter already bypass that proxy.  Keep proxy support for real
    remote endpoints while making local black-box runs deterministic.
    """
    if httpx is None:
        return None
    host = (urlsplit(base_url).hostname or "").lower()
    return httpx.Client(trust_env=False) if host in {"localhost", "127.0.0.1", "::1"} else httpx.Client()


def _post_run(
    client,
    base_url: str,
    task: str,
    session_id: str,
    max_tokens: int,
    request_options: dict | None = None,
) -> tuple[int, dict]:
    payload = {"task": task, "session_id": session_id, "max_tokens": max_tokens}
    payload.update(request_options or {})
    if httpx is not None:
        resp = client.post(
            f"{base_url}/v1/agent/run",
            json=payload,
            headers=_headers(),
            timeout=30,
        )
        return resp.status_code, _safe_json(resp)
    import urllib.request

    body = json_dumps(payload)
    req = urllib.request.Request(
        f"{base_url}/v1/agent/run", data=body.encode(),
        headers={"Content-Type": "application/json", **_headers()},
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


def run_one(
    problem: dict,
    base_url: str,
    workspace: Path,
    results_dir: Path,
    run_id: str,
    client,
    variant: str = "default",
    request_options: dict | None = None,
    workspace_id_prefix: str = "bench",
) -> dict:
    """Execute one problem end-to-end and return its result record."""
    qid = problem["id"]
    api_workspace_id = scoped_workspace_id(workspace_id_prefix, qid, run_id)
    workdir = prepare_workspace(workspace, api_workspace_id)
    # Only public task inputs enter the isolated agent workspace.
    for name, content in problem["context_files"].items():
        path = workdir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    log = results_dir / "logs" / f"{qid}.log"
    _log(log, f"start problem={qid}")

    task = effective_prompt(problem)
    session_id = f"eval-{qid}-{run_id}"
    # Watchdogs measure elapsed duration, so they must not use the wall clock:
    # NTP/manual clock adjustments can otherwise expire a task immediately or
    # keep it alive past its limit.
    started = time.monotonic()
    deadline = started + int(problem["timeout_seconds"])
    perf: dict | None = None

    scoped_options = {**(request_options or {}), "workspace_id": api_workspace_id}
    status_code, resp = _post_run(
        client, base_url, task, session_id, int(problem["max_tokens"]), scoped_options
    )
    _log(log, f"POST /run -> {status_code} {resp}")
    if status_code != 202:
        return _result(
            problem, "failed", None, None, 0, None, "RUN_REJECTED",
            f"http {status_code}: {resp}", variant=variant, perf=perf,
            workspace_id=api_workspace_id,
        )

    agent_status = None
    token_usage = None
    error: dict | None = None
    perf: dict | None = None
    # The agent's final answer, kept so the scorer's quality dimension has
    # something to judge. Bounded so a chatty run cannot bloat raw_results.json.
    final_output: str | None = None
    while time.monotonic() < deadline:
        if httpx is not None:
            sr = client.get(
                f"{base_url}/v1/agent/status/{session_id}", headers=_headers(), timeout=30
            )
            if sr.status_code == 200:
                data = _safe_json(sr)
                agent_status = data.get("status")
                token_usage = data.get("token_usage")
                if data.get("output"):
                    final_output = data["output"]
                if data.get("error"):
                    error = data["error"]
                if data.get("perf"):
                    perf = data["perf"]
                _log(log, f"poll status={agent_status} token_usage={token_usage}")
        else:  # pragma: no cover - exercised only without the httpx extra
            import urllib.error
            import urllib.request

            req = urllib.request.Request(
                f"{base_url}/v1/agent/status/{session_id}", headers=_headers()
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as status_response:
                    data = json_loads(status_response.read())
                agent_status = data.get("status")
                token_usage = data.get("token_usage")
                if data.get("output"):
                    final_output = data["output"]
                error = data.get("error") or error
                perf = data.get("perf") or perf
            except urllib.error.HTTPError:
                pass
        time.sleep(_POLL_INTERVAL)
        if agent_status in _TERMINAL_STATUSES:
            break

    duration = round(time.monotonic() - started, 2)
    if agent_status not in _TERMINAL_STATUSES:
        agent_status = "timeout"
        _log(log, "watchdog: marked timeout (agent did not reach terminal status)")
        return _result(
            problem, "timeout", None, None, duration, token_usage, "TIMEOUT",
            "exceeded timeout_seconds", variant=variant, perf=perf,
            workspace_id=api_workspace_id, answer=final_output,
        )

    # verification
    if agent_status == "success":
        _log(log, "agent succeeded, running verification")
        try:
            v = verify(problem, workdir)
            _log(
                log,
                f"verify exit={v['exit_code']} collected={v['tests_collected']}/{v['tests_total']} "
                f"passed={v['tests_passed']} failed={v['failed']} errors={v['errors']}",
            )
            passed = verification_passed(v)
            error_cls = None if passed else "VERIFICATION_FAILED"
            error_msg = None if passed else (
                f"pytest exit={v['exit_code']} collected={v['tests_collected']}/{v['tests_total']} "
                f"passed={v['tests_passed']}/{v['tests_total']}"
            )
            return _result(
                problem, "success", v["tests_passed"], v["tests_total"], duration,
                token_usage, error_cls, error_msg, variant=variant, perf=perf,
                workspace_id=api_workspace_id, answer=final_output,
            )
        except subprocess.TimeoutExpired:
            return _result(
                problem, "success", None, None, duration, token_usage,
                "VERIFICATION_TIMEOUT", "pytest timed out", variant=variant,
                perf=perf, workspace_id=api_workspace_id, answer=final_output,
            )
        except VerificationIntegrityError as exc:
            return _result(
                problem, "success", None, None, duration, token_usage,
                "VERIFICATION_INTEGRITY", str(exc), variant=variant, perf=perf,
                workspace_id=api_workspace_id, answer=final_output,
            )

    error_cls = (error or {}).get("code") or "AGENT_FAILED"
    return _result(
        problem, "failed", None, None, duration, token_usage, error_cls,
        (error or {}).get("detail") or error_cls, variant=variant, perf=perf,
        workspace_id=api_workspace_id, answer=final_output,
    )


def _result(
    problem,
    agent_status,
    tests_passed,
    tests_total,
    duration,
    token_usage,
    error_cls,
    error_msg,
    variant="default",
    perf=None,
    workspace_id=None,
    answer=None,
) -> dict:
    return {
        "id": problem["id"],
        "category": problem["category"],
        "difficulty": problem["difficulty"],
        # The task text and the agent's final answer. Objective scoring needs
        # neither, but the quality dimension does: an LLM-as-Judge asked to
        # grade "correctness / faithfulness" has nothing to read otherwise, and
        # the transcript is not kept in the results directory.
        "question": problem.get("prompt"),
        "answer": (answer or "")[:ANSWER_MAX_CHARS] or None,
        "agent_status": agent_status,
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "duration_s": duration,
        "token_usage": token_usage,
        "error_class": error_cls,
        "error_msg": error_msg,
        "variant": variant,
        "perf": perf,
        "workspace_id": workspace_id,
        "benchmark_integrity_version": BENCHMARK_INTEGRITY_VERSION,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval_bench.runner", description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000", help="MyCoder API base URL")
    parser.add_argument("--dataset", default=str(Path(__file__).parent / "dataset.json"))
    parser.add_argument(
        "--workspace",
        default="workspaces/eval-api/local",
        help="host directory containing per-task API workspace roots",
    )
    parser.add_argument("--results", default=None, help="output dir (default results/<timestamp>)")
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="skip problems already recorded in --results/raw_results.json")
    parser.add_argument("--dry-run", action="store_true", help="validate the dataset schema and exit")
    parser.add_argument("--tag", default="default", help="variant label recorded on every result (for scorer --compare)")
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        metavar="ID",
        help="run only the listed problem ids (selection is frozen in manifest)",
    )
    parser.add_argument(
        "--workspace-id",
        default="bench",
        help="prefix for unique per-task API workspace ids (never use the API default root)",
    )
    parser.add_argument("--execution-mode", choices=("single", "multi"), default="multi")
    parser.add_argument(
        "--reasoning-strategy",
        choices=("auto", "react", "plan_execute", "reflection"),
        default="auto",
    )
    parser.add_argument(
        "--orchestration-strategy",
        choices=("auto", "sequential", "parallel", "conditional"),
        default="auto",
    )
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset).resolve()
    data = load_dataset(dataset_path)
    violations = validate_dataset(data)
    try:
        scoped_workspace_id(args.workspace_id, "schema-probe", "00000000")
    except ValueError as exc:
        violations.append(str(exc))
    if violations:
        for v in violations:
            print(f"[schema] {v}")
        print(f"[dry-run] dataset invalid: {len(violations)} violation(s)")
        return 1
    selected_ids = list(dict.fromkeys(args.ids or []))
    if selected_ids:
        by_id = {problem["id"]: problem for problem in data}
        unknown = [problem_id for problem_id in selected_ids if problem_id not in by_id]
        if unknown:
            print(f"[selection] unknown problem id(s): {', '.join(unknown)}")
            return 1
        data = [by_id[problem_id] for problem_id in selected_ids]
    if args.dry_run:
        print(f"[dry-run] runtime={runtime_config_snapshot()}")
        print(f"[dry-run] dataset OK: {len(data)} problems "
              f"({sum(1 for p in data if p['category']=='bugfix')} bugfix, "
              f"{sum(1 for p in data if p['category']=='refactor')} refactor, "
              f"{sum(1 for p in data if p['category']=='implement')} implement, "
              f"{sum(1 for p in data if p['category']=='cross_file')} cross_file)")
        return 0

    results_dir = Path(args.results) if args.results else _DEFAULT_RESULTS_ROOT / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "logs").mkdir(exist_ok=True)
    workspace = Path(args.workspace).resolve()
    run_id = uuid.uuid4().hex[:8]
    contract = {
        "benchmark_integrity_version": BENCHMARK_INTEGRITY_VERSION,
        "dataset": str(dataset_path),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "task_count": len(data),
        "base_url": args.base_url,
        "workspace_parent": str(workspace),
        "workspace_id_prefix": args.workspace_id,
        "variant": args.tag,
        "execution_mode": args.execution_mode,
        "reasoning_strategy": args.reasoning_strategy,
        "orchestration_strategy": args.orchestration_strategy,
        "runtime_config": runtime_config_snapshot(),
    }
    if selected_ids:
        contract["selected_ids"] = selected_ids
    manifest_path = results_dir / "manifest.json"
    raw_results_path = results_dir / "raw_results.json"
    if args.resume:
        if not manifest_path.exists():
            print("[resume] refusing results without an integrity-v2 manifest")
            return 1
        existing_manifest = json_load(manifest_path)
        if existing_manifest.get("contract") != contract:
            print("[resume] refusing to mix results from a different dataset or run configuration")
            return 1
    else:
        if raw_results_path.exists():
            print("[run] results already exist; use --resume with the identical configuration")
            return 1
        manifest_path.write_text(
            json_dumps(
                {
                    "schema_version": 1,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "contract": contract,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    done_ids: set[str] = set()
    if args.resume and raw_results_path.exists():
        previous_results = json_load(raw_results_path)
        incompatible = [
            result.get("id", "<unknown>")
            for result in previous_results
            if result.get("benchmark_integrity_version") != BENCHMARK_INTEGRITY_VERSION
        ]
        if incompatible:
            print(f"[resume] refusing legacy/unverified records: {', '.join(incompatible)}")
            return 1
        for r in previous_results:
            if r.get("agent_status") in ("success", "failed", "timeout"):
                done_ids.add(r["id"])
        print(f"[resume] skipping {len(done_ids)} already-completed problem(s)")

    client = _http_client(args.base_url)
    all_results: list[dict] = []
    todo = [p for p in data if p["id"] not in done_ids]
    print(f"[run] {len(todo)}/{len(data)} problems, parallel={args.parallel}, base_url={args.base_url}")
    print(f"[run] results -> {results_dir}")

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        request_options = {
            "execution_mode": args.execution_mode,
            "reasoning_strategy": args.reasoning_strategy,
            "orchestration_strategy": args.orchestration_strategy,
        }
        futures = {
            pool.submit(
                run_one, p, args.base_url, workspace, results_dir, run_id, client,
                args.tag, request_options, args.workspace_id,
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
                all_results.append(
                    {
                        "id": qid,
                        "agent_status": "failed",
                        "error_class": "RUNNER_ERROR",
                        "error_msg": str(exc),
                        "benchmark_integrity_version": BENCHMARK_INTEGRITY_VERSION,
                    }
                )

    # merge with any resumed results
    if done_ids and raw_results_path.exists():
        merged = list(json_load(raw_results_path)) + all_results
    else:
        merged = all_results
    out = raw_results_path
    out.write_text(json_dumps(merged, indent=2), encoding="utf-8")
    passed = sum(1 for r in merged if r.get("agent_status") == "success" and r.get("error_class") is None)
    print(f"[done] raw results -> {out}  ({passed}/{len(merged)} passed)")
    if client is not None:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
