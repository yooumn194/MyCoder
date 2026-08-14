"""TaskPlanner — LLM-driven task decomposition for the Orchestrator.

Turns an arbitrary user instruction into a dependency DAG of subagent
assignments. Design:

  * One non-streaming LLM call (`response_format={"type": "json_object"}`) via
    asyncio.to_thread, wrapped in a hard timeout (default 30s).
  * JSON parsing is tolerant: ```json fences are stripped, and the payload is
    coerced into a validated list[SubTask].
  * Every failure (bad JSON, invalid subagent name, dangling depends_on, DAG
    cycle, timeout) retries once, then falls back to a single explorer task so
    the Orchestrator always has a runnable plan.

Observability: structlog records task_id / llm_duration_ms / subtask_count /
retry_count, with a warning + reason on fallback. The plan itself is surfaced
to subagents through the shared blackboard (see orchestrator._decompose).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from mycoder.config import Config
from mycoder.llm import LLM
from mycoder.sandbox.logger import get_logger

from .planner_prompt import build_system_prompt, build_user_prompt

logger = get_logger("mycoder.planner")

# Subagent roles the planner may emit — must match BUILTIN_SUBAGENTS keys.
SubagentName = Literal["explorer", "planner", "implementer", "reviewer"]
VALID_SUBAGENT_NAMES: frozenset[str] = frozenset(
    ("explorer", "planner", "implementer", "reviewer")
)

PLANNER_TIMEOUT_SECONDS = 30.0
PLANNER_MAX_ATTEMPTS = 2  # initial attempt + one retry
MAX_SUBTASKS = 5

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class SubTask(BaseModel):
    """One node of the planned execution DAG (Pydantic V2)."""

    id: str = ""
    subagent_name: SubagentName
    instruction: str
    depends_on: list[str] = Field(default_factory=list)
    estimated_tokens: int = 0


def _extract_json(raw: str) -> str:
    """Strip a ```json ... ``` fence if present; otherwise return as-is."""
    if not raw:
        return ""
    match = _JSON_FENCE.search(raw)
    return match.group(1).strip() if match else raw.strip()


class TaskPlanner:
    def __init__(
        self,
        llm: LLM | None = None,
        *,
        llm_config: dict[str, Any] | None = None,
        timeout_seconds: float = PLANNER_TIMEOUT_SECONDS,
        max_subtasks: int = MAX_SUBTASKS,
    ) -> None:
        self._llm = llm if llm is not None else _resolve_llm(llm_config)
        self.timeout = timeout_seconds
        self.max_subtasks = max_subtasks

    # ------------------------------------------------------------------ API
    async def decompose(self, task: str, context: dict | None = None) -> list[SubTask]:
        """Return a validated plan. Never raises: every failure path ends in a
        single-explorer fallback so the orchestrator always has a plan."""
        task_id = (context or {}).get("task_id", "unknown")
        if self._llm is None:
            logger.warning(
                "planner_fallback",
                task_id=task_id,
                reason="no LLM configured",
                retry_count=0,
                subtask_count=1,
            )
            return [self._default_subtask(task)]

        retry_count = 0
        last_reason = "unknown"
        for attempt in range(1, PLANNER_MAX_ATTEMPTS + 1):
            started = time.monotonic()
            try:
                raw = await self._call_llm(task, context)
                subtasks = self._sanitize_subtasks(raw)
                self._validate_dag(subtasks)
                logger.info(
                    "planner_ok",
                    task_id=task_id,
                    llm_duration_ms=_elapsed_ms(started),
                    subtask_count=len(subtasks),
                    retry_count=retry_count,
                )
                return subtasks
            except Exception as exc:  # noqa: BLE001 - any failure retries/falls back
                retry_count += 1
                last_reason = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "planner_retry",
                    task_id=task_id,
                    llm_duration_ms=_elapsed_ms(started),
                    reason=last_reason,
                    attempt=attempt,
                    retry_count=retry_count,
                )

        logger.warning(
            "planner_fallback",
            task_id=task_id,
            reason=last_reason,
            retry_count=retry_count,
            subtask_count=1,
        )
        return [self._default_subtask(task)]

    # ------------------------------------------------------------- internals
    async def _call_llm(self, task: str, context: dict | None) -> str:
        """One non-streaming LLM call with a hard timeout. Runs in a thread
        because LLM.chat is a blocking stream; wait_for guards the deadline."""
        assert self._llm is not None
        messages = [
            {"role": "system", "content": build_system_prompt(self.max_subtasks)},
            {"role": "user", "content": build_user_prompt(task, context)},
        ]
        response = await asyncio.wait_for(
            asyncio.to_thread(
                self._llm.chat, messages, response_format={"type": "json_object"}
            ),
            timeout=self.timeout,
        )
        return str(getattr(response, "content", response))

    def _sanitize_subtasks(self, raw: str) -> list[SubTask]:
        """Parse + coerce the LLM output into a validated list[SubTask].

        Raises ValueError on anything unusable so decompose() can retry/fallback.
        """
        text = _extract_json(raw)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON from planner: {exc}") from exc
        if not isinstance(data, list) or not data:
            raise ValueError("planner output is not a non-empty JSON array")

        seen_ids: set[str] = set()
        subtasks: list[SubTask] = []
        for index, item in enumerate(data[: self.max_subtasks]):
            if not isinstance(item, dict):
                raise ValueError(f"subtask {index} is not an object")
            name = str(item.get("subagent_name", "")).strip()
            if name not in VALID_SUBAGENT_NAMES:
                raise ValueError(f"invalid subagent_name {name!r}")
            instruction = str(item.get("instruction", "")).strip()
            if not instruction:
                raise ValueError(f"subtask {index} has an empty instruction")

            st_id = str(item.get("id", "")).strip() or f"t{len(subtasks) + 1}"
            if st_id in seen_ids:  # duplicate ids break the DAG -> force unique
                st_id = f"{st_id}_{len(subtasks)}"
            seen_ids.add(st_id)

            depends_on = [str(d).strip() for d in item.get("depends_on", []) if str(d).strip()]
            estimated = _as_int(item.get("estimated_tokens", 0))
            subtasks.append(
                SubTask(
                    id=st_id,
                    subagent_name=cast(SubagentName, name),
                    instruction=instruction,
                    depends_on=depends_on,
                    estimated_tokens=estimated,
                )
            )
        if not subtasks:
            raise ValueError("planner returned zero usable subtasks")
        return subtasks

    def _validate_dag(self, subtasks: list[SubTask]) -> bool:
        """Kahn's algorithm over depends_on edges. Returns True; raises
        ValueError on dangling references, self-dependency, or a cycle."""
        ids = {st.id for st in subtasks}
        for st in subtasks:
            missing = set(st.depends_on) - ids
            if missing:
                raise ValueError(f"depends_on references unknown subtask: {sorted(missing)}")

        indegree = {st.id: 0 for st in subtasks}
        children: dict[str, list[str]] = {st.id: [] for st in subtasks}
        for st in subtasks:
            for dep in st.depends_on:
                if dep == st.id:
                    raise ValueError(f"subtask {st.id} depends on itself")
                children[dep].append(st.id)
                indegree[st.id] += 1

        queue = [st.id for st in subtasks if indegree[st.id] == 0]
        visited = 0
        while queue:
            node = queue.pop(0)
            visited += 1
            for child in children[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if visited != len(subtasks):
            raise ValueError("dependency cycle detected in plan")
        return True

    @staticmethod
    def _default_subtask(task: str) -> SubTask:
        return SubTask(id="t1", subagent_name="explorer", instruction=task)


# ---------------------------------------------------------------------------
def _resolve_llm(llm_config: dict[str, Any] | None) -> LLM | None:
    """Build an LLM from an explicit config dict, else from the environment.
    Returns None when no API key is available (planner degrades to fallback)."""
    cfg = Config.from_env()
    api_key = (llm_config or {}).get("api_key") or cfg.api_key
    if not api_key:
        return None
    model = str((llm_config or {}).get("model") or cfg.model)
    base_url = (llm_config or {}).get("base_url") or cfg.base_url
    return LLM(
        model=model,
        api_key=str(api_key),
        base_url=base_url,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
