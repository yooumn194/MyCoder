"""Model router: task complexity -> model tier.

Phase 3's "smart decision-maker" is also about spending model budget wisely.
The router maps a task description (+ an optional file count) onto a model
tier using YAML-configured keyword/file-count rules (optimization point #3).

The config file is hot-reloaded on mtime change, so editing
config/model_routing.yaml takes effect without a restart. If the file is
absent (installed wheel), a built-in default config is used.
"""

import os
from enum import Enum
from pathlib import Path

import yaml

from mycoder.resources import config_path
from mycoder.sandbox.logger import get_logger

logger = get_logger("mycoder.model_router")

DEFAULT_CONFIG_PATH = config_path("model_routing.yaml")

# Fallback used when the config file is not present (e.g. installed from a wheel).
DEFAULT_YAML = """\
tiers:
  fast: "gpt-5.4-mini"
  standard: "gpt-5.4"
  powerful: "gpt-5.5"

providers:
  openai:
    tiers:
      fast: "gpt-5.4-mini"
      standard: "gpt-5.4"
      powerful: "gpt-5.5"
    fallbacks:
      powerful: [standard, fast, base]
      standard: [fast, base]
      fast: [base]
  deepseek:
    tiers:
      fast: "deepseek-flash"
      standard: "deepseek-flash"
      powerful: "deepseek-flash"
    fallbacks:
      powerful: [standard, fast]
      standard: [fast]
      fast: []
  openrouter:
    tiers:
      fast: "openrouter/free"
      standard: "minimax/minimax-m3:free"
      powerful: "minimax/minimax-m3:free"
    fallbacks:
      powerful: [standard, fast, base]
      standard: [fast, base]
      fast: [base]

routing_rules:
  # 按优先级从上到下匹配，第一条命中即停
  - tier: powerful
    keywords: ["重构", "架构", "设计模式", "多文件迁移", "breaking change"]
    min_file_count: 3

  - tier: standard
    keywords: ["实现", "添加功能", "修复bug", "单元测试"]
    max_file_count: 2

  - tier: fast
    keywords: ["搜索", "读取", "格式化", "lint", "类型检查"]

  # 兜底
  - tier: standard
    keywords: []
"""


class ModelTier(str, Enum):
    FAST = "fast"
    STANDARD = "standard"
    POWERFUL = "powerful"


class ModelRouter:
    """Keyword + file-count routing rules with mtime-based hot reload."""

    def __init__(self, config_path: Path | str | None = None, load=yaml.safe_load) -> None:
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._load = load
        self._read()

    def _read(self) -> None:
        if self.config_path.exists():
            with open(self.config_path, encoding="utf-8") as fh:
                self.config = self._load(fh)
        else:
            self.config = self._load(DEFAULT_YAML)
        self._mtime = self.config_path.stat().st_mtime if self.config_path.exists() else None

    def _hot_reload(self) -> None:
        """Reload the config when the file changed on disk (no restart needed)."""
        if not self.config_path.exists():
            return
        mtime = self.config_path.stat().st_mtime
        if mtime != self._mtime:
            self._read()

    def classify_complexity(self, task_desc: str, file_count: int = 1) -> ModelTier:
        """First matching rule wins; empty-keyword rule is the absolute fallback."""
        self._hot_reload()
        for rule in self.config.get("routing_rules", []):
            kw_match = any(k in task_desc for k in rule.get("keywords", []))
            fc_match = True
            if "min_file_count" in rule:
                fc_match = fc_match and file_count >= rule["min_file_count"]
            if "max_file_count" in rule:
                fc_match = fc_match and file_count <= rule["max_file_count"]
            if kw_match and fc_match:
                return ModelTier(rule["tier"])
        return ModelTier.STANDARD  # 绝对兜底

    def model_for(self, task_desc: str, file_count: int = 1, provider: str | None = None) -> str | None:
        """Resolve a task to the concrete model name for its tier."""
        tier = self.classify_complexity(task_desc, file_count)
        return self.resolve_tier_model(tier.value, provider=provider)

    def _provider_config(self, provider: str | None) -> dict | None:
        providers = self.config.get("providers", {})
        if not providers or provider is None:
            return None
        return providers.get(provider, {}) or {}

    def resolve_tier_model(self, tier: str, provider: str | None = None) -> str | None:
        """Concrete model name for a given tier (used to build per-tier LLMs)."""
        self._hot_reload()
        provider_config = self._provider_config(provider)
        if provider_config is not None:
            return provider_config.get("tiers", {}).get(tier)
        return self.config.get("tiers", {}).get(tier)

    def resolve_candidates(self, tier: str, provider: str | None, base_model: str | None) -> list[str]:
        """Return a de-duplicated, same-provider primary/fallback model chain.

        Fallback entries can name another tier, a concrete model, or ``base``
        (the caller's configured model). Legacy configs without ``providers``
        or ``fallbacks`` keep their previous one-model behavior.
        """
        self._hot_reload()
        provider_config = self._provider_config(provider)
        tiers = provider_config.get("tiers", {}) if provider_config is not None else self.config.get("tiers", {})
        primary = tiers.get(tier)
        if not primary:
            return []
        refs = provider_config.get("fallbacks", {}).get(tier, []) if provider_config is not None else []
        models: list[str] = [primary]
        for ref in refs:
            if ref == "base":
                model = base_model
            else:
                model = tiers.get(ref, ref)
            if model and model not in models:
                models.append(model)
        return models

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def tier_from_env() -> str:
        """Allow an explicit env override (MYCODER_MODEL_TIER)."""
        return os.getenv("MYCODER_MODEL_TIER", ModelTier.STANDARD.value)


def build_model_factory(base_llm, router: ModelRouter | None = None):
    """Return ``factory(tier) -> LLM | None`` for sub-agent model-tier routing.

    Wires the ModelRouter (config/model_routing.yaml) into production: given a
    sub-agent's ``model_tier`` (fast/standard/powerful), build a same-class LLM
    for the tier's concrete model. Returns ``None`` when there is no tier, no
    model configured for it, or the tier model equals the base model — in all
    those cases the caller keeps using the shared ``base_llm`` (cost + behavior
    unchanged).

    The tier LLM inherits the base LLM's class (LLM vs LiteLLM), api_key /
    base_url (read off the OpenAI client for the plain ``LLM`` backend, which
    does not store them as attributes), tracer (so tier calls stay observable)
    and extra kwargs (temperature / max_tokens).

    A None base_llm (e.g. the API layer with no API key) yields a no-op
    factory — sub-agents then simply have no model and fail closed upstream.
    """
    router = router or ModelRouter()
    if base_llm is None:
        return lambda _tier: None
    if os.getenv("MYCODER_LOCK_BASE_MODEL", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        # Reproducible single-model calibration: every sub-agent inherits the
        # configured base model instead of silently changing model by role.
        return lambda _tier: base_llm

    def _api_key() -> str | None:
        key = getattr(base_llm, "api_key", None)
        if key:
            return key
        return getattr(getattr(base_llm, "client", None), "api_key", None)

    def _base_url() -> str | None:
        url = getattr(base_llm, "base_url", None)
        if url:
            return str(url)
        client_url = getattr(getattr(base_llm, "client", None), "base_url", None)
        return str(client_url) if client_url else None

    provider = _infer_provider(base_llm)

    def _build(model: str):
        if model == getattr(base_llm, "model", None):
            return base_llm
        if not getattr(base_llm, "builds_tier_clients", True):
            # A replay (or any mock) base LLM serves every tier itself.
            # Constructing a second client here would spend real tokens or hit
            # the network in the middle of a supposedly offline replay.
            return base_llm
        kwargs = dict(getattr(base_llm, "extra", {}))
        if hasattr(base_llm, "provider"):
            kwargs["provider"] = provider
        if hasattr(base_llm, "tool_dialect"):
            kwargs["tool_dialect"] = getattr(base_llm, "tool_dialect")
        # Build against the *undecorated* client, then restore the decorator
        # chain: recording has to survive a second client, or the run log would
        # be missing exactly the calls that are hardest to reproduce.
        from .observability.run_log import llm_core, llm_like

        core = llm_core(base_llm)
        return llm_like(
            base_llm,
            core.__class__(
                model=model,
                api_key=_api_key(),
                base_url=_base_url(),
                tracer=getattr(core, "_tracer", None),
                caller="model_router",
                **kwargs,
            ),
        )

    def factory(tier: str | None):
        if tier is None:
            return None
        try:
            models = router.resolve_candidates(tier, provider, getattr(base_llm, "model", None))
        except AttributeError:
            # Small third-party/legacy routers only expose resolve_tier_model.
            try:
                model = router.resolve_tier_model(tier, provider=provider)
            except TypeError:
                model = router.resolve_tier_model(tier)
            models = [model] if model else []
        if not models:
            return None
        candidates = [_build(model) for model in models]
        if len(candidates) == 1:
            return None if candidates[0] is base_llm else candidates[0]
        return FallbackLLM(candidates)

    return factory


def _infer_provider(llm) -> str:
    explicit = getattr(llm, "provider", None)
    if explicit:
        return str(explicit).lower()
    url = str(getattr(llm, "base_url", None) or getattr(getattr(llm, "client", None), "base_url", "")).lower()
    if "openrouter.ai" in url:
        return "openrouter"
    if "deepseek.com" in url:
        return "deepseek"
    if "localhost:11434" in url or "127.0.0.1:11434" in url:
        return "ollama"
    return "openai"


def _is_recoverable_model_error(exc: Exception) -> bool:
    """Only fail over for availability/rate/transport failures, never auth."""
    status = getattr(exc, "status_code", None)
    if status in {401, 402, 403}:
        return False
    if status in {404, 408, 409, 429} or (status is not None and status >= 500):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    name = type(exc).__name__.lower()
    if any(term in name for term in ("timeout", "connection", "ratelimit")):
        return True
    message = str(exc).lower()
    return any(
        term in message
        for term in (
            "model unavailable",
            "model not found",
            "model does not exist",
            "no endpoints found",
            "no available provider",
        )
    )


class FallbackLLM:
    """Try a provider-local model chain without replaying visible side effects."""

    def __init__(self, candidates: list) -> None:
        self.candidates = candidates
        self.model = candidates[0].model
        self.models = [candidate.model for candidate in candidates]
        self.provider = getattr(candidates[0], "provider", "openai")
        self.api_key = getattr(candidates[0], "api_key", None)
        self.base_url = getattr(candidates[0], "base_url", None)
        self.extra = getattr(candidates[0], "extra", {})
        self._tracer = getattr(candidates[0], "_tracer", None)
        self.caller = "model_router"
        self.selected_model = self.model
        self.fallback_count = 0

    @property
    def total_prompt_tokens(self) -> int:
        return sum(getattr(llm, "total_prompt_tokens", 0) for llm in self.candidates)

    @property
    def total_completion_tokens(self) -> int:
        return sum(getattr(llm, "total_completion_tokens", 0) for llm in self.candidates)

    @property
    def total_reasoning_tokens(self) -> int:
        return sum(getattr(llm, "total_reasoning_tokens", 0) for llm in self.candidates)

    @property
    def estimated_cost(self) -> float | None:
        costs = [getattr(llm, "estimated_cost", None) for llm in self.candidates]
        known = [cost for cost in costs if cost is not None]
        return sum(known) if known else None

    def chat(self, *args, **kwargs):
        for index, candidate in enumerate(self.candidates):
            emitted = False
            side_effect_started = False
            call_args = list(args)
            call_kwargs = dict(kwargs)

            on_token = call_kwargs.get("on_token")
            if len(call_args) >= 3:
                on_token = call_args[2]

            if on_token is not None:

                def tracked_token(token, callback=on_token):
                    nonlocal emitted
                    emitted = True
                    callback(token)

                if len(call_args) >= 3:
                    call_args[2] = tracked_token
                else:
                    call_kwargs["on_token"] = tracked_token

            predictive = call_kwargs.get("predictive_executor")
            if len(call_args) >= 5:
                predictive = call_args[4]
            if predictive is not None:

                def tracked_predictive(tool_call, callback=predictive):
                    nonlocal side_effect_started
                    side_effect_started = True
                    return callback(tool_call)

                if len(call_args) >= 5:
                    call_args[4] = tracked_predictive
                else:
                    call_kwargs["predictive_executor"] = tracked_predictive

            try:
                response = candidate.chat(*call_args, **call_kwargs)
                self.selected_model = candidate.model
                return response
            except Exception as exc:  # noqa: BLE001 - policy decides re-raise
                is_last = index == len(self.candidates) - 1
                if is_last or emitted or side_effect_started or not _is_recoverable_model_error(exc):
                    raise
                self.fallback_count += 1
                logger.warning(
                    "model_fallback",
                    provider=self.provider,
                    failed_model=candidate.model,
                    fallback_model=self.candidates[index + 1].model,
                    error_type=type(exc).__name__,
                )
        raise RuntimeError("model fallback chain is empty")
