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

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "model_routing.yaml"
)

# Fallback used when the config file is not present (e.g. installed from a wheel).
DEFAULT_YAML = """\
tiers:
  fast: "ollama/qwen2.5-coder:7b"
  standard: "deepseek-ai/DeepSeek-V4-Flash"
  powerful: "claude-opus-5"

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
        self._mtime = (
            self.config_path.stat().st_mtime if self.config_path.exists() else None
        )

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

    def model_for(self, task_desc: str, file_count: int = 1) -> str | None:
        """Resolve a task to the concrete model name for its tier."""
        tier = self.classify_complexity(task_desc, file_count)
        return self.config.get("tiers", {}).get(tier.value)

    def resolve_tier_model(self, tier: str) -> str | None:
        """Concrete model name for a given tier (used to build per-tier LLMs)."""
        self._hot_reload()
        return self.config.get("tiers", {}).get(tier)

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

    def factory(tier: str | None):
        if tier is None:
            return None
        model = router.resolve_tier_model(tier)
        if not model or model == getattr(base_llm, "model", None):
            return None
        return base_llm.__class__(
            model=model,
            api_key=_api_key(),
            base_url=_base_url(),
            tracer=getattr(base_llm, "_tracer", None),
            caller="model_router",
            **dict(getattr(base_llm, "extra", {})),
        )

    return factory
