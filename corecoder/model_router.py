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

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def tier_from_env() -> str:
        """Allow an explicit env override (CORECODER_MODEL_TIER)."""
        return os.getenv("CORECODER_MODEL_TIER", ModelTier.STANDARD.value)
