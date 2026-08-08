"""Tests for the Phase 3 model router (YAML rules + hot reload)."""

import os

import yaml

from corecoder.model_router import ModelRouter, ModelTier

_CONFIG = """\
tiers:
  fast: "fast-model"
  standard: "std-model"
  powerful: "big-model"

routing_rules:
  - tier: powerful
    keywords: ["重构", "架构", "multi-file"]
    min_file_count: 3
  - tier: standard
    keywords: ["实现", "fix bug"]
    max_file_count: 2
  - tier: fast
    keywords: ["搜索", "lint"]
  - tier: standard
    keywords: []
"""


def _router(tmp_path, content=_CONFIG):
    path = tmp_path / "model_routing.yaml"
    path.write_text(content, encoding="utf-8")
    return ModelRouter(config_path=path), path


def test_powerful_only_with_file_count(tmp_path):
    router, _ = _router(tmp_path)
    # "重构" with 1 file: powerful rule's min_file_count=3 not met -> fall to standard
    assert router.classify_complexity("重构 handler", file_count=1) == ModelTier.STANDARD
    # 3+ files -> powerful
    assert router.classify_complexity("重构 handler", file_count=3) == ModelTier.POWERFUL


def test_standard_and_fast(tmp_path):
    router, _ = _router(tmp_path)
    assert router.classify_complexity("实现登录", file_count=1) == ModelTier.STANDARD
    assert router.classify_complexity("搜索函数", file_count=1) == ModelTier.FAST
    assert router.classify_complexity("lint 检查", file_count=1) == ModelTier.FAST


def test_unknown_falls_back_to_standard(tmp_path):
    router, _ = _router(tmp_path)
    assert router.classify_complexity("随便说点什么", file_count=1) == ModelTier.STANDARD


def test_model_for_resolves_tier_model(tmp_path):
    router, _ = _router(tmp_path)
    assert router.model_for("搜索", file_count=1) == "fast-model"
    assert router.model_for("重构 x y z", file_count=4) == "big-model"
    assert router.model_for("xyz", file_count=1) == "std-model"


def test_hot_reload_picks_up_config_change(tmp_path):
    """Editing the YAML must take effect WITHOUT recreating the router."""
    router, path = _router(tmp_path)
    assert router.classify_complexity("搜索", file_count=1) == ModelTier.FAST

    # rewrite the config: '搜索' now routes to powerful; force a distinct mtime
    updated = _CONFIG.replace(
        "  - tier: fast\n    keywords: [\"搜索\", \"lint\"]",
        "  - tier: powerful\n    keywords: [\"搜索\", \"lint\"]",
    )
    path.write_text(updated, encoding="utf-8")
    os.utime(path, (1_700_000_000, 1_700_000_001))

    assert router.classify_complexity("搜索", file_count=1) == ModelTier.POWERFUL


def test_fallback_when_config_file_absent(tmp_path):
    """No file on disk -> the built-in default YAML is used."""
    router = ModelRouter(config_path=tmp_path / "missing.yaml")
    assert router.classify_complexity("重构", file_count=5) == ModelTier.POWERFUL
    assert router.classify_complexity("搜索", file_count=1) == ModelTier.FAST
    assert router.config.get("tiers", {}).get("standard")


def test_yaml_is_valid(tmp_path):
    _, path = _router(tmp_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert set(data["tiers"]) >= {"fast", "standard", "powerful"}
    assert data["routing_rules"][-1]["keywords"] == []
