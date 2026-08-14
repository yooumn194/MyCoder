"""Tests for the Phase 3 model router (YAML rules + hot reload)."""

import os

import yaml

from mycoder.model_router import ModelRouter, ModelTier

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


# ---------------------------------------------------------------------------
# build_model_factory — model-tier routing wired into production
# ---------------------------------------------------------------------------

class _FakeLLM:
    """Stand-in for mycoder.llm: stores api_key/base_url as attributes."""

    def __init__(
        self, model, api_key=None, base_url=None, *, tracer=None, caller="llm", **kwargs
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._tracer = tracer
        self.caller = caller
        self.extra = kwargs
        self.client = None


class _OpenAIStyleLLM(_FakeLLM):
    """The real mycoder.llm.LLM keeps api_key/base_url on the OpenAI client."""

    def __init__(self, model, api_key=None, base_url=None, **kw):
        super().__init__(model, api_key=api_key, base_url=base_url, **kw)
        self.client = type("C", (), {"api_key": api_key, "base_url": base_url})()


def test_factory_routes_tier_to_same_class_model():
    from mycoder.model_router import build_model_factory

    # 用最小 fake router 隔离真实 yaml
    class _Router:
        def resolve_tier_model(self, tier):
            return {"fast": "fast-model", "standard": "std-model"}.get(tier)

    base = _FakeLLM(model="std-model", api_key="k", base_url="http://x", temperature=0.2)
    factory = build_model_factory(base, router=_Router())

    fast = factory("fast")
    assert fast is not None
    assert isinstance(fast, _FakeLLM)          # same class as the base
    assert fast.model == "fast-model"
    assert fast.caller == "model_router"
    assert fast.extra == {"temperature": 0.2}  # inherits base kwargs


def test_factory_falls_back_when_same_model_missing_or_no_tier():
    from mycoder.model_router import build_model_factory

    class _Router:
        def resolve_tier_model(self, tier):
            return {"fast": "std-model"}.get(tier)  # fast == base model

    base = _FakeLLM(model="std-model", api_key="k")
    factory = build_model_factory(base, router=_Router())
    assert factory("fast") is None      # tier model == base model
    assert factory("missing") is None   # no model configured
    assert factory(None) is None        # no tier


def test_factory_reads_credentials_off_openai_client():
    """LLM backend keeps api_key/base_url on the client; the factory must read
    them there so tier models can actually be constructed."""
    from mycoder.model_router import build_model_factory

    class _Router:
        def resolve_tier_model(self, tier):
            return "fast-model"

    base = _OpenAIStyleLLM(model="std-model", api_key="sk-x", base_url="https://api.example.com/v1")
    factory = build_model_factory(base, router=_Router())
    fast = factory("fast")
    assert fast.api_key == "sk-x"
    assert fast.base_url == "https://api.example.com/v1"


def test_factory_none_base_is_noop():
    from mycoder.model_router import build_model_factory

    factory = build_model_factory(None)
    assert factory("fast") is None
