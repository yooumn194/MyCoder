"""Load config/memory.yaml (optional) and merge it over the defaults.

Follows the same pattern as corecoder/mcp/config.py: the YAML is best-effort —
a missing file, missing PyYAML, or malformed content all fall back to the
built-in defaults rather than raising.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_CONFIG: dict = {
    "memory": {
        "rrf_k": 60,
        "max_tokens": 2048,
        "decay_days": 30,
        "confidence_threshold": 0.1,
        "filter_sensitive": True,
        "embedder": {
            "backend": "fastembed",
            "model": "BAAI/bge-small-zh-v1.5",
        },
    }
}

_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "memory.yaml"
)


def _deep_merge(base: dict, override: dict) -> dict:
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_memory_config(path: str | Path | None = None) -> dict:
    """Return the memory config dict (always the full merged shape)."""
    config = _deep_merge(
        dict(DEFAULT_CONFIG),
        _load_yaml(Path(path) if path else _CONFIG_PATH),
    )
    if "CORECODER_MEMORY_EMBEDDER" in os.environ:
        config["memory"]["embedder"]["backend"] = os.environ[
            "CORECODER_MEMORY_EMBEDDER"
        ]
    return config


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:  # noqa: BLE001 - malformed yaml degrades to defaults
        pass
    return {}
