"""Pytest bootstrap: isolate the test suite from the developer's environment.

The repo may carry a local .env (gitignored) that sets MYCODER_MODEL and
friends. Config.from_env() re-loads .env on every call (override=False), so a
local .env would re-inject those values even after a test deletes the env var
first. This autouse fixture neutralizes dotenv loading and pins the config env
vars, making the suite deterministic on any machine.

(P1-1 from the Phase 2 peer review.)
"""

import pytest

import mycoder.config as config_mod

_CONFIG_ENV_VARS = (
    "MYCODER_MODEL",
    "MYCODER_MAX_TOKENS",
    "MYCODER_MAX_CONTEXT",
    "MYCODER_TEMPERATURE",
    "MYCODER_BASE_URL",
    "MYCODER_API_KEY",
    "MYCODER_PROVIDER",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
)


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch):
    """No developer .env leaks into the tests."""
    for key in _CONFIG_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    # from_env() re-reads .env; make that a no-op so a local .env can never
    # override the pinned state above.
    monkeypatch.setattr(config_mod, "_load_dotenv", lambda: None)


@pytest.fixture(autouse=True)
def _reset_memory_state():
    """Phase 5: reset process-wide memory singletons and integration hooks so a
    test that installs hooks / builds a singleton store never leaks into the
    next test (and never writes to the developer's real ~/.mycoder)."""
    yield
    from mycoder import planner
    from mycoder.memory import reset_store
    from mycoder.tools import correction

    reset_store()
    planner._memory_injector = None  # noqa: SLF001
    planner._pending_memory_section = ""  # noqa: SLF001
    planner._plan_complete_hook = None  # noqa: SLF001
    planner._active_plan = None  # noqa: SLF001
    planner._active_store = None  # noqa: SLF001
    correction.recovery_hook = None  # noqa: SLF001


# ---------------------------------------------------------------------------
# Phase 5 memory fixtures (tmp_path-backed; never touch ~/.mycoder)
# ---------------------------------------------------------------------------
class _ControlledEmbedder:
    """Deterministic char+position embedder used by tests.

    Each (character, position) votes into one of 512 buckets (L2-normalized), so:
      * near-identical texts (same chars at same positions) get cosine > 0.85
        and trigger dedup;
      * texts that merely share vocabulary get low cosine and stay distinct —
        so multi-doc ranking tests are not silently merged by dedup.
    """

    def embed(self, text):
        import hashlib

        import numpy as np

        vec = np.zeros(512, dtype=np.float32)
        for i, ch in enumerate(text):
            key = f"{ch}{i}"
            h = int.from_bytes(hashlib.md5(key.encode("utf-8")).digest()[:4], "little")
            vec[h % 512] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec


@pytest.fixture
def memory_store(tmp_path):
    from mycoder.memory import MemoryStore

    return MemoryStore(
        project_dir=tmp_path / "proj",
        global_dir=tmp_path / "glob",
        embedder=_ControlledEmbedder(),
    )


@pytest.fixture
def memory_retriever(memory_store):
    from mycoder.memory import HybridRetriever

    return HybridRetriever(memory_store)
