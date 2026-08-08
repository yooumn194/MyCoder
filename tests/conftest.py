"""Pytest bootstrap: isolate the test suite from the developer's environment.

The repo may carry a local .env (gitignored) that sets CORECODER_MODEL and
friends. Config.from_env() re-loads .env on every call (override=False), so a
local .env would re-inject those values even after a test deletes the env var
first. This autouse fixture neutralizes dotenv loading and pins the config env
vars, making the suite deterministic on any machine.

(P1-1 from the Phase 2 peer review.)
"""

import pytest

import corecoder.config as config_mod

_CONFIG_ENV_VARS = (
    "CORECODER_MODEL",
    "CORECODER_MAX_TOKENS",
    "CORECODER_MAX_CONTEXT",
    "CORECODER_TEMPERATURE",
    "CORECODER_BASE_URL",
    "CORECODER_API_KEY",
    "CORECODER_PROVIDER",
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
