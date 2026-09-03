"""Configuration - env vars and defaults."""

import os
from dataclasses import dataclass
from pathlib import Path


_PROVIDER_DEFAULTS = {
    "openai": {"model": "gpt-5.5", "base_url": None, "key_env": "OPENAI_API_KEY"},
    "deepseek": {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "openrouter": {
        "model": "minimax/minimax-m3:free",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
    },
    "ollama": {
        "model": "qwen2.5-coder:7b",
        "base_url": "http://localhost:11434/v1",
        "key_env": None,
    },
    "litellm": {"model": "gpt-5.5", "base_url": None, "key_env": None},
}
PROVIDERS = tuple(_PROVIDER_DEFAULTS)


def _detect_provider() -> str:
    """Resolve provider explicitly first, then from an unambiguous API key."""
    explicit = os.getenv("MYCODER_PROFILE") or os.getenv("MYCODER_PROVIDER")
    if explicit:
        return explicit.strip().lower()
    base_url = (os.getenv("MYCODER_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").lower()
    if "openrouter.ai" in base_url:
        return "openrouter"
    if "deepseek.com" in base_url:
        return "deepseek"
    if "localhost:11434" in base_url or "127.0.0.1:11434" in base_url:
        return "ollama"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.getenv("DEEPSEEK_API_KEY"):
        return "deepseek"
    return "openai"


def _load_dotenv():
    """Load .env from cwd, walking up to home dir. No-op if python-dotenv missing."""
    try:
        from dotenv import load_dotenv

        # search cwd first, then parent dirs up to ~
        env_path = Path(".env")
        if not env_path.exists():
            cur = Path.cwd()
            home = Path.home()
            while cur != home and cur != cur.parent:
                candidate = cur / ".env"
                if candidate.exists():
                    env_path = candidate
                    break
                cur = cur.parent
        load_dotenv(env_path, override=False)
    except ImportError:
        pass  # python-dotenv not installed, silently skip


@dataclass
class Config:
    model: str = "gpt-5.5"
    api_key: str = ""
    base_url: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.0
    max_context_tokens: int = 128_000
    provider: str = "openai"

    @classmethod
    def from_env(
        cls,
        *,
        provider_override: str | None = None,
        model_override: str | None = None,
    ) -> "Config":
        # load .env if present (won't override existing env vars)
        _load_dotenv()
        profile = (provider_override or os.getenv("MYCODER_PROFILE") or "").strip().lower()
        if profile and profile not in _PROVIDER_DEFAULTS:
            raise ValueError(
                f"unknown provider profile {profile!r}; choose one of: {', '.join(PROVIDERS)}"
            )
        provider = profile or _detect_provider()
        defaults = _PROVIDER_DEFAULTS.get(provider, _PROVIDER_DEFAULTS["openai"])
        provider_key_env = defaults.get("key_env")
        # Profile mode is a complete provider switch. Provider-specific values
        # win over generic settings left in .env for another provider.
        api_key = os.getenv(str(provider_key_env)) if profile and provider_key_env else None
        if not api_key:
            api_key = os.getenv("MYCODER_API_KEY")
        if not api_key and provider_key_env:
            api_key = os.getenv(str(provider_key_env))
        if not api_key and provider == "deepseek":
            api_key = os.getenv("OPENAI_API_KEY")
        # Custom OpenAI-compatible providers traditionally use OPENAI_API_KEY.
        if not api_key and provider not in {"openrouter", "deepseek"}:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key and provider == "ollama":
            api_key = "ollama"
        env_prefix = f"MYCODER_{provider.upper().replace('-', '_')}"
        configured_model = model_override or os.getenv(f"{env_prefix}_MODEL")
        if not configured_model and not profile:
            configured_model = os.getenv("MYCODER_MODEL")
        if not configured_model or configured_model.strip().lower() in {"auto", "default"}:
            configured_model = str(defaults["model"])
        if profile:
            configured_base_url = (
                os.getenv(f"{env_prefix}_BASE_URL") or defaults.get("base_url")
            )
        else:
            configured_base_url = (
                os.getenv("MYCODER_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
                or defaults.get("base_url")
            )
        return cls(
            model=configured_model,
            api_key=api_key or "",
            base_url=configured_base_url,
            max_tokens=int(os.getenv("MYCODER_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("MYCODER_TEMPERATURE", "0")),
            max_context_tokens=int(os.getenv("MYCODER_MAX_CONTEXT", "128000")),
            provider=provider,
        )
