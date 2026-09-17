"""Provider-facing tool protocol adapters.

The agent runtime keeps one internal tool vocabulary.  Model-specific names
and argument spellings are translated only at the provider boundary, so the
orchestrator, traces, permissions and tool implementations never depend on a
vendor wire format.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass


_ZCODE_NAMES = {
    "read_file": "Read",
    "write_file": "Write",
    "edit_file": "Edit",
    "execute_in_sandbox": "Bash",
    "list_files": "Glob",
    "grep_search": "Grep",
}

_ZCODE_ARGUMENTS = {
    "list_files": {"glob_pattern": "pattern"},
}


@dataclass(frozen=True)
class ProviderToolCapabilities:
    """Wire-level capabilities negotiated from the provider profile.

    A model identifier is not a protocol identifier.  Keep this metadata next
    to the adapter so callers can make a fail-closed decision before sending a
    requirement-critical request.
    """

    provider: str
    dialect: str
    supports_tool_choice: bool = True
    supports_named_tool_choice: bool = True
    supports_stream_options: bool = True


_PROVIDER_CAPABILITIES: dict[str, ProviderToolCapabilities] = {
    "openai": ProviderToolCapabilities("openai", "native"),
    # DeepSeek Flash on OpenRouter currently follows the provider's textual
    # ZCode vocabulary.  This is a provider+model capability, not a global
    # model-name switch; other OpenRouter models remain native below.
    "openrouter": ProviderToolCapabilities("openrouter", "native"),
    # DeepSeek Flash thinking accepts the ordinary ``tools`` field, but its
    # thinking endpoint rejects both ``tool_choice=required`` and a named
    # function choice.  Requirement enforcement therefore stays in the agent
    # loop (which narrows the advertised catalog and validates every call),
    # while the provider request must leave ``tool_choice`` unset.  Strict
    # benchmark mutation calls have a bounded LLM-layer retry that disables
    # thinking for that one request, where named choices are accepted.
    "deepseek": ProviderToolCapabilities(
        "deepseek",
        "zcode",
        supports_tool_choice=False,
        supports_named_tool_choice=False,
    ),
    "ollama": ProviderToolCapabilities("ollama", "native", supports_tool_choice=False),
    "litellm": ProviderToolCapabilities("litellm", "native"),
}
_PROVIDER_MODEL_DIALECTS = {
    # The hosted DeepSeek Flash endpoints emit the ZCode/DSML tool vocabulary;
    # this exception is deliberately scoped by both provider and model.
    ("openai", "deepseek-flash"): "zcode",  # legacy OpenAI-compatible setup
    ("openrouter", "deepseek-flash"): "zcode",
    ("deepseek", "deepseek-flash"): "zcode",
}


def provider_tool_capabilities(
    provider: str | None,
    model: str,
    requested: str | None = None,
) -> ProviderToolCapabilities:
    """Return capabilities for an explicit provider profile.

    ``requested`` remains an escape hatch for compatibility and local custom
    gateways.  Auto resolution is provider-first; only the legacy helper
    below uses the historical model-name heuristic.
    """
    provider_name = (provider or "openai").strip().lower()
    base = _PROVIDER_CAPABILITIES.get(provider_name)
    requested_value = (
        requested if requested is not None else os.getenv("MYCODER_TOOL_DIALECT", "")
    ).strip().lower()
    dialect = (
        resolve_tool_dialect(model, requested)
        if requested_value and requested_value != "auto"
        else (
            _PROVIDER_MODEL_DIALECTS.get(
                (provider_name, model.strip().lower()),
                base.dialect if base is not None else "native",
            )
        )
    )
    if dialect not in {"native", "zcode"}:
        raise ValueError("tool dialect must be one of: auto, native, zcode")
    if base is None:
        return ProviderToolCapabilities(provider_name, dialect)
    return ProviderToolCapabilities(
        provider=base.provider,
        dialect=dialect,
        supports_tool_choice=base.supports_tool_choice,
        supports_named_tool_choice=base.supports_named_tool_choice,
        supports_stream_options=base.supports_stream_options,
    )


def resolve_tool_dialect(model: str, requested: str | None = None) -> str:
    """Resolve an explicit dialect or the model profile's safe default."""
    value = (requested or os.getenv("MYCODER_TOOL_DIALECT", "auto")).strip().lower()
    if value not in {"auto", "native", "zcode"}:
        raise ValueError("tool dialect must be one of: auto, native, zcode")
    if value != "auto":
        return value
    # deepseek-flash is distributed for coding-agent use and follows the
    # ZCode-style built-in tool vocabulary more reliably than project-local
    # snake_case aliases.  Other models keep the native MyCoder protocol.
    return "zcode" if model.strip().lower() == "deepseek-flash" else "native"


@dataclass(frozen=True)
class ToolProtocolAdapter:
    """Bidirectional translation between the internal and wire protocols."""

    dialect: str = "native"
    provider: str = "openai"
    supports_tool_choice: bool = True
    supports_named_tool_choice: bool = True
    supports_stream_options: bool = True

    @classmethod
    def for_model(
        cls,
        model: str,
        requested: str | None = None,
        provider: str | None = None,
    ) -> "ToolProtocolAdapter":
        capabilities = provider_tool_capabilities(provider, model, requested)
        return cls(
            capabilities.dialect,
            capabilities.provider,
            capabilities.supports_tool_choice,
            capabilities.supports_named_tool_choice,
            capabilities.supports_stream_options,
        )

    @property
    def _names(self) -> dict[str, str]:
        return _ZCODE_NAMES if self.dialect == "zcode" else {}

    def to_wire_name(self, name: str) -> str:
        return self._names.get(name, name)

    def from_wire_name(self, name: str) -> str:
        reverse = {wire: internal for internal, wire in self._names.items()}
        return reverse.get(name, name)

    def to_wire_arguments(self, internal_name: str, arguments: dict) -> dict:
        mapping = _ZCODE_ARGUMENTS.get(internal_name, {}) if self.dialect == "zcode" else {}
        return {mapping.get(key, key): value for key, value in arguments.items()}

    def from_wire_arguments(self, wire_name: str, arguments: dict) -> dict:
        internal_name = self.from_wire_name(wire_name)
        mapping = _ZCODE_ARGUMENTS.get(internal_name, {}) if self.dialect == "zcode" else {}
        reverse = {wire: internal for internal, wire in mapping.items()}
        return {reverse.get(key, key): value for key, value in arguments.items()}

    def tools_to_wire(self, tools: list[dict] | None) -> list[dict] | None:
        if not tools or self.dialect == "native":
            return tools
        converted = copy.deepcopy(tools)
        for schema in converted:
            function = schema.get("function")
            if not isinstance(function, dict):
                continue
            internal_name = str(function.get("name") or "")
            function["name"] = self.to_wire_name(internal_name)
            rename = _ZCODE_ARGUMENTS.get(internal_name, {})
            parameters = function.get("parameters")
            if not rename or not isinstance(parameters, dict):
                continue
            properties = parameters.get("properties")
            if isinstance(properties, dict):
                parameters["properties"] = {
                    rename.get(key, key): value for key, value in properties.items()
                }
            required = parameters.get("required")
            if isinstance(required, list):
                parameters["required"] = [rename.get(str(key), str(key)) for key in required]
        return converted

    def messages_to_wire(self, messages: list[dict]) -> list[dict]:
        if self.dialect == "native":
            return messages
        converted = copy.deepcopy(messages)
        for message in converted:
            if message.get("role") == "system" and isinstance(message.get("content"), str):
                content = message["content"]
                for internal, wire in sorted(
                    self._names.items(), key=lambda item: len(item[0]), reverse=True
                ):
                    content = content.replace(internal, wire)
                message["content"] = content
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if not isinstance(function, dict):
                    continue
                internal_name = str(function.get("name") or "")
                function["name"] = self.to_wire_name(internal_name)
                raw = function.get("arguments")
                if not isinstance(raw, str):
                    continue
                try:
                    arguments = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                function["arguments"] = json.dumps(
                    self.to_wire_arguments(internal_name, arguments),
                    ensure_ascii=False,
                )
        return converted

    def tool_choice_to_wire(self, choice):
        if self.dialect == "native" or not isinstance(choice, dict):
            return choice
        converted = copy.deepcopy(choice)
        function = converted.get("function")
        if isinstance(function, dict) and function.get("name"):
            function["name"] = self.to_wire_name(str(function["name"]))
        return converted
