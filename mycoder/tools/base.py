"""Base class for all tools."""

from abc import ABC, abstractmethod


class Tool(ABC):
    """Minimal tool interface. Subclass this to add new capabilities."""

    name: str
    description: str
    parameters: dict  # JSON Schema for the function args

    # Idempotency contract (P0, tools/idempotency.py):
    #   True  -> executing the same (args) twice is safe to reuse, and transient
    #            failures may be auto-retried (read-only / naturally idempotent
    #            writes like write_file with identical content).
    #   False -> side-effecting or non-deterministic (e.g. running an arbitrary
    #            sandbox command). Never auto-retried; identical calls are not
    #            deduped, so a retry can't double-apply a side effect.
    idempotent: bool = True

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Run the tool and return a text result."""
        ...

    def schema(self) -> dict:
        """OpenAI function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
