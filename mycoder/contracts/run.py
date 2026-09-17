"""Runtime completion contract shared by API and agent entry points."""

from __future__ import annotations

from dataclasses import dataclass, replace

# Who decides whether a run passed: the agent's own recorded evidence, or the
# harness running a command and reading its exit code. They are not additive —
# asking both is duplicate work for the same question, and the agent-side
# answer is the weaker one ("did the model run a check?" vs "do the tests
# pass?"), so `harness` REPLACES `agent` rather than adding to it.
VERIFICATION_OWNER_AGENT = "agent"
VERIFICATION_OWNER_HARNESS = "harness"


@dataclass(frozen=True)
class RunContract:
    """Postconditions that must hold before a run can be reported successful."""

    require_mutation: bool = False
    require_verification: bool = False
    # Recorded, NOT enforced: since the tool catalog became constant for a whole
    # turn the agent never sends a provider ``tool_choice``, so nothing is
    # "strictly" forced any more. Kept because callers log the contract and
    # because removing it would silently change the serialized context.
    strict_tool_choice: bool = False
    mutation_reserved_tokens: int | None = None
    verification_reserved_tokens: int | None = None
    # ``agent`` or ``harness`` — see VERIFICATION_OWNER_* above.
    verification_owned_by: str = VERIFICATION_OWNER_AGENT

    @classmethod
    def from_policy(cls, sandbox_policy: str) -> "RunContract":
        benchmark = str(sandbox_policy).strip().lower() == "benchmark"
        return cls(
            require_mutation=benchmark,
            require_verification=benchmark,
            strict_tool_choice=benchmark,
        )

    def with_harness_verification(self) -> "RunContract":
        """Hand the verdict to the harness, dropping the agent-side requirement.

        Used when a benchmark run has a verification command configured: the
        exit code of that command is the verdict, so (a) the agent is no longer
        forced to run a check of its own before it may finish — which is what
        burned rounds and produced "no successful verification command" runs —
        and (b) the LLM verifier subagent is no longer spawned, since it would
        judge the same workspace a second time and could fail a run whose
        harness command passes.
        """
        return replace(
            self,
            require_verification=False,
            verification_owned_by=VERIFICATION_OWNER_HARNESS,
        )

    @property
    def harness_owned_verification(self) -> bool:
        return self.verification_owned_by == VERIFICATION_OWNER_HARNESS

    def as_context(self) -> dict[str, object]:
        """Serialize the contract for subagent runners and checkpoints."""
        context: dict[str, object] = {
            "require_patch": self.require_mutation,
            "require_verification": self.require_verification,
            "strict_tool_choice": self.strict_tool_choice,
            "verification_owned_by": self.verification_owned_by,
        }
        if self.mutation_reserved_tokens is not None:
            context["mutation_reserved_tokens"] = max(0, int(self.mutation_reserved_tokens))
        if self.verification_reserved_tokens is not None:
            context["verification_reserved_tokens"] = max(0, int(self.verification_reserved_tokens))
        return context
