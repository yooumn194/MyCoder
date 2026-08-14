"""ToolOutputValidator — a lightweight contract for what subagent tool calls
must return, so a malformed output is caught BEFORE the subagent reasons on it.

The subagent's final envelope is already Pydantic-validated; this closes the
inner gap: a tool returning e.g. a grep result without `matches` would let the
subagent continue on wrong data and produce a legal-but-wrong envelope.
"""

from typing import Any, Awaitable, Callable, Optional

MAX_RETRIES = 2


class ToolOutputValidator:
    # tool name -> output schema
    TOOL_SCHEMAS: dict[str, dict] = {
        "grep_search": {
            "required_fields": ["matches", "total_count"],
            "max_matches": 100,
        },
        "read_file": {
            "required_fields": ["content", "line_count"],
            "max_lines": 300,
        },
        "list_files": {
            "required_fields": ["files"],
            "max_files": 1000,
        },
        "execute_in_sandbox": {
            "required_fields": ["exit_code", "stdout", "stderr"],
        },
    }

    def validate(self, tool_name: str, output: Any) -> tuple[bool, Optional[str]]:
        """Return (is_valid, error_message); tools without a schema pass."""
        schema = self.TOOL_SCHEMAS.get(tool_name)
        if schema is None:
            return True, None
        if not isinstance(output, dict):
            return False, f"output must be an object, got {type(output).__name__}"

        for field in schema.get("required_fields", []):
            if field not in output:
                return False, f"{tool_name} output missing required field: {field}"

        if "max_matches" in schema:
            matches = output.get("matches")
            if not isinstance(matches, list):
                return False, f"{tool_name} output 'matches' must be a list"
            if len(matches) > schema["max_matches"]:
                return False, f"{tool_name} matches exceeds {schema['max_matches']}"
        if "max_lines" in schema:
            if output.get("line_count", 0) > schema["max_lines"]:
                return False, f"{tool_name} exceeds {schema['max_lines']} lines"
        if "max_files" in schema:
            files = output.get("files")
            if isinstance(files, list) and len(files) > schema["max_files"]:
                return False, f"{tool_name} files exceeds {schema['max_files']}"
        return True, None

    async def validate_and_retry(
        self,
        tool_name: str,
        params: dict,
        call_fn: Callable[[str, dict], Awaitable[Any]],
        max_retries: int = MAX_RETRIES,
    ) -> tuple[Any, bool]:
        """Call a tool, validate its output, and retry on schema violations.

        Returns (last_output, success). On retry, the validation error is
        passed to call_fn via params['_validation_error'] so a real subagent
        harness can feed it back as correction context. Fails gracefully
        (returns the invalid output, success=False) after max_retries — it
        never raises.
        """
        last_error: Optional[str] = None
        output = None
        for attempt in range(max_retries + 1):
            call_params = dict(params)
            if attempt > 0:
                call_params["_validation_error"] = last_error
            output = await call_fn(tool_name, call_params)
            valid, error = self.validate(tool_name, output)
            if valid:
                return output, True
            last_error = error or "unknown validation error"
        return output, False
