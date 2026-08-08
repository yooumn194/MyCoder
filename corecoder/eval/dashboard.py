"""Incremental verification dashboard — raises an alarm when the last three
results fail, optionally attributing them to a failure pattern.
"""

from typing import Optional

from .failure_kb import FailurePattern

ALARM_THRESHOLD = 3


class IncrementalDashboard:
    def __init__(self) -> None:
        self.results: list[bool] = []
        self.consecutive_failures: int = 0
        self.last_failure_pattern: Optional[FailurePattern] = None

    def add_result(self, passed: bool, pattern: Optional[FailurePattern] = None) -> dict:
        self.results.append(passed)
        total = len(self.results)
        passed_count = sum(self.results)
        pass_rate = passed_count / total

        if not passed:
            self.consecutive_failures += 1
            self.last_failure_pattern = pattern
            if self.consecutive_failures >= ALARM_THRESHOLD:
                return {
                    "status": "alarm",
                    "message": (
                        f"连续 {ALARM_THRESHOLD} 次失败，模式: "
                        f"{(pattern.value if pattern else 'unknown').upper()}"
                    ),
                    "pass_rate": pass_rate,
                }
        else:
            self.consecutive_failures = 0

        return {
            "status": "ok",
            "pass_rate": pass_rate,
            "total": total,
            "passed": passed_count,
        }
