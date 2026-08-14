"""SLO alerting on the LLM-observability metrics (P2, alerts).

An AlertManager evaluates a metrics dict (session summary from LLMTracer, or a
test harness) against AlertRules — success rate, p95 latency, token budget —
and emits a structlog.error per fired alert, debounced by a per-(rule, session)
cooldown so a stuck session can't spam.

Rules are threshold checks with an operator; wiring: attach the manager to an
LLMTracer and it evaluates after every recorded call.
"""

from __future__ import annotations

import time

from mycoder.sandbox.logger import get_logger

logger = get_logger("mycoder.alerts")


class AlertRule:
    """A threshold on one metric: value <op> threshold -> breach."""

    def __init__(
        self,
        name: str,
        metric: str,
        threshold: float,
        op: str = ">=",
        severity: str = "warning",
        cooldown_seconds: int = 60,
    ) -> None:
        self.name = name
        self.metric = metric
        self.threshold = float(threshold)
        self.op = op
        self.severity = severity
        self.cooldown_seconds = cooldown_seconds

    def breached(self, value: float) -> bool:
        if self.op == ">=":
            return value >= self.threshold
        if self.op == "<=":
            return value <= self.threshold
        if self.op == "<":
            return value < self.threshold
        if self.op == ">":
            return value > self.threshold
        return False


def default_rules() -> list[AlertRule]:
    """Sensible defaults for an LLM service session."""
    return [
        AlertRule("low_success_rate", "success_rate", 0.9, op="<", severity="critical"),
        AlertRule("high_p95_latency", "p95_duration_ms", 5000.0, op=">", severity="warning"),
        AlertRule("budget_high_usage", "budget_ratio", 0.8, op=">=", severity="warning"),
    ]


class AlertManager:
    """Evaluates metrics against rules and emits debounced alerts."""

    def __init__(self, rules: list[AlertRule] | None = None, log=logger) -> None:
        self.rules = rules if rules is not None else default_rules()
        self._log = log
        self._last_fired: dict[tuple[str, str], float] = {}

    def evaluate(self, session_id: str, metrics: dict) -> list[dict]:
        """Return alerts fired for `metrics` (respecting per-rule cooldown)."""
        now = time.time()
        fired: list[dict] = []
        for rule in self.rules:
            value = metrics.get(rule.metric)
            if value is None:
                continue
            if not rule.breached(float(value)):
                continue
            key = (session_id, rule.name)
            last = self._last_fired.get(key)
            if last is not None and now - last < rule.cooldown_seconds:
                continue
            self._last_fired[key] = now
            alert = {
                "session_id": session_id,
                "rule": rule.name,
                "metric": rule.metric,
                "value": value,
                "threshold": rule.threshold,
                "severity": rule.severity,
            }
            fired.append(alert)
            self._log.error(
                "alert_fired",
                session_id=session_id,
                rule=rule.name,
                metric=rule.metric,
                value=value,
                threshold=rule.threshold,
                severity=rule.severity,
            )
        return fired

    def reset(self) -> None:
        self._last_fired.clear()
