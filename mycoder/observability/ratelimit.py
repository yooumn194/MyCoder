"""Rate limiting primitives (P2) — token bucket / leaky bucket / sliding window.

Classic 八股 and a genuine agent-service concern: a runaway agent loop firing
LLM calls must not blow the provider's rate limit. Three algorithms, all pure
with an injectable clock so they are unit-testable without sleeping:

  * TokenBucket         — burst capacity + steady refill; allow if >= n tokens.
  * LeakyBucket         — fixed drain rate; allow if there is room in the queue.
  * SlidingWindowCounter— max requests per fixed window (rolling).

`RateLimiter` is the per-key sliding-window facade used by the API (opt-in via
MYCODER_RATE_LIMIT=requests-per-minute on /v1/agent/run).
"""

from __future__ import annotations

import time
from collections.abc import Callable

Clock = Callable[[], float]


def _default_clock() -> float:
    return time.monotonic()


class TokenBucket:
    """Classic token bucket: burst up to `capacity`, refilled continuously."""

    def __init__(
        self, capacity: float, refill_per_second: float, now: Clock | None = None
    ) -> None:
        self.capacity = capacity
        self.refill_rate = refill_per_second
        self._clock = now or _default_clock
        self._tokens = capacity
        self._last = self._clock()

    def consume(self, n: int = 1) -> bool:
        now = self._clock()
        elapsed = now - self._last
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
        self._last = now
        if self._tokens >= n:
            self._tokens -= n
            return True
        return False

    @property
    def tokens(self) -> float:
        return self._tokens


class LeakyBucket:
    """Fixed drain rate; a request is admitted only if the queue has room."""

    def __init__(
        self, capacity: float, leak_per_second: float, now: Clock | None = None
    ) -> None:
        self.capacity = capacity
        self.leak_rate = leak_per_second
        self._clock = now or _default_clock
        self._water = 0.0
        self._last = self._clock()

    def consume(self, n: int = 1) -> bool:
        now = self._clock()
        self._water = max(0.0, self._water - (now - self._last) * self.leak_rate)
        self._last = now
        if self._water + n <= self.capacity:
            self._water += n
            return True
        return False

    @property
    def water(self) -> float:
        return self._water


class SlidingWindowCounter:
    """Rolling-window rate counter: `max_requests` per `window_seconds`."""

    def __init__(
        self, window_seconds: float, max_requests: int, now: Clock | None = None
    ) -> None:
        self.window = window_seconds
        self.max_requests = max_requests
        self._clock = now or _default_clock
        self._hits: list[float] = []

    def allow(self) -> bool:
        now = self._clock()
        self._hits = [t for t in self._hits if now - t < self.window]
        if len(self._hits) >= self.max_requests:
            return False
        self._hits.append(now)
        return True

    def __len__(self) -> int:
        return len(self._hits)


class RateLimiter:
    """Per-key sliding-window limiter (the service-layer facade)."""

    def __init__(self, requests_per_minute: int, now: Clock | None = None) -> None:
        self.requests_per_minute = requests_per_minute
        self._clock = now or _default_clock
        self._limiters: dict[str, SlidingWindowCounter] = {}

    def allow(self, key: str) -> bool:
        limiter = self._limiters.setdefault(
            key, SlidingWindowCounter(60.0, self.requests_per_minute, self._clock)
        )
        return limiter.allow()

    @classmethod
    def from_env(cls) -> "RateLimiter | None":
        """MYCODER_RATE_LIMIT=<requests/min> -> limiter, or None when unset."""
        import os

        raw = os.getenv("MYCODER_RATE_LIMIT", "").strip()
        if not raw:
            return None
        try:
            return cls(int(raw))
        except ValueError:
            return None
