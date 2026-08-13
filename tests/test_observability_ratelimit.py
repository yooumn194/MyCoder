"""P2 rate limiting: token bucket / leaky bucket / sliding window / facade."""

from corecoder.observability.ratelimit import (
    LeakyBucket,
    RateLimiter,
    SlidingWindowCounter,
    TokenBucket,
)


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def test_token_bucket_burst_and_refill():
    clock = _Clock()
    tb = TokenBucket(capacity=3, refill_per_second=1.0, now=clock)
    assert tb.consume() and tb.consume() and tb.consume()  # burst of 3
    assert not tb.consume()  # empty
    clock.advance(2.0)
    assert tb.consume() and tb.consume()  # 2 refilled
    assert not tb.consume()


def test_leaky_bucket_fixed_drain():
    clock = _Clock()
    lb = LeakyBucket(capacity=3, leak_per_second=1.0, now=clock)
    assert lb.consume() and lb.consume() and lb.consume()
    assert not lb.consume()  # queue full
    clock.advance(2.0)
    assert lb.consume() and lb.consume()  # 2 leaked out
    assert not lb.consume() and not lb.consume()  # 2 admitted, 1 more fails


def test_sliding_window_counter():
    clock = _Clock()
    sw = SlidingWindowCounter(window_seconds=10, max_requests=3, now=clock)
    assert sw.allow() and sw.allow() and sw.allow()
    assert not sw.allow()  # window full
    clock.advance(10.0)
    assert sw.allow()  # window rolled


def test_rate_limiter_is_per_key():
    clock = _Clock()
    rl = RateLimiter(requests_per_minute=2, now=clock)
    assert rl.allow("alice") and rl.allow("alice")
    assert not rl.allow("alice")  # alice exhausted
    assert rl.allow("bob")  # bob unaffected


def test_rate_limiter_from_env(monkeypatch):
    assert RateLimiter.from_env() is None
    monkeypatch.setenv("CORECODER_RATE_LIMIT", "5")
    limiter = RateLimiter.from_env()
    assert limiter is not None and limiter.requests_per_minute == 5
