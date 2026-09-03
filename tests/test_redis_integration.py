"""Real Redis smoke tests, enabled only by CI or an explicit local URL."""

from __future__ import annotations

import os
import time
import uuid

import pytest

from api.state_backend import RedisStateBackend
from mycoder.observability.store import RedisObservabilityStore

REDIS_URL = os.getenv("MYCODER_TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(not REDIS_URL, reason="MYCODER_TEST_REDIS_URL is unset")


@pytest.mark.asyncio
async def test_redis_state_is_visible_across_backend_instances():
    session_id = f"ci-{uuid.uuid4().hex}"
    first = RedisStateBackend(REDIS_URL)
    second = RedisStateBackend(REDIS_URL)
    try:
        await first.save_session(session_id, {"status": "running", "tenant_id": "ci"})
        assert await second.get_session(session_id) == {
            "status": "running",
            "tenant_id": "ci",
        }
    finally:
        await first._redis.delete(f"mycoder:session:{session_id}")  # noqa: SLF001
        await first.close()
        await second.close()


def test_redis_observability_is_shared_and_atomic():
    suffix = uuid.uuid4().hex
    session_id = f"ci-{suffix}"
    rate_key = f"ci-rate-{suffix}"
    first = RedisObservabilityStore(REDIS_URL, ttl_seconds=60, max_alerts=10)
    second = RedisObservabilityStore(REDIS_URL, ttl_seconds=60, max_alerts=10)
    trace = {
        "call_id": suffix,
        "session_id": session_id,
        "timestamp": time.time(),
        "model": "ci-model",
    }
    try:
        first.append_trace(trace)
        assert second.list_traces(session_id) == [trace]
        assert first.rate_limit_allow(rate_key, 1)
        assert not second.rate_limit_allow(rate_key, 1)

        alert = {
            "session_id": session_id,
            "rule": "ci-rule",
            "metric": "error_count",
            "value": 1,
        }
        assert first.claim_alert(session_id, "ci-rule", 60, alert)
        assert not second.claim_alert(session_id, "ci-rule", 60, alert)
    finally:
        redis = first._redis  # noqa: SLF001
        redis.delete(first._trace_key(session_id))  # noqa: SLF001
        redis.delete(f"mycoder:observability:rate:{first._digest(rate_key)}")  # noqa: SLF001
        redis.delete(
            "mycoder:observability:alert-cooldown:"
            f"{first._digest(f'{session_id}:ci-rule')}"  # noqa: SLF001
        )
        first.close()
        second.close()
