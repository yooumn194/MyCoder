"""A mutex that can be awaited from any event loop."""

import asyncio
import threading


class AsyncMutex:
    """async-compatible mutual exclusion without loop affinity.

    The agent runs tools on a thread pool (corecoder/agent.py) and each tool
    call spins up its own event loop via asyncio.run(), so an `asyncio.Lock`
    would be bound to whichever loop created it and raise on the next call.
    A `threading.Lock` has no loop affinity; acquisition happens in a worker
    thread, making the await safe from *any* loop while still never blocking
    the loop itself.

    Used to serialize access to a single Docker container, whose `docker exec`
    is not safe to issue concurrently from multiple loops.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    async def __aenter__(self) -> "AsyncMutex":
        await asyncio.to_thread(self._lock.acquire)  # blocking -> off-loop
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._lock.release()
