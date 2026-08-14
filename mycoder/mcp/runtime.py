"""A persistent event loop for the MCP layer.

MCP transports run background I/O tasks (stdio reader/writer, SSE reader). The
agent calls tools synchronously, so a naive `asyncio.run()` per call would
close the loop and kill those tasks — and futures created in one loop cannot be
resolved from another. The whole MCP stack therefore shares ONE long-lived loop
in a daemon thread; every async entry point submits work to it via
run_coroutine_threadsafe and blocks for the result.
"""

import asyncio
import threading

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None


def get_loop() -> asyncio.AbstractEventLoop:
    """The shared MCP loop (lazily created, lives for the process)."""
    global _loop, _thread
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        _thread = threading.Thread(target=_loop.run_forever, daemon=True, name="mcp-loop")
        _thread.start()
    return _loop


def run_in_loop(coro, timeout=None):
    """Run a coroutine on the shared loop from any thread, blocking for the result.

    ``timeout`` bounds the wait (e.g. teardown during interpreter shutdown must
    not hang on a stuck transport); a timeout raises concurrent.futures.TimeoutError.
    """
    loop = get_loop()
    return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=timeout)


def shutdown() -> None:
    """Stop the shared loop (used by tests / process teardown)."""
    global _loop, _thread
    if _loop is not None:
        _loop.call_soon_threadsafe(_loop.stop)
        if _thread is not None:
            _thread.join(timeout=2)
        _loop.close()
        _loop = None
        _thread = None
