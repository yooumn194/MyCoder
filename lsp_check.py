#!/usr/bin/env python3
"""Verify the LSP preconditions: gopls / pylsp / tsserver can start.

Launches each language server as a stdio subprocess (Content-Length-framed
JSON-RPC), performs the LSP `initialize` handshake, then shuts it down.
Reports PASS/FAIL per server and exits non-zero if any required server fails.

Usage:
    python3 lsp_check.py            # check all three
    python3 lsp_check.py pylsp      # check just one
"""

import json
import os
import select
import subprocess
import sys
import time

# (label, argv, language).
# TypeScript: the raw `tsserver` binary speaks its OWN protocol, not LSP. The
# LSP entry point is `typescript-language-server` (wraps tsserver, translating
# LSP <-> native); that is what editors launch, and what we verify here.
SERVERS = [
    ("gopls", ["gopls"], "Go"),
    ("pylsp", ["pylsp"], "Python"),
    ("tsserver", ["typescript-language-server", "--stdio"], "TypeScript"),
]

TIMEOUT = 30  # seconds to wait for the initialize response


def _send(proc, msg):
    body = json.dumps(msg).encode()
    proc.stdin.write(
        b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
    )
    proc.stdin.flush()


def _read_messages(proc, deadline):
    """Yield parsed JSON-RPC messages from stdout until the deadline.

    Uses read1() + a persistent buffer: BufferedReader.read(n) blocks for n
    bytes on a pipe, and read(1)+select() breaks because the buffered chunk
    hides from select. read1() returns whatever is available, exactly like the
    MCP stdio transport framing.
    """
    buffer = b""
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining <= 0:
            return
        r, _, _ = select.select([proc.stdout], [], [], min(0.5, remaining))
        if not r:
            continue  # a 0.5s select timeout is NOT the deadline — keep waiting
        chunk = proc.stdout.read1(65536)
        if not chunk:
            return  # EOF / server exited
        buffer += chunk
        while True:
            header_end = buffer.find(b"\r\n\r\n")
            if header_end == -1:
                break  # incomplete header
            length = 0
            for line in buffer[:header_end].decode(errors="replace").split("\r\n"):
                if line.lower().startswith("content-length:"):
                    length = int(line.split(":", 1)[1].strip())
            if len(buffer) < header_end + 4 + length:
                break  # half body — wait for the rest
            body = buffer[header_end + 4 : header_end + 4 + length]
            buffer = buffer[header_end + 4 + length :]
            try:
                yield json.loads(body.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue


def check_server(name, argv, lang):
    try:
        proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        return f"{name:9s}  FAIL  binary not found: {argv[0]}"
    try:
        _send(proc, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "processId": os.getpid(),
                "rootUri": "file:///tmp",
                "capabilities": {},
                "workspaceFolders": None,
            },
        })
        deadline = time.time() + TIMEOUT
        result = None
        for msg in _read_messages(proc, deadline):
            if msg.get("id") == 1 and "result" in msg:
                result = msg["result"]
                break
        if result is None:
            return f"{name:9s}  FAIL  no initialize response within {TIMEOUT}s"
        caps = result.get("capabilities", {})
        # clean shutdown
        _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "shutdown"})
        _send(proc, {"jsonrpc": "2.0", "method": "exit"})
        return f"{name:9s}  PASS  {lang:8s}  capabilities: {len(caps)} entries"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    targets = sys.argv[1:] or [s[0] for s in SERVERS]
    print("LSP server precondition check\n" + "-" * 48)
    failures = 0
    for name, argv, lang in SERVERS:
        if name not in targets:
            continue
        line = check_server(name, argv, lang)
        print(line)
        if "FAIL" in line:
            failures += 1
    print("-" * 48)
    if failures:
        print(f"{failures} LSP server(s) FAILED the precondition.")
        return 1
    print("All LSP preconditions OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
