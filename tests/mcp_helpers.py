"""Shared fake MCP stdio server for the Phase 3.5 transport tests.

A real MCP server speaks Content-Length-framed JSON-RPC over stdio. This fake
mirrors that protocol so the StdioTransport can be tested without external
dependencies. NOTE: it reads with `read1()` — BufferedReader.read(n) blocks on
a pipe until n bytes or EOF, so partial frames would never be processed.
"""

FAKE_MCP_SERVER = r"""
import json, sys, re

def send(msg):
    body = json.dumps(msg).encode()
    sys.stdout.buffer.write(
        b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
    )
    sys.stdout.buffer.flush()

def reply(msg, result=None, error=None):
    out = {"jsonrpc": "2.0", "id": msg["id"]}
    if error is not None:
        out["error"] = error
    else:
        out["result"] = result if result is not None else {}
    send(out)

sys.stderr.write("server starting\n")
sys.stderr.flush()

buf = b""
while True:
    chunk = sys.stdin.buffer.read1(4096)  # read1: returns partial pipe data
    if not chunk:
        break
    buf += chunk
    while True:
        header_end = buf.find(b"\r\n\r\n")
        if header_end == -1:
            break
        m = re.search(rb"Content-Length:\s*(\d+)", buf[:header_end])
        if not m:
            break
        n = int(m.group(1))
        start = header_end + 4
        if len(buf) < start + n:
            break
        msg = json.loads(buf[start:start + n])
        buf = buf[start + n:]
        method = msg.get("method")
        if method == "initialize":
            reply(msg, {"capabilities": {}, "protocolVersion": "2024-11-05"})
        elif method == "ping":
            reply(msg, {})
        elif method == "tools/list":
            reply(msg, {"tools": [{
                "name": "echo",
                "description": "echo back the text argument",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            }]})
        elif method == "tools/call":
            args = msg.get("params", {}).get("arguments", {})
            name = msg.get("params", {}).get("name", "")
            if name == "echo":
                reply(msg, {"content": [{"type": "text", "text": "echo:" + args.get("text", "")}]})
            elif name == "noise":
                sys.stderr.write("index ready\n")
                sys.stderr.flush()
                reply(msg, {"content": [{"type": "text", "text": "noised"}]})
            elif name == "err":
                reply(msg, error={"code": -32602, "message": "invalid params"})
            elif name == "crash":
                sys.stderr.write("FATAL: boom\n")
                sys.stderr.flush()
                sys.exit(1)
            else:
                reply(msg, error={"code": -32601, "message": "method not found"})
"""


def write_fake_server(tmp_path, name: str = "fake_mcp.py"):
    """Materialize the fake server script into a temp dir."""
    path = tmp_path / name
    path.write_text(FAKE_MCP_SERVER, encoding="utf-8")
    return str(path)
