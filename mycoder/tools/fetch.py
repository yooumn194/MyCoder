"""Read-only HTTP fetch tool with SSRF-safe destination validation."""

from __future__ import annotations

import http.client
import ipaddress
import os
import socket
import urllib.error
import urllib.parse
import urllib.request

from .base import Tool


_MAX_BODY_BYTES = 1_000_000
_MAX_OUTPUT_CHARS = 8_000
_MAX_TIMEOUT_SECONDS = 30


class UnsafeFetchTarget(ValueError):
    """Raised when a URL could reach a non-public network destination."""


def _allowed_hosts() -> set[str]:
    """Explicit escape hatch for trusted local development destinations."""
    return {
        host.strip().lower().rstrip(".")
        for host in os.getenv("MYCODER_FETCH_ALLOWED_HOSTS", "").split(",")
        if host.strip()
    }


def _is_public_address(value: str) -> bool:
    address = ipaddress.ip_address(value.split("%", 1)[0])
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return address.is_global and not any(
        (
            address.is_private,
            address.is_loopback,
            address.is_link_local,
            address.is_multicast,
            address.is_reserved,
            address.is_unspecified,
        )
    )


def _resolve_addresses(hostname: str, port: int) -> list[str]:
    hostname = hostname.lower().rstrip(".")
    try:
        addresses = [str(ipaddress.ip_address(hostname))]
    except ValueError:
        try:
            answers = socket.getaddrinfo(
                hostname,
                port,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise UnsafeFetchTarget("hostname could not be resolved") from exc
        addresses = list(dict.fromkeys(answer[4][0] for answer in answers))

    if not addresses:
        raise UnsafeFetchTarget("hostname did not resolve to an address")
    if hostname not in _allowed_hosts() and any(
        not _is_public_address(address) for address in addresses
    ):
        raise UnsafeFetchTarget(
            "private, loopback, link-local, or reserved destinations are blocked"
        )
    return addresses


def validate_fetch_url(url: str) -> str:
    """Validate scheme, credentials and every DNS answer before connecting.

    Rejecting a hostname when *any* answer is non-public prevents a resolver
    from returning a convenient public address alongside an internal target.
    Redirects are validated separately by ``SafeRedirectHandler`` below.
    """
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise UnsafeFetchTarget("malformed URL") from exc
    if parsed.scheme.lower() not in {"http", "https"}:
        raise UnsafeFetchTarget("only http and https URLs are supported")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeFetchTarget("credentials in URLs are not allowed")
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if not hostname:
        raise UnsafeFetchTarget("URL hostname is required")
    _resolve_addresses(
        hostname,
        port or (443 if parsed.scheme.lower() == "https" else 80),
    )
    return url


def _safe_url(url: str) -> str:
    """Return a log-safe URL without credentials, query parameters or fragments."""
    try:
        parsed = urllib.parse.urlsplit(url)
        hostname = parsed.hostname or "<invalid>"
        if parsed.port:
            hostname = f"{hostname}:{parsed.port}"
        return urllib.parse.urlunsplit((parsed.scheme, hostname, parsed.path, "", ""))
    except ValueError:
        return "<invalid-url>"


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-apply destination validation to every redirect hop."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_fetch_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_pinned_socket(connection: http.client.HTTPConnection):
    """Resolve, validate, then connect to that exact address (DNS pinning)."""
    last_error: OSError | None = None
    for address in _resolve_addresses(connection.host, connection.port):
        try:
            return socket.create_connection(
                (address, connection.port),
                connection.timeout,
                connection.source_address,
            )
        except OSError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise UnsafeFetchTarget("hostname did not resolve to an address")


class SafeHTTPConnection(http.client.HTTPConnection):
    """HTTP connection pinned to an address validated by this process."""

    def connect(self):
        self.sock = _open_pinned_socket(self)
        if self._tunnel_host:
            self._tunnel()


class SafeHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS variant that preserves the original hostname for TLS SNI."""

    def connect(self):
        raw_socket = _open_pinned_socket(self)
        server_hostname = self.host
        if self._tunnel_host:
            self.sock = raw_socket
            self._tunnel()
            raw_socket = self.sock
            server_hostname = self._tunnel_host
        self.sock = self._context.wrap_socket(
            raw_socket,
            server_hostname=server_hostname,
        )


class SafeHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(SafeHTTPConnection, req)


class SafeHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(
            SafeHTTPSConnection,
            req,
            context=self._context,
        )


class FetchUrlTool(Tool):
    cacheable = False  # remote resources can change between requests
    name = "fetch_url"
    description = (
        "Fetch text from a public http(s) URL. Private, loopback, link-local "
        "and reserved network destinations are blocked."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The public http:// or https:// URL to fetch",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (1-30, default 15)",
            },
        },
        "required": ["url"],
    }

    def execute(self, url: str, timeout: int = 15) -> str:
        try:
            validate_fetch_url(url)
            timeout = max(1, min(int(timeout), _MAX_TIMEOUT_SECONDS))
            req = urllib.request.Request(url, headers={"User-Agent": "MyCoder"})
            # Ignore ambient proxy variables: a proxy would resolve the target
            # independently and invalidate the address checks performed here.
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({}),
                SafeRedirectHandler(),
                SafeHTTPHandler(),
                SafeHTTPSHandler(),
            )
            with opener.open(req, timeout=timeout) as resp:
                validate_fetch_url(resp.geturl())
                raw = resp.read(_MAX_BODY_BYTES)
                text = raw.decode("utf-8", errors="replace")
        except (UnsafeFetchTarget, ValueError) as exc:
            return f"Error: unsafe fetch target ({exc})"
        except (urllib.error.URLError, OSError) as exc:
            return f"Error fetching {_safe_url(url)}: {exc}"
        if len(text) > _MAX_OUTPUT_CHARS:
            text = text[:6000] + f"\n... (truncated, {len(text)} chars) ...\n" + text[-1000:]
        return text
