"""SSRF boundary tests for the host-network fetch tool."""

from __future__ import annotations

import socket
import urllib.error
import urllib.request

import pytest

from mycoder.tools.fetch import (
    FetchUrlTool,
    SafeHTTPConnection,
    SafeHTTPHandler,
    SafeHTTPSHandler,
    SafeRedirectHandler,
    UnsafeFetchTarget,
    validate_fetch_url,
)


class _Response:
    def __init__(self, url: str, body: bytes = b"public body") -> None:
        self._url = url
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def geturl(self) -> str:
        return self._url

    def read(self, limit: int) -> bytes:
        return self._body[:limit]


class _Opener:
    def __init__(self, response=None, error=None) -> None:
        self.response = response
        self.error = error
        self.timeout = None

    def open(self, _request, timeout):
        self.timeout = timeout
        if self.error is not None:
            raise self.error
        return self.response


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://169.254.169.254/latest/meta-data",
        "http://10.0.0.1/internal",
        "http://[::1]/admin",
        "http://0.0.0.0/",
        "http://224.0.0.1/",
    ],
)
def test_private_and_special_ip_literals_are_blocked(url):
    with pytest.raises(UnsafeFetchTarget):
        validate_fetch_url(url)


def test_hostname_resolving_to_private_address_is_blocked(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", 80))
        ],
    )

    with pytest.raises(UnsafeFetchTarget):
        validate_fetch_url("http://internal.example/resource")


def test_mixed_public_and_private_dns_answers_fail_closed(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80)),
        ],
    )

    with pytest.raises(UnsafeFetchTarget):
        validate_fetch_url("http://rebinding.example/resource")


def test_redirect_to_private_destination_is_blocked():
    handler = SafeRedirectHandler()
    request = urllib.request.Request("https://93.184.216.34/start")

    with pytest.raises(UnsafeFetchTarget):
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "http://127.0.0.1/admin",
        )


def test_connection_revalidates_dns_and_never_connects_after_rebinding(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))
        ],
    )
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *_args, **_kwargs: pytest.fail("unsafe address must not be connected"),
    )

    with pytest.raises(UnsafeFetchTarget):
        SafeHTTPConnection("public.example", 80).connect()


def test_credentials_in_url_are_rejected():
    with pytest.raises(UnsafeFetchTarget, match="credentials"):
        validate_fetch_url("https://user:secret@93.184.216.34/path")


def test_public_fetch_uses_proxy_free_opener_and_caps_timeout(monkeypatch):
    response = _Response("https://93.184.216.34/resource")
    opener = _Opener(response=response)
    handlers = []

    def build_opener(*items):
        handlers.extend(items)
        return opener

    monkeypatch.setattr(urllib.request, "build_opener", build_opener)

    result = FetchUrlTool().execute("https://93.184.216.34/resource", timeout=999)

    assert result == "public body"
    assert opener.timeout == 30
    assert any(isinstance(item, urllib.request.ProxyHandler) for item in handlers)
    assert any(isinstance(item, SafeRedirectHandler) for item in handlers)
    assert any(isinstance(item, SafeHTTPHandler) for item in handlers)
    assert any(isinstance(item, SafeHTTPSHandler) for item in handlers)


def test_fetch_error_does_not_echo_query_secret(monkeypatch):
    opener = _Opener(error=urllib.error.URLError("offline"))
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_items: opener)

    result = FetchUrlTool().execute(
        "https://93.184.216.34/resource?token=super-secret#fragment"
    )

    assert "super-secret" not in result
    assert "fragment" not in result


def test_explicit_allowlist_supports_trusted_local_development(monkeypatch):
    monkeypatch.setenv("MYCODER_FETCH_ALLOWED_HOSTS", "localhost")
    opener = _Opener(response=_Response("http://localhost:8080/health", b"ok"))
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_items: opener)

    assert FetchUrlTool().execute("http://localhost:8080/health") == "ok"
