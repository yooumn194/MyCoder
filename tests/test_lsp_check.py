"""LSP server precondition check (Phase 4 gate).

Skips itself when any of the language servers is not installed, so CI and
minimal dev machines stay green; on machines with gopls/pylsp/typescript it
actually launches each server and completes the LSP initialize handshake.
"""

import shutil

import pytest

from lsp_check import SERVERS, check_server


def _all_available() -> bool:
    return all(shutil.which(argv[0]) for _, argv, _ in SERVERS)


@pytest.mark.skipif(not _all_available(), reason="LSP servers not installed")
@pytest.mark.parametrize("name,argv,lang", SERVERS, ids=[s[0] for s in SERVERS])
def test_lsp_server_starts(name, argv, lang):
    """Each configured LSP server must start and answer the initialize handshake."""
    line = check_server(name, argv, lang)
    assert "PASS" in line, line
