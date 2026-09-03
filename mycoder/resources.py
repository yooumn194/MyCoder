"""Resolve configuration files in a source checkout or an installed wheel."""

from __future__ import annotations

from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent
_SOURCE_CONFIG_ROOT = _PACKAGE_ROOT.parent / "config"
_PACKAGED_CONFIG_ROOT = _PACKAGE_ROOT / "config_data"


def config_path(filename: str) -> Path:
    """Return the repository config when present, otherwise packaged data.

    Editable/source installs keep hot-reload behavior against ``config/``.
    Wheels contain the same defaults under ``mycoder/config_data`` so runtime
    behavior does not depend on the caller's current working directory.
    """
    if Path(filename).name != filename:
        raise ValueError("configuration filename must not contain a path")
    source = _SOURCE_CONFIG_ROOT / filename
    if source.is_file():
        return source
    return _PACKAGED_CONFIG_ROOT / filename
