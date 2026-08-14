"""Tests for PathGuard — the shared path-safety gate.

Phase 2 security red line: no tool may ever touch a file outside the project
root, whether via `../` traversal, an absolute path, or a symlink escape.
"""

import os
import sys

import pytest

from mycoder.tools.path_guard import PathGuard, PathTraversalError


def test_parent_traversal_rejected(tmp_path):
    """`../etc/passwd` escapes the root and must be rejected."""
    guard = PathGuard(project_root=tmp_path)
    with pytest.raises(PathTraversalError) as exc:
        guard.resolve("../etc/passwd")
    assert "路径超出项目范围" in str(exc.value)


def test_absolute_path_outside_root_rejected(tmp_path):
    guard = PathGuard(project_root=tmp_path)
    with pytest.raises(PathTraversalError):
        guard.resolve("/etc/passwd")


def test_normal_path_resolves(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.txt").write_text("x")
    guard = PathGuard(project_root=tmp_path)
    assert guard.resolve("sub/a.txt") == (tmp_path / "sub" / "a.txt").resolve()
    assert guard.resolve(".") == tmp_path.resolve()


def test_absolute_path_inside_root_maps(tmp_path):
    """An absolute path that IS inside the root is allowed (mapped, not rejected)."""
    (tmp_path / "a.txt").write_text("x")
    guard = PathGuard(project_root=tmp_path)
    assert guard.resolve(str(tmp_path / "a.txt")) == (tmp_path / "a.txt").resolve()


def test_symlink_escape_rejected(tmp_path):
    """A symlink inside the project pointing to a sibling outside must be rejected."""
    outside_dir = tmp_path.parent / "outside_victims"
    outside_dir.mkdir()
    victim = outside_dir / "secret.txt"
    victim.write_text("secret")
    link = tmp_path / "link_to_secret"
    link.symlink_to(victim)
    guard = PathGuard(project_root=tmp_path)
    with pytest.raises(PathTraversalError):
        guard.resolve("link_to_secret")


def test_symlink_pointing_outside_root_rejected(tmp_path):
    """A symlink whose TARGET is outside the root (e.g. /etc/passwd) is rejected."""
    if sys.platform == "win32" and not os.path.exists("/etc/passwd"):
        pytest.skip("no /etc/passwd on this platform")
    outside = tmp_path.parent.parent / "outside.txt"
    outside.write_text("outside")
    link = tmp_path / "evil_link"
    link.symlink_to(outside)
    guard = PathGuard(project_root=tmp_path)
    with pytest.raises(PathTraversalError):
        guard.resolve("evil_link")


def test_symlink_to_internal_file_allowed(tmp_path):
    """A symlink whose target stays inside the root is fine."""
    (tmp_path / "real.txt").write_text("x")
    (tmp_path / "alias").symlink_to(tmp_path / "real.txt")
    guard = PathGuard(project_root=tmp_path)
    assert guard.resolve("alias") == (tmp_path / "real.txt").resolve()


def test_symlink_ancestor_not_false_positive(tmp_path):
    """Regression: a project root whose LITERAL path goes through a symlink
    (macOS /var -> /private/var) must not be rejected as a symlink escape.

    The candidate's absolute ancestors (e.g. /var) are system symlinks; only
    the user-controlled portion below the project root may be inspected.
    """
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    proj = link / "proj"
    (real / "proj").mkdir(parents=True)
    (real / "proj" / "a.txt").write_text("x")

    guard = PathGuard(project_root=str(proj))  # literal path via the symlink
    resolved = guard.resolve(str(proj / "a.txt"))
    assert resolved == (real / "proj" / "a.txt").resolve()
