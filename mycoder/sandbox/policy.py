"""Permission-style confirmation for risky-but-legal commands.

A lightweight mirror of Claude Code's permission system. The hard pre-check
(tools/sandbox_tool.py) *blocks* catastrophic commands outright; this layer
sits between the pre-check and execution and *asks* the operator to confirm
commands that are risky but legitimate — the sandbox would contain them, but
the operator may still want to watch (or stop) them.

How it maps onto Claude Code's permission model:

    rule       ->  CONFIRM_RULES: ConfirmRule(id, category, pattern, reason)
                   consulted in order; a match means "ask".
    decision   ->  remembered for the session keyed by (rule_id, base_command)
                   (mirrors "always allow for this session"). Stripping
                   options from the command means `git push --force origin main`
                   shares the cache entry with `git push origin main` — an
                   intentional trade-off, see _extract_base_command.
    unattended ->  fails closed unless MYCODER_ALLOW_RISKY_COMMANDS=1
                   (the analogue of --dangerously-skip-permissions).

Every decision is awaited with a hard timeout (MYCODER_CONFIRM_TIMEOUT, 60s)
and a SIGINT (Ctrl+C) is treated as DENY — a confirmation you cannot finish is
never allowed to slip through. The confirmer is injectable, so a richer
permission UI (persistent allowlists, MCP-backed approval) can be plugged in
later without touching the sandbox backends.
"""

import asyncio
import os
import re
import shlex
import sys
import threading
from dataclasses import dataclass

from .logger import get_logger

logger = get_logger()

ALLOW_RISKY_ENV = "MYCODER_ALLOW_RISKY_COMMANDS"
OPERATOR_ID_ENV = "MYCODER_OPERATOR_ID"
DEFAULT_OPERATOR_ID = "local_tty"
_TRUE = {"1", "true", "yes", "on"}

# Confirmation prompt deadline. Longer than the default 300s of a human prompt
# so a forgotten terminal can't hold the agent hostage, but generous enough
# that a real operator has time to decide. Fails closed on expiry.
CONFIRM_TIMEOUT_SECONDS = int(os.getenv("MYCODER_CONFIRM_TIMEOUT", "60"))


@dataclass(frozen=True)
class ConfirmRule:
    """One "ask" rule: a risky-but-legal class of command."""

    id: str
    category: str
    pattern: re.Pattern[str]
    reason: str
    # A truncating shape only *destroys* something when the target already
    # exists: `pytest -q > build.log` on a fresh path merely creates a file,
    # while `> src/main.py` throws away a source file. Rules that only matter
    # when they overwrite set this flag, and the policy then consults
    # `target_probe` against the paths extracted by `overwrite_targets`. This
    # is what keeps the rule from firing on the (ubiquitous) scratch-file
    # redirect — a confirmation nobody can answer costs a whole round trip.
    overwrite_only: bool = False
    # Which *kind* of existing path the rule is about, when that changes how
    # dangerous the overwrite is:
    #   True  — only when a target is tracked by the repository (or cannot be
    #           resolved, which fails closed). Rewriting a tracked file
    #           destroys repository content, i.e. the work a patch is made of.
    #   False — only when every target is a path the repository does NOT track:
    #           a repro script, build output, a scratch file. Overwriting one
    #           costs nothing beyond the run.
    #   None  — the rule does not care.
    # One shape (`> path`) is split across both flags because the same command
    # means two very different things depending on what it lands on. Splitting
    # it is what lets the benchmark policy keep auto-approving scratch writes
    # (heredocs, `pytest > build.log`) while a command that would clobber a
    # tracked file still has to be answered.
    target_tracked: bool | None = None


def _rule(
    id_: str,
    category: str,
    regex: str,
    reason: str,
    *,
    overwrite_only: bool = False,
    target_tracked: bool | None = None,
) -> ConfirmRule:
    return ConfirmRule(
        id=id_,
        category=category,
        pattern=re.compile(regex, re.I),
        reason=reason,
        overwrite_only=overwrite_only,
        target_tracked=target_tracked,
    )


# Paths a command would clobber, for the `overwrite_only` rules. Deliberately
# literal-only: a token carrying `$`, a glob or `~` cannot be resolved to a
# real file here, and an unresolvable target is treated as existing (fail
# closed) rather than silently waved through.
_OVERWRITE_TARGET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # `> path` (but not `>>`, `2>&1`, `>&`, or /dev/*)
    re.compile(r"(?<![>])>(?![&>])\s*(?!/dev/)(?P<target>[^\s;|&]+)"),
    # open('path', 'w' | 'wb' | mode='w') — read modes ('r') are not destructive
    re.compile(
        r"\bopen\s*\((?P<target>['\"][^'\"]+['\"])[^)]*?"
        r"(?:\bmode\s*=\s*)?['\"]w[b+]?['\"]",
        re.I,
    ),
    # shutil.copyfile(src, dst) / copy(src, dst)
    re.compile(r"\bcopy(?:file)?\s*\([^)]*?,\s*(?P<target>['\"][^'\"]+['\"])", re.I),
    # `cp src dst`, `mv src dst`, `install -m 644 src dst`: the destination is
    # the last argument and every one of them replaces what is already there.
    re.compile(r"\b(?:cp|mv|install)\b[^|;&]*\s(?P<target>[^\s;|&]+)\s*(?=$|[|;&])", re.I),
    # `ln -sf src dst` — with -f the link is created over an existing path.
    re.compile(
        r"\bln\b[^|;&]*\s-\w*f\w*[^|;&]*\s(?P<target>[^\s;|&]+)\s*(?=$|[|;&])",
        re.I,
    ),
    # `tee dst` truncates it (`tee -a` appends, which loses nothing).
    re.compile(
        r"\btee\b(?![^|;&]*(?:\s-\w*a\w*|--append\b))[^|;&]*?\s"
        r"(?P<target>[^\s;|&]+)\s*(?=$|[|;&])",
        re.I,
    ),
    # `sed -i` / `perl -i` rewrite the file they are handed in place.
    re.compile(
        r"\b(?:sed|perl)\b[^|;&]*\s-\w*i\w*[^|;&]*\s(?P<target>[^\s;|&]+)\s*(?=$|[|;&])",
        re.I,
    ),
)

_UNRESOLVABLE_TARGET = re.compile(r"[$*?~`]")


def is_unresolvable_target(token: str) -> bool:
    """Could this path token be a glob, a variable or a home reference?

    A token like ``$OUT/x`` or ``build/*.log`` cannot be resolved to one real
    file here, so every caller that has to decide something about it (does it
    exist, is it tracked) is expected to fail closed. Exposed so the sandbox
    manager's probes answer the same question the policy does.
    """
    return bool(_UNRESOLVABLE_TARGET.search(token))


def overwrite_targets(cmd: str) -> list[str]:
    """Candidate paths a command would truncate, best effort and literal-only."""
    targets: list[str] = []
    for pattern in _OVERWRITE_TARGET_PATTERNS:
        for match in pattern.finditer(cmd):
            token = match.group("target").strip().strip("'\"")
            if token:
                targets.append(token)
    return targets


# Commands that replace an existing file's contents without a `rm` or a `git`
# in sight, and with no destructive-looking flag for the pre-check to match.
# Each shape has a matching entry in _OVERWRITE_TARGET_PATTERNS so the policy
# can tell *what* would be replaced before asking.
_OVERWRITE_SHAPES = (
    r"(?<![>])>(?![&>])\s*(?!/dev/)\S"
    r"|\bopen\s*\([^)]*(?:\bmode\s*=\s*)?['\"]w[b+]?['\"]"
    r"|\.write_text\s*\(|\.write_bytes\s*\(|\.truncate\s*\(|os\.truncate\s*\("
    r"|\bcopy(?:file)?\s*\([^)]*,"
    r"|\btruncate\s+(?:-s\s+)?0\b"
    r"|\bdd\b[^|;&]*\bof="
    r"|\bcp\s+/dev/null\b"
    # `cp`/`mv`/`install` over an existing destination, `tee` (not `tee -a`),
    # an in-place `sed -i`/`perl -i`, and `ln -sf` replace a file's contents
    # just as thoroughly as a redirect does.
    r"|\b(?:cp|mv|install)\b[^|;&]*\s\S+\s*(?=$|[|;&])"
    r"|\bln\b[^|;&]*\s-\w*f\w*\s"
    r"|\btee\b(?![^|;&]*(?:\s-\w*a\w*|--append\b))"
    r"|\b(?:sed|perl)\b[^|;&]*\s-\w*i\w*"
)


# --- git shapes that throw uncommitted work away ---------------------------
# A benchmark patch IS "uncommitted work", so every spelling that discards it
# has to reach this layer. Written as named pieces because the carve-outs
# matter more than the matches: `git restore --staged <path>` only unstages,
# leaving the worktree untouched, and that is the one harmless spelling.
#
#   git restore <path>              default mode is --worktree: edits are gone
#   git restore --worktree <path>   explicit
#   git restore --source=<ref> .    restores the tree from another commit
#   git restore --staged <path>     carve-out: index only, worktree untouched
#   git switch -f / --discard-changes
#                                   moves the branch and drops the edits
#   git checkout -f/--force <ref>   the same, older spelling
#   git checkout .                  the whole tree
#   git checkout <path>             a path-like argument is a file restore, not
#                                   a branch switch (same as `checkout -- <path>`)
_SHORT_FLAG_F = r"(?<=\s)-\w*f(?![\w-])"
# A path-like argument: `src/main.py`, but not a branch (`feature/x`) or a
# version tag (`v1.2.3`). The pre-check on branch creation (`-b`/`-B`/`-t`)
# keeps `git checkout -b feature` out of this rule.
_PATHLIKE_ARG = r"(?:\S*/)?[^\s/]*\.[A-Za-z][A-Za-z0-9]{0,5}(?=\s|$)"
_GIT_DISCARD_PATTERNS: tuple[str, ...] = (
    # Any `git restore` except the index-only form. The old rule required a
    # `--`, a `-f` or a bare `.`, so the modern `git restore src/` — which
    # discards exactly like `git checkout -- src/` — ran with no prompt at all.
    r"\bgit\s+restore\b(?![^|;&]*\s(?:--staged|-S)(?![^|;&]*(?:--worktree|-W|--source)))",
    r"\bgit\s+checkout\b[^|;&]*(?:--\s|" + _SHORT_FLAG_F + r"|--force\b|\s\.(?:\s|$))",
    r"\bgit\s+checkout\b(?![^|;&]*\s(?:-b|-B|--branch|-t|--track|--orphan)\b)"
    r"[^|;&]*\s" + _PATHLIKE_ARG,
    # Directory paths have no extension, but `git checkout src/` is still a
    # worktree restore.  A trailing slash (or an explicit ./../ prefix) is
    # unambiguously a path rather than a branch name.
    r"\bgit\s+checkout\b(?![^|;&]*\s(?:-b|-B|--branch|-t|--track|--orphan)\b)"
    r"[^|;&]*\s(?:\./\S*|\.\./\S*|[^\s/]+/)(?=\s|$)",
    r"\bgit\s+switch\b[^|;&]*(?:" + _SHORT_FLAG_F + r"|--force\b|--discard-changes)",
)

# Ordered, first match wins. Each is deliberately coarse: a false positive just
# costs one confirmation. Categories double as the keys for ALTERNATIVE_HINTS.
CONFIRM_RULES: tuple[ConfirmRule, ...] = (
    _rule("network", "network", r"\b(curl|wget|scp|ssh|rsync|nc)\b", "network command"),
    _rule(
        "install",
        "install",
        r"\b(pip|pip3|npm|pnpm|yarn|brew|apt|apt-get|apk|dnf|gem|cargo|go)"
        r"\s+(install|add|uninstall|remove|upgrade|update)\b",
        "installs or changes packages",
    ),
    _rule(
        "git_rewrite",
        "git_rewrite",
        r"\bgit\s+(push|reset\s+--hard|checkout\s+--\s|clean\s+-\w*f"
        r"|filter-branch|rebase|rm\s+-r)\b",
        "git command that rewrites or publishes work",
    ),
    # Discarding the working tree is as destructive as rm: `git checkout .`,
    # `git checkout <ref> -- .`, `git restore <path>` and `git switch -f` all
    # throw away uncommitted edits with no prompt of their own. A benchmark
    # patch is exactly "uncommitted edits", so this was the one hole that could
    # silently turn a solved instance into an empty diff. See
    # _GIT_DISCARD_PATTERNS for the spellings each piece covers.
    _rule(
        "git_discard",
        "git_rewrite",
        "|".join(_GIT_DISCARD_PATTERNS),
        "git command that discards uncommitted changes",
    ),
    # Any `git clean` with clean flags removes untracked files — the one class
    # of work no git object in the repository holds. The previous pattern
    # (`clean\s+-\w*f`) only matched flags ENDING in f, so the idiomatic
    # `git clean -fdx` was let through while `git clean -xdf` was caught.
    # `-n` / `--dry-run` is a report, not a deletion.
    _rule(
        "git_clean",
        "recursive_delete",
        r"\bgit\s+clean\b(?![^|;&]*(?:-\w*n|--dry-run))",
        "git clean deletes untracked files",
    ),
    # Commands that destroy the object store itself: the restore point, the
    # reflog and unreachable commits all live there, and `git update-ref` can
    # rewrite any ref pointer. These reach the host repository through the
    # read-write bind mount like everything else.
    _rule(
        "git_object_store",
        "git_rewrite",
        r"\bgit\s+(?:update-ref|prune\b|reflog\s+expire|gc\b[^|;&]*--prune"
        r"|stash\s+(?:drop|clear)\b|branch\s+-D\b|tag\s+-d\b)",
        "git command that can destroy reachable history or the restore point",
    ),
    _rule("recursive_delete", "recursive_delete", r"\brm\s+-\w*r\w*\s+\S", "recursive delete"),
    # `find -delete` removes files with no `rm` in sight, and it is the idiomatic
    # way to sweep a tree. It reaches the read-write bind mount like any other
    # command, so the pre-check's `rm` patterns never see it.
    _rule(
        "find_delete",
        "recursive_delete",
        r"\bfind\b[^|;]*\s-delete\b",
        "deletes every file find matches",
    ),
    # The same deletion, routed around a plain `rm <path>`: a pipe
    # (`find . | xargs rm`), an -exec action, or a shredder. The pre-check
    # looks for `rm` with flags on real paths, so none of these show up there.
    _rule(
        "indirect_delete",
        "recursive_delete",
        r"\bfind\b[^|;]*-exec\s+rm\b|\bxargs\b[^|;]*\brm\b|\bshred\b",
        "deletes files without a plain `rm <path>`",
    ),
    # Same story through an interpreter: the shell never sees a delete command.
    _rule(
        "inline_delete",
        "recursive_delete",
        r"\b(?:shutil\.rmtree|os\.removedirs|os\.remove|os\.unlink|"
        r"Path\([^)]*\)\.unlink|\.rmdir)\s*\(",
        "deletes files from inside an interpreter",
    ),
    # Truncation with no `rm` and no `git` in sight. This is the largest
    # remaining hole for a read-write bind mount: `> src/main.py`, a `'w'`-mode
    # `open()` inside `python -c`, or `truncate -s 0` all erase a file's
    # contents while looking like an ordinary write. `overwrite_only=True`
    # keeps it from firing on scratch output (`pytest -q > build.log`).
    #
    # Split in two by target_tracked, because `> path` means two different
    # things depending on what it lands on. This pair is listed
    # tracked-first so a command that hits both kinds takes the stricter rule.
    _rule(
        "workspace_overwrite_tracked",
        "workspace_overwrite_tracked",
        _OVERWRITE_SHAPES,
        "overwrites a file tracked by the repository",
        overwrite_only=True,
        target_tracked=True,
    ),
    _rule(
        "workspace_overwrite",
        "workspace_overwrite",
        _OVERWRITE_SHAPES,
        "overwrites the contents of an existing file",
        overwrite_only=True,
        target_tracked=False,
    ),
    _rule("permission", "permission", r"\b(chmod|chown)\b", "changes permissions or ownership"),
    _rule("process", "process", r"\b(kill|pkill|killall)\b", "kills processes"),
    _rule("disk", "disk", r"^\s*(dd|mkfs)\b", "low-level disk command"),
    _rule("system_path", "system_path", r">\s*(/etc/|/usr/|/bin/|/sbin/)", "writes into a system path"),
    _rule("eval", "eval", r"\beval\s+", "evaluates a shell string"),
)

# Denial guidance keyed by ConfirmRule.category: what a human can do instead.
ALTERNATIVE_HINTS: dict[str, str] = {
    "network": "使用沙箱内已有依赖，或将资源预先下载到项目中",
    "install": "在 Dockerfile 中预装，或使用已有虚拟环境",
    "git_rewrite": "丢弃改动用 git restore --staged <file>（只撤索引，保留工作区）；发布改动请新建 git branch 替代 force push",
    "recursive_delete": "使用 git checkout -- <file> 恢复特定文件",
    "workspace_overwrite": "把输出写到新的临时文件，避免覆盖已有文件",
    "workspace_overwrite_tracked": "用 write_file / edit_file 修改仓库内的文件；沙箱命令只写临时文件",
    "permission": "在 Dockerfile 或 docker-compose 中设置文件权限",
    "process": "使用 timeout 命令限制子进程生命周期",
    "disk": "写入项目目录内，由部署脚本处理系统路径",
    "system_path": "写入项目目录内，由部署脚本处理系统路径",
    "eval": "将动态代码写入 .py 文件后执行",
}


def _extract_base_command(cmd: str) -> str:
    """Strip option flags, keep the command skeleton.

    'git push --force origin main' -> 'git push origin main'
    'rm -rf /tmp/x'               -> 'rm /tmp/x'

    INTERVIEW_NOTE: --force is deliberately NOT kept, so its approval shares
    the cache entry of the plain variant. --force is an aggressive form of the
    same operation, not a different operation. If finer granularity is ever
    wanted, move the "must re-confirm" flags into a never-strip allowlist.
    """
    try:
        tokens = shlex.split(cmd)
    except ValueError:  # unterminated quote — can't tokenize, cache as-is
        return cmd
    base = [t for t in tokens if not t.startswith("-")]
    return " ".join(base) if base else cmd


def _operator_id() -> str:
    """Who approved/denied. Defaults to local_tty; an MCP approval service can
    inject a reviewer ID via MYCODER_OPERATOR_ID."""
    return os.getenv(OPERATOR_ID_ENV, DEFAULT_OPERATOR_ID)


class ConfirmPolicy:
    """Decides whether a command may run, consulting a confirmer.

    Thread-safe: the agent runs tools on a thread pool (mycoder/agent.py), so
    the session approval cache is guarded by a lock.
    """

    def __init__(
        self,
        *,
        confirmer=None,
        rules: tuple[ConfirmRule, ...] | None = None,
        auto_approve_categories: set[str] | frozenset[str] | None = None,
        target_probe=None,
        tracked_probe=None,
    ) -> None:
        self._confirmer = confirmer or _default_confirmer
        self._rules = rules if rules is not None else CONFIRM_RULES
        # "Does this path exist?" — supplied by the sandbox manager, which is
        # the only layer that knows the project root the command will run in.
        # None means "assume every target exists" (fail closed); the policy
        # itself never touches the filesystem, which keeps it unit-testable.
        self._target_probe = target_probe
        # "Is this path tracked by the repository?" — the second half of the
        # same question, and the one that separates "rewriting a repro script"
        # from "rewriting the file the patch is supposed to change". None means
        # "cannot tell", which the tracked-shaped rules treat as tracked.
        self._tracked_probe = tracked_probe
        # A caller may grant a deliberately narrow, per-session capability.
        # This is separate from MYCODER_ALLOW_RISKY_COMMANDS, which remains the
        # explicit global escape hatch for every confirmation category.
        self._auto_approve_categories = frozenset(auto_approve_categories or ())
        # Cache keyed by (rule_id, base_command): approving `git push A` must
        # NOT auto-approve `git push B`.
        self._approved_signatures: set[tuple[str, str]] = set()
        self._lock = threading.Lock()

    def check(self, cmd: str) -> ConfirmRule | None:
        """First matching rule, or None if the command needs no confirmation."""
        for rule in self._rules:
            if not rule.pattern.search(cmd):
                continue
            if rule.overwrite_only and not self._destroys_something(cmd):
                continue
            if rule.target_tracked is not None and (
                self._targets_are_tracked(cmd) is not rule.target_tracked
            ):
                continue
            return rule
        return None

    def _targets_are_tracked(self, cmd: str) -> bool:
        """Would this overwrite a path the repository tracks?

        True when any target is tracked, and also when the question cannot be
        answered (no literal target, no probe, an unresolvable token) — the
        tracked-shaped rule is the stricter of the pair, so "unknown" has to
        fall to it. False only when every identifiable target is a literal path
        the repository does not track, i.e. a scratch file.
        """
        targets = overwrite_targets(cmd)
        if not targets or self._tracked_probe is None:
            return True
        for target in targets:
            if is_unresolvable_target(target):
                return True
            if self._tracked_probe(target):
                return True
        return False

    def _destroys_something(self, cmd: str) -> bool:
        """Would this truncation actually erase something that exists?

        True (ask) when a target exists, when a target cannot be resolved
        (``$OUT/x``, a glob — unknowable here, so fail closed), or when the
        shape matched but no literal path could be identified at all (e.g.
        ``open(path_variable, 'w')``). False only for the unambiguous case:
        every identifiable target is a literal path that does not exist yet,
        i.e. the command is creating a file rather than overwriting one.
        """
        targets = overwrite_targets(cmd)
        if not targets:
            return True
        probe = self._target_probe
        for target in targets:
            if is_unresolvable_target(target):
                return True
            if probe is None or probe(target):
                return True
        return False

    async def decide(self, cmd: str) -> tuple[bool, ConfirmRule | None]:
        """Return (may_run, matched_rule). rule is non-None only when denied."""
        rule = self.check(cmd)
        if rule is None:
            return True, None

        base_cmd = _extract_base_command(cmd)
        sig = (rule.id, base_cmd)
        if sig in self._approved_signatures:
            logger.info(
                "sandbox.confirm",
                rule_id=rule.id,
                reason=rule.reason,
                decision="approved",
                source="session_cache",
                operator_id=_operator_id(),
                cmd=cmd[:128],
            )
            return True, rule

        if rule.category in self._auto_approve_categories:
            self.record(rule.id, cmd, "approved")
            logger.info(
                "sandbox.confirm",
                rule_id=rule.id,
                reason=rule.reason,
                decision="approved",
                source="scoped_policy",
                operator_id=_operator_id(),
                cmd=cmd[:128],
            )
            return True, rule

        if self._auto_allow():
            self.record(rule.id, cmd, "approved")
            logger.info(
                "sandbox.confirm",
                rule_id=rule.id,
                reason=rule.reason,
                decision="approved",
                source="env_auto",
                operator_id=_operator_id(),
                cmd=cmd[:128],
            )
            return True, rule

        decision = await self.confirm(cmd, rule)
        if decision == "approved":
            self.record(rule.id, cmd, "approved")
            logger.info(
                "sandbox.confirm",
                rule_id=rule.id,
                reason=rule.reason,
                decision="approved",
                source="operator",
                operator_id=_operator_id(),
                cmd=cmd[:128],
            )
            return True, rule

        logger.warning(
            "sandbox.confirm",
            rule_id=rule.id,
            reason=rule.reason,
            decision="denied",
            source="operator",
            operator_id=_operator_id(),
            cmd=cmd[:128],
        )
        return False, rule

    async def confirm(self, cmd: str, rule: ConfirmRule) -> str:
        """Ask the operator, bounded by a hard timeout; fail closed."""
        try:
            decision = await asyncio.wait_for(
                asyncio.to_thread(self._confirmer, cmd, rule.reason),
                timeout=CONFIRM_TIMEOUT_SECONDS,
            )
            return decision if decision == "approved" else "denied"
        except asyncio.TimeoutError:
            logger.warning(
                "sandbox.confirm_timeout",
                rule_id=rule.id,
                reason=rule.reason,
                cmd=cmd[:128],
                timeout=CONFIRM_TIMEOUT_SECONDS,
                operator_id=_operator_id(),
            )
            return "denied"  # fail-closed
        except Exception:
            return "denied"

    def record(self, rule_id: str, cmd: str, decision: str) -> None:
        """Remember an approved command for the rest of the session."""
        if decision != "approved":
            return
        base_cmd = _extract_base_command(cmd)
        with self._lock:
            self._approved_signatures.add((rule_id, base_cmd))

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _auto_allow() -> bool:
        return os.getenv(ALLOW_RISKY_ENV, "").strip().lower() in _TRUE


def _default_confirmer(cmd: str, reason: str) -> str:
    """Interactive y/N prompt returning "approved" / "denied".

    Fails closed in every ambiguous case:
      * no TTY (CI, daemons, redirected stdin)      -> denied
      * Ctrl+C (SIGINT) during the prompt           -> denied
      * EOF (stdin closed)                          -> denied
      * anything other than a literal "y"           -> denied
    """
    if not sys.stdin.isatty():
        return "denied"
    try:
        response = input(f"\n⚠️  危险命令确认\n  命令: {cmd}\n  原因: {reason}\n  执行? [y/N]: ").strip().lower()
        return "approved" if response == "y" else "denied"
    except KeyboardInterrupt:
        print("\n确认被用户中断，视为拒绝。")
        return "denied"
    except EOFError:
        return "denied"


def _truncate(text: str, limit: int = 256) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
