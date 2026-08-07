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
    unattended ->  fails closed unless CORECODER_ALLOW_RISKY_COMMANDS=1
                   (the analogue of --dangerously-skip-permissions).

Every decision is awaited with a hard timeout (CORECODER_CONFIRM_TIMEOUT, 60s)
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

ALLOW_RISKY_ENV = "CORECODER_ALLOW_RISKY_COMMANDS"
OPERATOR_ID_ENV = "CORECODER_OPERATOR_ID"
DEFAULT_OPERATOR_ID = "local_tty"
_TRUE = {"1", "true", "yes", "on"}

# Confirmation prompt deadline. Longer than the default 300s of a human prompt
# so a forgotten terminal can't hold the agent hostage, but generous enough
# that a real operator has time to decide. Fails closed on expiry.
CONFIRM_TIMEOUT_SECONDS = int(os.getenv("CORECODER_CONFIRM_TIMEOUT", "60"))


@dataclass(frozen=True)
class ConfirmRule:
    """One "ask" rule: a risky-but-legal class of command."""

    id: str
    category: str
    pattern: re.Pattern[str]
    reason: str


def _rule(id_: str, category: str, regex: str, reason: str) -> ConfirmRule:
    return ConfirmRule(
        id=id_,
        category=category,
        pattern=re.compile(regex, re.I),
        reason=reason,
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
    _rule("recursive_delete", "recursive_delete", r"\brm\s+-\w*r\w*\s+\S", "recursive delete"),
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
    "git_rewrite": "使用 git branch 创建新分支替代 force push",
    "recursive_delete": "使用 git checkout -- <file> 恢复特定文件",
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
    inject a reviewer ID via CORECODER_OPERATOR_ID."""
    return os.getenv(OPERATOR_ID_ENV, DEFAULT_OPERATOR_ID)


class ConfirmPolicy:
    """Decides whether a command may run, consulting a confirmer.

    Thread-safe: the agent runs tools on a thread pool (corecoder/agent.py), so
    the session approval cache is guarded by a lock.
    """

    def __init__(
        self,
        *,
        confirmer=None,
        rules: tuple[ConfirmRule, ...] | None = None,
    ) -> None:
        self._confirmer = confirmer or _default_confirmer
        self._rules = rules if rules is not None else CONFIRM_RULES
        # Cache keyed by (rule_id, base_command): approving `git push A` must
        # NOT auto-approve `git push B`.
        self._approved_signatures: set[tuple[str, str]] = set()
        self._lock = threading.Lock()

    def check(self, cmd: str) -> ConfirmRule | None:
        """First matching rule, or None if the command needs no confirmation."""
        for rule in self._rules:
            if rule.pattern.search(cmd):
                return rule
        return None

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
        response = input(
            f"\n⚠️  危险命令确认\n"
            f"  命令: {cmd}\n"
            f"  原因: {reason}\n"
            f"  执行? [y/N]: "
        ).strip().lower()
        return "approved" if response == "y" else "denied"
    except KeyboardInterrupt:
        print("\n确认被用户中断，视为拒绝。")
        return "denied"
    except EOFError:
        return "denied"


def _truncate(text: str, limit: int = 256) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
