"""Interactive REPL - the user-facing terminal interface."""

import argparse
import atexit
import os
import signal
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings

from .agent import Agent
from .agent_factory import AgentFactory
from .tools import ALL_TOOLS
from .llm import LLM, LiteLLM
from .config import Config, PROVIDERS
from .session import save_session, load_session, list_sessions
from . import __version__

console = Console()


def _parse_args():
    p = argparse.ArgumentParser(
        prog="mycoder",
        description="Minimal AI coding agent. Works with any OpenAI-compatible LLM.",
    )
    p.add_argument("-m", "--model", help="Model name (default: selected provider profile)")
    p.add_argument(
        "--provider",
        choices=PROVIDERS,
        help="Provider profile; selects its key, default model and base URL",
    )
    p.add_argument("--base-url", help="API base URL (default: $OPENAI_BASE_URL)")
    p.add_argument("--api-key", help="API key (default: $OPENAI_API_KEY)")
    p.add_argument("-p", "--prompt", help="One-shot prompt (non-interactive mode)")
    p.add_argument("-r", "--resume", metavar="ID", help="Resume a saved session")
    p.add_argument(
        "--run-log",
        metavar="PATH",
        help=(
            "Record an append-only event log (JSONL file, or a directory to name per run) "
            "that `python -m mycoder.replay` can re-execute without a provider. "
            "Also settable via $MYCODER_RUN_LOG."
        ),
    )
    p.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    return p.parse_args()


# Set by main() when memory is enabled; run once on exit (decay + compact so
# auto memories don't grow unbounded — see #6).
_memory_maintenance = None


def _run_memory_maintenance() -> None:
    if _memory_maintenance is None:
        return
    try:
        _memory_maintenance()
    except Exception:  # never mask the real exit cause
        pass


def _cleanup_sandbox_on_exit() -> None:
    """Teardown process-global resources so nothing lingers after exit.

    * sandbox: stop the manager's container + volume (auto-close; see plan);
    * MCP: close every transport (stdio child processes, SSE/HTTP sessions).

    Runs from atexit / a SIGTERM handler, i.e. while the interpreter is
    shutting down — asyncio.run() can no longer spawn executor threads there,
    so the sandbox uses the synchronous stop_sync() path instead of async
    stop(), and MCP teardown is bounded + best-effort. Memory maintenance
    (decay + compact, optional merge) runs last.
    """
    from .sandbox.executor import get_active_manager

    manager = get_active_manager()
    if manager is not None:
        try:
            manager.stop_sync()
        except Exception:  # never mask the real exit cause
            pass

    try:
        from .mcp import shutdown_mcp_tools

        shutdown_mcp_tools()
    except Exception:  # MCP is optional; never mask the real exit cause
        pass

    _run_memory_maintenance()


def _register_exit_cleanup() -> None:
    """Wire sandbox teardown into process exit.

    atexit covers every normal path (quit, Ctrl+C at the prompt, one-shot,
    sys.exit from errors). SIGTERM (kill, container orchestration) is routed
    through sys.exit so the same atexit hook runs; SIGINT is already handled by
    the REPL, and its exit path also reaches atexit.
    """
    atexit.register(_cleanup_sandbox_on_exit)
    signal.signal(signal.SIGTERM, lambda _sig, _frm: sys.exit(0))


def _announce_restore_point() -> None:
    """Record and show where the working tree stands before the first command.

    P0-1 made the container's /workspace the project directory itself (mounted
    read-write), so a destructive command is not contained by a throwaway copy
    any more. The restore point is what makes an approved mistake recoverable;
    printing it also means the operator sees the recovery command before it is
    needed. Best-effort: never block the REPL on it.
    """
    try:
        from .sandbox import restore_hint
        from .tools.sandbox_tool import capture_session_restore_point

        hint = restore_hint(capture_session_restore_point())
    except Exception:  # noqa: BLE001 - recovery must not break startup
        return
    if hint:
        console.print(f"[dim]{hint}[/dim]")


def main():
    _register_exit_cleanup()
    args = _parse_args()
    config = Config.from_env(
        provider_override=args.provider,
        model_override=args.model,
    )

    # CLI args override env vars
    if args.base_url:
        config.base_url = args.base_url
    if args.api_key:
        config.api_key = args.api_key

    if not config.api_key:
        console.print("[red bold]No API key found.[/]")
        console.print(
            "Set one of: OPENAI_API_KEY, OPENROUTER_API_KEY, "
            "DEEPSEEK_API_KEY, or MYCODER_API_KEY\n"
            "\nExamples:\n"
            "  # OpenAI\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "\n"
            "  # OpenRouter\n"
            "  export MYCODER_PROVIDER=openrouter OPENROUTER_API_KEY=sk-or-...\n"
            "\n"
            "  # DeepSeek\n"
            "  export OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.deepseek.com\n"
            "\n"
            "  # Ollama (local)\n"
            "  export OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1 MYCODER_MODEL=qwen2.5-coder\n"
        )
        sys.exit(1)

    # P2 token budget (#10): the CLI attaches an LLMTracer so sub-agent token
    # usage can be tracked per session and enforced by TokenBudgetGuard.
    from .observability.trace import LLMTracer

    tracer = LLMTracer()
    llm_cls = LiteLLM if config.provider == "litellm" else LLM
    llm = llm_cls(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        provider=config.provider,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        tracer=tracer,
    )
    # Trace replay (opt-in): wrap the LLM so every exchange is appended to a run
    # log, and hand the same recorder to the agent for tool/control events.
    run_log_target = args.run_log or os.getenv("MYCODER_RUN_LOG")
    run_recorder = None
    if run_log_target:
        from .observability.run_log import RecordingLLM, RunLogRecorder

        run_recorder = RunLogRecorder(
            run_log_target,
            run_id=args.resume or "cli",
            run_context={"execution_mode": "single", "sandbox_policy": "interactive"},
        )
        llm = RecordingLLM(llm, run_recorder)
        console.print(f"[green]Run log: {run_recorder.path}[/green]")
    # Phase 3.5: MCP servers (config/mcp_servers.yaml, all opt-in by default).
    # A broken MCP server never blocks the REPL.
    mcp_tools = []
    try:
        from .mcp import load_mcp_tools

        mcp_tools = load_mcp_tools()
        tools = [*ALL_TOOLS, *mcp_tools]
    except Exception:  # noqa: BLE001 - MCP is optional
        tools = ALL_TOOLS

    # P2 token budget (#10): per-subagent budget protection (default 100k
    # tokens/session; MYCODER_SUBAGENT_BUDGET to tune). The tracer attached
    # to the shared LLM above is the source of truth the guard reads.
    from .observability.budget import TokenBudgetGuard

    budget_guard = TokenBudgetGuard(
        max_tokens_per_session=int(os.getenv("MYCODER_SUBAGENT_BUDGET", "100000")),
        tracer=tracer,
    )
    factory = AgentFactory.from_defaults(
        llm=llm,
        tools=tools,
        max_context_tokens=config.max_context_tokens,
        budget_guard=budget_guard,
        additional_tool_names={t.name for t in mcp_tools},
        run_recorder=run_recorder,
    )
    global _memory_maintenance

    def _maintain_factory_memory() -> None:
        factory.maintain_memory(close=True)

    _memory_maintenance = _maintain_factory_memory
    agent = factory.build()

    # P0-1: one filesystem means the agent's shell can reach this checkout, so
    # a restore point is captured before the first command can run.
    _announce_restore_point()

    # resume saved session
    if args.resume:
        loaded = load_session(args.resume)
        if loaded:
            agent.messages, loaded_model = loaded
            # restore the model from the saved session unless overridden by CLI
            if not args.model:
                agent.llm.model = loaded_model
                config.model = loaded_model
            console.print(f"[green]Resumed session: {args.resume} (model: {agent.llm.model})[/green]")
        else:
            console.print(f"[red]Session '{args.resume}' not found.[/red]")
            sys.exit(1)

    try:
        # one-shot mode
        if args.prompt:
            _run_once(agent, args.prompt)
            return

        # interactive REPL
        _repl(agent, config)
    finally:
        if run_recorder is not None:
            run_recorder.close()


def _run_once(agent: Agent, prompt: str):
    """Non-interactive: run one prompt and exit."""
    def on_token(tok):
        print(tok, end="", flush=True)

    def on_tool(name, kwargs):
        console.print(f"\n[dim]> {name}({_brief(kwargs)})[/dim]")

    try:
        agent.chat(prompt, on_token=on_token, on_tool=on_tool)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)
    print()


def _repl(agent: Agent, config: Config):
    """Interactive read-eval-print loop."""
    console.print(Panel(
        f"[bold]MyCoder[/bold] v{__version__}\n"
        f"Model: [cyan]{config.model}[/cyan]"
        + (f"  Base: [dim]{config.base_url}[/dim]" if config.base_url else "")
        + "\nType [bold]/help[/bold] for commands, [bold]Ctrl+C[/bold] to cancel, [bold]quit[/bold] to exit.",
        border_style="blue",
    ))

    hist_path = os.path.expanduser("~/.mycoder_history")
    history = FileHistory(hist_path)

    # Enter submits, Escape+Enter inserts a newline (for pasting code blocks etc.)
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event):
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")
    def _newline(event):
        event.current_buffer.insert_text("\n")

    while True:
        try:
            user_input = pt_prompt(
                "You > ",
                history=history,
                multiline=True,
                key_bindings=kb,
                prompt_continuation="...  ",
            ).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye!")
            break

        if not user_input:
            continue

        # built-in commands
        if user_input.lower() in ("quit", "exit", "/quit", "/exit"):
            break
        if user_input == "/help":
            _show_help()
            continue
        if user_input == "/reset":
            agent.reset()
            console.print("[yellow]Conversation reset.[/yellow]")
            continue
        if user_input == "/tokens":
            p = agent.llm.total_prompt_tokens
            c = agent.llm.total_completion_tokens
            line = f"Tokens: [cyan]{p}[/cyan] prompt + [cyan]{c}[/cyan] completion = [bold]{p+c}[/bold] total"
            reasoning = getattr(agent.llm, "total_reasoning_tokens", 0)
            if reasoning:
                line += f"  ([magenta]{reasoning}[/magenta] reasoning)"
            cost = agent.llm.estimated_cost
            if cost is not None:
                line += f"  (~${cost:.4f})"
            console.print(line)
            continue
        if user_input == "/strategy" or user_input.startswith("/strategy "):
            name = user_input[len("/strategy"):].strip()
            if not name:
                mode = (
                    "auto（按任务切换）"
                    if agent._strategy_mode == "auto"
                    else agent.reasoning_strategy
                )
                console.print(
                    f"[cyan]当前推理策略: {mode}[/cyan]"
                    "  (/strategy auto|react|plan_execute|reflection)"
                )
            else:
                console.print(agent.set_strategy(name))
            continue
        if user_input == "/model" or user_input.startswith("/model "):
            new_model = user_input[7:].strip() if user_input.startswith("/model ") else ""
            if new_model:
                agent.llm.model = new_model
                config.model = new_model
                console.print(f"Switched to [cyan]{new_model}[/cyan]")
            else:
                console.print(f"Current model: [cyan]{config.model}[/cyan]")
            continue
        if user_input == "/compact":
            from .context import estimate_tokens
            before = estimate_tokens(agent.messages)
            compressed = agent.context.maybe_compress(agent.messages, agent.llm)
            after = estimate_tokens(agent.messages)
            if compressed:
                console.print(f"[green]Compressed: {before} → {after} tokens ({len(agent.messages)} messages)[/green]")
            else:
                console.print(f"[dim]Nothing to compress ({before} tokens, {len(agent.messages)} messages)[/dim]")
            continue
        if user_input == "/save":
            sid = save_session(agent.messages, config.model)
            console.print(f"[green]Session saved: {sid}[/green]")
            console.print(f"Resume with: mycoder -r {sid}")
            continue
        if user_input == "/diff":
            from .tools.edit import _changed_files
            if not _changed_files:
                console.print("[dim]No files modified this session.[/dim]")
            else:
                console.print(f"[bold]Files modified this session ({len(_changed_files)}):[/bold]")
                for f in sorted(_changed_files):
                    console.print(f"  [cyan]{f}[/cyan]")
            continue
        if user_input == "/sessions":
            sessions = list_sessions()
            if not sessions:
                console.print("[dim]No saved sessions.[/dim]")
            else:
                for s in sessions:
                    console.print(f"  [cyan]{s['id']}[/cyan] ({s['model']}, {s['saved_at']}) {s['preview']}")
            continue

        # an unknown /command shouldn't be sent to the model as a prompt
        if user_input.startswith("/"):
            console.print(f"[yellow]Unknown command: {user_input.split()[0]} (try /help)[/yellow]")
            continue

        # call the agent
        streamed: list[str] = []

        def on_token(tok):
            streamed.append(tok)
            print(tok, end="", flush=True)

        def on_tool(name, kwargs):
            console.print(f"\n[dim]> {name}({_brief(kwargs)})[/dim]")

        try:
            response = agent.chat(user_input, on_token=on_token, on_tool=on_tool)
            if streamed:
                print()  # newline after streamed tokens
            else:
                # response wasn't streamed (came after tool calls)
                console.print(Markdown(response))
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


def _show_help():
    console.print(Panel(
        "[bold]Commands:[/bold]\n"
        "  /help          Show this help\n"
        "  /reset         Clear conversation history\n"
        "  /model         Show current model\n"
        "  /model <name>  Switch model mid-conversation\n"
        "  /strategy      Show current reasoning strategy\n"
        "  /strategy <auto|react|plan_execute|reflection>  Switch strategy\n"
        "  /tokens        Show token usage\n"
        "  /compact       Compress conversation context\n"
        "  /diff          Show files modified this session\n"
        "  /save          Save session to disk\n"
        "  /sessions      List saved sessions\n"
        "  quit           Exit MyCoder\n"
        "\n"
        "[bold]Input:[/bold]\n"
        "  Enter          Submit message\n"
        "  Esc+Enter      Insert newline (for pasting code)",
        title="MyCoder Help",
        border_style="dim",
    ))


def _brief(kwargs: dict, maxlen: int = 80) -> str:
    s = ", ".join(f"{k}={repr(v)[:40]}" for k, v in kwargs.items())
    return s[:maxlen] + ("..." if len(s) > maxlen else "")
