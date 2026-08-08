# LSP 集成故障排查

CoreCoder 不内置 LSP Server——它通过 MCP 集成外部的 `mcp-server-lsp`
（Phase 4 Module B）。本文覆盖配置、常见启动失败与性能调优。

## 1. 常见配置（Python / TypeScript / Go）

`config/mcp_servers.yaml` 中的 `lsp` 条目通过 `LSP_SERVERS` 环境变量声明各
语言使用的 Server：

```yaml
servers:
  lsp:
    transport: stdio
    command: npx
    args: ["-y", "mcp-server-lsp"]
    enabled: false
    warmup: true
    warmup_timeout: 30
    env:
      LSP_SERVERS: |
        {
          "python": { "command": "pylsp", "args": [] },
          "typescript": { "command": "typescript-language-server", "args": ["--stdio"] },
          "go": { "command": "gopls", "args": [] }
        }
```

先确认每个语言 Server 能独立启动并完成 LSP `initialize` 握手——项目根目录
有现成的检查脚本：

```bash
python3 lsp_check.py          # 三个都查
python3 lsp_check.py pylsp    # 只查 Python
```

## 2. 启动失败排查

| 症状 | 可能原因 | 处理 |
|------|---------|------|
| `npx: command not found` | 未装 Node/npx | `brew install node`（或 `apt install nodejs npm`） |
| `exec: "pylsp": not found` | `python-lsp-server` 未装 | `pip install python-lsp-server` |
| `exec: "gopls": not found` | Go 工具链未装 | `brew install gopls` 或 `go install golang.org/x/tools/gopls@latest` |
| `exec: "tsserver"` 相关报错 | 用了裸 `tsserver` | TypeScript 的 LSP 入口是 `typescript-language-server`（包装 tsserver），不是 tsserver 本身 |
| MCP Server 启动即退出 | 包名/版本不兼容 | 看 stderr（已关联当前请求 id）：`[MCP:lsp] ...` |

## 3. 版本兼容性

- `mcp-server-lsp` 需要 `npx`（Node ≥ 18）。
- `pylsp`（python-lsp-server）当前主线 1.x。
- `gopls` 与 `typescript-language-server` 跟随上游语义版本即可。
- CoreCoder 侧的 MCP 协议层（stdio / SSE / Streamable HTTP）与具体语言 Server
  无关——出问题先隔离：`python3 lsp_check.py <server>` 单独验证握手。

## 4. 性能调优

- **预热**：`warmup: true` 让 CoreCoder 启动时完成 `initialize` 握手，消除
  首次 LSP 调用 2-5 秒冷启动。预热超时会打 WARNING 并在首次调用时重试，
  不阻塞启动。
- **连接池**：每个 MCP Server 一个 stdio 子进程（长连接）。语言 Server 按需
  由 `mcp-server-lsp` 派生；减少 LSP_SERVERS 里声明但用不到的语言可省内存。
- **结果压缩**：`LSPResultCompressor` 会把大量引用/诊断去重、排序、截断
  （引用最多 20 条）——不要自行绕过，否则会把上下文撑爆。
