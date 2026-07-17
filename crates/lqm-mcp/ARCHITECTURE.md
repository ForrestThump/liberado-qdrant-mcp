# lqm-mcp

MCP server binary. Agents talk to this; it delegates all work to `lqm-core`.

## Stack

- **MCP SDK:** turbomcp v3 (`#[server]`, `#[tool]`)
- **Transport:** stdio (default) + streamable HTTP (via `serve` subcommand)
- **Runtime:** tokio
- **CLI:** clap v4 (derive)

## Dual-mode

```
lqm-mcp              → stdio transport (spawn-on-demand, zero daemon)
lqm-mcp serve        → HTTP server (persistent, single model in RAM)
  --host 0.0.0.0     → bind address (default)
  --port 8080        → port (default)
```

## Testing

Integration tests use turbomcp's `channel` transport + `McpTestClient`.
A connectivity check on Qdrant ensures graceful skip when Qdrant is unavailable.
