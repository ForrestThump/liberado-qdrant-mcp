# liberado-qdrant-mcp

Lightweight **headless RAG** for LLM agents: Qdrant vectors, pluggable embedders,
MCP tools (`lqm-mcp`) and HTTP API (`lqm-api`).

## Quick start

```bash
# Qdrant (gRPC 6334)
export QDRANT_URL=http://127.0.0.1:6334

# MCP over stdio (Claude Desktop / Cursor)
cargo run -p lqm-mcp

# MCP HTTP
cargo run -p lqm-mcp -- serve --port 3000

# REST API
cargo run -p lqm-api -- --port 8080
```

Optional API auth: `LQM_API_TOKEN=secret` then `Authorization: Bearer secret` on `/api/*`.

## Agent documentation

See **[docs/AGENTS.md](docs/AGENTS.md)** for:

- Tool matrix (MCP ↔ HTTP)
- When to use `search` vs `get_relevant_context`
- Claude Desktop / Cursor / stdio vs `serve` setup
- Stable Qdrant payload schema

Roadmap and design: [docs/ROADMAP.md](docs/ROADMAP.md), [docs/PLAN.md](docs/PLAN.md).
