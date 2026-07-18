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

## Scoped filtering (clearance-safe isolation)

Share one collection across projects or sensitivity levels without multi-user auth:

| Payload field | Meaning | Search constraint |
|---------------|---------|-------------------|
| `scope` | Exact partition (e.g. `team-a`, `personal`) | `scope=…` — only that partition |
| `clearance` | `public` < `internal` < `confidential` < `restricted` | `max_clearance=…` — that level and below |

- **Ingest** (MCP `ingest_text` / HTTP `POST /api/ingest`): optional `scope` and `clearance`. If clearance is omitted, points are stored as **`public`**.
- **Search / context**: optional `scope` and `max_clearance`. When omitted, behavior is unchanged (unscoped).
- **Delete by filter**: same fields for lifecycle cleanup.

Example (HTTP):

```bash
# Ingest into a scope
curl -s localhost:8080/api/ingest -H 'Content-Type: application/json' -d '{
  "text": "Internal runbook for team-a",
  "scope": "team-a",
  "clearance": "internal"
}'

# Search only that scope, max internal
curl -s localhost:8080/api/search -H 'Content-Type: application/json' -d '{
  "query": "runbook",
  "scope": "team-a",
  "max_clearance": "internal"
}'
```

This is **payload isolation for agents**, not multi-tenant authentication.

## Agent documentation

See **[docs/AGENTS.md](docs/AGENTS.md)** for:

- Tool matrix (MCP ↔ HTTP)
- When to use `search` vs `get_relevant_context`
- Scoped filtering / hybrid retrieval
- Claude Desktop / Cursor / stdio vs `serve` setup
- Stable Qdrant payload schema

| Doc | Contents |
|-----|----------|
| [docs/ROADMAP.md](docs/ROADMAP.md) | Shipped phases P0–P5 + backlog |
| [docs/PLAN.md](docs/PLAN.md) | Design rationale |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Crate graph, data flows, scaling |
| [docs/AUDIT.md](docs/AUDIT.md) | Maintainability audit dispositions |
| [AnythingLLM comparison](liberado-qdrant-mcp_vs_AnythingLLM_Analysis_and_Implementation_Roadmap.md) | Gaps vs AnythingLLM knowledge/MCP layer (~80–85% headless path) |

### Ingest parity

MCP, HTTP (`POST /api/ingest*`), and CLI `ingest` all use the same
structure-aware chunking (`RagCore::expand_to_chunks`) so search quality does
not depend on which surface ingested the document.
