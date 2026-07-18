# liberado-qdrant-mcp

Lightweight **headless RAG** for LLM agents: Qdrant vectors, pluggable embedders,
MCP tools (`lqm-mcp`) and HTTP API (`lqm-api`).

Built to replace AnythingLLM’s **knowledge / retrieval** layer for agents—not
chat UI, multi-user product shell, or agent orchestration. Rough coverage of
that headless path: ~**80–85%** (see [gap map](liberado-qdrant-mcp_vs_AnythingLLM_Analysis_and_Implementation_Roadmap.md)).

## Capabilities (shipped)

| Area | What agents get |
|------|-----------------|
| Collections | create / list / info / delete |
| Ingest | text, path, URL, batch (`ingest_many`); structure-aware chunking; skip/replace by content hash |
| Lifecycle | `list_sources`, `delete_by_source`, `delete_by_filter` |
| Source reconstruction | `list_chunks` / `get_source` (ordered by `chunk_index`, paginated), `expand_context` (±N neighbors) |
| Retrieval | filtered search, pagination, `get_relevant_context`, optional hybrid (keyword_index / sparse / scroll backends) + MMR |
| Memories | `store_memory` / `recall_memories` with optional recency blend |
| Isolation | payload `scope` + `clearance` / `max_clearance` (not multi-user auth) |
| Surfaces | MCP (stdio + `serve`) ↔ HTTP parity; CLI for ops |

Full tool matrix and payload keys: **[docs/AGENTS.md](docs/AGENTS.md)**.

## Storage model

Qdrant points hold **dense vectors + payload**, not a media vault.

- **Stored:** chunk **text**, metadata (`source`, `source_type`, tags, chunk
  indices, scope/clearance, …), and the embedding.
- **`source` is a pointer** (file path, URL, or id). Originals stay where they
  already live; agents open them with other MCPs if needed.
- **Not stored:** PDF/audio/image **binaries**. PDF ingest keeps extracted text
  only; audio is currently a filterable placeholder until transcription lands.

Search / context return **readable passages + citations**, not raw vectors.

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
| [docs/ROADMAP.md](docs/ROADMAP.md) | **Next** work only (forward-looking sequence) |
| [docs/PLAN.md](docs/PLAN.md) | Design rationale |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Crate graph, data flows, storage model, scaling |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Architectural decision log |
| [docs/AUDIT.md](docs/AUDIT.md) | Maintainability audit dispositions |
| [AnythingLLM comparison](liberado-qdrant-mcp_vs_AnythingLLM_Analysis_and_Implementation_Roadmap.md) | Capability matrix vs AnythingLLM knowledge/MCP layer |

### Ingest parity

MCP, HTTP (`POST /api/ingest*`), and CLI `ingest` all use the same
structure-aware chunking (`RagCore::expand_to_chunks`) so search quality does
not depend on which surface ingested the document.
