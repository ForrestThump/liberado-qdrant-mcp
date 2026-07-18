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

## MCP tools

| Tool | Role |
|------|------|
| `ingest_text` | Structure-aware chunk → embed/upsert; skip/replace by source; `file_results` |
| `ingest_path` | Walk file/dir; per-file `file_results`; md/code chunking; PDF enabled |
| `ingest_url` | Fetch + hardened HTML (title, size cap) → structure-aware chunk/upsert |
| `ingest_many` | Batch texts/paths/urls with one upsert and per-item results |
| `search` | Semantic search; filters (source/project/tags must·should·must_not), offset pagination |
| `get_relevant_context` | Same filters + markdown passages, char budget, optional MMR |
| `list_collections` | Collection names |
| `create_collection` | Create/ensure; dim defaults to active embedder |
| `delete_collection` | Drop collection |
| `get_collection_info` | Points, vector size, distance, status |
| `list_sources` | Distinct sources + counts in a collection |
| `delete_by_source` | Remove all points for one source |
| `delete_by_filter` | Delete by source/source_type/project/tags (AND) |

Ingest tools return `{ inserted, skipped, replaced, chunks }` for agent accounting.

All tools parse args and delegate to `RagCore` / `lqm-ingest` — no Qdrant/embed reimplementation in this binary.

## Testing

Integration tests use turbomcp's `channel` transport + `McpTestClient`.
A connectivity check on Qdrant ensures graceful skip when Qdrant is unavailable.
Pure unit tests for HTML extraction and context formatting live in `lqm-ingest` / `lqm-core`.
