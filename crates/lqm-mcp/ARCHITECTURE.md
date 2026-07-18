# lqm-mcp

MCP server binary. Agents talk to this; it delegates all work to `lqm-core`.

## Stack

- **MCP SDK:** turbomcp (fork pin in Cargo.toml; `stdio`, `http`, `channel` features)
- **Transport:** stdio (default) + streamable HTTP (`serve` subcommand)
- **Runtime:** tokio
- **CLI:** clap v4 (`env` for `QDRANT_URL` / `LQM_CONFIG`)
- **Construction:** `RagCore::from_env`

## Dual-mode

```
lqm-mcp              → stdio transport (spawn-on-demand)
lqm-mcp serve        → streamable HTTP (persistent, single model in RAM)
  --host 0.0.0.0
  --port …
```

## MCP tools

| Tool | Role |
|------|------|
| `ingest_text` | `expand_to_chunks` → embed/upsert; skip/replace; accounting |
| `ingest_path` | Walk file/dir; per-file `file_results`; PDF feature on |
| `ingest_url` | `fetch_url` + structure-aware chunk/upsert |
| `ingest_many` | Batch texts/paths/urls; one upsert; per-item results |
| `search` | Dense or hybrid; filters; offset pagination |
| `get_relevant_context` | Same filters + markdown, char budget, optional MMR |
| `list_collections` | Names |
| `create_collection` | Create/ensure; dim defaults to embedder |
| `delete_collection` | Drop |
| `get_collection_info` | Points, vector size, distance, status |
| `list_sources` | Distinct sources + counts |
| `delete_by_source` | Remove one source |
| `delete_by_filter` | source / source_type / project / tags / scope / clearance |
| `get_embedder_info` | id, dimension, model |
| `store_memory` / `recall_memories` | Long-term notes (default collection `memories`) |

Ingest tools return `{ inserted, skipped, replaced, chunks }` (+ `file_results` where multi-item).

Optional ingest fields: `scope`, `clearance`. Search/context: `scope`, `max_clearance`, `hybrid`, `hybrid_alpha`.

All tools stay thin: parse args → `RagCore` / `lqm-ingest`.

## Testing

- **Live smokes** against real Qdrant when available (`LQM_LIVE=1` hard-requires).
- Workspace CI skips when Qdrant is down; optional `live-qdrant` job runs smokes.
- Channel transport + `McpTestClient` offline suite is **planned** (see `docs/ROADMAP.md` item 4 / AUDIT TC11), not yet the primary harness.
