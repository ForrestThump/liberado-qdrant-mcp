# Agent guide — liberado-qdrant-mcp (lqm)

Headless RAG for LLM agents: **Qdrant + embeddings + MCP/HTTP**. No chat UI.

## When to use which tool

| Need | Prefer | Avoid |
|------|--------|--------|
| Paste LLM-ready passages with citations | `get_relevant_context` | Raw `search` (unless you post-process yourself) |
| Inspect scores / build custom prompts | `search` | — |
| Add knowledge from text | `ingest_text` | — |
| Add a file or vault tree | `ingest_path` | — |
| Add a webpage | `ingest_url` | — |
| Batch many items | `ingest_many` | N sequential single-item tools when possible |
| See what is already indexed | `list_sources` | — |
| Remove one document | `delete_by_source` | `delete_collection` (wipes everything) |
| Create a scoped KB | `create_collection` | — |
| Check embedder dim/model | `get_embedder_info` | Guessing dims from docs alone |

**Search vs context:** `search` returns JSON hits (`text`, `score`, `payload`) plus pagination (`offset`, `has_more`, `next_offset`). `get_relevant_context` reuses the same filters/pagination, optionally applies MMR and a char budget, and returns markdown with numbered passages and a `sources` array.

## MCP tool matrix ↔ HTTP

| MCP tool | HTTP |
|----------|------|
| `ingest_text` | `POST /api/ingest` |
| `ingest_path` | `POST /api/ingest/path` `{ "path": "..." }` |
| `ingest_url` | `POST /api/ingest/url` `{ "url": "..." }` |
| `ingest_many` | `POST /api/ingest/many` `{ "texts"?, "paths"?, "urls"? }` |
| `search` | `POST /api/search` |
| `get_relevant_context` | `POST /api/context` |
| `list_collections` | `GET /api/collections` |
| `create_collection` | `POST /api/collections` |
| `get_collection_info` | `GET /api/collections/{name}` |
| `delete_collection` | `DELETE /api/collections/{name}` |
| `list_sources` | `GET /api/collections/{name}/sources` |
| `delete_by_source` | `DELETE /api/collections/{name}/sources/{source}` |
| `delete_by_filter` | `POST /api/collections/{name}/delete_by_filter` |
| `get_embedder_info` | `GET /api/embedder` |

Errors on HTTP are JSON: `{ "code": "validation_error", "message": "...", "error": "..." }` (`error` mirrors `message` for older clients).

## Run modes

### stdio (default — Claude Desktop, Cursor, etc.)

```bash
# Env
export QDRANT_URL=http://127.0.0.1:6334
# optional: LQM_CONFIG=/path/to/lqm.toml

lqm-mcp
```

Claude Desktop `claude_desktop_config.json` sketch:

```json
{
  "mcpServers": {
    "liberado-qdrant": {
      "command": "lqm-mcp",
      "args": [],
      "env": {
        "QDRANT_URL": "http://127.0.0.1:6334",
        "RUST_LOG": "info"
      }
    }
  }
}
```

Cursor / other hosts: same pattern — spawn `lqm-mcp` with `QDRANT_URL` set.

### HTTP serve (persistent process)

```bash
lqm-mcp serve --host 0.0.0.0 --port 3000
# MCP streamable HTTP on that bind

# REST API (separate binary)
lqm-api --host 127.0.0.1 --port 8080
```

Optional API auth: set `LQM_API_TOKEN=secret` then send `Authorization: Bearer secret` on all `/api/*` routes (`/health` stays open).

## Stable payload schema (Qdrant points)

Written by the shared upsert path:

| Key | Meaning |
|-----|---------|
| `text` | Chunk body |
| `ingest_hash` | SHA-256 of text |
| `source` | Origin path/URL/id |
| `source_type` | e.g. text, webpage, pdf |
| `tags` | string array |
| `project` | optional scope |
| `timestamp` / `last_modified` | optional strings |
| `chunk_index` | 0-based index in parent doc |
| `total_chunks` | parent doc chunk count |
| `embedding_model` | model name or embedder id |

## Live tests

Integration smokes skip when Qdrant is down (CI-safe). Hard-require:

```bash
LQM_LIVE=1 QDRANT_URL=http://127.0.0.1:6334 cargo test -p lqm-mcp live_smoke
```
