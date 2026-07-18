# lqm-api

HTTP server for REST clients and a minimal static search UI. Reuses `lqm-core`
for all RAG logic; stays in parity with MCP tools (see `docs/AGENTS.md` matrix).

## Stack

- **HTTP:** axum 0.8
- **CORS / static:** tower-http
- **Core:** `RagCore::from_env`; structure-aware `expand_chunks` → `expand_to_chunks`
- **Auth:** optional `LQM_API_TOKEN` → Bearer on `/api/*` (`/health` open)

## Endpoints

| Method | Path | Notes |
|--------|------|--------|
| `GET` | `/health` | `{ status, version }` |
| `GET` | `/api/collections` | list |
| `POST` | `/api/collections` | create |
| `GET` | `/api/collections/{name}` | info |
| `DELETE` | `/api/collections/{name}` | delete |
| `GET` | `/api/collections/{name}/sources` | list_sources |
| `GET` | `/api/collections/{name}/sources/{source}` | get_source |
| `GET` | `/api/collections/{name}/sources/{source}/chunks` | list_chunks (`offset`/`limit` query) |
| `DELETE` | `/api/collections/{name}/sources/{source}` | delete_by_source |
| `POST` | `/api/collections/{name}/delete_by_filter` | filter delete |
| `POST` | `/api/search` | filters, hybrid, pagination |
| `POST` | `/api/context` | get_relevant_context |
| `POST` | `/api/expand_context` | neighbor window for source + chunk_index |
| `POST` | `/api/ingest` | text; **structure-aware chunking** |
| `POST` | `/api/ingest/path` | file/dir |
| `POST` | `/api/ingest/url` | remote URL |
| `POST` | `/api/ingest/many` | batch |
| `GET` | `/api/embedder` | embedder info |
| `POST` | `/api/memories` | store_memory |
| `POST` | `/api/memories/recall` | recall_memories |
| `GET` | `/` | static search UI |
| — | `static/` | ServeDir fallback |

Errors: JSON `{ "code", "message", "error" }` via `lqm_core::structured_error`.

## Configuration

```
lqm-api --host 127.0.0.1 --port 8080
lqm-api --config embedder.toml --qdrant-url http://localhost:6334
# or QDRANT_URL env
```

## Static UI

`static/index.html` is an interim dark-mode search page until a Dioxus SPA
(`docs/ROADMAP.md` later/optional). Not the product goal — agents should prefer MCP/HTTP tools.
