# Agent guide — liberado-qdrant-mcp (lqm)

Headless RAG for LLM agents: **Qdrant + embeddings + MCP/HTTP**. No chat UI.

## When to use which tool

| Need | Prefer | Avoid |
|------|--------|--------|
| Paste LLM-ready passages with citations | `get_relevant_context` | Raw `search` (unless you post-process yourself) |
| Inspect scores / build custom prompts | `search` | — |
| Rare exact tokens dense may bury | `search` with `hybrid=true` (optional `hybrid_alpha`, lower = more keyword) | Expecting pure dense to match nonsense tokens |
| Isolate one team/project partition | `scope` on ingest + `scope` on search | Relying on freeform tags alone for hard isolation |
| Cap sensitivity of hits | `clearance` on ingest + `max_clearance` on search | Multi-user auth (not provided — payload isolation only) |
| Add knowledge from text | `ingest_text` | — |
| Add a file or vault tree | `ingest_path` | — |
| Add a webpage | `ingest_url` | — |
| Batch many items | `ingest_many` | N sequential single-item tools when possible |
| See what is already indexed | `list_sources` | — |
| Remove one document | `delete_by_source` | `delete_collection` (wipes everything) |
| Create a scoped KB | `create_collection` | — |
| Check embedder dim/model | `get_embedder_info` | Guessing dims from docs alone |
| Save a long-term preference/fact | `store_memory` | Putting prefs into a random doc collection without `source_type=memory` |
| Retrieve past prefs/facts by meaning | `recall_memories` (optional `use_recency`) | Full-collection `search` without memory filter |

**Search vs context:** `search` returns JSON hits (`text`, `score`, `payload`) plus pagination (`offset`, `has_more`, `next_offset`). `get_relevant_context` reuses the same filters/pagination, optionally applies MMR and a char budget, and returns markdown with numbered passages and a `sources` array.

**Hybrid retrieval:** set `hybrid=true` on `search` / `get_relevant_context` (or JSON body for HTTP). Core over-fetches dense hits, merges keyword-matching scroll candidates, and fuses with weighted dense + keyword scores and RRF. Response includes `"hybrid": true`. Default remains dense-only when `hybrid` is omitted/false. **Scaling:** keyword merge scrolls the collection (O(n) payloads); fine for homelab size; see `docs/ARCHITECTURE.md` scaling section for large corpora.

**Ingest parity:** MCP, HTTP, and CLI all structure-aware-chunk via `RagCore::expand_to_chunks` before embed/upsert.

**Scoped filtering:** optional payload keys `scope` (exact partition) and `clearance` (`public` | `internal` | `confidential` | `restricted`). Search/context accept `scope` and `max_clearance` (admits that level and all lower). Unscoped search is the default. Delete-by-filter accepts the same constraints. Not multi-user auth — agents pass the scope they are allowed to see.

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
| `store_memory` | `POST /api/memories` |
| `recall_memories` | `POST /api/memories/recall` |

Errors on HTTP are JSON: `{ "code": "validation_error", "message": "...", "error": "..." }` (`error` mirrors `message` for older clients).

### Memories

- Default collection: **`memories`** (`DEFAULT_MEMORY_COLLECTION`)
- Points use `source_type=memory`, `source=memory://{memory_id}`, plus `importance` (0–1), `last_accessed` (unix secs string), `memory_id`
- `store_memory` reuses skip/replace-by-source (same id+text → skip; same id new text → replace)
- `recall_memories` semantic search filtered to `source_type=memory`; `use_recency=true` re-ranks with importance + exponential recency blend (host still generates answers)
- Generation is **not** performed server-side

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

## Storage model (what agents can read)

Qdrant stores **chunk text + metadata + vectors**. `source` is a **pointer**
(path/URL/id), not a copy of the original file. Search and
`get_relevant_context` return **passages and citations**, not raw embeddings.
Original binaries are not in lqm—use other MCPs with `source` if you need the
file itself. See `docs/ARCHITECTURE.md` and decision **016**.

## Stable payload schema (Qdrant points)

Written by the shared upsert path:

| Key | Meaning |
|-----|---------|
| `text` | Chunk body |
| `ingest_hash` | SHA-256 of text |
| `source` | Origin path/URL/id |
| `source_type` | e.g. text, webpage, pdf, markdown, code, memory, `audio_placeholder` |
| `tags` | string array |
| `project` | optional scope |
| `timestamp` / `last_modified` | optional strings |
| `chunk_index` | 0-based index in parent doc |
| `total_chunks` | parent doc chunk count |
| `embedding_model` | model name or embedder id |
| `importance` | memory weight 0–1 as **string** (memories; survives Qdrant StringValue round-trip) |
| `last_accessed` | unix seconds string (memories) |
| `memory_id` | stable memory identifier |
| `scope` | isolation partition (exact match when filtering) |
| `clearance` | `public` / `internal` / `confidential` / `restricted` (default `public` on upsert) |

## Live tests

Integration smokes skip when Qdrant is down (CI-safe). Hard-require:

```bash
LQM_LIVE=1 QDRANT_URL=http://127.0.0.1:6334 cargo test -p lqm-mcp live_smoke
```
