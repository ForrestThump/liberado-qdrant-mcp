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
| Ingest voice notes / podcasts | `ingest_path` on audio files (DeepInfra STT → `source_type=audio`) | Expecting audio binaries in Qdrant; multimodal embeddings |
| Add a webpage | `ingest_url` | — |
| Batch many items | `ingest_many` | N sequential single-item tools when possible |
| See what is already indexed | `list_sources` | — |
| Read all chunks of one source in order | `get_source` / `list_chunks` | Re-running `search` and guessing filters |
| Expand a hit with neighboring chunks | `expand_context` (same `source` + `chunk_index` ±N) | Inventing text outside the index |
| Remove one document | `delete_by_source` | `delete_collection` (wipes everything) |
| Create a scoped KB | `create_collection` (optional `model_label` to tag with embedder name) | — |
| Check embedder dim/model | `get_embedder_info` / `get_collection_info` (recorded + current dim) | Guessing dims from docs alone |
| Save a long-term preference/fact | `store_memory` | Putting prefs into a random doc collection without `source_type=memory` |
| Retrieve past prefs/facts by meaning | `recall_memories` (optional `use_recency`) | Full-collection `search` without memory filter |

**Search vs context:** `search` returns JSON hits (`text`, `score`, `payload`) plus pagination (`offset`, `has_more`, `next_offset`). `get_relevant_context` reuses the same filters/pagination, optionally applies MMR and a char budget, and returns markdown with numbered passages and a `sources` array.

**Hybrid retrieval:** set `hybrid=true` on `search` / `get_relevant_context` (or JSON body for HTTP). Core over-fetches dense hits, merges keyword candidates, and fuses with weighted dense + keyword scores and RRF. Response includes `"hybrid": true`. Default remains dense-only when `hybrid` is omitted/false.

**Keyword backend (process env, not a per-request param):** `LQM_HYBRID_KEYWORD_BACKEND`:

| Value | Use when |
|-------|----------|
| `keyword_index` (default) | Existing dense collections; index-backed text match (no full scroll) |
| `sparse` | Large corpora; recreate/create collection with sparse schema + re-ingest |
| `scroll` | Legacy O(n) full scroll for A/B comparison only |

See `docs/ARCHITECTURE.md` scaling section.

**Ingest parity:** MCP, HTTP, and CLI all structure-aware-chunk via `RagCore::expand_to_chunks` before embed/upsert.

**Scoped filtering:** optional payload keys `scope` (exact partition) and `clearance` (`public` | `internal` | `confidential` | `restricted`). Search/context accept `scope` and `max_clearance` (admits that level and all lower). Unscoped search is the default. Delete-by-filter accepts the same constraints. Not multi-user auth — agents pass the scope they are allowed to see.

**Collection ↔ embedder guarantees:** every `create_collection` records the
active embedder's id + dimension in the reserved `_lqm_config` collection.
`search` and all ingest tools validate the current embedder dim matches the
target collection before performing expensive operations, returning a clear
error instead of a cryptic Qdrant gRPC rejection. Collections created before
this feature (no config entry) pass validation for backward compatibility.
Use `get_collection_info` to inspect the recorded and current embedder dims
side by side.

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
| `create_collection` | `POST /api/collections` `{ "name", "vector_dim"?, "model_label"? }` |
| `get_collection_info` | `GET /api/collections/{name}` (also returns `embedder_id`, `recorded_vector_dim`, `current_embedder_id`) |
| `delete_collection` | `DELETE /api/collections/{name}` |
| `list_sources` | `GET /api/collections/{name}/sources` |
| `list_chunks` | `GET /api/collections/{name}/sources/{source}/chunks?offset=&limit=` |
| `get_source` | `GET /api/collections/{name}/sources/{source}` |
| `expand_context` | `POST /api/expand_context` `{ "source", "chunk_index", "neighbors"?, "collection"? }` |
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
# optional: LQM_HYBRID_KEYWORD_BACKEND=keyword_index|sparse|scroll
# audio STT (DeepInfra file ASR — feature asr-deepinfra on MCP/API/CLI):
# export DEEPINFRA_API_KEY=...          # or DEEPINFRA_TOKEN
# export LQM_ASR_BACKEND=whisper        # default; or nemotron
# export LQM_ASR_MODEL=openai/whisper-large-v3-turbo  # optional slug override
# export LQM_ASR_FALLBACK_PLACEHOLDER=1 # optional: placeholder when key missing

lqm-mcp
```

**Audio:** `ingest_path` / file items in `ingest_many` use async extract. With
`DEEPINFRA_API_KEY` (or `TOKEN`), audio extensions are transcribed to text and
stored as `source_type=audio` (file stays on disk as a pointer only). Without a
key, ingest **errors** unless `LQM_ASR_FALLBACK_PLACEHOLDER=1` (then
`audio_placeholder`). Filter real transcripts with `source_type=audio`; exclude
stubs with `audio_placeholder`.

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

**Source reconstruction:** after a hit, call `list_chunks` / `get_source` with
that `source` to read indexed chunks **ordered by `chunk_index`** (paginated via
`offset`/`limit`/`has_more`/`next_offset`). Use `expand_context` for ±N neighbors
of the same source around a `chunk_index`. Missing `chunk_index` sorts last.
This rebuilds text from the index only — not original files.

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

## Tests

**Offline MCP** (no Qdrant): `McpTestClient` exercises tool registration and a
subset of dispatch (`get_embedder_info`, validation errors) with FakeEmbedder:

```bash
cargo test -p lqm-mcp offline_mcp
```

**Live smokes** skip when Qdrant is down (CI-safe). Hard-require:

```bash
LQM_LIVE=1 QDRANT_URL=http://127.0.0.1:6334 cargo test -p lqm-mcp live_smoke
```
