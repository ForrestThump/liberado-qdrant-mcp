# Roadmap

Living document. Move items from active phases into **Shipped** as they land.
Prioritization is by **leverage** (payoff / effort) for the stated goal:
**headless agent tooling that can replace AnythingLLM’s backend knowledge layer
+ HTTP API + MCP** — not chat UI, multi-user product, or agent orchestration.

**Architecture rule:** new capability → `lqm-core` first → thin MCP + API (+ CLI
if ops-useful) in the same change. Live smoke against Qdrant for every new tool.

---

## North star (scope reminder)

| In scope | Out of scope (for now) |
|----------|-------------------------|
| Collections as scoped knowledge bases | Multi-user auth / multi-tenancy |
| Ingest (text, path, URL, more formats) | Full chat UI / workspace chat history |
| Semantic search + LLM-ready context | AnythingLLM agent CRUD / `invoke_agent` |
| Document lifecycle (list/delete/replace) | Broad SaaS connectors (Notion, etc.) before lifecycle is solid |
| MCP + HTTP API parity over `RagCore` | Competing on product shell |

Rough progress as a **headless agent KB** (P0–P4 + hybrid/scope shipped):
~**80–85%** of the AnythingLLM *agent knowledge/retrieval* path without product
surface. Remaining: scale (sparse hybrid), audio transcription, offline MCP
tests, optional SPA/connectors. See
[`liberado-qdrant-mcp_vs_AnythingLLM_Analysis_and_Implementation_Roadmap.md`](../liberado-qdrant-mcp_vs_AnythingLLM_Analysis_and_Implementation_Roadmap.md).

---

## Shipped

### Foundations (M0–M8)

- **M0 — Scaffolding** (Cargo workspace, crate skeletons, docs, AGENTS.md)
- **M1 — Core MVP** (Embedder trait, FakeEmbedder, chunking, Qdrant wrapper, types, semaphore)
- **M2 — MCP binary (stdio)** (turbomcp: ingest_text, search, list_collections)
- **M3 — Dual mode** (stdio + streamable HTTP via `lqm-mcp serve`)
- **M4 — CLI + benchmarking** (ingest/list/delete/bench, file walker, mtime)
- **M5 — Hash + indexes** (SHA256 `ingest_hash` stored, auto payload indexes, `last_modified`)
- **M6 — More mediums** (PDF behind `pdf` feature, audio as `audio_placeholder`, extension detection)
- **M7 — Ollama + OpenAI embedders** (feature-gated, TOML + env, factory)
- **M8 — HTTP API + simple UI** (axum REST, dark-mode search page)

### Post-M8 hardening

- Audit fixes, fastembed default + zero-config model, structured logging
- `ingest_path` MCP tool; CI (fmt, clippy, test)
- Deploy path (Dockerfile, origin policy, on-demand collection create)

### Phase 1 — Agent RAG foundation (high leverage, shipped)

- Collection tools: `create_collection`, `delete_collection`, `get_collection_info`
- Remote ingest: `fetch_url` / HTML extract + `ingest_url`
- Agent context: `format_relevant_context` + `get_relevant_context`
- Live `test_all_mcp_tools_live_smoke` against real Qdrant
- Search `limit=0` → empty results (Qdrant-safe)

### Phase P0 — Document lifecycle + true idempotency (shipped)

- **Source lifecycle (core)** — `list_sources`, `delete_by_source`, `delete_by_filter` (tags / source_type / project); scroll + count helpers
- **Re-ingest policy** — same source + same hash multiset → skip; different → delete source then upsert; `IngestReport { inserted, skipped, replaced, chunks }`
- **MCP** — `list_sources`, `delete_by_source`, `delete_by_filter`; ingest tools return accounting fields
- **HTTP API** — create/info collection; list/delete sources; delete_by_filter; ingest stats
- **Live smoke** — `test_p0_lifecycle_live_smoke` + extended `test_all_mcp_tools_live_smoke`
- Indexes: `project` + `tags` added to auto-created keyword indexes

**Done when (met):** an agent can fully manage a throwaway KB without `delete_collection`.

---

## Priority order (by leverage)

Leverage = agent payoff ÷ implementation effort. Do **P1 before P2**, etc.
Within a band, list order is the suggested implementation sequence.

| Band | Theme | Why high leverage | Status |
|------|--------|-------------------|--------|
| **P0** | Document lifecycle + true idempotency | Agents cannot curate KBs without list/delete/replace | **Shipped** |
| **P1** | Richer retrieval | Better filters/pagination/context budget → better answers without more product | **Shipped** |
| **P2** | Ingest quality & reporting | Fewer bad chunks; agents can act on per-file errors | **Shipped** |
| **P3** | MCP ↔ API parity + agent ergonomics | Remaining parity (path/url ingest API, docs, errors) | **Shipped** |
| **P4** | Memories | Agent long-term notes; valuable but after curation/retrieval | **Shipped** |
| **P5** | Nice-to-haves | Lower payoff or out of core headless path | **Hybrid + scoped filtering shipped**; rest backlog |

---

## P0 — Document lifecycle + idempotency (shipped)

- [x] **Core: source lifecycle**
  - `list_sources(collection)` — distinct `source` (+ counts, `source_type`, sample `last_modified`)
  - `delete_by_source(collection, source)`
  - `delete_by_filter` (tags / `source_type` / `project` at minimum)
  - Scroll/count helpers as needed (Qdrant scroll + payload filters)
- [x] **True re-ingest policy** (use existing `ingest_hash` index)
  - Same source + same content hash → **skip**
  - Same source + different content → **delete old points for source, then upsert**
  - Report `{ inserted, skipped, replaced, chunks }` on ingest paths
- [x] **MCP tools** for list/delete-by-source/filter; extend ingest responses
- [x] **HTTP API** mirrors of the same (do not leave lifecycle MCP-only)
- [x] **Live smoke:** create → ingest → list_sources → re-ingest (skip/replace) → delete_by_source → search confirms

**Done when:** an agent can fully manage a throwaway KB without `delete_collection`. ✅

---

## P1 — Richer retrieval (shipped)

*High payoff once content can be curated; builds on existing search/context tools.*

- [x] **Richer filters** shared by `search` and `get_relevant_context`
  - `source`, `project`, tag must (`tags`) / should (`tags_should`) / must_not (`tags_must_not`)
  - `SearchFilter` → Qdrant `Filter` via `search_filter_to_qdrant` (ergonomic MCP args, not raw JSON blob)
- [x] **Offset / pagination** — `SearchPage { results, offset, limit, has_more, next_offset }` (limit+1 probe)
- [x] **Context budget** on `get_relevant_context` — `max_chars_per_passage`, `max_total_chars`, always-cite sources, clearer empty-result tip
- [x] Optional **simple MMR** (`mmr` / `mmr_lambda`) — score + token-Jaccard diversity, pure post-process

**Done when:** filtered, paginated context stays within a token budget with stable citations. ✅

### Phase P1 notes (shipped)

- Core: `SearchOptions` / `search_page`, `ContextOptions` / `format_relevant_context_with`, `mmr_rerank`
- MCP + API: same filter/pagination fields; `POST /api/context` for formatted retrieval

---

## P2 — Ingest quality & reporting (shipped)

*Medium effort; large quality gain for vaults/code/web agents use daily.*

- [x] **Structured ingest reports** — per-file `{ path, ok, error, chunks }` via `file_results` on `ingest_path` / `ingest_many` / text+url
- [x] **Markdown heading-aware chunking** — AT1–H6 sections then size limits (`chunk_markdown`)
- [x] **Code-aware chunking** — fn/def/class/struct boundaries (`chunk_code`)
- [x] **HTML hardening** — `<title>`, strip nav/footer/header, 2MB body cap, timeout
- [x] **PDF** — `lqm-mcp` enables `lqm-ingest` `pdf` feature by default
- [x] **`ingest_many`** — batch texts/paths/urls with one upsert and per-item results

**Done when:** bulk path ingest returns actionable errors and chunks respect doc structure. ✅

### Phase P2 notes

- Core: `chunk_for_ingest` / `ChunkKind` dispatch from path extension + source_type
- MCP: structure-aware chunking on all ingest tools; `file_results` accounting

---

## P3 — Surface parity + agent ergonomics (shipped)

*Unlocks “backend + API” replacement and reduces tool-calling mistakes.*

- [x] **HTTP API parity with MCP** — path/url/many ingest routes; create/info/sources/search/context already present
- [x] **Stable payload schema** — `chunk_index`, `total_chunks`, `embedding_model` on upsert; constants in `payload_schema`
- [x] **`get_embedder_info`** — MCP tool + `GET /api/embedder` (id, dimension, model)
- [x] **Structured errors** — HTTP `{ code, message, error }`; shared `api_error` helpers
- [x] **Agent docs** — `docs/AGENTS.md` tool matrix, search vs context, stdio/`serve`
- [x] **HTTP bearer** — optional `LQM_API_TOKEN` → `Authorization: Bearer …` on `/api/*`
- [x] **CI:** optional `live-qdrant` job with Qdrant service + `LQM_LIVE=1` smokes
- [x] Offline embedder response parsers (`parse_ollama_embeddings` / `parse_openai_embeddings`) unit-tested (full wiremock HTTP still optional)

**Done when:** every MCP capability has an HTTP equivalent and docs teach the tool set. ✅

### Phase P3 notes

- See `docs/AGENTS.md` for the tool/API matrix and host config sketches
- Payload keys: `docs/AGENTS.md` § Stable payload schema

---

## P4 — Memories (shipped)

*Long-term agent notes; generation stays in the host agent.*

- [x] Memory schema — default collection `memories`, `source_type=memory`, payload `importance` / `last_accessed` / `memory_id`
- [x] `store_memory` / `recall_memories` on RagCore + MCP + HTTP (`POST /api/memories`, `POST /api/memories/recall`)
- [x] Optional recency blend (`use_recency`, half-life post-process over semantic scores)
- [x] Pure unit tests for payload + blend ranking; live smoke `test_p4_memory_live_smoke` (skip if no Qdrant)

**Done when:** agents can store notes and recall them by query without a chat-with-workspace tool. ✅

### Phase P4 notes

- Constants: `DEFAULT_MEMORY_COLLECTION`, `MEMORY_SOURCE_TYPE` in `lqm_core::memory`
- Replace-by-source: same text+id re-ingest skips; changed text replaces via `memory://{id}` source

---

## P5 — Hybrid retrieval (partial; headless slice shipped)

*Highest-leverage P5 item first: dense + keyword fusion without a second DB.*

### Shipped

- [x] Hybrid search mode on `SearchOptions` / `RagCore::search_page` (`hybrid`, `hybrid_alpha`)
- [x] Pure fusion helpers in `lqm_core::hybrid` (tokenize, keyword_score, weighted + RRF fuse, merge candidates)
- [x] MCP `search` / `get_relevant_context` and HTTP `POST /api/search` + `/api/context` accept `hybrid` / `hybrid_alpha` (dense-only default)
- [x] Unit tests for fusion ranking; live smoke `test_p5_hybrid_live_smoke` (skip if no Qdrant)

**Done when (met for hybrid):** agents can request hybrid retrieval so rare keywords still surface among hits. ✅

### Still backlog (not shipped)

- Audio transcription (whisper-rs) — replace `audio_placeholder` stubs
- Dioxus richer SPA (MCP + API remain priority over UI)
- WASM build of core for browser-side use
- Chat-with-context tool (only if a host cannot generate itself)
- Background re-index workers / heavy queues
- Native Qdrant sparse vectors (optional upgrade; current hybrid is post-query fusion over dense + text; keyword path is O(n) scroll)
- Channel-transport + `McpTestClient` offline MCP integration tests
- Full HTTP router integration tests (`tower::ServiceExt`)

### Scoped filtering (shipped)

- [x] Payload keys `scope` + `clearance` (indexed); pure helpers in `lqm_core::scope`
- [x] `SearchFilter` / `PayloadFilter` constraints `scope` + `max_clearance` → Qdrant filters
- [x] MCP + HTTP ingest/search/context/delete_by_filter thin adapters
- [x] Live smoke `test_scope_filter_live_smoke` (skip if no Qdrant); README + AGENTS docs

**Done when (met):** agents can exclude other scopes / higher clearances without multi-user auth. ✅

---

## Explicit non-goals

Do not prioritize these while P0–P3 are open:

- Multi-user auth, API-key admin consoles, multi-tenancy
- AnythingLLM-style agent CRUD / `invoke_agent`
- Full product chat UI and workspace chat history as a goal
- Breadth-first connector race (Drive, Notion, …) before lifecycle + retrieval are solid

---

## Suggested next milestone (single PR stack)

**P0–P4 (and part of P5) are shipped.** Prefer a single coherent slice from remaining
backlog rather than re-doing early milestones:

1. Channel-transport MCP integration tests (`McpTestClient`) without live Qdrant
2. Sparse / keyword index for hybrid search (replace full collection scroll at scale)
3. Real audio transcription (replace `audio_placeholder` stubs)

Or pick the highest-value remaining P5 item for your deployment (Dioxus SPA, connectors).
)
