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

Rough progress as a **headless agent KB** (post Phase 1): ~55–60%. Phase 2A–2B
targets ~80%+ of the AnythingLLM agent knowledge path without product surface.

---

## Shipped

### Foundations (M0–M8)

- **M0 — Scaffolding** (Cargo workspace, crate skeletons, docs, AGENTS.md)
- **M1 — Core MVP** (Embedder trait, FakeEmbedder, chunking, Qdrant wrapper, types, semaphore)
- **M2 — MCP binary (stdio)** (turbomcp: ingest_text, search, list_collections)
- **M3 — Dual mode** (stdio + streamable HTTP via `lqm-mcp serve`)
- **M4 — CLI + benchmarking** (ingest/list/delete/bench, file walker, mtime)
- **M5 — Hash + indexes** (SHA256 `ingest_hash` stored, auto payload indexes, `last_modified`)
- **M6 — More mediums** (PDF behind `pdf` feature, audio placeholder, extension detection)
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
| **P4** | Memories | Agent long-term notes; valuable but after curation/retrieval | Next |
| **P5** | Nice-to-haves | Lower payoff or out of core headless path | Open |

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
- [ ] Mock HTTP tests for Ollama/OpenAI embedders (deferred — lower urgency)

**Done when:** every MCP capability has an HTTP equivalent and docs teach the tool set. ✅

### Phase P3 notes

- See `docs/AGENTS.md` for the tool/API matrix and host config sketches
- Payload keys: `docs/AGENTS.md` § Stable payload schema

---

## P4 — Memories (do next)

*Valuable for long-running agents; weaker leverage until P0–P1 exist.*

- [ ] Memory schema (dedicated collection or `source_type=memory` + `importance` / `last_accessed`)
- [ ] `store_memory` / `recall_memories` (semantic + optional recency)
- [ ] Keep generation in the host agent (no required chat-with-workspace tool)

---

## P5 — Backlog (lower leverage or non-core)

- Hybrid / sparse + dense search (if keyword misses hurt after P1)
- Audio transcription (whisper-rs) — replace audio placeholder
- Clearance-safe / scoped filtering beyond simple payload tags
- Dioxus richer SPA (MCP + API remain priority over UI)
- WASM build of core for browser-side use
- Chat-with-context tool (only if a host cannot generate itself)
- Background re-index workers / heavy queues

---

## Explicit non-goals

Do not prioritize these while P0–P3 are open:

- Multi-user auth, API-key admin consoles, multi-tenancy
- AnythingLLM-style agent CRUD / `invoke_agent`
- Full product chat UI and workspace chat history as a goal
- Breadth-first connector race (Drive, Notion, …) before lifecycle + retrieval are solid

---

## Suggested next milestone (single PR stack)

Ship **P0** as one coherent slice:

1. Core scroll/filter + `list_sources` / `delete_by_source` (+ filter delete)
2. Ingest skip/replace-by-source using `ingest_hash`
3. MCP + API wiring + live smoke extensions

Then **P1** filters/offset/context budget, then **P2** chunking/reports, then **P3** parity/docs.
)
