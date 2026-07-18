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

**Still partial after Phase 1:** `ingest_hash` is stored/indexed but **not** used
to skip or replace duplicates; document lifecycle tools are missing; HTTP API
lags MCP.

---

## Priority order (by leverage)

Leverage = agent payoff ÷ implementation effort. Do **P0 before P1**, etc.
Within a band, list order is the suggested implementation sequence.

| Band | Theme | Why high leverage |
|------|--------|-------------------|
| **P0** | Document lifecycle + true idempotency | Agents cannot curate KBs without list/delete/replace; highest gap vs AnythingLLM knowledge backend |
| **P1** | Richer retrieval | Better filters/pagination/context budget → better answers without more product |
| **P2** | Ingest quality & reporting | Fewer bad chunks; agents can act on per-file errors |
| **P3** | MCP ↔ API parity + agent ergonomics | “Backend + API” story; adoption friction |
| **P4** | Memories | Agent long-term notes; valuable but after curation/retrieval |
| **P5** | Nice-to-haves | Lower payoff or out of core headless path |

---

## P0 — Document lifecycle + idempotency (do next)

*Highest leverage remaining. Collections are write-mostly dumps until this lands.*

- [ ] **Core: source lifecycle**
  - `list_sources(collection)` — distinct `source` (+ counts, `source_type`, sample `last_modified`)
  - `delete_by_source(collection, source)`
  - `delete_by_filter` (tags / `source_type` / `project` at minimum)
  - Scroll/count helpers as needed (Qdrant scroll + payload filters)
- [ ] **True re-ingest policy** (use existing `ingest_hash` index)
  - Same source + same hash → **skip**
  - Same source + different hash → **delete old points for source, then upsert**
  - Report `{ inserted, skipped, replaced, chunks }` on ingest paths
- [ ] **MCP tools** for list/delete-by-source/filter; extend ingest responses
- [ ] **HTTP API** mirrors of the same (do not leave lifecycle MCP-only)
- [ ] **Live smoke:** create → ingest → list_sources → re-ingest (skip/replace) → delete_by_source → search confirms

**Done when:** an agent can fully manage a throwaway KB without `delete_collection`.

---

## P1 — Richer retrieval

*High payoff once content can be curated; builds on existing search/context tools.*

- [ ] **Richer filters** shared by `search` and `get_relevant_context`
  - `source`, `project`, tag must / should / must_not
  - Optional small JSON filter blob → Qdrant `Filter` (keep MCP args ergonomic)
- [ ] **Offset / pagination** (+ `has_more` or explicit next offset)
- [ ] **Context budget** on `get_relevant_context` (max chars/tokens, always-cite sources, clear empty-result copy)
- [ ] Optional **simple rerank or MMR** behind a flag (defer heavy cross-encoders if costly)

**Done when:** filtered, paginated context stays within a token budget with stable citations.

---

## P2 — Ingest quality & reporting

*Medium effort; large quality gain for vaults/code/web agents use daily.*

- [ ] **Structured ingest reports** — per-file `{ path, ok|error, chunks }` for `ingest_path`; same shape where useful for URL/text
- [ ] **Markdown heading-aware chunking**
- [ ] **Code-aware chunking** (function/class boundaries where cheap)
- [ ] **HTML hardening** — title/metadata, size/timeout limits, less boilerplate
- [ ] **PDF** — enable or document clearly for default MCP deploy if homelab needs it
- [ ] Optional **`ingest_many`** batch tool for agent loops

**Done when:** bulk path ingest returns actionable errors and chunks respect doc structure.

---

## P3 — Surface parity + agent ergonomics

*Unlocks “backend + API” replacement and reduces tool-calling mistakes.*

- [ ] **HTTP API parity with MCP** — create/info collection, ingest path/url, get_relevant_context, source lifecycle
- [ ] **Stable payload schema** — document + optionally add `chunk_index`, `total_chunks`, `embedding_model`
- [ ] **`get_embedder_info`** tool/endpoint (id, dim, model) — avoid dim mismatches
- [ ] **Structured errors** — machine-readable `code` + message (not only free-text internal errors)
- [ ] **Agent docs** — tool matrix, when to use `search` vs `get_relevant_context`, Claude Desktop / Cursor / stdio vs `serve` examples
- [ ] Optional **HTTP bearer** for non-private binds
- [ ] **CI:** optional Qdrant service job so live smoke is not local-only
- [ ] Mock HTTP tests for Ollama/OpenAI embedders (lower urgency than live Qdrant smoke)

**Done when:** every MCP capability has an HTTP equivalent and docs teach the tool set.

---

## P4 — Memories (Phase 3)

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
