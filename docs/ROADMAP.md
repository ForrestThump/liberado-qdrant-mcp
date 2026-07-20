# Roadmap

**Forward-looking only.** What has already shipped lives in:

| Doc | Role |
|-----|------|
| [`README.md`](../README.md) | Capabilities overview, quick start, storage model |
| [`docs/AGENTS.md`](AGENTS.md) | MCP ↔ HTTP tool matrix, payload schema, host setup |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | Crate graph, data flows, scaling boundaries |
| [`docs/DECISIONS.md`](DECISIONS.md) | Why we chose what we chose |
| [`docs/PLAN.md`](PLAN.md) | Design rationale (historical milestones) |
| [AnythingLLM gap map](../liberado-qdrant-mcp_vs_AnythingLLM_Analysis_and_Implementation_Roadmap.md) | Capability matrix vs AnythingLLM knowledge layer |

When a roadmap item lands: implement it, document it in the rows above, then
**remove it from this file** (or move it only into historical docs if needed).
Do not accumulate a “Shipped” checklist here.

---

## Goal

Headless agent knowledge layer: **collections + ingest + lifecycle + retrieval
+ memories** over Qdrant via MCP and HTTP — enough to replace AnythingLLM’s
backend knowledge/MCP path for agents. Not a chat UI, multi-user product, or
agent orchestrator.

**Architecture rule:** new capability → `lqm-core` first → thin MCP + API
(+ CLI if ops-useful) in the same change. Live smoke against Qdrant for every
new tool.

**Storage model (fixed):** Qdrant holds **chunk text + metadata + vectors**.
`source` is a **pointer** (path/URL/id) for agents and other MCPs — originals
are not copied into a blob store. See ARCHITECTURE and DECISIONS.

---

## Active sequence (by leverage)

Do these **in order** unless a deployment constraint forces a jump.

### 1. Collection ↔ embedder hard guarantees

**Leverage:** High — prevents silent dim/model mismatch errors that confuse agents.

**What:** Refuse search/ingest when the configured embedder's dimension or model
doesn't match the target collection. Return clear validation errors with the
expected vs actual dimension.

**Implementation outline:**
- `lqm-core`: store `embedder_id` + `vector_dim` per-collection in a reserved
  `_lqm_config` collection (or as collection metadata). Read on `ensure_collection`
  create; validate in `search_page` and `embed_and_upsert_batch`.
- `lqm-core`: `RagCore::validate_collection_dim(collection_name)` — returns
  `LqmError::Validation` with a human-readable mismatch message.
- `lqm-mcp` / `lqm-api`: `create_collection` accepts optional `model_label` for
  human-readable tracking. `get_collection_info` returns `embedder_id` and
  `vector_dim` alongside existing stats.
- Tests: offline dimension mismatch rejection; live smoke for create+search
  with matching/mismatching embedders.

**Rationale for ordering #1:** Smallest item (~50–80 lines in core). Establishes
the `_lqm_config` collection pattern that item 2 (per-collection chunk config)
depends on, and item 5 (list_sources performance) can share.

### 2. Per-collection chunk configuration

**Leverage:** High — code repos and prose vaults should not share one `chunk_size`.

**What:** Allow `chunk_size`, `chunk_overlap`, and `chunk_kind` to be set
per-collection at create time. Ingest and search paths read the collection's
config; fall back to global defaults when none is set.

**Implementation outline:**
- `lqm-core`: `ChunkConfig` stored per-collection in `_lqm_config` (same pattern
  as item 1). `RagCore::chunk_config_for(collection_name)` reads it.
- `lqm-core`: `expand_to_chunks` takes an optional `ChunkConfig` override;
  `embed_and_upsert_batch` resolves the right config per-collection.
- `lqm-mcp` / `lqm-api`: `create_collection` accepts optional `chunk_size`,
  `chunk_overlap`, `chunk_kind` params.
- `lqm-cli`: `ingest` reads or accepts chunk config.
- Tests: round-trip create+ingest+search with non-default chunk sizes; fallback
  to global defaults for collections created before this feature.

**Rationale for ordering #2:** Depends on the config-collection storage from
item 1. Touches the full ingest pipeline (RagCore, chunking, ensure_collection)
— the largest item in the sequence. Shipping this unlocks per-source-type
tuning for retrieval quality.

### 3. Tool: `get_collection_stats`

**Leverage:** High — the #1 agent ergonomic gap. Agents need to know "what's in
this collection?" before formulating a search strategy.

**What:** MCP `get_collection_stats` and HTTP `GET /api/collections/{name}/stats`
return point counts grouped by `source_type`, `project`, and `tags`. Optional
`group_by` param for custom aggregation.

**Implementation outline:**
- `lqm-core`: `RagCore::collection_stats(name, group_by)` uses repeated
  `QdrantClient::count_points` with filters — cheap gRPC calls, no scroll needed.
  Returns `CollectionStats { total_points, total_sources, source_types: Map,
  projects: Map, tags: Map }` plus dynamic `group_by` support.
- `lqm-mcp`: `#[tool] get_collection_stats(collection, group_by?)`.
- `lqm-api`: `GET /api/collections/{name}/stats?group_by=source_type`.
- Tests: offline via `McpTestClient` (returns error with lazy client); live
  smoke with known data.

**Rationale for ordering #3:** Independent of items 1–2 (pure `count_points`
gRPC, no config collection dependency). Ships fast. Agents can introspect
collections immediately — including chunk config info from item 2 once that
lands alongside.

### 4. Tool: `get_similar_to_source`

**Leverage:** Medium — closes an AnythingLLM parity gap. Common agent workflow:
"I found this doc — what else is related?"

**What:** Embed the joined text of a source and search with that vector,
excluding the source itself. Returns ranked similar sources with scores.

**Implementation outline:**
- `lqm-core`: `RagCore::similar_to_source(source, collection, limit, filters?)`:
  1. Call `get_source(collection, source)` to join all chunk text.
  2. `embed_batch(vec![joined_text])` to get a single vector.
  3. `search_page` with that vector + filter `source != {source}`.
  Return `Vec<SearchResult>`.
- `lqm-mcp`: `#[tool] get_similar_to_source(source, collection?, limit?, …)`.
- `lqm-api`: `POST /api/similar { source, collection?, limit?, … }`.
- Tests: live smoke with two related sources; offline tool registration test.

**Rationale for ordering #4:** Independent — wraps two existing primitives
(`get_source` + `search_page`). Pure additive, no refactoring of existing
paths. Good to ship after stats (item 3) so agents can first explore
collections, find a source, then find similar ones.

### 5. `list_sources` performance overhaul

**Leverage:** High — `list_sources` currently scrolls every payload in the
collection (O(n) points). The biggest perf wart at scale.

**What:** Two-phase approach:
- **Phase A (fast, ships now):** Use Qdrant scroll with payload field mask to
  exclude `text` and vector data — only fetch `source`, `source_type`, `tags`,
  `scope`. Still O(n) but drastically less data over the wire.
- **Phase B (proper, ships after):** Maintain a `_lqm_sources` collection
  keyed by `source` pointer, updated on every `embed_and_upsert_batch`.
  `list_sources` then reads from this small collection (O(sources), not O(points)).
  Pairs naturally with document status tracking.

**Implementation outline (Phase A):**
- `lqm-core`: `QdrantClient::scroll_payloads` accepts optional field mask.
  `RagCore::list_sources` passes `SOURCE_FIELD_MASK` to skip heavy fields.

**Implementation outline (Phase B):**
- `lqm-core`: `RagCore::upsert_source_summary(source, collection, metadata)`
  called from `embed_and_upsert_batch`. `list_sources` queries `_lqm_sources`.
- `lqm-api`: enriched response with `preview`, `total_chars`, `status`.

**Rationale for ordering #5:** Phase A is minimal and safe. Phase B can share
the config-collection infrastructure from items 1–2 (same reserved-collection
pattern). Ships last because it's performance hardening — the feature
work (items 1–4) has higher agent-visible impact.

---

## Later / optional (not sequenced)

Pick only if a concrete need appears:

| Item | Notes |
|------|--------|
| Document status tracking | Per-source ingest status (complete/partial/error); pairs with `list_sources` phase B |
| Richer `list_sources` previews | Sample title / first-chunk preview / total chars; pairs with `list_sources` phase B |
| Additional source extractors (csv, docx, epub, xlsx, pptx, html) | Feature-gated in `lqm-ingest`; csv and docx first |
| Full offline Qdrant double for ingest/search | MVP uses `McpTestClient` + FakeEmbedder + lazy client; full mock needs a seam |
| HTTP router tests (`tower::ServiceExt`) | Companion to offline MCP harness |
| Cross-encoder re-ranking | Trait `Reranker` behind `rerank` feature; post-MMR on `get_relevant_context` |
| Re-ingest / refresh tool | Single-tool delete+re-ingest with `last_modified` skip |
| Background re-index workers | Only if bulk refresh becomes painful |
| Bulk ingest batch sizing cap | Prevent OOM on large `ingest_many` calls |
| `estimate_token_count` tool | Simple tokenizer for context window budgeting |
| Cursor-based pagination (`after`) | More robust than `offset` under concurrent mutation |
| Better error messages with suggestions | Levenshtein collection name suggestions; dim mismatch guidance |
| Sparse vector quality (BM25 / learned sparse) | Beyond current hash-TF; revisit if `keyword_index` is insufficient |
| Dioxus SPA | Demo UX; agents should prefer MCP/HTTP |
| WASM core | Browser-side story, not agent replacement |
| Chat-with-context tool | Only if a host cannot generate from `get_relevant_context` |
| Magic-values audit: `SourceType` on domain types | Still `Option<String>` on chunks/filters; enum exists in `source_type.rs` |
| Magic-values audit: `Importance` / `Scope` newtypes | Clamp/reject empty at construction; not started |
| Magic-values audit: full `response_keys` migration | Partial wiring in MCP/API; remaining JSON builders still use string literals |
| Magic-values audit: ingest/ASR string constants | `lqm-ingest` payload keys / content-type / API path literals |
| Magic-values audit: FNV + binary port defaults | Named constants in hybrid.rs and CLI/MCP/API listen defaults |

---

## Explicit non-goals

Do not prioritize these for the headless agent-KB goal:

- Multi-user auth, API-key admin consoles, multi-tenancy
- AnythingLLM-style agent CRUD / `invoke_agent`
- Full product chat UI and workspace chat history
- Breadth-first SaaS connectors (Drive, Notion, …) — agents use other MCPs +
  `ingest_*` with `source` pointers
- Storing original media binaries inside lqm (paths/URLs as pointers only)

---

## How to use this doc

1. Implement the **next unchecked item** in the active sequence (or jump with a
   written reason in the PR).
2. Update AGENTS / ARCHITECTURE / DECISIONS / README as needed.
3. Drop the item from this roadmap when merged.
4. Keep the AnythingLLM gap map in sync if capability vs product changes.
