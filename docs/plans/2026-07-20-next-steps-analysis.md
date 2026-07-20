# 2026-07-20 Next Steps Analysis

Branch: `analysis/next-steps` (off `develop`)

Four-part analysis: **feature richness**, **performance**, **agentic ergonomics**,
**feature parity with AnythingLLM + anythingllm-mcp-server**.

**Scoping constraints:** web frontend and agent MCP connectors are **out of scope**.
This project is an MCP server — not a chat UI, NotebookLM replacement, or agent
orchestrator. Multi-user auth, tenant ACLs, and chat/agent CRUD are non-goals.

---

## Current Landscape Summary

| Capability | Status |
|------------|--------|
| Collections (CRUD) | Shipped |
| Text/markdown/code/URL/PDF ingest | Shipped |
| Audio ASR (DeepInfra Whisper/Nemotron) | Shipped |
| Hybrid search (3 backends: keyword_index, sparse, scroll) | Shipped |
| MMR diversity re-ranking | Shipped |
| Source reconstruction (list/get/expand) | Shipped |
| Long-term agent memories (store/recall + recency blend) | Shipped |
| Scope/clearance payload isolation | Shipped |
| Idempotent ingest (SHA-256 skip/replace) | Shipped |
| Offset pagination on search + chunk listing | Shipped |
| LLM context formatting (`get_relevant_context`) | Shipped |
| MCP + HTTP surface parity | Shipped (audited and fixed) |
| Multiple embedder backends (fastembed, ollama, openai) | Shipped |
| CLI admin/bench | Shipped |

**Estimate:** ~80–85% coverage of the AnythingLLM headless knowledge path
(as stated in README).

---

## 1. Feature Richness

What's missing to make this a truly complete headless RAG server.

### P1 — Must-do (high leverage, foundations for other work)

#### 1.1 Per-collection chunk configuration
**Gap:** `ChunkConfig` is global. A codebase collection and an Obsidian vault
get the same `chunk_size`/`overlap`, but they shouldn't.

**Implementation:**
- Store `chunk_config` (size, overlap, kind) in collection-level metadata
  (Qdrant doesn't natively support collection metadata, so use a reserved
  `_lqm_config` collection with per-collection config points).
- `ensure_collection` writes config on create; `expand_to_chunks` reads it.
- MCP `create_collection` and HTTP `POST /api/collections` accept optional
  `chunk_size`, `chunk_overlap`, `chunk_kind` params.
- Backward-compatible: if no config exists, use global defaults.

**Leverage:** High. Agents ingesting code vs prose need different chunking.
This is the #1 missing feature for retrieval quality.

#### 1.2 Collection ↔ embedder hard guarantees
**Gap:** Nothing prevents searching collection A (dim 384) with embedder B
(dim 768). Qdrant rejects the gRPC call, but error messages are cryptic.

**Implementation:**
- On collection create, store `embedder_id` + `vector_dim` in the `_lqm_config`
  collection.
- `search_page` / `embed_and_upsert_batch` validate dim match before the
  expensive operation.
- Return a clear `LqmError::Validation("collection 'foo' uses dim 384 but
  embedder produces dim 768")` message.
- Offer a `model_label` field on `create_collection` for human-readable tracking.

**Leverage:** High. Prevents silent footguns when switching embedders.

#### 1.3 Tool: `get_similar_to_source`
**Gap:** Agents can search by query but not by "find documents similar to
document X". AnythingLLM has "similar documents" recommendations.

**Implementation:**
- MCP `get_similar_to_source(source, collection, limit, ...)`:
  1. Fetch all chunk texts for `source` via `get_source`.
  2. Embed the joined text.
  3. Search with that vector, excluding the source itself by filter.
- HTTP `POST /api/similar` with same semantics.

**Leverage:** Medium. Common agent UX: "what else relates to this doc I just
found?"

---

### P2 — Should-do (improves completeness)

#### 1.4 Additional source extractors
**Gap:** Text / md / code / URL / PDF / audio covered. Missing formats common in
knowledge work: docx, pptx, xlsx, csv, epub, plain HTML files (not URL fetch).

**Implementation (per format, feature-gated in `lqm-ingest`):**
- **docx** — `docx-rs` or `calamine` crate (no C deps)
- **csv** — `csv` crate (already pure Rust, minimal)
- **epub** — `epub` crate
- **xlsx** — `calamine` crate
- **pptx** — extract text nodes from ZIP XML (no dedicated crate needed)
- **html files** — reuse `html_to_text` from `url.rs` for local `.html` paths

**Each format is a feature gate + a new `Extractor` impl.** Start with csv and
docx (highest value per implementation cost).

**Leverage:** Medium. Fills the "document format" gap vs AnythingLLM.

#### 1.5 Richer `list_sources` previews
**Gap:** `list_sources` returns `{source, count, sample_tags, source_type}`.
No title, first-chunk preview, total chars, ingest timestamp.

**Implementation:**
- Add optional payload fields: `title`, `first_chunk_preview` (first 200 chars),
  `total_chars`, `ingest_timestamp`.
- `list_sources` aggregates min/max/sum during the scroll.
- Add `preview=true` param to control whether first-chunk preview is included
  (scroll cost: reading payloads anyway for counts; preview is no extra Qdrant
  work).

**Leverage:** Medium. Agents routinely need to preview sources before deciding
what to search.

#### 1.6 Document status tracking
**Gap:** No way to know if a source is fully ingested, partially ingested, or had
errors during last ingest. AnythingLLM tracks pending/processing/ready/failed.

**Implementation:**
- During ingest, write `ingest_status = "complete" | "partial"` and optional
  `ingest_error` to a `_lqm_sources` collection (or a lightweight payload field
  on every chunk — simpler but heavier).
- `list_sources` returns `status` and `error` fields.
- Errors from individual files in batch ingest are already reported in the
  response JSON — just need to persist them.

**Leverage:** Low–Medium. Useful for ops visibility but less critical for
agent-driven workflows.

---

### P3 — Nice-to-have

#### 1.7 Re-ingest / refresh tool
**Gap:** To update a source after the file changes, agents must `delete_by_source`
then `ingest_path` again. No single-tool refresh.

**Implementation:**
- `reingest_source(source, collection)` → delete + re-ingest atomically.
- Check `last_modified` on file sources to skip if unchanged.
- Return `{status: "updated" | "skipped" | "error"}`.

**Leverage:** Low. Two-tool workaround is fine for now; this is syntactic sugar.

#### 1.8 Cross-encoder re-ranking
**Gap:** MMR is diversity-based, not relevance-based. A cross-encoder
(e.g., `BAAI/bge-reranker-base`) would improve precision on `get_relevant_context`.

**Implementation:**
- Trait `Reranker` behind a feature gate (`rerank`).
- Integrate into `get_relevant_context` as a post-MMR pass.
- Default: off (costs latency). Opt-in via `rerank=true` param.

**Leverage:** Medium–High for retrieval quality, but adds model download/compute
burden. Defer until concrete agent quality complaints.

#### 1.9 Query rewriting / expansion
**Gap:** Short agent queries ("what's the network config?") don't match
technical prose well. Query expansion (synonyms, hypothetically-generated queries)
improves recall.

**Implementation:**
- Could be done agent-side (host already has an LLM). Doing it server-side adds
  latency and dependency on an LLM — against the headless design.
- **Recommendation:** document in AGENTS.md that hosts should expand queries
  before calling `search`. Not worth building server-side.

**Verdict:** Not for lqm. Agents should handle this.

---

## 2. Performance

### P1 — Must-do

#### 2.1 `list_sources` performance overhaul
**Gap:** `list_sources` scrolls **all** payloads in the collection, deserializes
every one to JSON, then aggregates in Rust. O(n) where n = total points. For
10k+ chunks, this is seconds.

**Implementation (option A — pragmatic, ships fast):**
- Use Qdrant's `scroll` with `with_payload = false` or a field mask for only
  `source` + `source_type` + `tags` — skip `text` and `embedding`.
- Still O(n) but drastically less data over the wire.

**Implementation (option B — proper, more work):**
- Create a separate `_lqm_sources` collection keyed by `source` pointer.
- On `embed_and_upsert_batch`, also upsert/update a source summary point.
- `list_sources` then reads from this small collection (O(sources), not O(points)).

**Recommendation:** Option A is low-effort and immediate. Option B is the right
long-term design and pairs naturally with doc status tracking (1.6).

**Leverage:** High. `list_sources` is called often by agents exploring what's
available. Current perf is the biggest scaling wart.

#### 2.2 Payload→JSON allocation optimization
**Gap:** `qdrant_payload_to_json` allocates a `serde_json::Map` + `Value` for
every search result. For 20 results, that's 20 allocations of ~500B each.

**Implementation:**
- Return `Payload` as `Arc<HashMap<String, qdrant::Value>>` from search/scroll.
- Lazily convert to JSON only when serialized (e.g., `Serialize` impl on
  `SearchResult` that streams to writer).
- Avoids double allocation in the hot path.

**Leverage:** Medium. Marginal improvement for typical agent page sizes (5–20
results), but adds up under concurrent load in HTTP mode.

#### 2.3 Chunking allocation reduction
**Gap:** `chunk_markdown` and `chunk_code` allocate `String` per heading/block
even when the block fits within `chunk_size` and isn't split.

**Implementation:**
- Use `&str` windows where possible.
- Copy only when a section is larger than `chunk_size` (rare for typical docs).
- Audit EF4 noted this; deferred as micro-optimization but adds up for large
  vault ingests.

**Leverage:** Low–Medium. Only meaningful for bulk ingestion of large repos/vaults.

---

### P2 — Should-do

#### 2.4 Embedding cache
**Gap:** Common queries ("network config", "homelab setup") are re-embedded on
every invocation. In stdio mode, the model is freshly loaded each time anyway;
in HTTP/serve mode, the model stays in RAM but query embeddings are recomputed.

**Implementation:**
- LRU cache (e.g., `moka` crate) on `embed_batch` calls, keyed by SHA-256 of
  the query text.
- Configurable TTL + max size.
- HTTP/serve mode only (stdio mode loses the cache on exit).

**Leverage:** Low. Most agent queries are novel. Only useful in high-throughput
HTTP mode with repetitive queries.

#### 2.5 Bulk ingest batch sizing
**Gap:** `ingest_many` collects all chunks from all texts/paths/URLs, then does
one `embed_and_upsert_batch` call. For 1000 files, this is one giant batch —
embedding model may OOM.

**Implementation:**
- Cap batch size at `MAX_BATCH_CHUNKS` (e.g., 256).
- Stream ingest: embed+upsert in chunks of N, report progress.
- Already partially handled by the embed semaphore limit, but explicit batching
  is safer.

**Leverage:** Medium. Prevents production footgun.

---

### P3 — Nice-to-have

#### 2.6 Qdrant gRPC connection pooling revisit
**Gap:** `QdrantClient` uses a single `Arc<QdrantGrpc>`. Recent `qdrant-client`
has optional connection pooling.

**Implementation:** Evaluate `qdrant-client >= 1.13` pooling and enable if
beneficial under concurrent search load.

**Leverage:** Low. Single-connection grpc is fine for homelab concurrency.

#### 2.7 Sparse vector quality: BM25 or learned sparse
**Gap:** Current sparse vectors use hash-TF (FNV-1a into fixed modulus). It's
a coarse bag-of-words, not true BM25 with IDF.

**Implementation:**
- BM25-Sparse (e.g., `Splade` style) via ONNX model behind `sparse-bm25` feature.
- Or: compute IDF from `count_points` per-token on the fly.
- ROADMAP already flagged this as "revisit if needed".

**Leverage:** Low. `keyword_index` backend is the default and already gives
reasonable keyword recall for most use cases.

---

## 3. Agentic Ergonomics

How well the MCP tool surface serves LLM agents.

### P1 — Must-do

#### 3.1 Tool: `get_collection_stats`
**Gap:** Agents can `get_collection_info` (points count, vector size, status)
but can't get distribution info: "how many source_type=code points?" or
"how many points tagged with 'networking'?"

**Implementation:**
- MCP `get_collection_stats(collection, group_by: Option<String>)`:
  - Without group_by: returns `{total_points, total_sources, source_types: {...}, tags: {...}}`
  - With group_by: returns counts keyed by payload field value.
- Uses `count_points` with repeated filters — cheap gRPC calls (no scroll).
- HTTP `GET /api/collections/{name}/stats?group_by=source_type`.

**Leverage:** High. This is the #1 ergonomic gap: agents need to know "what's in
here?" before formulating a search strategy.

#### 3.2 `get_relevant_context` hybrid + MMR interplay audit
**Gap:** `get_relevant_context` passes `hybrid` and `mmr` independently. If
both are true, hybrid fusion runs first, then MMR on the fused results. This is
correct but undocumented. Agents don't know the order of operations.

**Implementation:**
- Document the pipeline order in tool descriptions and AGENTS.md.
- Add a `pipeline_info` field in the response: `{hybrid, mmr, rerank: false}`.
- Test edge case: hybrid=false + mmr=true (current behavior is dense-only MMR,
  which is correct).

**Leverage:** Medium. Agents that combine flags need to trust the composition.

#### 3.3 Pagination ergonomics: `after` cursor
**Gap:** Current pagination uses `offset` + `has_more` + `next_offset`. Many
agents prefer cursor-based `after` tokens, especially when results can shift
between calls (adds/deletes). Current `offset` is fragile under mutation.

**Implementation:**
- Add optional `after` (string cursor) param to `search` and `get_relevant_context`.
- Cursor encodes `(offset, query_hash, timestamp)` so the server can detect
  drift.
- Backward-compatible: if `after` is passed, ignore `offset`. If neither,
  default offset=0.
- HTTP already has offset; add `after` as an alternative.

**Leverage:** Medium. Most agent loops are single-page; this matters for
"show me more" follow-up calls where data may have changed.

---

### P2 — Should-do

#### 3.4 Tool: `estimate_token_count`
**Gap:** Agents building context windows need to know how much text fits.
`get_relevant_context` has `max_total_chars`, but agents may want to budget
before calling.

**Implementation:**
- MCP `estimate_token_count(text, model?)` using a simple tokenizer (e.g.,
  `tiktoken-rs` behind feature).
- Cheap: no embedding, no Qdrant. Approximate counts are fine.
- Can also be used before ingest to warn about oversized chunks.

**Leverage:** Medium. Agents frequently ask "how big is this?" before including
in context.

#### 3.5 Better error messages for agent consumption
**Gap:** Current errors return `{code, message}`. For agent consumption,
structured suggestions would help: "collection not found — did you mean
'homelab-notes'? Available: [homelab, work-notes, memories]".

**Implementation:**
- `LqmError` variants carry optional `suggestions: Vec<String>`.
- When `collection_exists` check fails, suggest closest match (Levenshtein).
- When dim mismatch, suggest the correct dim.
- MCP and HTTP both serialize `suggestions` when present.

**Leverage:** Low–Medium. Nice-to-have for agent self-correction loops.

#### 3.6 Tool descriptions with usage guidance
**Gap:** Current `#[tool]` descriptions are factual but don't guide agents on
*when* to use each tool. The AGENTS.md has a "When to use which tool" table,
but agents reading tool descriptions don't see it.

**Implementation:**
- Add usage guidance to `#[tool(description = "...")]` macros:
  - `search`: "Use for scored results with metadata. Prefer get_relevant_context
    for LLM-ready passages."
  - `get_relevant_context`: "Use for pasting directly into LLM context. Returns
    markdown with citations."
  - `ingest_many`: "Prefer over multiple ingest_text calls. Batch all items in
    one call."
- AGENTS.md "When to use which tool" table stays as host-side reference.

**Leverage:** Medium. MCP hosts expose descriptions to the LLM directly.

---

### P3 — Nice-to-have

#### 3.7 Tool: `merge_collections`
**Gap:** No way to combine two collections (e.g., merge a staging collection into
production after review).

**Implementation:**
- `merge_collections(source, dest, strategy)` — scroll + re-upsert.
- Strategies: skip-duplicates (by source + ingest_hash), overwrite, append.
- Warn about dim mismatch.

**Leverage:** Low. Manual workaround via bulk scroll + reingest exists.

#### 3.8 Tool: `export_collection`
**Gap:** No way to export collection data for backup or migration.

**Implementation:**
- `export_collection(name, format)` where format is `jsonl` or `csv`.
- Scrolls chunks and writes to temp file, returns path.
- Agents can then read the file with other tools.

**Leverage:** Low. Backup via `qdrant-client` directly is fine for operators.

---

## 4. Feature Parity with AnythingLLM + anythingllm-mcp-server

Comparing lqm against:
- **AnythingLLM's knowledge/MCP layer:** document/workspace management, search,
  embedding, vector database admin
- **`anythingllm-mcp-server`** (raqueljezweb/andreperez): 39 tools wrapping the
  AnythingLLM API

### Out of scope (by project design)

These AnythingLLM features are explicitly not relevant to lqm:

| AnythingLLM Feature | Why out of scope |
|---------------------|------------------|
| Chat with workspace / thread management | This is a chat UI concern. lqm is headless. |
| Agent CRUD / `invoke_agent` | lqm is not an agent orchestrator. |
| User management / API key admin | Multi-user auth is a non-goal. |
| LLM provider configuration | lqm doesn't call LLMs (embedding only). |
| Vector DB provider switching | lqm is Qdrant-only by definition. |
| System settings / monitoring dashboard | Product shell concern. |
| Workspace settings / prompts | Chat UI concern. |

### Parity gaps for the knowledge/retrieval path

| AnythingLLM Feature | lqm Status | Action |
|---------------------|------------|--------|
| **Document upload (many formats)** | CSV/epub/xlsx/docx missing | P2 item 1.4 |
| **Document status (pending/ready/failed)** | Not tracked | P1 item 1.6 |
| **Similar documents** | Missing | P1 item 1.3 |
| **Custom chunking per workspace** | Global only | P1 item 1.1 |
| **URL processing** | `ingest_url` (shipped) | ✓ parity |
| **Webpage embedding** | `ingest_url` (shipped) | ✓ parity |
| **Workspace CRUD** | Collection CRUD (shipped) | ✓ parity |
| **Delete document** | `delete_by_source` (shipped) | ✓ parity |
| **List documents** | `list_sources` (shipped) | ✓ parity |
| **Vector search** | `search` (shipped) | ✓ parity |
| **Get document vectors** | `get_source` + `list_chunks` | ✓ parity (better: ordered reconstruction) |
| **Multiple embedder backends** | fastembed, ollama, openai | ✓ parity |
| **MCP server** | `lqm-mcp` (shipped) | ✓ parity |
| **HTTP API** | `lqm-api` (shipped) | ✓ parity |
| **PIN/importance on documents** | `importance` for memories only | Low priority — document pinning is a UI/product feature |
| **OCR on images** | Not supported | Defer — agents can use vision models via other MCPs |
| **Re-index / refresh** | Two-step workaround | P3 item 1.7 |
| **Cross-encoder re-ranking** | MMR only | P3 item 1.8 |
| **Streaming responses (SSE)** | Not supported | Defer — MCP/HTTP responses are small; streaming adds complexity for marginal gain |
| **Embed text directly** | `ingest_text` (shipped) | ✓ parity |
| **Batch embed** | `ingest_many` (shipped) | ✓ parity |
| **Search with multiple filters** | Full combinatorics (shipped) | ✓ parity (better: must/should/must_not tags) |
| **Clearance / scope isolation** | Shipped | ✓ parity (better: explicit clearance levels) |
| **Memory / long-term recall** | Shipped (store_memory/recall_memories) | ✓ parity (AnythingLLM doesn't have this) |

---

## Prioritized Next Steps (by leverage)

**Ranked by impact/effort ratio, respecting out-of-scope constraints.**

### High leverage (start here)

| # | Item | Dimension | Why first |
|---|------|-----------|-----------|
| 1 | **Per-collection chunk configuration** (1.1) | Feature Richness | Foundation for retrieval quality. Code vs prose need different chunking. |
| 2 | **Collection ↔ embedder hard guarantees** (1.2) | Feature Richness | Silent dim mismatch errors confuse agents. Defense in depth. |
| 3 | **`get_collection_stats` tool** (3.1) | Agentic Ergonomics | Agents need introspection before searching. #1 ergonomic gap. |
| 4 | **`list_sources` performance overhaul** (2.1) | Performance | O(n) scroll is the biggest perf wart. Affects every agent exploration. |

### Medium leverage

| # | Item | Dimension | Why |
|---|------|-----------|-----|
| 5 | **`get_similar_to_source` tool** (1.3) | Feature Richness + Parity | Common agent workflow; closes AnythingLLM gap. |
| 6 | **Richer `list_sources` previews** (1.5) | Feature Richness | Agents need to preview before searching. |
| 7 | **Document status tracking** (1.6) | Feature Richness + Parity | Ops visibility; pairs with `list_sources` rework. |
| 8 | **Pagination cursor (`after`)** (3.3) | Agentic Ergonomics | Robustness under concurrent mutation. |
| 9 | **Tool description usage guidance** (3.6) | Agentic Ergonomics | LLMs see tool descriptions; guide them to the right tool. |
| 10 | **Hybrid + MMR interplay docs** (3.2) | Agentic Ergonomics | Undocumented pipeline order erodes trust. |
| 11 | **Bulk ingest batch sizing** (2.5) | Performance | Prevents OOM on large `ingest_many` calls. |

### Lower leverage (do when needed)

| # | Item | Dimension | Why |
|---|------|-----------|-----|
| 12 | Additional extractors (csv, docx, epub, xlsx) (1.4) | Parity | Feature-gated; each format is small. |
| 13 | Re-ingest / refresh tool (1.7) | Parity | Syntactic sugar over delete+ingest. |
| 14 | `estimate_token_count` tool (3.4) | Agentic Ergonomics | Helpful but agents can estimate. |
| 15 | Better error messages with suggestions (3.5) | Agentic Ergonomics | Nice for self-correction loops. |
| 16 | Embedding cache (2.4) | Performance | Only useful in high-throughput HTTP mode. |
| 17 | Payload→JSON allocation optimization (2.2) | Performance | Marginal gain for typical page sizes. |
| 18 | Cross-encoder re-ranking (1.8) | Feature Richness | Defer until concrete quality complaints. |
| 19 | Merge/export collections (3.7, 3.8) | Ergonomics | Operational tools; low agent use. |
| 20 | BM25 / learned sparse vectors (2.7) | Performance | Keyword_index backend is adequate default. |

### Deferred / Not for lqm

| Item | Reason |
|------|--------|
| Query rewriting / expansion | Agent/host responsibility (they have the LLM). |
| OCR on images | Agents use vision models via other MCPs. |
| Streaming SSE responses | Marginal gain for small agent responses. |
| Full offline Qdrant double | `McpTestClient` + FakeEmbedder + lazy client works for CI. |
| Background re-index workers | Single-user homelab doesn't need persistent workers. |
| Dioxus SPA | Explicitly out of scope. |

---

## Summary

lqm is a mature v1 headless RAG MCP server. The next wave of work falls into
two clear themes:

1. **Collection-level configurability** — per-collection chunking, embedder
   guarantees, collection stats. These unlock better retrieval quality and
   agent self-service.

2. **Performance at scale** — `list_sources` is the bottleneck; batch ingest
   needs capping. These raise the ceiling before users hit walls.

AnythingLLM parity is already strong (~80–85%) on the knowledge/retrieval path.
The remaining gaps are document format breadth, similar-document recommendations,
and status tracking — all with clear, bounded implementations.

**Recommended first task:** 1.1 (per-collection chunk configuration), because it
touches the config layer, ingest pipeline, and search path, and establishes
patterns for all other collection-level features.
