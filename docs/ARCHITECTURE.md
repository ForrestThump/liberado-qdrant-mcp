# Architecture

High-level system architecture for `liberado-qdrant-mcp`.

## Crate Dependency Graph

```
                         ┌─────────────┐
                         │  lqm-core   │  (lib — types, chunking, embedding, RagCore)
                         └──────┬──────┘
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
       ┌───────────┐     ┌──────────┐      ┌─────────┐
       │ lqm-ingest│     │ lqm-mcp  │      │ lqm-cli │
       │  (lib)    │     │ (binary) │      │(binary) │
       └─────┬─────┘     └────┬─────┘      └─────────┘
             │                │ turbomcp
             └───────┬────────┘
                     ▼
              ┌──────────┐
              │ lqm-api  │  (axum binary; REST + static UI)
              └──────────┘
```

- `lqm-core` — no MCP/HTTP *frameworks*. Owns chunking, pluggable `Embedder`,
  Qdrant wrapper, payload types, hybrid/memory/scope, semaphore. Binaries
  depend on this.
- `lqm-ingest` — extractors (text, PDF feature, audio placeholder) + URL fetch
  (`fetch-url` feature). Returns raw text; core owns chunking.
- `lqm-mcp` — turbomcp tools. Stdio default; `serve` for streamable HTTP.
- `lqm-cli` — admin bulk ingest / list / delete / bench (same `expand_to_chunks`).
- `lqm-api` — axum REST parity with MCP + interim static search page.

## Data Flow

### Ingestion

```
file / dir / text / url
        │
        ▼
  lqm-ingest extract_text (if path/url)
        │
        ▼
  RagCore::expand_to_chunks  (structure-aware: md / code / paragraphs)
        │
        ▼
  embed_batch + embed_and_upsert_batch  (skip/replace by ingest_hash)
        │
        ▼
     Qdrant points
```

All MCP, HTTP, and CLI ingest paths call the same `expand_to_chunks` helper so
chunk boundaries stay consistent across surfaces.

### What is stored vs pointed at

Each Qdrant point is `{ dense vector, payload }`. The payload includes the
chunk **`text`** and metadata such as **`source`** (path/URL/id),
`source_type`, `chunk_index` / `total_chunks`, tags, project, scope, clearance.

| In Qdrant | Not in Qdrant |
|-----------|----------------|
| Chunk text (searchable / LLM-readable) | Original file/URL bytes |
| Embedding vector | A separate blob/object store |
| Provenance pointer (`source`) | Guaranteed live filesystem access |

Agents retrieve **text + citations**, not raw vectors alone. Opening the
original media (if still available) is the job of other host tools/MCPs using
`source` as a handle. PDF ingest stores extracted text only; audio currently
stores a filterable placeholder until transcription ships.

### Source reconstruction

```
collection + source
        │
        ▼
  scroll payloads (filter source=…)
        │
        ▼
  parse chunk_index (number or string) → sort (missing index last)
        │
        ├── list_chunks  → paginate (offset/limit/has_more)
        ├── get_source   → all chunks + joined text
        └── expand_context(center, ±N) → window of same-source chunks
```

Pure helpers live in `lqm_core::reconstruction`; MCP/HTTP are thin adapters.

### Search (dense)

```
query → embed(query) → Qdrant dense search (+ SearchFilter)
                     → Vec<SearchResult> / SearchPage
```

### Hybrid search

```
query → dense search (over-fetch)
      → keyword candidates (backend-selected; see below)
      → merge_and_fuse_hybrid (weighted + RRF)
      → page slice
```

Keyword backends (`LQM_HYBRID_KEYWORD_BACKEND`, default `keyword_index`):

| Backend | How candidates are found | Schema needs |
|---------|--------------------------|--------------|
| `keyword_index` | Scroll with `MatchTextAny` on payload `text` (full-text index) | Text index on `text` (auto on ensure) |
| `sparse` | Qdrant sparse ANN on named vector `sparse` | Collection created with sparse config; points store sparse at ingest |
| `scroll` | Full-collection payload scroll + client TF scoring (legacy O(n)) | None beyond existing payload |

Dense-only remains the default when `hybrid` is omitted/false. Sparse collections
that lack the sparse schema fall back to `keyword_index` then `scroll`.

### Memory recall

```
query → search_page(collection=memories, source_type=memory)
      → rank_memory_hits (optional recency/importance blend)
```

Chunking and metadata are always applied the same way because both paths go
through `lqm-core`. The web UI and agents see consistent results.

## Scaling boundaries

Homelab-friendly defaults; know these before large corpora:

| Operation | Complexity | Notes |
|-----------|------------|--------|
| Hybrid keyword (`keyword_index`) | ~O(matches) | Text-index-backed `MatchTextAny` scroll; candidate cap `KEYWORD_CANDIDATE_LIMIT`. Prefer for existing dense-only collections. |
| Hybrid keyword (`sparse`) | ~O(log n) sparse ANN | Named sparse vector `sparse` + TF encoding at ingest/query. Create collection with sparse schema (default when backend=`sparse`). Re-ingest needed if collection was dense-only. |
| Hybrid keyword (`scroll`) | O(n) points | Legacy full payload walk (`KEYWORD_SCROLL_PAGE`). A/B / small corpora only. |
| `list_sources` | O(n) payloads | Full payload scroll to aggregate source counts. |
| Embed concurrency | `num_cpus` semaphore | `embed_semaphore` default = CPU count; raise carefully under heavy concurrent ingest. |
| Payload→JSON conversion | O(keys × hits) | Shared `qdrant_value_to_json`; still allocates per hit — acceptable for typical agent page sizes. |

**A/B backends:** set `LQM_HYBRID_KEYWORD_BACKEND=sparse` or `keyword_index` (or
`scroll`) before starting MCP/API/CLI. Same `hybrid` / `hybrid_alpha` tool params.

## Concurrency Model

- `embed_semaphore` (configurable `Semaphore`) limits parallel embedding calls
  to avoid CPU thrash. Default size = `num_cpus::get()`.
- Searches bypass the semaphore — Qdrant handles high read concurrency natively.
- In stdio mode, concurrency is limited to however many agents spawn the binary
  simultaneously (usually 1–3). Model reloaded each time.
- In HTTP/serve mode, one model in RAM, all requests multiplexed via async tasks
  inside one process.

## Key Seams (for future extension)

| Seam                      | Location              | What it enables                                    |
|---------------------------|-----------------------|----------------------------------------------------|
| `Embedder` trait          | `lqm-core/embedding`  | Swap fastembed ↔ Ollama ↔ candle ↔ custom          |
| Source extractors         | `lqm-ingest`          | Markdown → PDF → audio transcription → anything    |
| Payload schema            | `lqm-core/types`      | Extensible metadata per source type                |
| Transport (turbomcp)      | `lqm-mcp`             | Stdio, HTTP, WebSocket, TCP, Unix, channel          |
| Filter policy             | `lqm-core/qdrant`     | Customizable search scoping                        |
| Frontend / API            | `lqm-api`             | REST parity with MCP; static UI interim            |

## Related docs

| Doc | Role |
|-----|------|
| [`docs/PLAN.md`](PLAN.md) | Design rationale |
| [`docs/ROADMAP.md`](ROADMAP.md) | Forward-looking next work only |
| [`docs/AGENTS.md`](AGENTS.md) | Tool matrix for hosts |
| [`docs/AUDIT.md`](AUDIT.md) | Maintainability audit + dispositions |
| [`../liberado-qdrant-mcp_vs_AnythingLLM_Analysis_and_Implementation_Roadmap.md`](../liberado-qdrant-mcp_vs_AnythingLLM_Analysis_and_Implementation_Roadmap.md) | Gaps vs AnythingLLM knowledge layer |

## Design Principles

1. **Loose coupling.** Each crate has one clear responsibility. New embedders
   and source types are feature-gated additions, not rewrites.
2. **Core avoids product frameworks.** `lqm-core` does not depend on MCP or
   HTTP frameworks (it does talk to Qdrant via `qdrant-client`).
3. **Batching by default.** Embedding is batch-oriented; MCP tools expose batch
   variants.
4. **Testable seams.** `Embedder` trait + live smokes; channel + `McpTestClient`
   planned for offline MCP integration tests.
5. **Config-driven, not hard-coded.** Model choice, chunking params, semaphore
   size, Qdrant addr are all configurable.
6. **Surface parity.** MCP, HTTP, and CLI share `expand_to_chunks` and the same
   payload schema so agents see consistent retrieval quality.
