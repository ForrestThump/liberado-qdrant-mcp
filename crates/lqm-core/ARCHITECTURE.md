# lqm-core

Reusable core library for the liberado-qdrant-mcp RAG system.

## Responsibility

Owns everything shared across binaries: types, the `Embedder` trait, chunking
logic, Qdrant client interactions, payload schema, concurrency control
(semaphore), content hashing, and payload index management. Zero MCP/HTTP/CLI
dependencies.

## Module map

```
src/
├── lib.rs       — crate root, module declarations, re-exports
├── types.rs     — DocumentChunk, SearchResult, CollectionInfoSummary, RagConfig, INDEX_FIELDS
├── error.rs     — LqmError enum (Embed, Qdrant, Validation, Io, Other)
├── chunking.rs  — ChunkingStrategy, paragraph-aware sliding window chunk_text()
├── config.rs    — EmbedderConfig with TOML/env/load_or_default, create_embedder() factory
├── context.rs   — format_relevant_context / _with, mmr_rerank, char budgets
├── lifecycle.rs — decide_source_reingest (skip/replace/insert); pure, unit-tested
├── memory.rs    — MemoryNote/Hit, DEFAULT_MEMORY_COLLECTION, recency/importance blend ranking
├── hybrid.rs    — tokenize/keyword_score, RRF + weighted dense+keyword fuse (pure, offline-tested)
├── scope.rs     — scope match + clearance ranks (public…restricted); pure inclusion helpers
├── embedding.rs — Embedder trait, FakeEmbedder, FastEmbedder/OllamaEmbedder/OpenAIEmbedder (feature-gated)
└── qdrant.rs    — QdrantClient wrapper, RagCore orchestrator, search_page (hybrid optional), lifecycle I/O, store/recall_memory
```

## Key design decisions

- `RagCore` uses `Arc<Semaphore>` to limit concurrent embedding calls
- `compute_ingest_hash()` produces SHA256 hex; re-ingest compares hash multisets per source
- `embed_and_upsert_batch` returns `IngestReport` (inserted/skipped/replaced/chunks)
- `list_sources` / `delete_by_source` / `delete_by_filter` share scroll + filter builders
- `search_page` + `SearchFilter` power filtered retrieval with offset/`has_more` (fetch limit+1)
- `format_relevant_context_with` supports total char budget + optional MMR diversity
- `ensure_indexes()` creates keyword indexes on source, source_type, collection, ingest_hash, project, tags
- `create_collection` / `get_collection_info` / `delete_collection` are first-class on `RagCore` (create defaults vector dim from the active embedder)
- Embedders are feature-gated (`embed-fastembed`, `embed-ollama`, `embed-openai`)
- `DEFAULT_COLLECTION_NAME = "default"` — single constant used by all consumers
- Memories default to collection `memories` with `source_type=memory` and `source=memory://{id}`; store reuses skip/replace; recall post-processes with optional recency blend (generation stays host-side)
- Hybrid search (`SearchOptions.hybrid`) fuses dense hits with keyword overlap on payload text (+ bounded scroll candidates); dense-only remains default
- Scoped filtering: payload `scope` (exact) + `clearance` (ordinal); search/lifecycle `scope` + `max_clearance`; unscoped default
