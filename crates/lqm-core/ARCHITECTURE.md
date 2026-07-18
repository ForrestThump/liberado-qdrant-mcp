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
├── context.rs   — format_relevant_context() → markdown passages + structured sources
├── embedding.rs — Embedder trait, FakeEmbedder, FastEmbedder/OllamaEmbedder/OpenAIEmbedder (feature-gated)
└── qdrant.rs    — QdrantClient wrapper, RagCore orchestrator, compute_ingest_hash
```

## Key design decisions

- `RagCore` uses `Arc<Semaphore>` to limit concurrent embedding calls
- `compute_ingest_hash()` produces SHA256 hex for idempotent ingestion
- `ensure_indexes()` creates keyword indexes on `source`, `source_type`, `collection`, `ingest_hash`
- `create_collection` / `get_collection_info` / `delete_collection` are first-class on `RagCore` (create defaults vector dim from the active embedder)
- `format_relevant_context` is pure (no I/O) so MCP/API/CLI can share LLM-ready formatting
- Embedders are feature-gated (`embed-fastembed`, `embed-ollama`, `embed-openai`)
- `DEFAULT_COLLECTION_NAME = "default"` — single constant used by all consumers
