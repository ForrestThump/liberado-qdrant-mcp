# lqm-core

Reusable core library for the liberado-qdrant-mcp RAG system.

## Responsibility

Owns everything shared across binaries: types, the `Embedder` trait, structure-aware
chunking, Qdrant client interactions, payload schema, concurrency control
(semaphore), content hashing, hybrid fusion, memories, and scoped filtering.
Zero MCP/HTTP framework dependencies (structured error *codes* live here as pure
helpers; axum mapping stays in `lqm-api`).

## Module map

```
src/
├── lib.rs       — crate root, re-exports
├── types.rs     — DocumentChunk, SearchFilter/Options, payload_schema, INDEX_FIELDS,
│                  resolve_collection, unix_now_secs*, make_file_result, payload_str
├── error.rs     — LqmError
├── api_error.rs — stable error codes + suggested HTTP status (framework-free)
├── chunking.rs  — ChunkingStrategy, chunk_text / chunk_markdown / chunk_code / chunk_for_ingest
├── config.rs    — EmbedderConfig TOML/env, create_embedder()
├── context.rs   — format_relevant_context(_with), mmr_rerank
├── lifecycle.rs — decide_source_reingest (pure skip/replace/insert)
├── memory.rs    — MemoryNote/Hit, blend ranking, DEFAULT_MEMORY_COLLECTION
├── hybrid.rs    — keyword_score, RRF + weighted fuse (pure)
├── scope.rs     — scope match + clearance ranks
├── source_type.rs — SourceType enum (canonical as_str literals)
├── constants.rs — defaults (chunk size, search limits, SOURCE_TYPE_* mirrors)
├── embedding.rs — Embedder trait, Fake/FastEmbed/Ollama/OpenAI, parse_*_embeddings
└── qdrant.rs    — QdrantClient, RagCore (from_env, expand_to_chunks, search_page, lifecycle I/O)
```

## Key design decisions

- `RagCore::from_env` builds embedder + Qdrant + default config for all binaries
- `RagCore::expand_to_chunks` is the **only** structure-aware expansion path (MCP/API/CLI)
- `compute_ingest_hash()` SHA-256; re-ingest compares hash multisets per source
- `embed_and_upsert_batch` → `IngestReport { inserted, skipped, replaced, chunks }`
- `ensure_collection(name, Option<dim>)` — `None` uses active embedder dimension
- `ensure_indexes()` keyword indexes: source, source_type, ingest_hash, project, tags, scope, clearance  
  (no payload `collection` field — Qdrant collection name is the namespace)
- `search_page` + hybrid optional (`scroll_payloads` is O(n); see workspace ARCHITECTURE scaling)
- Memories: collection `memories`, `source_type=memory`, optional recency blend
- Scoped filtering: `scope` exact + `clearance` ordinal / `max_clearance`
- Embedders feature-gated: `embed-fastembed`, `embed-ollama`, `embed-openai`
- `DEFAULT_COLLECTION_NAME = "default"` via `resolve_collection`
