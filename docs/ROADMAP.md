# Roadmap

Living document. Move items from "Next" to "Shipped" as milestones are completed.

## Shipped

- **M0 — Scaffolding** (Cargo workspace, crate skeletons, docs, AGENTS.md)
- **M1 — Core MVP** (Embedder trait, FakeEmbedder, chunking, Qdrant wrapper, types, semaphore)
- **M2 — MCP binary (stdio)** (turbomcp server: ingest_text, search, list_collections; channel tests)
- **M3 — Dual mode** (stdio + streamable HTTP via `lqm-mcp serve`)
- **M4 — CLI + benchmarking** (ingest/list/delete/bench, file walker, mtime tracking)
- **M5 — Quality & idempotency** (SHA256 content hashing, auto payload indexes, `last_modified`)
- **M6 — More mediums** (PDF extractor behind `pdf` feature, audio framework, extension auto-detection)
- **M7 — Ollama + OpenAI embedders** (HTTP-based, feature-gated, TOML config + env vars, factory fn)
- **M8 — Web frontend** (axum HTTP server, REST API, dark-mode search UI)

## Shipped (post-M8 improvements)

- **Audit fixes** — removed leaked deps, collection-aware upsert, `from_config()` factory, .gitignore
- **Fastembed default** — config defaults to fastembed, graceful fallback when feature not compiled
- **Structured logging** — `log` + `env_logger` across all crates, request middleware in lqm-api
- **ingest_path MCP tool** — agents can ingest files/directories directly
- **ingest_hash computed** — idempotency hash now actually stored during upsert
- **PDF integration test** — minimal PDF fixture test for `pdf` feature
- **CI/CD pipeline** — GitHub Actions: fmt, clippy, test on push/PR

## Next

- Per-source-type chunking (markdown heading-aware, code function-aware)
- `collection_info` MCP tool
- Mock HTTP tests for Ollama/OpenAI embedders

## Backlog

- Dioxus web frontend (richer SPA)
- Audio transcription integration (whisper-rs)
- Clearance-safe / scoped filtering
- Batch-friendly `ingest_many` MCP tool
- WASM build of core for browser-side use
