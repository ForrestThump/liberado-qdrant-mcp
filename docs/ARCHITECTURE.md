# Architecture

High-level system architecture for `liberado-qdrant-mcp`.

## Crate Dependency Graph

```
                         ┌─────────────┐
                         │  lqm-core   │  (lib — types, chunking, embedding, qdrant)
                         └──────┬──────┘
                     ┌──────────┼──────────┐
                     ▼          ▼          ▼
              ┌───────────┐ ┌────────┐ ┌─────────┐
              │ lqm-ingest│ │lqm-mcp │ │ lqm-cli │
              │  (lib)    │ │(binary)│ │(binary) │
              └───────────┘ └────────┘ └─────────┘
                                   │
                              turbomcp
                              (MCP SDK)
```

- `lqm-core` — zero side effects. Owns chunking, embedding orchestration
  (pluggable `Embedder` trait), Qdrant client wrapper, payload types, and
  concurrency control (semaphore). Everything else depends on this.
- `lqm-ingest` — source extractors (markdown, code, text, conversation logs
  now; PDF, audio later). Produces `DocumentChunk`s consumed by core.
- `lqm-mcp` — turbomcp server exposing tools. Stdio by default; HTTP optional.
- `lqm-cli` — admin, bulk ingest, embedder benchmarking.
- `lqm-api` (future) — HTTP server (axum) powering a Dioxus frontend.

## Data Flow

### Ingestion

```
file / dir → lqm-ingest extractor → Vec<DocumentChunk>
                                         │
                                         ▼
                                    lqm-core
                              ┌─────────────────┐
                              │ chunk if needed  │
                              │ embed_batch()    │
                              │ upsert Qdrant ◄──┼── embed_semaphore
                              └─────────────────┘
```

### Search

```
query text → lqm-core → embed(query)
                      → Qdrant.search(vector, filters)
                      → Vec<SearchResult>
```

Chunking and metadata are always applied the same way because both paths go
through `lqm-core`. The web UI and agents see consistent results.

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
| Frontend / API            | `lqm-api` (future)    | Same core powers both agent tools and web UI       |

## Design Principles

1. **Loose coupling.** Each crate has one clear responsibility. New embedders
   and source types are feature-gated additions, not rewrites.
2. **Core is pure.** `lqm-core` never spawns processes, opens sockets, or
   depends on MCP/HTTP frameworks.
3. **Batching by default.** Embedding is batch-oriented; MCP tools expose batch
   variants.
4. **Testable seams.** `Embedder` trait + turbomcp `channel` transport +
   `McpTestClient` make both core and MCP layers testable in-process.
5. **Config-driven, not hard-coded.** Model choice, chunking params, semaphore
   size, Qdrant addr are all configurable.
