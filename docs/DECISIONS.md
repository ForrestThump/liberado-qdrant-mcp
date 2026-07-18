# Decision Log

Record reasoning for architectural choices as they are made. Update when
decisions are revisited or superseded.

## 001 — Direct MCP → Qdrant (not MCP → separate backend)

**Date:** 2026-07-13

**What:** The MCP server binary talks directly to Qdrant via the core library.
There is no intermediate backend process (HTTP/gRPC).

**Why:** Stdio MCP servers are spawned on-demand and exit after the call.
Adding a separate persistent backend process is an extra hop, extra latency,
extra memory footprint, and more moving parts for no benefit in a single-user
homelab context.

**Revisit if:** We later need persistent in-memory state, background
re-indexing queues, multi-user auth, or horizontal scaling.

---

## 002 — Multiple crates/binaries (not monolithic)

**Date:** 2026-07-13

**What:** A Cargo workspace with separate crates for core, ingestion, MCP, CLI,
and API. Multiple crate-specific `ARCHITECTURE.md` files.

**Why:** Enforces loose coupling and makes it trivial to add new embedders,
source types, or transports behind features. Matches the project guidelines
that decomposition is a priority.

---

## 003 — turbomcp as MCP SDK

**Date:** 2026-07-13

**What:** Use `turbomcp` (Epistates/turbomcp) v3.1.x for the MCP server
instead of `rmcp` or raw JSON-RPC.

**Why:**
- Zero-boilerplate macros (`#[server]`, `#[tool]`, `#[resource]`, `#[prompt]`)
  with compile-time JSON schema generation.
- Feature-gated transports (stdio, HTTP, WebSocket, TCP, Unix, channel) match
  our dual-mode plan.
- `channel` transport + `McpTestClient` enables fast in-process integration
  tests without spawning subprocesses.
- Active maintenance, extensive examples, 25-crate modular architecture.
- Project requirement.

---

## 004 — fastembed as v1 embedding backend

**Date:** 2026-07-13

**What:** Default embedding backend is the `fastembed` crate, behind the
`Embedder` trait.

**Why:**
- In-process — no external server, no extra runtime dependency.
- Multiple small/fast models suitable for homelab hardware.
- Models cached after first download.
- The `Embedder` trait keeps it swappable; Ollama + other backends can be added
  as optional features later.

---

## 006 — PDF extraction via pluggable extractor trait

**Date:** 2026-07-13

**What:** PDF support is added as an optional `pdf` feature in `lqm-ingest`
behind the `Extractor` trait. The `pdf-extract` crate provides one-call text
extraction with no C dependencies.

**Why:**
- The `Extractor` trait already supports adding new source types without core
  changes. PDF fits naturally — just another `supported_extensions()` entry.
- `pdf-extract` is pure Rust, no system deps, matches the "runs anywhere" goal.
- Behind a cargo feature so users who don't need PDF don't pay the compile cost.
- Audio files follow the same pattern: metadata-only stub with
  `source_type=audio_placeholder` until transcription ships.

---

## 007 — Ollama and OpenAI HTTP embedders

**Date:** 2026-07-13

**What:** Added `OllamaEmbedder` (`POST /api/embed`) and `OpenAIEmbedder`
(`POST /v1/embeddings`) behind `embed-ollama` and `embed-openai` cargo features.
Both are pure HTTP/REST clients with zero native dependencies.

**Why:**
- HTTP-based embedders work on any platform (no glibc/ONNX issues).
- Ollama is the most popular local LLM runtime — users already have it running.
- OpenAI-compatible API covers OpenAI, Azure, Together, LiteLLM, and any proxy.
- Behind the same `Embedder` trait — zero code changes in RagCore, MCP, or CLI.
- Config driven via a single TOML file (`[embedding]` section) with env var
  fallbacks. Factory fn `create_embedder(config)` picks the right backend.

---

## 008 — HTTP API server (axum)

**Date:** 2026-07-13

**What:** Added `lqm-api` crate — an axum HTTP server that reuses `lqm-core`
for all RAG logic, exposing REST endpoints for search, ingest, and collection
management. Includes a dark-mode static HTML search UI.

**Why:**
- Same `lqm-core` powers both agent tools (via MCP) and the web UI — no
  duplication of chunking, embedding, or Qdrant logic.
- Enables browsing and searching documents from a browser, not just agents.
- CORS configured for homelab use (all origins allowed).
- Static file serving with fallback to embedded HTML — works with zero config.
- Keeps the architecture clean: MCP for agents, HTTP for humans, same core.

---

## 009 — FakeEmbedder as default requires explicit opt-in

**Date:** 2026-07-13

**What:** `lqm-core` defaults `EMBEDDING_BACKEND` to `fastembed` in constants,
but the core crate’s default *features* are empty (no fastembed). MCP and API
enable `embed-fastembed` so production binaries get a real local model. Selecting
`fake` still logs a loud warning via `create_embedder`.

**Why:**
- Library consumers pay for fastembed only when they enable the feature.
- MCP/API must not silently store zero vectors — feature wiring is intentional
  (documented in `lqm-mcp/Cargo.toml`).

---

## 010 — GitHub Actions CI with fmt, clippy, test

**Date:** 2026-07-16

**What:** CI pipeline using GitHub Actions on `ubuntu-24.04`. Runs `cargo fmt
--check`, `cargo clippy --workspace -- -D warnings`, and `cargo test --workspace`
on every push to `main`/`session/*` and all PRs to `main`.

**Why:**
- Prevents formatting drift, clippy warnings, and test regressions from landing
  without notice.
- Single job with three sequential steps — simpler than multi-job fan-out for a
  project this size (2,600 lines), and avoids duplicate compilation.
- `ubuntu-24.04` selected because it ships glibc 2.39 — compatible with the
  ONNX runtime used by `fastembed` (the `embed-fastembed` feature). CI can
  actually build and test with fastembed enabled, unlike our sandbox containers
  which run glibc 2.35.
- Uses `dtolnay/rust-toolchain` for Rust (vs `actions-rs` which is deprecated)
  and `Swatinem/rust-cache` for fast incremental builds.
- System deps (`build-essential`, `libssl-dev`, `pkg-config`) installed
  explicitly so the workflow is self-contained.
## 011 — Memories as dedicated collection + host-side generation

**Date:** 2026-07-17

**What:** Long-term agent notes live in a default Qdrant collection `memories`
with `source_type=memory`, stable `source=memory://{memory_id}`, and payload
fields `importance`, `last_accessed`, `memory_id`. `store_memory` /
`recall_memories` exist on RagCore, MCP, and HTTP. Recall is semantic search
plus optional importance/recency re-rank; no server-side LLM generation.

**Why:**
- Separates preferences/facts from document chunks without a second database.
- Reuses ingest skip/replace-by-source for idempotent updates.
- Keeps lqm headless RAG: host agents generate answers; lqm only stores/recalls.

**Revisit if:** agents need multi-turn memory sessions, automatic last_accessed
touch-on-recall writes, or hybrid sparse+dense over memories only.

---
## 012 — Hybrid search via post-query dense+keyword fusion (not native sparse)

**Date:** 2026-07-18

**What:** Hybrid retrieval is an optional flag on `search_page` / MCP `search` /
HTTP `/api/search`. It over-fetches dense hits, optionally merges keyword-
matching scroll candidates, and fuses scores with weighted normalized dense+
keyword signals plus reciprocal rank fusion (RRF). No Qdrant sparse vector
schema or second index.

**Why:**
- Ships keyword rescue for rare tokens without collection migrations.
- Fusion helpers are pure and unit-tested offline (CI without Qdrant).
- Dense-only default keeps existing agents/tests stable.

**Revisit if:** collections grow large enough that scroll candidate scans hurt
latency, or agents need true sparse BM25 indexes in Qdrant.

---
## 013 — Scoped filtering via payload scope + clearance (not multi-user auth)

**Date:** 2026-07-18

**What:** Optional payload fields `scope` (exact partition) and `clearance`
(public < internal < confidential < restricted). Search and lifecycle filters
accept `scope` and `max_clearance`. Upsert defaults missing clearance to
`public`. Unscoped paths remain the default.

**Why:**
- ROADMAP "clearance-safe scoped filtering" without inventing user accounts.
- Keyword indexes on `scope`/`clearance` reuse existing Qdrant filter patterns.
- Pure helpers unit-test inclusion/exclusion offline.

**Revisit if:** true multi-tenant ACLs or host-issued capability tokens are
required.

---

## 014 — Shared expand_to_chunks for MCP / HTTP / CLI parity

**Date:** 2026-07-18

**What:** Structure-aware document expansion lives only on `RagCore::expand_to_chunks`.
MCP, HTTP `POST /api/ingest*`, and CLI `ingest` all call it before
`embed_and_upsert_batch`. Extractors still return raw text (not chunks).

**Why:**
- Audit AP3–AP5: duplicated expand helpers and raw single-chunk HTTP/CLI ingest
  produced different search quality depending on surface.
- One implementation keeps markdown/code/plain boundaries identical.
- Avoids coupling `Extractor` to `ChunkConfig` (OC5 deferred redesign).

**Revisit if:** extractors need stream/async chunk production with per-format
config that cannot live on RagCore.

---

## 015 — RagCore::from_env factory for binaries

**Date:** 2026-07-18

**What:** `RagCore::from_env(qdrant_url?, config_path?)` loads embedder config,
creates the embedder, connects Qdrant (with `list_collections` health check),
and applies `RagConfig::default()`.

**Why:**
- Eliminated triple-copy construction in MCP/CLI/API (audit DP1).
- Centralizes `QDRANT_URL` / default URL resolution.
- Explicit CLI/API flags still override URL when provided.

---
