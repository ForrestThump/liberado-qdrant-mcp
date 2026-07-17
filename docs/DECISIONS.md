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
and (future) API. Multiple crate-specific `ARCHITECTURE.md` files.

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
- Audio files follow the same pattern (metadata-only placeholder, transcription
  behind a future feature).

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

**What:** The default embedding backend is `fake` (zero-value vectors). This is
safe for testing but produces garbage results in production. The `create_embedder`
factory emits a loud stderr warning when a `fake` backend is selected outside
of test configuration.

**Why:**
- Prevents silent failures where a user forgets to configure an embedder.
- Tests use `fake` intentionally — no warning needed in `#[cfg(test)]`.
- Production paths (MCP server, CLI, API) all trigger the warning unless a real
  backend is explicitly configured via TOML file or env vars.

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
