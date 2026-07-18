# AGENTS.md

Guidance for both human developers and AI agents working on this repository.
Read this first.

## Project

**liberado-qdrant-mcp** — a lightweight RAG MCP server that talks directly to
Qdrant. Built in Rust as a Cargo workspace with loose coupling as a priority.

Shorthand: `lqm`.

## Quick Orientation

```
.github/
  workflows/ci.yml  ← CI workflow: fmt, clippy, test on push

docs/
  PLAN.md          ← overall design and rationale (read first)
  ARCHITECTURE.md  ← system-level crate graph, data flow, seams
  ROADMAP.md       ← forward-looking next work only (no shipped checklist)
  DECISIONS.md     ← why we chose what we chose
  AUDIT.md         ← audit findings and fix status
  AGENTS.md        ← MCP↔HTTP tool matrix for host agents
  plans/           ← executable implementation plans for /goal (acceptance + verification)

liberado-qdrant-mcp_vs_AnythingLLM_Analysis_and_Implementation_Roadmap.md
                   ← gap map vs AnythingLLM knowledge/MCP layer

crates/
  lqm-core/        ← core library: types, Embedder trait, Qdrant client,
                      chunking, concurrency (embed_semaphore). Zero MCP/HTTP.
  lqm-ingest/      ← source extractors: markdown, code, text → chunks.
  lqm-mcp/         ← MCP server binary (turbomcp). Tools exposed to agents.
  lqm-cli/         ← CLI binary: admin, bulk ingest, benchmarking.
  lqm-api/         ← HTTP server (axum) for web frontend.
```

Each crate also has its own `ARCHITECTURE.md`.

## Rules for Agents

1. **Read `docs/PLAN.md` before making major changes.**
2. **Prefer more, smaller crates over monolithic code.** Decompose when seams
   appear.
3. **Add tests with every feature.** Unit tests in the relevant crate.
   Integration tests use turbomcp's `channel` transport + `McpTestClient`.
4. **Keep docs in sync.** When you add a crate, create its `ARCHITECTURE.md`.
   When you ship a roadmap item: document it in README / AGENTS / ARCHITECTURE /
   DECISIONS as appropriate, then **remove it from** `docs/ROADMAP.md` (that
   file is forward-looking only — no shipped checklist).
5. **Record decisions.** If you make a choice with non-obvious tradeoffs, add an
   entry to `docs/DECISIONS.md`.
6. **Audit regularly.** Before landing a non-trivial PR, scan for:
   - Duplicated logic that should live in `lqm-core`.
   - Tight coupling that can be loosened with a trait or config.
   - Opportunities to split a crate.
7. **Naming:** project is `liberado-qdrant-mcp`. Crate/bin names use `lqm-*`.
   Do not use "myrag".
8. **Lint before committing:** `cargo fmt && cargo clippy -- -D warnings`.
   CI enforces this on every push.
9. **Don't hard-code.** Model name, dimension, chunk size, semaphore size,
   Qdrant URL — all configurable.
10. **Use logging.** Use `log::info!`/`log::warn!`/`log::debug!`/`log::error!`
    instead of `eprintln!` or `println!`. Set `RUST_LOG=info` to see output.

## Key Dependencies

- `turbomcp` v3.1.x (MCP SDK — `#[server]`, `#[tool]`, transport features)
- `fastembed` (v1 embedding; in-process, no external server, behind feature)
- `qdrant-client` (async)
- `tokio` (runtime)
- `log` / `env_logger` (structured logging)
- `clap` (CLI crate)
- `axum` + `tower-http` (API crate)
- Dioxus (future web frontend)
