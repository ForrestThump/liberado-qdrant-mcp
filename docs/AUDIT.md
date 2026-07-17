# Audit — M0–M8 Post-Shipment Findings

## Anti-patterns

| # | Finding | Location | Fix |
|---|---------|----------|-----|
| A1 | `unwrap()` on semaphore acquire panics on shutdown | `lqm-core/embed_batch` | Return `LqmError` instead |
| A2 | `connect()` declared async with no `.await` — misleading | `lqm-core/qdrant` | Drop `async` or add connection validation |
| A3 | `"default"` collection name hardcoded in 3 places | mcp, cli, api | Extract `DEFAULT_COLLECTION` const in core |
| A4 | `core.qdrant().delete_collection()` breaks abstraction | cli | Add `RagCore::delete_collection()` |
| A5 | `FakeEmbedder` is default — silent garbage results | core/config | Warn loudly or refuse startup without config |

## Code Duplication

| # | Item | Copies | Fix |
|---|------|--------|-----|
| D1 | `load_embedder_config()` fn | mcp, cli, api (3 copies) | Move into `lqm-core::config` |
| D2 | RagCore construction pattern | mcp, cli, api (3 copies) | Add `RagCore::from_env()` factory |
| D3 | QDRANT_URL env reading | mcp, cli, api (3 variants) | Standardize via `RagConfig` |
| D4 | `cmd_list` / `list_collections` endpoint | cli, api | Both call same core method — acceptable |
| D5 | `cmd_delete` / `delete_collection` endpoint | cli, api | Add `RagCore::delete_collection()` |

## Test Coverage Gaps

| # | Gap | Crate | Severity |
|---|-----|-------|----------|
| T1 | `FastEmbedder` zero tests | core | Medium — feature-gated |
| T2 | `OllamaEmbedder` zero tests | core | Medium — needs mock HTTP |
| T3 | `OpenAIEmbedder` zero tests | core | Medium — needs mock HTTP |
| T4 | `ensure_indexes()` never tested | core | Low — needs Qdrant |
| T5 | `ingest_text` MCP tool never exercised | mcp | Medium — needs Qdrant |
| T6 | PDF extractor untested | ingest | Medium — needs fixture |
| T7 | CLI `ingest` with actual files untested | cli | Medium |
| T8 | `collection_info` planned but not implemented | mcp | Low — plan gap |

## Documentation Gaps

| # | Gap | Fix |
|---|-----|-----|
| DG1 | ROADMAP.md says "M0 not started" | Update to list all shipped M0–M8 |
| DG2 | ARCHITECTURE.md shows lqm-api as "(future)" | Mark as current |
| DG3 | DECISIONS.md missing M6, M7, M8 entries | Add entries for PDF, embedders, API |
| DG4 | Per-crate ARCHITECTURE.md not committed | Commit crate-level docs |
| DG5 | PLAN.md §7 lists unimplemented tools | Remove or annotate as backlog |
| DG6 | No `example.toml` config file | Add to repo root |
| DG7 | No API request/response docs | Add to lqm-api ARCHITECTURE.md |

## Fix Priority

### Must fix (correctness / architecture / de-duplication)
1. **D1** — Move `load_embedder_config()` into `lqm-core`
2. **A4** — Add `RagCore::delete_collection()` to fix abstraction leak
3. **A3** — Extract `DEFAULT_COLLECTION` const into `lqm-core`
4. **DG1** — Update ROADMAP.md

### Should fix (quality / robustness)
5. ✅ **A1** — Replace `unwrap()` on semaphore (fixed: `.map_err(LqmError::Other)`)
6. **A5** — Warn on FakeEmbedder default *(code fix documented below — requires crate rebuild)*
7. ✅ **DG3** — Update DECISIONS.md (entries 006–009 added)
8. ✅ **DG6** — Create `example.toml` config file (created in repo root)

### Nice to have (test coverage / polish)
9. **T1–T3** — Mock tests for HTTP embedders
10. **T6** — PDF fixture test
11. **DG4** — Commit crate ARCHITECTURE.md files
12. **DG5** — Clean up PLAN.md planned tools

---

## A5 Code Fix (documented — apply on next rebuild)

In `lqm-core/src/config.rs` (or wherever `create_embedder` is defined), add a
stderr warning when the `fake` backend is selected outside of test configuration:

```rust
pub fn create_embedder(
    config: &EmbedderConfig,
) -> Result<Box<dyn Embedder>, LqmError> {
    match config.backend.as_str() {
        "fake" => {
            eprintln!("WARNING: Using fake embedder (all-zero vectors).");
            eprintln!("  Configure a real backend via --config or EMBEDDING_BACKEND env var.");
            eprintln!("  Supported: fastembed, ollama, openai.");
            Ok(Box::new(FakeEmbedder::new(
                config.dimension.unwrap_or(384),
            )))
        }
        // ... other backends unchanged
    }
}
```

Tests using `FakeEmbedder` directly via the struct won't trigger this warning
since they construct `FakeEmbedder::new()` bypassing the factory.
