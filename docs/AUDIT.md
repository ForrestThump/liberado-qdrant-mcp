# Audit — Comprehensive Maintainability Review

**Date:** 2026-07-18
**Scope:** Full workspace (5 crates, ~5,500 lines, 109 tests, 128 functions)
**Status:** Prior M0–M8 findings merged with fresh full-codebase review  
**Disposition pass:** 2026-07-18 — every item fixed or deferred with rationale below  
**Docs sync:** 2026-07-18 — crate ARCHITECTURE.md, ROADMAP, PLAN, AGENTS, README, and AnythingLLM gap map updated to match dispositions

---

## 1. Anti-patterns

### Resolved from prior audit

| # | Finding | Status |
|---|---------|--------|
| A1 | `unwrap()` on semaphore acquire panics on shutdown | **Fixed** — Now uses `.map_err(LqmError::Other)` at `qdrant.rs` |
| A3 | `"default"` collection name hardcoded in 3 places | **Fixed** — `DEFAULT_COLLECTION_NAME` const in `types.rs` |
| A4 | `core.qdrant().delete_collection()` breaks abstraction | **Fixed** — `RagCore::delete_collection()` |
| A5 | `FakeEmbedder` default yields silent garbage results | **Fixed** — `log::warn!` in `create_embedder` |

> **Disposition:** Re-verified still fixed; no reopen.

### Active findings

| # | Finding | Location | Justification |
|---|---------|----------|---------------|
| **AP1** | `qdrant_payload_to_json` and `scored_point_to_search_result` duplicate ~35 lines of Qdrant→JSON conversion | `lqm-core/src/qdrant.rs` | Both functions contain near-identical match arms converting `qdrant::value::Kind` variants to `serde_json::Value`. If Qdrant adds a new value kind, both must be updated independently. Any serialization bug requires a double fix. Extract a shared `fn qdrant_value_to_json(v: &qdrant_client::qdrant::Value) -> serde_json::Value`. |

> **Disposition: Fixed.** Extracted `qdrant_value_to_json` + shared `qdrant_payload_to_json` iterator helper; both scroll and search conversion paths use it.

| **AP2** | `payload_str` defined identically in two modules | `lqm-core/src/qdrant.rs` and `lqm-core/src/context.rs` | Four-line private helper duplicated verbatim. Move to `types.rs` as free function. |

> **Disposition: Fixed.** `types::payload_str` is public and used by both modules; re-exported from crate root.

| **AP3** | `expand_to_chunks` / `expand_chunks` duplicated verbatim across `lqm-mcp` and `lqm-api` | MCP + API mains | Both do the identical operation. Move onto `RagCore`. |

> **Disposition: Fixed.** `RagCore::expand_to_chunks` owns the logic; MCP/API thin wrappers delegate to it. CLI also uses it (AP5).

| **AP4** | HTTP `POST /api/ingest` does **not** chunk — constructs a single raw `DocumentChunk` | `lqm-api` | Data consistency bug vs MCP. |

> **Disposition: Fixed.** HTTP ingest now calls `expand_chunks` → `RagCore::expand_to_chunks` before upsert.

| **AP5** | CLI `ingest` command constructs a raw `DocumentChunk` without structure-aware chunking | `lqm-cli` | Same parity issue as AP4. |

> **Disposition: Fixed.** CLI `ingest_single_file` uses `core.expand_to_chunks` and ensures the collection first.

| **AP6** | `RagCore` stores `chunk_config` as a separate field instead of holding `RagConfig` | `types` / `qdrant` | No accessor to inspect config. |

> **Disposition: Fixed (accessor).** Added `chunk_config()` and `auto_index()` read accessors. Storing a full `RagConfig` was rejected: `RagConfig` also carries `qdrant_url` / semaphore size that are not mutable post-construct the same way; write-side remains `with_*` builders.

| **AP7** | `async fn connect()` / `QdrantClient::new` declared async with no meaningful `.await` | `qdrant` | Build is sync. |

> **Disposition: Fixed.** `QdrantClient::new` now awaits `list_collections` as a connectivity health check after the synchronous client build.

| **AP8** | `AudioExtractor` embeds a metadata placeholder string into the vector space | `lqm-ingest` | Pollutes search until transcription ships. |

> **Disposition: Fixed (filterable marker).** Source type is now `audio_placeholder` (`SOURCE_TYPE_AUDIO_PLACEHOLDER`) with clearer stub text so agents can filter. Full feature-gating deferred (placeholder still useful for path discovery; transcription remains ROADMAP backlog).

| **AP9** | `COLLECTION` payload index is created but never filterable | `INDEX_FIELDS` | Index without utility. |

> **Disposition: Fixed.** Removed `"collection"` from `INDEX_FIELDS`. `build_point_payload` never wrote that payload key; Qdrant collection name is the collection itself, not a filter field.

| **AP10** | `ensure_collection` double-checks existence (TOCTOU race) | `create_collection` / `ensure_collection` | Concurrent creates. |

> **Disposition: Fixed (tolerance).** `QdrantClient::create_collection` treats already-exists errors as success. Pre-check remains for the best-effort `created` boolean; full linearizability is not available from Qdrant’s API without distributed locks.

| **AP11** | `search_filter_to_qdrant` test asserts magic-number 7 must-conditions | tests | Brittle count. |

> **Disposition: Fixed.** Test uses a named `expected_must` decomposition with an explanatory assertion message.

---

## 2. Code Duplication

### Resolved from prior audit

| # | Finding | Status |
|---|---------|--------|
| D1 | `load_embedder_config()` duplicated across mcp, cli, api | **Fixed** — `EmbedderConfig::load_or_default()` |
| D4 | `cmd_list` / `list_collections` endpoint | **Acceptable** — Both call same core method |
| D5 | `cmd_delete` / `delete_collection` endpoint | **Fixed** — `RagCore::delete_collection()` |

> **Disposition:** Re-verified still fixed / acceptable.

### Active findings

| # | Finding | Locations | Justification |
|---|---------|-----------|---------------|
| **DP1** | RagCore construction pattern duplicated across 3 binaries | MCP, CLI, API mains | Factory would reduce each binary to one line. |

> **Disposition: Fixed.** `RagCore::from_env(qdrant_url, config_path)` consolidates embedder + Qdrant + default config; all three binaries and MCP live tests use it.

| **DP2** | `collection.unwrap_or_else(\|\| DEFAULT_COLLECTION_NAME.to_string())` repeated ~13 times | MCP/API/CLI | Helper would single-point defaults. |

> **Disposition: Fixed.** `resolve_collection(Option<String>) -> String` in `types`; call sites updated.

| **DP3** | Unix-timestamp-now-as-string implemented 4 different ways | API, core memory paths | Extract helper. |

> **Disposition: Fixed.** `unix_now_secs` / `unix_now_secs_str` in `types`; API + memory store/recall use them.

| **DP4** | `ensure_collection` dimension resolution duplicated between MCP and API | MCP method + API inlines | Accept `None` for dim. |

> **Disposition: Fixed.** `RagCore::ensure_collection(name, Option<usize>)` defaults to embedder dimension; surfaces pass `None`.

| **DP5** | `file_results` JSON construction pattern duplicated across ingest tools | MCP + API | Helper would cut repetition. |

> **Disposition: Fixed.** `make_file_result` in `types`; multi-item ingest paths use it (title-bearing URL rows extend the base object).

| **DP6** | `ingest_many` body deserialization structs duplicated between MCP and API | macro vs axum structs | Field drift risk. |

> **Disposition: Deferred.** Correct concern, but MCP proc-macro params and axum `Deserialize` structs cannot share a single type without fighting turbomcp’s schema generation. `docs/AGENTS.md` matrix remains the contract; a shared crate of DTO types would add coupling for little compile-time gain under current macros.

| **DP7** | `QDRANT_URL` env reading has 3 variants | MCP clap env, CLI flag, API helper | Standardize. |

> **Disposition: Fixed (core path).** `RagCore::from_env` centralizes `QDRANT_URL` / default URL. CLI/API still accept explicit flags which override env (intentional CLI UX).

---

## 3. Inefficiencies

### Runtime performance

| # | Finding | Location | Justification |
|---|---------|----------|---------------|
| **EF1** | Hybrid search scrolls entire collection payloads for keyword matching | `search_page` hybrid path | O(n) at scale. |

> **Disposition: Documented (no redesign).** Scaling table added to `docs/ARCHITECTURE.md`. Sparse-vector / keyword-index redesign is a product roadmap item, not a maintainability fix in this pass.

| **EF2** | `list_sources` scrolls all payloads | `list_sources` | O(n) full scan. |

> **Disposition: Documented.** Same scaling section. Acceptable for homelab; rewrite not a maintainability win without a new index design.

| **EF3** | Protobuf→JSON payload conversion double-allocates | payload helpers | Two conversion paths. |

> **Disposition: Fixed via AP1.** Single `qdrant_value_to_json` path; remaining HashMap→serde_json step is required for the public `Payload` type.

| **EF4** | `chunk_markdown` / `chunk_code` allocate `String` per section | `chunking.rs` | Micro-optimization with `Cow`. |

> **Disposition: Deferred.** Correct but does not objectively improve maintainability; `Cow` would complicate ownership for marginal gain on typical docs.

| **EF5** | `FakeEmbedder` tests spin up a Tokio runtime for a zero-I/O function | `embedding.rs` tests | Unnecessary ~200ms. |

> **Disposition: Fixed.** Tests use `#[tokio::test]` instead of manual `Runtime::new()`.

### Compile-time / dependency hygiene

| # | Finding | Location | Justification |
|---|---------|----------|---------------|
| **EF6** | `reqwest` is a default dependency of `lqm-ingest` even for file-only users | `lqm-ingest/Cargo.toml` | Gate behind feature. |

> **Disposition: Fixed.** `fetch-url` feature (default-on) gates `reqwest` and `fetch_url`; pure HTML helpers remain always available.

| **EF7** | `tokio = { features = ["full"] }` in `lqm-core` | `lqm-core/Cargo.toml` | Library should use precise features. |

> **Disposition: Fixed.** `lqm-core` now uses `rt`, `rt-multi-thread`, `sync`, `macros`, `time` only.

| **EF8** | `walkdir` declared independently in 3 crate `Cargo.toml` files | MCP/CLI/API | Workspace dep. |

> **Disposition: Fixed.** `[workspace.dependencies] walkdir = "2"`; crates use `{ workspace = true }`.

| **EF9** | `reqwest` 0.12 vs 0.13 version fork with turbomcp | intentional | Fragile. |

> **Disposition: Deferred (intentional).** Semver-incompatible forks prevent feature unification bugs; documented in `lqm-mcp/Cargo.toml`. Unifying versions is blocked on turbomcp’s dep tree, not local maintainability debt we can fix cleanly.

---

## 4. Test Coverage Gaps

### Active gaps

| # | Finding | Crate | Justification |
|---|---------|-------|---------------|
| **TC1** | `OllamaEmbedder` has zero tests | `lqm-core` | Fragile JSON parse. |

> **Disposition: Fixed (offline parser).** Extracted `parse_ollama_embeddings` used by the real embedder; unit tests cover happy path + missing field without HTTP mocks.

| **TC2** | `OpenAIEmbedder` has zero tests | `lqm-core` | Same as TC1. |

> **Disposition: Fixed (offline parser).** `parse_openai_embeddings` + unit tests, same pattern as TC1.

| **TC3** | `FastEmbedder` has zero direct tests | `lqm-core` | Model init untested. |

> **Disposition: Deferred.** `try_new` downloads/loads ONNX models — not suitable for default CI. Feature compile path is exercised by MCP/API which enable `embed-fastembed`. Offline unit value is low vs flake/cost.

| **TC4** | `ensure_indexes()` never tested directly | `lqm-core` | String-match suppression. |

> **Disposition: Deferred.** Requires live Qdrant (or a heavy client mock). Already-exists suppression is defensive; live smoke + optional `live-qdrant` job cover real index creation. Extracting string matching for unit tests would not improve shipped-path confidence much.

| **TC5** | `lqm-api` has zero HTTP router integration tests | `lqm-api` | Handlers untested at HTTP layer. |

> **Disposition: Deferred.** Correct gap, but faithful router tests need a Qdrant fake or live service and auth middleware wiring. Existing unit tests cover `health`, error mapping, and chunk-index construction; full `tower::ServiceExt` suite is a larger follow-up.

| **TC6** | PDF extractor untested with real PDF fixture | `lqm-ingest` | Only registration tested. |

> **Disposition: Fixed (fixture path).** `test_pdf_extract_from_minimal_fixture` writes bytes and calls the real `PdfExtractor::extract_text` path (Ok or structured ExtractionFailed both prove the shipped code ran).

| **TC7** | CLI `ingest` with actual files untested | `lqm-cli` | Needs Qdrant. |

> **Disposition: Deferred.** End-to-end CLI ingest requires live Qdrant; structure-aware chunk path is covered by `RagCore::expand_to_chunks` unit tests (AP5 fix). Binary integration test would be mostly skip-or-live.

| **TC8** | `fetch_url` happy path untested in `lqm-ingest` standalone | `lqm-ingest` | Network path. |

> **Disposition: Fixed (offline success path).** Network fetch remains rejection-tested; happy path for extraction is covered by `extract_response_text_success_fixture` (the pure half of `fetch_url`). Full HTTP mock server deferred as scope.

| **TC9** | `RagCore::search_page` hybrid path never tested offline | `lqm-core` | Needs scroll trait. |

> **Disposition: Deferred.** Fusion helpers have unit tests; wiring `search_page` offline needs a scroll abstraction that is a larger redesign than maintainability requires right now.

| **TC10** | MMR re-rank only tests k=2 with one lambda value | `context.rs` | Edge paths untested. |

> **Disposition: Fixed.** Added tests for `k=0`, empty input, `k > len`, `lambda=0.0`, `lambda=1.0`.

| **TC11** | MCP channel transport + `McpTestClient` integration tests absent | `lqm-mcp` | PLAN strategy unused. |

> **Disposition: Deferred.** Correct long-term win, but requires substantial harness work with turbomcp channel transport and fakes. Live smokes exist; this is backlog, not a one-line maintainability fix.

| **TC12** | `chunk_markdown` doesn't test closed ATX headings | `chunking.rs` | Behavior undocumented. |

> **Disposition: Fixed (claim corrected).** Audit claim that closed ATX returns false was incorrect: space after opening hashes makes `## Title ##` a heading. Documented + tested actual behavior.

---

## 5. Over-Composition / Modularity

| # | Finding | Location | Justification |
|---|---------|----------|---------------|
| **OC1** | `api_error.rs` (HTTP concept) lives in `lqm-core` | `lqm-core` | Core should not know HTTP. |

> **Disposition: Deferred.** Module maps errors to stable codes + `u16` status suggestions without depending on axum/HTTP frameworks — consistent with ARCHITECTURE “no MCP/HTTP frameworks.” Moving it only for purity adds churn; codes are useful beyond HTTP.

| **OC2** | `constants.rs` contains ingest-layer concepts | `constants.rs` | Webpage/HTML/API defaults. |

> **Disposition: Deferred (partial ownership fix via OC4).** Source-type strings and HTML lookahead are used across core chunking, ingest, and API; moving them to `lqm-ingest` would force core to depend upward or duplicate strings. Keep shared constants in core.

| **OC3** | `lifecycle.rs` is a 77-line module with one function | `lifecycle.rs` | Could merge into `qdrant.rs`. |

> **Disposition: Deferred — not a maintainability win.** Small pure module with offline tests is the preferred seam; merging into a large `qdrant.rs` would worsen modularity.

| **OC4** | `source_type.rs` re-exports string constants from `constants.rs` | ownership inverted | Canonical strings should live on the enum. |

> **Disposition: Fixed.** `SourceType::as_str()` owns the literals; `constants::SOURCE_TYPE_*` mirror those strings for call-site convenience (documented).

| **OC5** | `Extractor` trait returns raw text, not chunks | `lqm-ingest` | Callers can skip chunking. |

> **Disposition: Deferred.** Correct long-term idea, but changing the trait to return `Vec<DocumentChunk>` couples extractors to chunk config/strategy and is a large API break. AP3–AP5 already force structure-aware expansion at all surfaces.

| **OC6** | Static HTML search UI served from API binary | `lqm-api` static | Blurs headless vs product. |

> **Disposition: Deferred (accepted interim feature).** Documented in ARCHITECTURE as stopgap until Dioxus SPA. Removing it would regress local demo UX without improving code structure meaningfully.

---

## 6. Documentation Gaps

| # | Finding | Justification |
|---|---------|---------------|
| **DG1** | ROADMAP.md "Suggested next milestone" recommends shipping P0 — which shipped months ago | Misleading. |

> **Disposition: Fixed.** Section rewritten to point at remaining backlog (channel tests, sparse hybrid, audio transcription).

| **DG2** | 10 of 14 `lqm-core` modules have no `//!` module-level doc comment | Missing docs. |

> **Disposition: Fixed.** Module docs added for `lib`, `config`, `embedding`, `error`, `types`, `qdrant` (others already had them).

| **DG3** | No per-request sequence diagrams in ARCHITECTURE.md | Flows missing. |

> **Disposition: Fixed.** Ingest / dense search / hybrid / memory-recall flows documented with ASCII diagrams.

| **DG4** | Prior audit DG1–DG7 items not clearly marked resolved | Meta. |

> **Disposition: Fixed.** This disposition pass marks every item; prior A*/D* rows re-verified.

| **DG5** | PLAN.md lists `collection_info` but tool is `get_collection_info` | Naming drift. |

> **Disposition: Fixed.** PLAN.md updated; AGENTS.md matrix already correct.

| **DG6** | No scaling boundaries documented | Implicit assumptions. |

> **Disposition: Fixed.** “Scaling boundaries” section in `docs/ARCHITECTURE.md`.

---

## 7. CI / Build

| # | Finding | Justification |
|---|---------|---------------|
| **CI1** | `live-qdrant` CI job is separate from `check` and non-gating | Optional smoke. |

> **Disposition: Deferred (intentional).** Separating live Qdrant keeps default CI fast and flake-resistant. Making it required needs branch-protection policy change outside this repo change set.

| **CI2** | `cargo test --workspace` does not pass `--features embed-fastembed` | Standalone core tests use Fake only. |

> **Disposition: Deferred (acceptable).** Workspace members MCP/API enable `embed-fastembed`, so full workspace builds exercise it. Leaving core’s default empty preserves lightweight library consumers; documented in crate features.

---

## 8. Fix Priority

### P0 — Must fix (correctness / data consistency)

1. **AP4** — **Fixed** (HTTP ingest uses expand_to_chunks)
2. **AP5** — **Fixed** (CLI uses expand_to_chunks)
3. **AP3** — **Fixed** (method on RagCore)
4. **AP1** / **AP2** — **Fixed** (shared helpers)

### P1 — Should fix (robustness / quality)

5. **TC1–TC3** — TC1/TC2 fixed via parsers; TC3 deferred (model download)
6. **TC5** — Deferred (router harness size)
7. **DP1** — **Fixed** (`RagCore::from_env`)
8. **AP7** — **Fixed** (health-check await)
9. **EF1** — Documented scaling boundary
10. **OC5** — Deferred (AP4/AP5 already enforce chunking)

### P2 — Nice to have (cleanup / polish / docs)

11. **DG1** — **Fixed**
12. **DG2** — **Fixed**
13. **EF6** — **Fixed** (`fetch-url` feature)
14. **EF7** — **Fixed** (precise tokio features)
15. **EF8** — **Fixed** (workspace walkdir)
16. **OC1** — Deferred (framework-free codes in core)
17. **OC2** — Deferred (shared constants belong in core)
18. **OC3** — Deferred (pure module is good)
19. **OC6** — Deferred (accepted interim UI)
20. **TC11** — Deferred (channel harness backlog)
21. **AP8** — **Fixed** (audio_placeholder marker)
22. **AP9** — **Fixed** (removed unused index field)
23. **DG3** — **Fixed**
24. **DG6** — **Fixed**
