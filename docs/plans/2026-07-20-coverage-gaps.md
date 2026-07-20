# Test Coverage Gap Analysis — 2026-07-20

**Source:** `cargo llvm-cov --package lqm-core --package lqm-ingest --package lqm-cli`
(67.66% total; lqm-mcp and lqm-api excluded due to ONNX/glibc 2.39 requirement)

**Focus:** Gaps that matter for the 5 active roadmap items. These are **offline-testable
pure functions** with zero or partial coverage — not the expected low coverage on
gRPC calls that require a live Qdrant.

---

## Critical Gaps — Must-Fix Before Roadmap Work

These are pure functions with **zero direct tests** that form the backbone of
correctness-critical paths.

### `qdrant.rs` — Protobuf construction (all upserts depend on these)

| Function | Line | Risk |
|----------|------|------|
| `make_dense_vector` | 211 | Used by every upsert. Pure `Vec<f32>` → protobuf. No test. |
| `make_point_vectors` | 220 | Branching sparse-vs-dense logic. Sparse present+nonempty → `NamedVectors`; sparse absent/empty → bare `Vector`. No test for any branch. |
| `qdrant_value_to_json` | 658 | All scroll/search payload deserialization depends on this. No test for `StringValue`, `ListValue`, or fallback paths. |
| `qdrant_payload_to_json` | 678 | All payload-to-JSON conversion. No test for empty iterator, single/multi keys. |
| `scored_point_to_search_result` | 691 | Every search result passes through this. No test for text extraction, score propagation, missing text. |
| `payload_filter_to_qdrant` | 578 | Used by `hashes_for_source`, `payloads_for_source`, `delete_by_filter`. Converts `PayloadFilter` → Qdrant `Filter`. No unit test. |
| `and_filter` | 1516 | Combines two Qdrant `Filter`s with AND. Used in hybrid keyword_index path. No test. |

### `qdrant.rs` — Security-critical filtering (scope + clearance)

| Function | Line | Risk |
|----------|------|------|
| `clearance_max_condition` | 540 | Builds clearance ceiling filter. No test for each `Clearance` variant's `allowed_levels()` count; no test for `None` return when no levels apply. |
| `push_scope_and_clearance` | 562 | Pushes scope + clearance into filter `must` vec. Empty scope filtering (`!s.is_empty()`) untested. Whitespace-only scope untested. Both-present vs one-present vs neither untested. |

### `qdrant.rs` — Payload construction edge cases

| Function | Line | Gap |
|----------|------|-----|
| `build_point_payload` | 1683 | Three existing tests cover basics, but untested: **timestamp** field (line 1722, never set), **last_modified** (line 1734, never set in assert), **scope** (line 1775, whitespace filtering), **explicit clearance** (line 1785, only default-to-public tested), **importance clamping** (line 1757, `clamp(0.0, 1.0)` never tested with out-of-range). |

---

## Roadmap-Critical Gaps — Per-Collection Chunk Config (Item #2)

When `chunk_size`/`chunk_overlap` become per-collection configurable, values
diverge from the global defaults (e.g., `chunk_size=256` for code repos vs
`2048` for prose). This hits untested paths that are never triggered with the
current fixed large defaults.

### `chunking.rs` — Oversized-section splitting (zero coverage)

| Path | Line | Risk |
|------|------|------|
| `chunk_markdown` → `chunk_text` fallback | 174–175 | When a markdown section exceeds `chunk_size`, it falls back to `chunk_text` (which may invoke `sliding_window_words`). **Completely untested.** Small per-collection `chunk_size` values will routinely trigger this. |
| `chunk_code` → `sliding_window_lines` fallback | 214 | Same — when a code block exceeds `chunk_size`, it falls into `sliding_window_lines`. **Completely untested.** |
| `sliding_window_words` oversized-word path | 266–268 | If a single word exceeds `chunk_size`, it becomes its own chunk. Small `chunk_size` values make this more likely. **Untested.** |
| `sliding_window_words` small-advance fallback | 275–276 | When overlap consumes most of the window, advance is forced to 1. **Untested.** |
| `sliding_window_lines` oversized-line path | 305–306 | Same as words. **Untested.** |
| `sliding_window_lines` overlap behavior | 311 | `SLIDING_WINDOW_LINE_OVERLAP_DIV=40` with `overlap=200` produces 5 lines of overlap. **Untested.** |

### `chunking.rs` — Dispatch and boundary detection

| Path | Line | Gap |
|------|------|-----|
| `chunk_kind_for` with `source_type` as canonical string | 97, 101 | `st == SOURCE_TYPE_MARKDOWN` and `st == SOURCE_TYPE_CODE` branches never hit (tests pass `source_type=Some("text")`). When per-collection config stores a `chunk_kind`, this path may become primary. |
| `chunk_for_ingest` Plain dispatch | 118 | Only Markdown and Code tested. Plain path (e.g., no extension, no source_type) **untested**. |
| `is_atx_heading` variants | 225–237 | H1, H6, H7+, tab separator, indented heading — all untested. Only `##` tested. |
| `is_code_boundary` comment bypass | 244 | `//` comment, `#` comment, `#!` shebang, `#[` attribute — untested. Only `fn` prefix tested. |
| `is_code_boundary` indented boundaries | 239 | `trim_start()` is exercised but all test inputs are unindented. Indented `fn`, `class`, etc. untested. |
| `chunk_kind_for` case insensitivity | 95 | `to_ascii_lowercase()` exercised but never with mixed-case input like `"MARKDOWN"`. |
| `chunk_kind_for` upper-case extensions | 91 | All test extensions are lower-case. `file.MD` or `main.RS` untested. |

---

## Moderate Gaps — Other Roadmap Items

### `qdrant.rs` — For items 3 (`get_collection_stats`), 4 (`get_similar_to_source`), 5 (`list_sources`)

| Function | Line | Gap |
|----------|------|-----|
| `search_page` pagination math | 1238–1254 | `has_more`, `next_offset`, truncation for dense-only results is pure math after Qdrant returns. Could be tested with mock `SearchResult` vectors. |
| `search_filter_to_qdrant` tags_should-only | 645 | Filter with only `tags_should` (no `must` conditions) — does the empty check still produce a valid filter? Untested. |
| `keyword_match` direct assertions | 525 | Only tested indirectly. No test verifying the protobuf `Field`/`Keyword` variant. |
| `search_page` limit=0 early return | 1196–1203 | Tested (line 1944). OK. |

### `memory.rs` — For future memory work (not in current roadmap, but nearby)

| Function | Line | Gap |
|----------|------|-----|
| `parse_importance_value` i64 branch | 36–38 | `json!(5)` never hits `as_i64()`. Untested. |
| `parse_importance_value` clamping | 37 | `clamp(0.0, 1.0)` never tested with out-of-range values. |
| `parse_unix_secs` standalone | 152–154 | Never tested directly. Only exercised inside `rank_memory_hits`. |
| `rank_memory_hits` numeric `last_accessed` | 215–216 | `v.as_u64()` and `v.as_i64()` branches untested. Only string-form timestamps tested. |
| `rank_memory_hits` empty results | 202 | `rank_memory_hits(&[], ...)` untested. |
| `rank_memory_hits` `use_recency=false` ordering | 268 | When `use_recency=false`, hits keep original `SearchResult` order (not re-sorted). This implicit guarantee is untested. |

### `source_type.rs`

| Function | Line | Gap |
|----------|------|-----|
| `UnknownSourceType::Display` | 79–93 | **Completely untested.** Error message formatting (suggestions list, `self.0`) has zero coverage. |

---

## Already Well-Tested (no action needed)

These modules/directories have excellent coverage and are safe to build on:

| Module | Coverage | Notes |
|--------|----------|-------|
| `scope.rs` | 100% | Clearance enum, scope helpers, parse — fully covered |
| `lifecycle.rs` | 100% | Reingest decision logic — fully covered |
| `reconstruction.rs` | 98.68% | `list_chunks`, `get_source`, `expand_context` — near-perfect |
| `context.rs` | 94.52% | LLM context formatting, MMR re-ranking — excellent |
| `embedding.rs` | 96.41% | Embedder trait, FakeEmbedder — near-perfect |
| `hybrid.rs` | 92.04% | Fusion helpers, tokenization, normalization — strong |
| `chunking.rs` (core paths) | 84.24% | `chunk_text`, `chunk_markdown` (non-oversized), `chunk_code` (non-oversized) — well-covered |

---

## Recommended Fix Order (by roadmap dependency)

These should be filled **before or alongside** the roadmap items, not deferred.

### Before Item 1 (Collection ↔ embedder hard guarantees)

| # | Test | What it covers | Why before |
|---|------|---------------|------------|
| 1 | `payload_filter_to_qdrant` unit tests | Empty, source-only, multi-tag, full filter | Item 1 requires per-collection filter queries; this is the foundation |
| 2 | `clearance_max_condition` unit tests | Each clearance variant; None return; empty levels | Critical for dim-mismatch validation of scoped collections |

### Before/with Item 2 (Per-collection chunk config)

| # | Test | What it covers | Why before |
|---|------|---------------|------------|
| 3 | `chunk_markdown` oversized section → chunk_text | `chunk_size=30`, long paragraph under a heading → multiple output chunks | Highest-risk gap for small per-collection chunk sizes |
| 4 | `chunk_code` oversized block → sliding_window_lines | `chunk_size=30`, long function body → multiple output chunks | Same — code repos with `chunk_size=256` will trigger this |
| 5 | `is_atx_heading` all variants | H1, H6, H7+, tab, indent | Refactoring `chunk_kind_for` will touch heading detection |
| 6 | `is_code_boundary` all prefixes + comments | All 52 boundary prefixes, `//`/`#` comment exclusion, indent | Same reason |
| 7 | `chunk_kind_for` source_type canonical path | `source_type=Some("markdown")`, `source_type=Some("code")` | Per-collection config stores `chunk_kind` — the source_type dispatch becomes primary |
| 8 | `build_point_payload` timestamp/scope/clearance | Non-None timestamp, explicit clearance levels, whitespace scope | Payload schema will carry chunk config policy metadata |

### Before/with Item 3 (get_collection_stats)

| # | Test | What it covers | Why before |
|---|------|---------------|------------|
| 9 | `search_filter_to_qdrant` tags_should-only | Filter with no must conditions | Stats uses filtered count_points — filter building is shared |

### Before/with Item 5 (list_sources performance)

| # | Test | What it covers | Why before |
|---|------|---------------|------------|
| 10 | `qdrant_value_to_json` + `qdrant_payload_to_json` | StringValue, ListValue, empty map, fallback | Phase A field masking changes what payload data flows through these |
| 11 | `scored_point_to_search_result` | Text extraction, missing text, score propagation | Phase B source-summary reads flow through this conversion |

### Can be done anytime (not blocking roadmap items)

| # | Test |
|---|------|
| 12 | `make_dense_vector` / `make_point_vectors` — protobuf shape assertions |
| 13 | `and_filter` — filter combination logic |
| 14 | `keyword_match` — protobuf field assertions |
| 15 | `push_scope_and_clearance` — edge cases |
| 16 | `UnknownSourceType::Display` — error message formatting |
| 17 | `rank_memory_hits` empty results / numeric timestamps / use_recency=false ordering |

---

## Summary

**7 critical pure functions with zero tests** in `qdrant.rs` form correctness
and security risks for the roadmap. **7 untested paths** in `chunking.rs` will
become active as per-collection chunk sizes diverge from global defaults.
Filling the top 8 gaps before or alongside roadmap implementation is the
recommended path — these are all fast, offline, deterministic unit tests with
no external dependencies.
