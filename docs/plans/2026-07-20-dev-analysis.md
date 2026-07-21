# Development Analysis — July 2026

Current state: 20 MCP tools, ~85% AnythingLLM knowledge-layer parity, 213 passing tests,
`_lqm_config` metadata collection shipped (roadmap #1).

## Areas Evaluated

### 1. Ingestion Expansion (Extractors)

**Current:** TextExtractor (30+ extensions), PdfExtractor (pdf), AudioExtractor (8 audio
formats + DeepInfra STT). URL fetch + HTML-to-text.

**Gaps vs AnythingLLM:** csv, docx, xlsx, pptx, epub, html files (local).

**Effort/reward matrix:**

| Format | Crate | Effort | Value |
|--------|-------|--------|-------|
| csv | `csv` (pure Rust) | ~50 lines | High — structured data is common in knowledge bases |
| docx | `calamine` or `docx-rs` (pure Rust) | ~80 lines | High — most business docs are docx |
| xlsx | `calamine` | ~50 lines (shared with docx) | Medium |
| html files | reuse `html_to_text` from url.rs | ~30 lines | Medium — same logic, different entry |
| epub | `epub` crate | ~60 lines | Low — niche format |
| pptx | ZIP XML extraction | ~100 lines | Low |

**Verdict:** csv + docx are the highest-value-per-effort. Adding both behind feature
gates is ~130 lines of new code with zero core changes. But "more formats" is additive
— it doesn't unlock new system capabilities.

### 2. Ingestion Hardening

**Current state:**
- `chunk_kind_for` is purely extension-based — a `.txt` file containing markdown gets
  paragraph-chunked instead of heading-split. No magic bytes, no content sniffing.
- `embed_and_upsert_batch` has no batch capping — 1000 files in `ingest_many` means
  one giant embed call that can OOM the embedding model.
- `IngestReport` is aggregate only (inserted/skipped/replaced/chunks). Per-file
  reporting is in the MCP tool response but not persisted.
- `ingest_path` takes only `path` + `collection` — no tags, project, scope, clearance
  overrides at the path level (you must re-ingest to add metadata).

**Verdict:** Content-type detection and batch capping are the highest-value hardening
items. Content-type detection directly improves chunk quality without agents needing
to know file types. Batch capping prevents a production footgun. Both are small scoped
changes (~50–80 lines each).

### 3. Indexing (Per-Collection Chunk Config — Roadmap #2)

**Current:** `ChunkConfig { chunk_size: 2048, overlap: 200 }` is global. A codebase
collection and an Obsidian vault get identical chunking.

**What it unlocks:**
- Code repos with `chunk_size=512` — smaller, tighter chunks for precise code search
- Prose vaults with `chunk_size=4096` — larger, context-rich chunks for narrative
- `chunk_kind` override — force markdown/code/plain regardless of extension
- Stored in `_lqm_config` (same pattern as embedder guarantees)

**Effort:** ~200 lines in core, ~30 lines per surface. Depends on `_lqm_config`
pattern already shipped.

**Verdict:** **Highest algorithmic leverage.** This is the single change that most
directly improves retrieval quality. Everything ingested after this gets better
chunk boundaries tuned to content type.

### 4. Agent Ergonomics

| Tool | Status | Impact |
|------|--------|--------|
| `get_collection_stats` | Roadmap #3 | High — agents can't answer "what's in here?" today |
| `get_similar_to_source` | Roadmap #4 | Medium — wraps existing get_source + search |
| `estimate_token_count` | Later/optional | Medium — context window budgeting |
| Cursor-based pagination | Later/optional | medium — robust under concurrent mutation |
| Tool description guidance | Later/optional | Low-medium — agents read descriptions |
| Better error suggestions | Later/optional | Low — Levenshtein collection name hints |

**Verdict:** `get_collection_stats` is the #1 ergonomic gap. It's pure
`count_points` gRPC — independent, ships fast, no dependency on other roadmap items.

### 5. Algorithm Improvements

| Improvement | Status | Impact |
|-------------|--------|--------|
| Cross-encoder re-ranking | Later/optional | Medium-high for precision, adds model dependency |
| Sparse vector quality (BM25) | Later/optional | Low — keyword_index default is adequate |
| Query expansion | Not for lqm | Agent-side (host has the LLM) |
| MMR improvements | Done — well-covered at 94.5% | — |

**Verdict:** Cross-encoder re-ranking would improve `get_relevant_context` precision
measurably but adds compute burden and a model download. Defer until retrieval quality
complaints surface. The algorithmic improvement that matters NOW is per-collection
chunk config.

### 6. Access (HTTP API)

**Current:** 18 REST endpoints — full parity with MCP tools. Auth via `LQM_API_TOKEN`.

**Gaps:** No streaming/SSE responses, no rate limiting, no metrics/telemetry,
no cursor-based pagination.

**Verdict:** Surface parity is already strong. These are nice-to-have but not
mission-critical for the headless agent goal.

---

## Recommended Sequence

### Push 1: Foundation (2 items, ~half day)

**1A. Per-collection chunk config** (roadmap #2)

The logical sequel to the `_lqm_config` pattern shipped in #1. Chunk config lives
in the same config collection. Ingest and search read per-collection settings.

**Why first:** Touches the full ingest pipeline, establishes the configuration
subsystem, unlocks per-source-type tuning. The test coverage for chunking is
already at 98.13% — safe to refactor.

**Effort:** ~260 lines. Core types, qdrant.rs (config read/write, expand_to_chunks
override), MCP + API create_collection params, live smoke tests.

**1B. `get_collection_stats`** (roadmap #3)

Independent of chunk config — pure `count_points` gRPC calls, no config collection
dependency. Ships fast alongside chunk config.

**Why in push 1:** Agents can verify per-collection chunk settings are reflected
in point distributions. Makes the config system observable.

**Effort:** ~140 lines. RagCore::collection_stats, MCP tool, HTTP endpoint.

### Push 2: Ingestion Quality (2 items, ~half day)

**2A. Content-type detection**

Add `looks_like_markdown` and `looks_like_code` heuristics to `chunk_kind_for`.
When extension is ambiguous or missing, inspect first 256 bytes of file content
(ATX headings → markdown, `fn `/`def `/`class ` → code). Fall back to
extension-based detection.

**Why here:** Per-collection chunk config tells the system HOW to chunk. Content-type
detection tells it WHAT to chunk. Together they make ingestion self-tuning.

**Effort:** ~60 lines in chunking.rs + ~30 lines in lqm-ingest.

**2B. Batch size capping**

Cap `embed_and_upsert_batch` at `MAX_BATCH_CHUNKS` (256). Stream large ingests
in chunks with progress logging. Prevents OOM on `ingest_many` with 1000+ files.

**Why here:** Hardening that becomes more important as ingestion handles more
formats and per-collection configs produce different chunk counts.

**Effort:** ~40 lines in qdrant.rs (embed_and_upsert_batch).

### Push 3: Format Expansion (2 items, ~half day)

**3A. CSV extractor**

`csv` crate is pure Rust. Extract all cell text joined per row, or column-aware
extraction. Feature-gated `csv` in lqm-ingest.

**Effort:** ~50 lines.

**3B. `get_similar_to_source`** (roadmap #4)

Wraps `get_source` + `search_page`. Independent, pure additive, closes AnythingLLM gap.

**Effort:** ~100 lines. RagCore::similar_to_source, MCP tool, HTTP endpoint.

---

## What NOT to do right now

- **Cross-encoder re-ranking** — adds model dependency, no concrete quality complaints yet
- **Background re-index workers** — single-user homelab doesn't need persistent workers
- **BM25/learned sparse** — keyword_index default is adequate, low urgency
- **Streaming/SSE** — agent responses are small, no demand
- **Export/backup tools** — manual Qdrant backup is fine for single-user
- **`estimate_token_count`** — agents can estimate on their own
- **list_sources perf** — not a bottleneck until collections exceed 10k+ points

---

## Summary

The highest-leverage next move is **push 1**: per-collection chunk config +
get_collection_stats. This pairs the biggest algorithmic win (tuned chunking) with
the biggest ergonomic gap (collection introspection), both building on the
`_lqm_config` infrastructure we just shipped. After that, content-type detection
and batch capping make ingestion more robust, and csv + get_similar_to_source
round out format coverage and AnythingLLM parity.
