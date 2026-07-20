# Magic Values Audit — Implementation Plan

Audit findings: 45 issues across 5 crates. This plan sequences fixes from
most impactful (type-system changes) to least (named constants for isolated
magic numbers), with each change compile-verified before continuing.

## Phase 1: `lqm-core` type-system foundations

These changes touch the public types and propagate outward.

### 1.1 `Clearance` enum (`scope.rs`)

Replace `CLEARANCE_LEVELS: &[&str]` with `enum Clearance { Public, Internal,
Confidential, Restricted }` using `#[repr(u8)]` and `Ord` derived from variant
order. Keep all existing free functions as `impl Clearance` methods. Replace
`CLEARANCE_LEVELS`, `DEFAULT_CLEARANCE`, and the helper functions with methods
on the enum.

### 1.2 Use `Clearance` and `SourceType` on domain types (`types.rs`)

- `DocumentChunk::clearance: Option<String>` → `Option<Clearance>`
- `DocumentChunk::source_type: Option<String>` → `Option<SourceType>`
- `PayloadFilter::source_type` and `max_clearance` → use the enums
- `SearchFilter::source_type` and `max_clearance` → use the enums
- `SourceSummary::source_type` → use the enum

Update `build_point_payload` in `qdrant.rs` to convert enums to their canonical
strings for Qdrant storage.

### 1.3 `ErrorCode` enum (`api_error.rs`)

Define `ErrorCode` enum with `ValidationError`, `EmbedError`, `QdrantError`,
`IoError`, `InternalError` and `Display` impl. Use it in `error_code()`,
`StructuredError`, `http_status()`. Replace bare string literals throughout
`lqm-api` and `lqm-mcp` that currently match on these codes.

### 1.4 `EmbedderConfig.backend` → `EmbedderBackend` (`config.rs`)

Change `backend: String` to `backend: EmbedderBackend` with custom
`Deserialize` that parses on deserialization. Remove the re-parse in
`create_embedder`. Update `from_env()` to parse the env var into the enum.

### 1.5 `Importance` newtype (`types.rs`)

`Importance(f32)` that clamps to 0.0–1.0 on construction. `Display` impl
emits the string for Qdrant payloads. Apply to `DocumentChunk::importance`.

### 1.6 `Scope` newtype (`types.rs`)

`Scope(String)` with fallible construction rejecting empty/whitespace.
Prevents mixing scope strings with other string fields.

## Phase 2: Payload field name constants

### 2.1 Fix `qdrant.rs` filter builders

`payload_filter_to_qdrant` (lines 583-593) and `search_filter_to_qdrant`
(lines 619-639) use bare string literals `"source"`, `"source_type"`,
`"project"`, `"tags"`. Replace with `payload_schema::SOURCE` etc.

### 2.2 Fix `qdrant.rs` scored point extraction

Line 693: `sp.payload.get("text")` → `sp.payload.get(payload_schema::TEXT)`.

### 2.3 Fix `context.rs` payload field lookups

Lines 92-94: `payload_str(.., "source")`, `"source_type"`, `"project"` →
use `payload_schema::SOURCE`, etc.

## Phase 3: Shared JSON response keys

### 3.1 Create `response_keys` module in `lqm-core`

`pub mod response_keys` with constants for every JSON key repeated in MCP and
API tool responses: `STATUS`, `OK`, `CHUNKS`, `INSERTED`, `SKIPPED`,
`REPLACED`, `COLLECTION`, `HAS_MORE`, `NEXT_OFFSET`, `OFFSET`, `LIMIT`,
`RESULTS`, `SOURCES`, `FILES`, `FILE_RESULTS`, `PATH`, `ERROR`, `EXISTS`,
`EMBEDDER`, `NAME`, `POINTS_COUNT`, `VECTOR_SIZE`, `DISTANCE`, `SEGMENTS`,
`INDEXED_VECTORS_COUNT`.

### 3.2 Replace literals in `lqm-mcp` and `lqm-api`

Do a pass through both binaries replacing bare JSON key strings with the
constants from `lqm_core::response_keys`.

## Phase 4: DRY and named constants

### 4.1 Config env var names (`config.rs`)

Define `pub const` items for every `EMBEDDING_*` env var used in `from_env()`.
Replace the bare string literals.

### 4.2 Deduplicate default URLs in `from_env()`

Lines 161, 172 use inline `"http://localhost:11434"` and
`"https://api.openai.com/v1"` instead of calling `default_ollama_url()` and
`default_openai_url()`. Fix to call the functions.

### 4.3 Memory prefixes

Define `MEMORY_ID_PREFIX` and `MEMORY_SOURCE_PREFIX` constants. Add helper
functions `memory_source_uri(id: &str) -> String` and
`parse_memory_source(source: &str) -> Option<&str>`.

### 4.4 Default importance constant

Define `const DEFAULT_MEMORY_IMPORTANCE: f32 = 0.5;` in `constants.rs`.
Replace the two duplicate `unwrap_or(0.5)` occurrences in `memory.rs`.

### 4.5 Search/context limit defaults

Replace `unwrap_or(10)` and `unwrap_or(8)` in `lqm-mcp/main.rs` with
`constants::DEFAULT_SEARCH_LIMIT` and `constants::DEFAULT_CONTEXT_LIMIT`.

### 4.6 Embedding API paths

Define `const OLLAMA_EMBED_PATH: &str = "/api/embed";` and
`const OPENAI_EMBED_PATH: &str = "/embeddings";` in `embedding.rs`.

### 4.7 `list_available_backends` returns `Vec<EmbedderBackend>` not `Vec<&str>`

Currently returns `Vec<&'static str>`. Return `Vec<EmbedderBackend>` instead
and call `.as_str()` at the display site.

### 4.8 FNV hash constants (`hybrid.rs`)

Name the `2_166_136_261` offset basis and `16_777_619` prime constants.

### 4.9 Server defaults in binaries

Name constants for default ports, hosts in `lqm-mcp`, `lqm-cli`, `lqm-api`.

### 4.10 `"Bearer "` prefix in `lqm-api`

Define constant, use case-insensitive matching instead of two `strip_prefix` calls.

## Phase 5: Ingestion crate cleanup

### 5.1 Payload field keys in `chunk_from_extract`

Replace `base.get("collection")`, `base.get("tags")`, etc. with
`payload_schema::*` constants.

### 5.2 ASR module string constants

Define constants for `"application/octet-stream"`, `"/v1/inference/"`, and
JSON response field names in `asr.rs`.

## Execution order

Build and test after each phase:

1. Phase 1.1 → `cargo build` to verify compile
2. Phase 1.2 → compile (propagates through all consumers)
3. Phase 1.3 → compile
4. Phase 1.4 → compile
5. Phase 1.5 → compile
6. Phase 1.6 → compile
7. Phase 2 → compile + run tests
8. Phase 3 → compile + run tests
9. Phases 4–5 → compile + run tests

Final verification: `cargo fmt && cargo clippy --workspace -- -D warnings &&
cargo test --workspace`

## Risks

- **Serde compatibility**: Changing `String` to enum/newtype on domain types
  that derive `Serialize/Deserialize` is the main risk. Custom serde impls
  or `#[serde(from = "String")]`/`#[serde(into = "String")]` patterns will
  keep JSON serialization format identical.
- **SourceType is already defined**: The enum exists in `source_type.rs` with
  `Display`/`FromStr` but isn't used on domain types. Migration is low-risk.
- **Qdrant payload format must not change**: The string representation stored
  in Qdrant payloads must remain identical. `Display` impls on the new types
  must produce the same strings.
