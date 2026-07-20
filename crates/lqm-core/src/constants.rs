//! Shared defaults and magic-number replacements for the entire workspace.
//!
//! Prefer these named consts over bare literals throughout the crate.

// ── Embedding defaults ──────────────────────────────────────────────
pub const DEFAULT_FAKE_DIM: usize = 384;
pub const DEFAULT_FASTEMBED_DIM: usize = 384;
pub const DEFAULT_OLLAMA_DIM: usize = 768;
pub const DEFAULT_OPENAI_DIM: usize = 1536;
pub const DEFAULT_FASTEMBED_MODEL: &str = "AllMiniLML6V2";
pub const DEFAULT_BACKEND: &str = "fastembed";

// ── Chunking ────────────────────────────────────────────────────────
pub const DEFAULT_CHUNK_SIZE: usize = 2048;
pub const DEFAULT_CHUNK_OVERLAP: usize = 200;
/// Overlap divisor for word-level sliding window (advance = overlap / DIV).
pub const SLIDING_WINDOW_WORD_OVERLAP_DIV: usize = 5;
/// Overlap divisor for line-level sliding window.
pub const SLIDING_WINDOW_LINE_OVERLAP_DIV: usize = 40;
pub const SLIDING_WINDOW_MIN_OVERLAP_LINES: usize = 1;

// ── Search defaults ─────────────────────────────────────────────────
pub const DEFAULT_SEARCH_LIMIT: u64 = 10;
pub const DEFAULT_CONTEXT_LIMIT: u64 = 8;
/// Extra hit fetched to detect `has_more` without a separate count call.
pub const HAS_MORE_EXTRA: u64 = 1;
pub const SCROLL_PAGE_SIZE: u32 = 256;
pub const KEYWORD_SCROLL_PAGE: u32 = 128;
/// Max chunks embedded in one batch call (prevents OOM on large ingests).
pub const MAX_EMBED_BATCH_CHUNKS: usize = 256;

// ── Hybrid retrieval ────────────────────────────────────────────────
/// Phrase / substring bonus added to keyword score when full query appears.
pub const KEYWORD_PHRASE_BONUS: f32 = 0.25;
/// Blend weight for weighted + RRF score fusion.
pub const HYBRID_FUSE_WEIGHTED: f32 = 0.5;
pub const HYBRID_FUSE_RRF: f32 = 0.5;
/// Named sparse vector used when `LQM_HYBRID_KEYWORD_BACKEND=sparse`.
pub const SPARSE_VECTOR_NAME: &str = "sparse";
/// Hash-space modulus for deterministic token → sparse index mapping.
pub const SPARSE_HASH_MODULUS: u32 = 2_000_003;
/// Cap non-zero sparse dimensions per document (after frequency merge).
pub const SPARSE_MAX_DIMS: usize = 256;
/// Max keyword candidates pulled from index/sparse paths before fusion.
pub const KEYWORD_CANDIDATE_LIMIT: u64 = 64;
/// Env var for hybrid keyword backend (`scroll` | `sparse` | `keyword_index`).
pub const ENV_HYBRID_KEYWORD_BACKEND: &str = "LQM_HYBRID_KEYWORD_BACKEND";

// ── Context formatting ──────────────────────────────────────────────
pub const DEFAULT_MMR_LAMBDA: f32 = 0.7;
pub const TEXT_PREVIEW_CHARS: usize = 160;

// ── HTML extraction (lqm-ingest) ────────────────────────────────────
pub const HTML_LOOKAHEAD_CHARS: usize = 256;

// ── Memory blend weights ────────────────────────────────────────────
pub const MEMORY_BLEND_SEM_WEIGHT: f32 = 0.75;
pub const MEMORY_BLEND_IMP_WEIGHT: f32 = 0.25;
pub const MEMORY_BLEND_RECENCY_SEM_WEIGHT: f32 = 0.60;
pub const MEMORY_BLEND_RECENCY_IMP_WEIGHT: f32 = 0.25;
pub const MEMORY_BLEND_RECENCY_REC_WEIGHT: f32 = 0.15;
/// Default half-life for recency decay (~7 days in seconds).
pub const MEMORY_RECENCY_HALF_LIFE: f32 = 7.0 * 86400.0;
/// Default importance for memories (0.0–1.0).
pub const DEFAULT_MEMORY_IMPORTANCE: f32 = 0.5;

// ── Memory IDs ───────────────────────────────────────────────────────
pub const MEMORY_ID_PREFIX: &str = "mem-";
pub const MEMORY_SOURCE_PREFIX: &str = "memory://";

// ── Default source strings (canonical values owned by `SourceType`) ─
// These re-export `SourceType::as_str()` so call sites can keep using constants
// without circular imports at const time (literal duplicates of the enum strings).
pub const SOURCE_TYPE_TEXT: &str = "text";
pub const SOURCE_TYPE_WEBPAGE: &str = "webpage";
pub const SOURCE_TYPE_URL: &str = "url";
pub const SOURCE_TYPE_PDF: &str = "pdf";
pub const SOURCE_TYPE_AUDIO: &str = "audio";
/// Placeholder audio (no transcription yet) — filterable distinct from real audio.
pub const SOURCE_TYPE_AUDIO_PLACEHOLDER: &str = "audio_placeholder";
pub const SOURCE_TYPE_MARKDOWN: &str = "markdown";
pub const SOURCE_TYPE_CODE: &str = "code";
pub const SOURCE_TYPE_MEMORY: &str = "memory";

/// Default source for single-shot API ingest.
pub const DEFAULT_API_INGEST_SOURCE: &str = "api-ingest";

/// Default Qdrant gRPC endpoint.
pub const DEFAULT_QDRANT_URL: &str = "http://localhost:6334";

// ── Env var names ────────────────────────────────────────────────────
pub const ENV_EMBEDDING_BACKEND: &str = "EMBEDDING_BACKEND";
pub const ENV_EMBEDDING_DIMENSION: &str = "EMBEDDING_DIMENSION";
pub const ENV_EMBEDDING_OLLAMA_MODEL: &str = "EMBEDDING_OLLAMA_MODEL";
pub const ENV_EMBEDDING_OLLAMA_URL: &str = "EMBEDDING_OLLAMA_URL";
pub const ENV_EMBEDDING_OLLAMA_DIMENSION: &str = "EMBEDDING_OLLAMA_DIMENSION";
pub const ENV_EMBEDDING_OPENAI_MODEL: &str = "EMBEDDING_OPENAI_MODEL";
pub const ENV_EMBEDDING_OPENAI_URL: &str = "EMBEDDING_OPENAI_URL";
pub const ENV_EMBEDDING_OPENAI_API_KEY: &str = "EMBEDDING_OPENAI_API_KEY";
pub const ENV_EMBEDDING_OPENAI_DIMENSION: &str = "EMBEDDING_OPENAI_DIMENSION";
pub const ENV_EMBEDDING_FASTEMBED_MODEL: &str = "EMBEDDING_FASTEMBED_MODEL";
