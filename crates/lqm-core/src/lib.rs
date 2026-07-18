pub mod api_error;
pub mod chunking;

pub use chunking::{
    ChunkKind, ChunkingStrategy, chunk_code, chunk_for_ingest, chunk_kind_for, chunk_markdown,
    chunk_text,
};
pub mod config;
pub mod context;
pub mod embedding;
pub mod error;
pub mod hybrid;
pub mod lifecycle;
pub mod memory;
pub mod qdrant;
pub mod scope;
pub mod types;

pub use api_error::{StructuredError, error_code, http_status, structured_error};
pub use context::{
    ContextSource, FormattedContext, format_relevant_context, format_relevant_context_with,
    mmr_rerank,
};
pub use hybrid::{
    DEFAULT_HYBRID_ALPHA, DEFAULT_RRF_K, fuse_dense_keyword, hybrid_dense_fetch_limit,
    keyword_score, merge_and_fuse_hybrid, tokenize_for_keyword,
};
pub use lifecycle::decide_source_reingest;
pub use memory::{
    DEFAULT_MEMORY_COLLECTION, MEMORY_SOURCE_TYPE, MemoryHit, MemoryNote, blend_memory_score,
    memory_note_to_chunk, parse_importance_value, rank_memory_hits,
};
pub use qdrant::QdrantClient;
pub use qdrant::RagCore;
pub use qdrant::build_point_payload;
pub use scope::{
    CLEARANCE_LEVELS, DEFAULT_CLEARANCE, allowed_clearance_levels, clearance_allowed,
    clearance_rank, normalize_clearance, point_in_scope, scope_matches,
};
pub use types::{
    ContextOptions, DEFAULT_COLLECTION_NAME, EmbedderInfo, IngestReport, PayloadFilter,
    ReingestAction, SearchFilter, SearchOptions, SearchPage, SourceSummary, payload_schema,
};
