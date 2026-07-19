//! `lqm-core` — types, chunking, embedding, Qdrant client, and search helpers.
//!
//! Zero MCP/HTTP framework dependencies. Binaries (`lqm-mcp`, `lqm-cli`, `lqm-api`)
//! and `lqm-ingest` depend on this crate.

pub mod api_error;
pub mod chunking;
pub mod config;
pub mod constants;
pub mod context;
pub mod embedding;
pub mod error;
pub mod hybrid;
pub mod lifecycle;
pub mod memory;
pub mod qdrant;
pub mod reconstruction;
pub mod response_keys;
pub mod scope;
pub mod source_type;
pub mod types;

pub use api_error::{ErrorCode, StructuredError, error_code, http_status, structured_error};
pub use chunking::{
    ChunkKind, ChunkingStrategy, chunk_code, chunk_for_ingest, chunk_kind_for, chunk_markdown,
    chunk_text,
};
pub use constants::*;
pub use context::{
    ContextSource, FormattedContext, format_relevant_context, format_relevant_context_with,
    mmr_rerank,
};
pub use embedding::FakeEmbedder;
pub use hybrid::{
    DEFAULT_HYBRID_ALPHA, DEFAULT_RRF_K, HybridKeywordBackend, SparseEncoding, encode_sparse_tf,
    fuse_dense_keyword, hash_token, hybrid_dense_fetch_limit, hybrid_keyword_backend_from_env,
    keyword_candidates_from_payloads, keyword_score, merge_and_fuse_hybrid, text_index_query,
    tokenize_for_keyword,
};
pub use lifecycle::decide_source_reingest;
pub use memory::{
    DEFAULT_MEMORY_COLLECTION, MEMORY_SOURCE_TYPE, MemoryHit, MemoryNote, blend_memory_score,
    memory_note_to_chunk, parse_importance_value, rank_memory_hits,
};
pub use qdrant::QdrantClient;
pub use qdrant::RagCore;
pub use qdrant::build_point_payload;
pub use reconstruction::{
    DEFAULT_EXPAND_NEIGHBORS, DEFAULT_LIST_CHUNKS_LIMIT, SourceChunk, SourceChunkPage,
    SourceDocument, expand_chunk_neighbors, paginate_source_chunks, parse_chunk_index_value,
    sort_source_chunks, source_chunk_from_payload, source_document_from_chunks,
};
pub use scope::{
    Clearance, DEFAULT_CLEARANCE, UnknownClearance, clearance_allowed, normalize_clearance,
    point_in_scope, scope_matches,
};
pub use source_type::{SourceType, UnknownSourceType};
pub use types::{
    ContextOptions, DEFAULT_COLLECTION_NAME, EmbedderInfo, IngestReport, PayloadFilter,
    ReingestAction, SearchFilter, SearchOptions, SearchPage, SourceSummary, make_file_result,
    payload_schema, payload_str, resolve_collection, unix_now_secs, unix_now_secs_str,
};
