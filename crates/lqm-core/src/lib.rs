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
pub mod lifecycle;
pub mod qdrant;
pub mod types;

pub use api_error::{StructuredError, error_code, http_status, structured_error};
pub use context::{
    ContextSource, FormattedContext, format_relevant_context, format_relevant_context_with,
    mmr_rerank,
};
pub use lifecycle::decide_source_reingest;
pub use qdrant::QdrantClient;
pub use qdrant::RagCore;
pub use qdrant::build_point_payload;
pub use types::{
    ContextOptions, DEFAULT_COLLECTION_NAME, EmbedderInfo, IngestReport, PayloadFilter,
    ReingestAction, SearchFilter, SearchOptions, SearchPage, SourceSummary, payload_schema,
};
