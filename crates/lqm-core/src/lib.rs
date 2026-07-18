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

pub use context::{
    ContextSource, FormattedContext, format_relevant_context, format_relevant_context_with,
    mmr_rerank,
};
pub use lifecycle::decide_source_reingest;
pub use qdrant::QdrantClient;
pub use qdrant::RagCore;
pub use types::{
    ContextOptions, DEFAULT_COLLECTION_NAME, IngestReport, PayloadFilter, ReingestAction,
    SearchFilter, SearchOptions, SearchPage, SourceSummary,
};
