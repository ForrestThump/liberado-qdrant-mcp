pub mod chunking;
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
