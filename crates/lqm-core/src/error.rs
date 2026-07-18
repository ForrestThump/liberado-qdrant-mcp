//! Top-level `LqmError` enum used across core, MCP, CLI, and API surfaces.

use crate::embedding::EmbedError;
use crate::qdrant::QdrantError;
use qdrant_client::QdrantError as QdrantClientError;

#[derive(Debug, thiserror::Error)]
pub enum LqmError {
    #[error("embed error: {0}")]
    Embed(#[from] EmbedError),
    #[error("qdrant error: {0}")]
    Qdrant(#[from] QdrantError),
    #[error("qdrant client error: {0}")]
    QdrantClient(#[from] QdrantClientError),
    #[error("validation error: {0}")]
    Validation(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Other(String),
}
