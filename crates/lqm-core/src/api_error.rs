//! Machine-readable error codes shared by HTTP API (and MCP where useful).

use crate::error::LqmError;
use serde::Serialize;

/// Stable error code + human message for agent/HTTP clients.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct StructuredError {
    pub code: String,
    pub message: String,
}

impl StructuredError {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

/// Map an `LqmError` to a stable code string.
pub fn error_code(err: &LqmError) -> &'static str {
    match err {
        LqmError::Validation(_) => "validation_error",
        LqmError::Embed(_) => "embed_error",
        LqmError::Qdrant(_) | LqmError::QdrantClient(_) => "qdrant_error",
        LqmError::Io(_) => "io_error",
        LqmError::Other(_) => "internal_error",
    }
}

/// Suggested HTTP status (as u16) for API mapping.
pub fn http_status(err: &LqmError) -> u16 {
    match err {
        LqmError::Validation(_) => 400,
        LqmError::Io(_) => 400,
        _ => 500,
    }
}

/// Full structured error from any `LqmError`.
pub fn structured_error(err: &LqmError) -> StructuredError {
    StructuredError::new(error_code(err), err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_maps_to_400_and_code() {
        let e = LqmError::Validation("bad name".into());
        assert_eq!(error_code(&e), "validation_error");
        assert_eq!(http_status(&e), 400);
        let s = structured_error(&e);
        assert_eq!(s.code, "validation_error");
        assert!(s.message.contains("bad name"));
    }

    #[test]
    fn embed_maps_to_500() {
        let e = LqmError::Embed(crate::embedding::EmbedError::EmbeddingFailed("x".into()));
        assert_eq!(error_code(&e), "embed_error");
        assert_eq!(http_status(&e), 500);
    }
}
