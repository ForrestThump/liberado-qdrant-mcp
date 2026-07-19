//! Machine-readable error codes shared by HTTP API (and MCP where useful).

use crate::error::LqmError;
use serde::Serialize;

/// Stable, typed error code for agent/HTTP clients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(into = "String")]
pub enum ErrorCode {
    ValidationError,
    IoError,
    EmbedError,
    QdrantError,
    InternalError,
    Unauthorized,
    FetchError,
}

impl ErrorCode {
    pub const fn as_str(&self) -> &'static str {
        match self {
            ErrorCode::ValidationError => "validation_error",
            ErrorCode::IoError => "io_error",
            ErrorCode::EmbedError => "embed_error",
            ErrorCode::QdrantError => "qdrant_error",
            ErrorCode::InternalError => "internal_error",
            ErrorCode::Unauthorized => "unauthorized",
            ErrorCode::FetchError => "fetch_error",
        }
    }
}

impl std::fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<ErrorCode> for String {
    fn from(c: ErrorCode) -> Self {
        c.as_str().to_string()
    }
}

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

/// Map an `LqmError` to a stable `ErrorCode`.
pub fn error_code(err: &LqmError) -> ErrorCode {
    match err {
        LqmError::Validation(_) => ErrorCode::ValidationError,
        LqmError::Embed(_) => ErrorCode::EmbedError,
        LqmError::Qdrant(_) | LqmError::QdrantClient(_) => ErrorCode::QdrantError,
        LqmError::Io(_) => ErrorCode::IoError,
        LqmError::Other(_) => ErrorCode::InternalError,
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
    StructuredError::new(error_code(err).to_string(), err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_maps_to_400_and_code() {
        let e = LqmError::Validation("bad name".into());
        assert_eq!(error_code(&e), ErrorCode::ValidationError);
        assert_eq!(http_status(&e), 400);
        let s = structured_error(&e);
        assert_eq!(s.code, "validation_error");
        assert!(s.message.contains("bad name"));
    }

    #[test]
    fn embed_maps_to_500() {
        let e = LqmError::Embed(crate::embedding::EmbedError::EmbeddingFailed("x".into()));
        assert_eq!(error_code(&e), ErrorCode::EmbedError);
        assert_eq!(http_status(&e), 500);
    }
}
