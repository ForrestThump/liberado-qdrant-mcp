//! Canonical JSON response keys shared by MCP and HTTP API tool responses.
//!
//! Use these constants instead of bare string literals in JSON builders
//! to prevent drift and typos across the two transport surfaces.

pub const STATUS: &str = "status";
pub const OK: &str = "ok";
pub const ERROR: &str = "error";
pub const CHUNKS: &str = "chunks";
pub const INSERTED: &str = "inserted";
pub const SKIPPED: &str = "skipped";
pub const REPLACED: &str = "replaced";
pub const COLLECTION: &str = "collection";
pub const HAS_MORE: &str = "has_more";
pub const NEXT_OFFSET: &str = "next_offset";
pub const OFFSET: &str = "offset";
pub const LIMIT: &str = "limit";
pub const RESULTS: &str = "results";
pub const SOURCES: &str = "sources";
pub const FILES: &str = "files";
pub const FILE_RESULTS: &str = "file_results";
pub const PATH: &str = "path";
pub const EXISTS: &str = "exists";
pub const EMBEDDER: &str = "embedder";
pub const NAME: &str = "name";
pub const POINTS_COUNT: &str = "points_count";
pub const VECTOR_SIZE: &str = "vector_size";
pub const DISTANCE: &str = "distance";
pub const SEGMENTS_COUNT: &str = "segments_count";
pub const INDEXED_VECTORS_COUNT: &str = "indexed_vectors_count";
