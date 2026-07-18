//! Shared domain types: chunks, search filters, payload schema, and helpers.

use serde::{Deserialize, Serialize};

pub type Payload = serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub text: String,
    pub source: Option<String>,
    pub source_type: Option<String>,
    pub collection: Option<String>,
    pub tags: Option<Vec<String>>,
    pub timestamp: Option<String>,
    pub project: Option<String>,
    pub last_modified: Option<String>,
    /// 0-based index within the parent document's chunk set (when known).
    pub chunk_index: Option<usize>,
    /// Total chunks produced for the parent document (when known).
    pub total_chunks: Option<usize>,
    /// Memory importance 0.0–1.0 (only for `source_type=memory`).
    pub importance: Option<f32>,
    /// Stable memory id (written as payload `memory_id`).
    pub memory_id: Option<String>,
    /// Isolation partition (e.g. team / tenant-like agent scope). Exact match on search.
    pub scope: Option<String>,
    /// Sensitivity level: public | internal | confidential | restricted.
    pub clearance: Option<String>,
}

impl DocumentChunk {
    /// Convenience constructor for a single unsplit blob (no chunk indices).
    pub fn simple(
        text: impl Into<String>,
        source: Option<String>,
        source_type: Option<String>,
        collection: Option<String>,
    ) -> Self {
        Self {
            text: text.into(),
            source,
            source_type,
            collection,
            tags: None,
            timestamp: None,
            project: None,
            last_modified: None,
            chunk_index: None,
            total_chunks: None,
            importance: None,
            memory_id: None,
            scope: None,
            clearance: None,
        }
    }
}

/// Documented Qdrant payload keys written by the shared ingest path.
///
/// Agents and HTTP clients should treat these names as stable.
pub mod payload_schema {
    pub const TEXT: &str = "text";
    pub const INGEST_HASH: &str = "ingest_hash";
    pub const SOURCE: &str = "source";
    pub const SOURCE_TYPE: &str = "source_type";
    pub const TAGS: &str = "tags";
    pub const TIMESTAMP: &str = "timestamp";
    pub const PROJECT: &str = "project";
    pub const LAST_MODIFIED: &str = "last_modified";
    pub const CHUNK_INDEX: &str = "chunk_index";
    pub const TOTAL_CHUNKS: &str = "total_chunks";
    pub const EMBEDDING_MODEL: &str = "embedding_model";
    pub const IMPORTANCE: &str = "importance";
    pub const LAST_ACCESSED: &str = "last_accessed";
    pub const MEMORY_ID: &str = "memory_id";
    pub const SCOPE: &str = "scope";
    pub const CLEARANCE: &str = "clearance";
}

/// Snapshot of the active embedder for agents and HTTP clients.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbedderInfo {
    pub id: String,
    pub dimension: usize,
    /// Backend-specific model name when known (e.g. `AllMiniLML6V2`, Ollama model).
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub text: String,
    pub score: f32,
    pub payload: Payload,
}

/// Serializable snapshot of a Qdrant collection for agents and tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionInfoSummary {
    pub name: String,
    pub points_count: Option<u64>,
    pub indexed_vectors_count: Option<u64>,
    pub segments_count: u64,
    pub status: String,
    pub vector_size: Option<u64>,
    pub distance: Option<String>,
}

/// Distinct document source within a collection (agent curation).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceSummary {
    pub source: String,
    pub count: u64,
    pub source_type: Option<String>,
    pub last_modified: Option<String>,
}

/// Payload filters for delete/list operations (AND of provided fields).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PayloadFilter {
    pub source: Option<String>,
    pub source_type: Option<String>,
    pub project: Option<String>,
    pub tags: Option<Vec<String>>,
    /// Exact scope partition match.
    pub scope: Option<String>,
    /// Max clearance level (admits this level and all lower).
    pub max_clearance: Option<String>,
}

impl PayloadFilter {
    pub fn is_empty(&self) -> bool {
        self.source.is_none()
            && self.source_type.is_none()
            && self.project.is_none()
            && self.tags.as_ref().map(|t| t.is_empty()).unwrap_or(true)
            && self
                .scope
                .as_ref()
                .map(|s| s.trim().is_empty())
                .unwrap_or(true)
            && self
                .max_clearance
                .as_ref()
                .map(|s| s.trim().is_empty())
                .unwrap_or(true)
    }

    pub fn for_source(source: impl Into<String>) -> Self {
        Self {
            source: Some(source.into()),
            ..Default::default()
        }
    }
}

/// Search-time payload filters shared by `search` and `get_relevant_context`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchFilter {
    pub source: Option<String>,
    pub source_type: Option<String>,
    pub project: Option<String>,
    /// Tags that must all match (AND → Qdrant `must`).
    pub tags: Option<Vec<String>>,
    /// Tags where any match (OR → Qdrant `should`).
    pub tags_should: Option<Vec<String>>,
    /// Tags that must not match (Qdrant `must_not`).
    pub tags_must_not: Option<Vec<String>>,
    /// Exact `scope` payload match — excludes other scopes when set.
    pub scope: Option<String>,
    /// Admit points with clearance at or below this level (`public`…`restricted`).
    pub max_clearance: Option<String>,
}

impl SearchFilter {
    pub fn is_empty(&self) -> bool {
        self.source.is_none()
            && self.source_type.is_none()
            && self.project.is_none()
            && self.tags.as_ref().map(|t| t.is_empty()).unwrap_or(true)
            && self
                .tags_should
                .as_ref()
                .map(|t| t.is_empty())
                .unwrap_or(true)
            && self
                .tags_must_not
                .as_ref()
                .map(|t| t.is_empty())
                .unwrap_or(true)
            && self
                .scope
                .as_ref()
                .map(|s| s.trim().is_empty())
                .unwrap_or(true)
            && self
                .max_clearance
                .as_ref()
                .map(|s| s.trim().is_empty())
                .unwrap_or(true)
    }
}

/// Options for paginated, filtered semantic search.
#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    pub collection: Option<String>,
    pub limit: Option<u64>,
    pub offset: Option<u64>,
    pub min_score: Option<f32>,
    pub filter: SearchFilter,
    /// When true, fuse dense scores with keyword overlap on payload text (hybrid retrieval).
    /// Dense-only remains the default (`false` / unset).
    pub hybrid: bool,
    /// Dense weight in hybrid fusion ∈ [0, 1]; keyword weight is `1 - alpha`.
    /// Default [`crate::hybrid::DEFAULT_HYBRID_ALPHA`] when hybrid is on and this is `None`.
    pub hybrid_alpha: Option<f32>,
}

/// One page of search hits with pagination metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchPage {
    pub results: Vec<SearchResult>,
    pub offset: u64,
    pub limit: u64,
    pub has_more: bool,
    pub next_offset: Option<u64>,
}

/// Options for LLM-ready context formatting.
#[derive(Debug, Clone, Default)]
pub struct ContextOptions {
    /// Truncate each passage body to this many chars (0 / None = no per-passage cap).
    pub max_chars_per_passage: Option<usize>,
    /// Stop adding passages once total markdown body would exceed this.
    pub max_total_chars: Option<usize>,
    /// If true, apply simple MMR diversity after score order (before budget).
    pub mmr: bool,
    /// MMR λ in [0, 1]: 1.0 = pure relevance, 0.0 = pure diversity. Default 0.7.
    pub mmr_lambda: Option<f32>,
}

/// Outcome of an ingest that may skip or replace by source.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct IngestReport {
    /// Source groups (or sourceless batches) newly written.
    pub inserted: usize,
    /// Source groups skipped because content hashes matched existing points.
    pub skipped: usize,
    /// Source groups where old points were deleted then rewritten.
    pub replaced: usize,
    /// Points actually upserted (`inserted` + `replaced` groups' chunk totals).
    pub chunks: usize,
}

/// Pure decision for re-ingest of one source's chunk set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReingestAction {
    /// No existing points for this source — write new points.
    Insert,
    /// Existing points have the same multiset of content hashes — skip write.
    Skip,
    /// Existing points differ — delete source points, then write.
    Replace,
}

#[derive(Debug, Clone)]
pub struct CollectionConfig {
    pub name: String,
    pub vector_dim: usize,
}

#[derive(Debug, Clone)]
pub struct ChunkConfig {
    pub chunk_size: usize,
    pub overlap: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: crate::constants::DEFAULT_CHUNK_SIZE,
            overlap: crate::constants::DEFAULT_CHUNK_OVERLAP,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RagConfig {
    pub qdrant_url: String,
    pub embed_semaphore_permits: usize,
    pub chunk: ChunkConfig,
    pub auto_index: bool,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            qdrant_url: crate::constants::DEFAULT_QDRANT_URL.to_string(),
            embed_semaphore_permits: num_cpus::get(),
            chunk: ChunkConfig::default(),
            auto_index: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct UpsertPoint {
    pub id: String,
    pub vector: Vec<f32>,
    pub payload: Payload,
}

pub const DEFAULT_COLLECTION_NAME: &str = "default";

/// Resolve an optional collection name to the workspace default.
pub fn resolve_collection(collection: Option<String>) -> String {
    collection.unwrap_or_else(|| DEFAULT_COLLECTION_NAME.to_string())
}

/// Current unix time in seconds (0 if the clock is before the epoch).
pub fn unix_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Current unix time in seconds as a decimal string.
pub fn unix_now_secs_str() -> String {
    unix_now_secs().to_string()
}

/// Read a string field from a JSON payload object.
pub fn payload_str(payload: &serde_json::Value, key: &str) -> Option<String> {
    payload
        .get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Per-item result object used by multi-file ingest tools (MCP + HTTP).
pub fn make_file_result(
    path: &str,
    ok: bool,
    error: Option<&str>,
    chunks: usize,
) -> serde_json::Value {
    serde_json::json!({
        "path": path,
        "ok": ok,
        "error": error,
        "chunks": chunks,
    })
}

/// Keyword payload indexes created on new collections (filter + lifecycle paths).
///
/// Note: Qdrant collection name is the collection itself — not a filterable payload key.
/// `build_point_payload` does not write a `collection` field.
pub const INDEX_FIELDS: &[&str] = &[
    "source",
    "source_type",
    "ingest_hash",
    "project",
    "tags",
    "scope",
    "clearance",
];
