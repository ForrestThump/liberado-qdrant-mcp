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
}

impl PayloadFilter {
    pub fn is_empty(&self) -> bool {
        self.source.is_none()
            && self.source_type.is_none()
            && self.project.is_none()
            && self.tags.as_ref().map(|t| t.is_empty()).unwrap_or(true)
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
            chunk_size: 2048,
            overlap: 200,
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
            qdrant_url: "http://localhost:6334".to_string(),
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

/// Keyword payload indexes created on new collections (filter + lifecycle paths).
pub const INDEX_FIELDS: &[&str] = &[
    "source",
    "source_type",
    "collection",
    "ingest_hash",
    "project",
    "tags",
];
