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

pub const INDEX_FIELDS: &[&str] = &["source", "source_type", "collection", "ingest_hash"];
