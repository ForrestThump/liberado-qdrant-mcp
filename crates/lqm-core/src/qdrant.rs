use crate::chunking::{ChunkingStrategy, chunk_text};
use crate::embedding::Embedder;
use crate::error::LqmError;
use crate::types::{
    ChunkConfig, CollectionInfoSummary, DocumentChunk, INDEX_FIELDS, SearchResult, UpsertPoint,
};
use qdrant_client::Qdrant as QdrantGrpc;
use qdrant_client::qdrant::{
    CollectionStatus, CreateCollection, CreateFieldIndexCollectionBuilder, DeleteCollection,
    DenseVector, Distance, FieldType, Filter, PointId, PointStruct, ScoredPoint, SearchPoints,
    UpsertPoints, Vector, VectorParams, Vectors, WithPayloadSelector, vector, vectors_config,
};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use tokio::sync::Semaphore;

#[derive(Debug, thiserror::Error)]
pub enum QdrantError {
    #[error("qdrant error: {0}")]
    Qdrant(String),
}

impl From<qdrant_client::QdrantError> for QdrantError {
    fn from(e: qdrant_client::QdrantError) -> Self {
        QdrantError::Qdrant(e.to_string())
    }
}

#[derive(Clone)]
pub struct QdrantClient {
    inner: Arc<QdrantGrpc>,
}

impl QdrantClient {
    pub async fn new(url: &str) -> Result<Self, QdrantError> {
        let qdrant = QdrantGrpc::from_url(url)
            .build()
            .map_err(|e| QdrantError::Qdrant(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(qdrant),
        })
    }

    pub async fn list_collections(&self) -> Result<Vec<String>, QdrantError> {
        let resp = self.inner.list_collections().await?;
        Ok(resp.collections.into_iter().map(|c| c.name).collect())
    }

    pub async fn collection_exists(&self, name: &str) -> Result<bool, QdrantError> {
        let collections = self.list_collections().await?;
        Ok(collections.iter().any(|c| c == name))
    }

    pub async fn create_collection(&self, name: &str, vector_dim: u64) -> Result<(), QdrantError> {
        let exists = self.collection_exists(name).await?;
        if exists {
            return Ok(());
        }
        self.inner
            .create_collection(CreateCollection {
                collection_name: name.to_string(),
                vectors_config: Some(qdrant_client::qdrant::VectorsConfig {
                    config: Some(qdrant_client::qdrant::vectors_config::Config::Params(
                        VectorParams {
                            size: vector_dim,
                            distance: Distance::Cosine.into(),
                            ..Default::default()
                        },
                    )),
                }),
                ..Default::default()
            })
            .await?;
        Ok(())
    }

    pub async fn delete_collection(&self, name: &str) -> Result<bool, QdrantError> {
        let exists = self.collection_exists(name).await?;
        if !exists {
            return Ok(false);
        }
        self.inner
            .delete_collection(DeleteCollection {
                collection_name: name.to_string(),
                timeout: None,
            })
            .await?;
        Ok(true)
    }

    /// Fetch collection stats and vector config for agent-facing inspection.
    pub async fn get_collection_info(
        &self,
        name: &str,
    ) -> Result<Option<CollectionInfoSummary>, QdrantError> {
        if !self.collection_exists(name).await? {
            return Ok(None);
        }
        let resp = self.inner.collection_info(name).await?;
        let info = resp
            .result
            .ok_or_else(|| QdrantError::Qdrant("collection info missing result".to_string()))?;

        let status = CollectionStatus::try_from(info.status)
            .map(|s| format!("{s:?}"))
            .unwrap_or_else(|_| format!("unknown({})", info.status));

        let (vector_size, distance) = info
            .config
            .as_ref()
            .and_then(|c| c.params.as_ref())
            .and_then(|p| p.vectors_config.as_ref())
            .and_then(|vc| vc.config.as_ref())
            .and_then(|cfg| match cfg {
                vectors_config::Config::Params(params) => {
                    let dist = Distance::try_from(params.distance)
                        .map(|d| format!("{d:?}"))
                        .unwrap_or_else(|_| format!("unknown({})", params.distance));
                    Some((Some(params.size), Some(dist)))
                }
                vectors_config::Config::ParamsMap(map) => {
                    // Single unnamed dense vector is the common case; take the first entry.
                    map.map.values().next().map(|params| {
                        let dist = Distance::try_from(params.distance)
                            .map(|d| format!("{d:?}"))
                            .unwrap_or_else(|_| format!("unknown({})", params.distance));
                        (Some(params.size), Some(dist))
                    })
                }
            })
            .unwrap_or((None, None));

        Ok(Some(CollectionInfoSummary {
            name: name.to_string(),
            points_count: info.points_count,
            indexed_vectors_count: info.indexed_vectors_count,
            segments_count: info.segments_count,
            status,
            vector_size,
            distance,
        }))
    }

    #[allow(deprecated)]
    fn make_dense_vector(data: Vec<f32>) -> Vector {
        Vector {
            data: vec![],
            indices: None,
            vectors_count: None,
            vector: Some(vector::Vector::Dense(DenseVector { data })),
        }
    }

    pub async fn upsert_points(
        &self,
        collection: &str,
        points: Vec<UpsertPoint>,
    ) -> Result<(), QdrantError> {
        use qdrant_client::qdrant::Value;
        use qdrant_client::qdrant::point_id::PointIdOptions;

        let qpoints: Vec<PointStruct> = points
            .into_iter()
            .map(|p| {
                let payload_map: std::collections::HashMap<String, Value> = p
                    .payload
                    .as_object()
                    .into_iter()
                    .flat_map(|map| {
                        map.iter().map(|(k, v)| {
                            let qv = match v {
                                serde_json::Value::String(s) => Value {
                                    kind: Some(qdrant_client::qdrant::value::Kind::StringValue(
                                        s.clone(),
                                    )),
                                },
                                serde_json::Value::Array(arr) => {
                                    let items: Vec<Value> = arr
                                        .iter()
                                        .filter_map(|item| {
                                            item.as_str().map(|s| Value {
                                                kind: Some(
                                                    qdrant_client::qdrant::value::Kind::StringValue(
                                                        s.to_string(),
                                                    ),
                                                ),
                                            })
                                        })
                                        .collect();
                                    Value {
                                        kind: Some(qdrant_client::qdrant::value::Kind::ListValue(
                                            qdrant_client::qdrant::ListValue { values: items },
                                        )),
                                    }
                                }
                                _ => Value {
                                    kind: Some(qdrant_client::qdrant::value::Kind::StringValue(
                                        v.to_string(),
                                    )),
                                },
                            };
                            (k.clone(), qv)
                        })
                    })
                    .collect();

                PointStruct {
                    id: Some(PointId {
                        point_id_options: Some(PointIdOptions::Uuid(p.id)),
                    }),
                    vectors: Some(Vectors {
                        vectors_options: Some(
                            qdrant_client::qdrant::vectors::VectorsOptions::Vector(
                                Self::make_dense_vector(p.vector),
                            ),
                        ),
                    }),
                    payload: payload_map,
                }
            })
            .collect();

        self.inner
            .upsert_points(UpsertPoints {
                collection_name: collection.to_string(),
                wait: Some(true),
                points: qpoints,
                ordering: None,
                shard_key_selector: None,
                update_filter: None,
                timeout: None,
                update_mode: None,
            })
            .await?;
        Ok(())
    }

    pub async fn search(
        &self,
        collection: &str,
        vector: Vec<f32>,
        limit: u64,
        filter: Option<Filter>,
        min_score: Option<f32>,
    ) -> Result<Vec<SearchResult>, QdrantError> {
        let search = SearchPoints {
            collection_name: collection.to_string(),
            vector: vector.clone(),
            limit,
            filter,
            with_payload: Some(WithPayloadSelector {
                selector_options: Some(
                    qdrant_client::qdrant::with_payload_selector::SelectorOptions::Enable(true),
                ),
            }),
            params: None,
            score_threshold: min_score,
            offset: None,
            vector_name: None,
            with_vectors: None,
            read_consistency: None,
            timeout: None,
            shard_key_selector: None,
            sparse_indices: None,
        };

        let resp = self.inner.search_points(search).await?;

        let results = resp
            .result
            .into_iter()
            .map(|p| scored_point_to_search_result(&p))
            .collect();

        Ok(results)
    }

    pub async fn create_field_index(
        &self,
        collection: &str,
        field_name: &str,
    ) -> Result<(), QdrantError> {
        self.inner
            .create_field_index(CreateFieldIndexCollectionBuilder::new(
                collection,
                field_name,
                FieldType::Keyword,
            ))
            .await?;
        Ok(())
    }
}

fn scored_point_to_search_result(sp: &ScoredPoint) -> SearchResult {
    let text = sp
        .payload
        .get("text")
        .and_then(|v| v.kind.as_ref())
        .and_then(|k| match k {
            qdrant_client::qdrant::value::Kind::StringValue(s) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_default();

    let payload: crate::types::Payload = serde_json::to_value(
        sp.payload
            .iter()
            .map(|(k, v)| {
                let json_val = match &v.kind {
                    Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => {
                        serde_json::Value::String(s.clone())
                    }
                    Some(qdrant_client::qdrant::value::Kind::ListValue(lv)) => {
                        serde_json::Value::Array(
                            lv.values
                                .iter()
                                .map(|v| match &v.kind {
                                    Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => {
                                        serde_json::Value::String(s.clone())
                                    }
                                    _ => serde_json::Value::String(format!("{:?}", v)),
                                })
                                .collect(),
                        )
                    }
                    _ => serde_json::Value::String(format!("{:?}", v)),
                };
                (k.clone(), json_val)
            })
            .collect::<std::collections::HashMap<_, _>>(),
    )
    .unwrap_or_default();

    SearchResult {
        text,
        score: sp.score,
        payload,
    }
}

pub struct RagCore {
    pub qdrant: QdrantClient,
    pub embedder: Arc<dyn Embedder>,
    embed_semaphore: Arc<Semaphore>,
    chunk_config: ChunkConfig,
    auto_index: bool,
}

impl RagCore {
    pub fn new(
        qdrant: QdrantClient,
        embedder: Box<dyn Embedder>,
        semaphore_size: Option<usize>,
    ) -> Self {
        let permits = semaphore_size.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        });
        Self {
            qdrant,
            embedder: Arc::from(embedder),
            embed_semaphore: Arc::new(Semaphore::new(permits)),
            chunk_config: ChunkConfig::default(),
            auto_index: true,
        }
    }

    pub fn with_chunk_config(mut self, config: ChunkConfig) -> Self {
        self.chunk_config = config;
        self
    }

    pub fn with_auto_index(mut self, auto_index: bool) -> Self {
        self.auto_index = auto_index;
        self
    }

    pub fn from_config(
        qdrant: QdrantClient,
        embedder: Box<dyn Embedder>,
        config: &crate::types::RagConfig,
    ) -> Self {
        Self::new(qdrant, embedder, Some(config.embed_semaphore_permits))
            .with_chunk_config(config.chunk.clone())
            .with_auto_index(config.auto_index)
    }

    pub async fn delete_collection(&self, name: &str) -> Result<bool, LqmError> {
        log::info!("deleting collection '{}'", name);
        Ok(self.qdrant.delete_collection(name).await?)
    }

    /// Create (or ensure) a collection. When `vector_dim` is `None`, uses the active embedder's
    /// dimension — the only size that will accept later upserts from this process.
    ///
    /// Returns `true` if the collection was newly created, `false` if it already existed.
    pub async fn create_collection(
        &self,
        name: &str,
        vector_dim: Option<usize>,
    ) -> Result<bool, LqmError> {
        if name.trim().is_empty() {
            return Err(LqmError::Validation(
                "collection name must not be empty".to_string(),
            ));
        }
        let dim = vector_dim.unwrap_or_else(|| self.embedder.dimension());
        if dim == 0 {
            return Err(LqmError::Validation(
                "vector dimension must be greater than zero".to_string(),
            ));
        }
        let existed = self.qdrant.collection_exists(name).await?;
        self.ensure_collection(name, dim).await?;
        Ok(!existed)
    }

    pub async fn get_collection_info(
        &self,
        name: &str,
    ) -> Result<Option<CollectionInfoSummary>, LqmError> {
        Ok(self.qdrant.get_collection_info(name).await?)
    }

    pub async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, LqmError> {
        log::debug!("embedding batch of {} texts", texts.len());
        let _permit = self
            .embed_semaphore
            .acquire()
            .await
            .map_err(|e| LqmError::Other(format!("semaphore acquire failed: {}", e)))?;
        Ok(self.embedder.embed_batch(texts).await?)
    }

    pub async fn embed_and_upsert_batch(
        &self,
        chunks: Vec<DocumentChunk>,
    ) -> Result<usize, LqmError> {
        let count = chunks.len();
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();

        let embeddings = self.embed_batch(texts).await?;

        let mut by_collection: std::collections::HashMap<String, Vec<UpsertPoint>> =
            std::collections::HashMap::new();

        for (chunk, vector) in chunks.into_iter().zip(embeddings) {
            let ingest_hash = compute_ingest_hash(&chunk.text);
            let target = chunk
                .collection
                .clone()
                .unwrap_or_else(|| crate::types::DEFAULT_COLLECTION_NAME.to_string());

            let mut payload = serde_json::Map::new();
            payload.insert("text".to_string(), serde_json::Value::String(chunk.text));
            payload.insert(
                "ingest_hash".to_string(),
                serde_json::Value::String(ingest_hash),
            );
            if let Some(source) = chunk.source {
                payload.insert("source".to_string(), serde_json::Value::String(source));
            }
            if let Some(source_type) = chunk.source_type {
                payload.insert(
                    "source_type".to_string(),
                    serde_json::Value::String(source_type),
                );
            }
            if let Some(tags) = chunk.tags {
                payload.insert(
                    "tags".to_string(),
                    serde_json::Value::Array(
                        tags.into_iter().map(serde_json::Value::String).collect(),
                    ),
                );
            }
            if let Some(timestamp) = chunk.timestamp {
                payload.insert(
                    "timestamp".to_string(),
                    serde_json::Value::String(timestamp),
                );
            }
            if let Some(project) = chunk.project {
                payload.insert("project".to_string(), serde_json::Value::String(project));
            }
            if let Some(last_modified) = chunk.last_modified {
                payload.insert(
                    "last_modified".to_string(),
                    serde_json::Value::String(last_modified),
                );
            }

            let point = UpsertPoint {
                id: uuid::Uuid::new_v4().to_string(),
                vector,
                payload: serde_json::Value::Object(payload),
            };

            by_collection.entry(target).or_default().push(point);
        }

        log::info!(
            "upserting {} chunks across {} collections",
            count,
            by_collection.len()
        );
        for (collection, points) in by_collection {
            self.qdrant.upsert_points(&collection, points).await?;
        }

        Ok(count)
    }

    pub async fn search(
        &self,
        query: &str,
        collection: Option<&str>,
        limit: Option<u64>,
        tags: Option<Vec<String>>,
        source_type: Option<&str>,
        min_score: Option<f32>,
    ) -> Result<Vec<SearchResult>, LqmError> {
        let collection = collection.unwrap_or(crate::types::DEFAULT_COLLECTION_NAME);
        let limit = limit.unwrap_or(10);
        // Qdrant rejects limit=0 with a validation error; treat as intentional empty result
        // so MCP/CLI callers get a clean [] instead of a hard failure.
        if limit == 0 {
            return Ok(vec![]);
        }

        log::debug!(
            "searching '{}' in '{}' (limit:{})",
            query,
            collection,
            limit
        );

        let query_embedding = self.embed_batch(vec![query.to_string()]).await?;
        let query_vector = query_embedding
            .into_iter()
            .next()
            .ok_or_else(|| LqmError::Other("embedding returned empty".to_string()))?;

        let mut filter = Filter {
            must: vec![],
            should: vec![],
            must_not: vec![],
            min_should: None,
        };

        if let Some(tags) = tags {
            let tag_conditions: Vec<_> = tags
                .into_iter()
                .map(|t| qdrant_client::qdrant::Condition {
                    condition_one_of: Some(
                        qdrant_client::qdrant::condition::ConditionOneOf::Field(
                            qdrant_client::qdrant::FieldCondition {
                                key: "tags".to_string(),
                                r#match: Some(qdrant_client::qdrant::Match {
                                    match_value: Some(
                                        qdrant_client::qdrant::r#match::MatchValue::Keyword(t),
                                    ),
                                }),
                                ..Default::default()
                            },
                        ),
                    ),
                })
                .collect();
            for cond in tag_conditions {
                filter.must.push(cond);
            }
        }

        if let Some(st) = source_type {
            filter.must.push(qdrant_client::qdrant::Condition {
                condition_one_of: Some(qdrant_client::qdrant::condition::ConditionOneOf::Field(
                    qdrant_client::qdrant::FieldCondition {
                        key: "source_type".to_string(),
                        r#match: Some(qdrant_client::qdrant::Match {
                            match_value: Some(qdrant_client::qdrant::r#match::MatchValue::Keyword(
                                st.to_string(),
                            )),
                        }),
                        ..Default::default()
                    },
                )),
            });
        }

        let results = self
            .qdrant
            .search(
                collection,
                query_vector,
                limit,
                if filter.must.is_empty() && filter.should.is_empty() {
                    None
                } else {
                    Some(filter)
                },
                min_score,
            )
            .await?;

        Ok(results)
    }

    pub async fn list_collections(&self) -> Result<Vec<String>, LqmError> {
        Ok(self.qdrant.list_collections().await?)
    }

    pub async fn ensure_collection(&self, name: &str, vector_dim: usize) -> Result<(), LqmError> {
        let exists = self.qdrant.collection_exists(name).await?;
        if exists {
            log::debug!("collection '{}' already exists", name);
        } else {
            log::info!("creating collection '{}' (dim={})", name, vector_dim);
            self.qdrant
                .create_collection(name, vector_dim as u64)
                .await?;
        }

        if self.auto_index {
            self.ensure_indexes(name).await?;
        }

        Ok(())
    }

    pub async fn ensure_indexes(&self, collection: &str) -> Result<(), LqmError> {
        for field in INDEX_FIELDS {
            log::debug!("creating index on {}.{}", collection, field);
            self.qdrant
                .create_field_index(collection, field)
                .await
                .map_err(LqmError::Qdrant)?;
        }
        Ok(())
    }
}

impl RagCore {
    pub fn chunk_text_method(&self, text: &str) -> Vec<String> {
        let strategy =
            ChunkingStrategy::new(self.chunk_config.chunk_size, self.chunk_config.overlap);
        chunk_text(text, &strategy)
    }
}

pub fn compute_ingest_hash(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::FakeEmbedder;

    fn make_test_qdrant_client() -> QdrantClient {
        QdrantClient {
            inner: Arc::new(
                QdrantGrpc::from_url("http://localhost:6334")
                    .build()
                    .expect("failed to build qdrant client for test"),
            ),
        }
    }

    fn make_fake_core(semaphore_size: usize) -> RagCore {
        let embedder = Box::new(FakeEmbedder::new(128));
        RagCore::new(make_test_qdrant_client(), embedder, Some(semaphore_size))
    }

    #[test]
    fn test_core_creation() {
        let core = make_fake_core(2);
        assert_eq!(core.embedder.dimension(), 128);
        assert_eq!(core.embedder.id(), "fake");
    }

    #[test]
    fn test_compute_ingest_hash() {
        let hash = compute_ingest_hash("hello world");
        assert_eq!(hash.len(), 64);
        let hash2 = compute_ingest_hash("hello world");
        assert_eq!(hash, hash2);
        let hash3 = compute_ingest_hash("different");
        assert_ne!(hash, hash3);
    }

    #[test]
    fn test_chunk_text_method() {
        let embedder = Box::new(FakeEmbedder::new(128));
        let core = RagCore::new(make_test_qdrant_client(), embedder, Some(2));
        let chunks = core.chunk_text_method("short text");
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0], "short text");
    }

    #[tokio::test]
    async fn test_qdrant_connection() {
        let qdrant = QdrantClient::new("http://localhost:6334").await;
        match qdrant {
            Ok(client) => {
                let collections = client.list_collections().await;
                if collections.is_err() {
                    eprintln!("Skipping: Qdrant connection succeeded but API call failed");
                }
            }
            Err(e) => {
                eprintln!("Qdrant not available (expected in CI): {:?}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_ensure_collection() {
        let core = make_fake_core(2);
        let result = core.ensure_collection("lqm_core_test_ensure", 128).await;
        match result {
            Ok(()) => {
                let collections = core.list_collections().await.unwrap();
                assert!(collections.iter().any(|c| c == "lqm_core_test_ensure"));
                let _ = core.delete_collection("lqm_core_test_ensure").await;
            }
            Err(e) => {
                eprintln!("Qdrant operation skipped (likely not running): {:?}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_delete_collection_nonexistent() {
        let core = make_fake_core(2);
        let result = core
            .delete_collection("lqm_nonexistent_collection_12345")
            .await;
        match result {
            Ok(deleted) => {
                assert!(!deleted, "Nonexistent collection should return false");
            }
            Err(e) => {
                eprintln!("Qdrant operation skipped (likely not running): {:?}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_create_collection_validation() {
        let core = make_fake_core(2);
        let empty = core.create_collection("  ", None).await;
        assert!(matches!(empty, Err(LqmError::Validation(_))));
        let zero_dim = core.create_collection("ok_name", Some(0)).await;
        assert!(matches!(zero_dim, Err(LqmError::Validation(_))));
    }

    #[tokio::test]
    async fn test_create_and_get_collection_info() {
        let core = make_fake_core(2);
        let coll = "lqm_core_test_info";
        let _ = core.delete_collection(coll).await;

        match core.create_collection(coll, None).await {
            Ok(created) => {
                assert!(created, "first create should report newly created");
                let created_again = core.create_collection(coll, None).await.unwrap();
                assert!(!created_again, "second create should report existing");

                let info = core.get_collection_info(coll).await.unwrap();
                let info = info.expect("collection info should exist");
                assert_eq!(info.name, coll);
                assert_eq!(info.vector_size, Some(128));
                assert!(info.distance.is_some());
                assert!(info.segments_count >= 1);

                let missing = core
                    .get_collection_info("lqm_core_missing_info_xyz")
                    .await
                    .unwrap();
                assert!(missing.is_none());

                let _ = core.delete_collection(coll).await;
            }
            Err(e) => {
                eprintln!("Qdrant operation skipped (likely not running): {:?}", e);
            }
        }
    }
}
