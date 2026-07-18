//! Qdrant client wrapper and `RagCore` (embed + upsert + search orchestration).
//!
//! All agent/HTTP/CLI surfaces should go through `RagCore` so chunking, payload
//! schema, and re-ingest policy stay consistent.

use crate::chunking::{ChunkingStrategy, chunk_text};
use crate::embedding::Embedder;
use crate::error::LqmError;
use crate::hybrid::{
    HybridKeywordBackend, encode_sparse_tf, keyword_candidates_from_payloads, text_index_query,
    tokenize_for_keyword,
};
use crate::lifecycle::decide_source_reingest;
use crate::types::{
    ChunkConfig, CollectionInfoSummary, DocumentChunk, EmbedderInfo, INDEX_FIELDS, IngestReport,
    PayloadFilter, ReingestAction, SearchFilter, SearchOptions, SearchPage, SearchResult,
    SourceSummary, UpsertPoint, payload_schema, payload_str,
};
use qdrant_client::Qdrant as QdrantGrpc;
use qdrant_client::qdrant::{
    CollectionStatus, Condition, CountPointsBuilder, CreateCollection,
    CreateFieldIndexCollectionBuilder, DeleteCollection, DeletePointsBuilder, DenseVector,
    Distance, FieldType, Filter, NamedVectors, PointId, PointStruct, ScoredPoint,
    ScrollPointsBuilder, SearchPoints, SparseIndices, SparseVectorConfig, SparseVectorParams,
    TokenizerType, UpsertPoints, Vector, VectorParams, Vectors, WithPayloadSelector, vector,
    vectors_config,
};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
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
    /// Build a client and validate connectivity with `list_collections`.
    ///
    /// The gRPC client constructor is synchronous; the `.await` is the health check.
    pub async fn new(url: &str) -> Result<Self, QdrantError> {
        let client = Self::new_lazy(url)?;
        // Fail fast when Qdrant is unreachable rather than on the first tool call.
        client.list_collections().await?;
        Ok(client)
    }

    /// Build a gRPC client **without** a connectivity probe.
    ///
    /// Useful for offline MCP harnesses that need a `RagCore` for tools that never
    /// touch Qdrant (e.g. `get_embedder_info`, pure validation). Production paths
    /// should keep using [`Self::new`].
    pub fn new_lazy(url: &str) -> Result<Self, QdrantError> {
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
        self.create_collection_with_sparse(name, vector_dim, false)
            .await
    }

    /// Create a collection with unnamed dense cosine vectors; optionally add named sparse `"sparse"`.
    pub async fn create_collection_with_sparse(
        &self,
        name: &str,
        vector_dim: u64,
        with_sparse: bool,
    ) -> Result<(), QdrantError> {
        let exists = self.collection_exists(name).await?;
        if exists {
            return Ok(());
        }
        let sparse_vectors_config = if with_sparse {
            Some(SparseVectorConfig {
                map: HashMap::from([(
                    crate::constants::SPARSE_VECTOR_NAME.to_string(),
                    SparseVectorParams::default(),
                )]),
            })
        } else {
            None
        };
        match self
            .inner
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
                sparse_vectors_config,
                ..Default::default()
            })
            .await
        {
            Ok(_) => Ok(()),
            // Concurrent creators may race past collection_exists; treat already-exists as success.
            Err(e) => {
                let msg = e.to_string().to_lowercase();
                if msg.contains("already") || msg.contains("exists") {
                    Ok(())
                } else {
                    Err(QdrantError::from(e))
                }
            }
        }
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

    fn make_point_vectors(dense: Vec<f32>, sparse: Option<(Vec<u32>, Vec<f32>)>) -> Vectors {
        match sparse {
            Some((indices, values)) if !indices.is_empty() && indices.len() == values.len() => {
                let named = NamedVectors::default()
                    .add_vector("", Self::make_dense_vector(dense))
                    .add_vector(
                        crate::constants::SPARSE_VECTOR_NAME,
                        Vector::new_sparse(indices, values),
                    );
                Vectors::from(named)
            }
            _ => Vectors {
                vectors_options: Some(qdrant_client::qdrant::vectors::VectorsOptions::Vector(
                    Self::make_dense_vector(dense),
                )),
            },
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
                    vectors: Some(Self::make_point_vectors(p.vector, p.sparse)),
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
        offset: Option<u64>,
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
            offset,
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

    /// Sparse-vector search against named sparse vector [`crate::constants::SPARSE_VECTOR_NAME`].
    pub async fn search_sparse(
        &self,
        collection: &str,
        indices: Vec<u32>,
        values: Vec<f32>,
        limit: u64,
        filter: Option<Filter>,
    ) -> Result<Vec<SearchResult>, QdrantError> {
        if indices.is_empty() || indices.len() != values.len() {
            return Ok(vec![]);
        }
        let search = SearchPoints {
            collection_name: collection.to_string(),
            vector: values,
            limit,
            filter,
            with_payload: Some(WithPayloadSelector {
                selector_options: Some(
                    qdrant_client::qdrant::with_payload_selector::SelectorOptions::Enable(true),
                ),
            }),
            params: None,
            score_threshold: None,
            offset: None,
            vector_name: Some(crate::constants::SPARSE_VECTOR_NAME.to_string()),
            with_vectors: None,
            read_consistency: None,
            timeout: None,
            shard_key_selector: None,
            sparse_indices: Some(SparseIndices { data: indices }),
        };

        let resp = self.inner.search_points(search).await?;
        Ok(resp
            .result
            .into_iter()
            .map(|p| scored_point_to_search_result(&p))
            .collect())
    }

    pub async fn create_field_index(
        &self,
        collection: &str,
        field_name: &str,
    ) -> Result<(), QdrantError> {
        match self
            .inner
            .create_field_index(CreateFieldIndexCollectionBuilder::new(
                collection,
                field_name,
                FieldType::Keyword,
            ))
            .await
        {
            Ok(_) => Ok(()),
            Err(e) => {
                let msg = e.to_string().to_lowercase();
                // Re-running ensure_indexes on existing collections is expected.
                if msg.contains("already") || msg.contains("exists") {
                    Ok(())
                } else {
                    Err(QdrantError::from(e))
                }
            }
        }
    }

    /// Full-text payload index on a string field (Word tokenizer, lowercase, min token len 2).
    pub async fn create_text_field_index(
        &self,
        collection: &str,
        field_name: &str,
    ) -> Result<(), QdrantError> {
        use qdrant_client::qdrant::TextIndexParamsBuilder;
        use qdrant_client::qdrant::payload_index_params::IndexParams;

        let params = TextIndexParamsBuilder::new(TokenizerType::Word)
            .lowercase(true)
            .min_token_len(2)
            .build();
        match self
            .inner
            .create_field_index(
                CreateFieldIndexCollectionBuilder::new(collection, field_name, FieldType::Text)
                    .field_index_params(IndexParams::TextIndexParams(params)),
            )
            .await
        {
            Ok(_) => Ok(()),
            Err(e) => {
                let msg = e.to_string().to_lowercase();
                if msg.contains("already") || msg.contains("exists") {
                    Ok(())
                } else {
                    Err(QdrantError::from(e))
                }
            }
        }
    }

    /// Scroll payload-only points (optionally filtered). Paginates until exhausted.
    pub async fn scroll_payloads(
        &self,
        collection: &str,
        filter: Option<Filter>,
        page_size: u32,
    ) -> Result<Vec<crate::types::Payload>, QdrantError> {
        let mut out = Vec::new();
        let mut offset: Option<PointId> = None;
        let page = page_size.max(1);
        loop {
            let mut builder = ScrollPointsBuilder::new(collection)
                .limit(page)
                .with_payload(true)
                .with_vectors(false);
            if let Some(ref f) = filter {
                builder = builder.filter(f.clone());
            }
            if let Some(off) = offset.clone() {
                builder = builder.offset(off);
            }
            let resp = self.inner.scroll(builder).await?;
            for pt in resp.result {
                out.push(qdrant_payload_to_json(pt.payload.iter()));
            }
            match resp.next_page_offset {
                Some(next) => offset = Some(next),
                None => break,
            }
        }
        Ok(out)
    }

    pub async fn count_points(
        &self,
        collection: &str,
        filter: Option<Filter>,
    ) -> Result<u64, QdrantError> {
        let mut builder = CountPointsBuilder::new(collection).exact(true);
        if let Some(f) = filter {
            builder = builder.filter(f);
        }
        let resp = self.inner.count(builder).await?;
        Ok(resp.result.map(|r| r.count).unwrap_or(0))
    }

    pub async fn delete_points_by_filter(
        &self,
        collection: &str,
        filter: Filter,
    ) -> Result<u64, QdrantError> {
        let deleted = self.count_points(collection, Some(filter.clone())).await?;
        if deleted == 0 {
            return Ok(0);
        }
        self.inner
            .delete_points(
                DeletePointsBuilder::new(collection)
                    .points(filter)
                    .wait(true),
            )
            .await?;
        Ok(deleted)
    }
}

fn keyword_match(key: &str, value: String) -> Condition {
    Condition {
        condition_one_of: Some(qdrant_client::qdrant::condition::ConditionOneOf::Field(
            qdrant_client::qdrant::FieldCondition {
                key: key.to_string(),
                r#match: Some(qdrant_client::qdrant::Match {
                    match_value: Some(qdrant_client::qdrant::r#match::MatchValue::Keyword(value)),
                }),
                ..Default::default()
            },
        )),
    }
}

/// Nested must-condition: clearance is one of the allowed levels (OR).
fn clearance_max_condition(max_clearance: &str) -> Option<Condition> {
    let levels = crate::scope::allowed_clearance_levels(max_clearance);
    if levels.is_empty() {
        return None;
    }
    let should: Vec<Condition> = levels
        .into_iter()
        .map(|l| keyword_match(payload_schema::CLEARANCE, l.to_string()))
        .collect();
    // Nested filter with only `should` → at least one clearance level must match.
    Some(Condition {
        condition_one_of: Some(qdrant_client::qdrant::condition::ConditionOneOf::Filter(
            Filter {
                must: vec![],
                should,
                must_not: vec![],
                min_should: None,
            },
        )),
    })
}

fn push_scope_and_clearance(
    must: &mut Vec<Condition>,
    scope: Option<&str>,
    max_clearance: Option<&str>,
) {
    if let Some(s) = scope.map(str::trim).filter(|s| !s.is_empty()) {
        must.push(keyword_match(payload_schema::SCOPE, s.to_string()));
    }
    if let Some(max) = max_clearance.map(str::trim).filter(|s| !s.is_empty())
        && let Some(cond) = clearance_max_condition(max)
    {
        must.push(cond);
    }
}

/// Build a Qdrant filter from payload fields (AND of all set fields).
pub fn payload_filter_to_qdrant(f: &PayloadFilter) -> Option<Filter> {
    if f.is_empty() {
        return None;
    }
    let mut must = Vec::new();
    if let Some(ref source) = f.source {
        must.push(keyword_match("source", source.clone()));
    }
    if let Some(ref st) = f.source_type {
        must.push(keyword_match("source_type", st.clone()));
    }
    if let Some(ref project) = f.project {
        must.push(keyword_match("project", project.clone()));
    }
    if let Some(ref tags) = f.tags {
        for t in tags {
            must.push(keyword_match("tags", t.clone()));
        }
    }
    push_scope_and_clearance(&mut must, f.scope.as_deref(), f.max_clearance.as_deref());
    if must.is_empty() {
        None
    } else {
        Some(Filter {
            must,
            should: vec![],
            must_not: vec![],
            min_should: None,
        })
    }
}

/// Build a Qdrant filter for search (must / should / must_not tag modes + scope/clearance).
pub fn search_filter_to_qdrant(f: &SearchFilter) -> Option<Filter> {
    if f.is_empty() {
        return None;
    }
    let mut must = Vec::new();
    let mut should = Vec::new();
    let mut must_not = Vec::new();

    if let Some(ref source) = f.source {
        must.push(keyword_match("source", source.clone()));
    }
    if let Some(ref st) = f.source_type {
        must.push(keyword_match("source_type", st.clone()));
    }
    if let Some(ref project) = f.project {
        must.push(keyword_match("project", project.clone()));
    }
    if let Some(ref tags) = f.tags {
        for t in tags {
            must.push(keyword_match("tags", t.clone()));
        }
    }
    if let Some(ref tags) = f.tags_should {
        for t in tags {
            should.push(keyword_match("tags", t.clone()));
        }
    }
    if let Some(ref tags) = f.tags_must_not {
        for t in tags {
            must_not.push(keyword_match("tags", t.clone()));
        }
    }
    push_scope_and_clearance(&mut must, f.scope.as_deref(), f.max_clearance.as_deref());

    if must.is_empty() && should.is_empty() && must_not.is_empty() {
        None
    } else {
        Some(Filter {
            must,
            should,
            must_not,
            min_should: None,
        })
    }
}

/// Convert a single Qdrant protobuf `Value` into `serde_json::Value`.
fn qdrant_value_to_json(v: &qdrant_client::qdrant::Value) -> serde_json::Value {
    match &v.kind {
        Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => {
            serde_json::Value::String(s.clone())
        }
        Some(qdrant_client::qdrant::value::Kind::ListValue(lv)) => serde_json::Value::Array(
            lv.values
                .iter()
                .map(|item| match &item.kind {
                    Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => {
                        serde_json::Value::String(s.clone())
                    }
                    _ => serde_json::Value::String(format!("{:?}", item)),
                })
                .collect(),
        ),
        _ => serde_json::Value::String(format!("{:?}", v)),
    }
}

fn qdrant_payload_to_json<'a, I>(payload: I) -> crate::types::Payload
where
    I: IntoIterator<Item = (&'a String, &'a qdrant_client::qdrant::Value)>,
{
    serde_json::to_value(
        payload
            .into_iter()
            .map(|(k, v)| (k.clone(), qdrant_value_to_json(v)))
            .collect::<HashMap<_, _>>(),
    )
    .unwrap_or_default()
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

    let payload: crate::types::Payload = qdrant_payload_to_json(sp.payload.iter());

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
    hybrid_keyword_backend: HybridKeywordBackend,
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
            hybrid_keyword_backend: HybridKeywordBackend::default(),
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

    pub fn with_hybrid_keyword_backend(mut self, backend: HybridKeywordBackend) -> Self {
        self.hybrid_keyword_backend = backend;
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
            .with_hybrid_keyword_backend(config.hybrid_keyword_backend)
    }

    /// Build `RagCore` from environment / optional config file and `QDRANT_URL`.
    ///
    /// - Embedder: `EmbedderConfig::load_or_default(config_path)`
    /// - Qdrant URL: `qdrant_url` if provided, else `QDRANT_URL`, else `DEFAULT_QDRANT_URL`
    /// - Hybrid keyword backend: `LQM_HYBRID_KEYWORD_BACKEND` (default `keyword_index`)
    pub async fn from_env(
        qdrant_url: Option<&str>,
        config_path: Option<&str>,
    ) -> Result<Self, LqmError> {
        let url = qdrant_url
            .map(|s| s.to_string())
            .or_else(|| std::env::var("QDRANT_URL").ok())
            .unwrap_or_else(|| crate::constants::DEFAULT_QDRANT_URL.to_string());
        let embedder_config = crate::config::EmbedderConfig::load_or_default(config_path)?;
        let embedder = crate::config::create_embedder(&embedder_config)?;
        let qdrant = QdrantClient::new(&url).await?;
        Ok(Self::from_config(
            qdrant,
            embedder,
            &crate::types::RagConfig::default(),
        ))
    }

    /// Chunking config currently applied to structure-aware ingest.
    pub fn chunk_config(&self) -> &ChunkConfig {
        &self.chunk_config
    }

    /// Whether payload indexes are auto-created on `ensure_collection`.
    pub fn auto_index(&self) -> bool {
        self.auto_index
    }

    /// Hybrid keyword candidate backend (`scroll` | `sparse` | `keyword_index`).
    pub fn hybrid_keyword_backend(&self) -> HybridKeywordBackend {
        self.hybrid_keyword_backend
    }

    pub async fn delete_collection(&self, name: &str) -> Result<bool, LqmError> {
        log::info!("deleting collection '{}'", name);
        Ok(self.qdrant.delete_collection(name).await?)
    }

    /// Create (or ensure) a collection. When `vector_dim` is `None`, uses the active embedder's
    /// dimension — the only size that will accept later upserts from this process.
    ///
    /// Returns `true` if the collection did not exist before this call (best-effort under
    /// concurrent creators; both may observe `created=true` while only one actually creates).
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
        // ensure_collection re-checks + creates with already-exists tolerance (TOCTOU-safe).
        self.ensure_collection(name, Some(dim)).await?;
        if !existed {
            log::info!(
                "collection '{}' ready (dim={}, hybrid_keyword_backend={}, sparse_schema={})",
                name,
                dim,
                self.hybrid_keyword_backend,
                self.hybrid_keyword_backend.needs_sparse_schema()
            );
        }
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

    /// Embed and upsert chunks with skip/replace-by-source policy.
    ///
    /// For each `(collection, source)` group: compare content hashes to existing
    /// points for that source — skip if identical multiset, replace (delete then
    /// write) if different, insert if none exist. Chunks without a source always insert.
    pub async fn embed_and_upsert_batch(
        &self,
        chunks: Vec<DocumentChunk>,
    ) -> Result<IngestReport, LqmError> {
        if chunks.is_empty() {
            return Ok(IngestReport::default());
        }

        // Group by (collection, source key). Empty source key = always-insert path.
        let mut groups: HashMap<(String, String), Vec<DocumentChunk>> = HashMap::new();
        for chunk in chunks {
            let coll = chunk
                .collection
                .clone()
                .unwrap_or_else(|| crate::types::DEFAULT_COLLECTION_NAME.to_string());
            let source_key = chunk.source.clone().unwrap_or_default();
            groups.entry((coll, source_key)).or_default().push(chunk);
        }

        let mut report = IngestReport::default();

        for ((collection, source_key), group_chunks) in groups {
            let new_hashes: Vec<String> = group_chunks
                .iter()
                .map(|c| compute_ingest_hash(&c.text))
                .collect();

            let action = if source_key.is_empty() {
                ReingestAction::Insert
            } else {
                let existing = self.hashes_for_source(&collection, &source_key).await?;
                decide_source_reingest(&existing, &new_hashes)
            };

            match action {
                ReingestAction::Skip => {
                    log::info!(
                        "skipping re-ingest of source '{}' in '{}' ({} chunks unchanged)",
                        source_key,
                        collection,
                        group_chunks.len()
                    );
                    report.skipped += 1;
                    continue;
                }
                ReingestAction::Replace => {
                    log::info!(
                        "replacing source '{}' in '{}' ({} new chunks)",
                        source_key,
                        collection,
                        group_chunks.len()
                    );
                    self.delete_by_source(&collection, &source_key).await?;
                    report.replaced += 1;
                }
                ReingestAction::Insert => {
                    report.inserted += 1;
                }
            }

            let texts: Vec<String> = group_chunks.iter().map(|c| c.text.clone()).collect();
            let embeddings = self.embed_batch(texts).await?;
            let mut points = Vec::with_capacity(group_chunks.len());
            let group_len = group_chunks.len();
            let embedding_model = self
                .embedder
                .model()
                .map(|s| s.to_string())
                .unwrap_or_else(|| self.embedder.id().to_string());

            let write_sparse = self.hybrid_keyword_backend.needs_sparse_schema();
            for (pos, (chunk, vector)) in group_chunks.into_iter().zip(embeddings).enumerate() {
                let chunk_index = chunk.chunk_index.unwrap_or(pos);
                let total_chunks = chunk.total_chunks.unwrap_or(group_len);
                let payload =
                    build_point_payload(&chunk, chunk_index, total_chunks, &embedding_model);

                let sparse = if write_sparse {
                    let enc = encode_sparse_tf(&chunk.text);
                    if enc.is_empty() {
                        None
                    } else {
                        Some((enc.indices, enc.values))
                    }
                } else {
                    None
                };

                points.push(UpsertPoint {
                    id: uuid::Uuid::new_v4().to_string(),
                    vector,
                    sparse,
                    payload: serde_json::Value::Object(payload),
                });
            }

            let n = points.len();
            self.qdrant.upsert_points(&collection, points).await?;
            report.chunks += n;
        }

        log::info!(
            "ingest report: inserted={} skipped={} replaced={} chunks={}",
            report.inserted,
            report.skipped,
            report.replaced,
            report.chunks
        );
        Ok(report)
    }

    async fn hashes_for_source(
        &self,
        collection: &str,
        source: &str,
    ) -> Result<Vec<String>, LqmError> {
        let filter = payload_filter_to_qdrant(&PayloadFilter::for_source(source));
        let payloads = self
            .qdrant
            .scroll_payloads(collection, filter, crate::constants::SCROLL_PAGE_SIZE)
            .await?;
        Ok(payloads
            .iter()
            .filter_map(|p| payload_str(p, "ingest_hash"))
            .collect())
    }

    /// List distinct sources in a collection with point counts and sample metadata.
    pub async fn list_sources(&self, collection: &str) -> Result<Vec<SourceSummary>, LqmError> {
        if !self.qdrant.collection_exists(collection).await? {
            return Ok(vec![]);
        }
        let payloads = self
            .qdrant
            .scroll_payloads(collection, None, crate::constants::SCROLL_PAGE_SIZE)
            .await?;
        let mut map: HashMap<String, SourceSummary> = HashMap::new();
        for p in payloads {
            let Some(source) = payload_str(&p, "source") else {
                continue;
            };
            let entry = map.entry(source.clone()).or_insert_with(|| SourceSummary {
                source,
                count: 0,
                source_type: None,
                last_modified: None,
            });
            entry.count += 1;
            if entry.source_type.is_none() {
                entry.source_type = payload_str(&p, "source_type");
            }
            if entry.last_modified.is_none() {
                entry.last_modified = payload_str(&p, "last_modified");
            }
        }
        let mut sources: Vec<_> = map.into_values().collect();
        sources.sort_by(|a, b| a.source.cmp(&b.source));
        Ok(sources)
    }

    pub async fn delete_by_source(&self, collection: &str, source: &str) -> Result<u64, LqmError> {
        if source.trim().is_empty() {
            return Err(LqmError::Validation(
                "source must not be empty for delete_by_source".to_string(),
            ));
        }
        self.delete_by_filter(collection, &PayloadFilter::for_source(source))
            .await
    }

    /// Fetch all payload points for a source (unordered scroll).
    async fn payloads_for_source(
        &self,
        collection: &str,
        source: &str,
    ) -> Result<Vec<crate::types::Payload>, LqmError> {
        if source.trim().is_empty() {
            return Err(LqmError::Validation(
                "source must not be empty for source reconstruction".to_string(),
            ));
        }
        if !self.qdrant.collection_exists(collection).await? {
            return Ok(vec![]);
        }
        let filter = payload_filter_to_qdrant(&PayloadFilter::for_source(source));
        Ok(self
            .qdrant
            .scroll_payloads(collection, filter, crate::constants::SCROLL_PAGE_SIZE)
            .await?)
    }

    /// List chunks for a source ordered by `chunk_index` with pagination.
    ///
    /// Missing `chunk_index` sorts last. `limit=0` returns an empty page (same
    /// as search). Default limit is [`crate::reconstruction::DEFAULT_LIST_CHUNKS_LIMIT`].
    pub async fn list_chunks(
        &self,
        collection: &str,
        source: &str,
        offset: Option<u64>,
        limit: Option<u64>,
    ) -> Result<crate::reconstruction::SourceChunkPage, LqmError> {
        let payloads = self.payloads_for_source(collection, source).await?;
        let chunks: Vec<_> = payloads
            .into_iter()
            .map(crate::reconstruction::source_chunk_from_payload)
            .collect();
        let offset = offset.unwrap_or(0);
        let limit = limit.unwrap_or(crate::reconstruction::DEFAULT_LIST_CHUNKS_LIMIT);
        Ok(crate::reconstruction::paginate_source_chunks(
            chunks, source, offset, limit,
        ))
    }

    /// Reconstruct a full source: all chunks ordered + joined text.
    pub async fn get_source(
        &self,
        collection: &str,
        source: &str,
    ) -> Result<crate::reconstruction::SourceDocument, LqmError> {
        let payloads = self.payloads_for_source(collection, source).await?;
        let chunks: Vec<_> = payloads
            .into_iter()
            .map(crate::reconstruction::source_chunk_from_payload)
            .collect();
        Ok(crate::reconstruction::source_document_from_chunks(
            source, chunks,
        ))
    }

    /// Neighboring chunks of the same source around `chunk_index` (±`neighbors`).
    ///
    /// Default neighbors is [`crate::reconstruction::DEFAULT_EXPAND_NEIGHBORS`].
    /// Only indexed chunks already stored for that source are returned.
    pub async fn expand_context(
        &self,
        collection: &str,
        source: &str,
        chunk_index: u64,
        neighbors: Option<u64>,
    ) -> Result<Vec<crate::reconstruction::SourceChunk>, LqmError> {
        let payloads = self.payloads_for_source(collection, source).await?;
        let chunks: Vec<_> = payloads
            .into_iter()
            .map(crate::reconstruction::source_chunk_from_payload)
            .collect();
        let neighbors = neighbors.unwrap_or(crate::reconstruction::DEFAULT_EXPAND_NEIGHBORS);
        Ok(crate::reconstruction::expand_chunk_neighbors(
            &chunks,
            chunk_index,
            neighbors,
        ))
    }

    pub async fn delete_by_filter(
        &self,
        collection: &str,
        filter: &PayloadFilter,
    ) -> Result<u64, LqmError> {
        if filter.is_empty() {
            return Err(LqmError::Validation(
                "delete_by_filter requires at least one of source, source_type, project, tags, scope, max_clearance"
                    .to_string(),
            ));
        }
        if !self.qdrant.collection_exists(collection).await? {
            return Ok(0);
        }
        let qf = payload_filter_to_qdrant(filter).ok_or_else(|| {
            LqmError::Validation("delete_by_filter produced empty qdrant filter".to_string())
        })?;
        Ok(self.qdrant.delete_points_by_filter(collection, qf).await?)
    }

    /// Convenience search returning only hits (no pagination metadata).
    pub async fn search(
        &self,
        query: &str,
        collection: Option<&str>,
        limit: Option<u64>,
        tags: Option<Vec<String>>,
        source_type: Option<&str>,
        min_score: Option<f32>,
    ) -> Result<Vec<SearchResult>, LqmError> {
        let page = self
            .search_page(
                query,
                SearchOptions {
                    collection: collection.map(|s| s.to_string()),
                    limit,
                    offset: None,
                    min_score,
                    filter: SearchFilter {
                        source: None,
                        source_type: source_type.map(|s| s.to_string()),
                        project: None,
                        tags,
                        tags_should: None,
                        tags_must_not: None,
                        scope: None,
                        max_clearance: None,
                    },
                    hybrid: false,
                    hybrid_alpha: None,
                },
            )
            .await?;
        Ok(page.results)
    }

    /// Filtered semantic search with offset pagination (`has_more` / `next_offset`).
    ///
    /// When `opts.hybrid` is true, over-fetches dense hits, merges keyword
    /// candidates from [`Self::hybrid_keyword_backend`], and fuses with
    /// [`crate::hybrid::merge_and_fuse_hybrid`]. Dense-only when hybrid is false.
    pub async fn search_page(
        &self,
        query: &str,
        opts: SearchOptions,
    ) -> Result<SearchPage, LqmError> {
        let collection = opts
            .collection
            .as_deref()
            .unwrap_or(crate::types::DEFAULT_COLLECTION_NAME);
        let limit = opts.limit.unwrap_or(10);
        let offset = opts.offset.unwrap_or(0);

        // Qdrant rejects limit=0; empty page is intentional.
        if limit == 0 {
            return Ok(SearchPage {
                results: vec![],
                offset,
                limit: 0,
                has_more: false,
                next_offset: None,
            });
        }

        log::debug!(
            "searching '{}' in '{}' (limit:{} offset:{} hybrid:{} keyword_backend:{})",
            query,
            collection,
            limit,
            offset,
            opts.hybrid,
            self.hybrid_keyword_backend
        );

        let query_embedding = self.embed_batch(vec![query.to_string()]).await?;
        let query_vector = query_embedding
            .into_iter()
            .next()
            .ok_or_else(|| LqmError::Other("embedding returned empty".to_string()))?;

        let filter = search_filter_to_qdrant(&opts.filter);

        if !opts.hybrid {
            let fetch_limit = limit.saturating_add(crate::constants::HAS_MORE_EXTRA);
            let mut results = self
                .qdrant
                .search(
                    collection,
                    query_vector,
                    fetch_limit,
                    filter,
                    opts.min_score,
                    Some(offset),
                )
                .await?;

            let has_more = results.len() as u64 > limit;
            if has_more {
                results.truncate(limit as usize);
            }
            let next_offset = if has_more {
                Some(offset.saturating_add(limit))
            } else {
                None
            };

            return Ok(SearchPage {
                results,
                offset,
                limit,
                has_more,
                next_offset,
            });
        }

        // Hybrid path: over-fetch dense, merge keyword candidates, fuse.
        let alpha = opts
            .hybrid_alpha
            .unwrap_or(crate::hybrid::DEFAULT_HYBRID_ALPHA);
        let dense_fetch = crate::hybrid::hybrid_dense_fetch_limit(limit);
        // Fetch dense from offset 0 for a stable fusion pool when hybrid; apply
        // pagination after fusion. (Dense-only keeps native Qdrant offset.)
        let dense_pool = self
            .qdrant
            .search(
                collection,
                query_vector,
                dense_fetch,
                filter.clone(),
                None, // min_score applied after fusion if set
                Some(0),
            )
            .await?;

        let tokens = tokenize_for_keyword(query);
        let keyword_candidates = self
            .hybrid_keyword_candidates(collection, query, &tokens, filter)
            .await;

        let mut fused = crate::hybrid::merge_and_fuse_hybrid(
            &dense_pool,
            &keyword_candidates,
            query,
            alpha,
            crate::hybrid::DEFAULT_RRF_K,
        );

        if let Some(min) = opts.min_score {
            fused.retain(|r| r.score >= min);
        }

        let total = fused.len() as u64;
        let start = offset.min(total) as usize;
        let end = (offset.saturating_add(limit)).min(total) as usize;
        let page_results = if start < fused.len() {
            fused[start..end].to_vec()
        } else {
            Vec::new()
        };
        let has_more = end < fused.len();
        let next_offset = if has_more {
            Some(offset.saturating_add(limit))
        } else {
            None
        };

        Ok(SearchPage {
            results: page_results,
            offset,
            limit,
            has_more,
            next_offset,
        })
    }

    /// Collect keyword candidates for hybrid fusion according to configured backend.
    ///
    /// Never fails the search: sparse schema mismatch falls back to keyword_index
    /// then scroll; empty tokens yield no candidates.
    async fn hybrid_keyword_candidates(
        &self,
        collection: &str,
        query: &str,
        tokens: &[String],
        filter: Option<Filter>,
    ) -> Vec<SearchResult> {
        if tokens.is_empty() {
            return Vec::new();
        }

        match self.hybrid_keyword_backend {
            HybridKeywordBackend::Sparse => {
                match self
                    .keyword_candidates_sparse(collection, query, filter.clone())
                    .await
                {
                    Ok(cands) if !cands.is_empty() => cands,
                    Ok(_) => {
                        log::debug!(
                            "sparse hybrid returned no candidates; trying keyword_index fallback"
                        );
                        self.keyword_candidates_text_index(collection, tokens, filter)
                            .await
                    }
                    Err(e) => {
                        log::warn!(
                            "sparse hybrid candidate search failed ({e}); falling back to keyword_index"
                        );
                        self.keyword_candidates_text_index(collection, tokens, filter)
                            .await
                    }
                }
            }
            HybridKeywordBackend::KeywordIndex => {
                self.keyword_candidates_text_index(collection, tokens, filter)
                    .await
            }
            HybridKeywordBackend::Scroll => {
                self.keyword_candidates_scroll(collection, tokens, filter)
                    .await
            }
        }
    }

    async fn keyword_candidates_sparse(
        &self,
        collection: &str,
        query: &str,
        filter: Option<Filter>,
    ) -> Result<Vec<SearchResult>, LqmError> {
        let enc = encode_sparse_tf(query);
        if enc.is_empty() {
            return Ok(vec![]);
        }
        let limit = crate::constants::KEYWORD_CANDIDATE_LIMIT;
        let mut results = self
            .qdrant
            .search_sparse(collection, enc.indices, enc.values, limit, filter)
            .await?;
        // Dense component for fusion is 0; text keyword_score drives re-rank.
        for r in &mut results {
            r.score = 0.0;
        }
        Ok(results)
    }

    async fn keyword_candidates_text_index(
        &self,
        collection: &str,
        tokens: &[String],
        base_filter: Option<Filter>,
    ) -> Vec<SearchResult> {
        let text_q = text_index_query(tokens);
        if text_q.is_empty() {
            return Vec::new();
        }
        let text_cond = Condition::matches_text_any(payload_schema::TEXT, text_q);
        let filter = and_filter(base_filter.clone(), Filter::must([text_cond]));
        match self
            .qdrant
            .scroll_payloads(
                collection,
                filter,
                crate::constants::KEYWORD_SCROLL_PAGE.min(
                    crate::constants::KEYWORD_CANDIDATE_LIMIT
                        .try_into()
                        .unwrap_or(u32::MAX),
                ),
            )
            .await
        {
            Ok(payloads) => {
                let mut cands = keyword_candidates_from_payloads(payloads, tokens);
                cands.truncate(crate::constants::KEYWORD_CANDIDATE_LIMIT as usize);
                cands
            }
            Err(e) => {
                log::warn!(
                    "keyword_index scroll failed ({e}); falling back to full collection scroll"
                );
                self.keyword_candidates_scroll(collection, tokens, base_filter)
                    .await
            }
        }
    }

    async fn keyword_candidates_scroll(
        &self,
        collection: &str,
        tokens: &[String],
        filter: Option<Filter>,
    ) -> Vec<SearchResult> {
        match self
            .qdrant
            .scroll_payloads(collection, filter, crate::constants::KEYWORD_SCROLL_PAGE)
            .await
        {
            Ok(payloads) => {
                let mut cands = keyword_candidates_from_payloads(payloads, tokens);
                cands.truncate(crate::constants::KEYWORD_CANDIDATE_LIMIT as usize);
                cands
            }
            Err(e) => {
                log::warn!("keyword scroll failed: {e}");
                Vec::new()
            }
        }
    }

    pub async fn list_collections(&self) -> Result<Vec<String>, LqmError> {
        Ok(self.qdrant.list_collections().await?)
    }

    /// Ensure a collection exists. When `vector_dim` is `None`, uses the active embedder dimension.
    ///
    /// New collections get a sparse vector schema when
    /// [`HybridKeywordBackend::needs_sparse_schema`] is true for the configured backend.
    pub async fn ensure_collection(
        &self,
        name: &str,
        vector_dim: Option<usize>,
    ) -> Result<(), LqmError> {
        let dim = vector_dim.unwrap_or_else(|| self.embedder.dimension());
        let exists = self.qdrant.collection_exists(name).await?;
        if exists {
            log::debug!("collection '{}' already exists", name);
        } else {
            let with_sparse = self.hybrid_keyword_backend.needs_sparse_schema();
            log::info!(
                "creating collection '{}' (dim={}, sparse={})",
                name,
                dim,
                with_sparse
            );
            self.qdrant
                .create_collection_with_sparse(name, dim as u64, with_sparse)
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
        // Text index enables keyword_index hybrid without full-collection scroll.
        // Harmless for scroll backend; sparse still benefits if it falls back.
        if self.hybrid_keyword_backend.needs_text_index()
            || matches!(self.hybrid_keyword_backend, HybridKeywordBackend::Sparse)
        {
            log::debug!(
                "creating text index on {}.{}",
                collection,
                payload_schema::TEXT
            );
            self.qdrant
                .create_text_field_index(collection, payload_schema::TEXT)
                .await
                .map_err(LqmError::Qdrant)?;
        }
        Ok(())
    }
}

/// Combine two optional filters with AND semantics on `must`/`should`/`must_not`.
fn and_filter(a: Option<Filter>, b: Filter) -> Option<Filter> {
    match a {
        None => Some(b),
        Some(mut base) => {
            base.must.extend(b.must);
            base.should.extend(b.should);
            base.must_not.extend(b.must_not);
            if base.min_should.is_none() {
                base.min_should = b.min_should;
            }
            Some(base)
        }
    }
}

impl RagCore {
    pub fn chunk_text_method(&self, text: &str) -> Vec<String> {
        let strategy =
            ChunkingStrategy::new(self.chunk_config.chunk_size, self.chunk_config.overlap);
        chunk_text(text, &strategy)
    }

    /// Structure-aware chunking (markdown headings / code defs / plain paragraphs).
    pub fn chunk_for_ingest(
        &self,
        text: &str,
        source_type: Option<&str>,
        path_hint: Option<&str>,
    ) -> Vec<String> {
        let strategy =
            ChunkingStrategy::new(self.chunk_config.chunk_size, self.chunk_config.overlap);
        crate::chunking::chunk_for_ingest(text, source_type, path_hint, &strategy)
    }

    /// Expand a source document into structure-aware `DocumentChunk`s for upsert.
    ///
    /// Shared by MCP, HTTP API, and CLI so all surfaces produce identical chunk boundaries.
    #[allow(clippy::too_many_arguments)]
    pub fn expand_to_chunks(
        &self,
        text: &str,
        source: Option<String>,
        source_type: Option<String>,
        collection: String,
        tags: Option<Vec<String>>,
        project: Option<String>,
        last_modified: Option<String>,
        path_hint: Option<&str>,
        scope: Option<String>,
        clearance: Option<String>,
    ) -> Vec<DocumentChunk> {
        let pieces = self.chunk_for_ingest(
            text,
            source_type.as_deref(),
            path_hint.or(source.as_deref()),
        );
        let total = pieces.len();
        pieces
            .into_iter()
            .enumerate()
            .map(|(i, text)| DocumentChunk {
                text,
                source: source.clone(),
                source_type: source_type.clone(),
                collection: Some(collection.clone()),
                tags: tags.clone(),
                timestamp: None,
                project: project.clone(),
                last_modified: last_modified.clone(),
                chunk_index: Some(i),
                total_chunks: Some(total),
                importance: None,
                memory_id: None,
                scope: scope.clone(),
                clearance: clearance.clone(),
            })
            .collect()
    }

    /// Active embedder identity for agent/HTTP introspection.
    pub fn embedder_info(&self) -> EmbedderInfo {
        EmbedderInfo {
            id: self.embedder.id().to_string(),
            dimension: self.embedder.dimension(),
            model: self.embedder.model().map(|s| s.to_string()),
        }
    }

    /// Persist a long-term agent memory (dedicated collection by default).
    ///
    /// Uses the same skip/replace-by-source path as document ingest (`source=memory://id`).
    /// Returns `(report, effective_memory_id)` — the id is caller-supplied or auto-assigned.
    pub async fn store_memory(
        &self,
        note: crate::memory::MemoryNote,
        collection: Option<&str>,
    ) -> Result<(IngestReport, String), LqmError> {
        if note.text.trim().is_empty() {
            return Err(LqmError::Validation(
                "memory text must not be empty".to_string(),
            ));
        }
        let coll = collection.unwrap_or(crate::memory::DEFAULT_MEMORY_COLLECTION);
        self.ensure_collection(coll, None).await?;

        let now = crate::types::unix_now_secs();
        let chunk = crate::memory::memory_note_to_chunk(&note, coll, now);
        let memory_id = chunk
            .memory_id
            .clone()
            .unwrap_or_else(|| format!("mem-{now}"));
        let report = self.embed_and_upsert_batch(vec![chunk]).await?;
        Ok((report, memory_id))
    }

    /// Semantic recall over the memory collection with optional recency/importance blend.
    pub async fn recall_memories(
        &self,
        query: &str,
        collection: Option<&str>,
        limit: Option<u64>,
        use_recency: bool,
        project: Option<&str>,
        tags: Option<Vec<String>>,
    ) -> Result<Vec<crate::memory::MemoryHit>, LqmError> {
        if query.trim().is_empty() {
            return Err(LqmError::Validation(
                "recall query must not be empty".to_string(),
            ));
        }
        let coll = collection.unwrap_or(crate::memory::DEFAULT_MEMORY_COLLECTION);
        let page = self
            .search_page(
                query,
                SearchOptions {
                    collection: Some(coll.to_string()),
                    limit: Some(limit.unwrap_or(8)),
                    offset: None,
                    min_score: None,
                    filter: SearchFilter {
                        source: None,
                        source_type: Some(crate::memory::MEMORY_SOURCE_TYPE.to_string()),
                        project: project.map(|s| s.to_string()),
                        tags,
                        tags_should: None,
                        tags_must_not: None,
                        scope: None,
                        max_clearance: None,
                    },
                    hybrid: false,
                    hybrid_alpha: None,
                },
            )
            .await?;

        let now = crate::types::unix_now_secs();
        let hits = crate::memory::rank_memory_hits(
            &page.results,
            now,
            use_recency,
            crate::constants::MEMORY_RECENCY_HALF_LIFE,
        );
        Ok(hits)
    }
}

/// Build the Qdrant payload map for a single chunk (shared schema; unit-tested offline).
pub fn build_point_payload(
    chunk: &DocumentChunk,
    chunk_index: usize,
    total_chunks: usize,
    embedding_model: &str,
) -> serde_json::Map<String, serde_json::Value> {
    let ingest_hash = compute_ingest_hash(&chunk.text);
    let mut payload = serde_json::Map::new();
    payload.insert(
        payload_schema::TEXT.to_string(),
        serde_json::Value::String(chunk.text.clone()),
    );
    payload.insert(
        payload_schema::INGEST_HASH.to_string(),
        serde_json::Value::String(ingest_hash),
    );
    if let Some(ref source) = chunk.source {
        payload.insert(
            payload_schema::SOURCE.to_string(),
            serde_json::Value::String(source.clone()),
        );
    }
    if let Some(ref source_type) = chunk.source_type {
        payload.insert(
            payload_schema::SOURCE_TYPE.to_string(),
            serde_json::Value::String(source_type.clone()),
        );
    }
    if let Some(ref tags) = chunk.tags {
        payload.insert(
            payload_schema::TAGS.to_string(),
            serde_json::Value::Array(
                tags.iter()
                    .cloned()
                    .map(serde_json::Value::String)
                    .collect(),
            ),
        );
    }
    if let Some(ref timestamp) = chunk.timestamp {
        payload.insert(
            payload_schema::TIMESTAMP.to_string(),
            serde_json::Value::String(timestamp.clone()),
        );
    }
    if let Some(ref project) = chunk.project {
        payload.insert(
            payload_schema::PROJECT.to_string(),
            serde_json::Value::String(project.clone()),
        );
    }
    if let Some(ref last_modified) = chunk.last_modified {
        payload.insert(
            payload_schema::LAST_MODIFIED.to_string(),
            serde_json::Value::String(last_modified.clone()),
        );
    }
    payload.insert(
        payload_schema::CHUNK_INDEX.to_string(),
        serde_json::Value::Number(chunk_index.into()),
    );
    payload.insert(
        payload_schema::TOTAL_CHUNKS.to_string(),
        serde_json::Value::Number(total_chunks.into()),
    );
    payload.insert(
        payload_schema::EMBEDDING_MODEL.to_string(),
        serde_json::Value::String(embedding_model.to_string()),
    );
    if let Some(imp) = chunk.importance {
        // String form survives Qdrant StringValue round-trip (upsert maps non-strings
        // via v.to_string(); scored_point restores StringValue as JSON String).
        payload.insert(
            payload_schema::IMPORTANCE.to_string(),
            serde_json::Value::String(format!("{}", imp.clamp(0.0, 1.0))),
        );
    }
    if let Some(ref mid) = chunk.memory_id {
        payload.insert(
            payload_schema::MEMORY_ID.to_string(),
            serde_json::Value::String(mid.clone()),
        );
    }
    // last_accessed mirrors timestamp for memories when set
    if chunk.source_type.as_deref() == Some(crate::memory::MEMORY_SOURCE_TYPE)
        && let Some(ref ts) = chunk.timestamp
    {
        payload.insert(
            payload_schema::LAST_ACCESSED.to_string(),
            serde_json::Value::String(ts.clone()),
        );
    }
    if let Some(ref scope) = chunk.scope {
        let s = scope.trim();
        if !s.is_empty() {
            payload.insert(
                payload_schema::SCOPE.to_string(),
                serde_json::Value::String(s.to_string()),
            );
        }
    }
    // Always write clearance when set or default to public for clearance-safe filters.
    let clearance = chunk
        .clearance
        .as_deref()
        .and_then(crate::scope::normalize_clearance)
        .unwrap_or(crate::scope::DEFAULT_CLEARANCE);
    payload.insert(
        payload_schema::CLEARANCE.to_string(),
        serde_json::Value::String(clearance.to_string()),
    );
    payload
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
    fn test_build_point_payload_schema_keys() {
        let chunk = DocumentChunk {
            text: "hello".into(),
            source: Some("s".into()),
            source_type: Some("text".into()),
            collection: Some("c".into()),
            tags: Some(vec!["t".into()]),
            timestamp: None,
            project: Some("p".into()),
            last_modified: None,
            chunk_index: Some(1),
            total_chunks: Some(3),
            importance: None,
            memory_id: None,
            scope: None,
            clearance: None,
        };
        let payload = build_point_payload(&chunk, 1, 3, "AllMiniLML6V2");
        assert_eq!(payload[payload_schema::TEXT], "hello");
        assert!(payload[payload_schema::INGEST_HASH].as_str().unwrap().len() == 64);
        assert_eq!(payload[payload_schema::SOURCE], "s");
        assert_eq!(payload[payload_schema::CHUNK_INDEX], 1);
        assert_eq!(payload[payload_schema::TOTAL_CHUNKS], 3);
        assert_eq!(payload[payload_schema::EMBEDDING_MODEL], "AllMiniLML6V2");
    }

    #[test]
    fn test_build_point_payload_memory_importance_is_string() {
        let chunk = DocumentChunk {
            text: "note".into(),
            source: Some("memory://m1".into()),
            source_type: Some(crate::memory::MEMORY_SOURCE_TYPE.into()),
            collection: Some(crate::memory::DEFAULT_MEMORY_COLLECTION.into()),
            tags: None,
            timestamp: Some("1700000000".into()),
            project: None,
            last_modified: Some("1700000000".into()),
            chunk_index: Some(0),
            total_chunks: Some(1),
            importance: Some(0.85),
            memory_id: Some("m1".into()),
            scope: None,
            clearance: None,
        };
        let payload = build_point_payload(&chunk, 0, 1, "fake");
        // Must be a JSON string so Qdrant StringValue path is a no-op identity.
        assert_eq!(
            payload[payload_schema::IMPORTANCE].as_str(),
            Some("0.85"),
            "importance should be string for round-trip: {:?}",
            payload[payload_schema::IMPORTANCE]
        );
        assert_eq!(
            payload[payload_schema::LAST_ACCESSED].as_str(),
            Some("1700000000")
        );
        assert_eq!(payload[payload_schema::MEMORY_ID].as_str(), Some("m1"));
    }

    #[test]
    fn test_embedder_info_from_fake() {
        let core = make_fake_core(1);
        let info = core.embedder_info();
        assert_eq!(info.id, "fake");
        assert_eq!(info.dimension, 128);
    }

    #[test]
    fn test_search_filter_to_qdrant_must_should_must_not() {
        assert!(search_filter_to_qdrant(&SearchFilter::default()).is_none());

        let f = SearchFilter {
            source: Some("s1".into()),
            source_type: Some("text".into()),
            project: Some("p".into()),
            tags: Some(vec!["a".into(), "b".into()]),
            tags_should: Some(vec!["c".into()]),
            tags_must_not: Some(vec!["d".into()]),
            scope: Some("team-a".into()),
            max_clearance: Some("internal".into()),
        };
        let qf = search_filter_to_qdrant(&f).expect("filter");
        // Decomposition of must conditions for a full filter:
        //   1 source + 1 source_type + 1 project + 2 tags (AND) + 1 scope + 1 clearance
        let expected_must = 1 + 1 + 1 + 2 + 1 + 1;
        assert_eq!(
            qf.must.len(),
            expected_must,
            "must conditions: source(1)+source_type(1)+project(1)+tags(2)+scope(1)+clearance(1)"
        );
        assert_eq!(qf.should.len(), 1, "tags_should contributes one should");
        assert_eq!(
            qf.must_not.len(),
            1,
            "tags_must_not contributes one must_not"
        );

        // Scope-only excludes empty filter
        let scoped = search_filter_to_qdrant(&SearchFilter {
            scope: Some("only-me".into()),
            ..Default::default()
        })
        .expect("scope filter");
        assert_eq!(scoped.must.len(), 1);

        // Unknown max_clearance adds no clearance condition (strict pure helper rejects in unit tests)
        let bad_clear = search_filter_to_qdrant(&SearchFilter {
            max_clearance: Some("nope".into()),
            ..Default::default()
        });
        // empty allowed list → no condition → filter still empty of useful must
        assert!(bad_clear.is_none());
    }

    #[tokio::test]
    async fn test_search_page_limit_zero() {
        let core = make_fake_core(2);
        // Does not need live Qdrant — early return for limit 0.
        let page = core
            .search_page(
                "q",
                SearchOptions {
                    limit: Some(0),
                    ..Default::default()
                },
            )
            .await
            .expect("limit 0");
        assert!(page.results.is_empty());
        assert!(!page.has_more);
        assert_eq!(page.limit, 0);
    }

    #[test]
    fn test_chunk_text_method() {
        let embedder = Box::new(FakeEmbedder::new(128));
        let core = RagCore::new(make_test_qdrant_client(), embedder, Some(2));
        let chunks = core.chunk_text_method("short text");
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0], "short text");
    }

    #[test]
    fn expand_to_chunks_sets_indices() {
        let core = make_fake_core(1);
        let chunks = core.expand_to_chunks(
            "# A\n\none\n\n# B\n\ntwo",
            Some("doc.md".into()),
            Some("text".into()),
            "default".into(),
            None,
            None,
            None,
            Some("doc.md"),
            None,
            None,
        );
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].chunk_index, Some(0));
        assert_eq!(chunks[0].total_chunks, Some(chunks.len()));
        assert_eq!(chunks.last().unwrap().chunk_index, Some(chunks.len() - 1));
        assert_eq!(chunks[0].source.as_deref(), Some("doc.md"));
        assert_eq!(chunks[0].collection.as_deref(), Some("default"));
    }

    #[test]
    fn resolve_collection_and_helpers() {
        assert_eq!(
            crate::types::resolve_collection(None),
            crate::types::DEFAULT_COLLECTION_NAME
        );
        assert_eq!(crate::types::resolve_collection(Some("x".into())), "x");
        let fr = crate::types::make_file_result("p", true, None, 3);
        assert_eq!(fr["ok"], true);
        assert_eq!(fr["chunks"], 3);
        assert!(!crate::types::unix_now_secs_str().is_empty());
        assert!(!crate::types::INDEX_FIELDS.contains(&"collection"));
    }

    #[test]
    fn chunk_config_accessors() {
        let core = make_fake_core(1);
        assert!(core.auto_index());
        assert_eq!(
            core.chunk_config().chunk_size,
            crate::constants::DEFAULT_CHUNK_SIZE
        );
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
        let result = core
            .ensure_collection("lqm_core_test_ensure", Some(128))
            .await;
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

    /// Rare-token hybrid recall for a given keyword backend (skips if Qdrant down).
    async fn hybrid_rare_token_smoke(backend: HybridKeywordBackend, coll: &str) {
        let core = make_fake_core(2).with_hybrid_keyword_backend(backend);
        let _ = core.delete_collection(coll).await;
        match core.create_collection(coll, None).await {
            Ok(_) => {}
            Err(e) => {
                eprintln!("Qdrant hybrid smoke skipped for {backend} (likely not running): {e:?}");
                return;
            }
        }

        let rare = "rarekeytokenxqzv9f3a";
        let chunks = vec![
            DocumentChunk::simple(
                "Fluffy clouds drift over the quiet mountain lake at dawn.",
                Some("smoke://hybrid-generic".into()),
                Some("text".into()),
                Some(coll.into()),
            ),
            DocumentChunk::simple(
                format!(
                    "Agent notes about Liberado retrieval and the token {rare} for hybrid smoke."
                ),
                Some("smoke://hybrid-keyword".into()),
                Some("text".into()),
                Some(coll.into()),
            ),
        ];
        core.embed_and_upsert_batch(chunks)
            .await
            .expect("ingest hybrid smoke docs");

        let page = core
            .search_page(
                rare,
                SearchOptions {
                    collection: Some(coll.into()),
                    limit: Some(5),
                    hybrid: true,
                    hybrid_alpha: Some(0.35),
                    ..Default::default()
                },
            )
            .await
            .expect("hybrid search");
        assert!(!page.results.is_empty(), "hybrid ({backend}) expected hits");
        let joined: String = page
            .results
            .iter()
            .map(|r| r.text.as_str())
            .collect::<Vec<_>>()
            .join(" | ");
        assert!(
            joined.contains(rare),
            "hybrid ({backend}) must surface rare token; got: {joined}"
        );
        let _ = core.delete_collection(coll).await;
    }

    #[tokio::test]
    async fn test_hybrid_keyword_index_live_smoke() {
        hybrid_rare_token_smoke(
            HybridKeywordBackend::KeywordIndex,
            "lqm_core_hybrid_kw_index",
        )
        .await;
    }

    #[tokio::test]
    async fn test_hybrid_sparse_live_smoke() {
        hybrid_rare_token_smoke(HybridKeywordBackend::Sparse, "lqm_core_hybrid_sparse").await;
    }

    #[tokio::test]
    async fn test_hybrid_scroll_live_smoke() {
        hybrid_rare_token_smoke(HybridKeywordBackend::Scroll, "lqm_core_hybrid_scroll").await;
    }

    #[test]
    fn test_hybrid_keyword_backend_default_is_keyword_index() {
        // Default on RagCore::new is KeywordIndex (scalable, no recreate required).
        let core = make_fake_core(1);
        assert_eq!(
            core.hybrid_keyword_backend(),
            HybridKeywordBackend::KeywordIndex
        );
    }
}
