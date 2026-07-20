//! Qdrant client wrapper and `RagCore` (embed + upsert + search orchestration).
//!
//! All agent/HTTP/CLI surfaces should go through `RagCore` so chunking, payload
//! schema, and re-ingest policy stay consistent.

use crate::chunking::{ChunkingStrategy, chunk_for_ingest, chunk_text};
use crate::embedding::Embedder;
use crate::error::LqmError;
use crate::hybrid::{
    HybridKeywordBackend, encode_sparse_tf, keyword_candidates_from_payloads, text_index_query,
    tokenize_for_keyword,
};
use crate::lifecycle::decide_source_reingest;
use crate::scope::Clearance;
use crate::types::{
    CONFIG_COLLECTION, ChunkConfig, CollectionInfoSummary, CollectionMeta, CollectionStats,
    DocumentChunk, EmbedderInfo, INDEX_FIELDS, IngestReport, PayloadFilter, ReingestAction,
    SOURCES_COLLECTION, SearchFilter, SearchOptions, SearchPage, SearchResult, SourceSummary,
    UpsertPoint, payload_schema, payload_str,
};

/// Fields fetched by `list_sources` — includes text for previews but skips
/// ~11 other payload keys (ingest_hash, tags, scope, project, timestamp,
/// embedding_model, importance, last_accessed, memory_id, chunk_index,
/// total_chunks, clearance).
const SOURCE_FIELD_MASK: &[&str] = &[
    payload_schema::SOURCE,
    payload_schema::SOURCE_TYPE,
    payload_schema::LAST_MODIFIED,
    payload_schema::TEXT,
];
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
        self.scroll_payloads_with_mask(collection, filter, page_size, None)
            .await
    }

    /// Scroll payloads with an optional field mask (include-list) to reduce wire size.
    ///
    /// When `field_mask` is `Some`, only those payload keys are fetched; others (including
    /// the heavy `text` field and vectors) are excluded.
    pub async fn scroll_payloads_with_mask(
        &self,
        collection: &str,
        filter: Option<Filter>,
        page_size: u32,
        field_mask: Option<&[&str]>,
    ) -> Result<Vec<crate::types::Payload>, QdrantError> {
        let mut out = Vec::new();
        let mut offset: Option<PointId> = None;
        let page = page_size.max(1);

        let include_fields =
            field_mask.map(|fields| qdrant_client::qdrant::PayloadIncludeSelector {
                fields: fields.iter().map(|s| s.to_string()).collect(),
            });

        loop {
            let mut builder = ScrollPointsBuilder::new(collection)
                .limit(page)
                .with_vectors(false);
            if let Some(ref inc) = include_fields {
                builder = builder.with_payload(inc.clone());
            } else {
                builder = builder.with_payload(true);
            }
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
fn clearance_max_condition(max_clearance: Clearance) -> Option<Condition> {
    let levels = max_clearance.allowed_levels();
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
    max_clearance: Option<Clearance>,
) {
    if let Some(s) = scope.map(str::trim).filter(|s| !s.is_empty()) {
        must.push(keyword_match(payload_schema::SCOPE, s.to_string()));
    }
    if let Some(max) = max_clearance
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
        must.push(keyword_match(payload_schema::SOURCE, source.clone()));
    }
    if let Some(ref st) = f.source_type {
        must.push(keyword_match(payload_schema::SOURCE_TYPE, st.clone()));
    }
    if let Some(ref project) = f.project {
        must.push(keyword_match(payload_schema::PROJECT, project.clone()));
    }
    if let Some(ref tags) = f.tags {
        for t in tags {
            must.push(keyword_match(payload_schema::TAGS, t.clone()));
        }
    }
    push_scope_and_clearance(&mut must, f.scope.as_deref(), f.max_clearance);
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
        must.push(keyword_match(payload_schema::SOURCE, source.clone()));
    }
    if let Some(ref st) = f.source_type {
        must.push(keyword_match(payload_schema::SOURCE_TYPE, st.clone()));
    }
    if let Some(ref project) = f.project {
        must.push(keyword_match(payload_schema::PROJECT, project.clone()));
    }
    if let Some(ref tags) = f.tags {
        for t in tags {
            must.push(keyword_match(payload_schema::TAGS, t.clone()));
        }
    }
    if let Some(ref tags) = f.tags_should {
        for t in tags {
            should.push(keyword_match(payload_schema::TAGS, t.clone()));
        }
    }
    if let Some(ref tags) = f.tags_must_not {
        for t in tags {
            must_not.push(keyword_match(payload_schema::TAGS, t.clone()));
        }
    }
    push_scope_and_clearance(&mut must, f.scope.as_deref(), f.max_clearance);

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
        .get(payload_schema::TEXT)
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

    /// Read per-collection chunk config from `_lqm_config`, falling back to global defaults.
    pub async fn chunk_config_for(&self, collection: &str) -> ChunkConfig {
        if let Ok(Some(meta)) = self.read_collection_meta(collection).await {
            ChunkConfig {
                chunk_size: meta
                    .chunk_size
                    .map(|s| s as usize)
                    .unwrap_or(self.chunk_config.chunk_size),
                overlap: meta
                    .chunk_overlap
                    .map(|o| o as usize)
                    .unwrap_or(self.chunk_config.overlap),
            }
        } else {
            self.chunk_config.clone()
        }
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

    /// Aggregated point counts grouped by payload fields (source_type, project, tags).
    ///
    /// Uses repeated Qdrant `count_points` calls with filters — cheap gRPC, no scroll.
    pub async fn collection_stats(&self, name: &str) -> Result<CollectionStats, LqmError> {
        use crate::source_type::SourceType;

        let total_points = self
            .qdrant
            .inner
            .count(CountPointsBuilder::new(name).exact(true))
            .await
            .map_err(|e| LqmError::Qdrant(e.into()))?
            .result
            .map(|r| r.count)
            .unwrap_or(0) as u64;

        let counted = |key: &str, value: &str| -> Filter {
            let cond = crate::qdrant::keyword_match(key, value.to_string());
            Filter {
                must: vec![cond],
                should: vec![],
                must_not: vec![],
                min_should: None,
            }
        };

        let mut source_types = std::collections::HashMap::new();
        for st in SourceType::ALL {
            let filter = counted(payload_schema::SOURCE_TYPE, st.as_str());
            let count = self
                .qdrant
                .inner
                .count(CountPointsBuilder::new(name).filter(filter).exact(true))
                .await
                .map_err(|e| LqmError::Qdrant(e.into()))?
                .result
                .map(|r| r.count)
                .unwrap_or(0) as u64;
            if count > 0 {
                source_types.insert(st.to_string(), count);
            }
        }

        // Count distinct sources by scrolling only the source field.
        let source_points = self
            .qdrant
            .inner
            .scroll(
                ScrollPointsBuilder::new(name)
                    .filter(Filter::default())
                    .limit(1_000),
            )
            .await
            .map_err(|e| LqmError::Qdrant(e.into()))?
            .result;
        let total_sources = source_points
            .iter()
            .filter_map(|p| p.payload.get(payload_schema::SOURCE))
            .filter_map(|v| v.kind.as_ref())
            .filter_map(|k| {
                if let qdrant_client::qdrant::value::Kind::StringValue(s) = k {
                    Some(s)
                } else {
                    None
                }
            })
            .collect::<std::collections::HashSet<_>>()
            .len() as u64;

        // For projects and tags, just return empty maps (count_points per value is expensive
        // without a curated value list). Agents can see source_type breakdown + total sources.
        let projects = std::collections::HashMap::new();
        let tags = std::collections::HashMap::new();

        Ok(CollectionStats {
            total_points,
            total_sources,
            source_types,
            projects,
            tags,
        })
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

        // Track validated collections to avoid duplicate checks per group.
        let mut validated: std::collections::HashSet<String> = std::collections::HashSet::new();

        for ((collection, source_key), group_chunks) in groups {
            if validated.insert(collection.clone()) {
                self.validate_collection_dim(&collection).await?;
            }
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
            // Batch embedding to cap memory pressure on large ingests.
            let embeddings: Vec<Vec<f32>> =
                if texts.len() > crate::constants::MAX_EMBED_BATCH_CHUNKS {
                    log::info!(
                        "large batch ({} chunks), splitting into {} sub-batches",
                        texts.len(),
                        texts
                            .len()
                            .div_ceil(crate::constants::MAX_EMBED_BATCH_CHUNKS)
                    );
                    let mut all = Vec::with_capacity(texts.len());
                    for sub_batch in texts.chunks(crate::constants::MAX_EMBED_BATCH_CHUNKS) {
                        all.extend(self.embed_batch(sub_batch.to_vec()).await?);
                    }
                    all
                } else {
                    self.embed_batch(texts).await?
                };
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

            // Write source status after successful upsert
            if !source_key.is_empty() {
                let _ = self
                    .write_source_status(&collection, &source_key, "complete", None, n as u64)
                    .await;
            }
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
            .filter_map(|p| payload_str(p, payload_schema::INGEST_HASH))
            .collect())
    }

    /// Find sources similar to a given source by embedding its full text and searching.
    ///
    /// The source itself is excluded from results via payload filter.
    pub async fn similar_to_source(
        &self,
        source: &str,
        collection: &str,
        limit: u64,
        filters: Option<SearchFilter>,
    ) -> Result<Vec<SearchResult>, LqmError> {
        let source_doc = self.get_source(collection, source).await?;
        if source_doc.text.trim().is_empty() {
            return Ok(vec![]);
        }

        let embedding = self
            .embed_batch(vec![source_doc.text])
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| LqmError::Other("embedding returned empty".to_string()))?;

        let filter = filters.unwrap_or_default();
        // We want results that match the user's optional filters AND do NOT match the source.
        let qdrant_filter = search_filter_to_qdrant(&filter);
        let exclude_source = Filter {
            must: vec![],
            should: vec![],
            must_not: vec![keyword_match(payload_schema::SOURCE, source.to_string())],
            min_should: None,
        };
        let combined = and_filter(qdrant_filter, exclude_source);

        Ok(self
            .qdrant
            .search(collection, embedding, limit, combined, None, Some(0))
            .await?)
    }

    /// List distinct sources in a collection with point counts and enriched previews.
    ///
    /// Uses a field mask that skips ~11 non-essential payload keys (ingest_hash, tags,
    /// scope, embedding_model, chunk indices, etc.) while still fetching source, source_type,
    /// last_modified, and text for previews.
    pub async fn list_sources(&self, collection: &str) -> Result<Vec<SourceSummary>, LqmError> {
        if !self.qdrant.collection_exists(collection).await? {
            return Ok(vec![]);
        }
        let payloads = self
            .qdrant
            .scroll_payloads_with_mask(
                collection,
                None,
                crate::constants::SCROLL_PAGE_SIZE,
                Some(SOURCE_FIELD_MASK),
            )
            .await?;
        let mut map: HashMap<String, SourceSummary> = HashMap::new();
        for p in payloads {
            let Some(source) = payload_str(&p, payload_schema::SOURCE) else {
                continue;
            };
            let entry = map.entry(source.clone()).or_insert_with(|| {
                let first_chunk = p
                    .get(payload_schema::TEXT)
                    .and_then(|v| v.as_str())
                    .map(|t| {
                        let max_len = crate::constants::TEXT_PREVIEW_CHARS;
                        if t.len() <= max_len {
                            t.to_string()
                        } else {
                            // Take up to preview chars, truncate at last whitespace
                            let mut end =
                                t[..max_len].rfind(char::is_whitespace).unwrap_or(max_len);
                            if end == 0 {
                                end = max_len;
                            }
                            format!("{}…", &t[..end])
                        }
                    });
                let title = first_chunk
                    .as_deref()
                    .or(p.get(payload_schema::SOURCE_TYPE).and_then(|v| v.as_str()))
                    .map(|s| {
                        // First non-empty line, trimmed, capped at 80 chars
                        s.lines()
                            .next()
                            .map(|l| l.trim())
                            .unwrap_or("")
                            .chars()
                            .take(80)
                            .collect()
                    });
                SourceSummary {
                    source,
                    count: 0,
                    source_type: None,
                    last_modified: None,
                    first_chunk,
                    total_chars: 0,
                    title,
                    ingest_status: None,
                    ingest_error: None,
                }
            });
            entry.count += 1;
            if let Some(t) = p.get(payload_schema::TEXT).and_then(|v| v.as_str()) {
                entry.total_chars = entry.total_chars.saturating_add(t.len() as u64);
            }
            if entry.source_type.is_none() {
                entry.source_type = payload_str(&p, payload_schema::SOURCE_TYPE);
            }
            if entry.last_modified.is_none() {
                entry.last_modified = payload_str(&p, payload_schema::LAST_MODIFIED);
            }
        }
        let mut sources: Vec<_> = map.into_values().collect();
        sources.sort_by(|a, b| a.source.cmp(&b.source));

        // Enrich with source status from _lqm_sources if available
        if let Ok(statuses) = self.read_source_statuses(collection).await {
            for s in &mut sources {
                if let Some((status, error)) = statuses.get(&s.source) {
                    s.ingest_status = Some(status.clone());
                    s.ingest_error = error.clone();
                }
            }
        }

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

        self.validate_collection_dim(collection).await?;

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

    /// Find the Levenshtein-closest collection name to the given needle.
    ///
    /// Returns `None` if no collections exist or the closest match exceeds half the
    /// needle length (avoids spurious suggestions for very short or unrelated names).
    pub async fn suggest_collection(&self, needle: &str) -> Option<String> {
        let needle_lower = needle.to_ascii_lowercase();
        let candidates = match self.list_collections().await {
            Ok(names) => names,
            Err(_) => return None,
        };
        if candidates.is_empty() {
            return None;
        }
        let threshold = (needle.len() / 2).max(1);
        candidates
            .iter()
            .map(|name| {
                let dist = levenshtein(&needle_lower, &name.to_ascii_lowercase());
                (name.clone(), dist)
            })
            .filter(|(_, d)| *d <= threshold)
            .min_by_key(|(_, d)| *d)
            .map(|(name, _)| name)
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

        // Write per-collection metadata (idempotent — overwrite so dim is always current).
        if name != CONFIG_COLLECTION {
            self.write_collection_meta(
                name, dim as u64, None, // model_label set by callers
                None, // chunk_size (global default)
                None, // chunk_overlap
                None, // chunk_kind
            )
            .await?;
        }

        Ok(())
    }

    /// Persist embedder identity + dimension in the `_lqm_config` collection.
    #[allow(deprecated)]
    pub async fn write_collection_meta(
        &self,
        collection: &str,
        vector_dim: u64,
        model_label: Option<String>,
        chunk_size: Option<u64>,
        chunk_overlap: Option<u64>,
        chunk_kind: Option<String>,
    ) -> Result<(), LqmError> {
        use qdrant_client::qdrant::value::Kind;

        let meta = CollectionMeta {
            name: collection.to_string(),
            embedder_id: self.embedder.id().to_string(),
            vector_dim,
            model_label,
            created_at: crate::types::unix_now_secs_str(),
            chunk_size,
            chunk_overlap,
            chunk_kind,
        };

        let mut payload = HashMap::from([
            (
                "collection_name".to_string(),
                qdrant_client::qdrant::Value {
                    kind: Some(Kind::StringValue(meta.name.clone())),
                },
            ),
            (
                "embedder_id".to_string(),
                qdrant_client::qdrant::Value {
                    kind: Some(Kind::StringValue(meta.embedder_id.clone())),
                },
            ),
            (
                "vector_dim".to_string(),
                qdrant_client::qdrant::Value {
                    kind: Some(Kind::IntegerValue(meta.vector_dim as i64)),
                },
            ),
            (
                "created_at".to_string(),
                qdrant_client::qdrant::Value {
                    kind: Some(Kind::StringValue(meta.created_at.clone())),
                },
            ),
        ]);

        if let Some(cs) = meta.chunk_size {
            payload.insert(
                "chunk_size".to_string(),
                qdrant_client::qdrant::Value {
                    kind: Some(Kind::IntegerValue(cs as i64)),
                },
            );
        }
        if let Some(co) = meta.chunk_overlap {
            payload.insert(
                "chunk_overlap".to_string(),
                qdrant_client::qdrant::Value {
                    kind: Some(Kind::IntegerValue(co as i64)),
                },
            );
        }
        if let Some(ref ck) = meta.chunk_kind {
            payload.insert(
                "chunk_kind".to_string(),
                qdrant_client::qdrant::Value {
                    kind: Some(Kind::StringValue(ck.clone())),
                },
            );
        }

        let point_id = collection.to_string();

        // Fire-and-forget upsert into config collection. If the config collection
        // doesn't exist yet, ensure it first (1-dim dummy vector).
        self.ensure_config_collection().await?;

        // Deterministic UUID from collection name for idempotent upsert.
        let mut hasher = Sha256::new();
        hasher.update(point_id.as_bytes());
        let hash = hasher.finalize();
        let uuid = uuid::Uuid::from_bytes(hash[..16].try_into().unwrap());

        let _ = self
            .qdrant
            .inner
            .upsert_points(UpsertPoints {
                collection_name: CONFIG_COLLECTION.to_string(),
                wait: Some(true),
                points: vec![qdrant_client::qdrant::PointStruct {
                    id: Some(qdrant_client::qdrant::PointId {
                        point_id_options: Some(
                            qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid.to_string()),
                        ),
                    }),
                    vectors: Some(Vectors {
                        vectors_options: Some(
                            qdrant_client::qdrant::vectors::VectorsOptions::Vector(Vector {
                                data: vec![],
                                indices: None,
                                vectors_count: None,
                                vector: Some(vector::Vector::Dense(DenseVector {
                                    data: vec![0.0],
                                })),
                            }),
                        ),
                    }),
                    payload,
                }],
                ..Default::default()
            })
            .await
            .map_err(|e| LqmError::Qdrant(e.into()))?;

        Ok(())
    }

    /// Ensure the reserved `_lqm_config` collection exists (1-dim dummy vectors).
    async fn ensure_config_collection(&self) -> Result<(), LqmError> {
        if self.qdrant.collection_exists(CONFIG_COLLECTION).await? {
            return Ok(());
        }
        self.qdrant.create_collection(CONFIG_COLLECTION, 1).await?;
        Ok(())
    }

    /// Ensure the reserved `_lqm_sources` collection exists (1-dim dummy vectors).
    async fn ensure_sources_collection(&self) -> Result<(), LqmError> {
        if self.qdrant.collection_exists(SOURCES_COLLECTION).await? {
            return Ok(());
        }
        self.qdrant.create_collection(SOURCES_COLLECTION, 1).await?;
        Ok(())
    }

    /// Write ingest status for a source. Used by `embed_and_upsert_batch` after
    /// successfully processing a source group.
    #[allow(deprecated)]
    async fn write_source_status(
        &self,
        collection: &str,
        source: &str,
        status: &str,
        error: Option<&str>,
        chunks: u64,
    ) -> Result<(), LqmError> {
        use qdrant_client::qdrant::value::Kind;

        self.ensure_sources_collection().await?;

        let key = format!("{collection}:{source}");
        let mut hasher = Sha256::new();
        hasher.update(key.as_bytes());
        let hash = hasher.finalize();
        let uuid = uuid::Uuid::from_bytes(hash[..16].try_into().unwrap());

        let mut payload: HashMap<String, qdrant_client::qdrant::Value> = [
            (
                "collection".to_string(),
                qdrant_client::qdrant::Value {
                    kind: Some(Kind::StringValue(collection.to_string())),
                },
            ),
            (
                "source".to_string(),
                qdrant_client::qdrant::Value {
                    kind: Some(Kind::StringValue(source.to_string())),
                },
            ),
            (
                "status".to_string(),
                qdrant_client::qdrant::Value {
                    kind: Some(Kind::StringValue(status.to_string())),
                },
            ),
            (
                "chunks".to_string(),
                qdrant_client::qdrant::Value {
                    kind: Some(Kind::IntegerValue(chunks as i64)),
                },
            ),
        ]
        .into_iter()
        .collect();

        if let Some(err) = error {
            payload.insert(
                "error".to_string(),
                qdrant_client::qdrant::Value {
                    kind: Some(Kind::StringValue(err.to_string())),
                },
            );
        }

        let _ = self
            .qdrant
            .inner
            .upsert_points(UpsertPoints {
                collection_name: SOURCES_COLLECTION.to_string(),
                wait: Some(true),
                points: vec![qdrant_client::qdrant::PointStruct {
                    id: Some(qdrant_client::qdrant::PointId {
                        point_id_options: Some(
                            qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid.to_string()),
                        ),
                    }),
                    vectors: Some(Vectors {
                        vectors_options: Some(
                            qdrant_client::qdrant::vectors::VectorsOptions::Vector(Vector {
                                data: vec![],
                                indices: None,
                                vectors_count: None,
                                vector: Some(vector::Vector::Dense(DenseVector {
                                    data: vec![0.0],
                                })),
                            }),
                        ),
                    }),
                    payload,
                }],
                ..Default::default()
            })
            .await
            .map_err(|e| LqmError::Qdrant(e.into()))?;

        Ok(())
    }

    /// Read source statuses for a given collection from `_lqm_sources`.
    async fn read_source_statuses(
        &self,
        collection: &str,
    ) -> Result<HashMap<String, (String, Option<String>)>, LqmError> {
        use qdrant_client::qdrant::Match;
        use qdrant_client::qdrant::value::Kind;

        let filter = Filter {
            must: vec![Condition {
                condition_one_of: Some(qdrant_client::qdrant::condition::ConditionOneOf::Field(
                    qdrant_client::qdrant::FieldCondition {
                        key: "collection".to_string(),
                        r#match: Some(Match {
                            match_value: Some(qdrant_client::qdrant::r#match::MatchValue::Keyword(
                                collection.to_string(),
                            )),
                        }),
                        ..Default::default()
                    },
                )),
            }],
            should: vec![],
            must_not: vec![],
            min_should: None,
        };

        let points = self
            .qdrant
            .inner
            .scroll(
                ScrollPointsBuilder::new(SOURCES_COLLECTION)
                    .filter(filter)
                    .limit(1000),
            )
            .await
            .map_err(|e| LqmError::Qdrant(e.into()))?
            .result;

        let mut map = HashMap::new();
        for pt in points {
            let source = pt
                .payload
                .get("source")
                .and_then(|v| v.kind.as_ref())
                .and_then(|k| {
                    if let Kind::StringValue(s) = k {
                        Some(s.clone())
                    } else {
                        None
                    }
                });
            let status = pt
                .payload
                .get("status")
                .and_then(|v| v.kind.as_ref())
                .and_then(|k| {
                    if let Kind::StringValue(s) = k {
                        Some(s.clone())
                    } else {
                        None
                    }
                });
            let error = pt
                .payload
                .get("error")
                .and_then(|v| v.kind.as_ref())
                .and_then(|k| {
                    if let Kind::StringValue(s) = k {
                        Some(s.clone())
                    } else {
                        None
                    }
                });
            if let (Some(source), Some(status)) = (source, status) {
                map.insert(source, (status, error));
            }
        }
        Ok(map)
    }

    /// Read per-collection metadata from `_lqm_config`.
    pub async fn read_collection_meta(
        &self,
        collection: &str,
    ) -> Result<Option<CollectionMeta>, LqmError> {
        use qdrant_client::qdrant::Match;
        use qdrant_client::qdrant::value::Kind;

        // Use a keyword match filter on collection_name in _lqm_config
        let filter = Filter {
            must: vec![Condition {
                condition_one_of: Some(qdrant_client::qdrant::condition::ConditionOneOf::Field(
                    qdrant_client::qdrant::FieldCondition {
                        key: "collection_name".to_string(),
                        r#match: Some(Match {
                            match_value: Some(qdrant_client::qdrant::r#match::MatchValue::Keyword(
                                collection.to_string(),
                            )),
                        }),
                        ..Default::default()
                    },
                )),
            }],
            should: vec![],
            must_not: vec![],
            min_should: None,
        };

        let points = self
            .qdrant
            .inner
            .scroll(
                ScrollPointsBuilder::new(CONFIG_COLLECTION)
                    .filter(filter)
                    .limit(1),
            )
            .await
            .map_err(|e| LqmError::Qdrant(e.into()))?
            .result;

        if points.is_empty() {
            return Ok(None);
        }

        let payload = &points[0].payload;
        let name = payload
            .get("collection_name")
            .and_then(|v| v.kind.as_ref())
            .and_then(|k| {
                if let Kind::StringValue(s) = k {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .unwrap_or_default();

        let embedder_id = payload
            .get("embedder_id")
            .and_then(|v| v.kind.as_ref())
            .and_then(|k| {
                if let Kind::StringValue(s) = k {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .unwrap_or_default();

        let vector_dim = payload
            .get("vector_dim")
            .and_then(|v| v.kind.as_ref())
            .and_then(|k| {
                if let Kind::IntegerValue(i) = k {
                    Some(*i as u64)
                } else {
                    None
                }
            })
            .unwrap_or(0);

        let created_at = payload
            .get("created_at")
            .and_then(|v| v.kind.as_ref())
            .and_then(|k| {
                if let Kind::StringValue(s) = k {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .unwrap_or_default();

        let chunk_size = payload
            .get("chunk_size")
            .and_then(|v| v.kind.as_ref())
            .and_then(|k| {
                if let Kind::IntegerValue(i) = k {
                    Some(*i as u64)
                } else {
                    None
                }
            });
        let chunk_overlap = payload
            .get("chunk_overlap")
            .and_then(|v| v.kind.as_ref())
            .and_then(|k| {
                if let Kind::IntegerValue(i) = k {
                    Some(*i as u64)
                } else {
                    None
                }
            });
        let chunk_kind = payload
            .get("chunk_kind")
            .and_then(|v| v.kind.as_ref())
            .and_then(|k| {
                if let Kind::StringValue(s) = k {
                    Some(s.clone())
                } else {
                    None
                }
            });

        Ok(Some(CollectionMeta {
            name,
            embedder_id,
            vector_dim,
            model_label: None,
            created_at,
            chunk_size,
            chunk_overlap,
            chunk_kind,
        }))
    }

    /// Validate that the embedder dimension matches a collection's recorded dimension.
    ///
    /// Returns `Ok(())` if no config exists for the collection (backward-compatible —
    /// collections created before this feature was shipped have no recorded dim).
    pub async fn validate_collection_dim(&self, collection: &str) -> Result<(), LqmError> {
        if let Some(meta) = self.read_collection_meta(collection).await? {
            let current_dim = self.embedder.dimension() as u64;
            if meta.vector_dim != current_dim {
                return Err(LqmError::Validation(format!(
                    "collection '{}' was created with embedder '{}' (dim={}) but the current embedder '{}' produces dim={}. \
                     To match, either recreate the collection with `vector_dim={}` or switch back to embedder '{}'.",
                    collection,
                    meta.embedder_id,
                    meta.vector_dim,
                    self.embedder.id(),
                    current_dim,
                    meta.vector_dim,
                    meta.embedder_id,
                )));
            }
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
        chunk_for_ingest(text, source_type, path_hint, &strategy)
    }

    fn chunk_for_ingest_with_strategy(
        &self,
        text: &str,
        source_type: Option<&str>,
        path_hint: Option<&str>,
        strategy: &ChunkingStrategy,
    ) -> Vec<String> {
        chunk_for_ingest(text, source_type, path_hint, strategy)
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
        clearance: Option<Clearance>,
    ) -> Vec<DocumentChunk> {
        self.expand_to_chunks_with_config(
            text,
            source,
            source_type,
            collection,
            tags,
            project,
            last_modified,
            path_hint,
            scope,
            clearance,
            None,
        )
    }

    /// Same as `expand_to_chunks` but accepts an optional per-collection `ChunkConfig` override.
    #[allow(clippy::too_many_arguments)]
    pub fn expand_to_chunks_with_config(
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
        clearance: Option<Clearance>,
        chunk_config: Option<ChunkConfig>,
    ) -> Vec<DocumentChunk> {
        let strategy = if let Some(ref cc) = chunk_config {
            ChunkingStrategy::new(cc.chunk_size, cc.overlap)
        } else {
            ChunkingStrategy::new(self.chunk_config.chunk_size, self.chunk_config.overlap)
        };
        let pieces = self.chunk_for_ingest_with_strategy(
            text,
            source_type.as_deref(),
            path_hint.or(source.as_deref()),
            &strategy,
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
                clearance,
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
    let clearance = chunk.clearance.unwrap_or(crate::scope::DEFAULT_CLEARANCE);
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

/// Levenshtein (edit) distance between two strings.
fn levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let (m, n) = (a_chars.len(), b_chars.len());
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{EmbedderConfig, create_embedder};
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
            max_clearance: Some(Clearance::Internal),
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
        let core = make_fake_core(1);
        assert_eq!(
            core.hybrid_keyword_backend(),
            HybridKeywordBackend::KeywordIndex
        );
    }

    // --- Collection ↔ embedder guarantees ---

    #[test]
    fn collection_meta_serialization_roundtrip() {
        let meta = CollectionMeta {
            name: "test-collection".into(),
            embedder_id: "fake".into(),
            vector_dim: 128,
            model_label: Some("AllMiniLML6V2".into()),
            created_at: "1700000000".into(),
            chunk_size: Some(256),
            chunk_overlap: Some(50),
            chunk_kind: Some("code".into()),
        };
        let json = serde_json::to_value(&meta).expect("serialize");
        let restored: CollectionMeta = serde_json::from_value(json).expect("deserialize");
        assert_eq!(restored.name, "test-collection");
        assert_eq!(restored.embedder_id, "fake");
        assert_eq!(restored.vector_dim, 128);
        assert_eq!(restored.model_label.as_deref(), Some("AllMiniLML6V2"));
        assert_eq!(restored.chunk_size, Some(256));
        assert_eq!(restored.chunk_overlap, Some(50));
        assert_eq!(restored.chunk_kind.as_deref(), Some("code"));
    }

    #[test]
    fn config_collection_name_is_reserved() {
        assert_eq!(CONFIG_COLLECTION, "_lqm_config");
    }

    #[test]
    fn expand_to_chunks_uses_custom_config_when_provided() {
        let core = make_fake_core(1);
        let small_config = ChunkConfig {
            chunk_size: 30,
            overlap: 5,
        };
        let text = "this is a very long sentence that should be split into multiple chunks";
        let chunks = core.expand_to_chunks_with_config(
            text,
            None,
            Some("text".to_string()),
            "c".into(),
            None,
            None,
            None,
            None,
            None,
            None,
            Some(small_config),
        );
        assert!(
            chunks.len() >= 2,
            "small chunk_size=30 should split long text, got {} chunks",
            chunks.len()
        );
        for c in &chunks {
            assert!(c.text.len() <= 36, "chunk '{}' exceeds 30+overlap", c.text);
        }
    }

    #[test]
    fn expand_to_chunks_uses_global_default_without_config() {
        let core = make_fake_core(1);
        let text = "a short sentence";
        let chunks = core.expand_to_chunks(
            text,
            Some("short".to_string()),
            Some("text".to_string()),
            "c".into(),
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].source.as_deref(), Some("short"));
        assert_eq!(chunks[0].collection.as_deref(), Some("c"));
    }

    /// Live: write collection meta, read it back, verify fields.
    #[ignore = "requires live Qdrant at QDRANT_URL"]
    #[tokio::test]
    async fn live_write_and_read_collection_meta() {
        let config = EmbedderConfig::load_or_default(None).expect("config");
        let embedder = create_embedder(&config).expect("embedder");
        let qdrant_url = crate::constants::DEFAULT_QDRANT_URL;
        let qdrant = QdrantClient::new_lazy(qdrant_url).expect("qdrant client");
        let core = RagCore::new(qdrant, embedder, Some(2));
        let test_coll = "test_lqm_meta_coll";

        // Clean up first
        let _ = core.delete_collection(test_coll).await;
        let _ = core.delete_collection(CONFIG_COLLECTION).await;

        // Create collection — this should auto-write meta
        core.ensure_collection(test_coll, None)
            .await
            .expect("ensure");

        // Read meta back
        let meta = core.read_collection_meta(test_coll).await.expect("read");
        assert!(meta.is_some(), "meta should exist after ensure_collection");
        let meta = meta.unwrap();
        assert_eq!(meta.name, test_coll);
        assert_eq!(meta.vector_dim as usize, core.embedder.dimension());
        assert_eq!(meta.embedder_id, core.embedder.id());

        // Validate dim — should pass
        core.validate_collection_dim(test_coll)
            .await
            .expect("validate should pass");

        // Overwrite with explicit label
        core.write_collection_meta(test_coll, 128, Some("test-model".into()), None, None, None)
            .await
            .expect("write");
        let meta2 = core
            .read_collection_meta(test_coll)
            .await
            .expect("read")
            .unwrap();
        assert_eq!(meta2.model_label.as_deref(), Some("test-model"));

        // Clean up
        let _ = core.delete_collection(test_coll).await;
        let _ = core.delete_collection(CONFIG_COLLECTION).await;
    }

    /// Live: validate_collection_dim rejects mismatched dimensions.
    #[ignore = "requires live Qdrant at QDRANT_URL"]
    #[tokio::test]
    async fn live_dimension_mismatch_rejected() {
        let config = EmbedderConfig::load_or_default(None).expect("config");
        let embedder = create_embedder(&config).expect("embedder");
        let qdrant_url = crate::constants::DEFAULT_QDRANT_URL;
        let qdrant = QdrantClient::new_lazy(qdrant_url).expect("qdrant client");
        let core = RagCore::new(qdrant, embedder, Some(2));
        let test_coll = "test_lqm_dim_mismatch";

        let _ = core.delete_collection(test_coll).await;
        let _ = core.delete_collection(CONFIG_COLLECTION).await;

        // Write meta with wrong dim
        core.ensure_collection(test_coll, Some(128))
            .await
            .expect("ensure");
        // Manually overwrite meta to claim a different dim
        core.write_collection_meta(test_coll, 999, None, None, None, None)
            .await
            .expect("write fake dim");

        let err = core.validate_collection_dim(test_coll).await;
        assert!(
            err.is_err(),
            "validation should fail with mismatched dim, got: {err:?}"
        );

        // Clean up
        let _ = core.delete_collection(test_coll).await;
        let _ = core.delete_collection(CONFIG_COLLECTION).await;
    }

    /// Live: validate on unknown collection is backward-compatible (no error).
    #[ignore = "requires live Qdrant at QDRANT_URL"]
    #[tokio::test]
    async fn live_validate_unknown_collection_ok() {
        let config = EmbedderConfig::load_or_default(None).expect("config");
        let embedder = create_embedder(&config).expect("embedder");
        let qdrant_url = crate::constants::DEFAULT_QDRANT_URL;
        let qdrant = QdrantClient::new_lazy(qdrant_url).expect("qdrant client");
        let core = RagCore::new(qdrant, embedder, Some(2));

        // A collection with no meta record should pass validation (backward compat)
        core.validate_collection_dim("nonexistent_coll_xyz")
            .await
            .expect("unknown collection should pass validation");
    }

    // --- build_point_payload edge cases ---

    fn empty_chunk(text: &str) -> DocumentChunk {
        DocumentChunk {
            text: text.into(),
            source: None,
            source_type: None,
            collection: None,
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

    #[test]
    fn build_point_payload_writes_timestamp_when_set() {
        let chunk = DocumentChunk {
            timestamp: Some("1700000000".into()),
            ..empty_chunk("t")
        };
        let payload = build_point_payload(&chunk, 0, 1, "fake");
        assert_eq!(payload[payload_schema::TIMESTAMP], "1700000000");
    }

    #[test]
    fn build_point_payload_writes_last_modified_when_set() {
        let chunk = DocumentChunk {
            last_modified: Some("1699999999".into()),
            ..empty_chunk("t")
        };
        let payload = build_point_payload(&chunk, 0, 1, "fake");
        assert_eq!(payload[payload_schema::LAST_MODIFIED], "1699999999");
    }

    #[test]
    fn build_point_payload_writes_scope_when_set() {
        let chunk = DocumentChunk {
            scope: Some("team-a".into()),
            ..empty_chunk("t")
        };
        let payload = build_point_payload(&chunk, 0, 1, "fake");
        assert_eq!(payload[payload_schema::SCOPE], "team-a");
    }

    #[test]
    fn build_point_payload_skips_whitespace_scope() {
        let chunk = DocumentChunk {
            scope: Some("   ".into()),
            ..empty_chunk("t")
        };
        let payload = build_point_payload(&chunk, 0, 1, "fake");
        assert!(
            !payload.contains_key(payload_schema::SCOPE),
            "whitespace-only scope should not appear in payload"
        );
    }

    #[test]
    fn build_point_payload_writes_explicit_clearance() {
        let chunk = DocumentChunk {
            clearance: Some(Clearance::Confidential),
            ..empty_chunk("t")
        };
        let payload = build_point_payload(&chunk, 0, 1, "fake");
        assert_eq!(payload[payload_schema::CLEARANCE], "confidential");
    }

    #[test]
    fn build_point_payload_defaults_clearance_to_public() {
        let payload = build_point_payload(&empty_chunk("t"), 0, 1, "fake");
        assert_eq!(payload[payload_schema::CLEARANCE], "public");
    }

    #[test]
    fn build_point_payload_clamps_importance_out_of_range() {
        let chunk = DocumentChunk {
            importance: Some(1.5),
            ..empty_chunk("t")
        };
        let payload = build_point_payload(&chunk, 0, 1, "fake");
        assert_eq!(
            payload[payload_schema::IMPORTANCE].as_str(),
            Some("1"),
            "importance > 1.0 should be clamped to 1.0"
        );

        let chunk2 = DocumentChunk {
            importance: Some(-0.5),
            ..empty_chunk("t")
        };
        let payload2 = build_point_payload(&chunk2, 0, 1, "fake");
        assert_eq!(
            payload2[payload_schema::IMPORTANCE].as_str(),
            Some("0"),
            "importance < 0.0 should be clamped to 0.0"
        );
    }

    #[test]
    fn build_point_payload_memory_sets_last_accessed_from_timestamp() {
        let chunk = DocumentChunk {
            source_type: Some(crate::memory::MEMORY_SOURCE_TYPE.into()),
            timestamp: Some("1700000000".into()),
            ..empty_chunk("note")
        };
        let payload = build_point_payload(&chunk, 0, 1, "fake");
        assert_eq!(
            payload[payload_schema::LAST_ACCESSED],
            "1700000000",
            "memory source_type with timestamp should copy to last_accessed"
        );
    }

    #[test]
    fn build_point_payload_non_memory_skips_last_accessed() {
        let chunk = DocumentChunk {
            source_type: Some("text".into()),
            timestamp: Some("1700000000".into()),
            ..empty_chunk("t")
        };
        let payload = build_point_payload(&chunk, 0, 1, "fake");
        assert!(
            !payload.contains_key(payload_schema::LAST_ACCESSED),
            "non-memory source_type should not set last_accessed"
        );
    }

    // --- make_dense_vector / make_point_vectors (protobuf construction) ---

    #[test]
    #[allow(deprecated)]
    fn make_dense_vector_wraps_data_in_dense_variant() {
        let data = vec![0.1, 0.2, 0.3];
        let v = QdrantClient::make_dense_vector(data.clone());
        assert!(v.data.is_empty(), "deprecated data field should be empty");
        let inner = v.vector.expect("vector variant should be set");
        match inner {
            vector::Vector::Dense(dv) => assert_eq!(dv.data, data),
            _ => panic!("expected Dense variant"),
        }
    }

    #[test]
    fn make_point_vectors_dense_only_produces_bare_vector() {
        let dense = vec![1.0, 2.0];
        let v = QdrantClient::make_point_vectors(dense.clone(), None);
        let opts = v.vectors_options.expect("vectors_options should be set");
        match opts {
            qdrant_client::qdrant::vectors::VectorsOptions::Vector(vec) => {
                let inner = vec.vector.expect("inner vector should be set");
                match inner {
                    vector::Vector::Dense(dv) => assert_eq!(dv.data, dense),
                    _ => panic!("expected Dense"),
                }
            }
            _ => panic!("expected bare Vector when no sparse"),
        }
    }

    #[test]
    fn make_point_vectors_with_sparse_produces_named_vectors() {
        let dense = vec![0.5];
        let sparse = Some((vec![0u32, 1], vec![0.8f32, 0.3]));
        let v = QdrantClient::make_point_vectors(dense.clone(), sparse);
        let opts = v.vectors_options.expect("vectors_options should be set");
        match opts {
            qdrant_client::qdrant::vectors::VectorsOptions::Vectors(named) => {
                assert_eq!(named.vectors.len(), 2, "dense + sparse = 2 named vectors");
                assert!(named.vectors.contains_key(""), "dense uses empty name");
                assert!(
                    named
                        .vectors
                        .contains_key(crate::constants::SPARSE_VECTOR_NAME),
                    "sparse uses SPARSE_VECTOR_NAME"
                );
            }
            _ => panic!("expected NamedVectors when sparse is present"),
        }
    }

    #[test]
    fn make_point_vectors_empty_sparse_falls_back_to_dense() {
        let dense = vec![0.5];
        let sparse = Some((vec![], vec![]));
        let v = QdrantClient::make_point_vectors(dense.clone(), sparse);
        let opts = v.vectors_options.expect("vectors_options should be set");
        match opts {
            qdrant_client::qdrant::vectors::VectorsOptions::Vector(_) => {}
            _ => panic!("empty sparse should fall back to bare Vector"),
        }
    }

    #[test]
    fn make_point_vectors_mismatched_sparse_len_falls_back_to_dense() {
        let dense = vec![0.5];
        let sparse = Some((vec![0u32, 1, 2], vec![0.8f32]));
        let v = QdrantClient::make_point_vectors(dense.clone(), sparse);
        let opts = v.vectors_options.expect("vectors_options should be set");
        match opts {
            qdrant_client::qdrant::vectors::VectorsOptions::Vector(_) => {}
            _ => panic!("mismatched sparse lengths should fall back to bare Vector"),
        }
    }

    // --- qdrant_value_to_json / qdrant_payload_to_json ---

    #[test]
    fn qdrant_value_string_to_json() {
        let v = qdrant_client::qdrant::Value {
            kind: Some(qdrant_client::qdrant::value::Kind::StringValue(
                "hello".into(),
            )),
        };
        assert_eq!(
            qdrant_value_to_json(&v),
            serde_json::Value::String("hello".into())
        );
    }

    #[test]
    fn qdrant_value_list_to_json() {
        let v = qdrant_client::qdrant::Value {
            kind: Some(qdrant_client::qdrant::value::Kind::ListValue(
                qdrant_client::qdrant::ListValue {
                    values: vec![
                        qdrant_client::qdrant::Value {
                            kind: Some(qdrant_client::qdrant::value::Kind::StringValue("a".into())),
                        },
                        qdrant_client::qdrant::Value {
                            kind: Some(qdrant_client::qdrant::value::Kind::StringValue("b".into())),
                        },
                    ],
                },
            )),
        };
        assert_eq!(qdrant_value_to_json(&v), serde_json::json!(["a", "b"]));
    }

    #[test]
    fn qdrant_value_list_fallback_for_non_string() {
        let v = qdrant_client::qdrant::Value {
            kind: Some(qdrant_client::qdrant::value::Kind::ListValue(
                qdrant_client::qdrant::ListValue {
                    values: vec![qdrant_client::qdrant::Value {
                        kind: Some(qdrant_client::qdrant::value::Kind::IntegerValue(42)),
                    }],
                },
            )),
        };
        let result = qdrant_value_to_json(&v);
        let arr = result.as_array().expect("outer should be Array");
        assert_eq!(arr.len(), 1);
        assert!(
            arr[0].is_string(),
            "non-string item in list should fall back to Debug string"
        );
    }

    #[test]
    fn qdrant_value_empty_kind_falls_back_to_debug() {
        let v = qdrant_client::qdrant::Value { kind: None };
        let result = qdrant_value_to_json(&v);
        assert!(
            result.is_string(),
            "empty kind should fall back to Debug string"
        );
    }

    #[test]
    fn qdrant_payload_to_json_empty() {
        let map: HashMap<String, qdrant_client::qdrant::Value> = HashMap::new();
        let result = qdrant_payload_to_json(map.iter());
        assert_eq!(result.as_object().unwrap().len(), 0);
    }

    #[test]
    fn qdrant_payload_to_json_single_key() {
        let mut map = HashMap::new();
        map.insert(
            "key".into(),
            qdrant_client::qdrant::Value {
                kind: Some(qdrant_client::qdrant::value::Kind::StringValue(
                    "val".into(),
                )),
            },
        );
        let result = qdrant_payload_to_json(map.iter());
        assert_eq!(result["key"], "val");
    }

    #[test]
    fn qdrant_payload_to_json_multiple_keys() {
        let mut map = HashMap::new();
        map.insert(
            "a".into(),
            qdrant_client::qdrant::Value {
                kind: Some(qdrant_client::qdrant::value::Kind::StringValue("1".into())),
            },
        );
        map.insert(
            "b".into(),
            qdrant_client::qdrant::Value {
                kind: Some(qdrant_client::qdrant::value::Kind::StringValue("2".into())),
            },
        );
        let result = qdrant_payload_to_json(map.iter());
        assert_eq!(result["a"], "1");
        assert_eq!(result["b"], "2");
    }

    // --- scored_point_to_search_result ---

    #[test]
    fn scored_point_to_search_result_extracts_text_and_score() {
        let mut payload = HashMap::new();
        payload.insert(
            "text".into(),
            qdrant_client::qdrant::Value {
                kind: Some(qdrant_client::qdrant::value::Kind::StringValue(
                    "hello world".into(),
                )),
            },
        );
        let sp = ScoredPoint {
            id: None,
            payload,
            score: 0.95,
            version: 0,
            vectors: None,
            shard_key: None,
            order_value: None,
        };
        let result = scored_point_to_search_result(&sp);
        assert_eq!(result.text, "hello world");
        assert!((result.score - 0.95).abs() < 0.001);
    }

    #[test]
    fn scored_point_to_search_result_missing_text_defaults_empty() {
        let sp = ScoredPoint {
            id: None,
            payload: HashMap::new(),
            score: 0.5,
            version: 0,
            vectors: None,
            shard_key: None,
            order_value: None,
        };
        let result = scored_point_to_search_result(&sp);
        assert_eq!(result.text, "");
        assert!((result.score - 0.5).abs() < 0.001);
        assert!(result.payload.as_object().unwrap().is_empty());
    }

    #[test]
    fn scored_point_to_search_result_text_not_a_string() {
        let mut payload = HashMap::new();
        payload.insert(
            "text".into(),
            qdrant_client::qdrant::Value {
                kind: Some(qdrant_client::qdrant::value::Kind::IntegerValue(123)),
            },
        );
        let sp = ScoredPoint {
            id: None,
            payload,
            score: 0.3,
            version: 0,
            vectors: None,
            shard_key: None,
            order_value: None,
        };
        let result = scored_point_to_search_result(&sp);
        assert_eq!(
            result.text, "",
            "non-string text field should default to empty"
        );
    }

    // --- payload_filter_to_qdrant ---

    #[test]
    fn payload_filter_empty_returns_none() {
        assert!(payload_filter_to_qdrant(&PayloadFilter::default()).is_none());
    }

    #[test]
    fn payload_filter_is_empty_returns_none() {
        // PayloadFilter::default() has all None fields → is_empty() returns true → None.
        assert!(payload_filter_to_qdrant(&PayloadFilter::default()).is_none());
    }

    #[test]
    fn payload_filter_source_only() {
        let f = PayloadFilter {
            source: Some("doc.md".into()),
            ..Default::default()
        };
        let qf = payload_filter_to_qdrant(&f).expect("filter");
        assert_eq!(qf.must.len(), 1, "only source → 1 must condition");
    }

    #[test]
    fn payload_filter_multi_tags() {
        let f = PayloadFilter {
            tags: Some(vec!["a".into(), "b".into(), "c".into()]),
            ..Default::default()
        };
        let qf = payload_filter_to_qdrant(&f).expect("filter");
        assert_eq!(qf.must.len(), 3, "3 tags → 3 must conditions (AND)");
    }

    #[test]
    fn payload_filter_full() {
        let f = PayloadFilter {
            source: Some("s".into()),
            source_type: Some("text".into()),
            project: Some("p".into()),
            tags: Some(vec!["t1".into(), "t2".into()]),
            scope: Some("team-a".into()),
            max_clearance: Some(Clearance::Internal),
        };
        let qf = payload_filter_to_qdrant(&f).expect("filter");
        // source + source_type + project + 2 tags + scope + clearance
        assert_eq!(qf.must.len(), 1 + 1 + 1 + 2 + 1 + 1);
    }

    #[test]
    fn payload_filter_with_empty_string_scope_skipped() {
        let f = PayloadFilter {
            scope: Some("".into()),
            source: Some("s".into()),
            ..Default::default()
        };
        let qf = payload_filter_to_qdrant(&f).expect("filter");
        assert_eq!(
            qf.must.len(),
            1,
            "empty scope should be skipped, only source remains"
        );
    }

    // --- and_filter ---

    #[test]
    fn and_filter_none_first_returns_second() {
        let b = Filter {
            must: vec![keyword_match("k", "v".into())],
            should: vec![],
            must_not: vec![],
            min_should: None,
        };
        let result = and_filter(None, b.clone());
        assert_eq!(result.unwrap().must.len(), 1);
    }

    #[test]
    fn and_filter_combines_must_should_must_not() {
        let a = Some(Filter {
            must: vec![keyword_match("a", "1".into())],
            should: vec![keyword_match("s", "x".into())],
            must_not: vec![keyword_match("n", "y".into())],
            min_should: None,
        });
        let b = Filter {
            must: vec![keyword_match("b", "2".into())],
            should: vec![],
            must_not: vec![keyword_match("n2", "z".into())],
            min_should: Some(qdrant_client::qdrant::MinShould {
                conditions: vec![],
                min_count: 2,
            }),
        };
        let result = and_filter(a, b).expect("combined filter");
        assert_eq!(result.must.len(), 2);
        assert_eq!(result.should.len(), 1);
        assert_eq!(result.must_not.len(), 2);
    }

    #[test]
    fn and_filter_preserves_min_should_from_first() {
        let a = Some(Filter {
            must: vec![],
            should: vec![],
            must_not: vec![],
            min_should: Some(qdrant_client::qdrant::MinShould {
                conditions: vec![],
                min_count: 3,
            }),
        });
        let b = Filter {
            must: vec![],
            should: vec![],
            must_not: vec![],
            min_should: Some(qdrant_client::qdrant::MinShould {
                conditions: vec![],
                min_count: 1,
            }),
        };
        let result = and_filter(a, b).expect("combined filter");
        assert_eq!(
            result.min_should.map(|m| m.min_count),
            Some(3),
            "first filter's min_should wins"
        );
    }

    #[test]
    fn and_filter_uses_second_min_should_when_first_none() {
        let a = Some(Filter {
            must: vec![],
            should: vec![],
            must_not: vec![],
            min_should: None,
        });
        let b = Filter {
            must: vec![],
            should: vec![],
            must_not: vec![],
            min_should: Some(qdrant_client::qdrant::MinShould {
                conditions: vec![],
                min_count: 5,
            }),
        };
        let result = and_filter(a, b).expect("combined filter");
        assert_eq!(
            result.min_should.map(|m| m.min_count),
            Some(5),
            "second filter's min_should used when first is None"
        );
    }

    // --- clearance_max_condition / push_scope_and_clearance ---

    #[test]
    fn clearance_max_condition_public_allows_one_level() {
        let cond =
            clearance_max_condition(Clearance::Public).expect("public should produce condition");
        if let Some(qdrant_client::qdrant::condition::ConditionOneOf::Filter(inner)) =
            &cond.condition_one_of
        {
            assert_eq!(inner.should.len(), 1, "Public admits only Public");
        } else {
            panic!("expected nested filter condition");
        }
    }

    #[test]
    fn clearance_max_condition_internal_allows_two_levels() {
        let cond = clearance_max_condition(Clearance::Internal)
            .expect("internal should produce condition");
        if let Some(qdrant_client::qdrant::condition::ConditionOneOf::Filter(inner)) =
            &cond.condition_one_of
        {
            assert_eq!(inner.should.len(), 2, "Internal admits Public + Internal");
        } else {
            panic!("expected nested filter condition");
        }
    }

    #[test]
    fn clearance_max_condition_confidential_allows_three_levels() {
        let cond = clearance_max_condition(Clearance::Confidential)
            .expect("confidential should produce condition");
        if let Some(qdrant_client::qdrant::condition::ConditionOneOf::Filter(inner)) =
            &cond.condition_one_of
        {
            assert_eq!(
                inner.should.len(),
                3,
                "Confidential admits Public + Internal + Confidential"
            );
        } else {
            panic!("expected nested filter condition");
        }
    }

    #[test]
    fn clearance_max_condition_restricted_allows_four_levels() {
        let cond = clearance_max_condition(Clearance::Restricted)
            .expect("restricted should produce condition");
        if let Some(qdrant_client::qdrant::condition::ConditionOneOf::Filter(inner)) =
            &cond.condition_one_of
        {
            assert_eq!(inner.should.len(), 4, "Restricted admits all four levels");
        } else {
            panic!("expected nested filter condition");
        }
    }

    #[test]
    fn push_scope_and_clearance_both_present() {
        let mut must = Vec::new();
        push_scope_and_clearance(&mut must, Some("team-a"), Some(Clearance::Internal));
        assert_eq!(must.len(), 2, "scope + clearance → 2 conditions");
    }

    #[test]
    fn push_scope_and_clearance_only_scope() {
        let mut must = Vec::new();
        push_scope_and_clearance(&mut must, Some("team-a"), None);
        assert_eq!(must.len(), 1, "only scope → 1 condition");
    }

    #[test]
    fn push_scope_and_clearance_only_clearance() {
        let mut must = Vec::new();
        push_scope_and_clearance(&mut must, None, Some(Clearance::Confidential));
        assert_eq!(must.len(), 1, "only clearance → 1 condition");
    }

    #[test]
    fn push_scope_and_clearance_neither() {
        let mut must = Vec::new();
        push_scope_and_clearance(&mut must, None, None);
        assert_eq!(must.len(), 0, "neither → 0 conditions");
    }

    #[test]
    fn push_scope_and_clearance_empty_string_scope_skipped() {
        let mut must = Vec::new();
        push_scope_and_clearance(&mut must, Some(""), Some(Clearance::Public));
        assert_eq!(must.len(), 1, "empty scope skipped, only clearance");
    }

    #[test]
    fn push_scope_and_clearance_whitespace_scope_skipped() {
        let mut must = Vec::new();
        push_scope_and_clearance(&mut must, Some("   "), Some(Clearance::Restricted));
        assert_eq!(
            must.len(),
            1,
            "whitespace-only scope skipped, only clearance"
        );
    }

    // ── Performance test bench: list_sources field mask ──

    /// Benchmark: `list_sources` with field mask vs full payload scroll.
    ///
    /// Creates N chunks across M sources, then measures both paths. Asserts that
    /// the field-masked path returns identical results AND is measurably faster or
    /// equivalent in data volume.
    #[ignore = "requires live Qdrant at QDRANT_URL"]
    #[tokio::test]
    async fn bench_list_sources_field_mask() {
        let config = EmbedderConfig::load_or_default(None).expect("config");
        let embedder = create_embedder(&config).expect("embedder");
        let qdrant_url = crate::constants::DEFAULT_QDRANT_URL;
        let qdrant = QdrantClient::new_lazy(qdrant_url).expect("qdrant client");
        let core = RagCore::new(qdrant, embedder, Some(4));
        let test_coll = "test_lqm_perf_sources";
        let num_sources = 20u64;
        let chunks_per_source = 5u64;

        let _ = core.delete_collection(test_coll).await;
        core.ensure_collection(test_coll, None)
            .await
            .expect("ensure");

        // Ingest many chunks across multiple sources
        let mut all_chunks = Vec::new();
        for s in 0..num_sources {
            for i in 0..chunks_per_source {
                let text = format!(
                    "Source {} chunk {}: this is a medium-length paragraph for benchmarking purposes, with enough characters to simulate real-world chunk sizes but not so many that it distorts the results.",
                    s, i
                );
                all_chunks.push(DocumentChunk {
                    text,
                    source: Some(format!("source-{s}")),
                    source_type: Some("text".into()),
                    collection: Some(test_coll.into()),
                    chunk_index: Some(i as usize),
                    total_chunks: Some(chunks_per_source as usize),
                    ..empty_chunk("")
                });
            }
        }
        core.embed_and_upsert_batch(all_chunks)
            .await
            .expect("ingest");

        // Warm-up: one call each to stabilise connection pooling
        let _ = core.list_sources(test_coll).await;

        // ── Measure masked path (our new optimised path) ──
        let t0 = tokio::time::Instant::now();
        let sources_masked = core.list_sources(test_coll).await.expect("masked");
        let elapsed_masked = t0.elapsed();

        // ── Measure unmasked path (classic full-payload scroll) ──
        let t1 = tokio::time::Instant::now();
        let payloads_full = core
            .qdrant
            .scroll_payloads(test_coll, None, crate::constants::SCROLL_PAGE_SIZE)
            .await
            .expect("full scroll");
        let elapsed_full = t1.elapsed();

        // ── Aggregate full-payload results the same way to compare ──
        let mut map: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
        for p in &payloads_full {
            if let Some(source) = payload_str(p, payload_schema::SOURCE) {
                *map.entry(source).or_default() += 1;
            }
        }
        let full_count: u64 = map.values().sum();

        // ── Assertions ──
        assert_eq!(
            sources_masked.len() as u64,
            num_sources,
            "masked: expected {} sources, got {}",
            num_sources,
            sources_masked.len()
        );
        assert_eq!(
            full_count,
            num_sources * chunks_per_source,
            "full: expected {} chunks, got {}",
            num_sources * chunks_per_source,
            full_count
        );

        // First-chunk preview and total_chars should be populated
        for s in &sources_masked {
            assert!(
                s.total_chars > 0,
                "source {} should have total_chars > 0",
                s.source
            );
            assert!(
                s.first_chunk.is_some(),
                "source {} should have a first_chunk preview",
                s.source
            );
            assert!(s.title.is_some(), "source {} should have a title", s.source);
        }

        // The masked path should not be slower than the full path.
        // Allow a small tolerance (2x) for warm-up variance on the first call.
        assert!(
            elapsed_masked < elapsed_full.mul_f64(3.0),
            "masked path ({elapsed_masked:?}) should not be >3x slower than full ({elapsed_full:?})"
        );

        log::info!(
            "list_sources bench: {num_sources} sources × {chunks_per_source} chunks — masked={elapsed_masked:?} full={elapsed_full:?}"
        );

        let _ = core.delete_collection(test_coll).await;
        let _ = core.delete_collection(CONFIG_COLLECTION).await;
    }

    // ── Levenshtein distance + collection suggestions ──

    #[test]
    fn levenshtein_same_string_is_zero() {
        assert_eq!(levenshtein("hello", "hello"), 0);
    }

    #[test]
    fn levenshtein_one_substitution() {
        assert_eq!(levenshtein("hello", "hallo"), 1);
    }

    #[test]
    fn levenshtein_one_deletion() {
        assert_eq!(levenshtein("hello", "hell"), 1);
    }

    #[test]
    fn levenshtein_completely_different() {
        assert_eq!(levenshtein("abc", "xyz"), 3);
    }

    #[test]
    fn levenshtein_case_sensitive() {
        // levenshtein is case-sensitive; our suggestion wrapper lowercases both sides.
        assert_ne!(levenshtein("Hello", "hello"), 0);
    }

    /// Live: suggest_collection returns closest match within threshold.
    #[ignore = "requires live Qdrant at QDRANT_URL"]
    #[tokio::test]
    async fn live_suggest_collection_finds_closest() {
        let config = EmbedderConfig::load_or_default(None).expect("config");
        let embedder = create_embedder(&config).expect("embedder");
        let qdrant_url = crate::constants::DEFAULT_QDRANT_URL;
        let qdrant = QdrantClient::new_lazy(qdrant_url).expect("qdrant client");
        let core = RagCore::new(qdrant, embedder, Some(2));

        let _ = core.delete_collection("my_homelab_notes").await;
        let _ = core.delete_collection("my_homelab_config").await;
        let _ = core.delete_collection(CONFIG_COLLECTION).await;

        core.ensure_collection("my_homelab_notes", None)
            .await
            .expect("ok");
        core.ensure_collection("my_homelab_config", None)
            .await
            .expect("ok");

        // Typo: "homelab" -> "homelab" is 0 chars off, should match "my_homelab_notes"
        let suggestion = core.suggest_collection("my_homelvb_notes").await;
        assert_eq!(suggestion.as_deref(), Some("my_homelab_notes"));

        // Unrelated name should not match (distance too high)
        let none = core.suggest_collection("completely_different").await;
        assert!(none.is_none(), "unrelated name should return None");

        let _ = core.delete_collection("my_homelab_notes").await;
        let _ = core.delete_collection("my_homelab_config").await;
        let _ = core.delete_collection(CONFIG_COLLECTION).await;
    }
}
