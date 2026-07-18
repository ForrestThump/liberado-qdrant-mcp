use crate::chunking::{ChunkingStrategy, chunk_text};
use crate::embedding::Embedder;
use crate::error::LqmError;
use crate::lifecycle::decide_source_reingest;
use crate::types::{
    ChunkConfig, CollectionInfoSummary, DocumentChunk, EmbedderInfo, INDEX_FIELDS, IngestReport,
    PayloadFilter, ReingestAction, SearchFilter, SearchOptions, SearchPage, SearchResult,
    SourceSummary, UpsertPoint, payload_schema,
};
use qdrant_client::Qdrant as QdrantGrpc;
use qdrant_client::qdrant::{
    CollectionStatus, Condition, CountPointsBuilder, CreateCollection,
    CreateFieldIndexCollectionBuilder, DeleteCollection, DeletePointsBuilder, DenseVector,
    Distance, FieldType, Filter, PointId, PointStruct, ScoredPoint, ScrollPointsBuilder,
    SearchPoints, UpsertPoints, Vector, VectorParams, Vectors, WithPayloadSelector, vector,
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
                out.push(qdrant_payload_to_json(&pt.payload));
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

/// Build a Qdrant filter for search (must / should / must_not tag modes).
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

fn qdrant_payload_to_json(
    payload: &HashMap<String, qdrant_client::qdrant::Value>,
) -> crate::types::Payload {
    serde_json::to_value(
        payload
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
                                .map(|item| match &item.kind {
                                    Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => {
                                        serde_json::Value::String(s.clone())
                                    }
                                    _ => serde_json::Value::String(format!("{:?}", item)),
                                })
                                .collect(),
                        )
                    }
                    _ => serde_json::Value::String(format!("{:?}", v)),
                };
                (k.clone(), json_val)
            })
            .collect::<HashMap<_, _>>(),
    )
    .unwrap_or_default()
}

fn payload_str(payload: &serde_json::Value, key: &str) -> Option<String> {
    payload
        .get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
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

            for (pos, (chunk, vector)) in group_chunks.into_iter().zip(embeddings).enumerate() {
                let chunk_index = chunk.chunk_index.unwrap_or(pos);
                let total_chunks = chunk.total_chunks.unwrap_or(group_len);
                let payload =
                    build_point_payload(&chunk, chunk_index, total_chunks, &embedding_model);

                points.push(UpsertPoint {
                    id: uuid::Uuid::new_v4().to_string(),
                    vector,
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
        let payloads = self.qdrant.scroll_payloads(collection, filter, 256).await?;
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
        let payloads = self.qdrant.scroll_payloads(collection, None, 256).await?;
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

    pub async fn delete_by_filter(
        &self,
        collection: &str,
        filter: &PayloadFilter,
    ) -> Result<u64, LqmError> {
        if filter.is_empty() {
            return Err(LqmError::Validation(
                "delete_by_filter requires at least one of source, source_type, project, tags"
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
                    },
                },
            )
            .await?;
        Ok(page.results)
    }

    /// Filtered semantic search with offset pagination (`has_more` / `next_offset`).
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
            "searching '{}' in '{}' (limit:{} offset:{})",
            query,
            collection,
            limit,
            offset
        );

        let query_embedding = self.embed_batch(vec![query.to_string()]).await?;
        let query_vector = query_embedding
            .into_iter()
            .next()
            .ok_or_else(|| LqmError::Other("embedding returned empty".to_string()))?;

        let filter = search_filter_to_qdrant(&opts.filter);

        // Fetch one extra hit to detect has_more without a separate count call.
        let fetch_limit = limit.saturating_add(1);
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

        Ok(SearchPage {
            results,
            offset,
            limit,
            has_more,
            next_offset,
        })
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

    /// Active embedder identity for agent/HTTP introspection.
    pub fn embedder_info(&self) -> EmbedderInfo {
        EmbedderInfo {
            id: self.embedder.id().to_string(),
            dimension: self.embedder.dimension(),
            model: self.embedder.model().map(|s| s.to_string()),
        }
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
        };
        let qf = search_filter_to_qdrant(&f).expect("filter");
        // source + source_type + project + 2 tags must
        assert_eq!(qf.must.len(), 5);
        assert_eq!(qf.should.len(), 1);
        assert_eq!(qf.must_not.len(), 1);
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
