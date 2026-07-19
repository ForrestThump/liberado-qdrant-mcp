use axum::{
    Router,
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::Json,
    routing::{get, post},
};
use clap::Parser;
use lqm_core::ErrorCode;
use lqm_core::constants;
use lqm_core::format_relevant_context_with;
use lqm_core::scope::Clearance;
use lqm_core::structured_error;
use lqm_core::types::{
    ContextOptions, DocumentChunk, PayloadFilter, SearchFilter, SearchOptions, make_file_result,
    resolve_collection, unix_now_secs_str,
};
use lqm_core::{DEFAULT_MEMORY_COLLECTION, MemoryNote, RagCore};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;

#[derive(Parser)]
#[command(name = "lqm-api", about = "LQM HTTP API Server")]
struct Cli {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    #[arg(long, default_value_t = 8080)]
    port: u16,

    #[arg(long)]
    qdrant_url: Option<String>,

    #[arg(long)]
    config: Option<String>,
}

#[derive(Clone)]
struct AppState {
    core: Arc<RagCore>,
    embed_dimension: usize,
    /// When set, require `Authorization: Bearer <token>` on `/api/*` routes.
    api_token: Option<String>,
}

#[derive(Deserialize)]
struct SearchBody {
    query: String,
    collection: Option<String>,
    limit: Option<u64>,
    offset: Option<u64>,
    tags: Option<Vec<String>>,
    tags_should: Option<Vec<String>>,
    tags_must_not: Option<Vec<String>>,
    source: Option<String>,
    source_type: Option<String>,
    project: Option<String>,
    min_score: Option<f32>,
    /// Dense + keyword fusion when true (default false = dense-only).
    hybrid: Option<bool>,
    hybrid_alpha: Option<f32>,
    /// Exact scope partition; excludes other scopes when set.
    scope: Option<String>,
    /// Max clearance: public | internal | confidential | restricted.
    max_clearance: Option<String>,
}

#[derive(Deserialize)]
struct ContextBody {
    query: String,
    collection: Option<String>,
    limit: Option<u64>,
    offset: Option<u64>,
    tags: Option<Vec<String>>,
    tags_should: Option<Vec<String>>,
    tags_must_not: Option<Vec<String>>,
    source: Option<String>,
    source_type: Option<String>,
    project: Option<String>,
    min_score: Option<f32>,
    max_chars_per_passage: Option<u64>,
    max_total_chars: Option<u64>,
    mmr: Option<bool>,
    mmr_lambda: Option<f32>,
    hybrid: Option<bool>,
    hybrid_alpha: Option<f32>,
    scope: Option<String>,
    max_clearance: Option<String>,
}

#[derive(Deserialize)]
struct IngestBody {
    text: String,
    source: Option<String>,
    source_type: Option<String>,
    collection: Option<String>,
    tags: Option<Vec<String>>,
    project: Option<String>,
    last_modified: Option<u64>,
    scope: Option<String>,
    clearance: Option<String>,
}

#[derive(Deserialize)]
struct CreateCollectionBody {
    name: String,
    vector_dim: Option<u64>,
}

#[derive(Deserialize)]
struct DeleteByFilterBody {
    source: Option<String>,
    source_type: Option<String>,
    project: Option<String>,
    tags: Option<Vec<String>>,
    scope: Option<String>,
    max_clearance: Option<String>,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
}

#[derive(Serialize)]
struct CollectionsResponse {
    collections: Vec<String>,
}

#[derive(Serialize)]
struct ErrorResponse {
    code: String,
    message: String,
    /// Back-compat: same as message for older clients.
    error: String,
}

fn map_lqm_err(e: lqm_core::error::LqmError) -> (StatusCode, Json<ErrorResponse>) {
    let s = structured_error(&e);
    let status = StatusCode::from_u16(lqm_core::http_status(&e))
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (
        status,
        Json(ErrorResponse {
            code: s.code.clone(),
            message: s.message.clone(),
            error: s.message,
        }),
    )
}

fn map_msg(
    code: ErrorCode,
    message: impl Into<String>,
    status: StatusCode,
) -> (StatusCode, Json<ErrorResponse>) {
    let message = message.into();
    (
        status,
        Json(ErrorResponse {
            code: code.to_string(),
            message: message.clone(),
            error: message,
        }),
    )
}

/// Expand text into structure-aware chunks with stable payload indices.
#[allow(clippy::too_many_arguments)]
fn expand_chunks(
    core: &RagCore,
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
    core.expand_to_chunks(
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
    )
}

fn timestamp_now() -> String {
    unix_now_secs_str()
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

async fn list_collections(
    State(state): State<AppState>,
) -> Result<Json<CollectionsResponse>, (StatusCode, Json<ErrorResponse>)> {
    let collections = state.core.list_collections().await.map_err(map_lqm_err)?;
    Ok(Json(CollectionsResponse { collections }))
}

async fn delete_collection(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    let deleted = state
        .core
        .delete_collection(&name)
        .await
        .map_err(map_lqm_err)?;
    if deleted {
        log::info!("Deleted collection '{}'", name);
    }
    Ok(StatusCode::NO_CONTENT)
}

async fn get_embedder_info(State(state): State<AppState>) -> Json<serde_json::Value> {
    let info = state.core.embedder_info();
    Json(serde_json::json!({
        "status": "ok",
        "id": info.id,
        "dimension": info.dimension,
        "model": info.model,
    }))
}

async fn search(
    State(state): State<AppState>,
    Json(body): Json<SearchBody>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let hybrid_on = body.hybrid.unwrap_or(false);
    let page = state
        .core
        .search_page(
            &body.query,
            SearchOptions {
                collection: body.collection,
                limit: body.limit,
                offset: body.offset,
                min_score: body.min_score,
                filter: SearchFilter {
                    source: body.source,
                    source_type: body.source_type,
                    project: body.project,
                    tags: body.tags,
                    tags_should: body.tags_should,
                    tags_must_not: body.tags_must_not,
                    scope: body.scope,
                    max_clearance: body.max_clearance.and_then(|s| s.parse().ok()),
                },
                hybrid: hybrid_on,
                hybrid_alpha: body.hybrid_alpha,
            },
        )
        .await
        .map_err(map_lqm_err)?;

    Ok(Json(serde_json::json!({
        "results": page.results,
        "offset": page.offset,
        "limit": page.limit,
        "has_more": page.has_more,
        "next_offset": page.next_offset,
        "hybrid": hybrid_on,
    })))
}

async fn get_relevant_context(
    State(state): State<AppState>,
    Json(body): Json<ContextBody>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let coll = resolve_collection(body.collection.clone());
    let page = state
        .core
        .search_page(
            &body.query,
            SearchOptions {
                collection: Some(coll.clone()),
                limit: body.limit.or(Some(8)),
                offset: body.offset,
                min_score: body.min_score,
                filter: SearchFilter {
                    source: body.source,
                    source_type: body.source_type,
                    project: body.project,
                    tags: body.tags,
                    tags_should: body.tags_should,
                    tags_must_not: body.tags_must_not,
                    scope: body.scope,
                    max_clearance: body.max_clearance.and_then(|s| s.parse().ok()),
                },
                hybrid: body.hybrid.unwrap_or(false),
                hybrid_alpha: body.hybrid_alpha,
            },
        )
        .await
        .map_err(map_lqm_err)?;

    let formatted = format_relevant_context_with(
        &body.query,
        &page.results,
        &ContextOptions {
            max_chars_per_passage: body.max_chars_per_passage.map(|n| n as usize),
            max_total_chars: body.max_total_chars.map(|n| n as usize),
            mmr: body.mmr.unwrap_or(false),
            mmr_lambda: body.mmr_lambda,
        },
    );

    Ok(Json(serde_json::json!({
        "status": "ok",
        "query": body.query,
        "collection": coll,
        "passage_count": formatted.passage_count,
        "truncated_by_budget": formatted.truncated_by_budget,
        "offset": page.offset,
        "limit": page.limit,
        "has_more": page.has_more,
        "next_offset": page.next_offset,
        "context": formatted.context,
        "sources": formatted.sources,
    })))
}

async fn ingest(
    State(state): State<AppState>,
    Json(body): Json<IngestBody>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let collection = resolve_collection(body.collection);
    let source = Some(
        body.source
            .unwrap_or_else(|| constants::DEFAULT_API_INGEST_SOURCE.to_string()),
    );
    let source_type = Some(
        body.source_type
            .unwrap_or_else(|| constants::SOURCE_TYPE_TEXT.to_string()),
    );
    let mut chunks = expand_chunks(
        &state.core,
        &body.text,
        source,
        source_type,
        collection.clone(),
        Some(body.tags.unwrap_or_default()),
        body.project,
        body.last_modified.map(|ts| ts.to_string()),
        None,
        body.scope,
        body.clearance.and_then(|s| s.parse().ok()),
    );
    let ts = timestamp_now();
    for c in &mut chunks {
        c.timestamp = Some(ts.clone());
    }

    state
        .core
        .ensure_collection(&collection, None)
        .await
        .map_err(map_lqm_err)?;

    let report = state
        .core
        .embed_and_upsert_batch(chunks)
        .await
        .map_err(map_lqm_err)?;

    Ok(Json(serde_json::json!({
        "status": "ok",
        "collection": collection,
        "inserted": report.inserted,
        "skipped": report.skipped,
        "replaced": report.replaced,
        "chunks": report.chunks,
        "chunks_upserted": report.chunks,
    })))
}

async fn create_collection(
    State(state): State<AppState>,
    Json(body): Json<CreateCollectionBody>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let dim = body.vector_dim.map(|d| d as usize);
    let created = state
        .core
        .create_collection(&body.name, dim)
        .await
        .map_err(map_lqm_err)?;
    Ok(Json(serde_json::json!({
        "status": "ok",
        "name": body.name,
        "created": created,
        "vector_dim": dim.unwrap_or(state.embed_dimension),
    })))
}

async fn get_collection_info(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    match state.core.get_collection_info(&name).await {
        Ok(Some(info)) => Ok(Json(serde_json::json!({
            "status": "ok",
            "exists": true,
            "name": info.name,
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "segments_count": info.segments_count,
            "collection_status": info.status,
            "vector_size": info.vector_size,
            "distance": info.distance,
        }))),
        Ok(None) => Ok(Json(serde_json::json!({
            "status": "ok",
            "exists": false,
            "name": name,
        }))),
        Err(e) => Err(map_lqm_err(e)),
    }
}

async fn list_sources(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let sources = state.core.list_sources(&name).await.map_err(map_lqm_err)?;
    Ok(Json(serde_json::json!({
        "status": "ok",
        "collection": name,
        "sources": sources,
    })))
}

#[derive(Deserialize)]
struct ListChunksQuery {
    offset: Option<u64>,
    limit: Option<u64>,
}

async fn list_chunks(
    State(state): State<AppState>,
    Path((name, source)): Path<(String, String)>,
    Query(q): Query<ListChunksQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let page = state
        .core
        .list_chunks(&name, &source, q.offset, q.limit)
        .await
        .map_err(map_lqm_err)?;
    Ok(Json(serde_json::json!({
        "status": "ok",
        "collection": name,
        "source": page.source,
        "chunks": page.chunks,
        "offset": page.offset,
        "limit": page.limit,
        "total": page.total,
        "has_more": page.has_more,
        "next_offset": page.next_offset,
    })))
}

async fn get_source(
    State(state): State<AppState>,
    Path((name, source)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let doc = state
        .core
        .get_source(&name, &source)
        .await
        .map_err(map_lqm_err)?;
    Ok(Json(serde_json::json!({
        "status": "ok",
        "collection": name,
        "source": doc.source,
        "source_type": doc.source_type,
        "total": doc.total,
        "text": doc.text,
        "chunks": doc.chunks,
    })))
}

#[derive(Deserialize)]
struct ExpandContextBody {
    source: String,
    chunk_index: u64,
    collection: Option<String>,
    neighbors: Option<u64>,
}

async fn expand_context(
    State(state): State<AppState>,
    Json(body): Json<ExpandContextBody>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let collection = resolve_collection(body.collection);
    let chunks = state
        .core
        .expand_context(&collection, &body.source, body.chunk_index, body.neighbors)
        .await
        .map_err(map_lqm_err)?;
    Ok(Json(serde_json::json!({
        "status": "ok",
        "collection": collection,
        "source": body.source,
        "chunk_index": body.chunk_index,
        "neighbors": body.neighbors.unwrap_or(lqm_core::DEFAULT_EXPAND_NEIGHBORS),
        "count": chunks.len(),
        "chunks": chunks,
    })))
}

async fn delete_by_source(
    State(state): State<AppState>,
    Path((name, source)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let deleted = state
        .core
        .delete_by_source(&name, &source)
        .await
        .map_err(map_lqm_err)?;
    Ok(Json(serde_json::json!({
        "status": "ok",
        "collection": name,
        "source": source,
        "deleted": deleted,
    })))
}

async fn delete_by_filter(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(body): Json<DeleteByFilterBody>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let filter = PayloadFilter {
        source: body.source,
        source_type: body.source_type,
        project: body.project,
        tags: body.tags,
        scope: body.scope,
        max_clearance: body.max_clearance.and_then(|s| s.parse().ok()),
    };
    let deleted = state
        .core
        .delete_by_filter(&name, &filter)
        .await
        .map_err(map_lqm_err)?;
    Ok(Json(serde_json::json!({
        "status": "ok",
        "collection": name,
        "deleted": deleted,
        "filter": filter,
    })))
}

#[derive(Deserialize)]
struct IngestPathBody {
    path: String,
    collection: Option<String>,
}

#[derive(Deserialize)]
struct IngestUrlBody {
    url: String,
    collection: Option<String>,
    tags: Option<Vec<String>>,
    source_type: Option<String>,
    project: Option<String>,
    source: Option<String>,
}

#[derive(Deserialize)]
struct IngestManyBody {
    collection: Option<String>,
    texts: Option<Vec<String>>,
    paths: Option<Vec<String>>,
    urls: Option<Vec<String>>,
    tags: Option<Vec<String>>,
    project: Option<String>,
    scope: Option<String>,
    clearance: Option<String>,
}

async fn ingest_path(
    State(state): State<AppState>,
    Json(body): Json<IngestPathBody>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let collection = resolve_collection(body.collection);
    state
        .core
        .ensure_collection(&collection, None)
        .await
        .map_err(map_lqm_err)?;

    let metadata = std::fs::metadata(&body.path).map_err(|e| {
        map_msg(
            ErrorCode::IoError,
            format!("cannot access path: {e}"),
            StatusCode::BAD_REQUEST,
        )
    })?;

    let mut paths: Vec<std::path::PathBuf> = Vec::new();
    if metadata.is_dir() {
        for entry in walkdir::WalkDir::new(&body.path) {
            let entry = entry.map_err(|e| {
                map_msg(
                    ErrorCode::IoError,
                    format!("walk error: {e}"),
                    StatusCode::BAD_REQUEST,
                )
            })?;
            if entry.file_type().is_file() {
                paths.push(entry.path().to_path_buf());
            }
        }
    } else if metadata.is_file() {
        paths.push(std::path::PathBuf::from(&body.path));
    } else {
        return Err(map_msg(
            ErrorCode::ValidationError,
            format!("path is not a file or directory: {}", body.path),
            StatusCode::BAD_REQUEST,
        ));
    }

    let mut all_chunks = Vec::new();
    let mut file_results = Vec::new();
    let mut ok_files = 0usize;

    for p in paths {
        let display = p.to_string_lossy().to_string();
        match lqm_ingest::extract_file_async(&p, serde_json::json!({})).await {
            Ok(extracted) => {
                let mut n = 0usize;
                for doc in extracted {
                    let pieces = expand_chunks(
                        &state.core,
                        &doc.text,
                        doc.source.or_else(|| Some(display.clone())),
                        doc.source_type,
                        collection.clone(),
                        doc.tags,
                        doc.project,
                        doc.last_modified,
                        Some(&display),
                        None,
                        None,
                    );
                    n += pieces.len();
                    all_chunks.extend(pieces);
                }
                ok_files += 1;
                file_results.push(make_file_result(&display, true, None, n));
            }
            Err(e) => {
                let err = e.to_string();
                file_results.push(make_file_result(&display, false, Some(&err), 0));
            }
        }
    }

    if all_chunks.is_empty() {
        return Ok(Json(serde_json::json!({
            "status": "no files ingested",
            "files": ok_files,
            "file_results": file_results,
            "inserted": 0, "skipped": 0, "replaced": 0, "chunks": 0,
            "collection": collection,
        })));
    }

    let report = state
        .core
        .embed_and_upsert_batch(all_chunks)
        .await
        .map_err(map_lqm_err)?;

    Ok(Json(serde_json::json!({
        "status": "ok",
        "files": ok_files,
        "file_results": file_results,
        "collection": collection,
        "inserted": report.inserted,
        "skipped": report.skipped,
        "replaced": report.replaced,
        "chunks": report.chunks,
    })))
}

async fn ingest_url(
    State(state): State<AppState>,
    Json(body): Json<IngestUrlBody>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let collection = resolve_collection(body.collection);
    state
        .core
        .ensure_collection(&collection, None)
        .await
        .map_err(map_lqm_err)?;

    let fetched = lqm_ingest::fetch_url(&body.url, None)
        .await
        .map_err(|e| map_msg(ErrorCode::FetchError, e.to_string(), StatusCode::BAD_REQUEST))?;

    let source = body.source.unwrap_or_else(|| fetched.url.clone());
    let source_type = body.source_type.unwrap_or(fetched.source_type);
    let chunks = expand_chunks(
        &state.core,
        &fetched.text,
        Some(source.clone()),
        Some(source_type.clone()),
        collection.clone(),
        body.tags,
        body.project,
        None,
        Some(&source),
        None,
        None,
    );
    let file_chunks = chunks.len();
    if chunks.is_empty() {
        return Err(map_msg(
            ErrorCode::ValidationError,
            format!("no chunks for {}", body.url),
            StatusCode::BAD_REQUEST,
        ));
    }

    let report = state
        .core
        .embed_and_upsert_batch(chunks)
        .await
        .map_err(map_lqm_err)?;

    Ok(Json(serde_json::json!({
        "status": "ok",
        "url": body.url,
        "source": source,
        "source_type": source_type,
        "title": fetched.title,
        "content_type": fetched.content_type,
        "collection": collection,
        "inserted": report.inserted,
        "skipped": report.skipped,
        "replaced": report.replaced,
        "chunks": report.chunks,
        "file_results": [{
            "path": body.url,
            "ok": true,
            "error": null,
            "chunks": file_chunks,
        }],
    })))
}

async fn ingest_many(
    State(state): State<AppState>,
    Json(body): Json<IngestManyBody>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let collection = resolve_collection(body.collection);
    state
        .core
        .ensure_collection(&collection, None)
        .await
        .map_err(map_lqm_err)?;

    let mut all_chunks = Vec::new();
    let mut file_results = Vec::new();

    if let Some(texts) = body.texts {
        for (i, text) in texts.into_iter().enumerate() {
            let src = format!("ingest_many://text/{i}");
            let pieces = expand_chunks(
                &state.core,
                &text,
                Some(src.clone()),
                Some(constants::SOURCE_TYPE_TEXT.into()),
                collection.clone(),
                body.tags.clone(),
                body.project.clone(),
                None,
                None,
                body.scope.clone(),
                body.clearance.clone().and_then(|s| s.parse().ok()),
            );
            let n = pieces.len();
            all_chunks.extend(pieces);
            file_results.push(make_file_result(&src, true, None, n));
        }
    }

    if let Some(paths) = body.paths {
        for path in paths {
            match lqm_ingest::extract_file_async(std::path::Path::new(&path), serde_json::json!({}))
                .await
            {
                Ok(extracted) => {
                    let mut n = 0usize;
                    for doc in extracted {
                        let pieces = expand_chunks(
                            &state.core,
                            &doc.text,
                            doc.source.or_else(|| Some(path.clone())),
                            doc.source_type,
                            collection.clone(),
                            body.tags.clone().or(doc.tags),
                            body.project.clone().or(doc.project),
                            doc.last_modified,
                            Some(&path),
                            body.scope.clone(),
                            body.clearance.clone().and_then(|s| s.parse().ok()),
                        );
                        n += pieces.len();
                        all_chunks.extend(pieces);
                    }
                    file_results.push(make_file_result(&path, true, None, n));
                }
                Err(e) => {
                    let err = e.to_string();
                    file_results.push(make_file_result(&path, false, Some(&err), 0));
                }
            }
        }
    }

    if let Some(urls) = body.urls {
        for url in urls {
            match lqm_ingest::fetch_url(&url, None).await {
                Ok(fetched) => {
                    let pieces = expand_chunks(
                        &state.core,
                        &fetched.text,
                        Some(url.clone()),
                        Some(fetched.source_type),
                        collection.clone(),
                        body.tags.clone(),
                        body.project.clone(),
                        None,
                        Some(&url),
                        body.scope.clone(),
                        body.clearance.clone().and_then(|s| s.parse().ok()),
                    );
                    let n = pieces.len();
                    all_chunks.extend(pieces);
                    let mut row = make_file_result(&url, true, None, n);
                    if let Some(obj) = row.as_object_mut() {
                        obj.insert(
                            "title".into(),
                            serde_json::to_value(fetched.title).unwrap_or(serde_json::Value::Null),
                        );
                    }
                    file_results.push(row);
                }
                Err(e) => {
                    let err = e.to_string();
                    file_results.push(make_file_result(&url, false, Some(&err), 0));
                }
            }
        }
    }

    if all_chunks.is_empty() {
        return Ok(Json(serde_json::json!({
            "status": "no items ingested",
            "file_results": file_results,
            "inserted": 0, "skipped": 0, "replaced": 0, "chunks": 0,
            "collection": collection,
        })));
    }

    let report = state
        .core
        .embed_and_upsert_batch(all_chunks)
        .await
        .map_err(map_lqm_err)?;

    Ok(Json(serde_json::json!({
        "status": "ok",
        "collection": collection,
        "file_results": file_results,
        "inserted": report.inserted,
        "skipped": report.skipped,
        "replaced": report.replaced,
        "chunks": report.chunks,
    })))
}

#[derive(Deserialize)]
struct StoreMemoryBody {
    text: String,
    importance: Option<f32>,
    tags: Option<Vec<String>>,
    project: Option<String>,
    memory_id: Option<String>,
    collection: Option<String>,
}

#[derive(Deserialize)]
struct RecallMemoriesBody {
    query: String,
    collection: Option<String>,
    limit: Option<u64>,
    use_recency: Option<bool>,
    project: Option<String>,
    tags: Option<Vec<String>>,
}

async fn store_memory(
    State(state): State<AppState>,
    Json(body): Json<StoreMemoryBody>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let coll = body
        .collection
        .unwrap_or_else(|| DEFAULT_MEMORY_COLLECTION.to_string());
    let note = MemoryNote {
        text: body.text,
        importance: body.importance,
        tags: body.tags,
        project: body.project,
        memory_id: body.memory_id,
    };
    let (report, effective_id) = state
        .core
        .store_memory(note, Some(&coll))
        .await
        .map_err(map_lqm_err)?;
    Ok(Json(serde_json::json!({
        "status": "ok",
        "collection": coll,
        "memory_id": effective_id,
        "inserted": report.inserted,
        "skipped": report.skipped,
        "replaced": report.replaced,
        "chunks": report.chunks,
    })))
}

async fn recall_memories(
    State(state): State<AppState>,
    Json(body): Json<RecallMemoriesBody>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let coll = body
        .collection
        .unwrap_or_else(|| DEFAULT_MEMORY_COLLECTION.to_string());
    let hits = state
        .core
        .recall_memories(
            &body.query,
            Some(&coll),
            body.limit,
            body.use_recency.unwrap_or(true),
            body.project.as_deref(),
            body.tags,
        )
        .await
        .map_err(map_lqm_err)?;
    Ok(Json(serde_json::json!({
        "status": "ok",
        "collection": coll,
        "query": body.query,
        "count": hits.len(),
        "memories": hits,
    })))
}

async fn index_html() -> axum::response::Html<&'static str> {
    axum::response::Html(include_str!("../static/index.html"))
}

/// Optional bearer auth for `/api/*` when `LQM_API_TOKEN` / AppState token is set.
async fn require_bearer(
    State(state): State<AppState>,
    headers: HeaderMap,
    req: axum::extract::Request,
    next: Next,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    const BEARER_PREFIX: &str = "Bearer ";
    if let Some(ref expected) = state.api_token {
        let auth = headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let token = auth
            .strip_prefix("Bearer ")
            .or_else(|| auth.strip_prefix("bearer "))
            .unwrap_or("");
        if token != expected.as_str() {
            return Err(map_msg(
                ErrorCode::Unauthorized,
                "missing or invalid Authorization bearer token",
                StatusCode::UNAUTHORIZED,
            ));
        }
    }
    Ok(next.run(req).await)
}

async fn request_logger(
    req: axum::extract::Request,
    next: middleware::Next,
) -> axum::response::Response {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let start = std::time::Instant::now();
    let response = next.run(req).await;
    let status = response.status();
    let duration = start.elapsed();
    log::info!(
        "{} {} -> {} ({:?})",
        method,
        path,
        status.as_u16(),
        duration
    );
    response
}

fn build_router(state: AppState) -> Router {
    // Health stays unauthenticated; all `/api/*` routes can require bearer when configured.
    let api = Router::new()
        .route("/api/embedder", get(get_embedder_info))
        .route(
            "/api/collections",
            get(list_collections).post(create_collection),
        )
        .route(
            "/api/collections/{name}",
            get(get_collection_info).delete(delete_collection),
        )
        .route("/api/collections/{name}/sources", get(list_sources))
        .route(
            "/api/collections/{name}/sources/{source}",
            get(get_source).delete(delete_by_source),
        )
        .route(
            "/api/collections/{name}/sources/{source}/chunks",
            get(list_chunks),
        )
        .route(
            "/api/collections/{name}/delete_by_filter",
            post(delete_by_filter),
        )
        .route("/api/search", post(search))
        .route("/api/context", post(get_relevant_context))
        .route("/api/expand_context", post(expand_context))
        .route("/api/ingest", post(ingest))
        .route("/api/ingest/path", post(ingest_path))
        .route("/api/ingest/url", post(ingest_url))
        .route("/api/ingest/many", post(ingest_many))
        .route("/api/memories", post(store_memory))
        .route("/api/memories/recall", post(recall_memories))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            require_bearer,
        ))
        .with_state(state);

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/health", get(health))
        .merge(api)
        .route("/", get(index_html))
        .layer(cors)
        .layer(middleware::from_fn(request_logger))
        .fallback_service(ServeDir::new("static"))
}

fn qdrant_url(cli: &Cli) -> String {
    cli.qdrant_url
        .clone()
        .or_else(|| std::env::var("QDRANT_URL").ok())
        .unwrap_or_else(|| constants::DEFAULT_QDRANT_URL.to_string())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let cli = Cli::parse();

    let url = qdrant_url(&cli);
    let core = RagCore::from_env(Some(&url), cli.config.as_deref()).await?;
    log::info!(
        "Embedder: {} (dim={})",
        core.embedder.id(),
        core.embedder.dimension()
    );

    let dim = core.embedder.dimension();
    let api_token = std::env::var("LQM_API_TOKEN")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    if api_token.is_some() {
        log::info!("LQM_API_TOKEN set — /api/* requires Authorization: Bearer …");
    }
    let state = AppState {
        core: Arc::new(core),
        embed_dimension: dim,
        api_token,
    };

    let app = build_router(state);
    let addr = format!("{}:{}", cli.host, cli.port);
    log::info!("API server listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    let serve = axum::serve(listener, app.into_make_service());
    serve.await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lqm_core::error::LqmError;

    #[tokio::test]
    async fn test_health_endpoint() {
        let response = health().await;
        assert_eq!(response.0.status, "ok");
    }

    #[tokio::test]
    async fn test_health_endpoint_includes_version() {
        let response = health().await;
        assert_eq!(response.0.version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn test_map_lqm_err_structured() {
        let (status, Json(body)) = map_lqm_err(LqmError::Validation("empty name".into()));
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body.code, "validation_error");
        assert!(body.message.contains("empty name"));
        assert_eq!(body.error, body.message);
    }

    #[test]
    fn test_chunk_indices_on_document_chunks() {
        use lqm_core::{ChunkingStrategy, chunk_for_ingest};
        let strategy = ChunkingStrategy::text(2000, 20);
        let pieces = chunk_for_ingest(
            "# A\n\none\n\n# B\n\ntwo",
            Some("text"),
            Some("x.md"),
            &strategy,
        );
        assert!(pieces.len() >= 2);
        let total = pieces.len();
        let chunks: Vec<_> = pieces
            .into_iter()
            .enumerate()
            .map(|(i, text)| DocumentChunk {
                text,
                source: Some("x.md".into()),
                source_type: Some("text".into()),
                collection: Some("c".into()),
                tags: None,
                timestamp: None,
                project: None,
                last_modified: None,
                chunk_index: Some(i),
                total_chunks: Some(total),
                importance: None,
                memory_id: None,
                scope: None,
                clearance: None,
            })
            .collect();
        assert_eq!(chunks[0].chunk_index, Some(0));
        assert_eq!(chunks[0].total_chunks, Some(total));
        let payload = lqm_core::qdrant::build_point_payload(
            &chunks[0],
            chunks[0].chunk_index.unwrap(),
            total,
            "fake",
        );
        assert_eq!(payload[lqm_core::payload_schema::CHUNK_INDEX], 0);
        assert_eq!(payload[lqm_core::payload_schema::TOTAL_CHUNKS], total);
        assert_eq!(payload[lqm_core::payload_schema::EMBEDDING_MODEL], "fake");
    }
}
