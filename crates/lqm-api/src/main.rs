use axum::{
    Router,
    extract::{Path, State},
    http::StatusCode,
    middleware,
    response::Json,
    routing::{delete, get, post},
};
use clap::Parser;
use lqm_core::config::{EmbedderConfig, create_embedder};
use lqm_core::format_relevant_context_with;
use lqm_core::types::{
    ContextOptions, DEFAULT_COLLECTION_NAME, DocumentChunk, PayloadFilter, RagConfig, SearchFilter,
    SearchOptions,
};
use lqm_core::{QdrantClient, RagCore};
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
    error: String,
}

fn timestamp_now() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .to_string()
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
    let collections = state.core.list_collections().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;
    Ok(Json(CollectionsResponse { collections }))
}

async fn delete_collection(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    let deleted = state.core.delete_collection(&name).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;
    if deleted {
        log::info!("Deleted collection '{}'", name);
    }
    Ok(StatusCode::NO_CONTENT)
}

async fn search(
    State(state): State<AppState>,
    Json(body): Json<SearchBody>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
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
                },
            },
        )
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;

    Ok(Json(serde_json::json!({
        "results": page.results,
        "offset": page.offset,
        "limit": page.limit,
        "has_more": page.has_more,
        "next_offset": page.next_offset,
    })))
}

async fn get_relevant_context(
    State(state): State<AppState>,
    Json(body): Json<ContextBody>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let coll = body
        .collection
        .clone()
        .unwrap_or_else(|| DEFAULT_COLLECTION_NAME.to_string());
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
                },
            },
        )
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;

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
    let collection = body
        .collection
        .unwrap_or_else(|| DEFAULT_COLLECTION_NAME.to_string());

    let chunk = DocumentChunk {
        text: body.text,
        source: Some(body.source.unwrap_or_else(|| "api-ingest".to_string())),
        source_type: Some(body.source_type.unwrap_or_else(|| "text".to_string())),
        collection: Some(collection.clone()),
        tags: Some(body.tags.unwrap_or_default()),
        timestamp: Some(timestamp_now()),
        project: body.project,
        last_modified: body.last_modified.map(|ts| ts.to_string()),
    };

    state
        .core
        .ensure_collection(&collection, state.embed_dimension)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;

    let report = state
        .core
        .embed_and_upsert_batch(vec![chunk])
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;

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
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;
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
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )),
    }
}

async fn list_sources(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let sources = state.core.list_sources(&name).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;
    Ok(Json(serde_json::json!({
        "status": "ok",
        "collection": name,
        "sources": sources,
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
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;
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
    };
    let deleted = state
        .core
        .delete_by_filter(&name, &filter)
        .await
        .map_err(|e| {
            let status = if e.to_string().contains("validation") {
                StatusCode::BAD_REQUEST
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (
                status,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;
    Ok(Json(serde_json::json!({
        "status": "ok",
        "collection": name,
        "deleted": deleted,
        "filter": filter,
    })))
}

async fn index_html() -> axum::response::Html<&'static str> {
    axum::response::Html(include_str!("../static/index.html"))
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
    let api = Router::new()
        .route("/health", get(health))
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
            delete(delete_by_source),
        )
        .route(
            "/api/collections/{name}/delete_by_filter",
            post(delete_by_filter),
        )
        .route("/api/search", post(search))
        .route("/api/context", post(get_relevant_context))
        .route("/api/ingest", post(ingest))
        .with_state(state);

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
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
        .unwrap_or_else(|| "http://localhost:6334".to_string())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let cli = Cli::parse();

    let url = qdrant_url(&cli);
    let qdrant = QdrantClient::new(&url).await?;

    let embedder_config = EmbedderConfig::load_or_default(cli.config.as_deref())?;
    let embedder = create_embedder(&embedder_config)?;
    log::info!("Embedder: {} (dim={})", embedder.id(), embedder.dimension());

    let dim = embedder.dimension();
    let core = RagCore::from_config(qdrant, embedder, &RagConfig::default());
    let state = AppState {
        core: Arc::new(core),
        embed_dimension: dim,
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
}
