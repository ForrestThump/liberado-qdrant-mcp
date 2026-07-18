use clap::{Parser, Subcommand};
use lqm_core::RagCore;
use lqm_core::config::{EmbedderConfig, create_embedder};
use lqm_core::format_relevant_context;
use lqm_core::types::{DEFAULT_COLLECTION_NAME, DocumentChunk, RagConfig};
use serde_json::Value;
use std::sync::Arc;
use turbomcp::prelude::*;

#[derive(Parser)]
#[command(name = "lqm-mcp", about = "liberado-qdrant-mcp server")]
struct Cli {
    #[arg(long, env = "LQM_CONFIG")]
    config: Option<String>,

    /// Qdrant gRPC endpoint. Also settable via QDRANT_URL — in the container the flag would have to
    /// precede the subcommand, which fights ENTRYPOINT/CMD, so the env var is the practical path.
    #[arg(long, env = "QDRANT_URL", default_value = "http://localhost:6334")]
    qdrant_url: String,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Serve {
        // 0.0.0.0, not 127.0.0.1: this serves inside a container and Liberado reaches it across the
        // Docker bridge. Bound to loopback it would be unreachable from anywhere but its own netns.
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        #[arg(long, default_value_t = 3000)]
        port: u16,
    },
}

/// Origin policy for the MCP surface.
///
/// turbomcp validates the `Origin` header and answers 403 to anything it does not recognise unless
/// the peer is loopback. That guards against DNS rebinding, where a malicious page in a **browser**
/// aims XHR at an MCP server bound to localhost.
///
/// It cannot do that job here and it breaks the only client we have: Liberado is a server-side HTTP
/// client reaching this container across a private Docker bridge — not loopback, and it sends no
/// `Origin` at all, so it is refused every time. Browsers always send `Origin` on cross-origin
/// requests, so an Origin-less request is definitionally not the attack being defended against.
///
/// Default is permissive because this server is consumed by MCP clients on a private network and is
/// not published to the internet. Set `MCP_ALLOWED_ORIGINS` to a comma-separated list to enforce an
/// allow-list if it is ever exposed to a browser.
fn origin_policy() -> ServerConfig {
    let builder = ServerConfig::builder();

    match std::env::var("MCP_ALLOWED_ORIGINS") {
        Ok(raw) if !raw.trim().is_empty() => {
            let origins: Vec<String> = raw
                .split(',')
                .map(|o| o.trim().to_string())
                .filter(|o| !o.is_empty())
                .collect();
            log::info!("MCP origin validation: allow-listed: {origins:?}");
            builder
                .allow_origins(origins)
                .allow_any_origin(false)
                .build()
        }
        _ => {
            log::info!(
                "MCP origin validation: disabled (private-network default). \
                 Set MCP_ALLOWED_ORIGINS to enforce an allow-list."
            );
            builder.allow_any_origin(true).build()
        }
    }
}

#[derive(Clone)]
struct LqmServer {
    core: Arc<RagCore>,
}

impl LqmServer {
    /// Create the collection if it is missing, sized to the active embedder.
    ///
    /// Deliberately a plain inherent method outside the `#[server]` block: anything in there is a
    /// candidate for tool export, and this is internal plumbing, not a tool.
    ///
    /// The dimension must come from the embedder rather than config — a collection created at the
    /// wrong width makes every later upsert fail on a dimension mismatch, and Qdrant will not
    /// silently resize it.
    async fn ensure_collection(&self, collection: &str) -> McpResult<()> {
        let dim = self.core.embedder.dimension();
        self.core
            .ensure_collection(collection, dim)
            .await
            .map_err(|e| {
                McpError::internal(format!(
                    "could not ensure collection '{collection}' (dim={dim}): {e}"
                ))
            })
    }
}

#[server(name = "liberado-qdrant-mcp", version = "0.1.0")]
impl LqmServer {
    async fn new(core: RagCore) -> Self {
        Self {
            core: Arc::new(core),
        }
    }

    fn core(&self) -> &RagCore {
        &self.core
    }

    #[tool]
    #[allow(clippy::too_many_arguments)]
    async fn ingest_text(
        &self,
        text: String,
        source: Option<String>,
        source_type: Option<String>,
        collection: Option<String>,
        tags: Option<Vec<String>>,
        project: Option<String>,
        last_modified: Option<String>,
    ) -> McpResult<Value> {
        let core = self.core();
        let collection = collection.unwrap_or_else(|| DEFAULT_COLLECTION_NAME.to_string());

        // Qdrant rejects an upsert into a collection that does not exist, and this server exposes no
        // create_collection tool — so without this an agent could never ingest anything at all: even
        // the default collection 404s on a fresh Qdrant, with no in-band way to fix it. Create on
        // demand at the embedder's dimension, which is also the only place that knows the right size.
        self.ensure_collection(&collection).await?;

        let chunk = DocumentChunk {
            text,
            source,
            source_type,
            collection: Some(collection),
            tags,
            timestamp: None,
            project,
            last_modified,
        };
        match core.embed_and_upsert_batch(vec![chunk]).await {
            Ok(count) => Ok(serde_json::json!({"status": "ok", "chunks": count})),
            Err(e) => Err(McpError::internal(format!("ingest failed: {e}"))),
        }
    }

    #[tool]
    async fn search(
        &self,
        query: String,
        collection: Option<String>,
        limit: Option<u64>,
        tags: Option<Vec<String>>,
        source_type: Option<String>,
        min_score: Option<f32>,
    ) -> McpResult<Value> {
        let core = self.core();
        let coll = collection.clone();
        match core
            .search(
                &query,
                coll.as_deref(),
                Some(limit.unwrap_or(10)),
                tags,
                source_type.as_deref(),
                min_score,
            )
            .await
        {
            Ok(results) => {
                let json_results: Vec<Value> = results
                    .iter()
                    .map(|r| {
                        serde_json::json!({
                            "text": r.text,
                            "score": r.score,
                            "payload": r.payload,
                        })
                    })
                    .collect();
                Ok(serde_json::json!({"results": json_results}))
            }
            Err(e) => Err(McpError::internal(format!("search failed: {e}"))),
        }
    }

    /// Semantic search formatted as LLM-ready markdown context with citations.
    ///
    /// Reuses the same search path as `search`, then formats numbered passages with
    /// score/source metadata plus a structured `sources` array for tooling.
    #[tool]
    #[allow(clippy::too_many_arguments)]
    async fn get_relevant_context(
        &self,
        query: String,
        collection: Option<String>,
        limit: Option<u64>,
        tags: Option<Vec<String>>,
        source_type: Option<String>,
        min_score: Option<f32>,
        max_chars_per_passage: Option<u64>,
    ) -> McpResult<Value> {
        let core = self.core();
        let coll = collection.clone();
        let results = core
            .search(
                &query,
                coll.as_deref(),
                Some(limit.unwrap_or(8)),
                tags,
                source_type.as_deref(),
                min_score,
            )
            .await
            .map_err(|e| McpError::internal(format!("get_relevant_context search failed: {e}")))?;

        let max_chars = max_chars_per_passage.map(|n| n as usize);
        let formatted = format_relevant_context(&query, &results, max_chars);

        Ok(serde_json::json!({
            "status": "ok",
            "query": query,
            "collection": collection.unwrap_or_else(|| DEFAULT_COLLECTION_NAME.to_string()),
            "passage_count": formatted.passage_count,
            "context": formatted.context,
            "sources": formatted.sources,
        }))
    }

    #[tool]
    async fn list_collections(&self) -> McpResult<Value> {
        let core = self.core();
        match core.list_collections().await {
            Ok(collections) => Ok(serde_json::json!({"collections": collections})),
            Err(e) => Err(McpError::internal(format!("list failed: {e}"))),
        }
    }

    /// Create a Qdrant collection for agent-scoped knowledge.
    ///
    /// When `vector_dim` is omitted, the active embedder's dimension is used so later upserts
    /// match. Idempotent: existing collections are left alone and reported as `created: false`.
    #[tool]
    async fn create_collection(&self, name: String, vector_dim: Option<u64>) -> McpResult<Value> {
        let dim = vector_dim.map(|d| d as usize);
        match self.core().create_collection(&name, dim).await {
            Ok(created) => {
                let resolved_dim = dim.unwrap_or_else(|| self.core().embedder.dimension());
                Ok(serde_json::json!({
                    "status": "ok",
                    "name": name,
                    "created": created,
                    "vector_dim": resolved_dim,
                }))
            }
            Err(e) => Err(McpError::internal(format!("create_collection failed: {e}"))),
        }
    }

    /// Delete a collection and all of its points. Returns `deleted: false` if it did not exist.
    #[tool]
    async fn delete_collection(&self, name: String) -> McpResult<Value> {
        match self.core().delete_collection(&name).await {
            Ok(deleted) => Ok(serde_json::json!({
                "status": "ok",
                "name": name,
                "deleted": deleted,
            })),
            Err(e) => Err(McpError::internal(format!("delete_collection failed: {e}"))),
        }
    }

    /// Inspect a collection: point counts, vector size, distance metric, status.
    #[tool]
    async fn get_collection_info(&self, name: String) -> McpResult<Value> {
        match self.core().get_collection_info(&name).await {
            Ok(Some(info)) => Ok(serde_json::json!({
                "status": "ok",
                "exists": true,
                "name": info.name,
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "segments_count": info.segments_count,
                "collection_status": info.status,
                "vector_size": info.vector_size,
                "distance": info.distance,
            })),
            Ok(None) => Ok(serde_json::json!({
                "status": "ok",
                "exists": false,
                "name": name,
            })),
            Err(e) => Err(McpError::internal(format!(
                "get_collection_info failed: {e}"
            ))),
        }
    }

    /// Fetch a remote HTTP(S) URL, extract text (HTML stripped or plain), chunk/embed/upsert.
    ///
    /// Source metadata defaults to the URL; `source_type` defaults to the extractor's guess
    /// (`webpage` for HTML, `url` otherwise). Optional tags/project flow into the payload.
    #[tool]
    #[allow(clippy::too_many_arguments)]
    async fn ingest_url(
        &self,
        url: String,
        collection: Option<String>,
        tags: Option<Vec<String>>,
        source_type: Option<String>,
        project: Option<String>,
        source: Option<String>,
    ) -> McpResult<Value> {
        let collection = collection.unwrap_or_else(|| DEFAULT_COLLECTION_NAME.to_string());
        self.ensure_collection(&collection).await?;

        let fetched = lqm_ingest::fetch_url(&url, None)
            .await
            .map_err(|e| McpError::internal(format!("ingest_url fetch failed: {e}")))?;

        let resolved_source = source.unwrap_or_else(|| fetched.url.clone());
        let resolved_source_type = source_type.unwrap_or(fetched.source_type);

        // Chunk long pages so retrieval stays passage-level, same path as local file ingest.
        let pieces = self.core().chunk_text_method(&fetched.text);
        let chunks: Vec<DocumentChunk> = pieces
            .into_iter()
            .map(|text| DocumentChunk {
                text,
                source: Some(resolved_source.clone()),
                source_type: Some(resolved_source_type.clone()),
                collection: Some(collection.clone()),
                tags: tags.clone(),
                timestamp: None,
                project: project.clone(),
                last_modified: None,
            })
            .collect();

        if chunks.is_empty() {
            return Err(McpError::internal(format!(
                "ingest_url produced no chunks for {url}"
            )));
        }

        let count = self
            .core()
            .embed_and_upsert_batch(chunks)
            .await
            .map_err(|e| McpError::internal(format!("ingest_url upsert failed: {e}")))?;

        log::info!(
            "ingested url {} ({} chunks) into '{}'",
            url,
            count,
            collection
        );
        Ok(serde_json::json!({
            "status": "ok",
            "url": url,
            "source": resolved_source,
            "source_type": resolved_source_type,
            "content_type": fetched.content_type,
            "chunks": count,
            "collection": collection,
        }))
    }

    #[tool]
    async fn ingest_path(&self, path: String, collection: Option<String>) -> McpResult<Value> {
        let core = self.core();
        let collection = collection.unwrap_or_else(|| DEFAULT_COLLECTION_NAME.to_string());
        // Same reason as ingest_text: Qdrant rejects upserts into a missing collection and there is
        // no create_collection tool to reach for.
        self.ensure_collection(&collection).await?;
        let metadata = std::fs::metadata(&path)
            .map_err(|e| McpError::internal(format!("cannot access path: {e}")))?;

        let mut chunks: Vec<DocumentChunk> = Vec::new();
        let mut file_count = 0usize;

        if metadata.is_dir() {
            for entry in walkdir::WalkDir::new(&path) {
                let entry = entry.map_err(|e| McpError::internal(format!("walk error: {e}")))?;
                if entry.file_type().is_file() {
                    let base_payload = serde_json::json!({});
                    match lqm_ingest::extract_file(entry.path(), base_payload) {
                        Ok(mut extracted) => {
                            for c in &mut extracted {
                                c.collection = Some(collection.clone());
                            }
                            chunks.append(&mut extracted);
                            file_count += 1;
                        }
                        Err(e) => {
                            log::warn!("skipping {}: {}", entry.path().display(), e);
                        }
                    }
                }
            }
        } else if metadata.is_file() {
            let base_payload = serde_json::json!({});
            match lqm_ingest::extract_file(std::path::Path::new(&path), base_payload) {
                Ok(mut extracted) => {
                    for c in &mut extracted {
                        c.collection = Some(collection.clone());
                    }
                    chunks.append(&mut extracted);
                    file_count += 1;
                }
                Err(e) => {
                    return Err(McpError::internal(format!(
                        "extraction failed for {}: {e}",
                        path
                    )));
                }
            }
        } else {
            return Err(McpError::internal(format!(
                "path is not a file or directory: {path}"
            )));
        }

        if chunks.is_empty() {
            return Ok(serde_json::json!({"status": "no files ingested", "files": 0, "chunks": 0}));
        }

        let result = core
            .embed_and_upsert_batch(chunks)
            .await
            .map_err(|e| McpError::internal(format!("upsert failed: {e}")))?;

        log::info!(
            "ingested {} files ({} chunks) into '{}'",
            file_count,
            result,
            collection
        );
        Ok(serde_json::json!({
            "status": "ok",
            "files": file_count,
            "chunks": result,
            "collection": collection,
        }))
    }
}

#[cfg(test)]
fn test_qdrant_url() -> String {
    std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6334".to_string())
}

#[cfg(test)]
async fn create_test_server() -> Option<LqmServer> {
    let qdrant_url = test_qdrant_url();
    let config = EmbedderConfig::load_or_default(None).ok()?;
    let embedder = create_embedder(&config).ok()?;
    let qdrant = lqm_core::QdrantClient::new(&qdrant_url).await.ok()?;
    let core = RagCore::from_config(qdrant, embedder, &RagConfig::default());
    match core.list_collections().await {
        Ok(_) => Some(LqmServer::new(core).await),
        Err(_) => None,
    }
}

#[tokio::test]
async fn test_list_collections() {
    let server = create_test_server().await;
    if server.is_none() {
        return;
    }
    let result = server.unwrap().core().list_collections().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_collection_create_info_delete_tools() {
    let server = create_test_server().await;
    if server.is_none() {
        return;
    }
    let server = server.unwrap();
    let coll = "lqm_mcp_test_coll_mgmt";
    let _ = server.core().delete_collection(coll).await;

    let created = server
        .create_collection(coll.to_string(), None)
        .await
        .expect("create_collection");
    assert_eq!(created["status"], "ok");
    assert_eq!(created["created"], true);
    assert_eq!(
        created["vector_dim"],
        server.core().embedder.dimension() as u64
    );

    let info = server
        .get_collection_info(coll.to_string())
        .await
        .expect("get_collection_info");
    assert_eq!(info["exists"], true);
    assert_eq!(info["name"], coll);
    assert!(info["vector_size"].as_u64().is_some());

    let deleted = server
        .delete_collection(coll.to_string())
        .await
        .expect("delete_collection");
    assert_eq!(deleted["deleted"], true);

    let gone = server
        .get_collection_info(coll.to_string())
        .await
        .expect("get after delete");
    assert_eq!(gone["exists"], false);
}

#[tokio::test]
async fn test_ingest_and_search() {
    let server = create_test_server().await;
    if server.is_none() {
        return;
    }
    let server = server.unwrap();
    let core = server.core();

    let coll = "lqm_mcp_test_ingest";
    let _ = core.delete_collection(coll).await;

    let chunk = DocumentChunk {
        text: "Hello world from lqm-mcp test".to_string(),
        source: Some("test_source".to_string()),
        source_type: Some("text".to_string()),
        collection: Some(coll.to_string()),
        tags: None,
        timestamp: None,
        project: Some("test_project".to_string()),
        last_modified: None,
    };
    let ingest_result = core.embed_and_upsert_batch(vec![chunk]).await;
    if ingest_result.is_err() {
        return;
    }

    let search_result = core
        .search("Hello world", Some(coll), Some(5), None, None, None)
        .await;
    assert!(search_result.is_ok());

    let _ = core.delete_collection(coll).await;
}

#[tokio::test]
async fn test_search_edge_cases() {
    let server = create_test_server().await;
    if server.is_none() {
        return;
    }
    let server = server.unwrap();
    let core = server.core();

    let coll = "lqm_mcp_test_edge";
    let _ = core.delete_collection(coll).await;

    let chunk = DocumentChunk {
        text: "test content".to_string(),
        source: Some("s".to_string()),
        source_type: Some("text".to_string()),
        collection: Some(coll.to_string()),
        tags: None,
        timestamp: None,
        project: None,
        last_modified: None,
    };
    let _ = core.embed_and_upsert_batch(vec![chunk]).await;
    let result = core
        .search(
            "nonexistent_term_xyz",
            Some(coll),
            Some(10),
            None,
            None,
            None,
        )
        .await;
    assert!(result.is_ok());
    let result = core
        .search("test", Some(coll), Some(0), None, None, None)
        .await;
    assert!(result.is_ok());

    let _ = core.delete_collection(coll).await;
}

#[tokio::test]
async fn test_multiple_collections_independence() {
    let server = create_test_server().await;
    if server.is_none() {
        return;
    }
    let server = server.unwrap();
    let core = server.core();

    let coll_a = "lqm_mcp_test_a";
    let coll_b = "lqm_mcp_test_b";
    let _ = core.delete_collection(coll_a).await;
    let _ = core.delete_collection(coll_b).await;

    let chunk_a = DocumentChunk {
        text: "content a".to_string(),
        source: Some("s".to_string()),
        source_type: Some("text".to_string()),
        collection: Some(coll_a.to_string()),
        tags: None,
        timestamp: None,
        project: None,
        last_modified: None,
    };
    let _ = core.embed_and_upsert_batch(vec![chunk_a]).await;
    let chunk_b = DocumentChunk {
        text: "content b".to_string(),
        source: Some("s".to_string()),
        source_type: Some("text".to_string()),
        collection: Some(coll_b.to_string()),
        tags: None,
        timestamp: None,
        project: None,
        last_modified: None,
    };
    let _ = core.embed_and_upsert_batch(vec![chunk_b]).await;

    let result = core
        .search("content a", Some(coll_a), Some(5), None, None, None)
        .await;
    assert!(result.is_ok());
    let result = core
        .search("content b", Some(coll_b), Some(5), None, None, None)
        .await;
    assert!(result.is_ok());

    let _ = core.delete_collection(coll_a).await;
    let _ = core.delete_collection(coll_b).await;
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let cli = Cli::parse();
    let qdrant_url = cli.qdrant_url.clone();
    let config = EmbedderConfig::load_or_default(cli.config.as_deref())?;
    let embedder =
        create_embedder(&config).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
    log::info!("starting lqm-mcp, embedder backend: {}", embedder.id());
    let qdrant = lqm_core::QdrantClient::new(&qdrant_url).await?;
    let core = RagCore::from_config(qdrant, embedder, &RagConfig::default());

    match cli.command {
        Some(Commands::Serve { host, port }) => {
            log::info!("lqm-mcp server started");
            let server = LqmServer::new(core).await;
            let addr = format!("{}:{}", host, port);
            // Deliberately not `run_http(&addr)`: that convenience wrapper takes no ServerConfig, so
            // turbomcp's origin validation stays on defaults and 403s every Liberado request. Go
            // through the builder so origin_policy() actually applies.
            server
                .builder()
                .with_config(origin_policy())
                .transport(Transport::http(&addr))
                .serve()
                .await?;
        }
        None => {
            log::info!("lqm-mcp server started");
            let server = LqmServer::new(core).await;
            server.run_stdio().await?;
        }
    }

    Ok(())
}
