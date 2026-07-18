use clap::{Parser, Subcommand};
use lqm_core::RagCore;
use lqm_core::config::{EmbedderConfig, create_embedder};
use lqm_core::format_relevant_context_with;
use lqm_core::types::{
    ContextOptions, DEFAULT_COLLECTION_NAME, DocumentChunk, PayloadFilter, RagConfig, SearchFilter,
    SearchOptions,
};
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

    /// Expand a source document into structure-aware `DocumentChunk`s for upsert.
    #[allow(clippy::too_many_arguments)]
    fn expand_to_chunks(
        &self,
        text: &str,
        source: Option<String>,
        source_type: Option<String>,
        collection: String,
        tags: Option<Vec<String>>,
        project: Option<String>,
        last_modified: Option<String>,
        path_hint: Option<&str>,
    ) -> Vec<DocumentChunk> {
        let pieces = self.core().chunk_for_ingest(
            text,
            source_type.as_deref(),
            path_hint.or(source.as_deref()),
        );
        pieces
            .into_iter()
            .map(|text| DocumentChunk {
                text,
                source: source.clone(),
                source_type: source_type.clone(),
                collection: Some(collection.clone()),
                tags: tags.clone(),
                timestamp: None,
                project: project.clone(),
                last_modified: last_modified.clone(),
            })
            .collect()
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

        let path_hint = source.clone();
        let chunks = self.expand_to_chunks(
            &text,
            source,
            source_type,
            collection.clone(),
            tags,
            project,
            last_modified,
            path_hint.as_deref(),
        );
        if chunks.is_empty() {
            return Ok(serde_json::json!({
                "status": "ok",
                "collection": collection,
                "inserted": 0,
                "skipped": 0,
                "replaced": 0,
                "chunks": 0,
                "files": [{
                    "path": path_hint,
                    "ok": true,
                    "error": null,
                    "chunks": 0,
                }],
            }));
        }
        let file_chunks = chunks.len();
        match core.embed_and_upsert_batch(chunks).await {
            Ok(report) => Ok(serde_json::json!({
                "status": "ok",
                "collection": collection,
                "inserted": report.inserted,
                "skipped": report.skipped,
                "replaced": report.replaced,
                "chunks": report.chunks,
                "files": [{
                    "path": path_hint,
                    "ok": true,
                    "error": null,
                    "chunks": file_chunks,
                }],
            })),
            Err(e) => Err(McpError::internal(format!("ingest failed: {e}"))),
        }
    }

    /// Semantic search with filters and offset pagination.
    #[tool]
    #[allow(clippy::too_many_arguments)]
    async fn search(
        &self,
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
    ) -> McpResult<Value> {
        let page = self
            .core()
            .search_page(
                &query,
                SearchOptions {
                    collection: collection.clone(),
                    limit: Some(limit.unwrap_or(10)),
                    offset,
                    min_score,
                    filter: SearchFilter {
                        source,
                        source_type,
                        project,
                        tags,
                        tags_should,
                        tags_must_not,
                    },
                },
            )
            .await
            .map_err(|e| McpError::internal(format!("search failed: {e}")))?;

        let json_results: Vec<Value> = page
            .results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "text": r.text,
                    "score": r.score,
                    "payload": r.payload,
                })
            })
            .collect();
        Ok(serde_json::json!({
            "results": json_results,
            "offset": page.offset,
            "limit": page.limit,
            "has_more": page.has_more,
            "next_offset": page.next_offset,
        }))
    }

    /// Semantic search formatted as LLM-ready markdown context with citations.
    ///
    /// Shared filters/pagination with `search`. Optional `max_total_chars` budget and
    /// `mmr` diversity reordering before formatting.
    #[tool]
    #[allow(clippy::too_many_arguments)]
    async fn get_relevant_context(
        &self,
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
    ) -> McpResult<Value> {
        let coll = collection
            .clone()
            .unwrap_or_else(|| DEFAULT_COLLECTION_NAME.to_string());
        let page = self
            .core()
            .search_page(
                &query,
                SearchOptions {
                    collection: Some(coll.clone()),
                    limit: Some(limit.unwrap_or(8)),
                    offset,
                    min_score,
                    filter: SearchFilter {
                        source,
                        source_type,
                        project,
                        tags,
                        tags_should,
                        tags_must_not,
                    },
                },
            )
            .await
            .map_err(|e| McpError::internal(format!("get_relevant_context search failed: {e}")))?;

        let formatted = format_relevant_context_with(
            &query,
            &page.results,
            &ContextOptions {
                max_chars_per_passage: max_chars_per_passage.map(|n| n as usize),
                max_total_chars: max_total_chars.map(|n| n as usize),
                mmr: mmr.unwrap_or(false),
                mmr_lambda,
            },
        );

        Ok(serde_json::json!({
            "status": "ok",
            "query": query,
            "collection": coll,
            "passage_count": formatted.passage_count,
            "truncated_by_budget": formatted.truncated_by_budget,
            "offset": page.offset,
            "limit": page.limit,
            "has_more": page.has_more,
            "next_offset": page.next_offset,
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

        let chunks = self.expand_to_chunks(
            &fetched.text,
            Some(resolved_source.clone()),
            Some(resolved_source_type.clone()),
            collection.clone(),
            tags,
            project,
            None,
            Some(&resolved_source),
        );

        if chunks.is_empty() {
            return Err(McpError::internal(format!(
                "ingest_url produced no chunks for {url}"
            )));
        }
        let file_chunks = chunks.len();

        let report = self
            .core()
            .embed_and_upsert_batch(chunks)
            .await
            .map_err(|e| McpError::internal(format!("ingest_url upsert failed: {e}")))?;

        log::info!(
            "ingested url {} ({} chunks) into '{}' (inserted={} skipped={} replaced={})",
            url,
            report.chunks,
            collection,
            report.inserted,
            report.skipped,
            report.replaced
        );
        Ok(serde_json::json!({
            "status": "ok",
            "url": url,
            "source": resolved_source,
            "source_type": resolved_source_type,
            "title": fetched.title,
            "content_type": fetched.content_type,
            "collection": collection,
            "inserted": report.inserted,
            "skipped": report.skipped,
            "replaced": report.replaced,
            "chunks": report.chunks,
            "files": [{
                "path": url,
                "ok": true,
                "error": null,
                "chunks": file_chunks,
            }],
        }))
    }

    /// Ingest a file or directory with per-file `{ path, ok, error, chunks }` reporting.
    #[tool]
    async fn ingest_path(&self, path: String, collection: Option<String>) -> McpResult<Value> {
        let core = self.core();
        let collection = collection.unwrap_or_else(|| DEFAULT_COLLECTION_NAME.to_string());
        self.ensure_collection(&collection).await?;
        let metadata = std::fs::metadata(&path)
            .map_err(|e| McpError::internal(format!("cannot access path: {e}")))?;

        let mut paths: Vec<std::path::PathBuf> = Vec::new();
        if metadata.is_dir() {
            for entry in walkdir::WalkDir::new(&path) {
                let entry = entry.map_err(|e| McpError::internal(format!("walk error: {e}")))?;
                if entry.file_type().is_file() {
                    paths.push(entry.path().to_path_buf());
                }
            }
        } else if metadata.is_file() {
            paths.push(std::path::PathBuf::from(&path));
        } else {
            return Err(McpError::internal(format!(
                "path is not a file or directory: {path}"
            )));
        }

        let mut all_chunks: Vec<DocumentChunk> = Vec::new();
        let mut file_reports: Vec<Value> = Vec::new();
        let mut ok_files = 0usize;

        for p in paths {
            let display = p.to_string_lossy().to_string();
            match lqm_ingest::extract_file(&p, serde_json::json!({})) {
                Ok(extracted) => {
                    let mut file_chunk_count = 0usize;
                    for doc in extracted {
                        let pieces = self.expand_to_chunks(
                            &doc.text,
                            doc.source.or_else(|| Some(display.clone())),
                            doc.source_type,
                            collection.clone(),
                            doc.tags,
                            doc.project,
                            doc.last_modified,
                            Some(&display),
                        );
                        file_chunk_count += pieces.len();
                        all_chunks.extend(pieces);
                    }
                    ok_files += 1;
                    file_reports.push(serde_json::json!({
                        "path": display,
                        "ok": true,
                        "error": null,
                        "chunks": file_chunk_count,
                    }));
                }
                Err(e) => {
                    log::warn!("skipping {display}: {e}");
                    file_reports.push(serde_json::json!({
                        "path": display,
                        "ok": false,
                        "error": e.to_string(),
                        "chunks": 0,
                    }));
                }
            }
        }

        if all_chunks.is_empty() {
            return Ok(serde_json::json!({
                "status": "no files ingested",
                "files": ok_files,
                "file_results": file_reports,
                "inserted": 0,
                "skipped": 0,
                "replaced": 0,
                "chunks": 0,
            }));
        }

        let report = core
            .embed_and_upsert_batch(all_chunks)
            .await
            .map_err(|e| McpError::internal(format!("upsert failed: {e}")))?;

        log::info!(
            "ingested {} ok files ({} chunks) into '{}' (inserted={} skipped={} replaced={})",
            ok_files,
            report.chunks,
            collection,
            report.inserted,
            report.skipped,
            report.replaced
        );
        Ok(serde_json::json!({
            "status": "ok",
            "files": ok_files,
            "file_results": file_reports,
            "collection": collection,
            "inserted": report.inserted,
            "skipped": report.skipped,
            "replaced": report.replaced,
            "chunks": report.chunks,
        }))
    }

    /// Batch ingest: optional lists of texts, paths, and/or URLs into one collection.
    ///
    /// Returns per-item results under `file_results` plus aggregate insert/skip/replace stats.
    /// All items are extracted/chunked first, then a single upsert batch is applied.
    #[tool]
    async fn ingest_many(
        &self,
        collection: Option<String>,
        texts: Option<Vec<String>>,
        paths: Option<Vec<String>>,
        urls: Option<Vec<String>>,
        tags: Option<Vec<String>>,
        project: Option<String>,
    ) -> McpResult<Value> {
        let collection = collection.unwrap_or_else(|| DEFAULT_COLLECTION_NAME.to_string());
        self.ensure_collection(&collection).await?;

        let mut all_chunks: Vec<DocumentChunk> = Vec::new();
        let mut file_results: Vec<Value> = Vec::new();

        if let Some(texts) = texts {
            for (i, text) in texts.into_iter().enumerate() {
                let src = format!("ingest_many://text/{i}");
                let pieces = self.expand_to_chunks(
                    &text,
                    Some(src.clone()),
                    Some("text".to_string()),
                    collection.clone(),
                    tags.clone(),
                    project.clone(),
                    None,
                    None,
                );
                let n = pieces.len();
                all_chunks.extend(pieces);
                file_results.push(serde_json::json!({
                    "path": src,
                    "ok": true,
                    "error": null,
                    "chunks": n,
                }));
            }
        }

        if let Some(paths) = paths {
            for path in paths {
                let p = std::path::Path::new(&path);
                match lqm_ingest::extract_file(p, serde_json::json!({})) {
                    Ok(extracted) => {
                        let mut n = 0usize;
                        for doc in extracted {
                            let pieces = self.expand_to_chunks(
                                &doc.text,
                                doc.source.or_else(|| Some(path.clone())),
                                doc.source_type,
                                collection.clone(),
                                tags.clone().or(doc.tags),
                                project.clone().or(doc.project),
                                doc.last_modified,
                                Some(&path),
                            );
                            n += pieces.len();
                            all_chunks.extend(pieces);
                        }
                        file_results.push(serde_json::json!({
                            "path": path,
                            "ok": true,
                            "error": null,
                            "chunks": n,
                        }));
                    }
                    Err(e) => {
                        file_results.push(serde_json::json!({
                            "path": path,
                            "ok": false,
                            "error": e.to_string(),
                            "chunks": 0,
                        }));
                    }
                }
            }
        }

        if let Some(urls) = urls {
            for url in urls {
                match lqm_ingest::fetch_url(&url, None).await {
                    Ok(fetched) => {
                        let st = fetched.source_type.clone();
                        let pieces = self.expand_to_chunks(
                            &fetched.text,
                            Some(url.clone()),
                            Some(st),
                            collection.clone(),
                            tags.clone(),
                            project.clone(),
                            None,
                            Some(&url),
                        );
                        let n = pieces.len();
                        all_chunks.extend(pieces);
                        file_results.push(serde_json::json!({
                            "path": url,
                            "ok": true,
                            "error": null,
                            "chunks": n,
                            "title": fetched.title,
                        }));
                    }
                    Err(e) => {
                        file_results.push(serde_json::json!({
                            "path": url,
                            "ok": false,
                            "error": e.to_string(),
                            "chunks": 0,
                        }));
                    }
                }
            }
        }

        if all_chunks.is_empty() {
            return Ok(serde_json::json!({
                "status": "no items ingested",
                "file_results": file_results,
                "inserted": 0,
                "skipped": 0,
                "replaced": 0,
                "chunks": 0,
                "collection": collection,
            }));
        }

        let report = self
            .core()
            .embed_and_upsert_batch(all_chunks)
            .await
            .map_err(|e| McpError::internal(format!("ingest_many upsert failed: {e}")))?;

        Ok(serde_json::json!({
            "status": "ok",
            "collection": collection,
            "file_results": file_results,
            "inserted": report.inserted,
            "skipped": report.skipped,
            "replaced": report.replaced,
            "chunks": report.chunks,
        }))
    }

    /// List distinct document sources in a collection (for agent curation).
    #[tool]
    async fn list_sources(&self, collection: Option<String>) -> McpResult<Value> {
        let collection = collection.unwrap_or_else(|| DEFAULT_COLLECTION_NAME.to_string());
        match self.core().list_sources(&collection).await {
            Ok(sources) => Ok(serde_json::json!({
                "status": "ok",
                "collection": collection,
                "sources": sources,
            })),
            Err(e) => Err(McpError::internal(format!("list_sources failed: {e}"))),
        }
    }

    /// Delete all points for a given source within a collection.
    #[tool]
    async fn delete_by_source(
        &self,
        source: String,
        collection: Option<String>,
    ) -> McpResult<Value> {
        let collection = collection.unwrap_or_else(|| DEFAULT_COLLECTION_NAME.to_string());
        match self.core().delete_by_source(&collection, &source).await {
            Ok(deleted) => Ok(serde_json::json!({
                "status": "ok",
                "collection": collection,
                "source": source,
                "deleted": deleted,
            })),
            Err(e) => Err(McpError::internal(format!("delete_by_source failed: {e}"))),
        }
    }

    /// Delete points matching payload filters (AND of provided fields).
    #[tool]
    async fn delete_by_filter(
        &self,
        collection: Option<String>,
        source: Option<String>,
        source_type: Option<String>,
        project: Option<String>,
        tags: Option<Vec<String>>,
    ) -> McpResult<Value> {
        let collection = collection.unwrap_or_else(|| DEFAULT_COLLECTION_NAME.to_string());
        let filter = PayloadFilter {
            source,
            source_type,
            project,
            tags,
        };
        match self.core().delete_by_filter(&collection, &filter).await {
            Ok(deleted) => Ok(serde_json::json!({
                "status": "ok",
                "collection": collection,
                "deleted": deleted,
                "filter": filter,
            })),
            Err(e) => Err(McpError::internal(format!("delete_by_filter failed: {e}"))),
        }
    }
}

#[cfg(test)]
fn test_qdrant_url() -> String {
    // Prefer 127.0.0.1: some Windows proxies intercept "localhost" HTTP and break readiness probes;
    // gRPC to the same host is what the app uses.
    std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://127.0.0.1:6334".to_string())
}

#[cfg(test)]
async fn create_test_server() -> Option<LqmServer> {
    match create_test_server_detailed().await {
        Ok(s) => Some(s),
        Err(e) => {
            eprintln!("create_test_server skipped: {e}");
            None
        }
    }
}

#[cfg(test)]
async fn create_test_server_detailed() -> Result<LqmServer, String> {
    let qdrant_url = test_qdrant_url();
    let config = EmbedderConfig::load_or_default(None).map_err(|e| format!("config: {e}"))?;
    let embedder = create_embedder(&config).map_err(|e| format!("embedder: {e}"))?;
    let qdrant = lqm_core::QdrantClient::new(&qdrant_url)
        .await
        .map_err(|e| format!("qdrant connect {qdrant_url}: {e}"))?;
    let core = RagCore::from_config(qdrant, embedder, &RagConfig::default());
    core.list_collections()
        .await
        .map_err(|e| format!("list_collections {qdrant_url}: {e}"))?;
    Ok(LqmServer::new(core).await)
}

/// Live-smoke server selection for CI-safe workspace tests.
///
/// - Default: return `None` when Qdrant/embedder is unavailable (skip live smoke).
/// - `LQM_LIVE=1`: hard-require live Qdrant — panic if unreachable (intentional smoke).
#[cfg(test)]
async fn live_test_server() -> Option<LqmServer> {
    match create_test_server_detailed().await {
        Ok(server) => Some(server),
        Err(e) => {
            let hard = std::env::var("LQM_LIVE").ok().as_deref() == Some("1");
            if hard {
                panic!(
                    "LQM_LIVE=1 but Qdrant/embedder unavailable at {}: {e}",
                    test_qdrant_url()
                );
            }
            eprintln!(
                "skipping live smoke (Qdrant unavailable at {}): {e}",
                test_qdrant_url()
            );
            None
        }
    }
}

/// Minimal local HTTP fixture for live `ingest_url` (no public internet required).
#[cfg(test)]
async fn spawn_fixture_http_server() -> (String, tokio::task::JoinHandle<()>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fixture http");
    let addr = listener.local_addr().expect("local_addr");
    let handle = tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                break;
            };
            let mut buf = [0u8; 1024];
            let _ = socket.read(&mut buf).await;
            let body = b"<html><head><title>Smoke</title><script>evil()</script></head>\
<body><h1>Smoke Fixture Page</h1>\
<p>Liberado Qdrant MCP smoke test content about vector search and curated context.</p>\
</body></html>";
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = socket.write_all(header.as_bytes()).await;
            let _ = socket.write_all(body).await;
        }
    });
    (format!("http://127.0.0.1:{}", addr.port()), handle)
}

/// Live smoke: every `#[tool]` on `LqmServer` against real Qdrant when available.
///
/// Skips when Qdrant is down (CI has no Qdrant service). Set `LQM_LIVE=1` to hard-require.
#[tokio::test]
async fn test_all_mcp_tools_live_smoke() {
    let Some(server) = live_test_server().await else {
        return;
    };
    let coll = format!(
        "lqm_smoke_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    );
    let _ = server.delete_collection(coll.clone()).await;

    // 1) create_collection
    let created = server
        .create_collection(coll.clone(), None)
        .await
        .expect("create_collection");
    assert_eq!(created["status"], "ok", "create_collection: {created}");
    assert_eq!(created["created"], true, "create_collection: {created}");
    assert!(
        created["vector_dim"].as_u64().unwrap_or(0) > 0,
        "create_collection dim: {created}"
    );

    // 2) list_collections — must include smoke collection
    let listed = server.list_collections().await.expect("list_collections");
    let names = listed["collections"].as_array().expect("collections array");
    assert!(
        names.iter().any(|n| n.as_str() == Some(coll.as_str())),
        "list_collections missing {coll}: {listed}"
    );

    // 3) get_collection_info
    let info = server
        .get_collection_info(coll.clone())
        .await
        .expect("get_collection_info");
    assert_eq!(info["exists"], true, "get_collection_info: {info}");
    assert!(
        info["vector_size"].as_u64().is_some(),
        "vector_size missing: {info}"
    );

    // 4) ingest_text
    let text_body =
        "Unique smoke sentence about orange kites and vector retrieval for agents.".to_string();
    let ingested = server
        .ingest_text(
            text_body.clone(),
            Some("smoke://text".to_string()),
            Some("text".to_string()),
            Some(coll.clone()),
            Some(vec!["smoke".to_string()]),
            Some("lqm_smoke".to_string()),
            None,
        )
        .await
        .expect("ingest_text");
    assert_eq!(ingested["status"], "ok", "ingest_text: {ingested}");
    assert!(
        ingested["chunks"].as_u64().unwrap_or(0) >= 1,
        "ingest_text chunks: {ingested}"
    );
    assert_eq!(ingested["inserted"].as_u64().unwrap_or(0), 1);

    // 4b) re-ingest same content → skip
    let skipped = server
        .ingest_text(
            text_body.clone(),
            Some("smoke://text".to_string()),
            Some("text".to_string()),
            Some(coll.clone()),
            Some(vec!["smoke".to_string()]),
            Some("lqm_smoke".to_string()),
            None,
        )
        .await
        .expect("ingest_text skip");
    assert_eq!(
        skipped["skipped"].as_u64().unwrap_or(0),
        1,
        "skip: {skipped}"
    );
    assert_eq!(
        skipped["chunks"].as_u64().unwrap_or(0),
        0,
        "skip chunks: {skipped}"
    );

    // 4c) re-ingest changed content → replace
    let replaced = server
        .ingest_text(
            "Updated smoke sentence about orange kites (v2).".to_string(),
            Some("smoke://text".to_string()),
            Some("text".to_string()),
            Some(coll.clone()),
            Some(vec!["smoke".to_string()]),
            Some("lqm_smoke".to_string()),
            None,
        )
        .await
        .expect("ingest_text replace");
    assert_eq!(
        replaced["replaced"].as_u64().unwrap_or(0),
        1,
        "replace: {replaced}"
    );
    assert!(replaced["chunks"].as_u64().unwrap_or(0) >= 1);

    // 4d) list_sources
    let sources = server
        .list_sources(Some(coll.clone()))
        .await
        .expect("list_sources");
    assert_eq!(sources["status"], "ok");
    let src_arr = sources["sources"].as_array().expect("sources array");
    assert!(
        src_arr
            .iter()
            .any(|s| s["source"].as_str() == Some("smoke://text")),
        "list_sources missing smoke://text: {sources}"
    );

    // 5) ingest_path — real temp file
    let path_dir = std::env::temp_dir().join(format!("lqm_smoke_path_{coll}"));
    std::fs::create_dir_all(&path_dir).expect("mkdir smoke path");
    let path_file = path_dir.join("smoke_doc.txt");
    std::fs::write(
        &path_file,
        "Path-ingested smoke document discussing blue lanterns and Qdrant collections.\n",
    )
    .expect("write smoke path file");
    let path_ingested = server
        .ingest_path(path_file.to_string_lossy().to_string(), Some(coll.clone()))
        .await
        .expect("ingest_path");
    assert_eq!(
        path_ingested["status"], "ok",
        "ingest_path: {path_ingested}"
    );
    assert!(
        path_ingested["files"].as_u64().unwrap_or(0) >= 1,
        "ingest_path files: {path_ingested}"
    );
    assert!(
        path_ingested["chunks"].as_u64().unwrap_or(0) >= 1,
        "ingest_path chunks: {path_ingested}"
    );
    let _ = std::fs::remove_dir_all(&path_dir);

    // 6) ingest_url — local fixture HTTP server (real GET)
    let (fixture_url, http_handle) = spawn_fixture_http_server().await;
    let url_ingested = server
        .ingest_url(
            fixture_url.clone(),
            Some(coll.clone()),
            Some(vec!["url-smoke".to_string()]),
            None,
            None,
            None,
        )
        .await
        .expect("ingest_url");
    http_handle.abort();
    assert_eq!(url_ingested["status"], "ok", "ingest_url: {url_ingested}");
    assert!(
        url_ingested["chunks"].as_u64().unwrap_or(0) >= 1,
        "ingest_url chunks: {url_ingested}"
    );

    // 7) search — expect non-empty after ingest; pagination fields present
    let search = server
        .search(
            "orange kites vector retrieval".to_string(),
            Some(coll.clone()),
            Some(5),
            None, // offset
            None, // tags
            None, // tags_should
            None, // tags_must_not
            None, // source
            None, // source_type
            None, // project
            None, // min_score
        )
        .await
        .expect("search");
    let results = search["results"].as_array().expect("results array");
    assert!(
        !results.is_empty(),
        "search expected non-empty results: {search}"
    );
    assert!(
        results[0].get("score").is_some() && results[0].get("text").is_some(),
        "search hit missing score/text: {search}"
    );
    assert!(
        search.get("has_more").is_some(),
        "missing has_more: {search}"
    );
    assert!(search.get("offset").is_some(), "missing offset: {search}");

    // 7b) filtered search by source
    let filtered = server
        .search(
            "orange kites".to_string(),
            Some(coll.clone()),
            Some(5),
            None,
            None,
            None,
            None,
            Some("smoke://text".to_string()),
            None,
            None,
            None,
        )
        .await
        .expect("filtered search");
    assert!(
        filtered["results"].as_array().is_some(),
        "filtered search: {filtered}"
    );

    // 8) get_relevant_context — markdown with passages/scores + budget/mmr fields
    let ctx = server
        .get_relevant_context(
            "vector search curated context".to_string(),
            Some(coll.clone()),
            Some(5),
            None, // offset
            None, // tags
            None,
            None,
            None, // source
            None,
            None,
            None,        // min_score
            Some(200),   // max_chars_per_passage
            Some(4000),  // max_total_chars
            Some(false), // mmr
            None,
        )
        .await
        .expect("get_relevant_context");
    assert_eq!(ctx["status"], "ok", "get_relevant_context: {ctx}");
    let context = ctx["context"].as_str().expect("context string");
    assert!(!context.is_empty(), "empty context: {ctx}");
    assert!(
        context.contains("Passage") || context.contains("score"),
        "context missing passage/score structure: {context}"
    );
    assert!(
        ctx["passage_count"].as_u64().unwrap_or(0) >= 1,
        "passage_count: {ctx}"
    );
    assert!(ctx.get("truncated_by_budget").is_some());

    // 9) delete_by_source — curation without wiping the collection
    let del_src = server
        .delete_by_source("smoke://text".to_string(), Some(coll.clone()))
        .await
        .expect("delete_by_source");
    assert!(
        del_src["deleted"].as_u64().unwrap_or(0) >= 1,
        "delete_by_source: {del_src}"
    );
    let sources_after = server
        .list_sources(Some(coll.clone()))
        .await
        .expect("list_sources after delete");
    let remaining = sources_after["sources"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        !remaining
            .iter()
            .any(|s| s["source"].as_str() == Some("smoke://text")),
        "source should be gone: {sources_after}"
    );

    // 10) delete_by_filter on tags (path ingest used no tags on file source; filter by source_type text from path may remain)
    let _ = server
        .delete_by_filter(
            Some(coll.clone()),
            None,
            Some("text".to_string()),
            None,
            None,
        )
        .await
        .expect("delete_by_filter");

    // 11) delete_collection cleanup
    let deleted = server
        .delete_collection(coll.clone())
        .await
        .expect("delete_collection");
    assert_eq!(deleted["deleted"], true, "delete_collection: {deleted}");
    let gone = server
        .get_collection_info(coll)
        .await
        .expect("get after delete");
    assert_eq!(gone["exists"], false, "collection should be gone: {gone}");
}

/// P0 lifecycle smoke: list_sources, skip/replace, delete_by_source (skips if no Qdrant).
#[tokio::test]
async fn test_p0_lifecycle_live_smoke() {
    let Some(server) = live_test_server().await else {
        return;
    };
    let coll = "lqm_smoke_p0_lifecycle";
    let _ = server.delete_collection(coll.to_string()).await;
    server
        .create_collection(coll.to_string(), None)
        .await
        .expect("create");

    let src = "p0://doc-a";
    let v1 = "P0 lifecycle document version one about curated sources.";
    let first = server
        .ingest_text(
            v1.to_string(),
            Some(src.to_string()),
            Some("text".to_string()),
            Some(coll.to_string()),
            Some(vec!["p0".to_string()]),
            Some("proj-p0".to_string()),
            None,
        )
        .await
        .expect("ingest v1");
    assert_eq!(first["inserted"], 1);
    assert_eq!(first["chunks"], 1);

    let listed = server
        .list_sources(Some(coll.to_string()))
        .await
        .expect("list");
    assert!(
        listed["sources"]
            .as_array()
            .unwrap()
            .iter()
            .any(|s| s["source"] == src && s["count"] == 1),
        "{listed}"
    );

    let skip = server
        .ingest_text(
            v1.to_string(),
            Some(src.to_string()),
            Some("text".to_string()),
            Some(coll.to_string()),
            Some(vec!["p0".to_string()]),
            Some("proj-p0".to_string()),
            None,
        )
        .await
        .expect("skip");
    assert_eq!(skip["skipped"], 1, "{skip}");
    assert_eq!(skip["chunks"], 0, "{skip}");

    let v2 = "P0 lifecycle document version two — replaced content.";
    let rep = server
        .ingest_text(
            v2.to_string(),
            Some(src.to_string()),
            Some("text".to_string()),
            Some(coll.to_string()),
            Some(vec!["p0".to_string()]),
            Some("proj-p0".to_string()),
            None,
        )
        .await
        .expect("replace");
    assert_eq!(rep["replaced"], 1, "{rep}");
    assert_eq!(rep["chunks"], 1, "{rep}");

    let del = server
        .delete_by_source(src.to_string(), Some(coll.to_string()))
        .await
        .expect("delete_by_source");
    assert!(del["deleted"].as_u64().unwrap_or(0) >= 1, "{del}");

    let empty = server
        .list_sources(Some(coll.to_string()))
        .await
        .expect("list empty");
    assert!(
        empty["sources"]
            .as_array()
            .map(|a| a.is_empty())
            .unwrap_or(false),
        "{empty}"
    );

    // delete_by_filter validation
    let bad = server
        .delete_by_filter(Some(coll.to_string()), None, None, None, None)
        .await;
    assert!(bad.is_err(), "empty filter should fail");

    let _ = server.delete_collection(coll.to_string()).await;
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
    let dim = core.embedder.dimension();

    let coll = "lqm_mcp_test_ingest";
    let _ = core.delete_collection(coll).await;
    core.ensure_collection(coll, dim)
        .await
        .expect("ensure_collection");

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
    core.embed_and_upsert_batch(vec![chunk])
        .await
        .expect("upsert");

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
    let dim = core.embedder.dimension();

    let coll = "lqm_mcp_test_edge";
    let _ = core.delete_collection(coll).await;
    core.ensure_collection(coll, dim)
        .await
        .expect("ensure_collection");

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
    core.embed_and_upsert_batch(vec![chunk])
        .await
        .expect("upsert");
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
    let dim = core.embedder.dimension();

    let coll_a = "lqm_mcp_test_a";
    let coll_b = "lqm_mcp_test_b";
    let _ = core.delete_collection(coll_a).await;
    let _ = core.delete_collection(coll_b).await;
    core.ensure_collection(coll_a, dim).await.expect("ensure a");
    core.ensure_collection(coll_b, dim).await.expect("ensure b");

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
    core.embed_and_upsert_batch(vec![chunk_a])
        .await
        .expect("upsert a");
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
    core.embed_and_upsert_batch(vec![chunk_b])
        .await
        .expect("upsert b");

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
