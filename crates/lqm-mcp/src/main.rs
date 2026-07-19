use clap::{Parser, Subcommand};
use lqm_core::RagCore;
use lqm_core::constants;
use lqm_core::format_relevant_context_with;
use lqm_core::types::{
    ContextOptions, DocumentChunk, PayloadFilter, SearchFilter, SearchOptions, make_file_result,
    resolve_collection,
};
use lqm_core::{DEFAULT_MEMORY_COLLECTION, MemoryNote};
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
        self.core
            .ensure_collection(collection, None)
            .await
            .map_err(|e| {
                McpError::internal(format!("could not ensure collection '{collection}': {e}"))
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
        scope: Option<String>,
        clearance: Option<Clearance>,
    ) -> Vec<DocumentChunk> {
        self.core().expand_to_chunks(
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
        scope: Option<String>,
        clearance: Option<String>,
    ) -> McpResult<Value> {
        let core = self.core();
        let collection = resolve_collection(collection);

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
            scope,
            clearance: clearance.and_then(|s| s.parse().ok()),
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
    ///
    /// Set `hybrid=true` to fuse dense similarity with keyword overlap on chunk
    /// text (helps rare tokens dense-only can miss). Dense-only is the default.
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
        hybrid: Option<bool>,
        hybrid_alpha: Option<f32>,
        scope: Option<String>,
        max_clearance: Option<String>,
    ) -> McpResult<Value> {
        let hybrid_on = hybrid.unwrap_or(false);
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
                        scope,
                        max_clearance: max_clearance
                            .map(|s| {
                                s.parse::<Clearance>().map_err(|e| {
                                    McpError::invalid_request(format!(
                                        "invalid max_clearance '{}': {e}",
                                        s
                                    ))
                                })
                            })
                            .transpose()?,
                    },
                    hybrid: hybrid_on,
                    hybrid_alpha,
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
            "hybrid": hybrid_on,
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
        hybrid: Option<bool>,
        hybrid_alpha: Option<f32>,
        scope: Option<String>,
        max_clearance: Option<String>,
    ) -> McpResult<Value> {
        let coll = resolve_collection(collection.clone());
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
                        scope,
                        max_clearance: max_clearance
                            .map(|s| {
                                s.parse::<Clearance>().map_err(|e| {
                                    McpError::invalid_request(format!(
                                        "invalid max_clearance '{}': {e}",
                                        s
                                    ))
                                })
                            })
                            .transpose()?,
                    },
                    hybrid: hybrid.unwrap_or(false),
                    hybrid_alpha,
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
        let collection = resolve_collection(collection);
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
            None,
            None,
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
        let collection = resolve_collection(collection);
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
            match lqm_ingest::extract_file_async(&p, serde_json::json!({})).await {
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
                            None,
                            None,
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
        let collection = resolve_collection(collection);
        self.ensure_collection(&collection).await?;

        let mut all_chunks: Vec<DocumentChunk> = Vec::new();
        let mut file_results: Vec<Value> = Vec::new();

        if let Some(texts) = texts {
            for (i, text) in texts.into_iter().enumerate() {
                let src = format!("ingest_many://text/{i}");
                let pieces = self.expand_to_chunks(
                    &text,
                    Some(src.clone()),
                    Some(constants::SOURCE_TYPE_TEXT.to_string()),
                    collection.clone(),
                    tags.clone(),
                    project.clone(),
                    None,
                    None,
                    None,
                    None,
                );
                let n = pieces.len();
                all_chunks.extend(pieces);
                file_results.push(make_file_result(&src, true, None, n));
            }
        }

        if let Some(paths) = paths {
            for path in paths {
                let p = std::path::Path::new(&path);
                match lqm_ingest::extract_file_async(p, serde_json::json!({})).await {
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
                                None,
                                None,
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
                            None,
                            None,
                        );
                        let n = pieces.len();
                        all_chunks.extend(pieces);
                        let mut row = make_file_result(&url, true, None, n);
                        if let Some(obj) = row.as_object_mut() {
                            obj.insert(
                                "title".into(),
                                serde_json::to_value(fetched.title)
                                    .unwrap_or(serde_json::Value::Null),
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

    /// Active embedder identity (id, dimension, model) for diagnosing dim mismatches.
    #[tool]
    async fn get_embedder_info(&self) -> McpResult<Value> {
        let info = self.core().embedder_info();
        Ok(serde_json::json!({
            "status": "ok",
            "id": info.id,
            "dimension": info.dimension,
            "model": info.model,
        }))
    }

    /// Store a long-term agent memory (default collection `memories`).
    ///
    /// Host agent keeps generation; this only persists text + metadata for later recall.
    #[tool]
    async fn store_memory(
        &self,
        text: String,
        importance: Option<f32>,
        tags: Option<Vec<String>>,
        project: Option<String>,
        memory_id: Option<String>,
        collection: Option<String>,
    ) -> McpResult<Value> {
        let coll = collection.unwrap_or_else(|| DEFAULT_MEMORY_COLLECTION.to_string());
        let note = MemoryNote {
            text,
            importance,
            tags,
            project,
            memory_id,
        };
        match self.core().store_memory(note, Some(&coll)).await {
            Ok((report, effective_id)) => Ok(serde_json::json!({
                "status": "ok",
                "collection": coll,
                "memory_id": effective_id,
                "inserted": report.inserted,
                "skipped": report.skipped,
                "replaced": report.replaced,
                "chunks": report.chunks,
            })),
            Err(e) => Err(McpError::internal(format!("store_memory failed: {e}"))),
        }
    }

    /// Recall memories by semantic similarity; optional recency/importance blend.
    #[tool]
    async fn recall_memories(
        &self,
        query: String,
        collection: Option<String>,
        limit: Option<u64>,
        use_recency: Option<bool>,
        project: Option<String>,
        tags: Option<Vec<String>>,
    ) -> McpResult<Value> {
        let coll = collection.unwrap_or_else(|| DEFAULT_MEMORY_COLLECTION.to_string());
        match self
            .core()
            .recall_memories(
                &query,
                Some(&coll),
                limit,
                use_recency.unwrap_or(true),
                project.as_deref(),
                tags,
            )
            .await
        {
            Ok(hits) => Ok(serde_json::json!({
                "status": "ok",
                "collection": coll,
                "query": query,
                "count": hits.len(),
                "memories": hits,
            })),
            Err(e) => Err(McpError::internal(format!("recall_memories failed: {e}"))),
        }
    }

    /// List distinct document sources in a collection (for agent curation).
    #[tool]
    async fn list_sources(&self, collection: Option<String>) -> McpResult<Value> {
        let collection = resolve_collection(collection);
        match self.core().list_sources(&collection).await {
            Ok(sources) => Ok(serde_json::json!({
                "status": "ok",
                "collection": collection,
                "sources": sources,
            })),
            Err(e) => Err(McpError::internal(format!("list_sources failed: {e}"))),
        }
    }

    /// List indexed chunks for a source ordered by `chunk_index` (paginated).
    ///
    /// Reconstructs a parent document from the index without re-searching.
    /// `source` is a pointer (path/URL/id), not a blob. Missing chunk_index sorts last.
    #[tool]
    async fn list_chunks(
        &self,
        source: String,
        collection: Option<String>,
        offset: Option<u64>,
        limit: Option<u64>,
    ) -> McpResult<Value> {
        let collection = resolve_collection(collection);
        match self
            .core()
            .list_chunks(&collection, &source, offset, limit)
            .await
        {
            Ok(page) => Ok(serde_json::json!({
                "status": "ok",
                "collection": collection,
                "source": page.source,
                "chunks": page.chunks,
                "offset": page.offset,
                "limit": page.limit,
                "total": page.total,
                "has_more": page.has_more,
                "next_offset": page.next_offset,
            })),
            Err(e) => Err(McpError::internal(format!("list_chunks failed: {e}"))),
        }
    }

    /// Reconstruct a full source: all chunks in index order plus joined text.
    #[tool]
    async fn get_source(&self, source: String, collection: Option<String>) -> McpResult<Value> {
        let collection = resolve_collection(collection);
        match self.core().get_source(&collection, &source).await {
            Ok(doc) => Ok(serde_json::json!({
                "status": "ok",
                "collection": collection,
                "source": doc.source,
                "source_type": doc.source_type,
                "total": doc.total,
                "text": doc.text,
                "chunks": doc.chunks,
            })),
            Err(e) => Err(McpError::internal(format!("get_source failed: {e}"))),
        }
    }

    /// Neighboring chunks of the same source around `chunk_index` (±`neighbors`).
    ///
    /// Use after a search hit to expand context without inventing content.
    #[tool]
    async fn expand_context(
        &self,
        source: String,
        chunk_index: u64,
        collection: Option<String>,
        neighbors: Option<u64>,
    ) -> McpResult<Value> {
        let collection = resolve_collection(collection);
        match self
            .core()
            .expand_context(&collection, &source, chunk_index, neighbors)
            .await
        {
            Ok(chunks) => Ok(serde_json::json!({
                "status": "ok",
                "collection": collection,
                "source": source,
                "chunk_index": chunk_index,
                "neighbors": neighbors.unwrap_or(lqm_core::DEFAULT_EXPAND_NEIGHBORS),
                "count": chunks.len(),
                "chunks": chunks,
            })),
            Err(e) => Err(McpError::internal(format!("expand_context failed: {e}"))),
        }
    }

    /// Delete all points for a given source within a collection.
    #[tool]
    async fn delete_by_source(
        &self,
        source: String,
        collection: Option<String>,
    ) -> McpResult<Value> {
        let collection = resolve_collection(collection);
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
    #[allow(clippy::too_many_arguments)]
    async fn delete_by_filter(
        &self,
        collection: Option<String>,
        source: Option<String>,
        source_type: Option<String>,
        project: Option<String>,
        tags: Option<Vec<String>>,
        scope: Option<String>,
        max_clearance: Option<String>,
    ) -> McpResult<Value> {
        let collection = resolve_collection(collection);
        let filter = PayloadFilter {
            source,
            source_type,
            project,
            tags,
            scope,
            max_clearance: max_clearance
                .map(|s| {
                    s.parse::<Clearance>().map_err(|e| {
                        McpError::invalid_request(format!("invalid max_clearance '{}': {e}", s))
                    })
                })
                .transpose()?,
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
    let core = RagCore::from_env(Some(&qdrant_url), None)
        .await
        .map_err(|e| format!("from_env {qdrant_url}: {e}"))?;
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
            Some(constants::SOURCE_TYPE_TEXT.to_string()),
            Some(coll.clone()),
            Some(vec!["smoke".to_string()]),
            Some("lqm_smoke".to_string()),
            None,
            None,
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
            Some(constants::SOURCE_TYPE_TEXT.to_string()),
            Some(coll.clone()),
            Some(vec!["smoke".to_string()]),
            Some("lqm_smoke".to_string()),
            None,
            None,
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
            Some(constants::SOURCE_TYPE_TEXT.to_string()),
            Some(coll.clone()),
            Some(vec!["smoke".to_string()]),
            Some("lqm_smoke".to_string()),
            None,
            None,
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

    // 4e) list_chunks / get_source / expand_context (source reconstruction)
    let listed_chunks = server
        .list_chunks(
            "smoke://text".to_string(),
            Some(coll.clone()),
            Some(0),
            Some(10),
        )
        .await
        .expect("list_chunks");
    assert_eq!(listed_chunks["status"], "ok", "{listed_chunks}");
    assert!(
        listed_chunks["total"].as_u64().unwrap_or(0) >= 1,
        "list_chunks total: {listed_chunks}"
    );
    let got_src = server
        .get_source("smoke://text".to_string(), Some(coll.clone()))
        .await
        .expect("get_source");
    assert_eq!(got_src["status"], "ok", "{got_src}");
    assert!(
        got_src["text"]
            .as_str()
            .unwrap_or("")
            .contains("orange kites"),
        "get_source text: {got_src}"
    );
    let expanded = server
        .expand_context("smoke://text".to_string(), 0, Some(coll.clone()), Some(1))
        .await
        .expect("expand_context");
    assert_eq!(expanded["status"], "ok", "{expanded}");
    assert!(
        expanded["count"].as_u64().unwrap_or(0) >= 1,
        "expand_context: {expanded}"
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
            None, // hybrid
            None, // hybrid_alpha
            None, // scope
            None, // max_clearance
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
    assert_eq!(search["hybrid"], false);

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
            None,
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
            None, // hybrid
            None,
            None,
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
            Some(constants::SOURCE_TYPE_TEXT.to_string()),
            None,
            None,
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

/// P4 memory store → recall (skips if no Qdrant).
#[tokio::test]
async fn test_p4_memory_live_smoke() {
    let Some(server) = live_test_server().await else {
        return;
    };
    let coll = "lqm_smoke_memories_p4";
    let _ = server.delete_collection(coll.to_string()).await;

    let store = server
        .store_memory(
            "The user prefers concise Rust examples with Qdrant filters.".to_string(),
            Some(0.85),
            Some(vec!["prefs".to_string()]),
            Some("agent".to_string()),
            Some("pref-rust-style".to_string()),
            Some(coll.to_string()),
        )
        .await
        .expect("store_memory");
    assert_eq!(store["status"], "ok", "{store}");
    assert_eq!(
        store["memory_id"].as_str(),
        Some("pref-rust-style"),
        "effective memory_id: {store}"
    );
    assert!(
        store["chunks"].as_u64().unwrap_or(0) >= 1 || store["skipped"].as_u64().unwrap_or(0) >= 1,
        "{store}"
    );

    let recall = server
        .recall_memories(
            "Rust code style preferences".to_string(),
            Some(coll.to_string()),
            Some(5),
            Some(true),
            None,
            None,
        )
        .await
        .expect("recall_memories");
    assert_eq!(recall["status"], "ok", "{recall}");
    let mems = recall["memories"].as_array().expect("memories array");
    assert!(!mems.is_empty(), "expected recall hits: {recall}");
    let text = mems[0]["text"].as_str().unwrap_or("");
    assert!(
        text.to_lowercase().contains("rust") || text.to_lowercase().contains("qdrant"),
        "unexpected text: {text}"
    );
    assert!(mems[0].get("blended_score").is_some());
    assert!(mems[0].get("importance").is_some());

    let _ = server.delete_collection(coll.to_string()).await;
}

/// P5 hybrid search: rare keyword doc surfaces under hybrid=true (skips if no Qdrant).
#[tokio::test]
async fn test_p5_hybrid_live_smoke() {
    let Some(server) = live_test_server().await else {
        return;
    };
    let coll = "lqm_smoke_hybrid_p5";
    let _ = server.delete_collection(coll.to_string()).await;
    server
        .create_collection(coll.to_string(), None)
        .await
        .expect("create_collection");

    // Doc without the rare token.
    let _ = server
        .ingest_text(
            "Fluffy clouds drift over the quiet mountain lake at dawn.".to_string(),
            Some("smoke://hybrid-generic".to_string()),
            Some(constants::SOURCE_TYPE_TEXT.to_string()),
            Some(coll.to_string()),
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("ingest generic");

    // Doc with distinctive rare keyword that pure dense may bury.
    let rare = "rarekeytokenxqzv9f3a";
    let _ = server
        .ingest_text(
            format!("Agent notes about Liberado retrieval and the token {rare} for hybrid smoke."),
            Some("smoke://hybrid-keyword".to_string()),
            Some(constants::SOURCE_TYPE_TEXT.to_string()),
            Some(coll.to_string()),
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("ingest keyword doc");

    let hybrid = server
        .search(
            rare.to_string(),
            Some(coll.to_string()),
            Some(5),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(true), // hybrid
            Some(0.35), // lean keyword for rare-token queries
            None,       // scope
            None,       // max_clearance
        )
        .await
        .expect("hybrid search");
    assert_eq!(hybrid["hybrid"], true, "{hybrid}");
    let results = hybrid["results"].as_array().expect("results");
    assert!(!results.is_empty(), "hybrid expected hits: {hybrid}");
    let joined: String = results
        .iter()
        .filter_map(|r| r["text"].as_str())
        .collect::<Vec<_>>()
        .join(" | ");
    assert!(
        joined.contains(rare),
        "hybrid results must include rare keyword doc; got: {joined}"
    );
    assert!(results[0].get("score").is_some());
    assert!(results[0].get("payload").is_some() || results[0].get("text").is_some());

    let _ = server.delete_collection(coll.to_string()).await;
}

/// Scoped filtering: two scopes, search constrained to one (skips if no Qdrant).
#[tokio::test]
async fn test_scope_filter_live_smoke() {
    let Some(server) = live_test_server().await else {
        return;
    };
    let coll = "lqm_smoke_scope_filter";
    let _ = server.delete_collection(coll.to_string()).await;
    server
        .create_collection(coll.to_string(), None)
        .await
        .expect("create_collection");

    server
        .ingest_text(
            "Alpha scope secret recipe for applesauce.".to_string(),
            Some("smoke://scope-alpha".to_string()),
            Some(constants::SOURCE_TYPE_TEXT.to_string()),
            Some(coll.to_string()),
            None,
            None,
            None,
            Some("alpha".to_string()),
            Some("internal".to_string()),
        )
        .await
        .expect("ingest alpha");
    server
        .ingest_text(
            "Beta scope notes about bananas only.".to_string(),
            Some("smoke://scope-beta".to_string()),
            Some(constants::SOURCE_TYPE_TEXT.to_string()),
            Some(coll.to_string()),
            None,
            None,
            None,
            Some("beta".to_string()),
            Some("public".to_string()),
        )
        .await
        .expect("ingest beta");

    let scoped = server
        .search(
            "recipe notes applesauce bananas".to_string(),
            Some(coll.to_string()),
            Some(10),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("alpha".to_string()),
            None,
        )
        .await
        .expect("scoped search");
    let results = scoped["results"].as_array().expect("results");
    assert!(!results.is_empty(), "expected alpha hits: {scoped}");
    for r in results {
        let t = r["text"].as_str().unwrap_or("");
        assert!(
            t.to_lowercase().contains("apple") || t.to_lowercase().contains("alpha"),
            "cross-scope leak: {t}"
        );
        assert!(
            !t.to_lowercase().contains("banana"),
            "beta doc leaked into alpha scope: {t}"
        );
        if let Some(p) = r.get("payload")
            && let Some(s) = p.get("scope").and_then(|v| v.as_str())
        {
            assert_eq!(s, "alpha", "payload scope: {p}");
        }
    }

    let cleared = server
        .search(
            "bananas notes".to_string(),
            Some(coll.to_string()),
            Some(10),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("beta".to_string()),
            Some("public".to_string()),
        )
        .await
        .expect("clearance search");
    let cr = cleared["results"].as_array().expect("results");
    assert!(!cr.is_empty(), "{cleared}");
    for r in cr {
        let t = r["text"].as_str().unwrap_or("");
        assert!(
            !t.to_lowercase().contains("applesauce"),
            "alpha leaked: {t}"
        );
    }

    let _ = server.delete_collection(coll.to_string()).await;
}

/// Source reconstruction: multi-chunk ordered list, pagination, expand neighbors.
///
/// Skips when Qdrant is down. Set `LQM_LIVE=1` to hard-require.
#[tokio::test]
async fn test_source_reconstruction_live_smoke() {
    let Some(server) = live_test_server().await else {
        return;
    };
    let coll = "lqm_smoke_reconstruction";
    let src = "recon://multi-doc";
    let _ = server.delete_collection(coll.to_string()).await;
    server
        .create_collection(coll.to_string(), None)
        .await
        .expect("create_collection");

    // Upsert four ordered chunks for one source (via core path agents' ingest uses).
    let mut batch = Vec::new();
    for i in 0..4usize {
        batch.push(DocumentChunk {
            text: format!(
                "Reconstruction chunk index {i} with unique token_recon_{i} for ordering tests."
            ),
            source: Some(src.to_string()),
            source_type: Some(constants::SOURCE_TYPE_TEXT.to_string()),
            collection: Some(coll.to_string()),
            tags: Some(vec!["recon".to_string()]),
            timestamp: None,
            project: None,
            last_modified: None,
            chunk_index: Some(i),
            total_chunks: Some(4),
            importance: None,
            memory_id: None,
            scope: None,
            clearance: None,
        });
    }
    // Insert in reverse order so list_chunks must sort, not rely on upsert order.
    batch.reverse();
    server
        .core()
        .embed_and_upsert_batch(batch)
        .await
        .expect("upsert multi-chunk source");

    // Full ordered page
    let page = server
        .list_chunks(src.to_string(), Some(coll.to_string()), Some(0), Some(10))
        .await
        .expect("list_chunks all");
    assert_eq!(page["status"], "ok", "{page}");
    assert_eq!(page["total"], 4, "{page}");
    assert_eq!(page["has_more"], false, "{page}");
    let chunks = page["chunks"].as_array().expect("chunks");
    assert_eq!(chunks.len(), 4, "{page}");
    for (i, c) in chunks.iter().enumerate() {
        let idx = c["chunk_index"]
            .as_u64()
            .or_else(|| c["chunk_index"].as_str().and_then(|s| s.parse().ok()));
        assert_eq!(idx, Some(i as u64), "ordered chunk_index at {i}: {c}");
        assert!(
            c["text"]
                .as_str()
                .unwrap_or("")
                .contains(&format!("token_recon_{i}")),
            "text at {i}: {c}"
        );
        assert_eq!(c["source"].as_str(), Some(src));
    }

    // Pagination: limit < total
    let p0 = server
        .list_chunks(src.to_string(), Some(coll.to_string()), Some(0), Some(2))
        .await
        .expect("list_chunks page0");
    assert_eq!(p0["total"], 4);
    assert_eq!(p0["has_more"], true, "{p0}");
    assert_eq!(p0["next_offset"], 2, "{p0}");
    assert_eq!(p0["chunks"].as_array().unwrap().len(), 2);
    let p1 = server
        .list_chunks(src.to_string(), Some(coll.to_string()), Some(2), Some(2))
        .await
        .expect("list_chunks page1");
    assert_eq!(p1["has_more"], false, "{p1}");
    assert_eq!(p1["chunks"].as_array().unwrap().len(), 2);
    assert!(
        p1["chunks"][0]["text"]
            .as_str()
            .unwrap_or("")
            .contains("token_recon_2"),
        "{p1}"
    );

    // get_source joins ordered text
    let doc = server
        .get_source(src.to_string(), Some(coll.to_string()))
        .await
        .expect("get_source");
    assert_eq!(doc["total"], 4, "{doc}");
    let text = doc["text"].as_str().unwrap_or("");
    for i in 0..4 {
        assert!(
            text.contains(&format!("token_recon_{i}")),
            "missing token {i} in {text}"
        );
    }
    let pos0 = text.find("token_recon_0").expect("t0");
    let pos3 = text.find("token_recon_3").expect("t3");
    assert!(pos0 < pos3, "joined text should keep index order");

    // expand_context ±1 around chunk 1 → indices 0,1,2 only
    let exp = server
        .expand_context(src.to_string(), 1, Some(coll.to_string()), Some(1))
        .await
        .expect("expand_context");
    assert_eq!(exp["count"], 3, "{exp}");
    let exp_chunks = exp["chunks"].as_array().unwrap();
    let indices: Vec<u64> = exp_chunks
        .iter()
        .map(|c| {
            c["chunk_index"]
                .as_u64()
                .or_else(|| c["chunk_index"].as_str().and_then(|s| s.parse().ok()))
                .expect("chunk_index")
        })
        .collect();
    assert_eq!(indices, vec![0, 1, 2], "{exp}");
    for c in exp_chunks {
        assert_eq!(c["source"].as_str(), Some(src));
    }

    // empty source → validation error (no panic)
    let empty_src = server
        .list_chunks("".to_string(), Some(coll.to_string()), None, None)
        .await;
    assert!(
        empty_src.is_err(),
        "empty source should fail: {empty_src:?}"
    );

    // missing source → empty page
    let missing = server
        .list_chunks(
            "recon://does-not-exist".to_string(),
            Some(coll.to_string()),
            Some(0),
            Some(10),
        )
        .await
        .expect("list missing");
    assert_eq!(missing["total"], 0, "{missing}");
    assert!(
        missing["chunks"]
            .as_array()
            .map(|a| a.is_empty())
            .unwrap_or(false),
        "{missing}"
    );

    let _ = server.delete_collection(coll.to_string()).await;
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
            Some(constants::SOURCE_TYPE_TEXT.to_string()),
            Some(coll.to_string()),
            Some(vec!["p0".to_string()]),
            Some("proj-p0".to_string()),
            None,
            None,
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
            Some(constants::SOURCE_TYPE_TEXT.to_string()),
            Some(coll.to_string()),
            Some(vec!["p0".to_string()]),
            Some("proj-p0".to_string()),
            None,
            None,
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
            Some(constants::SOURCE_TYPE_TEXT.to_string()),
            Some(coll.to_string()),
            Some(vec!["p0".to_string()]),
            Some("proj-p0".to_string()),
            None,
            None,
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
        .delete_by_filter(Some(coll.to_string()), None, None, None, None, None, None)
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
    core.ensure_collection(coll, Some(dim))
        .await
        .expect("ensure_collection");

    let chunk = DocumentChunk {
        text: "Hello world from lqm-mcp test".to_string(),
        source: Some("test_source".to_string()),
        source_type: Some(constants::SOURCE_TYPE_TEXT.to_string()),
        collection: Some(coll.to_string()),
        tags: None,
        timestamp: None,
        project: Some("test_project".to_string()),
        last_modified: None,
        chunk_index: Some(0),
        total_chunks: Some(1),
        importance: None,
        memory_id: None,
        scope: None,
        clearance: None,
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
    core.ensure_collection(coll, Some(dim))
        .await
        .expect("ensure_collection");

    let chunk = DocumentChunk {
        text: "test content".to_string(),
        source: Some("s".to_string()),
        source_type: Some(constants::SOURCE_TYPE_TEXT.to_string()),
        collection: Some(coll.to_string()),
        tags: None,
        timestamp: None,
        project: None,
        last_modified: None,
        chunk_index: Some(0),
        total_chunks: Some(1),
        importance: None,
        memory_id: None,
        scope: None,
        clearance: None,
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
    core.ensure_collection(coll_a, Some(dim))
        .await
        .expect("ensure a");
    core.ensure_collection(coll_b, Some(dim))
        .await
        .expect("ensure b");

    let chunk_a = DocumentChunk {
        text: "content a".to_string(),
        source: Some("s".to_string()),
        source_type: Some(constants::SOURCE_TYPE_TEXT.to_string()),
        collection: Some(coll_a.to_string()),
        tags: None,
        timestamp: None,
        project: None,
        last_modified: None,
        chunk_index: Some(0),
        total_chunks: Some(1),
        importance: None,
        memory_id: None,
        scope: None,
        clearance: None,
    };
    core.embed_and_upsert_batch(vec![chunk_a])
        .await
        .expect("upsert a");
    let chunk_b = DocumentChunk {
        text: "content b".to_string(),
        source: Some("s".to_string()),
        source_type: Some(constants::SOURCE_TYPE_TEXT.to_string()),
        collection: Some(coll_b.to_string()),
        tags: None,
        timestamp: None,
        project: None,
        last_modified: None,
        chunk_index: Some(0),
        total_chunks: Some(1),
        importance: None,
        memory_id: None,
        scope: None,
        clearance: None,
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

// ── Offline MCP integration (McpTestClient + FakeEmbedder; no live Qdrant) ──

/// Build `LqmServer` with FakeEmbedder and a lazy Qdrant client (no health check).
///
/// Suitable for tool-registration and tools that never call Qdrant. Does **not**
/// replace live smokes for ingest/search/lifecycle.
#[cfg(test)]
async fn offline_test_server(dim: usize) -> LqmServer {
    let qdrant = lqm_core::QdrantClient::new_lazy("http://127.0.0.1:9")
        .expect("lazy Qdrant client builds without network");
    let embedder = Box::new(lqm_core::FakeEmbedder::new(dim));
    let core = RagCore::new(qdrant, embedder, Some(1));
    LqmServer::new(core).await
}

/// Tool names that must appear in MCP registration (representative of the surface).
#[cfg(test)]
const EXPECTED_OFFLINE_TOOLS: &[&str] = &[
    "ingest_text",
    "search",
    "get_relevant_context",
    "list_collections",
    "create_collection",
    "delete_collection",
    "get_collection_info",
    "ingest_path",
    "ingest_url",
    "ingest_many",
    "get_embedder_info",
    "store_memory",
    "recall_memories",
    "list_sources",
    "list_chunks",
    "get_source",
    "expand_context",
    "delete_by_source",
    "delete_by_filter",
];

/// Offline: McpTestClient lists tools from the real `#[server]` handler (no Qdrant).
#[tokio::test]
async fn offline_mcp_tool_registration() {
    use turbomcp::testing::McpTestClient;

    let server = offline_test_server(32).await;
    let client = McpTestClient::new(server);

    let info = client.server_info();
    assert_eq!(info.name, "liberado-qdrant-mcp");
    assert!(!info.version.is_empty(), "version should be set");

    let tools = client.list_tools();
    assert!(
        !tools.is_empty(),
        "registered tools must be non-empty (macro surface)"
    );
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    for expected in EXPECTED_OFFLINE_TOOLS {
        assert!(
            names.contains(expected),
            "missing tool {expected:?}; have {names:?}"
        );
        client.assert_tool_exists(expected);
    }
    assert!(
        tools.len() >= EXPECTED_OFFLINE_TOOLS.len(),
        "tool count {} < expected {}",
        tools.len(),
        EXPECTED_OFFLINE_TOOLS.len()
    );
}

/// Offline: get_embedder_info via real tool dispatch (FakeEmbedder, no Qdrant I/O).
#[tokio::test]
async fn offline_mcp_get_embedder_info() {
    use turbomcp::testing::McpTestClient;

    let dim = 48usize;
    let server = offline_test_server(dim).await;
    let client = McpTestClient::new(server);

    let result = client
        .call_tool("get_embedder_info", serde_json::json!({}))
        .await
        .expect("get_embedder_info should succeed offline");
    assert!(!result.is_error(), "tool should not report isError");
    let text = result
        .first_text()
        .expect("embedder info should return text/JSON content");
    let v: serde_json::Value =
        serde_json::from_str(text).unwrap_or_else(|_| serde_json::json!({ "raw": text }));
    // structuredContent may also be present; prefer parsing text or structured
    let payload = result.structured_content.clone().unwrap_or(v);
    assert_eq!(payload["status"], "ok", "{payload}");
    assert_eq!(payload["id"], "fake", "{payload}");
    assert_eq!(payload["dimension"], dim as u64, "{payload}");
}

/// Offline: validation paths that fail before any Qdrant RPC (tool dispatch via client).
#[tokio::test]
async fn offline_mcp_validation_errors() {
    use turbomcp::testing::McpTestClient;

    let server = offline_test_server(16).await;
    let client = McpTestClient::new(server);

    // Empty collection name → RagCore validation (no Qdrant call).
    let empty_name = client
        .call_tool("create_collection", serde_json::json!({ "name": "   " }))
        .await;
    assert!(
        empty_name.is_err() || empty_name.as_ref().map(|r| r.is_error()).unwrap_or(false),
        "empty collection name must fail: {empty_name:?}"
    );
    if let Err(e) = &empty_name {
        let msg = e.to_string().to_lowercase();
        assert!(
            msg.contains("empty") || msg.contains("validation") || msg.contains("create"),
            "unexpected error text: {e}"
        );
    }

    // Empty delete filter → validation before Qdrant.
    let empty_filter = client
        .call_tool(
            "delete_by_filter",
            serde_json::json!({
                "collection": "offline_dummy"
            }),
        )
        .await;
    assert!(
        empty_filter.is_err() || empty_filter.as_ref().map(|r| r.is_error()).unwrap_or(false),
        "empty delete_by_filter must fail: {empty_filter:?}"
    );
}

/// Offline: unknown tool name is rejected by the real handler dispatch.
#[tokio::test]
async fn offline_mcp_unknown_tool() {
    use turbomcp::testing::McpTestClient;

    let server = offline_test_server(8).await;
    let client = McpTestClient::new(server);
    let err = client
        .call_tool("definitely_not_a_real_tool", serde_json::json!({}))
        .await;
    assert!(err.is_err(), "unknown tool must error: {err:?}");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let cli = Cli::parse();
    let core = RagCore::from_env(Some(&cli.qdrant_url), cli.config.as_deref())
        .await
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
    log::info!("starting lqm-mcp, embedder backend: {}", core.embedder.id());

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
