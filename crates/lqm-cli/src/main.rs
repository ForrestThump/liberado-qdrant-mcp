use clap::{Parser, Subcommand};
use lqm_core::RagCore;
use lqm_core::types::resolve_collection;
use std::path::Path;

#[derive(Parser)]
#[command(name = "lqm-cli", about = "liberado-qdrant-mcp CLI")]
struct Cli {
    #[arg(long, default_value = "http://localhost:6334")]
    qdrant_url: String,

    #[arg(long)]
    config: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Ingest {
        #[arg(short, long)]
        path: String,

        #[arg(short, long)]
        collection: Option<String>,

        #[arg(long)]
        source_type: Option<String>,
    },
    List,
    Delete {
        #[arg(short, long)]
        name: String,
    },
    Bench {
        #[arg(short, long)]
        text: String,

        #[arg(short, long, default_value_t = 100)]
        iterations: usize,
    },
}

fn file_modified_secs(path: &Path) -> Option<String> {
    std::fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| {
            t.duration_since(std::time::UNIX_EPOCH)
                .ok()
                .map(|d| d.as_secs().to_string())
        })
}

async fn make_core(qdrant_url: &str, config_path: Option<&str>) -> Option<RagCore> {
    // from_env already validates Qdrant connectivity via list_collections.
    RagCore::from_env(Some(qdrant_url), config_path).await.ok()
}

async fn cmd_ingest(
    core: &RagCore,
    path: &str,
    collection: Option<String>,
    source_type: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let coll = resolve_collection(collection);
    log::info!("ingesting into collection '{}'", coll);
    core.ensure_collection(&coll, None).await?;

    let extractors = lqm_ingest::all_extractors();
    let p = Path::new(path);

    if p.is_dir() {
        for entry in walkdir::WalkDir::new(p) {
            let entry = entry?;
            if !entry.file_type().is_file() {
                continue;
            }
            ingest_single_file(
                core,
                entry.path(),
                &coll,
                source_type.as_deref(),
                &extractors,
            )
            .await;
        }
    } else if p.is_file() {
        ingest_single_file(core, p, &coll, source_type.as_deref(), &extractors).await;
    } else {
        log::info!("path not found: {path}");
    }

    Ok(())
}

async fn ingest_single_file(
    core: &RagCore,
    path: &Path,
    collection: &str,
    source_type: Option<&str>,
    extractors: &[Box<dyn lqm_ingest::Extractor>],
) {
    let ext = lqm_ingest::extension_lower(path);
    let extractor = match lqm_ingest::find_extractor(path, extractors) {
        Some(e) => e,
        None => {
            log::info!("skipping unsupported: {ext}");
            return;
        }
    };

    let text = match extractor.extract_text(path) {
        Ok(t) => t,
        Err(e) => {
            log::error!("failed to extract {path:?}: {e}");
            return;
        }
    };

    let source = path.to_string_lossy().to_string();
    let st = source_type
        .map(|s| s.to_string())
        .unwrap_or_else(|| extractor.source_type().to_string());
    let modified = file_modified_secs(path);

    let chunks = core.expand_to_chunks(
        &text,
        Some(source.clone()),
        Some(st),
        collection.to_string(),
        None,
        None,
        modified,
        Some(&source),
        None,
        None,
    );

    match core.embed_and_upsert_batch(chunks).await {
        Ok(report) => println!(
            "ingested {source}: chunks={} inserted={} skipped={} replaced={}",
            report.chunks, report.inserted, report.skipped, report.replaced
        ),
        Err(e) => log::error!("ingest error {source}: {e}"),
    }
}

async fn cmd_list(core: &RagCore) -> Result<(), Box<dyn std::error::Error>> {
    let collections = core.list_collections().await?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({"collections": collections}))?
    );
    Ok(())
}

async fn cmd_delete(core: &RagCore, name: &str) -> Result<(), Box<dyn std::error::Error>> {
    log::info!("deleting collection '{}'", name);
    let deleted = core.delete_collection(name).await?;
    if deleted {
        println!("deleted: {name}");
    } else {
        println!("collection not found: {name}");
    }
    Ok(())
}

async fn cmd_bench(
    core: &RagCore,
    text: &str,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let dim = core.embedder.dimension();
    let start = std::time::Instant::now();

    for i in 0..iterations {
        let t = format!("{text} [{i}]");
        match core.embed_batch(vec![t]).await {
            Ok(vectors) => {
                if i == 0 {
                    println!("dim: {dim}, vec len: {}", vectors[0].len());
                }
            }
            Err(e) => {
                log::error!("embed error at iter {i}: {e}");
                break;
            }
        }
    }

    let elapsed = start.elapsed();
    println!(
        "{} iterations in {:?} ({:.2} iters/sec)",
        iterations,
        elapsed,
        iterations as f64 / elapsed.as_secs_f64()
    );
    Ok(())
}

#[tokio::test]
async fn test_list_collections() {
    let core = make_core("http://localhost:6334", None).await;
    if core.is_none() {
        return;
    }
    let result = core.unwrap().list_collections().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_delete_nonexistent() {
    let core = make_core("http://localhost:6334", None).await;
    if core.is_none() {
        return;
    }
    let core = core.unwrap();
    let result = core.delete_collection("lqm_cli_nonexistent_test").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_bench() {
    let core = make_core("http://localhost:6334", None).await;
    if core.is_none() {
        return;
    }
    let core = core.unwrap();
    let result = cmd_bench(&core, "hello world", 5).await;
    assert!(result.is_ok());
}

#[test]
fn test_file_modified_secs() {
    let result = file_modified_secs(Path::new("/nonexistent/file.txt"));
    assert!(result.is_none());
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let cli = Cli::parse();

    let core = match make_core(&cli.qdrant_url, cli.config.as_deref()).await {
        Some(c) => c,
        None => {
            log::error!("failed to connect to qdrant at {}", cli.qdrant_url);
            std::process::exit(1);
        }
    };

    match cli.command {
        Commands::Ingest {
            path,
            collection,
            source_type,
        } => cmd_ingest(&core, &path, collection, source_type).await?,
        Commands::List => cmd_list(&core).await?,
        Commands::Delete { name } => cmd_delete(&core, &name).await?,
        Commands::Bench { text, iterations } => cmd_bench(&core, &text, iterations).await?,
    }

    Ok(())
}
