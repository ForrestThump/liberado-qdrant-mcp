# lqm-cli

CLI binary for admin tasks, bulk ingestion, and embedder benchmarking.

## Stack

- **CLI:** clap v4
- **Core:** `RagCore::from_env` (Qdrant + embedder)
- **Ingestion:** `lqm-ingest` extractors + `RagCore::expand_to_chunks` (same chunking as MCP/API)
- **Traversal:** workspace `walkdir`

## Commands

```
lqm-cli ingest --path <path>
  -c, --collection NAME   (default: resolve_collection / "default")
  --source-type TYPE
  --qdrant-url URL
  --config PATH

lqm-cli list              List collections (JSON)
lqm-cli delete --name N   Delete collection
lqm-cli bench -t TEXT [-i N]   Embedder benchmark
```

## Notes

- Ingest ensures the target collection, then structure-aware expands each file
  (markdown headings / code boundaries / plain windows) before upsert.
- Unsupported extensions are skipped with a log line.
- Feature `embed-fastembed` enables fastembed for this binary when desired.
