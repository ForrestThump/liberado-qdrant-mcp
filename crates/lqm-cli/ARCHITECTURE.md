# lqm-cli

CLI binary for admin tasks, bulk ingestion, and embedder benchmarking.

## Stack

- **CLI framework:** clap v4 (derive)
- **Core:** delegates to `lqm-core`
- **Ingestion:** delegates to `lqm-ingest` extractors
- **File traversal:** walkdir

## Commands

```
lqm-cli ingest <path>     Bulk ingest files/dirs into a collection
  -c, --collection NAME   Target collection (default: DEFAULT_COLLECTION_NAME)
  --source-type TYPE      Source label override
  --qdrant-url URL        Qdrant server
  --config PATH           TOML embedder config

lqm-cli list              List all collections (pretty JSON)
lqm-cli delete <name>     Delete a collection
lqm-cli bench             Benchmark embedders
  -t, --text TEXT         Text to embed
  -i, --iterations N      Iterations per backend
```

## Features

- `embed-fastembed` — enables fastembed in benchmark command
