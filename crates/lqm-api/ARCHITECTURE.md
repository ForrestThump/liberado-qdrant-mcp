# lqm-api

HTTP server powering the web frontend. Reuses `lqm-core` for all RAG logic.

## Stack

- **HTTP:** axum 0.8
- **CORS:** tower-http (any origin)
- **Core:** delegates to `lqm-core`

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | `{ status, version }` |
| `GET` | `/api/collections` | `{ collections: [...] }` |
| `DELETE` | `/api/collections/{name}` | `204 No Content` |
| `POST` | `/api/search` | `{ results: [...] }` |
| `POST` | `/api/ingest` | `{ status, collection }` |
| `GET` | `/` | RAG search UI |

## Configuration

```
lqm-api --host 127.0.0.1 --port 8080
lqm-api --config embedder.toml --qdrant-url http://localhost:6334
```

## Static files

`static/` directory served for static assets. Includes `index.html` — a
dark-themed RAG search UI with JavaScript form handling.
