# lqm-ingest

Source extractors that turn files and remote responses into raw text (callers
then chunk via `RagCore::expand_to_chunks`).

## Architecture

```
Extractor trait  →  extract_text(path) → String
├── TextExtractor   (txt, md, rs, py, js, …)
├── PdfExtractor    (pdf — feature "pdf")
└── AudioExtractor  (audio — source_type=audio_placeholder until transcription)

url.rs (always: pure HTML helpers; network behind feature "fetch-url")
├── html_to_text / extract_html_title / extract_response_text
└── fetch_url  (reqwest; default-on feature)
```

## Trait

```rust
pub trait Extractor: Send + Sync {
    fn supported_extensions(&self) -> &[&str];
    fn source_type(&self) -> &str;
    fn extract_text(&self, path: &Path) -> Result<String, ExtractError>;
}
```

The trait returns **text**, not chunks, by design: chunk strategy lives in
`lqm-core` so MCP/HTTP/CLI cannot diverge. Callers must use `expand_to_chunks`.

## Public API

- `all_extractors` / `find_extractor` / `extract_file` / `extension_lower`
- `html_to_text`, `extract_html_title`, `extract_response_text`
- `fetch_url` — gated on `fetch-url` (default feature)

## Features

| Feature | Default | Effect |
|---------|---------|--------|
| `fetch-url` | on | reqwest + `fetch_url` |
| `pdf` | off | pdf-extract + `PdfExtractor` (MCP/API enable it) |

## Audio

Audio extractors write a **placeholder** string and `source_type=audio_placeholder`
so agents can filter stubs until real transcription ships (see `docs/ROADMAP.md`).
