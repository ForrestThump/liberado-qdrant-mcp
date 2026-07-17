# lqm-ingest

Source extractors that turn files/streams into `DocumentChunk`s.

## Architecture

```
Extractor trait
├── TextExtractor   (txt, md, rs, py, js, ...)
├── PdfExtractor    (pdf — behind "pdf" feature)
└── AudioExtractor  (mp3, wav, flac, ogg, ... — placeholder)
```

## Trait

```rust
pub trait Extractor: Send + Sync {
    fn supported_extensions(&self) -> &[&str];
    fn source_type(&self) -> &str;
    fn extract_text(&self, path: &Path) -> Result<String, ExtractError>;
}
```

## Public API

- `all_extractors()` — returns all registered extractors
- `find_extractor(path, extractors)` — matches by file extension
- `extract_file(path, base_payload)` — one-shot extraction producing `DocumentChunk`s

## Features

- `pdf` — enables `pdf-extract` crate for PDF text extraction
