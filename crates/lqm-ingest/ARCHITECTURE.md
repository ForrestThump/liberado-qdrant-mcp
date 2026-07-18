# lqm-ingest

Source extractors that turn files and remote responses into raw text (callers
then chunk via `RagCore::expand_to_chunks`).

## Architecture

```
Extractor trait  →  extract_text(path) → String   (sync)
├── TextExtractor   (txt, md, rs, py, js, …)
├── PdfExtractor    (pdf — feature "pdf")
└── AudioExtractor  (sync → audio_placeholder only)

extract_file_async(path)  (preferred for MCP/API/CLI file ingest)
├── non-audio → same as extract_file
└── audio + feature asr-deepinfra → DeepInfra STT → source_type=audio

asr.rs (feature "asr-deepinfra")
├── AsrBackend: whisper | nemotron
├── AsrConfig::from_env
├── parse_transcript_json (pure)
└── transcribe_file (multipart POST)

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

- `all_extractors` / `find_extractor` / `extract_file` / `extract_file_async` / `extension_lower`
- `asr` module: `AsrBackend`, `AsrConfig`, `transcribe_file`, `parse_transcript_json`
- `html_to_text`, `extract_html_title`, `extract_response_text`
- `fetch_url` — gated on `fetch-url` (default feature)

## Features

| Feature | Default | Effect |
|---------|---------|--------|
| `fetch-url` | on | reqwest + `fetch_url` |
| `pdf` | off | pdf-extract + `PdfExtractor` (MCP/API enable it) |
| `asr-deepinfra` | on | DeepInfra file ASR (Whisper/Nemotron); MCP/API/CLI also request it |

## Audio

With `asr-deepinfra` + API key (`DEEPINFRA_API_KEY` or `DEEPINFRA_TOKEN`):
`extract_file_async` transcribes via DeepInfra and sets `source_type=audio`.
Backends: `LQM_ASR_BACKEND=whisper` (default slug
`openai/whisper-large-v3-turbo`) or `nemotron` (slug
`nvidia/Nemotron-3.5-ASR-Streaming-Multilingual-0.6b`). Optional `LQM_ASR_MODEL`
overrides the slug. Missing key fails closed unless
`LQM_ASR_FALLBACK_PLACEHOLDER=1`. Sync `extract_file` still returns
`audio_placeholder` for offline/simple callers.
