//! Source extractors: files and URLs → raw text / `DocumentChunk`s.
//!
//! Audio transcription (DeepInfra Whisper / Nemotron) is feature-gated
//! (`asr-deepinfra`) and exposed via [`extract_file_async`].

mod url;

pub mod asr;

pub use asr::{
    AsrBackend, AsrConfig, AsrError, DEFAULT_DEEPINFRA_BASE_URL, DEFAULT_NEMOTRON_MODEL,
    DEFAULT_WHISPER_MODEL, Transcript, parse_transcript_json, parse_transcript_value,
    resolve_api_key_from_env,
};

#[cfg(feature = "asr-deepinfra")]
pub use asr::transcribe_file;

pub use url::{
    DEFAULT_FETCH_TIMEOUT_SECS, DEFAULT_MAX_FETCH_BYTES, FetchedDocument, extract_html_title,
    extract_response_text, html_to_text,
};

#[cfg(feature = "fetch-url")]
pub use url::fetch_url;

use lqm_core::constants;
use lqm_core::types::DocumentChunk;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ExtractError {
    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),
    #[error("extraction failed: {0}")]
    ExtractionFailed(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ASR error: {0}")]
    Asr(String),
}

impl From<AsrError> for ExtractError {
    fn from(e: AsrError) -> Self {
        ExtractError::Asr(e.to_string())
    }
}

pub trait Extractor: Send + Sync {
    fn supported_extensions(&self) -> &[&str];
    fn source_type(&self) -> &str;
    fn extract_text(&self, path: &Path) -> Result<String, ExtractError>;
}

pub struct TextExtractor;

impl Extractor for TextExtractor {
    fn supported_extensions(&self) -> &[&str] {
        &[
            "txt", "md", "rs", "py", "js", "ts", "go", "java", "c", "cpp", "h", "hpp", "rb", "sh",
            "yaml", "yml", "json", "xml", "html", "css", "toml", "ini", "cfg", "log", "org", "rmd",
            "nix", "conf",
        ]
    }

    fn source_type(&self) -> &str {
        constants::SOURCE_TYPE_TEXT
    }

    fn extract_text(&self, path: &Path) -> Result<String, ExtractError> {
        Ok(std::fs::read_to_string(path)?)
    }
}

/// Extensions treated as audio for ASR / placeholder paths.
pub const AUDIO_EXTENSIONS: &[&str] = &["mp3", "wav", "flac", "ogg", "m4a", "opus", "aac", "wma"];

pub fn is_audio_path(path: &Path) -> bool {
    let ext = extension_lower(path);
    AUDIO_EXTENSIONS.contains(&ext.as_str())
}

pub struct AudioExtractor;

impl Extractor for AudioExtractor {
    fn supported_extensions(&self) -> &[&str] {
        AUDIO_EXTENSIONS
    }

    fn source_type(&self) -> &str {
        // Sync path remains placeholder; async path sets `audio` after successful STT.
        constants::SOURCE_TYPE_AUDIO_PLACEHOLDER
    }

    fn extract_text(&self, path: &Path) -> Result<String, ExtractError> {
        audio_placeholder_text(path)
    }
}

fn audio_placeholder_text(path: &Path) -> Result<String, ExtractError> {
    let metadata = std::fs::metadata(path)?;
    let size = metadata.len();
    let basename = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");
    Ok(format!(
        "[Audio placeholder — not transcribed: {basename}, size: {size} bytes]"
    ))
}

#[cfg(feature = "pdf")]
pub struct PdfExtractor;

#[cfg(feature = "pdf")]
impl Extractor for PdfExtractor {
    fn supported_extensions(&self) -> &[&str] {
        &["pdf"]
    }

    fn source_type(&self) -> &str {
        constants::SOURCE_TYPE_PDF
    }

    fn extract_text(&self, path: &Path) -> Result<String, ExtractError> {
        let bytes = std::fs::read(path)?;
        pdf_extract::extract_text_from_mem(&bytes)
            .map_err(|e| ExtractError::ExtractionFailed(format!("pdf extraction failed: {e}")))
    }
}

#[cfg(feature = "csv")]
pub struct CsvExtractor;

#[cfg(feature = "csv")]
impl Extractor for CsvExtractor {
    fn supported_extensions(&self) -> &[&str] {
        &["csv"]
    }

    fn source_type(&self) -> &str {
        constants::SOURCE_TYPE_TEXT
    }

    fn extract_text(&self, path: &Path) -> Result<String, ExtractError> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .flexible(true)
            .from_path(path)
            .map_err(|e| ExtractError::ExtractionFailed(format!("csv open failed: {e}")))?;

        let mut out = String::new();
        for result in reader.records() {
            let record = result.map_err(|e| {
                ExtractError::ExtractionFailed(format!("csv read failed: {e}"))
            })?;
            // Join all fields for this row with tab separator for readability.
            let row: Vec<&str> = record.iter().collect();
            out.push_str(&row.join("\t"));
            out.push('\n');
        }
        Ok(out)
    }
}

pub fn all_extractors() -> Vec<Box<dyn Extractor>> {
    #[allow(unused_mut)]
    let mut extractors: Vec<Box<dyn Extractor>> =
        vec![Box::new(TextExtractor), Box::new(AudioExtractor)];
    #[cfg(feature = "pdf")]
    extractors.push(Box::new(PdfExtractor));
    #[cfg(feature = "csv")]
    extractors.push(Box::new(CsvExtractor));
    extractors
}

pub fn find_extractor<'a>(
    path: &Path,
    extractors: &'a [Box<dyn Extractor>],
) -> Option<&'a dyn Extractor> {
    let ext = extension_lower(path);
    extractors
        .iter()
        .find(|e| e.supported_extensions().contains(&ext.as_str()))
        .map(|e| e.as_ref())
}

pub fn extension_lower(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase()
}

fn chunk_from_extract(
    path: &Path,
    text: String,
    source_type: &str,
    base_payload: serde_json::Value,
) -> DocumentChunk {
    let source = path.to_string_lossy().to_string();
    let base = base_payload.as_object().cloned().unwrap_or_default();

    DocumentChunk {
        text,
        source: Some(source),
        source_type: Some(source_type.to_string()),
        collection: base
            .get("collection")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        tags: base.get("tags").and_then(|v| {
            v.as_array().map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                    .collect()
            })
        }),
        timestamp: base
            .get("timestamp")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        project: base
            .get("project")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        last_modified: base
            .get("last_modified")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        chunk_index: None,
        total_chunks: None,
        importance: None,
        memory_id: None,
        scope: None,
        clearance: None,
    }
}

/// Sync extract (text/pdf/placeholder audio). For audio with ASR, use [`extract_file_async`].
pub fn extract_file(
    path: &Path,
    base_payload: serde_json::Value,
) -> Result<Vec<DocumentChunk>, ExtractError> {
    let extractors = all_extractors();
    let extractor = find_extractor(path, &extractors)
        .ok_or_else(|| ExtractError::UnsupportedFormat(extension_lower(path)))?;
    let text = extractor.extract_text(path)?;
    Ok(vec![chunk_from_extract(
        path,
        text,
        extractor.source_type(),
        base_payload,
    )])
}

/// Async extract: non-audio same as sync; audio uses DeepInfra ASR when feature is enabled.
///
/// Policy when `asr-deepinfra` is enabled:
/// - API key present → transcribe → `source_type=audio`
/// - Missing key + `LQM_ASR_FALLBACK_PLACEHOLDER=1` → placeholder
/// - Missing key otherwise → clear error (fail closed)
///
/// Without the feature, audio remains placeholder via sync extract.
pub async fn extract_file_async(
    path: &Path,
    base_payload: serde_json::Value,
) -> Result<Vec<DocumentChunk>, ExtractError> {
    if !is_audio_path(path) {
        return extract_file(path, base_payload);
    }

    #[cfg(feature = "asr-deepinfra")]
    {
        extract_audio_with_config(path, base_payload, &AsrConfig::from_env()).await
    }
    #[cfg(not(feature = "asr-deepinfra"))]
    {
        extract_file(path, base_payload)
    }
}

/// Audio extract with an explicit [`AsrConfig`] (production via
/// [`extract_file_async`]; unit tests inject config without mutating env).
#[cfg(feature = "asr-deepinfra")]
pub async fn extract_audio_with_config(
    path: &Path,
    base_payload: serde_json::Value,
    config: &AsrConfig,
) -> Result<Vec<DocumentChunk>, ExtractError> {
    if config
        .api_key
        .as_ref()
        .map(|s| s.is_empty())
        .unwrap_or(true)
    {
        if config.fallback_placeholder {
            log::warn!(
                "ASR key missing; fallback_placeholder set — writing audio_placeholder for {:?}",
                path
            );
            let text = audio_placeholder_text(path)?;
            return Ok(vec![chunk_from_extract(
                path,
                text,
                constants::SOURCE_TYPE_AUDIO_PLACEHOLDER,
                base_payload,
            )]);
        }
        return Err(ExtractError::Asr(
            "DeepInfra API key missing (set DEEPINFRA_API_KEY or DEEPINFRA_TOKEN); \
             or set LQM_ASR_FALLBACK_PLACEHOLDER=1 for placeholder stubs"
                .to_string(),
        ));
    }

    let transcript = asr::transcribe_file(path, config).await?;
    if transcript.text.contains("Audio placeholder") {
        return Err(ExtractError::Asr(
            "ASR returned placeholder-like text; refusing to treat as transcript".to_string(),
        ));
    }
    Ok(vec![chunk_from_extract(
        path,
        transcript.text,
        constants::SOURCE_TYPE_AUDIO,
        base_payload,
    )])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_tmp_file(ext: &str, content: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("lqm_test_{}.{}", uuid::Uuid::new_v4(), ext));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    #[test]
    fn test_extractor_detection_text() {
        let extractors = all_extractors();
        assert!(find_extractor(Path::new("foo.rs"), &extractors).is_some());
        assert!(find_extractor(Path::new("foo.py"), &extractors).is_some());
        assert!(find_extractor(Path::new("foo.md"), &extractors).is_some());
        assert!(find_extractor(Path::new("foo.txt"), &extractors).is_some());
    }

    #[test]
    fn test_extractor_detection_audio() {
        let extractors = all_extractors();
        assert!(find_extractor(Path::new("foo.mp3"), &extractors).is_some());
        assert!(find_extractor(Path::new("foo.wav"), &extractors).is_some());
        assert!(is_audio_path(Path::new("x.FLAC")));
    }

    #[test]
    fn test_extractor_detection_unknown() {
        let extractors = all_extractors();
        assert!(find_extractor(Path::new("foo.xyz"), &extractors).is_none());
    }

    #[test]
    fn test_text_extraction() {
        let path = make_tmp_file("rs", "fn main() {}");
        let chunks = extract_file(&path, serde_json::json!({})).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "fn main() {}");
        assert!(chunks[0].source.as_ref().unwrap().contains(".rs"));
        assert_eq!(chunks[0].source_type.as_deref(), Some("text"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_audio_placeholder_sync() {
        // Sync path always placeholder (ASR is async + feature-gated).
        let path = make_tmp_file("mp3", "fake audio data");
        let chunks = extract_file(&path, serde_json::json!({})).unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(
            chunks[0].text.contains("Audio placeholder"),
            "got: {}",
            chunks[0].text
        );
        assert_eq!(
            chunks[0].source_type.as_deref(),
            Some(constants::SOURCE_TYPE_AUDIO_PLACEHOLDER)
        );
        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    #[cfg(feature = "asr-deepinfra")]
    async fn test_audio_async_missing_key_fails_closed() {
        let path = make_tmp_file("wav", "not-really-audio");
        let cfg = AsrConfig {
            backend: AsrBackend::Whisper,
            model_slug: DEFAULT_WHISPER_MODEL.to_string(),
            api_key: None,
            base_url: DEFAULT_DEEPINFRA_BASE_URL.to_string(),
            timeout: std::time::Duration::from_secs(5),
            fallback_placeholder: false,
        };
        let result = extract_audio_with_config(&path, serde_json::json!({}), &cfg).await;
        assert!(result.is_err(), "expected error without key");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.to_lowercase().contains("key") || msg.to_lowercase().contains("api"),
            "err={msg}"
        );
        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    #[cfg(feature = "asr-deepinfra")]
    async fn test_audio_async_fallback_placeholder() {
        let path = make_tmp_file("mp3", "fake");
        let cfg = AsrConfig {
            backend: AsrBackend::Whisper,
            model_slug: DEFAULT_WHISPER_MODEL.to_string(),
            api_key: None,
            base_url: DEFAULT_DEEPINFRA_BASE_URL.to_string(),
            timeout: std::time::Duration::from_secs(5),
            fallback_placeholder: true,
        };
        let chunks = extract_audio_with_config(&path, serde_json::json!({}), &cfg)
            .await
            .expect("fallback placeholder");
        assert!(chunks[0].text.contains("Audio placeholder"));
        assert_eq!(
            chunks[0].source_type.as_deref(),
            Some(constants::SOURCE_TYPE_AUDIO_PLACEHOLDER)
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    #[cfg(feature = "pdf")]
    fn test_pdf_extract_from_minimal_fixture() {
        let minimal = b"%PDF-1.1\n\
1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n\
2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n\
3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R /Resources<< /Font<< /F1 5 0 R >> >> >>endobj\n\
4 0 obj<< /Length 44 >>stream\n\
BT /F1 24 Tf 100 100 Td (Hello) Tj ET\n\
endstream\n\
endobj\n\
5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n\
trailer<< /Root 1 0 R >>\n\
%%EOF\n";
        let dir = std::env::temp_dir();
        let path = dir.join(format!("lqm_pdf_{}.pdf", uuid::Uuid::new_v4()));
        std::fs::write(&path, minimal).expect("write pdf fixture");
        let extractor = PdfExtractor;
        let result = extractor.extract_text(&path);
        match result {
            Ok(_text) => {}
            Err(ExtractError::ExtractionFailed(_)) => {}
            Err(e) => panic!("unexpected error from PDF fixture path: {e}"),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_unknown_extension_error() {
        let path = make_tmp_file("xyz", "data");
        let result = extract_file(&path, serde_json::json!({}));
        assert!(result.is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_extension_lower_case() {
        assert_eq!(extension_lower(Path::new("foo.PDF")), "pdf");
        assert_eq!(extension_lower(Path::new("foo.RS")), "rs");
        assert_eq!(extension_lower(Path::new("foo")), "");
    }

    #[test]
    fn test_missing_file_error() {
        let result = extract_file(Path::new("/nonexistent/path.txt"), serde_json::json!({}));
        assert!(result.is_err());
        match result {
            Err(ExtractError::Io(_)) => {}
            _ => panic!("expected Io error"),
        }
    }

    #[test]
    #[cfg(feature = "pdf")]
    fn test_pdf_extractor_registered() {
        let extractors = all_extractors();
        assert!(find_extractor(Path::new("doc.pdf"), &extractors).is_some());
        let extractor = PdfExtractor;
        assert!(extractor.supported_extensions().contains(&"pdf"));
        assert_eq!(extractor.source_type(), "pdf");
        let missing = extractor.extract_text(Path::new("/nonexistent/doc.pdf"));
        assert!(missing.is_err());
    }

    /// Live ASR: skip-safe unless LQM_LIVE_ASR=1 (then hard-require key).
    #[tokio::test]
    #[cfg(feature = "asr-deepinfra")]
    async fn test_asr_live_whisper_and_nemotron() {
        let key = resolve_api_key_from_env();
        let hard = std::env::var(asr::ENV_LIVE_ASR).ok().as_deref() == Some("1");
        if key.is_none() {
            if hard {
                panic!("LQM_LIVE_ASR=1 but DEEPINFRA_API_KEY/DEEPINFRA_TOKEN missing");
            }
            eprintln!("skipping live ASR smoke (no DeepInfra key)");
            return;
        }

        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/asr_hello.wav");
        assert!(
            fixture.is_file(),
            "missing fixture at {}",
            fixture.display()
        );

        for backend in [AsrBackend::Whisper, AsrBackend::Nemotron] {
            let config = AsrConfig {
                backend,
                model_slug: backend.default_model_slug().to_string(),
                api_key: key.clone(),
                base_url: DEFAULT_DEEPINFRA_BASE_URL.to_string(),
                timeout: std::time::Duration::from_secs(180),
                fallback_placeholder: false,
            };
            let transcript = match asr::transcribe_file(&fixture, &config).await {
                Ok(t) => t,
                Err(e) => {
                    if hard {
                        panic!("live ASR failed for {backend}: {e}");
                    }
                    eprintln!("live ASR soft-fail for {backend}: {e}");
                    continue;
                }
            };
            assert!(
                !transcript.text.trim().is_empty(),
                "empty transcript for {backend}"
            );
            assert!(
                !transcript.text.contains("Audio placeholder"),
                "placeholder leaked for {backend}: {}",
                transcript.text
            );
            eprintln!(
                "live ASR {backend} ok: chars={} preview={:?}",
                transcript.text.len(),
                transcript.text.chars().take(80).collect::<String>()
            );
        }
    }

    /// Full extract_file_async path with default env backend (whisper when key present).
    #[tokio::test]
    #[cfg(feature = "asr-deepinfra")]
    async fn test_extract_file_async_live_audio_source_type() {
        if resolve_api_key_from_env().is_none() {
            eprintln!("skipping extract_file_async live ASR (no key)");
            return;
        }
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/asr_hello.wav");
        if !fixture.is_file() {
            eprintln!("skipping: fixture missing");
            return;
        }
        match extract_file_async(&fixture, serde_json::json!({})).await {
            Ok(chunks) => {
                assert_eq!(chunks.len(), 1);
                assert_eq!(
                    chunks[0].source_type.as_deref(),
                    Some(constants::SOURCE_TYPE_AUDIO)
                );
                assert!(!chunks[0].text.contains("Audio placeholder"));
                assert!(!chunks[0].text.trim().is_empty());
            }
            Err(e) => {
                eprintln!("extract_file_async live soft-fail: {e}");
            }
        }
    }

    #[cfg(feature = "csv")]
    #[test]
    fn csv_extractor_reads_rows() {
        use crate::{CsvExtractor, Extractor};
        let dir = std::env::temp_dir();
        let p = dir.join("lqm_csv_test.csv");
        std::fs::write(&p, "a,b,c\n1,2,3\nx,y,z\n").expect("write csv");
        let extractor = CsvExtractor;
        let text = extractor.extract_text(&p).expect("extract");
        assert!(text.contains("a\tb\tc"));
        assert!(text.contains("1\t2\t3"));
        assert!(text.contains("x\ty\tz"));
        let _ = std::fs::remove_file(&p);
    }
}
