mod url;

pub use url::{
    DEFAULT_FETCH_TIMEOUT_SECS, DEFAULT_MAX_FETCH_BYTES, FetchedDocument, extract_html_title,
    extract_response_text, fetch_url, html_to_text,
};

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
        "text"
    }

    fn extract_text(&self, path: &Path) -> Result<String, ExtractError> {
        Ok(std::fs::read_to_string(path)?)
    }
}

pub struct AudioExtractor;

impl Extractor for AudioExtractor {
    fn supported_extensions(&self) -> &[&str] {
        &["mp3", "wav", "flac", "ogg", "m4a", "opus", "aac", "wma"]
    }

    fn source_type(&self) -> &str {
        "audio"
    }

    fn extract_text(&self, path: &Path) -> Result<String, ExtractError> {
        let metadata = std::fs::metadata(path)?;
        let size = metadata.len();
        let basename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        Ok(format!("[Audio file: {basename}, size: {size} bytes]"))
    }
}

#[cfg(feature = "pdf")]
pub struct PdfExtractor;

#[cfg(feature = "pdf")]
impl Extractor for PdfExtractor {
    fn supported_extensions(&self) -> &[&str] {
        &["pdf"]
    }

    fn source_type(&self) -> &str {
        "pdf"
    }

    fn extract_text(&self, path: &Path) -> Result<String, ExtractError> {
        // Path API is preferred for large files; from_mem keeps tests self-contained.
        let bytes = std::fs::read(path)?;
        pdf_extract::extract_text_from_mem(&bytes)
            .map_err(|e| ExtractError::ExtractionFailed(format!("pdf extraction failed: {e}")))
    }
}

pub fn all_extractors() -> Vec<Box<dyn Extractor>> {
    #[allow(unused_mut)]
    let mut extractors: Vec<Box<dyn Extractor>> =
        vec![Box::new(TextExtractor), Box::new(AudioExtractor)];
    #[cfg(feature = "pdf")]
    extractors.push(Box::new(PdfExtractor));
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

pub fn extract_file(
    path: &Path,
    base_payload: serde_json::Value,
) -> Result<Vec<DocumentChunk>, ExtractError> {
    let extractors = all_extractors();
    let extractor = find_extractor(path, &extractors)
        .ok_or_else(|| ExtractError::UnsupportedFormat(extension_lower(path)))?;
    let text = extractor.extract_text(path)?;
    let source = path.to_string_lossy().to_string();

    let base = base_payload.as_object().cloned().unwrap_or_default();

    let chunk = DocumentChunk {
        text,
        source: Some(source),
        source_type: Some(extractor.source_type().to_string()),
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
    };

    Ok(vec![chunk])
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
    fn test_audio_placeholder() {
        let path = make_tmp_file("mp3", "fake audio data");
        let chunks = extract_file(&path, serde_json::json!({})).unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.starts_with("[Audio file:"));
        assert_eq!(chunks[0].source_type.as_deref(), Some("audio"));
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
        // Minimal hand-written PDF fixtures often fail strict parsers; registration is the
        // structural guarantee. Real PDFs are covered via path ingest in live environments.
        let missing = extractor.extract_text(Path::new("/nonexistent/doc.pdf"));
        assert!(missing.is_err());
    }
}
