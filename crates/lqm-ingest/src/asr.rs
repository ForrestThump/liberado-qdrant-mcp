//! DeepInfra file ASR (speech-to-text) for audio ingest.
//!
//! Backends share one multipart HTTP contract; only the model slug differs:
//! - `whisper` (default) → `openai/whisper-large-v3-turbo`
//! - `nemotron` → `nvidia/Nemotron-3.5-ASR-Streaming-Multilingual-0.6b`
//!
//! Gated behind cargo feature `asr-deepinfra`. No multimodal embeddings; transcript
//! text is returned for normal dense chunking/upsert.

use std::fmt;
use std::path::Path;
use std::str::FromStr;
use std::time::Duration;

use thiserror::Error;

/// Default DeepInfra API root (no trailing slash).
pub const DEFAULT_DEEPINFRA_BASE_URL: &str = "https://api.deepinfra.com";

/// Whisper large-v3-turbo (default ASR backend).
pub const DEFAULT_WHISPER_MODEL: &str = "openai/whisper-large-v3-turbo";

/// Nemotron multilingual ASR (batch file multipart; not streaming input).
pub const DEFAULT_NEMOTRON_MODEL: &str = "nvidia/Nemotron-3.5-ASR-Streaming-Multilingual-0.6b";

pub const ENV_ASR_BACKEND: &str = "LQM_ASR_BACKEND";
pub const ENV_ASR_MODEL: &str = "LQM_ASR_MODEL";
pub const ENV_DEEPINFRA_API_KEY: &str = "DEEPINFRA_API_KEY";
pub const ENV_DEEPINFRA_TOKEN: &str = "DEEPINFRA_TOKEN";
pub const ENV_DEEPINFRA_BASE_URL: &str = "DEEPINFRA_BASE_URL";
/// When `1`/`true`, missing key falls back to audio_placeholder instead of erroring.
pub const ENV_ASR_FALLBACK_PLACEHOLDER: &str = "LQM_ASR_FALLBACK_PLACEHOLDER";
/// When `1`, live ASR tests hard-require a key (must not skip).
pub const ENV_LIVE_ASR: &str = "LQM_LIVE_ASR";

/// Default HTTP timeout for transcription (long files).
pub const DEFAULT_ASR_TIMEOUT_SECS: u64 = 120;

/// Selectable DeepInfra ASR backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AsrBackend {
    #[default]
    Whisper,
    Nemotron,
}

impl AsrBackend {
    pub const ALL: &'static [AsrBackend] = &[AsrBackend::Whisper, AsrBackend::Nemotron];

    pub fn as_str(self) -> &'static str {
        match self {
            AsrBackend::Whisper => "whisper",
            AsrBackend::Nemotron => "nemotron",
        }
    }

    pub fn default_model_slug(self) -> &'static str {
        match self {
            AsrBackend::Whisper => DEFAULT_WHISPER_MODEL,
            AsrBackend::Nemotron => DEFAULT_NEMOTRON_MODEL,
        }
    }
}

impl fmt::Display for AsrBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for AsrBackend {
    type Err = UnknownAsrBackend;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "whisper" | "whisper-turbo" | "whisper_turbo" | "turbo" => Ok(AsrBackend::Whisper),
            "nemotron" | "nemo" | "nvidia" => Ok(AsrBackend::Nemotron),
            other => Err(UnknownAsrBackend(other.to_string())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownAsrBackend(pub String);

impl fmt::Display for UnknownAsrBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown ASR backend '{}'; expected one of: {}",
            self.0,
            AsrBackend::ALL
                .iter()
                .map(|b| b.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

impl std::error::Error for UnknownAsrBackend {}

/// Runtime ASR configuration (env-driven).
#[derive(Debug, Clone)]
pub struct AsrConfig {
    pub backend: AsrBackend,
    /// Full DeepInfra model slug used for the request path.
    pub model_slug: String,
    pub api_key: Option<String>,
    pub base_url: String,
    pub timeout: Duration,
    /// When true, missing key yields placeholder instead of error (ops escape hatch).
    pub fallback_placeholder: bool,
}

impl AsrConfig {
    /// Resolve from environment variables.
    pub fn from_env() -> Self {
        let backend = match std::env::var(ENV_ASR_BACKEND) {
            Ok(raw) => raw.parse().unwrap_or_else(|e: UnknownAsrBackend| {
                log::warn!("{e}; using whisper");
                AsrBackend::Whisper
            }),
            Err(_) => AsrBackend::Whisper,
        };
        let model_slug = std::env::var(ENV_ASR_MODEL)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| backend.default_model_slug().to_string());
        let api_key = resolve_api_key_from_env();
        let base_url = std::env::var(ENV_DEEPINFRA_BASE_URL)
            .ok()
            .map(|s| s.trim().trim_end_matches('/').to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| DEFAULT_DEEPINFRA_BASE_URL.to_string());
        let fallback_placeholder = env_truthy(ENV_ASR_FALLBACK_PLACEHOLDER);
        Self {
            backend,
            model_slug,
            api_key,
            base_url,
            timeout: Duration::from_secs(DEFAULT_ASR_TIMEOUT_SECS),
            fallback_placeholder,
        }
    }

    /// Inference URL for this config (`{base}/v1/inference/{slug}`).
    pub fn inference_url(&self) -> String {
        format!("{}/v1/inference/{}", self.base_url, self.model_slug)
    }
}

/// Prefer `DEEPINFRA_API_KEY`, then `DEEPINFRA_TOKEN`.
pub fn resolve_api_key_from_env() -> Option<String> {
    for key in [ENV_DEEPINFRA_API_KEY, ENV_DEEPINFRA_TOKEN] {
        if let Ok(v) = std::env::var(key) {
            let t = v.trim();
            if !t.is_empty() {
                return Some(t.to_string());
            }
        }
    }
    None
}

fn env_truthy(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on")
        }
        Err(_) => false,
    }
}

/// Successful transcript from DeepInfra.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transcript {
    pub text: String,
    pub language: Option<String>,
}

#[derive(Debug, Error)]
pub enum AsrError {
    #[error("DeepInfra API key missing (set DEEPINFRA_API_KEY or DEEPINFRA_TOKEN)")]
    MissingApiKey,
    #[error("ASR HTTP error: {0}")]
    Http(String),
    #[error("ASR response parse error: {0}")]
    Parse(String),
    #[error("ASR returned empty transcript")]
    EmptyTranscript,
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Extract transcript text from a DeepInfra ASR JSON body (pure, offline-tested).
///
/// Prefers top-level `text`; if empty/missing, joins non-empty `segments[].text`.
pub fn parse_transcript_json(body: &str) -> Result<Transcript, AsrError> {
    let v: serde_json::Value =
        serde_json::from_str(body).map_err(|e| AsrError::Parse(format!("invalid JSON: {e}")))?;
    parse_transcript_value(&v)
}

/// Same as [`parse_transcript_json`] from an already-parsed value.
pub fn parse_transcript_value(v: &serde_json::Value) -> Result<Transcript, AsrError> {
    let language = v
        .get("language")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string());

    let mut text = v
        .get("text")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    if text.is_empty()
        && let Some(segs) = v.get("segments").and_then(|s| s.as_array())
    {
        let joined: String = segs
            .iter()
            .filter_map(|s| s.get("text").and_then(|t| t.as_str()))
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        text = joined.trim().to_string();
    }

    if text.is_empty() {
        // Surface provider error messages when present.
        if let Some(err) = v
            .get("error")
            .and_then(|e| e.as_str())
            .or_else(|| v.get("detail").and_then(|d| d.as_str()))
        {
            return Err(AsrError::Parse(format!("provider error: {err}")));
        }
        return Err(AsrError::EmptyTranscript);
    }

    Ok(Transcript { text, language })
}

/// Transcribe an audio file via DeepInfra multipart inference (async).
#[cfg(feature = "asr-deepinfra")]
pub async fn transcribe_file(path: &Path, config: &AsrConfig) -> Result<Transcript, AsrError> {
    let api_key = config
        .api_key
        .as_deref()
        .filter(|s| !s.is_empty())
        .ok_or(AsrError::MissingApiKey)?;

    let bytes = std::fs::read(path)?;
    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("audio.bin")
        .to_string();

    let part = reqwest::multipart::Part::bytes(bytes)
        .file_name(filename)
        .mime_str("application/octet-stream")
        .map_err(|e| AsrError::Http(format!("multipart part: {e}")))?;
    let form = reqwest::multipart::Form::new().part("audio", part);

    let url = config.inference_url();
    log::debug!(
        "ASR request backend={} model={} url={}",
        config.backend,
        config.model_slug,
        url
    );

    let client = reqwest::Client::builder()
        .timeout(config.timeout)
        .build()
        .map_err(|e| AsrError::Http(format!("client build: {e}")))?;

    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {api_key}"))
        .multipart(form)
        .send()
        .await
        .map_err(|e| AsrError::Http(format!("request failed: {e}")))?;

    let status = resp.status();
    let body = resp
        .text()
        .await
        .map_err(|e| AsrError::Http(format!("read body: {e}")))?;

    if !status.is_success() {
        let snippet: String = body.chars().take(400).collect();
        return Err(AsrError::Http(format!("status {status}: {snippet}")));
    }

    let transcript = parse_transcript_json(&body)?;
    log::info!(
        "ASR ok backend={} model={} chars={} language={:?}",
        config.backend,
        config.model_slug,
        transcript.text.len(),
        transcript.language
    );
    Ok(transcript)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asr_backend_parse_and_defaults() {
        assert_eq!(
            "whisper".parse::<AsrBackend>().unwrap(),
            AsrBackend::Whisper
        );
        assert_eq!(
            "NEMOTRON".parse::<AsrBackend>().unwrap(),
            AsrBackend::Nemotron
        );
        assert_eq!("turbo".parse::<AsrBackend>().unwrap(), AsrBackend::Whisper);
        assert!("nope".parse::<AsrBackend>().is_err());
        assert_eq!(AsrBackend::default(), AsrBackend::Whisper);
        assert_eq!(
            AsrBackend::Whisper.default_model_slug(),
            DEFAULT_WHISPER_MODEL
        );
        assert_eq!(
            AsrBackend::Nemotron.default_model_slug(),
            DEFAULT_NEMOTRON_MODEL
        );
        assert_eq!(DEFAULT_WHISPER_MODEL, "openai/whisper-large-v3-turbo");
        assert_eq!(
            DEFAULT_NEMOTRON_MODEL,
            "nvidia/Nemotron-3.5-ASR-Streaming-Multilingual-0.6b"
        );
    }

    #[test]
    fn asr_config_default_backend_whisper_slug() {
        // from_env without LQM_ASR_BACKEND should default whisper (may inherit env in process).
        // Construct explicitly to prove default fields.
        let cfg = AsrConfig {
            backend: AsrBackend::default(),
            model_slug: AsrBackend::default().default_model_slug().to_string(),
            api_key: None,
            base_url: DEFAULT_DEEPINFRA_BASE_URL.to_string(),
            timeout: Duration::from_secs(DEFAULT_ASR_TIMEOUT_SECS),
            fallback_placeholder: false,
        };
        assert_eq!(cfg.backend, AsrBackend::Whisper);
        assert_eq!(cfg.model_slug, DEFAULT_WHISPER_MODEL);
        assert_eq!(
            cfg.inference_url(),
            format!("{DEFAULT_DEEPINFRA_BASE_URL}/v1/inference/{DEFAULT_WHISPER_MODEL}")
        );
        let nemo = AsrConfig {
            backend: AsrBackend::Nemotron,
            model_slug: AsrBackend::Nemotron.default_model_slug().to_string(),
            ..cfg.clone()
        };
        assert!(nemo.inference_url().contains(DEFAULT_NEMOTRON_MODEL));
    }

    #[test]
    fn parse_transcript_json_uses_text_field() {
        let body = r#"{
            "text": "  Hello liberado world  ",
            "segments": [{"start":0.0,"end":1.0,"text":"ignored when text set"}],
            "language": "en"
        }"#;
        let t = parse_transcript_json(body).unwrap();
        assert_eq!(t.text, "Hello liberado world");
        assert_eq!(t.language.as_deref(), Some("en"));
    }

    #[test]
    fn parse_transcript_json_falls_back_to_segments() {
        let body = r#"{
            "text": "  ",
            "segments": [
                {"start":0.0,"end":1.0,"text":"Hello"},
                {"start":1.0,"end":2.0,"text":" world "}
            ]
        }"#;
        let t = parse_transcript_json(body).unwrap();
        assert_eq!(t.text, "Hello world");
    }

    #[test]
    fn parse_transcript_json_empty_is_error() {
        assert!(matches!(
            parse_transcript_json(r#"{"text":""}"#),
            Err(AsrError::EmptyTranscript)
        ));
        assert!(parse_transcript_json("not-json").is_err());
    }

    #[test]
    fn parse_transcript_json_provider_error() {
        let err = parse_transcript_json(r#"{"error":"model busy"}"#).unwrap_err();
        assert!(err.to_string().contains("model busy"), "{err}");
    }
}
