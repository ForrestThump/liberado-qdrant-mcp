#[cfg(any(
    feature = "embed-fastembed",
    feature = "embed-ollama",
    feature = "embed-openai"
))]
use crate::embedding::EmbedError;
use crate::embedding::{Embedder, FakeEmbedder};
use crate::error::LqmError;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct FastembedSection {
    pub model: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OllamaSection {
    #[serde(default = "default_ollama_url")]
    pub url: String,
    pub model: String,
    pub dimension: Option<usize>,
}

fn default_ollama_url() -> String {
    "http://localhost:11434".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAISection {
    #[serde(default = "default_openai_url")]
    pub url: String,
    pub model: String,
    pub api_key: Option<String>,
    pub dimension: Option<usize>,
}

fn default_openai_url() -> String {
    "https://api.openai.com/v1".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbedderConfig {
    #[serde(default = "default_backend")]
    pub backend: String,
    pub dimension: Option<usize>,
    pub fastembed: Option<FastembedSection>,
    pub ollama: Option<OllamaSection>,
    pub openai: Option<OpenAISection>,
}

fn default_backend() -> String {
    "fastembed".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct ConfigFile {
    pub embedding: EmbedderConfig,
}

impl EmbedderConfig {
    #[allow(clippy::result_large_err)]
    pub fn load(path: impl AsRef<Path>) -> Result<Self, LqmError> {
        let content = std::fs::read_to_string(path)?;
        let config_file: ConfigFile = toml::from_str(&content)
            .map_err(|e| LqmError::Other(format!("failed to parse config: {}", e)))?;
        Ok(config_file.embedding)
    }

    pub fn from_env() -> Self {
        let backend =
            std::env::var("EMBEDDING_BACKEND").unwrap_or_else(|_| "fastembed".to_string());
        let dimension = std::env::var("EMBEDDING_DIMENSION")
            .ok()
            .and_then(|v| v.parse().ok());

        let mut config = Self {
            backend,
            dimension,
            fastembed: None,
            ollama: None,
            openai: None,
        };

        if let Ok(model) = std::env::var("EMBEDDING_OLLAMA_MODEL") {
            config.ollama = Some(OllamaSection {
                url: std::env::var("EMBEDDING_OLLAMA_URL")
                    .unwrap_or_else(|_| "http://localhost:11434".to_string()),
                model,
                dimension: std::env::var("EMBEDDING_OLLAMA_DIMENSION")
                    .ok()
                    .and_then(|v| v.parse().ok()),
            });
        }

        if let Ok(model) = std::env::var("EMBEDDING_OPENAI_MODEL") {
            config.openai = Some(OpenAISection {
                url: std::env::var("EMBEDDING_OPENAI_URL")
                    .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
                model,
                api_key: std::env::var("EMBEDDING_OPENAI_API_KEY").ok(),
                dimension: std::env::var("EMBEDDING_OPENAI_DIMENSION")
                    .ok()
                    .and_then(|v| v.parse().ok()),
            });
        }

        if let Ok(model) = std::env::var("EMBEDDING_FASTEMBED_MODEL") {
            config.fastembed = Some(FastembedSection { model: Some(model) });
        }

        config
    }

    #[allow(clippy::result_large_err)]
    pub fn load_or_default(path: Option<&str>) -> Result<Self, LqmError> {
        if let Some(p) = path {
            let path = Path::new(p);
            if path.exists() {
                return Self::load(path);
            }
        }
        Ok(Self::from_env())
    }
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

#[allow(clippy::result_large_err)]
pub fn create_embedder(config: &EmbedderConfig) -> Result<Box<dyn Embedder>, LqmError> {
    match config.backend.as_str() {
        "fake" => {
            log::warn!(
                "using fake embedder (all-zero vectors). Set EMBEDDING_BACKEND or use a config file."
            );
            let dim = config.dimension.unwrap_or(384);
            let embedder = FakeEmbedder::new(dim);
            log::info!("fake embedder created (dim={})", dim);
            Ok(Box::new(embedder))
        }
        #[cfg(feature = "embed-fastembed")]
        "fastembed" => {
            // Default model when neither TOML section nor EMBEDDING_FASTEMBED_MODEL is set —
            // backend defaults to "fastembed", so zero-config startup must still work.
            let default_section = FastembedSection {
                model: Some("AllMiniLML6V2".to_string()),
            };
            let section = config.fastembed.as_ref().unwrap_or(&default_section);
            let model_name = section.model.as_deref().unwrap_or("AllMiniLML6V2");
            let embedder = crate::embedding::FastEmbedder::try_new(model_name).map_err(|e| {
                LqmError::Embed(EmbedError::EmbeddingFailed(format!(
                    "failed to create fastembed: {}",
                    e
                )))
            })?;
            log::info!(
                "fastembed embedder created (model={}, dim={})",
                model_name,
                embedder.dimension()
            );
            Ok(Box::new(embedder))
        }
        #[cfg(not(feature = "embed-fastembed"))]
        "fastembed" => {
            log::warn!(
                "'fastembed' requested but lqm-core was built without embed-fastembed. Falling back to fake. Rebuild with: cargo build --features embed-fastembed"
            );
            let dim = config.dimension.unwrap_or(384);
            let embedder = FakeEmbedder::new(dim);
            log::info!("fake embedder created (dim={})", dim);
            Ok(Box::new(embedder))
        }
        #[cfg(feature = "embed-ollama")]
        "ollama" => {
            let section = config.ollama.as_ref().ok_or_else(|| {
                LqmError::Validation(
                    "ollama backend requires [embedding.ollama] section".to_string(),
                )
            })?;
            let embedder = crate::embedding::OllamaEmbedder::new(
                &section.url,
                &section.model,
                section.dimension.unwrap_or(768),
            );
            log::info!(
                "ollama embedder created (url={}, model={}, dim={})",
                section.url,
                section.model,
                embedder.dimension()
            );
            Ok(Box::new(embedder))
        }
        #[cfg(feature = "embed-openai")]
        "openai" => {
            let section = config.openai.as_ref().ok_or_else(|| {
                LqmError::Validation(
                    "openai backend requires [embedding.openai] section".to_string(),
                )
            })?;
            let api_key = section
                .api_key
                .clone()
                .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                .ok_or_else(|| {
                    LqmError::Validation(
                        "openai backend requires api_key or OPENAI_API_KEY env var".to_string(),
                    )
                })?;
            let embedder = crate::embedding::OpenAIEmbedder::new(
                &section.url,
                &section.model,
                api_key,
                section.dimension.unwrap_or(1536),
            );
            log::info!(
                "openai embedder created (url={}, model={}, dim={})",
                section.url,
                section.model,
                embedder.dimension()
            );
            Ok(Box::new(embedder))
        }
        other => {
            let available = {
                let mut v = vec![];
                #[cfg(feature = "embed-fastembed")]
                v.push("fastembed");
                #[cfg(feature = "embed-ollama")]
                v.push("ollama");
                #[cfg(feature = "embed-openai")]
                v.push("openai");
                if v.is_empty() {
                    v.push("fake");
                }
                v
            };
            Err(LqmError::Validation(format!(
                "unknown embedder backend '{}'. Available backends: {}",
                other,
                available.join(", ")
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal() {
        let toml_str = r#"
[embedding]
backend = "fake"
dimension = 384
"#;
        let config: ConfigFile = toml::from_str(toml_str).unwrap();
        assert_eq!(config.embedding.backend, "fake");
        assert_eq!(config.embedding.dimension, Some(384));
    }

    #[test]
    fn test_parse_ollama() {
        let toml_str = r#"
[embedding]
backend = "ollama"
dimension = 768

[embedding.ollama]
url = "http://localhost:11434"
model = "nomic-embed-text"
"#;
        let config: ConfigFile = toml::from_str(toml_str).unwrap();
        assert_eq!(config.embedding.backend, "ollama");
        assert_eq!(config.embedding.dimension, Some(768));
        let ollama = config.embedding.ollama.unwrap();
        assert_eq!(ollama.model, "nomic-embed-text");
    }

    #[test]
    fn test_parse_openai() {
        let toml_str = r#"
[embedding]
backend = "openai"
dimension = 1536

[embedding.openai]
url = "https://api.openai.com/v1"
model = "text-embedding-3-small"
api_key = "sk-test"
"#;
        let config: ConfigFile = toml::from_str(toml_str).unwrap();
        assert_eq!(config.embedding.backend, "openai");
        let openai = config.embedding.openai.unwrap();
        assert_eq!(openai.model, "text-embedding-3-small");
        assert_eq!(openai.api_key, Some("sk-test".to_string()));
    }

    #[test]
    fn test_load_or_default_no_file() {
        let result = EmbedderConfig::load_or_default(None);
        assert!(result.is_ok());
        let config = result.unwrap();
        assert_eq!(config.backend, "fastembed");
    }

    #[test]
    fn test_create_embedder_fake() {
        let config = EmbedderConfig {
            backend: "fake".to_string(),
            dimension: Some(128),
            fastembed: None,
            ollama: None,
            openai: None,
        };
        let embedder = create_embedder(&config).unwrap();
        assert_eq!(embedder.id(), "fake");
        assert_eq!(embedder.dimension(), 128);
    }

    #[test]
    #[cfg(feature = "embed-fastembed")]
    fn test_create_embedder_fastembed_defaults_without_section() {
        // Zero-config: backend=fastembed and no [embedding.fastembed] must still construct.
        let config = EmbedderConfig {
            backend: "fastembed".to_string(),
            dimension: None,
            fastembed: None,
            ollama: None,
            openai: None,
        };
        let embedder = create_embedder(&config).expect("fastembed should default model");
        assert_eq!(embedder.id(), "fastembed");
        assert!(embedder.dimension() > 0);
    }

    #[test]
    fn test_create_embedder_unknown() {
        let config = EmbedderConfig {
            backend: "nonexistent".to_string(),
            dimension: None,
            fastembed: None,
            ollama: None,
            openai: None,
        };
        let result = create_embedder(&config);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("nonexistent"), "err was: {err}");
        // Available backends depend on unified features (e.g. workspace crates enabling
        // embed-fastembed); always mention at least one known backend name.
        assert!(
            err.contains("Available backends")
                && (err.contains("fake")
                    || err.contains("fastembed")
                    || err.contains("ollama")
                    || err.contains("openai")),
            "err was: {err}"
        );
    }
}
