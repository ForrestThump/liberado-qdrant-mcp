//! Pluggable embedding backends (`Embedder` trait) and response parsers.
//!
//! Backends: `FakeEmbedder` (tests), optional FastEmbed / Ollama / OpenAI
//! behind cargo features. Pure JSON parsers are always available for offline tests.

use async_trait::async_trait;
use std::fmt;

#[derive(Debug)]
pub enum EmbedError {
    EmbeddingFailed(String),
}

impl fmt::Display for EmbedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmbedError::EmbeddingFailed(msg) => write!(f, "embedding failed: {}", msg),
        }
    }
}

impl std::error::Error for EmbedError {}

#[async_trait]
pub trait Embedder: Send + Sync + std::fmt::Debug {
    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedError>;
    fn dimension(&self) -> usize;
    fn id(&self) -> &str;
    /// Backend model identifier when known (fastembed/ollama/openai model name).
    fn model(&self) -> Option<&str> {
        None
    }
}

pub struct FakeEmbedder {
    dim: usize,
}

impl FakeEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl std::fmt::Debug for FakeEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakeEmbedder")
            .field("dim", &self.dim)
            .finish()
    }
}

#[async_trait]
impl Embedder for FakeEmbedder {
    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(texts.iter().map(|_| vec![0.0_f32; self.dim]).collect())
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn id(&self) -> &str {
        "fake"
    }
}

#[cfg(feature = "embed-fastembed")]
pub struct FastEmbedder {
    model: std::sync::Mutex<fastembed::TextEmbedding>,
    model_name: String,
    dim: usize,
}

#[cfg(feature = "embed-fastembed")]
impl std::fmt::Debug for FastEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastEmbedder")
            .field("model_name", &self.model_name)
            .field("dim", &self.dim)
            .finish()
    }
}

#[cfg(feature = "embed-fastembed")]
impl FastEmbedder {
    pub fn try_new(model_name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        log::info!("loading fastembed model: {}", model_name);
        use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
        let model = match model_name {
            "AllMiniLML6V2" => EmbeddingModel::AllMiniLML6V2,
            "BGEBaseEnV15" => EmbeddingModel::BGEBaseENV15,
            "BGESmallENV15" => EmbeddingModel::BGESmallENV15,
            other => {
                return Err(format!("unknown fastembed model: {}", other).into());
            }
        };
        let text_embedding = TextEmbedding::try_new(InitOptions::new(model))?;
        let dim = crate::constants::DEFAULT_FASTEMBED_DIM;
        Ok(Self {
            model: std::sync::Mutex::new(text_embedding),
            model_name: model_name.to_string(),
            dim,
        })
    }
}

#[cfg(feature = "embed-fastembed")]
#[async_trait]
impl Embedder for FastEmbedder {
    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedError> {
        log::debug!("fastembed embedding {} texts", texts.len());
        let mut model = self.model.lock().map_err(|e| {
            EmbedError::EmbeddingFailed(format!("failed to lock fastembed model: {}", e))
        })?;
        let embeddings = model
            .embed(texts, None)
            .map_err(|e| EmbedError::EmbeddingFailed(format!("fastembed error: {}", e)))?;
        Ok(embeddings)
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn id(&self) -> &str {
        "fastembed"
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model_name)
    }
}

#[cfg(feature = "embed-ollama")]
#[derive(Debug)]
pub struct OllamaEmbedder {
    client: reqwest::Client,
    url: String,
    model: String,
    dim: usize,
}

#[cfg(feature = "embed-ollama")]
impl OllamaEmbedder {
    pub fn new(url: &str, model: &str, dim: usize) -> Self {
        Self {
            client: reqwest::Client::new(),
            url: url.to_string(),
            model: model.to_string(),
            dim,
        }
    }
}

/// Parse Ollama `/api/embed` JSON into embedding vectors (testable offline).
pub fn parse_ollama_embeddings(json: &serde_json::Value) -> Result<Vec<Vec<f32>>, EmbedError> {
    let embeddings: Vec<Vec<f32>> = json["embeddings"]
        .as_array()
        .ok_or_else(|| EmbedError::EmbeddingFailed("ollama response missing 'embeddings'".into()))?
        .iter()
        .map(|v| {
            v.as_array()
                .unwrap_or(&vec![])
                .iter()
                .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                .collect()
        })
        .collect();
    Ok(embeddings)
}

#[cfg(feature = "embed-ollama")]
#[async_trait]
impl Embedder for OllamaEmbedder {
    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedError> {
        use serde_json::json;
        let body = json!({
            "model": self.model,
            "input": texts,
        });
        let resp = self
            .client
            .post(format!("{}/api/embed", self.url))
            .json(&body)
            .send()
            .await
            .map_err(|e| EmbedError::EmbeddingFailed(format!("ollama request failed: {}", e)))?;
        let json: serde_json::Value = resp.json().await.map_err(|e| {
            EmbedError::EmbeddingFailed(format!("ollama response parse failed: {}", e))
        })?;
        parse_ollama_embeddings(&json)
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn id(&self) -> &str {
        "ollama"
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }
}

#[cfg(feature = "embed-openai")]
#[derive(Debug)]
pub struct OpenAIEmbedder {
    client: reqwest::Client,
    url: String,
    model: String,
    api_key: String,
    dim: usize,
}

#[cfg(feature = "embed-openai")]
impl OpenAIEmbedder {
    pub fn new(url: &str, model: &str, api_key: String, dim: usize) -> Self {
        Self {
            client: reqwest::Client::new(),
            url: url.to_string(),
            model: model.to_string(),
            api_key,
            dim,
        }
    }
}

/// Parse OpenAI `/embeddings` JSON into embedding vectors (testable offline).
pub fn parse_openai_embeddings(json: &serde_json::Value) -> Result<Vec<Vec<f32>>, EmbedError> {
    let embeddings: Vec<Vec<f32>> = json["data"]
        .as_array()
        .ok_or_else(|| EmbedError::EmbeddingFailed("openai response missing 'data'".into()))?
        .iter()
        .map(|item| {
            item["embedding"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                .collect()
        })
        .collect();
    Ok(embeddings)
}

#[cfg(feature = "embed-openai")]
#[async_trait]
impl Embedder for OpenAIEmbedder {
    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedError> {
        use serde_json::json;
        let body = json!({
            "model": self.model,
            "input": texts,
        });
        let resp = self
            .client
            .post(format!("{}/embeddings", self.url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| EmbedError::EmbeddingFailed(format!("openai request failed: {}", e)))?;
        let json: serde_json::Value = resp.json().await.map_err(|e| {
            EmbedError::EmbeddingFailed(format!("openai response parse failed: {}", e))
        })?;
        parse_openai_embeddings(&json)
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn id(&self) -> &str {
        "openai"
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fake_embedder() {
        let embedder = FakeEmbedder::new(128);
        assert_eq!(embedder.dimension(), 128);
        assert_eq!(embedder.id(), "fake");
        let embeddings = embedder
            .embed_batch(vec!["hello".to_string()])
            .await
            .expect("fake embed");
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 128);
        assert!(embeddings[0].iter().all(|&x| x == 0.0));
    }

    #[tokio::test]
    async fn test_fake_embedder_batch() {
        let embedder = FakeEmbedder::new(64);
        let texts: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let embeddings = embedder.embed_batch(texts).await.expect("fake batch");
        assert_eq!(embeddings.len(), 3);
        for emb in embeddings {
            assert_eq!(emb.len(), 64);
            assert!(emb.iter().all(|&x| x == 0.0));
        }
    }

    #[test]
    fn test_embed_error_display() {
        let err = EmbedError::EmbeddingFailed("test error".to_string());
        assert!(err.to_string().contains("test error"));
    }

    /// Offline parse tests for remote embedder response shapes (no HTTP).
    #[test]
    fn parse_ollama_embeddings_happy_and_missing() {
        let ok = serde_json::json!({
            "embeddings": [[0.1, 0.2], [0.3, 0.4, 0.5]]
        });
        let vecs = parse_ollama_embeddings(&ok).expect("parse");
        assert_eq!(vecs.len(), 2);
        assert_eq!(vecs[0], vec![0.1, 0.2]);
        assert_eq!(vecs[1].len(), 3);

        let bad = serde_json::json!({ "data": [] });
        assert!(parse_ollama_embeddings(&bad).is_err());
    }

    #[test]
    fn parse_openai_embeddings_happy_and_missing() {
        let ok = serde_json::json!({
            "data": [
                { "embedding": [1.0, 2.0] },
                { "embedding": [3.0] }
            ]
        });
        let vecs = parse_openai_embeddings(&ok).expect("parse");
        assert_eq!(vecs.len(), 2);
        assert_eq!(vecs[0], vec![1.0, 2.0]);

        let bad = serde_json::json!({ "embeddings": [] });
        assert!(parse_openai_embeddings(&bad).is_err());
    }
}
