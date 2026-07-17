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
    dim: usize,
}

#[cfg(feature = "embed-fastembed")]
impl std::fmt::Debug for FastEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastEmbedder")
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
        let dim = 384;
        Ok(Self {
            model: std::sync::Mutex::new(text_embedding),
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
        let embeddings: Vec<Vec<f32>> = json["embeddings"]
            .as_array()
            .ok_or_else(|| {
                EmbedError::EmbeddingFailed("ollama response missing 'embeddings'".into())
            })?
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

    fn dimension(&self) -> usize {
        self.dim
    }

    fn id(&self) -> &str {
        "ollama"
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

    fn dimension(&self) -> usize {
        self.dim
    }

    fn id(&self) -> &str {
        "openai"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fake_embedder() {
        let embedder = FakeEmbedder::new(128);
        assert_eq!(embedder.dimension(), 128);
        assert_eq!(embedder.id(), "fake");
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(embedder.embed_batch(vec!["hello".to_string()]));
        assert!(result.is_ok());
        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 128);
        assert!(embeddings[0].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_fake_embedder_batch() {
        let embedder = FakeEmbedder::new(64);
        let rt = tokio::runtime::Runtime::new().unwrap();
        let texts: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let result = rt.block_on(embedder.embed_batch(texts));
        assert!(result.is_ok());
        let embeddings = result.unwrap();
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
}
