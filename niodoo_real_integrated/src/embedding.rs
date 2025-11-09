use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use tokio::sync::Mutex;
use tokio::task;
use tracing::{info, instrument, warn};

use tcs_ml::qwen_error::QwenError;
use tcs_ml::QwenEmbedder;

/// Trait for embedding providers - allows swapping embedders
#[async_trait::async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, prompt: &str) -> Result<Vec<f32>>;
}

/// Internal embedder enum for real or mock
enum EmbedderInner {
    Real(QwenEmbedder),
    Mock { expected_dim: usize },
}

/// Wraps the stateful ONNX Qwen embedder in an async-friendly API.
#[derive(Clone)]
pub struct QwenStatefulEmbedder {
    inner: Arc<Mutex<EmbedderInner>>,
    expected_dim: usize,
}

impl QwenStatefulEmbedder {
    pub fn new(model_path: impl AsRef<Path>, expected_dim: usize) -> Result<Self> {
        // Try QWEN_MODEL_PATH env var first, fallback to passed path
        let env_path = std::env::var("QWEN_MODEL_PATH").ok().map(PathBuf::from);
        let model_path = env_path.as_deref().unwrap_or_else(|| model_path.as_ref());
        
        // Check if MOCK_MODE is enabled (for testing)
        let mock_mode = std::env::var("MOCK_MODE")
            .ok()
            .and_then(|v| if v == "true" { Some(true) } else { None })
            .unwrap_or(false);
        
        if mock_mode {
            warn!("MOCK_MODE enabled - creating mock embedder");
            return Self::new_mock(expected_dim);
        }
        
        if !model_path.exists() {
            return Err(anyhow!("Qwen model not found at {}", model_path.display()));
        }

        // Try to initialize with timeout protection
        let timeout_secs = std::env::var("QWEN_INIT_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(30); // 30 second timeout for initialization

        let model_path_str = model_path.to_str()
            .ok_or_else(|| anyhow!("Model path contains invalid UTF-8: {}", model_path.display()))?
            .to_string();
        let (tx, rx) = std::sync::mpsc::channel();
        
        let handle = std::thread::spawn(move || {
            let result = QwenEmbedder::new(&model_path_str)
                .with_context(|| format!("failed to initialise Qwen embedder from {}", model_path_str));
            tx.send(result).ok();
        });

        let embedder = match rx.recv_timeout(std::time::Duration::from_secs(timeout_secs)) {
            Ok(Ok(emb)) => {
                handle.join().ok();
                emb
            }
            Ok(Err(e)) => {
                handle.join().ok();
                return Err(e);
            }
            Err(_) => {
                warn!(
                    timeout_secs = timeout_secs,
                    "Qwen embedder initialization timed out - this may indicate CUDA initialization issues"
                );
                warn!("Suggestions:");
                warn!("  1. Set QWEN_FORCE_CPU=true to disable CUDA");
                warn!("  2. Set QWEN_CUDA_INIT_TIMEOUT_SECS=<seconds> to increase timeout");
                warn!("  3. Check CUDA/GPU availability: nvidia-smi");
                return Err(anyhow!("Qwen embedder initialization timed out after {} seconds", timeout_secs));
            }
        };
        
        Ok(Self {
            inner: Arc::new(Mutex::new(EmbedderInner::Real(embedder))),
            expected_dim,
        })
    }

    /// Create a mock embedder for testing (returns random normalized vectors)
    fn new_mock(expected_dim: usize) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(Mutex::new(EmbedderInner::Mock { expected_dim })),
            expected_dim,
        })
    }
}

#[async_trait::async_trait]
impl Embedder for QwenStatefulEmbedder {
    #[instrument(skip_all, fields(tokens = prompt.len()))]
    async fn embed(&self, prompt: &str) -> Result<Vec<f32>> {
        let embedder = self.inner.clone();
        let prompt_owned = prompt.to_owned();
        let expected = self.expected_dim;
        let embedding = task::spawn_blocking(move || {
            let mut guard = embedder.blocking_lock();
            match &mut *guard {
                EmbedderInner::Real(emb) => emb.embed(&prompt_owned),
                EmbedderInner::Mock { expected_dim } => {
                    use rand::{Rng, SeedableRng};
                    use std::collections::hash_map::DefaultHasher;
                    use std::hash::{Hash, Hasher};
                    
                    let mut hasher = DefaultHasher::new();
                    prompt_owned.hash(&mut hasher);
                    let hash = hasher.finish();
                    
                    let mut rng = rand::rngs::StdRng::seed_from_u64(hash);
                    
                    let mut embedding = Vec::with_capacity(*expected_dim);
                    for _ in 0..*expected_dim {
                        embedding.push(rng.gen_range(-1.0..1.0));
                    }
                    
                    let norm: f32 = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for v in &mut embedding {
                            *v /= norm;
                        }
                    }
                    
                    Ok(embedding)
                }
            }
        })
        .await
        .context("embed task join error")?
        .map_err(|e: QwenError| anyhow!(e))?;

        let mut embedding = embedding;
        if embedding.len() != expected {
            if embedding.len() < expected {
                embedding.resize(expected, 0.0);
            } else {
                embedding.truncate(expected);
            }
        }

        normalize(&mut embedding);
        Ok(embedding)
    }
}

// Backward compatibility - implement the old method signature
impl QwenStatefulEmbedder {
    #[instrument(skip_all, fields(tokens = prompt.len()))]
    pub async fn embed(&self, prompt: &str) -> Result<Vec<f32>> {
        <Self as Embedder>::embed(self, prompt).await
    }
}

fn normalize(vec: &mut [f32]) {
    let norm = vec.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    if norm == 0.0 {
        return;
    }
    for v in vec.iter_mut() {
        *v = (*v as f64 / norm) as f32;
    }
    info!(dim = vec.len(), "normalized embedding to hypersphere");
}
