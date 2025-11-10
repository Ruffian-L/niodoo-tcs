use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use tokio::sync::Mutex;
use tokio::task;
use tracing::{info, instrument, warn};

use tcs_ml::qwen_error::QwenError;
use tcs_ml::QwenEmbedder as TcsQwenEmbedder;

/// Local embedder using ONNX runtime (via tcs-ml) - ported from legacy QwenStatefulEmbedder
#[derive(Clone)]
pub struct LocalEmbedder {
    inner: Arc<Mutex<TcsQwenEmbedder>>,
    expected_dim: usize,
}

impl LocalEmbedder {
    /// Initialize embedder from environment variables
    /// 
    /// Environment variables:
    /// - `NIODOO_EMBED_MODEL`: Path to Qwen ONNX model (required)
    /// - `NIODOO_EMBED_DIM`: Expected embedding dimension (default: 768 - Qwen MRL output)
    /// - `MOCK_MODE`: Set to "true" for mock mode (for testing)
    /// - `QWEN_INIT_TIMEOUT_SECS`: Timeout for initialization (default: 30)
    pub fn from_env() -> Result<Self> {
        let model_path = std::env::var("NIODOO_EMBED_MODEL")
            .context("NIODOO_EMBED_MODEL environment variable not set")?;
        
        let expected_dim = std::env::var("NIODOO_EMBED_DIM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(768);  // Qwen model configured to output 768 dimensions via MRL
        
        Self::new(&model_path, expected_dim)
    }

    /// Create embedder with explicit model path and dimension
    pub fn new(model_path: impl AsRef<Path>, expected_dim: usize) -> Result<Self> {
        let model_path = model_path.as_ref();
        
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
            let result = TcsQwenEmbedder::new(&model_path_str)
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
                warn!("  2. Set QWEN_INIT_TIMEOUT_SECS=<seconds> to increase timeout");
                warn!("  3. Check CUDA/GPU availability: nvidia-smi");
                return Err(anyhow!("Qwen embedder initialization timed out after {} seconds", timeout_secs));
            }
        };
        
        Ok(Self {
            inner: Arc::new(Mutex::new(embedder)),
            expected_dim,
        })
    }

    /// Create a mock embedder for testing (returns random normalized vectors)
    fn new_mock(_expected_dim: usize) -> Result<Self> {
        // Mock mode is handled by tcs_ml::QwenEmbedder when MOCK_MODE env var is set
        // For now, require a valid model path even in mock mode
        Err(anyhow!("Mock embedder requires MOCK_MODE env var set in tcs_ml - provide a valid model path"))
    }

    /// Embed text synchronously (blocks on async operation)
    /// 
    /// This is a convenience method for synchronous contexts.
    /// For async contexts, use `embed_async()` instead.
    #[instrument(skip_all, fields(tokens = prompt.len()))]
    pub fn embed(&self, prompt: &str) -> Result<Vec<f32>> {
        // Use spawn_blocking directly - this works even in async contexts
        let embedder = self.inner.clone();
        let prompt_owned = prompt.to_owned();
        let expected = self.expected_dim;
        
        let result: std::thread::Result<Result<Vec<f32>, QwenError>> = std::thread::scope(|s| {
            s.spawn(|| {
                let mut guard = embedder.blocking_lock();
                guard.embed(&prompt_owned)
            }).join()
        });
        
        let mut embedding = match result {
            Ok(Ok(emb)) => emb,
            Ok(Err(e)) => return Err(anyhow!("Embedding error: {:?}", e)),
            Err(e) => return Err(anyhow!("Thread join error: {:?}", e)),
        };
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

    /// Embed text asynchronously
    #[instrument(skip_all, fields(tokens = prompt.len()))]
    pub async fn embed_async(&self, prompt: &str) -> Result<Vec<f32>> {
        let embedder = self.inner.clone();
        let prompt_owned = prompt.to_owned();
        let expected = self.expected_dim;
        let embedding = task::spawn_blocking(move || {
            let mut guard = embedder.blocking_lock();
            guard.embed(&prompt_owned)
        })
        .await
        .context("embed task join error")?;

        let mut embedding = embedding.map_err(|e: QwenError| anyhow!(e))?;
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

