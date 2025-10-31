use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use tokio::sync::Mutex;
use tokio::task;
use tracing::{info, instrument};

use tcs_ml::qwen_error::QwenError;
use tcs_ml::QwenEmbedder;

/// Wraps the stateful ONNX Qwen embedder in an async-friendly API.
#[derive(Clone)]
pub struct QwenStatefulEmbedder {
    inner: Arc<Mutex<QwenEmbedder>>,
    expected_dim: usize,
}

impl QwenStatefulEmbedder {
    pub fn new(model_path: impl AsRef<Path>, expected_dim: usize) -> Result<Self> {
        let model_path = model_path.as_ref();
        if !model_path.exists() {
            return Err(anyhow!("Qwen model not found at {}", model_path.display()));
        }

        let embedder = QwenEmbedder::new(model_path.to_str().unwrap()).with_context(|| {
            format!(
                "failed to initialise Qwen embedder from {}",
                model_path.display()
            )
        })?;
        Ok(Self {
            inner: Arc::new(Mutex::new(embedder)),
            expected_dim,
        })
    }

    #[instrument(skip_all, fields(tokens = prompt.len()))]
    pub async fn embed(&self, prompt: &str) -> Result<Vec<f32>> {
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

    /// Enable Candle backend
    pub fn enable_candle(&mut self, _model_dir: &Path) {
        // Stub implementation
    }

    /// Set mock mode
    pub fn set_mock_mode(&mut self, _mock: bool) {
        // Stub implementation
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
