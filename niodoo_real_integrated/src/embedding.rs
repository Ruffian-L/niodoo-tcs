use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use tokio::sync::Mutex;
use tokio::task;
use tracing::{info, instrument, warn};

use tcs_ml::qwen_error::QwenError;
use tcs_ml::QwenEmbedder;

/// Wraps the stateful ONNX Qwen embedder in an async-friendly API.
#[derive(Clone)]
pub struct QwenStatefulEmbedder {
    inner: Arc<Mutex<Option<QwenEmbedder>>>, // Option to allow mock mode
    expected_dim: usize,
    mock_mode: bool,
}

impl QwenStatefulEmbedder {
    pub fn new(model_path: impl AsRef<Path>, expected_dim: usize) -> Result<Self> {
        let model_path = model_path.as_ref();
        let model_path_str = model_path.to_str().unwrap();
        
        // Check if mock mode is enabled
        let mock_mode = std::env::var("MOCK_MODE")
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);
        
        // If mock mode is enabled, return immediately with mock embedder
        if mock_mode {
            info!("Mock mode enabled for embeddings");
            return Ok(Self {
                inner: Arc::new(Mutex::new(None)),
                expected_dim,
                mock_mode: true,
            });
        }
        
        // Check if this is an Ollama model name (contains ':') or a file path
        let is_ollama_model = model_path_str.contains(':') && !model_path.exists();
        
        // If it's an Ollama model name, try to find a compatible ONNX model file
        if is_ollama_model {
            // Try to find a compatible ONNX model file
            let fallback_paths = [
                "/workspace/models/Qwen2.5-0.5B-Instruct/onnx/model_fp16.onnx",
                "/workspace/models/Qwen2-0.5B-Instruct/onnx/model_fp16.onnx",
                "/workspace/models/hf_cache/models--onnx-community--Qwen2.5-Coder-0.5B-Instruct/snapshots/f0292f665fd307846ff3c318a91a1bc29d091492/onnx/model_fp16.onnx",
                "./models/qwen2-0.5b-instruct-onnx/onnx/model_fp16.onnx",
            ];
            
            let mut found_path: Option<String> = None;
            for path_str in &fallback_paths {
                let path = Path::new(path_str);
                if path.exists() {
                    found_path = Some(path_str.to_string());
                    break;
                }
            }
            
            // If not found in exact paths, search recursively in hf_cache
            if found_path.is_none() {
                use std::fs;
                let search_dirs = [
                    "/workspace/models/hf_cache",
                ];
                for dir in &search_dirs {
                    if let Ok(entries) = fs::read_dir(dir) {
                        for entry in entries.flatten() {
                            let path = entry.path();
                            if path.is_dir() {
                                let onnx_path = path.join("snapshots").join("f0292f665fd307846ff3c318a91a1bc29d091492").join("onnx").join("model_fp16.onnx");
                                if onnx_path.exists() {
                                    found_path = Some(onnx_path.to_string_lossy().to_string());
                                    break;
                                }
                                // Also try any snapshot subdirectory
                                if let Ok(snapshots) = fs::read_dir(path.join("snapshots")) {
                                    for snap in snapshots.flatten() {
                                        let snap_path = snap.path();
                                        if snap_path.is_dir() {
                                            let onnx_path = snap_path.join("onnx").join("model_fp16.onnx");
                                            if onnx_path.exists() {
                                                found_path = Some(onnx_path.to_string_lossy().to_string());
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                            if found_path.is_some() {
                                break;
                            }
                        }
                    }
                    if found_path.is_some() {
                        break;
                    }
                }
            }
            
            if let Some(path) = found_path {
                info!("Using ONNX fallback {} for Ollama model {}", path, model_path_str);
                match QwenEmbedder::new(&path) {
                    Ok(embedder) => {
                        return Ok(Self {
                            inner: Arc::new(Mutex::new(Some(embedder))),
                            expected_dim,
                            mock_mode: false,
                        });
                    }
                    Err(e) => {
                        warn!("Failed to load ONNX model {}: {}, enabling mock mode", path, e);
                        return Ok(Self {
                            inner: Arc::new(Mutex::new(None)),
                            expected_dim,
                            mock_mode: true,
                        });
                    }
                }
            } else {
                // No ONNX model found - enable mock mode
                warn!("No ONNX model found for Ollama model '{}', enabling mock mode for embeddings", model_path_str);
                return Ok(Self {
                    inner: Arc::new(Mutex::new(None)),
                    expected_dim,
                    mock_mode: true,
                });
            }
        }
        
        // Check if file exists
        if !model_path.exists() {
            warn!("Model file not found at {}, enabling mock mode", model_path.display());
            return Ok(Self {
                inner: Arc::new(Mutex::new(None)),
                expected_dim,
                mock_mode: true,
            });
        }

        // For file paths, use the path as-is
        match QwenEmbedder::new(model_path_str) {
            Ok(embedder) => {
                Ok(Self {
                    inner: Arc::new(Mutex::new(Some(embedder))),
                    expected_dim,
                    mock_mode: false,
                })
            }
            Err(e) => {
                warn!("Failed to initialize Qwen embedder from {}: {}, enabling mock mode", model_path_str, e);
                Ok(Self {
                    inner: Arc::new(Mutex::new(None)),
                    expected_dim,
                    mock_mode: true,
                })
            }
        }
    }

    #[instrument(skip_all, fields(tokens = prompt.len()))]
    pub async fn embed(&self, prompt: &str) -> Result<Vec<f32>> {
        if self.mock_mode {
            // Return mock embedding - random normalized vector
            let mut embedding = vec![0.0f32; self.expected_dim];
            use rand::Rng;
            let mut rng = rand::thread_rng();
            for v in &mut embedding {
                *v = rng.gen_range(-1.0..1.0);
            }
            normalize(&mut embedding);
            return Ok(embedding);
        }
        
        // If mock mode is false but inner is None, fall back to mock mode
        let guard = self.inner.lock().await;
        if guard.is_none() {
            warn!("Embedder not initialized, falling back to mock mode");
            let mut embedding = vec![0.0f32; self.expected_dim];
            use rand::Rng;
            let mut rng = rand::thread_rng();
            for v in &mut embedding {
                *v = rng.gen_range(-1.0..1.0);
            }
            normalize(&mut embedding);
            return Ok(embedding);
        }
        
        // Clone the Arc before moving into the blocking task
        let embedder_arc = self.inner.clone();
        let prompt_owned = prompt.to_owned();
        let expected = self.expected_dim;
        
        let embedding = task::spawn_blocking(move || {
            let mut guard = embedder_arc.blocking_lock();
            match guard.as_mut() {
                Some(embedder) => embedder.embed(&prompt_owned),
                None => {
                    // Fallback: return mock embedding
                    let mut embedding = vec![0.0f32; expected];
                    use rand::Rng;
                    let mut rng = rand::thread_rng();
                    for v in &mut embedding {
                        *v = rng.gen_range(-1.0..1.0);
                    }
                    Ok(embedding)
                }
            }
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
    pub fn set_mock_mode(&mut self, mock: bool) {
        self.mock_mode = mock;
        // Note: Can't modify inner here since it's Arc<Mutex<>>, but that's okay
        // The mock_mode flag will be checked in embed()
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
