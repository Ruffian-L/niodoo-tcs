#![cfg_attr(not(test), allow(dead_code))]

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use anyhow::{Context, Result, anyhow};
use axum::{Router, extract::Json, routing::post};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

use crate::config::CliArgs;
use crate::pipeline::Pipeline;

/// Dimension expected by the embedding pipeline.
const EMBEDDING_DIMENSION: usize = 896;

static ENV_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

/// Guard that applies a batch of environment overrides and restores them on drop.
struct EnvGuard {
    previous: HashMap<&'static str, Option<String>>,
    _lock: MutexGuard<'static, ()>,
}

impl EnvGuard {
    fn apply(vars: Vec<(&'static str, String)>) -> Result<Self> {
        let lock = ENV_LOCK
            .lock()
            .map_err(|_| anyhow!("failed to acquire env mutex for test setup"))?;

        let mut previous = HashMap::with_capacity(vars.len());
        for (key, value) in vars.iter() {
            previous.insert(*key, std::env::var(key).ok());
            #[allow(unused_unsafe)]
            unsafe {
                std::env::set_var(key, value);
            }
        }

        Ok(Self {
            previous,
            _lock: lock,
        })
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, value) in self.previous.drain() {
            if let Some(value) = value {
                #[allow(unused_unsafe)]
                unsafe {
                    std::env::set_var(key, value);
                }
            } else {
                #[allow(unused_unsafe)]
                unsafe {
                    std::env::remove_var(key);
                }
            }
        }
    }
}

/// Harness returned by [`mock_pipeline`] keeping the mock server and env overrides alive.
pub struct PipelineHarness {
    pipeline: Pipeline,
    server: JoinHandle<()>,
    _env: EnvGuard,
}

impl PipelineHarness {
    /// Borrow the underlying pipeline immutably.
    pub fn pipeline(&self) -> &Pipeline {
        &self.pipeline
    }

    /// Borrow the underlying pipeline mutably.
    pub fn pipeline_mut(&mut self) -> &mut Pipeline {
        &mut self.pipeline
    }
}

impl Drop for PipelineHarness {
    fn drop(&mut self) {
        self.server.abort();
    }
}

#[derive(Deserialize)]
struct EmbedRequest {
    prompt: String,
    #[serde(default)]
    #[allow(dead_code)]
    model: String,
}

#[derive(Serialize)]
struct EmbedResponse {
    embedding: Vec<f32>,
}

async fn embed_handler(Json(request): Json<EmbedRequest>) -> Json<EmbedResponse> {
    let embedding = synthesize_embedding(&request.prompt);
    Json(EmbedResponse { embedding })
}

fn synthesize_embedding(prompt: &str) -> Vec<f32> {
    use blake3::Hasher;

    let mut hasher = Hasher::new();
    hasher.update(prompt.as_bytes());
    let mut reader = hasher.finalize_xof();

    let mut buffer = [0u8; 4];
    let mut values = Vec::with_capacity(EMBEDDING_DIMENSION);
    for _ in 0..EMBEDDING_DIMENSION {
        reader.fill(&mut buffer);
        let raw = u32::from_le_bytes(buffer);
        let scaled = (raw as f32 / u32::MAX as f32) * 2.0 - 1.0;
        values.push(scaled);
    }
    values
}

async fn spawn_mock_embed_server() -> Result<(SocketAddr, JoinHandle<()>)> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .context("failed to bind mock embed server socket")?;
    let addr = listener
        .local_addr()
        .context("failed to resolve mock embed server address")?;
    let app = Router::new().route("/api/embeddings", post(embed_handler));

    let handle = tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app).await {
            tracing::error!(?error, "mock embed server terminated unexpectedly");
        }
    });

    Ok((addr, handle))
}

/// Build a pipeline configured for offline testing by mocking external services.
pub async fn mock_pipeline(stage: &str) -> Result<PipelineHarness> {
    if stage != "embed" {
        return Err(anyhow!("unsupported stage '{stage}' for mock_pipeline"));
    }

    let (addr, server) = spawn_mock_embed_server().await?;
    let endpoint = format!("http://{addr}");

    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = crate_root
        .join("..")
        .canonicalize()
        .context("failed to resolve repository root for tests")?;

    let tokenizer_path = repo_root
        .join("tokenizer.json")
        .canonicalize()
        .context("tokenizer.json missing; run setup scripts")?;
    let training_data_path = repo_root
        .join("data/training_data/emotion_training_data.json")
        .canonicalize()
        .context("emotion training data not found")?;
    let continual_path = repo_root
        .join("data/training_data/existing_continual_training_data.json")
        .canonicalize()
        .context("existing continual training data not found")?;
    let models_dir = repo_root
        .join("models")
        .canonicalize()
        .context("models directory not found")?;
    let qwen_model = models_dir.join("qwen2.5-coder-0.5b-instruct-onnx/onnx/model_quantized.onnx");
    // Allow missing ONNX model in test environment; it's not used by Ollama-based embedding
    let qwen_model_str = if qwen_model.exists() {
        qwen_model.to_string_lossy().into()
    } else {
        // Use a fallback path for test mode; the ONNX model is not actually loaded
        "/dev/null".to_string()
    };

    let overrides = vec![
        ("MOCK_MODE", "1".to_string()),
        ("OLLAMA_URL", endpoint.clone()),
        ("OLLAMA_ENDPOINT", endpoint.clone()),
        ("EMBEDDING_MAX_CONCURRENCY", "1".to_string()),
        (
            "TRAINING_DATA_PATH",
            training_data_path.to_string_lossy().into(),
        ),
        (
            "EMOTIONAL_SEED_PATH",
            continual_path.to_string_lossy().into(),
        ),
        ("TOKENIZER_JSON", tokenizer_path.to_string_lossy().into()),
        ("MODELS_DIR", models_dir.to_string_lossy().into()),
        ("QWEN_MODEL_PATH", qwen_model_str),
        ("NODE_ID", "test-suite".to_string()),
        ("QDRANT_URL", "http://127.0.0.1:6333".to_string()),
        ("VLLM_ENDPOINT", "http://127.0.0.1:5001".to_string()),
        ("VLLM_MODEL", "mock-vllm".to_string()),
        ("GENERATION_TIMEOUT_SECS", "1".to_string()),
        ("GENERATION_MAX_TOKENS", "96".to_string()),
        ("DYNAMIC_TOKEN_MIN", "8".to_string()),
        ("DYNAMIC_TOKEN_MAX", "96".to_string()),
        ("CURATOR_MINIMUM_THRESHOLD", "0.0".to_string()),
        ("CURATOR_QUALITY_THRESHOLD", "0.0".to_string()),
        ("CURATOR_TIMEOUT_SECS", "1".to_string()),
        ("CURATOR_MAX_TOKENS", "64".to_string()),
    ];

    let env_guard = EnvGuard::apply(overrides)?;

    let args = CliArgs::default();
    let pipeline = Pipeline::initialise_with_seed(args, 1337).await?;

    Ok(PipelineHarness {
        pipeline,
        server,
        _env: env_guard,
    })
}
