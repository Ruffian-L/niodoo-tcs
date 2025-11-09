use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{anyhow, Context, Result};
use tcs_ml::qwen_config::QwenConfig;
use tcs_ml::qwen_error::QwenError;
use tcs_ml::QwenEmbedder;

/// Environment variable for overriding the ONNX model path.
const ENV_MODEL_PATH: &str = "NIODOO_EMBED_MODEL";
/// Environment variable for overriding the TOML config path.
const ENV_CONFIG_PATH: &str = "NIODOO_EMBED_CONFIG";
/// Environment variable to force CPU execution.
const ENV_FORCE_CPU: &str = "NIODOO_EMBED_FORCE_CPU";

/// Default ONNX model shipped with the workspace.
const DEFAULT_MODEL_PATH: &str = "/workspace/models/Qwen-Embedding/onnx/model_fp16.onnx";

/// Immutable configuration describing how to initialise the embeddder.
#[derive(Debug, Clone)]
pub struct EmbedderConfig {
    model_path: PathBuf,
    config_path: Option<PathBuf>,
    force_cpu: bool,
}

impl EmbedderConfig {
    /// Create a configuration by reading overrides from the environment.
    pub fn from_env() -> Result<Self> {
        let model_path =
            std::env::var(ENV_MODEL_PATH).unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string());
        let config_path = std::env::var(ENV_CONFIG_PATH).ok().map(PathBuf::from);
        let force_cpu = std::env::var(ENV_FORCE_CPU)
            .ok()
            .and_then(|raw| raw.parse::<bool>().ok())
            .unwrap_or(false);

        let path = PathBuf::from(model_path);
        if !path.exists() {
            return Err(anyhow!(
                "embedding model not found at {} (set {} to override)",
                path.display(),
                ENV_MODEL_PATH
            ));
        }

        if let Some(ref cfg) = config_path {
            if !cfg.exists() {
                return Err(anyhow!(
                    "embedding config not found at {} (set {} to override)",
                    cfg.display(),
                    ENV_CONFIG_PATH
                ));
            }
        }

        Ok(Self {
            model_path: path,
            config_path,
            force_cpu,
        })
    }

    /// Construct from explicit paths.
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let path = model_path.as_ref();
        if !path.exists() {
            return Err(anyhow!("embedding model not found at {}", path.display()));
        }
        Ok(Self {
            model_path: path.to_path_buf(),
            config_path: None,
            force_cpu: false,
        })
    }

    /// Optional TOML configuration path.
    pub fn with_config_path<P: AsRef<Path>>(mut self, config_path: P) -> Result<Self> {
        let cfg = config_path.as_ref();
        if !cfg.exists() {
            return Err(anyhow!("embedding config not found at {}", cfg.display()));
        }
        self.config_path = Some(cfg.to_path_buf());
        Ok(self)
    }

    /// Force CPU execution, bypassing CUDA even if available.
    pub fn with_force_cpu(mut self, force_cpu: bool) -> Self {
        self.force_cpu = force_cpu;
        self
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub fn config_path(&self) -> Option<&Path> {
        self.config_path.as_deref()
    }

    pub fn force_cpu(&self) -> bool {
        self.force_cpu
    }
}

/// Thread-safe wrapper around the Qwen stateful embedder used by NIODOO.
#[derive(Debug)]
pub struct LocalEmbedder {
    inner: Mutex<QwenEmbedder>,
    embed_dim: usize,
    model_path: PathBuf,
}

impl LocalEmbedder {
    /// Initialise the embedder using configuration from the environment.
    pub fn from_env() -> Result<Self> {
        let config = EmbedderConfig::from_env()?;
        Self::new(config)
    }

    /// Construct a new embedder instance using the provided configuration.
    pub fn new(config: EmbedderConfig) -> Result<Self> {
        if config.force_cpu() {
            // Propagate flag so the tcs-ml embedder will pick it up.
            std::env::set_var("QWEN_FORCE_CPU", "1");
        }

        let qwen_config = if let Some(cfg_path) = config.config_path() {
            QwenConfig::from_file(cfg_path.to_str().expect("non UTF-8 path"))
                .context("failed to parse embedding config")?
        } else {
            QwenConfig::default()
        };

        let embed_dim = qwen_config.embed_dim;

        let embedder = QwenEmbedder::with_config(
            config
                .model_path()
                .to_str()
                .ok_or_else(|| anyhow!("embedding model path contains invalid UTF-8"))?,
            qwen_config,
        )
        .map_err(map_qwen_error)
        .context("failed to initialise Qwen embedder")?;

        Ok(Self {
            inner: Mutex::new(embedder),
            embed_dim,
            model_path: config.model_path().to_path_buf(),
        })
    }

    /// Generate an embedding for a single prompt.
    pub fn embed(&self, prompt: &str) -> Result<Vec<f32>> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| anyhow!("embedder mutex poisoned"))?;
        guard.embed(prompt).map_err(map_qwen_error)
    }

    /// Generate embeddings for a batch of prompts, preserving order.
    pub fn embed_batch<S: AsRef<str>, I: IntoIterator<Item = S>>(
        &self,
        prompts: I,
    ) -> Result<Vec<Vec<f32>>> {
        prompts
            .into_iter()
            .map(|p| self.embed(p.as_ref()))
            .collect()
    }

    /// Dimension of the embedding vectors produced by this model.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Path to the underlying ONNX model.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }
}

fn map_qwen_error(err: QwenError) -> anyhow::Error {
    anyhow!(err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_config_defaults_exist() {
        // Ensure default model is present so the embedder can be constructed later.
        std::env::remove_var(ENV_MODEL_PATH);
        let cfg = EmbedderConfig::from_env().expect("default embed config");
        assert!(cfg.model_path().exists(), "default model path missing");
    }
}
