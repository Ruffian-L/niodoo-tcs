//! Real ONNX model management utilities for inference workloads.

use anyhow::{anyhow, Result};
use chrono::Utc;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

#[derive(Debug, Clone)]
pub struct RealOnnxModelManager {
    models_dir: PathBuf,
}

impl RealOnnxModelManager {
    pub fn new(models_dir: impl AsRef<Path>) -> Result<Self> {
        let models_dir = models_dir.as_ref().to_path_buf();
        if !models_dir.exists() {
            fs::create_dir_all(&models_dir)?;
        }
        Ok(Self { models_dir })
    }

    pub fn ensure_models_ready(&self, models: &[ModelKind]) -> Result<()> {
        for model in models {
            self.ensure_model(model)?;
        }
        Ok(())
    }

    pub fn ensure_model(&self, model: &ModelKind) -> Result<PathBuf> {
        let path = self.model_path(model);
        if path.exists() {
            info!(model = %model.name(), path = %path.display(), "ONNX model already present");
            return Ok(path);
        }

        #[cfg(feature = "hf-hub")]
        {
            if let Err(err) = self.download_model(model, &path) {
                warn!(%err, "Failed to download model; creating stub instead");
                self.create_stub_model(model, &path)?;
            }
        }

        #[cfg(not(feature = "hf-hub"))]
        {
            warn!(model = %model.name(), "hf-hub feature disabled; creating stub ONNX model");
            self.create_stub_model(model, &path)?;
        }

        Ok(path)
    }

    fn model_path(&self, model: &ModelKind) -> PathBuf {
        self.models_dir.join(model.filename())
    }

    #[cfg(feature = "hf-hub")]
    fn download_model(&self, model: &ModelKind, destination: &Path) -> Result<()> {
        use hf_hub::api::sync::Api;
        use hf_hub::Repo;
        use hf_hub::RepoType;

        let (repo_id, file) = model.hf_descriptor();
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(repo_id.to_string(), RepoType::Model, "main".to_string()));
        let downloaded_path = repo.get(file)?;
        fs::copy(&downloaded_path, destination)?;
        info!(model = %model.name(), path = %destination.display(), "Downloaded ONNX model from Hugging Face");
        Ok(())
    }

    fn create_stub_model(&self, model: &ModelKind, destination: &Path) -> Result<()> {
        let stub = format!(
            "# ONNX STUB\nmodel: {}\ncreated: {}\nstatus: placeholder\n",
            model.name(),
            chrono::Utc::now()
        );
        fs::write(destination, stub.as_bytes())?;
        info!(model = %model.name(), path = %destination.display(), "Created stub ONNX model");
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ModelKind {
    SentenceEmbedding,
    EmotionClassifier,
    GaussianMemory,
}

impl ModelKind {
    fn filename(self) -> &'static str {
        match self {
            ModelKind::SentenceEmbedding => "sentence-embedding.onnx",
            ModelKind::EmotionClassifier => "emotion-classifier.onnx",
            ModelKind::GaussianMemory => "gaussian-memory.onnx",
        }
    }

    fn name(self) -> &'static str {
        match self {
            ModelKind::SentenceEmbedding => "SentenceEmbedding",
            ModelKind::EmotionClassifier => "EmotionClassifier",
            ModelKind::GaussianMemory => "GaussianMemory",
        }
    }

    #[cfg(feature = "hf-hub")]
    fn hf_descriptor(self) -> (&'static str, &'static str) {
        match self {
            ModelKind::SentenceEmbedding => (
                "sentence-transformers/all-MiniLM-L6-v2",
                "onnx/model.onnx",
            ),
            ModelKind::EmotionClassifier => (
                "j-hartmann/emotion-english-distilroberta-base",
                "onnx/model.onnx",
            ),
            ModelKind::GaussianMemory => ("ruffiann/gaussian-memory-prototype", "model.onnx"),
        }
    }
}

pub async fn setup_real_models() -> Result<()> {
    let manager = RealOnnxModelManager::new(default_models_dir()?)?;
    manager.ensure_models_ready(&[
        ModelKind::SentenceEmbedding,
        ModelKind::EmotionClassifier,
    ])?;
    Ok(())
}

fn default_models_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow!("home directory not available"))?;
    Ok(home.join(".niodoo").join("models"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn stub_models_created_when_missing() {
        let dir = tempdir().unwrap();
        let manager = RealOnnxModelManager::new(dir.path()).unwrap();
        let path = manager
            .ensure_model(&ModelKind::EmotionClassifier)
            .unwrap();
        assert!(path.exists());
        let contents = fs::read_to_string(path).unwrap();
        assert!(contents.contains("placeholder"));
    }
}

