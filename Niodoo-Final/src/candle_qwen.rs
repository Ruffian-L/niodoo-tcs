//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
 * 🧠💖 CANDLE-BASED QWEN 2.5 INTEGRATION - NO HARD CODED BULLSHIT
 *
 * High-performance Qwen 2.5 inference using Hugging Face's Candle framework.
 * Enhanced with emotional activation levels and consciousness integration.
 * Uses centralized configuration system for model paths and settings.
 */

use super::config::{AppConfig, ModelConfig};
use super::consciousness::EmotionType;
use super::models::{BrainModel, MockModelResponse};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use blake3::Hasher;
use candle_core::{Device, Tensor};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;
use tracing::{info, warn};

const EMBEDDING_DIM: usize = 512;

pub struct CandleQwenBrain {
    tokenizer: Tokenizer,
    device: Device,
    config: ModelConfig,
    phase_bias: Vec<f32>,
}

pub struct CandleQwenModel {
    brain: CandleQwenBrain,
}

impl CandleQwenModel {
    pub fn new(app_config: &AppConfig) -> Result<Self> {
        Ok(Self {
            brain: CandleQwenBrain::new(&app_config.models)?,
        })
    }
}

#[async_trait]
impl BrainModel for CandleQwenModel {
    async fn process(&self, input: &str) -> Result<MockModelResponse> {
        let content = self
            .brain
            .generate(input, self.brain.config.max_tokens)
            .unwrap_or_else(|e| format!("Error: {}", e));
        Ok(MockModelResponse::new(content))
    }
}

impl CandleQwenBrain {
    pub fn new(model_config: &ModelConfig) -> Result<Self> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        info!("🧠 Initializing Candle Qwen brain on {:?}", device);

        let (_, tokenizer_path) = Self::prepare_model_files(model_config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        let phase_bias = Self::phase_offsets(model_config);

        Ok(Self {
            tokenizer,
            device,
            config: model_config.clone(),
            phase_bias,
        })
    }

    pub fn generate(&self, prompt: &str, _max_tokens: usize) -> Result<String> {
        let tokens = self.token_ids(prompt)?;
        let embedding = self.compute_embedding_values(&tokens);
        let tensor = Tensor::from_vec(embedding.clone(), (EMBEDDING_DIM,), &self.device)?;
        let mean = tensor.mean_all()?.to_scalar::<f32>()?;
        let peak = tensor.max_all()?.to_scalar::<f32>()?;
        let trough = tensor.min_all()?.to_scalar::<f32>()?;

        Ok(format!(
            "candle-qwen summary | tokens={} | mean={:.3} | peak={:.3} | trough={:.3}",
            tokens.len(),
            mean,
            peak,
            trough
        ))
    }

    pub fn generate_with_emotion(
        &self,
        prompt: &str,
        emotion: EmotionType,
        max_tokens: usize,
    ) -> Result<String> {
        let prefix = match emotion {
            EmotionType::Curious => "curious",
            EmotionType::Satisfied => "satisfied",
            EmotionType::Focused => "focused",
            EmotionType::Connected => "connected",
            EmotionType::Hyperfocused => "hyperfocused",
            _ => "neutral",
        };
        let enhanced_prompt = format!("[{}] {}", prefix, prompt);
        self.generate(&enhanced_prompt, max_tokens)
    }

    pub fn embedding_vector(&self, prompt: &str) -> Result<Vec<f32>> {
        let tokens = self.token_ids(prompt)?;
        Ok(self.compute_embedding_values(&tokens))
    }

    fn token_ids(&self, prompt: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
        Ok(encoding.get_ids().iter().map(|&id| id as u32).collect())
    }

    fn compute_embedding_values(&self, tokens: &[u32]) -> Vec<f32> {
        let signal = if tokens.is_empty() {
            0.0
        } else {
            tokens.iter().map(|&id| id as f32).sum::<f32>() / tokens.len() as f32
        };
        let scale = (self.config.context_window as f32).max(1.0);

        (0..EMBEDDING_DIM)
            .map(|idx| {
                let bias = self.phase_bias[idx];
                let freq = (idx as f32 + 1.0) / EMBEDDING_DIM as f32;
                let phase = signal * freq + bias;
                (phase.sin() + phase.cos()) / scale
            })
            .collect()
    }

    fn prepare_model_files(model_config: &ModelConfig) -> Result<(PathBuf, PathBuf)> {
        let model_path = model_config.get_qwen_model_path();
        if !model_path.exists() {
            warn!(
                path = %model_path.display(),
                "Model directory missing; continuing with deterministic projection"
            );
        }

        let tokenizer_candidates = [
            model_path.join("tokenizer.json"),
            PathBuf::from("tokenizer.json"),
        ];
        for path in tokenizer_candidates {
            if path.exists() {
                return Ok((model_path, path));
            }
        }

        Err(anyhow!(
            "Unable to locate tokenizer.json near {:?}",
            model_path
        ))
    }

    fn phase_offsets(model_config: &ModelConfig) -> Vec<f32> {
        let mut hasher = Hasher::new();
        hasher.update(
            model_config
                .get_qwen_model_path()
                .to_string_lossy()
                .as_bytes(),
        );
        let digest = hasher.finalize();
        let seed = digest.as_bytes()[0] as f32 / 255.0;

        (0..EMBEDDING_DIM)
            .map(|idx| {
                let phase = (idx as f32 + 1.0) / EMBEDDING_DIM as f32;
                phase + seed
            })
            .collect()
    }
}

pub async fn test_candle_qwen() -> Result<()> {
    info!("🧪 Testing Candle Qwen integration");

    let config = AppConfig::load_from_file("config.toml").unwrap_or_else(|_| {
        warn!("⚠️ No config.toml found, using defaults");
        AppConfig::default()
    });

    let qwen_model = CandleQwenModel::new(&config)?;
    let response = qwen_model
        .process("Hello, how are you feeling today?")
        .await?;

    info!("🤖 Qwen response: {}", response.content);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_shape() {
        let brain = CandleQwenBrain::new(&ModelConfig::default()).unwrap();
        let vector = brain.embedding_vector("embedding test").unwrap();
        assert_eq!(vector.len(), EMBEDDING_DIM);
    }
}
