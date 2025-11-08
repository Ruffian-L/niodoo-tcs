//! Ablation Studies Framework
//!
//! Systematic component testing to quantify each component's contribution
//! to performance and cognitive capabilities.

pub mod topology_ablation;
pub mod erag_ablation;
pub mod compass_ablation;
pub mod learning_ablation;
pub mod curator_ablation;

pub use topology_ablation::*;
pub use erag_ablation::*;
pub use compass_ablation::*;
pub use learning_ablation::*;
pub use curator_ablation::*;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Ablation experiment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationConfig {
    pub topology_enabled: bool,
    pub erag_enabled: bool,
    pub compass_enabled: bool,
    pub learning_enabled: bool,
    pub curator_enabled: bool,
}

impl Default for AblationConfig {
    fn default() -> Self {
        Self {
            topology_enabled: true,
            erag_enabled: true,
            compass_enabled: true,
            learning_enabled: true,
            curator_enabled: true,
        }
    }
}

/// Ablation experiment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationResult {
    pub config: AblationConfig,
    pub metrics: AblationMetrics,
    pub timestamp: String,
}

/// Metrics collected during ablation experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationMetrics {
    pub rouge_scores: Vec<f64>,
    pub latency_ms: Vec<f64>,
    pub memory_retrieval_accuracy: Option<f64>,
    pub learning_rate: Option<f64>,
    pub response_quality: Vec<f64>,
}

impl AblationMetrics {
    pub fn mean_rouge(&self) -> f64 {
        if self.rouge_scores.is_empty() {
            return 0.0;
        }
        self.rouge_scores.iter().sum::<f64>() / self.rouge_scores.len() as f64
    }

    pub fn mean_latency(&self) -> f64 {
        if self.latency_ms.is_empty() {
            return 0.0;
        }
        self.latency_ms.iter().sum::<f64>() / self.latency_ms.len() as f64
    }

    pub fn mean_quality(&self) -> f64 {
        if self.response_quality.is_empty() {
            return 0.0;
        }
        self.response_quality.iter().sum::<f64>() / self.response_quality.len() as f64
    }
}

/// Run ablation study comparing baseline vs component-disabled
pub async fn run_ablation_study(
    component_name: &str,
    config: AblationConfig,
    test_prompts: Vec<String>,
    output_dir: PathBuf,
) -> anyhow::Result<AblationResult> {
    // This will be implemented by specific ablation modules
    match component_name {
        "topology" => topology_ablation::run_topology_ablation(config, test_prompts, output_dir).await,
        "erag" => erag_ablation::run_erag_ablation(config, test_prompts, output_dir).await,
        "compass" => compass_ablation::run_compass_ablation(config, test_prompts, output_dir).await,
        "learning" => learning_ablation::run_learning_ablation(config, test_prompts, output_dir).await,
        "curator" => curator_ablation::run_curator_ablation(config, test_prompts, output_dir).await,
        _ => anyhow::bail!("Unknown component: {}", component_name),
    }
}



