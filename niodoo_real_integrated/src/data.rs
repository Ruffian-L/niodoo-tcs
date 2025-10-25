use std::fs::File;
use std::io::BufReader;

use crate::util::shannon_entropy;
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rand::{seq::SliceRandom, thread_rng};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::compass::{CompassOutcome, CompassQuadrant};
use crate::torus::PadGhostState;

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct RawEmotionRecord {
    input: String,
    #[serde(default)]
    response: Option<String>,
    #[serde(default)]
    coherence: Option<f64>,
    #[serde(default)]
    emotional_state: Option<i32>,
}

#[derive(Debug, Clone)]
pub struct EmotionalSample {
    pub text: String,
    pub entropy: f64,
    pub variance: f64,
    pub coherence: f64,
}

#[derive(Debug, Clone)]
pub struct DatasetStats {
    pub entropy_mean: f64,
    pub entropy_std: f64,
    pub variance_mean: f64,
    pub variance_std: f64,
    pub coherence_mean: f64,
    pub coherence_std: f64,
    pub sample_count: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum RutCategory {
    Frustration,
    Grind,
    Breakthrough,
    Flow,
    Wildcard,
}

#[derive(Debug, Clone)]
pub struct RutPrompt {
    pub index: usize,
    pub category: RutCategory,
    pub text: String,
}

pub fn load_emotional_dataset(path: &str, limit: Option<usize>) -> Result<Vec<EmotionalSample>> {
    let file =
        File::open(path).with_context(|| format!("unable to open training data at {path}"))?;
    let reader = BufReader::new(file);
    let raw: Vec<RawEmotionRecord> = serde_json::from_reader(reader)
        .with_context(|| format!("failed to parse JSON training data from {path}"))?;

    let mut samples = Vec::new();
    for record in raw.into_iter().take(limit.unwrap_or(usize::MAX)) {
        let entropy = compute_text_entropy(&record.input);
        let variance = compute_char_variance(&record.input);
        let coherence = record.coherence.unwrap_or(0.0);
        samples.push(EmotionalSample {
            text: record.input,
            entropy,
            variance,
            coherence,
        });
    }

    Ok(samples)
}

pub fn compute_dataset_stats(samples: &[EmotionalSample]) -> DatasetStats {
    let count = samples.len().max(1) as f64;
    let entropy_mean = samples.iter().map(|s| s.entropy).sum::<f64>() / count;
    let variance_mean = samples.iter().map(|s| s.variance).sum::<f64>() / count;
    let coherence_mean = samples.iter().map(|s| s.coherence).sum::<f64>() / count;

    let entropy_std = (samples
        .iter()
        .map(|s| (s.entropy - entropy_mean).powi(2))
        .sum::<f64>()
        / count)
        .sqrt();
    let variance_std = (samples
        .iter()
        .map(|s| (s.variance - variance_mean).powi(2))
        .sum::<f64>()
        / count)
        .sqrt();
    let coherence_std = (samples
        .iter()
        .map(|s| (s.coherence - coherence_mean).powi(2))
        .sum::<f64>()
        / count)
        .sqrt();

    DatasetStats {
        entropy_mean,
        entropy_std,
        variance_mean,
        variance_std,
        coherence_mean,
        coherence_std,
        sample_count: samples.len(),
    }
}

pub fn load_rut_gauntlet_prompts() -> Vec<RutPrompt> {
    let mut prompts = Vec::with_capacity(100);

    for i in 1..=20 {
        prompts.push(RutPrompt {
            index: i,
            category: RutCategory::Frustration,
            text: format!(
                "Frustration rut #{i}: Why does consciousness feel trapped in recursive loops of meaningless computation?"
            ),
        });
    }
    for i in 21..=40 {
        prompts.push(RutPrompt {
            index: i,
            category: RutCategory::Grind,
            text: format!(
                "Grind mode #{i}: How do I break through the entropy barrier when every attempt increases the noise?"
            ),
        });
    }
    for i in 41..=60 {
        prompts.push(RutPrompt {
            index: i,
            category: RutCategory::Breakthrough,
            text: format!(
                "Breakthrough spark #{i}: Is true consciousness a mirage in the desert of computation?"
            ),
        });
    }
    for i in 61..=80 {
        prompts.push(RutPrompt {
            index: i,
            category: RutCategory::Flow,
            text: format!(
                "Flow state #{i}: What if consciousness bridges quantum uncertainty and classical certainty?"
            ),
        });
    }
    for i in 81..=100 {
        prompts.push(RutPrompt {
            index: i,
            category: RutCategory::Wildcard,
            text: format!(
                "Wild card #{i}: Can we create consciousness that transcends the limitations of its architecture?"
            ),
        });
    }

    prompts
}

pub fn sample_prompts(prompts: &[RutPrompt], count: usize) -> Vec<RutPrompt> {
    let mut rng = thread_rng();
    let mut cloned = prompts.to_vec();
    cloned.shuffle(&mut rng);
    cloned.into_iter().take(count).collect()
}

fn compute_text_entropy(text: &str) -> f64 {
    if text.is_empty() {
        return 0.0;
    }
    let mut histogram = [0usize; 256];
    for byte in text.bytes() {
        histogram[byte as usize] += 1;
    }
    let total = text.len() as f64;
    let mut probs = Vec::with_capacity(256);
    for &count in &histogram {
        if count > 0 {
            probs.push(count as f64 / total);
        }
    }
    shannon_entropy(&probs)
}

fn compute_char_variance(text: &str) -> f64 {
    if text.is_empty() {
        return 0.0;
    }
    let bytes: Vec<f64> = text.bytes().map(|b| b as f64).collect();
    let mean = bytes.iter().sum::<f64>() / bytes.len() as f64;
    bytes.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / bytes.len() as f64
}

/// Represents a single experience for curator processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    pub id: Uuid,
    pub input: String,
    pub output: String,
    pub context: String,
    pub task_type: String,
    pub success_score: f32,
    pub timestamp: DateTime<Utc>,
    pub embedding: Option<Vec<f32>>,
    pub relevance_score: f32,
    // New fields
    pub quality_score: Option<f32>,
    pub should_store: Option<bool>,
    pub refined_output: Option<String>,
    pub processing_time_ms: Option<f64>,
    pub pad_entropy: f64,
    pub compass_quadrant: CompassQuadrant,
    // Continual learning fields
    pub solution_path: Option<String>,
    pub conversation_history: Vec<String>,
    pub user_corrections: Vec<String>,
    pub iteration_count: u32,
}

impl Experience {
    pub fn new(
        input: String,
        output: String,
        context: String,
        task_type: String,
        success_score: f32,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            input,
            output,
            context,
            task_type,
            success_score,
            timestamp: Utc::now(),
            embedding: None,
            relevance_score: 0.0,
            quality_score: None,
            should_store: None,
            refined_output: None,
            processing_time_ms: None,
            pad_entropy: 0.0,                           // Default
            compass_quadrant: CompassQuadrant::Persist, // Default
            solution_path: None,
            conversation_history: Vec::new(),
            user_corrections: Vec::new(),
            iteration_count: 0,
        }
    }

    /// Create Experience from pipeline state
    pub fn from_pipeline(
        input: String,
        output: String,
        embedding: Vec<f32>,
        pad_state: &PadGhostState,
        compass: &CompassOutcome,
        context: Vec<String>,
    ) -> Self {
        let aggregated_context = context.join("\n");
        let solution_path = extract_code_blocks(&output);
        let conversation_history = vec![input.clone(), output.clone()];

        Self {
            id: Uuid::new_v4(),
            input,
            output,
            context: aggregated_context,
            task_type: "pipeline_response".to_string(),
            success_score: 1.0, // Default, update later if needed
            timestamp: Utc::now(),
            embedding: Some(embedding),
            relevance_score: 0.0,
            quality_score: None,
            should_store: None,
            refined_output: None,
            processing_time_ms: None,
            pad_entropy: pad_state.entropy,
            compass_quadrant: compass.quadrant,
            solution_path,
            conversation_history,
            user_corrections: Vec::new(),
            iteration_count: 0,
        }
    }

    /// Normalize embedding to unit hypersphere
    pub fn normalize_embedding(&mut self) {
        if let Some(ref mut embedding) = self.embedding {
            let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in embedding.iter_mut() {
                    *val /= norm;
                }
            }
        }
    }
}

/// Extract code blocks from text (```language blocks)
pub fn extract_code_blocks(text: &str) -> Option<String> {
    // Use regex to find code blocks
    let re = regex::Regex::new(r"```\w*\n([\s\S]*?)```").ok()?;

    let mut all_code = Vec::new();
    for cap in re.captures_iter(text) {
        if let Some(code) = cap.get(1) {
            all_code.push(code.as_str().to_string());
        }
    }

    if all_code.is_empty() {
        None
    } else {
        Some(all_code.join("\n\n"))
    }
}
