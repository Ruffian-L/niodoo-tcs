use std::fs::File;
use std::io::BufReader;

use crate::util::shannon_entropy;
use anyhow::{Context, Result};
use rand::{seq::SliceRandom, thread_rng};
use serde::Deserialize;

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

/// Experience tuple for learning/replay buffers and curator
#[derive(Debug, Clone)]
pub struct Experience {
    pub state: Vec<f32>,
    pub action: usize,
    pub reward: f64,
    pub next_state: Vec<f32>,
    pub done: bool,
    pub output: String, // Response text for curator refinement
}

impl Experience {
    /// Create experience from pipeline
    pub fn from_pipeline(
        input: String,
        output: String,
        embedding: Vec<f32>,
        pad_state: &crate::torus::PadGhostState,
        compass: &crate::compass::CompassOutcome,
        context: Vec<String>,
    ) -> Self {
        let state = embedding.clone();
        let next_state = embedding;
        let action = match compass.quadrant {
            crate::compass::CompassQuadrant::Panic => 0,
            crate::compass::CompassQuadrant::Persist => 1,
            crate::compass::CompassQuadrant::Discover => 2,
            crate::compass::CompassQuadrant::Master => 3,
        };
        let reward = compass.intrinsic_reward;
        let done = false;

        Self {
            state,
            action,
            reward,
            next_state,
            done,
            output,
        }
    }
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
