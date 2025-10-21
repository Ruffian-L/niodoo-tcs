use std::fs;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use nalgebra::DVector;
use prometheus::{register_gauge, register_histogram, register_counter, Counter, Gauge, Histogram};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn, error};
use tracing_subscriber::fmt;

// Re-exports from other modules
use crate::embedding::QwenEmbedder;
use crate::emotional_mapping::EmotionalMapper;
use crate::compass::CompassEngine;
use crate::erag::ERAGSystem;
use crate::tokenizer::TokenizerEngine;
use crate::generation::GenerationEngine;
use crate::learning::LearningLoop;
use crate::empathy_network::CompleteEmpathyNetwork;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Prompt to process
    #[arg(short, long)]
    pub prompt: String,

    /// Number of swarm instances
    #[arg(short, long, default_value = "1")]
    pub swarm: usize,

    /// Output format: csv or json
    #[arg(short, long, default_value = "csv")]
    pub output: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalSample {
    pub text: String,
    pub pad_vector: [f64; 7], // Pleasure, Arousal, Dominance + 4 ghosts
    pub entropy: f64,
}

#[derive(Debug)]
pub struct PipelineMetrics {
    entropy_gauge: Gauge,
    latency_histogram: Histogram,
    threats_counter: Counter,
    healings_counter: Counter,
}

impl PipelineMetrics {
    fn new() -> Result<Self> {
        Ok(Self {
            entropy_gauge: register_gauge!("niodoo_entropy", "Current entropy level")?,
            latency_histogram: register_histogram!("niodoo_latency", "Processing latency")?,
            threats_counter: register_counter!("niodoo_threats", "Threat detection count")?,
            healings_counter: register_counter!("niodoo_healings", "Healing detection count")?,
        })
    }
}

#[derive(Debug)]
pub struct DynamicThresholds {
    pub entropy_high: f64,
    pub variance_spike: f64,
    pub similarity_threshold: f64,
    pub mcts_c: f64,
    pub mirage_sigma: f64,
}

impl DynamicThresholds {
    pub async fn compute_from_data(samples: &[EmotionalSample]) -> Result<Self> {
        let entropies: Vec<f64> = samples.iter().map(|s| s.entropy).collect();
        let entropy_mean = entropies.iter().sum::<f64>() / entropies.len() as f64;
        let entropy_std = (entropies.iter().map(|&e| (e - entropy_mean).powi(2)).sum::<f64>() / entropies.len() as f64).sqrt();

        // Compute variance from PAD vectors
        let variances: Vec<f64> = samples.iter().map(|s| {
            let mean = s.pad_vector.iter().sum::<f64>() / 7.0;
            s.pad_vector.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / 7.0
        }).collect();
        let variance_std = (variances.iter().map(|&v| (v - variances.iter().sum::<f64>() / variances.len() as f64).powi(2)).sum::<f64>() / variances.len() as f64).sqrt();

        // Similarity threshold from average pairwise similarity
        let mut similarities = Vec::new();
        for i in 0..samples.len().min(100) {
            for j in (i+1)..samples.len().min(100) {
                let sim = cosine_similarity(&samples[i].pad_vector, &samples[j].pad_vector);
                similarities.push(sim);
            }
        }
        let similarity_threshold = similarities.iter().sum::<f64>() / similarities.len() as f64;

        Ok(Self {
            entropy_high: entropy_mean + 0.5 * entropy_std, // Lowered threshold for triggers
            variance_spike: 0.05_f64.max(0.1 * variance_std), // Much lower for flat detection
            similarity_threshold,
            mcts_c: entropy_std * 0.2, // Increased exploration 
            mirage_sigma: 0.2 * entropy_mean, // More chaos noise
        })
    }
}

fn cosine_similarity(a: &[f64; 7], b: &[f64; 7]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (norm_a * norm_b)
}

#[derive(Debug)]
pub struct NiodooIntegrated {
    embedder: QwenEmbedder,
    empathy_network: CompleteEmpathyNetwork,
    erag: ERAGSystem,
    tokenizer: TokenizerEngine,
    generator: GenerationEngine,
    learner: LearningLoop,
    metrics: PipelineMetrics,
    thresholds: DynamicThresholds,
}

impl NiodooIntegrated {
    pub async fn new() -> Result<Self> {
        // Load emotional training data
        let samples = load_emotional_samples("emotion_training_data_mock.json")?;

        // Compute dynamic thresholds
        let thresholds = DynamicThresholds::compute_from_data(&samples).await?;

        // Initialize components
        let embedder = QwenEmbedder::new()?;
        let empathy_network = CompleteEmpathyNetwork::new()?;
        let erag = ERAGSystem::new(thresholds.similarity_threshold)?;
        let tokenizer = TokenizerEngine::new(thresholds.mirage_sigma)?;
        let generator = GenerationEngine::new()?;
        let learner = LearningLoop::new()?;

        let metrics = PipelineMetrics::new()?;

        Ok(Self {
            embedder,
            empathy_network,
            erag,
            tokenizer,
            generator,
            learner,
            metrics,
            thresholds,
        })
    }

    pub async fn process_pipeline(&mut self, prompt: &str) -> Result<PipelineResult> {
        let start_time = Instant::now();

        // 1. Qwen Embed (896D + hyperspherical norm)
        println!("🔍 Step 1: Starting embedding...");
        let step1_start = Instant::now();
        let embedding = self.embedder.embed(prompt).await?;
        let normalized_embedding = normalize_hypersphere(&embedding);
        println!("✅ Step 1: Embedding completed in {:?}", step1_start.elapsed());

        // 2. Actor-based empathy network: HeartNode processes social input
        println!("🔍 Step 2: Starting HeartNode processing...");
        let step2_start = Instant::now();
        let emotional_state = self.empathy_network.process_social_input(prompt.to_string(), normalized_embedding.clone()).await?;
        println!("✅ Step 2: HeartNode completed in {:?}", step2_start.elapsed());

        // 3. Actor-based empathy network: CognitiveNode processes emotional state
        println!("🔍 Step 3: Starting CognitiveNode processing...");
        let step3_start = Instant::now();
        let compass_result = self.empathy_network.process_cognitive_input(emotional_state.clone()).await?;
        println!("✅ Step 3: CognitiveNode completed in {:?}", step3_start.elapsed());

        // 4. ERAG collapse
        println!("🔍 Step 4: Starting ERAG collapse...");
        let step4_start = Instant::now();
        let erag_result = self.erag.collapse(&compass_result, &normalized_embedding).await?;
        println!("✅ Step 4: ERAG completed in {:?}", step4_start.elapsed());

        // 5. Tokenizer promo + rut mirage
        println!("🔍 Step 5: Starting tokenizer processing...");
        let step5_start = Instant::now();
        let tokenized = self.tokenizer.process(&erag_result, &emotional_state).await?;
        println!("✅ Step 5: Tokenizer completed in {:?}", step5_start.elapsed());

        // 6. vLLM gen (hybrid echo)
        println!("🔍 Step 6: Starting vLLM generation...");
        let step6_start = Instant::now();
        let generation = self.generator.generate(&tokenized).await?;
        println!("✅ Step 6: vLLM generation completed in {:?}", step6_start.elapsed());

        // 7. Learn loop (QLoRA events)
        println!("🔍 Step 7: Starting learning update...");
        let step7_start = Instant::now();
        let learning_result = self.learner.update(&generation, &emotional_state).await?;
        println!("✅ Step 7: Learning completed in {:?}", step7_start.elapsed());

        let latency = start_time.elapsed().as_millis() as f64;

        // Update metrics
        self.metrics.entropy_gauge.set(emotional_state.entropy);
        self.metrics.latency_histogram.observe(latency);
        if compass_result.is_threat { self.metrics.threats_counter.inc(); }
        if compass_result.is_healing { self.metrics.healings_counter.inc(); }

        Ok(PipelineResult {
            response: generation.text,
            entropy: emotional_state.entropy,
            is_threat: compass_result.is_threat,
            is_healing: compass_result.is_healing,
            latency,
            learning_events: learning_result.events,
        })
    }
}

#[derive(Debug, Serialize)]
pub struct PipelineResult {
    pub response: String,
    pub entropy: f64,
    pub is_threat: bool,
    pub is_healing: bool,
    pub latency: f64,
    pub learning_events: Vec<String>,
}

fn load_emotional_samples(path: &str) -> Result<Vec<EmotionalSample>> {
    let data = fs::read_to_string(path)
        .with_context(|| format!("Failed to read emotional training data from {}", path))?;
    serde_json::from_str(&data)
        .with_context(|| format!("Failed to parse JSON from {}", path))
}

fn normalize_hypersphere(embedding: &[f64]) -> Vec<f64> {
    let norm = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
    embedding.iter().map(|x| x / norm).collect()
}