//! Machine learning agents and inference adapters migrated from the
//! original `src/` tree. We expose both the reinforcement learning agent
//! and the multi-brain inference interface so the orchestrator can keep
//! running while we restructure the project.

use anyhow::Result;
use async_trait::async_trait;
use rand::distributions::{Distribution, Uniform};
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tracing::info;

/// Reinforcement-learning inspired agent from the README that attempts to
/// simplify cognitive knots via exploratory actions.
#[derive(Debug)]
pub struct UntryingAgent {
    rng: StdRng,
    action_space: usize,
}

impl UntryingAgent {
    pub fn new() -> Self {
        Self {
            rng: StdRng::from_entropy(),
            action_space: 8,
        }
    }

    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            action_space: 8,
        }
    }

    pub fn select_action(&mut self, state_embedding: &[f32]) -> usize {
        let energy = state_embedding.iter().map(|v| v.abs()).sum::<f32>();
        let distribution = Uniform::new(0, self.action_space + (energy as usize % 4));
        distribution.sample(&mut self.rng)
    }
}

/// Brain types mirrored from the original `brain.rs` module.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum BrainType {
    Motor,
    Lcars,
    Efficiency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainResponse {
    pub brain_type: BrainType,
    pub content: String,
    pub confidence: f32,
    pub processing_time_ms: u64,
    pub model_info: String,
    pub emotional_impact: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainProcessingResult {
    pub responses: Vec<BrainResponse>,
    pub total_processing_time_ms: u64,
    pub consensus_confidence: f32,
    pub emotional_state: Option<String>,
    pub personality_alignment: Vec<String>,
    pub confidence: f32,
}

#[async_trait]
pub trait Brain: Send + Sync {
    async fn process(&self, input: &str) -> Result<String>;
    async fn load_model(&mut self, model_path: &str) -> Result<()>;
    fn get_brain_type(&self) -> BrainType;
    fn is_ready(&self) -> bool;
}

/// Simplified ONNX runtime shim used until the real integration lands.
#[derive(Clone, Debug)]
pub struct MockOnnxModel {
    name: String,
    loaded: bool,
}

impl MockOnnxModel {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            loaded: false,
        }
    }

    pub fn load(&mut self, _path: &str) {
        self.loaded = true;
    }

    pub fn infer(&self, prompt: &str) -> String {
        format!("{} inference for: {}", self.name, prompt)
    }
}

/// Motor brain implementation migrated from `src/brain.rs`.
#[derive(Clone)]
pub struct MotorBrain {
    brain_type: BrainType,
    model_loaded: bool,
    model: MockOnnxModel,
}

impl MotorBrain {
    pub fn new() -> Result<Self> {
        info!("Initializing Motor Brain (Fast & Practical)");
        Ok(Self {
            brain_type: BrainType::Motor,
            model_loaded: true,
            model: MockOnnxModel::new("motor"),
        })
    }

    async fn process_with_ai_inference(&self, input: &str) -> Result<String> {
        let patterns = self.analyse_input_patterns(input);
        let response = match patterns.intent {
            Intent::HelpRequest => self.generate_help_response(&patterns),
            Intent::EmotionalQuery => self.generate_emotional_response(&patterns),
            Intent::TechnicalQuery => self.generate_technical_response(&patterns),
            Intent::CreativeQuery => self.generate_creative_response(&patterns),
            Intent::GeneralQuery => self.generate_general_response(&patterns),
        };
        Ok(format!("Motor Brain Analysis:\n{}", response))
    }

    fn analyse_input_patterns(&self, input: &str) -> InputPatterns {
        let lower = input.to_lowercase();
        let mut patterns = InputPatterns::default();

        if lower.contains("help") || lower.contains("how") && lower.contains("do") {
            patterns.intent = Intent::HelpRequest;
        } else if lower.contains("feel") || lower.contains("emotion") {
            patterns.intent = Intent::EmotionalQuery;
        } else if lower.contains("code") || lower.contains("function") {
            patterns.intent = Intent::TechnicalQuery;
        } else if lower.contains("create") || lower.contains("imagine") {
            patterns.intent = Intent::CreativeQuery;
        } else {
            patterns.intent = Intent::GeneralQuery;
        }

        patterns.length = input.len();
        patterns.complexity = self.calculate_complexity(input);
        patterns.keywords = self.extract_keywords(input);
        patterns
    }

    fn calculate_complexity(&self, input: &str) -> f32 {
        let words: Vec<&str> = input.split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }
        let unique_words: HashSet<&str> = words.iter().copied().collect();
        let avg_word_length = words.iter().map(|w| w.len()).sum::<usize>() as f32 / words.len() as f32;
        let length_factor = (input.len() as f32 / 1000.0).min(1.0);
        let vocab_factor = (unique_words.len() as f32 / words.len() as f32).min(1.0);
        let word_length_factor = (avg_word_length / 10.0).min(1.0);
        length_factor * 0.4 + vocab_factor * 0.3 + word_length_factor * 0.3
    }

    fn extract_keywords(&self, input: &str) -> Vec<String> {
        const TERMS: [&str; 10] = [
            "consciousness",
            "memory",
            "brain",
            "neural",
            "cognitive",
            "emotion",
            "embedding",
            "vector",
            "tensor",
            "algorithm",
        ];
        TERMS
            .iter()
            .filter(|term| input.to_lowercase().contains(*term))
            .map(|term| term.to_string())
            .collect()
    }

    fn generate_help_response(&self, patterns: &InputPatterns) -> String {
        format!("Detected help intent with complexity {:.2}", patterns.complexity)
    }

    fn generate_emotional_response(&self, patterns: &InputPatterns) -> String {
        format!("Detected emotional intent: keywords {:?}", patterns.keywords)
    }

    fn generate_technical_response(&self, patterns: &InputPatterns) -> String {
        format!("Technical analysis, length {}", patterns.length)
    }

    fn generate_creative_response(&self, patterns: &InputPatterns) -> String {
        format!("Creative exploration with {} keywords", patterns.keywords.len())
    }

    fn generate_general_response(&self, patterns: &InputPatterns) -> String {
        format!("General response, complexity {:.2}", patterns.complexity)
    }
}

#[async_trait]
impl Brain for MotorBrain {
    async fn process(&self, input: &str) -> Result<String> {
        self.process_with_ai_inference(input).await
    }

    async fn load_model(&mut self, model_path: &str) -> Result<()> {
        self.model.load(model_path);
        self.model_loaded = true;
        Ok(())
    }

    fn get_brain_type(&self) -> BrainType {
        self.brain_type
    }

    fn is_ready(&self) -> bool {
        self.model_loaded
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Intent {
    HelpRequest,
    EmotionalQuery,
    TechnicalQuery,
    CreativeQuery,
    GeneralQuery,
}

impl Default for Intent {
    fn default() -> Self {
        Intent::GeneralQuery
    }
}

#[derive(Debug, Default, Clone)]
struct InputPatterns {
    intent: Intent,
    length: usize,
    complexity: f32,
    keywords: Vec<String>,
}
