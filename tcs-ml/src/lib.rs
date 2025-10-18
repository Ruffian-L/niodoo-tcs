//! Machine learning agents and inference adapters migrated from the
//! original `src/` tree. We expose both the exploration agent and the
//! multi-brain inference interface so the orchestrator can keep
//! running while we restructure the project.

use anyhow::Result;
use async_trait::async_trait;
use rand::distributions::{Distribution, Uniform};
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tracing::{info, warn};

#[cfg(feature = "onnx")]
const CHAR_NORMALIZATION_DIVISOR: f32 = 255.0;
#[cfg(feature = "onnx")]
const INFERENCE_HEAD_SAMPLES: usize = 5;
const DEFAULT_ACTION_SPACE: usize = 8;
const ENERGY_PERTURBATION_MOD: usize = 4;
const COMPLEXITY_LENGTH_NORM: f32 = 1000.0;
const COMPLEXITY_WEIGHT_LENGTH: f32 = 0.4;
const COMPLEXITY_WEIGHT_VOCAB: f32 = 0.3;
const COMPLEXITY_WEIGHT_WORD_LEN: f32 = 0.3;
const COGNITIVE_KEYWORDS: [&str; 10] = [
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

mod inference_backend {
    #[cfg(not(feature = "onnx"))]
    pub use self::mock::ModelBackend;
    #[cfg(feature = "onnx")]
    pub use self::onnx::ModelBackend;

    #[cfg(feature = "onnx")]
    mod onnx {
        use std::path::PathBuf;
        use std::sync::{Arc, Mutex};

        use anyhow::{anyhow, Context, Result};
        use ndarray::{Array2, CowArray};
        use ort::{
            environment::Environment,
            session::{Session, SessionBuilder},
            value::Value,
        };

        #[derive(Clone)]
        pub struct ModelBackend {
            name: String,
            environment: Arc<Environment>,
            model_path: Arc<Mutex<Option<PathBuf>>>,
            session: Arc<Mutex<Option<Arc<Session>>>>,
        }

        impl ModelBackend {
            pub fn new(name: &str) -> Result<Self> {
                let environment = Environment::builder()
                    .with_name("tcs-ml")
                    .build()
                    .context("failed to build ORT environment")?;

                Ok(Self {
                    name: name.to_string(),
                    environment: Arc::new(environment),
                    model_path: Arc::new(Mutex::new(None)),
                    session: Arc::new(Mutex::new(None)),
                })
            }

            pub fn load(&self, model_path: &str) -> Result<()> {
                let session = SessionBuilder::new(&self.environment)
                    .context("failed to create ORT session builder")?
                    .with_model_from_file(model_path)
                    .with_context(|| format!("failed to load ONNX model from {model_path}"))?;

                *self.model_path.lock().expect("model path mutex poisoned") =
                    Some(PathBuf::from(model_path));

                *self.session.lock().expect("session mutex poisoned") = Some(Arc::new(session));

                Ok(())
            }

            pub fn infer(&self, prompt: &str) -> Result<String> {
                let session = {
                    let guard = self.session.lock().expect("session mutex poisoned");
                    guard
                        .as_ref()
                        .cloned()
                        .ok_or_else(|| anyhow!("ONNX session not available"))?
                };

                if session.inputs.is_empty() {
                    return Err(anyhow!("ONNX model has no inputs"));
                }

                let feature_len = prompt.len().max(1);
                let mut encoded = Vec::with_capacity(feature_len);
                if feature_len == 1 && prompt.is_empty() {
                    encoded.push(0.0);
                } else {
                    for ch in prompt.chars() {
                        // TODO(tokenizers): swap naive char normalization with HuggingFace `tokenizers` once the
                        // production ONNX language model lands.
                        encoded.push(ch as u32 as f32 / crate::CHAR_NORMALIZATION_DIVISOR);
                    }
                }

                let tensor = Array2::from_shape_vec((1, encoded.len()), encoded)
                    .context("failed to convert prompt into tensor")?
                    .into_dyn();
                let tensor = CowArray::from(tensor);
                let input_value = Value::from_array(session.allocator(), &tensor)
                    .context("failed to create ORT input tensor")?;

                let outputs = session
                    .run(vec![input_value])
                    .context("failed to execute ONNX session")?;

                let first_output = outputs
                    .first()
                    .ok_or_else(|| anyhow!("ONNX model produced no outputs"))?;

                let summary = match first_output.try_extract::<f32>() {
                    Ok(tensor) => {
                        let view = tensor.view();
                        let values: Vec<f32> = view
                            .iter()
                            .take(crate::INFERENCE_HEAD_SAMPLES)
                            .copied()
                            .collect();
                        format!(
                            "{} (onnx) produced {} values, head={:?}",
                            self.name,
                            view.len(),
                            values
                        )
                    }
                    Err(_) => format!(
                        "{} (onnx) executed successfully but output type was not f32",
                        self.name
                    ),
                };

                Ok(summary)
            }

            pub fn is_loaded(&self) -> bool {
                self.session
                    .lock()
                    .expect("session mutex poisoned")
                    .is_some()
            }
        }
    }

    #[cfg(not(feature = "onnx"))]
    mod mock {
        use anyhow::Result;
        use std::sync::{Arc, Mutex};

        #[derive(Clone, Debug)]
        pub struct ModelBackend {
            name: String,
            loaded: Arc<Mutex<bool>>,
        }

        impl ModelBackend {
            pub fn new(name: &str) -> Result<Self> {
                Ok(Self {
                    name: name.to_string(),
                    loaded: Arc::new(Mutex::new(true)),
                })
            }

            pub fn load(&self, _model_path: &str) -> Result<()> {
                *self.loaded.lock().expect("model mutex poisoned") = true;
                Ok(())
            }

            pub fn infer(&self, prompt: &str) -> Result<String> {
                Ok(format!("{} inference for: {}", self.name, prompt))
            }

            #[allow(dead_code)]
            pub fn is_loaded(&self) -> bool {
                *self.loaded.lock().expect("model mutex poisoned")
            }
        }
    }
}

use inference_backend::ModelBackend;

/// Simple exploration agent that perturbs cognitive knots with stochastic moves.
#[derive(Debug)]
pub struct ExplorationAgent {
    rng: StdRng,
    action_space: usize,
}

impl ExplorationAgent {
    pub fn new() -> Self {
        Self {
            rng: StdRng::from_entropy(),
            action_space: DEFAULT_ACTION_SPACE,
        }
    }

    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            action_space: DEFAULT_ACTION_SPACE,
        }
    }

    pub fn select_action(&mut self, state_embedding: &[f32]) -> usize {
        let energy = state_embedding.iter().map(|v| v.abs()).sum::<f32>();
        let perturbation = energy as usize % ENERGY_PERTURBATION_MOD;
        let distribution = Uniform::new(0, self.action_space + perturbation);
        distribution.sample(&mut self.rng)
    }
}

#[deprecated = "Use ExplorationAgent; this type alias remains for transitional compatibility."]
pub type UntryingAgent = ExplorationAgent;

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

/// Motor brain implementation migrated from `src/brain.rs`.
#[derive(Clone)]
pub struct MotorBrain {
    brain_type: BrainType,
    model: ModelBackend,
}

impl MotorBrain {
    pub fn new() -> Result<Self> {
        info!("Initializing Motor Brain (Fast & Practical)");
        Ok(Self {
            brain_type: BrainType::Motor,
            model: ModelBackend::new("motor")?,
        })
    }

    async fn process_with_ai_inference(&self, input: &str) -> Result<String> {
        let backend_summary = match self.model.infer(input) {
            Ok(summary) => summary,
            Err(err) => {
                warn!(
                    target: "tcs-ml::motor_brain",
                    error = %err,
                    "Inference backend unavailable; using heuristic-only response"
                );
                format!("{:?} backend unavailable ({err})", self.brain_type)
            }
        };
        let patterns = self.analyse_input_patterns(input);
        let response = match patterns.intent {
            Intent::HelpRequest => self.generate_help_response(&patterns),
            Intent::EmotionalQuery => self.generate_emotional_response(&patterns),
            Intent::TechnicalQuery => self.generate_technical_response(&patterns),
            Intent::CreativeQuery => self.generate_creative_response(&patterns),
            Intent::GeneralQuery => self.generate_general_response(&patterns),
        };
        Ok(format!(
            "Motor Brain Analysis:\n{}\n{}",
            backend_summary, response
        ))
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
        let avg_word_length =
            words.iter().map(|w| w.len()).sum::<usize>() as f32 / words.len() as f32;
        let length_factor = (input.len() as f32 / COMPLEXITY_LENGTH_NORM).min(1.0);
        let vocab_factor = (unique_words.len() as f32 / words.len() as f32).min(1.0);
        let word_length_factor = (avg_word_length / 10.0).min(1.0);
        length_factor * COMPLEXITY_WEIGHT_LENGTH
            + vocab_factor * COMPLEXITY_WEIGHT_VOCAB
            + word_length_factor * COMPLEXITY_WEIGHT_WORD_LEN
    }

    fn extract_keywords(&self, input: &str) -> Vec<String> {
        let lower = input.to_lowercase();
        COGNITIVE_KEYWORDS
            .iter()
            .filter(|term| lower.contains(*term))
            .map(|term| (*term).to_string())
            .collect()
    }

    fn generate_help_response(&self, patterns: &InputPatterns) -> String {
        format!(
            "Detected help intent with complexity {:.2}",
            patterns.complexity
        )
    }

    fn generate_emotional_response(&self, patterns: &InputPatterns) -> String {
        format!(
            "Detected emotional intent: keywords {:?}",
            patterns.keywords
        )
    }

    fn generate_technical_response(&self, patterns: &InputPatterns) -> String {
        format!("Technical analysis, length {}", patterns.length)
    }

    fn generate_creative_response(&self, patterns: &InputPatterns) -> String {
        format!(
            "Creative exploration with {} keywords",
            patterns.keywords.len()
        )
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
        self.model.load(model_path)?;
        Ok(())
    }

    fn get_brain_type(&self) -> BrainType {
        self.brain_type
    }

    fn is_ready(&self) -> bool {
        #[cfg(feature = "onnx")]
        {
            self.model.is_loaded()
        }

        #[cfg(not(feature = "onnx"))]
        {
            true
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exploration_agent_is_deterministic_with_seed() {
        let mut agent_a = ExplorationAgent::with_seed(42);
        let mut agent_b = ExplorationAgent::with_seed(42);
        let state = vec![0.1, 0.4, -0.2];
        assert_eq!(agent_a.select_action(&state), agent_b.select_action(&state));
    }
}
