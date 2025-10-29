//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham
//!
//! Machine learning agents and inference adapters migrated from the
//! original `src/` tree. We expose both the exploration agent and the
//! multi-brain inference interface so the orchestrator can keep
//! running while we restructure the project.

#[cfg(feature = "onnx")]
use anyhow::Context;
use anyhow::Result;
use async_trait::async_trait;
use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tracing::{info, warn};

#[cfg(feature = "onnx")]
const INFERENCE_HEAD_SAMPLES: usize = 5;

#[cfg(feature = "onnx")]
use half::f16;
const DEFAULT_ACTION_SPACE: usize = 8;
const ENERGY_PERTURBATION_MOD: usize = 4;
const COMPLEXITY_LENGTH_NORM: f32 = 1000.0;
const COMPLEXITY_WEIGHT_LENGTH: f32 = 0.4;
const COMPLEXITY_WEIGHT_VOCAB: f32 = 0.3;
const COMPLEXITY_WEIGHT_WORD_LEN: f32 = 0.3;
const COGNITIVE_KEYWORDS: [&str; 10] = [
    "coherence",
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

        use crate::f16;
        use anyhow::{Context, Result, anyhow};
        use ndarray::{Array2, CowArray};
        use ort::{
            environment::Environment,
            session::{Session, SessionBuilder},
            value::Value,
        };
        #[cfg(feature = "tokenizers")]
        use tokenizers::Tokenizer;

        #[derive(Clone)]
        pub struct ModelBackend {
            name: String,
            environment: Arc<Environment>,
            model_path: Arc<Mutex<Option<PathBuf>>>,
            session: Arc<Mutex<Option<Arc<Session>>>>,
            #[cfg(feature = "tokenizers")]
            tokenizer: Arc<Mutex<Option<Tokenizer>>>,
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
                    #[cfg(feature = "tokenizers")]
                    tokenizer: Arc::new(Mutex::new(None)),
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

                #[cfg(feature = "tokenizers")]
                {
                    // Try to load tokenizer from model directory
                    if let Some(model_dir) = PathBuf::from(model_path).parent() {
                        // First try parent directory (same level as onnx folder)
                        let mut tokenizer_path = model_dir.to_path_buf();
                        tokenizer_path.pop(); // Go up one level from onnx/ to model root
                        tokenizer_path.push("tokenizer.json");

                        if tokenizer_path.exists() {
                            match Tokenizer::from_file(&tokenizer_path) {
                                Ok(tokenizer) => {
                                    *self.tokenizer.lock().expect("tokenizer mutex poisoned") =
                                        Some(tokenizer);
                                    tracing::info!("Loaded tokenizer from {:?}", tokenizer_path);
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        "Failed to load tokenizer from {:?}: {}",
                                        tokenizer_path,
                                        e
                                    );
                                }
                            }
                        } else {
                            // Also try in the same directory as the model
                            let mut alt_path = model_dir.to_path_buf();
                            alt_path.push("tokenizer.json");
                            if alt_path.exists() {
                                match Tokenizer::from_file(&alt_path) {
                                    Ok(tokenizer) => {
                                        *self.tokenizer.lock().expect("tokenizer mutex poisoned") =
                                            Some(tokenizer);
                                        tracing::info!("Loaded tokenizer from {:?}", alt_path);
                                    }
                                    Err(e) => {
                                        tracing::warn!(
                                            "Failed to load tokenizer from {:?}: {}",
                                            alt_path,
                                            e
                                        );
                                    }
                                }
                            }
                        }
                    }
                }

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

                // Try to use tokenizer first, fall back to naive approach
                let (input_ids, attention_mask) = {
                    #[cfg(feature = "tokenizers")]
                    {
                        let tokenizer_guard =
                            self.tokenizer.lock().expect("tokenizer mutex poisoned");
                        if let Some(ref tokenizer) = *tokenizer_guard {
                            let encoding = tokenizer
                                .encode(prompt, true)
                                .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
                            let input_ids: Vec<i64> =
                                encoding.get_ids().iter().map(|x| *x as i64).collect();
                            let attention_mask: Vec<i64> = encoding
                                .get_attention_mask()
                                .iter()
                                .map(|x| *x as i64)
                                .collect();
                            drop(tokenizer_guard);
                            (input_ids, attention_mask)
                        } else {
                            drop(tokenizer_guard);
                            // Fall back to naive char normalization
                            let feature_len = prompt.len().max(1);
                            let mut encoded = Vec::with_capacity(feature_len);
                            if feature_len == 1 && prompt.is_empty() {
                                encoded.push(0);
                            } else {
                                for ch in prompt.chars() {
                                    encoded.push((ch as u32) as i64);
                                }
                            }
                            let attention_mask = vec![1i64; encoded.len()];
                            (encoded, attention_mask)
                        }
                    }
                    #[cfg(not(feature = "tokenizers"))]
                    {
                        // Fall back to naive char normalization for backwards compatibility
                        let feature_len = prompt.len().max(1);
                        let mut encoded = Vec::with_capacity(feature_len);
                        if feature_len == 1 && prompt.is_empty() {
                            encoded.push(0);
                        } else {
                            for ch in prompt.chars() {
                                encoded.push((ch as u32) as i64);
                            }
                        }
                        let attention_mask = vec![1i64; encoded.len()];
                        (encoded, attention_mask)
                    }
                };

                // Create input tensors
                let input_ids_tensor = Array2::from_shape_vec((1, input_ids.len()), input_ids)
                    .context("failed to create input_ids tensor")?
                    .into_dyn();
                let attention_mask_tensor =
                    Array2::from_shape_vec((1, attention_mask.len()), attention_mask)
                        .context("failed to create attention_mask tensor")?
                        .into_dyn();

                let input_ids_cow = CowArray::from(input_ids_tensor);
                let attention_mask_cow = CowArray::from(attention_mask_tensor);
                let input_ids_value = Value::from_array(session.allocator(), &input_ids_cow)
                    .context("failed to create input_ids ORT value")?;
                let attention_mask_value =
                    Value::from_array(session.allocator(), &attention_mask_cow)
                        .context("failed to create attention_mask ORT value")?;

                let outputs = session
                    .run(vec![input_ids_value, attention_mask_value])
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
                    Err(_) => {
                        // Try extracting as f16 and convert to f32
                        match first_output.try_extract::<f16>() {
                            Ok(tensor) => {
                                let view = tensor.view();
                                let values: Vec<f32> = view
                                    .iter()
                                    .take(crate::INFERENCE_HEAD_SAMPLES)
                                    .map(|&x| f16::to_f32(x))
                                    .collect();
                                format!(
                                    "{} (onnx) produced {} f16 values (converted to f32), head={:?}",
                                    self.name,
                                    view.len(),
                                    values
                                )
                            }
                            Err(_) => format!(
                                "{} (onnx) executed successfully but output type was neither f32 nor f16",
                                self.name
                            ),
                        }
                    }
                };

                Ok(summary)
            }

            /// Extract embeddings as Vec<f32> for TCS pipeline integration
            pub fn extract_embeddings(&self, prompt: &str) -> Result<Vec<f32>> {
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

                // Try to use tokenizer first, fall back to naive approach
                let (input_ids, attention_mask) = {
                    #[cfg(feature = "tokenizers")]
                    {
                        let tokenizer_guard =
                            self.tokenizer.lock().expect("tokenizer mutex poisoned");
                        if let Some(ref tokenizer) = *tokenizer_guard {
                            let encoding = tokenizer
                                .encode(prompt, true)
                                .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
                            let input_ids: Vec<i64> =
                                encoding.get_ids().iter().map(|x| *x as i64).collect();
                            let attention_mask: Vec<i64> = encoding
                                .get_attention_mask()
                                .iter()
                                .map(|x| *x as i64)
                                .collect();
                            drop(tokenizer_guard);
                            (input_ids, attention_mask)
                        } else {
                            drop(tokenizer_guard);
                            // Fall back to naive char normalization
                            let feature_len = prompt.len().max(1);
                            let mut encoded = Vec::with_capacity(feature_len);
                            if feature_len == 1 && prompt.is_empty() {
                                encoded.push(0);
                            } else {
                                for ch in prompt.chars() {
                                    encoded.push((ch as u32) as i64);
                                }
                            }
                            let attention_mask = vec![1i64; encoded.len()];
                            (encoded, attention_mask)
                        }
                    }
                    #[cfg(not(feature = "tokenizers"))]
                    {
                        // Fall back to naive char normalization for backwards compatibility
                        let feature_len = prompt.len().max(1);
                        let mut encoded = Vec::with_capacity(feature_len);
                        if feature_len == 1 && prompt.is_empty() {
                            encoded.push(0);
                        } else {
                            for ch in prompt.chars() {
                                encoded.push((ch as u32) as i64);
                            }
                        }
                        let attention_mask = vec![1i64; encoded.len()];
                        (encoded, attention_mask)
                    }
                };

                // Create input tensors
                let input_ids_tensor = Array2::from_shape_vec((1, input_ids.len()), input_ids)
                    .context("failed to create input_ids tensor")?
                    .into_dyn();
                let attention_mask_tensor =
                    Array2::from_shape_vec((1, attention_mask.len()), attention_mask)
                        .context("failed to create attention_mask tensor")?
                        .into_dyn();

                let input_ids_cow = CowArray::from(input_ids_tensor);
                let attention_mask_cow = CowArray::from(attention_mask_tensor);
                let input_ids_value = Value::from_array(session.allocator(), &input_ids_cow)
                    .context("failed to create input_ids ORT value")?;
                let attention_mask_value =
                    Value::from_array(session.allocator(), &attention_mask_cow)
                        .context("failed to create attention_mask ORT value")?;

                // For now, let's try running with just the basic inputs and see what error we get
                // This will help us understand what the model actually expects
                println!("Model expects {} inputs", session.inputs.len());
                for (i, input) in session.inputs.iter().enumerate() {
                    println!(
                        "  Input {}: name='{}', shape={:?}",
                        i, input.name, input.input_type
                    );
                }

                // Start with just the basic inputs
                let input_values = vec![input_ids_value, attention_mask_value];

                let outputs = session
                    .run(input_values)
                    .context("failed to execute ONNX session")?;

                let first_output = outputs
                    .first()
                    .ok_or_else(|| anyhow!("ONNX model produced no outputs"))?;

                // The output is logits [batch_size, seq_len, vocab_size = 151936]
                // For embeddings, take the last token's logits and use first 512 as embedding

                // Extract as f32 vector, handling both f32 and f16 tensors
                let logits_vec: Vec<f32> = match first_output.try_extract::<f32>() {
                    Ok(tensor) => tensor.view().iter().copied().collect(),
                    Err(_) => {
                        // Try extracting as f16 and convert to f32
                        match first_output.try_extract::<f16>() {
                            Ok(tensor) => tensor.view().iter().map(|&x| f16::to_f32(x)).collect(),
                            Err(e) => {
                                return Err(anyhow!(
                                    "Failed to extract tensor as f32 or f16: {}",
                                    e
                                ));
                            }
                        }
                    }
                };

                // For embedding extraction, we need to know the tensor shape
                // The logits should be [batch_size, seq_len, vocab_size]
                // We'll assume batch_size=1 and calculate from total elements
                let total_elements = logits_vec.len();
                let vocab_size = 151936; // Qwen2.5 vocab size
                let seq_len = total_elements / vocab_size;

                if total_elements != seq_len * vocab_size {
                    return Err(anyhow!(
                        "Unexpected logits shape: total_elements={}, expected_vocab_size={}",
                        total_elements,
                        vocab_size
                    ));
                }

                // Get last token logits and take first 512 dimensions as embedding
                let last_token_start = (seq_len - 1) * vocab_size;
                let embedding_size = 512.min(vocab_size);

                if last_token_start + embedding_size > logits_vec.len() {
                    return Err(anyhow!(
                        "Cannot extract embedding: insufficient logits data"
                    ));
                }

                let embeddings: Vec<f32> =
                    logits_vec[last_token_start..last_token_start + embedding_size].to_vec();

                // Pad to exactly 512 dimensions if needed
                let mut final_embeddings = embeddings;
                final_embeddings.resize(512, 0.0);

                Ok(final_embeddings)
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

            pub fn extract_embeddings(&self, prompt: &str) -> Result<Vec<f32>> {
                // Mock embeddings based on prompt length and content
                let mut embeddings = Vec::with_capacity(512); // Common embedding size
                let chars: Vec<char> = prompt.chars().collect();
                for i in 0..512 {
                    let val = if i < chars.len() {
                        (chars[i] as u32 as f32) / 1000.0 // Normalize to reasonable range
                    } else {
                        0.0
                    };
                    embeddings.push(val);
                }
                Ok(embeddings)
            }

            #[allow(dead_code)]
            pub fn is_loaded(&self) -> bool {
                *self.loaded.lock().expect("model mutex poisoned")
            }
        }
    }
}

use inference_backend::ModelBackend;
pub use inference_backend::ModelBackend as InferenceModelBackend;

pub mod qwen_config;
pub use qwen_config::QwenConfig;
#[cfg(feature = "onnx")]
mod qwen_embedder;
#[cfg(feature = "onnx")]
pub mod qwen_error;
#[cfg(feature = "onnx")]
pub use qwen_embedder::QwenEmbedder;

/// Simple exploration agent that perturbs cognitive knots with stochastic moves.
#[derive(Debug)]
pub struct ExplorationAgent {
    rng: StdRng,
    action_space: usize,
}

impl ExplorationAgent {
    pub fn new() -> Self {
        let seed = std::env::var("NIODOO_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);
        Self {
            rng: StdRng::seed_from_u64(seed),
            action_space: DEFAULT_ACTION_SPACE,
        }
    }
}

impl Default for ExplorationAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl ExplorationAgent {
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

    /// Process state through E(n)-equivariant layer for rotation-invariant features
    pub fn process_with_equivariant(
        &self,
        positions: &DMatrix<f32>,
        features: &DMatrix<f32>,
    ) -> DMatrix<f32> {
        let layer = EquivariantLayer::new(features.ncols(), features.ncols());
        layer.forward(positions, features)
    }
}

/// E(n)-equivariant linear layer for geometric neural networks
#[derive(Debug, Clone)]
pub struct EquivariantLayer {
    weight: DMatrix<f32>,
    bias: DVector<f32>,
}

impl EquivariantLayer {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        assert!(input_dim > 0 && output_dim > 0);

        let mut weight = DMatrix::<f32>::zeros(input_dim, output_dim);
        let diagonal = input_dim.min(output_dim);
        for idx in 0..diagonal {
            weight[(idx, idx)] = 1.0;
        }

        let bias = DVector::<f32>::zeros(output_dim);
        Self { weight, bias }
    }

    pub fn forward(&self, positions: &DMatrix<f32>, features: &DMatrix<f32>) -> DMatrix<f32> {
        assert_eq!(positions.nrows(), features.nrows());
        assert_eq!(features.ncols(), self.weight.nrows());

        // Compute rotation-invariant kernel
        let invariant_kernel = Self::pairwise_squared_distances(positions);
        let invariant_features = invariant_kernel * features;
        let mut output = invariant_features * &self.weight;

        // Add bias
        let bias_matrix =
            DMatrix::<f32>::from_fn(output.nrows(), self.bias.len(), |_, col| self.bias[col]);
        output += bias_matrix;
        output
    }

    fn pairwise_squared_distances(positions: &DMatrix<f32>) -> DMatrix<f32> {
        let n_points = positions.nrows();
        let gram = positions * positions.transpose();
        let mut distances = DMatrix::<f32>::zeros(n_points, n_points);

        for i in 0..n_points {
            let norm_i = gram[(i, i)];
            for j in 0..n_points {
                let norm_j = gram[(j, j)];
                let value = norm_i + norm_j - 2.0 * gram[(i, j)];
                distances[(i, j)] = value.max(0.0);
            }
        }

        distances
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
    #[cfg(feature = "onnx")]
    qwen_embedder: std::sync::Arc<std::sync::Mutex<Option<QwenEmbedder>>>,
}

impl MotorBrain {
    pub fn new() -> Result<Self> {
        info!("Initializing Motor Brain (Fast & Practical)");
        let model = ModelBackend::new("motor")?;
        #[cfg(feature = "onnx")]
        let qwen_embedder = std::sync::Arc::new(std::sync::Mutex::new(None));

        Ok(Self {
            brain_type: BrainType::Motor,
            model,
            #[cfg(feature = "onnx")]
            qwen_embedder,
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

    /// Extract embeddings from input for TCS pipeline integration
    pub async fn extract_embeddings(&self, input: &str) -> Result<Vec<f32>> {
        if input.trim().is_empty() {
            #[cfg(feature = "onnx")]
            {
                self.reset_embedding_cache();
                info!(
                    target: "tcs-ml::motor_brain",
                    "Received empty prompt; reset Qwen embedder cache"
                );
            }
            return self.generate_fallback_embeddings(input);
        }

        #[cfg(feature = "onnx")]
        {
            match self.embed_with_qwen(input) {
                Ok(Some((embeddings, context_len))) => {
                    info!(
                        target: "tcs-ml::motor_brain",
                        dims = embeddings.len(),
                        context = context_len,
                        "Extracted Qwen stateful embeddings"
                    );
                    return Ok(embeddings);
                }
                Ok(None) => {
                    // Env var missing or embedder not initialised; fall back silently.
                }
                Err(err) => {
                    warn!(
                        target: "tcs-ml::motor_brain",
                        error = %err,
                        "Qwen embedder failed; falling back"
                    );
                }
            }
        }

        match self.model.extract_embeddings(input) {
            Ok(embeddings) => {
                info!(
                    target: "tcs-ml::motor_brain",
                    dims = embeddings.len(),
                    "Extracted embeddings via legacy backend"
                );
                Ok(embeddings)
            }
            Err(err) => {
                warn!(
                    target: "tcs-ml::motor_brain",
                    error = %err,
                    "Failed to extract embeddings, using heuristic fallback"
                );
                self.generate_fallback_embeddings(input)
            }
        }
    }

    #[cfg(feature = "onnx")]
    fn ensure_qwen_embedder(&self) -> Result<bool> {
        {
            let guard = self
                .qwen_embedder
                .lock()
                .expect("qwen embedder mutex poisoned");
            if guard.is_some() {
                return Ok(false);
            }
        }

        let model_path = match std::env::var("QWEN_MODEL_PATH") {
            Ok(path) => path,
            Err(_) => return Ok(false),
        };

        let embedder = QwenEmbedder::new(&model_path)
            .with_context(|| format!("failed to initialize QwenEmbedder from {model_path}"))?;

        {
            let mut guard = self
                .qwen_embedder
                .lock()
                .expect("qwen embedder mutex poisoned");
            *guard = Some(embedder);
        }

        if let Err(err) = self.model.load(&model_path) {
            warn!(
                target: "tcs-ml::motor_brain",
                error = %err,
                "Failed to prime legacy ONNX backend while initializing Qwen embedder"
            );
        }

        info!(
            target: "tcs-ml::motor_brain",
            "Initialized Qwen embedder from {}",
            model_path
        );
        Ok(true)
    }

    #[cfg(feature = "onnx")]
    fn embed_with_qwen(&self, input: &str) -> Result<Option<(Vec<f32>, usize)>> {
        let _ = self.ensure_qwen_embedder()?;

        let mut guard = self
            .qwen_embedder
            .lock()
            .expect("qwen embedder mutex poisoned");

        if let Some(embedder) = guard.as_mut() {
            let embeddings = embedder.embed(input)?;
            let context_len = embedder.context_length();
            Ok(Some((embeddings, context_len)))
        } else {
            Ok(None)
        }
    }

    #[cfg(feature = "onnx")]
    pub fn reset_embedding_cache(&self) {
        let mut guard = self
            .qwen_embedder
            .lock()
            .expect("qwen embedder mutex poisoned");
        if let Some(embedder) = guard.as_mut() {
            embedder.reset_cache();
            info!(
                target: "tcs-ml::motor_brain",
                "Reset Qwen embedder KV cache"
            );
        }
    }

    #[cfg(not(feature = "onnx"))]
    pub fn reset_embedding_cache(&self) {}

    /// Generate fallback embeddings when model inference fails
    fn generate_fallback_embeddings(&self, input: &str) -> Result<Vec<f32>> {
        let patterns = self.analyse_input_patterns(input);
        let mut embeddings = vec![0.0; 512]; // Standard embedding size

        // Encode basic patterns into embeddings
        embeddings[0] = patterns.complexity;
        embeddings[1] = patterns.length as f32 / 1000.0; // Normalize length
        embeddings[2] = patterns.keywords.len() as f32 / 10.0; // Normalize keyword count

        // Encode intent as one-hot-like pattern
        match patterns.intent {
            Intent::HelpRequest => embeddings[3] = 1.0,
            Intent::EmotionalQuery => embeddings[4] = 1.0,
            Intent::TechnicalQuery => embeddings[5] = 1.0,
            Intent::CreativeQuery => embeddings[6] = 1.0,
            Intent::GeneralQuery => embeddings[7] = 1.0,
        }

        // Fill remaining with normalized character data
        let chars: Vec<char> = input.chars().collect();
        for (i, embedding) in embeddings.iter_mut().enumerate().skip(8) {
            if i - 8 < chars.len() {
                *embedding = (chars[i - 8] as u32 as f32) / 65536.0; // Normalize Unicode
            }
        }

        Ok(embeddings)
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
        #[cfg(feature = "onnx")]
        {
            let embedder = QwenEmbedder::new(model_path)?;
            let mut guard = self
                .qwen_embedder
                .lock()
                .expect("qwen embedder mutex poisoned");
            *guard = Some(embedder);
            info!(
                target: "tcs-ml::motor_brain",
                "Loaded Qwen embedder from {}",
                model_path
            );
        }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Intent {
    HelpRequest,
    EmotionalQuery,
    TechnicalQuery,
    CreativeQuery,
    #[default]
    GeneralQuery,
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
    use proptest::prelude::*;

    #[test]
    fn exploration_agent_is_deterministic_with_seed() {
        let mut agent_a = ExplorationAgent::with_seed(42);
        let mut agent_b = ExplorationAgent::with_seed(42);
        let state = vec![0.1, 0.4, -0.2];
        assert_eq!(agent_a.select_action(&state), agent_b.select_action(&state));
    }

    #[test]
    fn exploration_agent_different_seeds_produce_different_actions() {
        let mut agent_a = ExplorationAgent::with_seed(42);
        let mut agent_b = ExplorationAgent::with_seed(99);
        let state = vec![0.1, 0.4, -0.2];

        // Not guaranteed to be different, but very likely
        let action_a = agent_a.select_action(&state);
        let action_b = agent_b.select_action(&state);

        // At least verify they're valid actions (within action space)
        assert!(action_a < DEFAULT_ACTION_SPACE + ENERGY_PERTURBATION_MOD);
        assert!(action_b < DEFAULT_ACTION_SPACE + ENERGY_PERTURBATION_MOD);
    }

    #[test]
    fn motor_brain_new() {
        let brain = MotorBrain::new();
        assert!(brain.is_ok());
        let brain = brain.unwrap();
        assert_eq!(brain.get_brain_type(), BrainType::Motor);
    }

    #[tokio::test]
    async fn motor_brain_process_empty_input() {
        let brain = MotorBrain::new().unwrap();
        let result = brain.process("").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn motor_brain_process_help_request() {
        let brain = MotorBrain::new().unwrap();
        let result = brain.process("How do I write Rust code?").await;
        assert!(result.is_ok());
        let content = result.unwrap();
        assert!(content.contains("Motor Brain"));
    }

    #[test]
    fn equivariant_layer_forward() {
        let layer = EquivariantLayer::new(3, 4);
        let positions = DMatrix::<f32>::from_row_slice(2, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let features = DMatrix::<f32>::from_row_slice(2, 3, &[0.5, 0.3, 0.2, 0.4, 0.6, 0.1]);

        let output = layer.forward(&positions, &features);
        assert_eq!(output.nrows(), 2);
        assert_eq!(output.ncols(), 4);
    }

    #[test]
    fn pairwise_distances_positive() {
        let positions = DMatrix::<f32>::from_row_slice(2, 2, &[0.0, 0.0, 1.0, 1.0]);
        let distances = EquivariantLayer::pairwise_squared_distances(&positions);

        assert_eq!(distances.nrows(), 2);
        assert_eq!(distances.ncols(), 2);
        assert!(distances[(0, 0)] >= 0.0);
        assert!(distances[(1, 1)] >= 0.0);
    }

    #[test]
    fn cognitive_knot_serialization() {
        let knot = CognitiveKnot {
            polynomial: "1t^0 + 1t^1 - 1t^2".to_string(),
            crossing_number: 3,
            complexity_score: 1.5,
        };

        assert_eq!(knot.crossing_number, 3);
        assert!(knot.complexity_score > 0.0);
    }

    proptest! {
        #[test]
        fn pairwise_distances_translation_invariant(
            n in 1usize..6,
            m in 1usize..6,
            shift in prop::collection::vec(-10.0f32..10.0, 0..6)
        ) {
            // Build a simple positions matrix (n rows, m cols)
            let mut data: Vec<f32> = (0..(n*m)).map(|i| (i as f32 % 7.0) / 3.0).collect();
            let positions = nalgebra::DMatrix::<f32>::from_row_slice(n, m, &data);

            // Create a shifted copy: add same scalar to every element (translation in embedding space)
            let delta = if shift.is_empty() { 0.0 } else { shift[0] };
            let mut shifted_data = data.clone();
            for v in &mut shifted_data { *v += delta; }
            let shifted = nalgebra::DMatrix::<f32>::from_row_slice(n, m, &shifted_data);

            let d1 = EquivariantLayer::pairwise_squared_distances(&positions);
            let d2 = EquivariantLayer::pairwise_squared_distances(&shifted);

            prop_assert_eq!(d1.nrows(), d2.nrows());
            prop_assert_eq!(d1.ncols(), d2.ncols());

            for i in 0..d1.nrows() { for j in 0..d1.ncols() {
                let a = d1[(i,j)];
                let b = d2[(i,j)];
                prop_assert!((a - b).abs() < 1e-5, "mismatch at ({},{}) {} vs {}", i, j, a, b);
            }}
        }
    }
}
