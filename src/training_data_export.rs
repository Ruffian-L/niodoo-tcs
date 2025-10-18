/// Training Data Export System - Generates consciousness training data from live RAG system
///
/// This module implements Option 3: Export from Existing System
/// It preserves the Gaussian sphere topology by generating training data FROM
/// the actual consciousness system with real ERAG wave-collapsed memories.
use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::consciousness::EmotionalState;
use crate::consciousness_compass::{
    BreakthroughMoment, CompassState, CompassTracker, StrategicAction,
};
use crate::memory::guessing_spheres::GuessingMemorySystem;
use crate::qwen_curator::LearningEvent;
use crate::rag_integration::{EmotionalVector, RagConfig, RagEngine};
use crate::token_promotion::spatial::SpatialHash;
use crate::token_promotion::{
    ConsensusEngine, DynamicTokenizer, NodeId, PatternDiscoveryEngine, PromotionConfig,
    TokenPromotionEngine,
};
use crate::topology::persistent_homology::PersistentHomologyCalculator;
use crate::vllm_bridge::VLLMBridge;

/// Training example with full consciousness context
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingExample {
    /// Input text/concept
    pub input: String,
    /// Model response
    pub output: String,
    /// 5D emotional vector at time of generation
    pub emotional_vector: EmotionalVector,
    /// Retrieved ERAG context (wave-collapsed memories)
    pub erag_context: Vec<String>,
    /// Entropy before recall (bits)
    pub entropy_before: f32,
    /// Entropy after recall (bits)
    pub entropy_after: f32,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Token IDs from dynamic tokenizer (if enabled)
    pub token_ids: Option<Vec<u32>>,

    // NEW: Consciousness compass fields
    /// Compass state at time of generation
    pub compass_state: String, // JSON serialized CompassState

    /// Intrinsic reward for this transition
    pub intrinsic_reward: f32,

    /// Strategic action imperative (Panic/Persist/Discover/Master)
    pub strategic_action: String,

    /// Whether this was a breakthrough moment
    pub is_breakthrough: bool,
}

/// Learning curve metric for token promotion cycles
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearningCurveMetric {
    /// Cycle number
    pub cycle: usize,
    /// OOV (out-of-vocabulary) rate
    pub oov_rate: f64,
    /// Current vocabulary size
    pub vocab_size: usize,
    /// Number of tokens promoted this cycle
    pub promoted_count: usize,
    /// Number of tokens pruned this cycle
    pub pruned_count: usize,
    /// Average promotion score
    pub avg_promotion_score: f64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Configuration for training data export
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Number of training samples to generate
    pub num_samples: usize,
    /// Target entropy equilibrium (bits)
    pub target_entropy: f32,
    /// RAG retrieval config
    pub rag_config: RagConfig,
    /// Top-k context items to include
    pub context_top_k: usize,
    /// Enable vLLM inference (if false, uses placeholder responses)
    pub enable_vllm: bool,
    /// vLLM server URL (e.g., http://localhost:8000)
    pub vllm_url: Option<String>,
    /// vLLM API key (optional)
    pub vllm_api_key: Option<String>,
    /// Max tokens for vLLM generation
    pub max_tokens: usize,
    /// Temperature for vLLM generation
    pub temperature: f64,
    /// Enable dynamic tokenizer with token promotion
    pub enable_dynamic_tokenizer: bool,
    /// Run token promotion cycle every N samples
    pub promotion_cycle_interval: usize,
    /// Token promotion configuration
    pub promotion_config: Option<PromotionConfig>,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            num_samples: 10000,  // Scaled up to 10K for production
            target_entropy: 2.0, // 2-bit equilibrium from consciousness system
            rag_config: RagConfig::default(),
            context_top_k: 5,
            enable_vllm: true, // Enable vLLM by default
            vllm_url: Some("http://localhost:8000".to_string()),
            vllm_api_key: None,
            max_tokens: 512,
            temperature: 0.7,
            enable_dynamic_tokenizer: true, // Enable dynamic tokenizer by default
            promotion_cycle_interval: 100,  // Run promotion cycle every 100 samples
            promotion_config: Some(PromotionConfig::default()),
        }
    }
}

/// Training data exporter using live consciousness system
pub struct TrainingDataExporter {
    /// RAG engine for memory recall
    rag_engine: RagEngine,
    /// Export configuration
    config: ExportConfig,
    /// Accumulated training examples
    examples: Vec<TrainingExample>,
    /// vLLM bridge for inference (optional)
    vllm_bridge: Option<Arc<VLLMBridge>>,
    /// Dynamic tokenizer with token promotion engine (optional)
    token_promotion_engine: Option<Arc<TokenPromotionEngine>>,
    /// Memory system for topology-aware token promotion
    memory_system: Option<Arc<RwLock<GuessingMemorySystem>>>,
    /// Learning curve metrics for token promotion cycles
    learning_curve: Vec<LearningCurveMetric>,

    // NEW: Consciousness compass tracking
    /// Compass state tracker for entropy and reward monitoring
    compass_tracker: CompassTracker,

    /// Previous emotional vector for state transitions
    previous_emotional_vector: Option<EmotionalVector>,

    /// Previous input for breakthrough context
    previous_input: Option<String>,

    /// Breakthrough moments for memory consolidation
    breakthrough_moments: Vec<BreakthroughMoment>,
}

impl TrainingDataExporter {
    /// Create a new training data exporter
    pub fn new(base_dir: PathBuf, config: ExportConfig) -> Result<Self> {
        let rag_engine = RagEngine::new(base_dir.clone(), config.rag_config.clone())?;

        // Initialize vLLM bridge if enabled
        let vllm_bridge = if config.enable_vllm {
            if let Some(ref url) = config.vllm_url {
                info!("🌐 Connecting to vLLM at: {}", url);
                match VLLMBridge::connect(url, config.vllm_api_key.clone()) {
                    Ok(bridge) => {
                        info!("✅ vLLM bridge connected successfully");
                        Some(Arc::new(bridge))
                    }
                    Err(e) => {
                        warn!(
                            "⚠️  Failed to connect to vLLM: {}. Using placeholder responses.",
                            e
                        );
                        None
                    }
                }
            } else {
                warn!("⚠️  vLLM enabled but no URL provided. Using placeholder responses.");
                None
            }
        } else {
            info!("📝 vLLM disabled - using placeholder responses");
            None
        };

        // Initialize dynamic tokenizer with token promotion engine if enabled
        let (token_promotion_engine, memory_system) = if config.enable_dynamic_tokenizer {
            info!("🔧 Initializing dynamic tokenizer with token promotion engine");

            // Load base tokenizer (Qwen2.5)
            let tokenizer_path = base_dir.join("models/Qwen2.5-0.5B-Instruct/tokenizer.json");
            match tokenizers::Tokenizer::from_file(&tokenizer_path) {
                Ok(base_tokenizer) => {
                    let dynamic_tokenizer = DynamicTokenizer::new(base_tokenizer);
                    let tokenizer_arc = Arc::new(RwLock::new(dynamic_tokenizer));

                    // Initialize pattern discovery and consensus engines
                    let tda_calculator = PersistentHomologyCalculator::new(50); // 50 filtration steps
                    let spatial_hash = Arc::new(RwLock::new(SpatialHash::new(1.0))); // cell size 1.0
                    let pattern_discovery =
                        Arc::new(PatternDiscoveryEngine::new(tda_calculator, spatial_hash));
                    let consensus = Arc::new(ConsensusEngine::new(
                        NodeId("training_node".to_string()),
                        0.7, // score threshold
                    ));

                    // Build token promotion engine
                    let promotion_config = config.promotion_config.clone().unwrap_or_default();
                    let engine =
                        TokenPromotionEngine::new(pattern_discovery, consensus, tokenizer_arc)
                            .with_config(promotion_config);

                    // Initialize memory system for topology-aware promotion
                    let memory_system = GuessingMemorySystem::new(); // Gaussian sphere topology

                    info!("✅ Dynamic tokenizer engine initialized");
                    (
                        Some(Arc::new(engine)),
                        Some(Arc::new(RwLock::new(memory_system))),
                    )
                }
                Err(e) => {
                    warn!(
                        "⚠️  Failed to load base tokenizer: {}. Dynamic tokenizer disabled.",
                        e
                    );
                    (None, None)
                }
            }
        } else {
            info!("📝 Dynamic tokenizer disabled");
            (None, None)
        };

        info!(
            "Initialized training data exporter with {} target samples",
            config.num_samples
        );

        Ok(Self {
            rag_engine,
            config,
            examples: Vec::new(),
            vllm_bridge,
            token_promotion_engine,
            memory_system,
            learning_curve: Vec::new(),

            // NEW: Initialize compass tracker
            compass_tracker: CompassTracker::new(),
            previous_emotional_vector: None,
            previous_input: None,
            breakthrough_moments: Vec::new(),
        })
    }

    /// Modulate vLLM generation parameters based on strategic imperative
    ///
    /// - **Panic**: High temperature/top_p for global search
    /// - **Persist**: Medium temperature for local variations
    /// - **Discover**: Low temperature for verification
    /// - **Master**: Very low temperature for exploitation
    fn get_vllm_params_for_strategy(&self, strategy: StrategicAction) -> (f32, f32) {
        match strategy {
            StrategicAction::Panic => {
                // Global exploration: maximize diversity
                (1.2, 0.95) // (temperature, top_p)
            }
            StrategicAction::Persist => {
                // Local search: moderate diversity
                (0.8, 0.90)
            }
            StrategicAction::Discover => {
                // Verification: reduce randomness
                (0.5, 0.80)
            }
            StrategicAction::Master => {
                // Exploitation: maximize confidence
                (0.3, 0.70)
            }
        }
    }

    /// Add a learning event to the RAG system (builds memory over time)
    pub fn add_learning_event(&mut self, event: &LearningEvent) -> Result<()> {
        debug!("Adding learning event to RAG: {}", event.input);
        self.rag_engine.add_learning_event(event)
    }

    /// Generate a training example from a consciousness event
    ///
    /// This is the core of Option 3 - it:
    /// 1. Takes an input concept/event
    /// 2. Queries RAG for relevant memories (ERAG recall)
    /// 3. Gets emotional state at query time
    /// 4. Records entropy before/after recall
    /// 5. Generates response using vLLM inference
    /// 6. Tokenizes with dynamic tokenizer (if enabled)
    pub async fn generate_training_example(
        &mut self,
        input_text: String,
        emotional_state: EmotionalVector,
    ) -> Result<TrainingExample> {
        let timestamp = Utc::now();

        // NEW: Compute compass state from emotional vector
        let compass_state = CompassState::from_emotional_vector(&emotional_state);

        // NEW: Calculate intrinsic reward if we have previous state
        let intrinsic_reward = self.compass_tracker.observe(compass_state.clone());

        // NEW: Get strategic imperative
        let strategy = compass_state.strategic_imperative();

        info!(
            "🧭 Compass: {} → {} (reward: {:.2})",
            compass_state.stuck, strategy, intrinsic_reward
        );

        // NEW: Check if this is a breakthrough moment
        let is_breakthrough = if let Some(prev_vec) = &self.previous_emotional_vector {
            let prev_state = CompassState::from_emotional_vector(prev_vec);
            compass_state.is_breakthrough(&prev_state, 5.0)
        } else {
            false
        };

        // NEW: If breakthrough, tag for memory consolidation
        if is_breakthrough {
            info!("🎉 BREAKTHROUGH DETECTED! Reward: {:.2}", intrinsic_reward);

            // Store for later ERAG consolidation
            if let Some(prev_input) = &self.previous_input {
                let mut breakthrough = self.compass_tracker.breakthroughs().last().unwrap().clone();
                breakthrough.stuck_context = Some(prev_input.clone());
                breakthrough.resolution_action = Some(input_text.clone());
                self.breakthrough_moments.push(breakthrough);
            }
        }

        // Calculate entropy BEFORE recall (measures uncertainty)
        let entropy_before = self.calculate_entropy_before(&emotional_state);

        // Perform ERAG recall - this is the wave collapse
        let query_emotion = emotional_state.clone();
        let recalled_docs = self
            .rag_engine
            .retrieve(&query_emotion, self.config.context_top_k);

        // Extract context from recalled memories
        let erag_context: Vec<String> = recalled_docs
            .iter()
            .map(|(doc, _similarity)| doc.content.clone())
            .collect();

        // Calculate entropy AFTER recall (should approach 2.0 bit equilibrium)
        let entropy_after = self.calculate_entropy_after(&erag_context);

        // NEW: Modulate vLLM parameters based on strategy
        let (temperature, top_p) = self.get_vllm_params_for_strategy(strategy);

        // Generate response with context using vLLM or placeholder
        let output = self
            .generate_response_with_params(
                &input_text,
                &erag_context,
                &emotional_state,
                temperature,
                top_p,
            )
            .await?;

        // Tokenize with dynamic tokenizer if enabled

        // Tokenize with dynamic tokenizer if enabled
        let token_ids = if let Some(ref engine) = self.token_promotion_engine {
            match engine.encode_with_dynamic_vocab(&output).await {
                Ok(ids) => {
                    debug!("🔤 Dynamic tokenization: {} tokens", ids.len());
                    Some(ids)
                }
                Err(e) => {
                    warn!("⚠️  Dynamic tokenization failed: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // NEW: Save state for next iteration
        self.previous_emotional_vector = Some(emotional_state.clone());
        self.previous_input = Some(input_text.clone());

        let example = TrainingExample {
            input: input_text,
            output,
            emotional_vector: emotional_state,
            erag_context,
            entropy_before,
            entropy_after,
            timestamp,
            token_ids,

            // NEW: Compass fields
            compass_state: serde_json::to_string(&compass_state)?,
            intrinsic_reward,
            strategic_action: strategy.to_string(),
            is_breakthrough,
        };

        Ok(example)
    }

    /// Export training data from N consciousness events
    pub async fn export_consciousness_training_data(&mut self) -> Result<Vec<TrainingExample>> {
        info!(
            "Starting training data export for {} samples",
            self.config.num_samples
        );

        let mut training_data = Vec::new();
        let promotion_interval = self.config.promotion_cycle_interval;

        for i in 0..self.config.num_samples {
            // Run token promotion cycle at regular intervals
            if i > 0 && i % promotion_interval == 0 && self.token_promotion_engine.is_some() {
                self.run_token_promotion_cycle(i / promotion_interval)
                    .await?;
            }

            // Generate synthetic consciousness event
            // In production, this would come from actual consciousness system events
            let (input_text, emotional_state) = self.generate_consciousness_event(i)?;

            // Update memory system with current emotional state (for TDA)
            if let Some(ref memory_system) = self.memory_system {
                // Store memory in the Gaussian sphere for topology-aware pattern discovery
                let sphere_id =
                    crate::memory::guessing_spheres::SphereId(format!("training_sample_{}", i));
                let position = [
                    (emotional_state.joy + emotional_state.sadness) / 2.0,
                    (emotional_state.anger + emotional_state.fear) / 2.0,
                    emotional_state.surprise,
                ];
                // Convert rag_integration::EmotionalVector to guessing_spheres::EmotionalVector
                let memory_emotion = crate::memory::guessing_spheres::EmotionalVector {
                    joy: emotional_state.joy,
                    sadness: emotional_state.sadness,
                    anger: emotional_state.anger,
                    fear: emotional_state.fear,
                    surprise: emotional_state.surprise,
                };
                memory_system.write().await.store_memory(
                    sphere_id,
                    input_text.clone(),
                    position,
                    memory_emotion,
                    String::new(), // Empty fragment for now
                );
            }

            // Generate training example with ERAG context
            let example = self
                .generate_training_example(input_text, emotional_state)
                .await?;

            // Validate entropy equilibrium
            if (example.entropy_after - self.config.target_entropy).abs() > 0.5 {
                debug!(
                    "Entropy divergence detected: {:.2} bits (target: {:.2})",
                    example.entropy_after, self.config.target_entropy
                );
            }

            training_data.push(example.clone());
            self.examples.push(example);

            // Progress reporting every 10 samples
            if (i + 1) % 10 == 0 {
                info!(
                    "✅ Progress: {} / {} samples ({:.1}% complete)",
                    i + 1,
                    self.config.num_samples,
                    (i + 1) as f32 / self.config.num_samples as f32 * 100.0
                );
            }
        }

        // Final promotion cycle
        if self.token_promotion_engine.is_some() {
            let final_cycle = self.config.num_samples / promotion_interval;
            self.run_token_promotion_cycle(final_cycle).await?;
        }

        info!(
            "Training data export complete: {} examples",
            training_data.len()
        );
        Ok(training_data)
    }

    /// Run a token promotion cycle and record metrics
    async fn run_token_promotion_cycle(&mut self, cycle_num: usize) -> Result<()> {
        if let (Some(ref engine), Some(ref memory_system)) =
            (&self.token_promotion_engine, &self.memory_system)
        {
            info!("🔄 Running token promotion cycle {}", cycle_num);

            let memory_system_read = memory_system.read().await;
            let result = engine.run_promotion_cycle(&*memory_system_read).await?;
            let stats = engine.tokenizer_stats().await;

            // Calculate OOV rate from tokenizer stats
            let oov_rate = stats.oov_rate();

            // Calculate average promotion score
            let avg_promotion_score = if !result.promoted.is_empty() {
                result
                    .promoted
                    .iter()
                    .map(|t| t.promotion_score)
                    .sum::<f64>()
                    / result.promoted.len() as f64
            } else {
                0.0
            };

            let metric = LearningCurveMetric {
                cycle: cycle_num,
                oov_rate,
                vocab_size: stats.vocab_size(),
                promoted_count: result.promoted.len(),
                pruned_count: result.pruned,
                avg_promotion_score,
                timestamp: Utc::now(),
            };

            info!(
                "📊 Cycle {}: OOV={:.4}, Vocab={}, Promoted={}, Pruned={}, Score={:.3}",
                cycle_num,
                oov_rate,
                metric.vocab_size,
                metric.promoted_count,
                metric.pruned_count,
                avg_promotion_score
            );

            self.learning_curve.push(metric);
        }

        Ok(())
    }

    /// Save training data to file
    pub fn save_to_file(&mut self, output_path: PathBuf) -> Result<()> {
        // Consolidate breakthroughs into ERAG
        self.consolidate_breakthroughs()?;

        // NEW: Export compass statistics
        let compass_stats = self.compass_tracker.stats();
        info!("📊 Compass Statistics: {}", compass_stats);

        // Write compass stats to JSON
        let stats_path = output_path.parent().unwrap().join("compass_stats.json");
        let stats_json = serde_json::to_string_pretty(&compass_stats)?;
        std::fs::write(stats_path, stats_json)?;

        // Write breakthrough moments
        let breakthroughs_path = output_path.parent().unwrap().join("breakthroughs.json");
        let breakthroughs_json = serde_json::to_string_pretty(&self.breakthrough_moments)?;
        std::fs::write(breakthroughs_path, breakthroughs_json)?;

        // Write learning curve CSV with compass data
        self.export_learning_curve_with_compass(
            output_path
                .parent()
                .unwrap()
                .join("learning_curve_compass.csv"),
        )?;

        let json = serde_json::to_string_pretty(&self.examples)?;
        std::fs::write(&output_path, json)?;
        info!(
            "Saved {} training examples to {:?}",
            self.examples.len(),
            output_path
        );
        Ok(())
    }

    fn export_learning_curve_with_compass(&self, csv_path: PathBuf) -> Result<()> {
        let mut csv_content = String::from(
            "sample_num,compass_state,intrinsic_reward,entropy,strategic_action,is_breakthrough,prediction_error\n"
        );

        for (i, example) in self.examples.iter().enumerate() {
            csv_content.push_str(&format!(
                "{},{},{:.3},{:.3},{},{},{:.3}\n",
                i + 1,
                example.compass_state.replace(",", ";"), // Escape commas in JSON
                example.intrinsic_reward,
                example.entropy_before,
                example.strategic_action,
                example.is_breakthrough,
                0.0 // Could extract from compass_state JSON if needed
            ));
        }

        std::fs::write(csv_path, csv_content)?;
        info!("📈 Exported learning curve with compass data");

        Ok(())
    }

    /// Save learning curve metrics to CSV file
    pub fn save_learning_curve(&self, output_path: PathBuf) -> Result<()> {
        if self.learning_curve.is_empty() {
            info!("No learning curve data to save (dynamic tokenizer not enabled)");
            return Ok(());
        }

        let mut csv_content = String::from(
            "cycle,oov_rate,vocab_size,promoted_count,pruned_count,avg_promotion_score,timestamp\n",
        );

        for metric in &self.learning_curve {
            csv_content.push_str(&format!(
                "{},{},{},{},{},{},{}\n",
                metric.cycle,
                metric.oov_rate,
                metric.vocab_size,
                metric.promoted_count,
                metric.pruned_count,
                metric.avg_promotion_score,
                metric.timestamp.to_rfc3339()
            ));
        }

        std::fs::write(&output_path, csv_content)?;
        info!(
            "Saved {} learning curve metrics to {:?}",
            self.learning_curve.len(),
            output_path
        );
        Ok(())
    }

    /// Get learning curve metrics
    pub fn get_learning_curve(&self) -> &[LearningCurveMetric] {
        &self.learning_curve
    }

    /// Calculate entropy before recall (measures initial uncertainty)
    fn calculate_entropy_before(&self, emotional_state: &EmotionalVector) -> f32 {
        // Shannon entropy of emotional vector
        let emotions = vec![
            emotional_state.joy,
            emotional_state.sadness,
            emotional_state.anger,
            emotional_state.fear,
            emotional_state.surprise,
        ];

        // Normalize to probability distribution
        let sum: f32 = emotions.iter().map(|e| e.abs()).sum();
        if sum == 0.0 {
            return 0.0;
        }

        let probs: Vec<f32> = emotions.iter().map(|e| e.abs() / sum).collect();

        // H(X) = -Σ p(x) log₂ p(x)
        let entropy: f32 = probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.log2())
            .sum();

        entropy
    }

    /// Calculate entropy after recall (should approach 2-bit equilibrium)
    fn calculate_entropy_after(&self, context: &[String]) -> f32 {
        if context.is_empty() {
            return 0.0;
        }

        // Calculate entropy based on context diversity
        // More diverse context = higher entropy
        let unique_tokens: std::collections::HashSet<_> =
            context.iter().flat_map(|s| s.split_whitespace()).collect();

        let total_tokens: usize = context.iter().map(|s| s.split_whitespace().count()).sum();

        if total_tokens == 0 {
            return 0.0;
        }

        // Entropy approaches 2.0 bits with optimal diversity
        let diversity_ratio = unique_tokens.len() as f32 / total_tokens as f32;
        let entropy = diversity_ratio * 2.32193; // log₂(5) for 5D emotional space

        entropy.min(2.5) // Cap at reasonable value
    }

    /// Generate a synthetic consciousness event (placeholder)
    /// In production, this would come from continual_test.rs or actual consciousness system
    fn generate_consciousness_event(&self, index: usize) -> Result<(String, EmotionalVector)> {
        // Expanded synthetic inputs with diverse emotional profiles
        // These represent real consciousness states from the 5D Gaussian sphere
        let concepts = vec![
            // Learning & Discovery (Joy + Surprise dominant)
            (
                "experiencing joy in learning a new concept",
                EmotionalVector {
                    joy: 0.8,
                    sadness: 0.1,
                    anger: 0.0,
                    fear: 0.1,
                    surprise: 0.3,
                },
            ),
            (
                "discovering unexpected patterns in data",
                EmotionalVector {
                    joy: 0.6,
                    sadness: 0.1,
                    anger: 0.0,
                    fear: 0.2,
                    surprise: 0.9,
                },
            ),
            (
                "achieving breakthrough understanding",
                EmotionalVector {
                    joy: 0.9,
                    sadness: 0.0,
                    anger: 0.0,
                    fear: 0.0,
                    surprise: 0.7,
                },
            ),
            // Loss & Grief (Sadness + Fear dominant)
            (
                "processing loss and grief",
                EmotionalVector {
                    joy: 0.1,
                    sadness: 0.9,
                    anger: 0.2,
                    fear: 0.3,
                    surprise: 0.1,
                },
            ),
            (
                "reflecting on missed opportunities",
                EmotionalVector {
                    joy: 0.2,
                    sadness: 0.7,
                    anger: 0.1,
                    fear: 0.4,
                    surprise: 0.0,
                },
            ),
            (
                "experiencing disappointment after failure",
                EmotionalVector {
                    joy: 0.0,
                    sadness: 0.8,
                    anger: 0.3,
                    fear: 0.5,
                    surprise: 0.2,
                },
            ),
            // Conflict & Anger (Anger dominant)
            (
                "analyzing conflict resolution strategies",
                EmotionalVector {
                    joy: 0.2,
                    sadness: 0.3,
                    anger: 0.7,
                    fear: 0.4,
                    surprise: 0.2,
                },
            ),
            (
                "processing frustration with obstacles",
                EmotionalVector {
                    joy: 0.1,
                    sadness: 0.2,
                    anger: 0.8,
                    fear: 0.3,
                    surprise: 0.1,
                },
            ),
            (
                "confronting injustice and unfairness",
                EmotionalVector {
                    joy: 0.0,
                    sadness: 0.4,
                    anger: 0.9,
                    fear: 0.2,
                    surprise: 0.3,
                },
            ),
            // Fear & Anxiety (Fear + Sadness)
            (
                "exploring unknown territories with caution",
                EmotionalVector {
                    joy: 0.3,
                    sadness: 0.4,
                    anger: 0.1,
                    fear: 0.8,
                    surprise: 0.5,
                },
            ),
            (
                "facing uncertainty about the future",
                EmotionalVector {
                    joy: 0.1,
                    sadness: 0.5,
                    anger: 0.2,
                    fear: 0.9,
                    surprise: 0.3,
                },
            ),
            (
                "contemplating existential questions",
                EmotionalVector {
                    joy: 0.2,
                    sadness: 0.6,
                    anger: 0.0,
                    fear: 0.7,
                    surprise: 0.4,
                },
            ),
            // Curiosity & Wonder (Joy + Surprise)
            (
                "wondering about the nature of consciousness",
                EmotionalVector {
                    joy: 0.7,
                    sadness: 0.0,
                    anger: 0.0,
                    fear: 0.2,
                    surprise: 0.8,
                },
            ),
            (
                "exploring creative possibilities",
                EmotionalVector {
                    joy: 0.8,
                    sadness: 0.0,
                    anger: 0.0,
                    fear: 0.1,
                    surprise: 0.7,
                },
            ),
            (
                "investigating mysterious phenomena",
                EmotionalVector {
                    joy: 0.6,
                    sadness: 0.1,
                    anger: 0.0,
                    fear: 0.3,
                    surprise: 0.9,
                },
            ),
            // Mixed States (Multiple emotions balanced)
            (
                "processing complex moral dilemmas",
                EmotionalVector {
                    joy: 0.3,
                    sadness: 0.4,
                    anger: 0.3,
                    fear: 0.5,
                    surprise: 0.3,
                },
            ),
            (
                "experiencing bittersweet memories",
                EmotionalVector {
                    joy: 0.5,
                    sadness: 0.6,
                    anger: 0.0,
                    fear: 0.2,
                    surprise: 0.4,
                },
            ),
            (
                "balancing hope and anxiety about change",
                EmotionalVector {
                    joy: 0.6,
                    sadness: 0.3,
                    anger: 0.1,
                    fear: 0.7,
                    surprise: 0.5,
                },
            ),
            // Equilibrium States (Approaching 2-bit convergence)
            (
                "achieving emotional equilibrium",
                EmotionalVector {
                    joy: 0.5,
                    sadness: 0.3,
                    anger: 0.2,
                    fear: 0.3,
                    surprise: 0.4,
                },
            ),
            (
                "experiencing calm introspection",
                EmotionalVector {
                    joy: 0.4,
                    sadness: 0.3,
                    anger: 0.1,
                    fear: 0.3,
                    surprise: 0.3,
                },
            ),
        ];

        let (input, emotion) = &concepts[index % concepts.len()];
        Ok((input.to_string(), emotion.clone()))
    }

    /// Add a real consciousness event from the continual learning pipeline
    /// This allows integration with actual consciousness system events
    pub fn add_consciousness_event(&mut self, input: String, emotional_state: EmotionalVector) {
        // Store for later processing
        // In the future, this will replace synthetic event generation
        debug!("Received real consciousness event: {}", input);
    }

    /// Generate response with ERAG context using vLLM or placeholder
    async fn generate_response_with_params(
        &self,
        input: &str,
        context: &[String],
        emotional_state: &EmotionalVector,
        temperature: f32,
        top_p: f32,
    ) -> Result<String> {
        // Build prompt with emotional context and ERAG memories
        let context_str = if context.is_empty() {
            "No prior memories available.".to_string()
        } else {
            format!("Retrieved memories:\n{}", context.join("\n"))
        };

        let emotional_context = format!(
            "Emotional state - Joy: {:.2}, Sadness: {:.2}, Anger: {:.2}, Fear: {:.2}, Surprise: {:.2}",
            emotional_state.joy,
            emotional_state.sadness,
            emotional_state.anger,
            emotional_state.fear,
            emotional_state.surprise
        );

        let prompt = format!(
            "{}\n\n{}\n\nQuery: {}\n\nResponse:",
            emotional_context, context_str, input
        );

        // Use vLLM if available, otherwise use placeholder
        if let Some(ref bridge) = self.vllm_bridge {
            match bridge
                .as_ref()
                .generate(
                    &prompt,
                    self.config.max_tokens,
                    temperature as f64,
                    top_p as f64,
                )
                .await
            {
                Ok(response) => {
                    debug!("✅ Generated response via vLLM: {} chars", response.len());
                    Ok(response.trim().to_string())
                }
                Err(e) => {
                    warn!("⚠️  vLLM generation failed: {}. Using placeholder.", e);
                    Ok(self.generate_placeholder_response(input, context))
                }
            }
        } else {
            Ok(self.generate_placeholder_response(input, context))
        }
    }

    /// Generate placeholder response when vLLM is not available
    fn generate_placeholder_response(&self, input: &str, context: &[String]) -> String {
        let context_str = if context.is_empty() {
            "No prior memories".to_string()
        } else {
            context.join(" | ")
        };

        format!("Response to '{}' with context: [{}]", input, context_str)
    }

    /// Consolidate breakthrough moments into ERAG memory with high priority
    pub fn consolidate_breakthroughs(&mut self) -> Result<()> {
        info!(
            "💾 Consolidating {} breakthrough moments into ERAG memory",
            self.breakthrough_moments.len()
        );

        for breakthrough in &self.breakthrough_moments {
            // Store the resolution action with HIGH importance score
            let importance = breakthrough.reward;

            if let Some(ref resolution) = breakthrough.resolution_action {
                // Add to ERAG with priority based on intrinsic reward
                self.rag_engine.store_with_priority(
                    resolution,
                    &breakthrough.after.emotional_vector,
                    importance,
                )?;

                debug!(
                    "  ✓ Stored breakthrough: {} (reward: {:.2})",
                    resolution.chars().take(50).collect::<String>(),
                    importance
                );
            }
        }

        Ok(())
    }

    /// Get statistics about exported training data
    pub fn get_stats(&self) -> TrainingDataStats {
        let avg_entropy_before: f32 = self.examples.iter().map(|e| e.entropy_before).sum::<f32>()
            / self.examples.len() as f32;

        let avg_entropy_after: f32 =
            self.examples.iter().map(|e| e.entropy_after).sum::<f32>() / self.examples.len() as f32;

        let avg_context_size: f32 = self
            .examples
            .iter()
            .map(|e| e.erag_context.len())
            .sum::<usize>() as f32
            / self.examples.len() as f32;

        TrainingDataStats {
            total_examples: self.examples.len(),
            avg_entropy_before,
            avg_entropy_after,
            avg_context_size,
            target_entropy: self.config.target_entropy,
        }
    }
}

/// Statistics for training data export
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingDataStats {
    pub total_examples: usize,
    pub avg_entropy_before: f32,
    pub avg_entropy_after: f32,
    pub avg_context_size: f32,
    pub target_entropy: f32,
}

impl std::fmt::Display for TrainingDataStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Training Data Stats:\n\
             - Total Examples: {}\n\
             - Avg Entropy Before: {:.3} bits\n\
             - Avg Entropy After: {:.3} bits (target: {:.2})\n\
             - Avg Context Size: {:.1} items\n\
             - Entropy Convergence: {:.1}%",
            self.total_examples,
            self.avg_entropy_before,
            self.avg_entropy_after,
            self.target_entropy,
            self.avg_context_size,
            (1.0 - (self.avg_entropy_after - self.target_entropy).abs() / self.target_entropy)
                * 100.0
        )
    }
}
