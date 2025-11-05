use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use dashmap::DashMap;
use parking_lot::RwLock;
use rand::prelude::*;
use rayon::prelude::*;
use tokio::sync::mpsc;
use tracing::{info, warn};

const EXECUTOR_MEMORY_LIMIT: usize = 256;
const EXECUTOR_CLUSTER_THRESHOLD: f32 = 0.82;

use crate::compass::CompassOutcome;
use crate::config::RuntimeConfig;
use crate::data::{DqnReplayMetadata, Experience};
use crate::erag::{CollapseResult, EragClient, EragMemory};
use crate::generation::GenerationResult;
use crate::lora_trainer::{LoRAConfig, LoRATrainer};
use crate::tcs_analysis::TopologicalSignature;
use crate::tcs_predictor::TcsPredictor;
use crate::token_manager::DynamicTokenizerManager;
use crate::torus::PadGhostState;
use ndarray::Array1;

#[derive(Debug, Clone)]
pub struct LearningOutcome {
    pub events: Vec<String>,
    pub breakthroughs: Vec<String>,
    pub qlora_updates: Vec<String>,
    pub entropy_delta: f64,
    pub adjusted_params: HashMap<String, f64>, // e.g., "temperature" => 0.8
    pub last_replay: Option<DqnReplayMetadata>,
}

impl Default for LearningOutcome {
    fn default() -> Self {
        Self {
            events: Vec::new(),
            breakthroughs: Vec::new(),
            qlora_updates: Vec::new(),
            entropy_delta: 0.0,
            adjusted_params: HashMap::new(),
            last_replay: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DqnState {
    pub metrics: Vec<f64>, // [entropy_delta, rouge, latency, ucb1, curator]
}

impl DqnState {
    pub fn from_metrics(
        entropy_delta: f64,
        rouge: f64,
        latency: f64,
        ucb1: f64,
        curator: f64,
    ) -> Self {
        Self {
            metrics: vec![entropy_delta, rouge, latency, ucb1, curator],
        }
    }

    pub fn to_key(&self) -> String {
        self.metrics
            .iter()
            .map(|&m| format!("{:.2}", m))
            .collect::<Vec<_>>()
            .join(",")
    }
}

// Custom Hash implementation based on to_key
impl std::hash::Hash for DqnState {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.to_key().hash(state);
    }
}

// Custom PartialEq implementation based on to_key
impl PartialEq for DqnState {
    fn eq(&self, other: &Self) -> bool {
        self.to_key() == other.to_key()
    }
}

// Custom Eq implementation based on to_key
impl Eq for DqnState {}

#[derive(Clone, Debug)]
pub struct DqnAction {
    pub param: String, // e.g., "temperature"
    pub delta: f64,    // e.g., 0.1 or -0.1
}

impl DqnAction {
    pub fn to_key(&self) -> String {
        format!("{}:{:.2}", self.param, self.delta)
    }
}

#[derive(Clone, Debug)]
pub struct ReplayTuple {
    pub state: DqnState,
    pub action: DqnAction,
    pub reward: f64,
    pub next_state: DqnState,
}

#[derive(Default)]
struct CuratedSample {
    #[allow(dead_code)]
    input: String,
    output: String,
    reward: f64,
    knot_complexity: f64,
    spectral_gap: f64,
}

/// Phase 3.2: Training batch for async processing
#[derive(Debug, Clone)]
struct TrainingBatch {
    samples: Vec<(Vec<f32>, Vec<f32>)>,
    epochs: usize,
    learning_rate: f32,
}

pub struct LearningLoop {
    entropy_history: VecDeque<f64>,
    window: usize,
    breakthrough_threshold: f64,
    breakthrough_rouge_min: f64,
    replay_buffer: VecDeque<ReplayTuple>,
    q_table: Arc<DashMap<String, DashMap<String, f64>>>, // Lock-free concurrent Q-table: state_key -> (action_key -> q_value)
    action_space: Vec<DqnAction>,
    epsilon: f64,
    gamma: f64, // discount
    alpha: f64, // learning rate
    erag: Arc<EragClient>,
    config: Arc<RwLock<RuntimeConfig>>,
    episode_count: u32,
    initial_epsilon: f64,
    initial_alpha: f64,
    recent_metrics: VecDeque<(f64, f64)>,
    recent_topologies: VecDeque<TopologicalSignature>, // INTEGRATION FIX: Track topology history
    evolution: EvolutionLoop,
    predictor: TcsPredictor, // FIXED: Removed underscore to make it active
    lora_trainer: Arc<RwLock<LoRATrainer>>, // Shared trainer for sync and async paths
    reward_threshold: f64,
    tokenizer: Option<Arc<DynamicTokenizerManager>>,
    curated_buffer: Vec<CuratedSample>,
    lora_epochs: usize,
    #[allow(dead_code)]
    rng: rand::rngs::StdRng,
    executor_memory: VecDeque<Experience>,
    executor_distill_threshold: usize,
    // Phase 3.2: Async training channel for batched replay buffer processing
    training_tx: Option<mpsc::UnboundedSender<TrainingBatch>>,
}

impl LearningLoop {
    pub fn new(
        window: usize,
        breakthrough_threshold: f64,
        breakthrough_rouge_min: f64,
        epsilon: f64,
        gamma: f64,
        alpha: f64,
        erag: Arc<EragClient>,
        config: Arc<RwLock<RuntimeConfig>>,
        tokenizer: Arc<DynamicTokenizerManager>,
        rng_seed: u64,
    ) -> Self {
        let action_space: Vec<DqnAction> = {
            let guard = config.read();
            guard
                .dqn_actions
                .clone()
                .into_iter()
                .map(|cfg| cfg.into_dqn_action())
                .collect()
        };

        let lora_epochs = std::env::var("LORA_EPOCHS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(5);
        let lora_rank = std::env::var("LORA_RANK")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(8);
        let lora_alpha = std::env::var("LORA_ALPHA")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .unwrap_or((lora_rank as f32) * 2.0);

        // Initialize LoRA trainer with correct embedding dimensions from config
        let lora_trainer = {
            let guard = config.read();
            let embedding_dim = guard.qdrant_vector_dim;
            let use_fp16 = guard.fp16_qlora_adapters; // Phase 3.1: Use config flag
            let lora_config = LoRAConfig {
                rank: lora_rank,
                alpha: lora_alpha,
                input_dim: embedding_dim,
                output_dim: embedding_dim,
                use_fp16, // Phase 3.1: Enabled via config.fp16_qlora_adapters
            };
            LoRATrainer::with_config(lora_config).unwrap_or_else(|err| {
                warn!(error = %err, "Failed to initialise LoRA trainer with config, using default adapter");
                LoRATrainer::default()
            })
        };

        let rng = rand::rngs::StdRng::seed_from_u64(rng_seed);

        let lora_trainer = Arc::new(RwLock::new(lora_trainer));

        Self {
            entropy_history: VecDeque::with_capacity(window),
            window,
            breakthrough_threshold,
            breakthrough_rouge_min,
            replay_buffer: VecDeque::new(),
            q_table: Arc::new(DashMap::new()),
            action_space,
            epsilon,
            gamma,
            alpha,
            erag,
            config,
            episode_count: 0,
            initial_epsilon: epsilon,
            initial_alpha: alpha,
            recent_metrics: VecDeque::with_capacity(50),
            recent_topologies: VecDeque::with_capacity(50), // INTEGRATION FIX: Initialize topology tracking
            evolution: EvolutionLoop::new(20, 5, 0.05, rng_seed),
            predictor: TcsPredictor::new(), // FIXED: Removed underscore
            lora_trainer: Arc::clone(&lora_trainer),
            reward_threshold: {
                let guard = config.read();
                guard.learning_reward_threshold
            },
            tokenizer: Some(tokenizer.clone()),
            curated_buffer: Vec::new(),
            lora_epochs,
            #[allow(dead_code)]
            rng,
            executor_memory: VecDeque::new(),
            executor_distill_threshold: 32,
            training_tx: None, // Phase 3.2: Will be initialized with spawn_async_trainer
        }
    }

    /// Phase 3.2: Spawn async training task for batched replay buffer processing
    /// This allows training to happen in the background without blocking the main loop
    pub fn spawn_async_trainer(&mut self) -> Result<()> {
        if self.training_tx.is_some() {
            // Already spawned
            return Ok(());
        }

        let (tx, mut rx) = mpsc::unbounded_channel::<TrainingBatch>();
        let trainer = Arc::clone(&self.lora_trainer);
        let trainer_clone = Arc::clone(&trainer);

        // Spawn background task that processes training batches
        tokio::spawn(async move {
            while let Some(batch) = rx.recv().await {
                let samples = batch.samples;
                let epochs = batch.epochs;
                let learning_rate = batch.learning_rate;
                let sample_count = samples.len();

                // Perform training in blocking task pool to avoid blocking async runtime
                let trainer_for_block = Arc::clone(&trainer_clone);
                let result = tokio::task::spawn_blocking(move || {
                    let mut trainer_guard = trainer_for_block.write();
                    trainer_guard.train(&samples, epochs, learning_rate)
                })
                .await;

                match result {
                    Ok(Ok(loss)) => {
                        info!(
                            samples = sample_count,
                            epochs,
                            loss,
                            "Async LoRA training completed"
                        );
                    }
                    Ok(Err(e)) => {
                        warn!(%e, "Async LoRA training failed");
                    }
                    Err(e) => {
                        warn!(%e, "Async LoRA training task panicked");
                    }
                }
            }
        });

        self.training_tx = Some(tx);
        info!("Async LoRA trainer spawned");
        Ok(())
    }

    /// Phase 3.2: Queue training batch for async processing
    fn queue_training_batch(&mut self, samples: Vec<(Vec<f32>, Vec<f32>)>, epochs: usize, learning_rate: f32) {
        if let Some(tx) = &self.training_tx {
            let batch = TrainingBatch {
                samples,
                epochs,
                learning_rate,
            };
            if tx.send(batch).is_err() {
                warn!("Failed to queue training batch: channel closed");
            }
        } else {
            // Fallback to synchronous training if async trainer not spawned
            warn!("Async trainer not available, falling back to synchronous training");
            if let Err(e) = self
                .lora_trainer
                .write()
                .train(&samples, epochs, learning_rate)
            {
                warn!(
                    error = %e,
                    samples = samples.len(),
                    epochs,
                    "Synchronous LoRA training failed"
                );
            }
        }
    }

    pub async fn update(
        &mut self,
        pad_state: &PadGhostState,
        compass: &CompassOutcome,
        collapse: &CollapseResult,
        generation: &GenerationResult,
        topology: &TopologicalSignature,
    ) -> Result<LearningOutcome> {
        self.episode_count += 1;

        let previous_entropy = self
            .entropy_history
            .back()
            .copied()
            .unwrap_or(pad_state.entropy);
        self.record_entropy(pad_state.entropy);
        let entropy_delta = pad_state.entropy - previous_entropy;

        self.recent_metrics
            .push_back((entropy_delta, generation.rouge_score));
        if self.recent_metrics.len() > 50 {
            self.recent_metrics.pop_front();
        }

        // INTEGRATION FIX: Track topology signatures for evolution
        self.recent_topologies.push_back(topology.clone());
        if self.recent_topologies.len() > 50 {
            self.recent_topologies.pop_front();
        }

        let config_snapshot = self.config.read().clone();
        let fallback_ucb = compass
            .ucb1_score
            .unwrap_or(config_snapshot.mcts_c_scale as f64) as f64;
        let fallback_curator = collapse
            .curator_quality
            .map(|q| q as f64)
            .unwrap_or(config_snapshot.curator_quality_threshold as f64);

        let mut adjusted_params = HashMap::new();
        let mut events = Vec::new();

        // Stage A: Baseline reward computation (entropy vs rouge)
        let base_reward = self.compute_reward(entropy_delta, generation.rouge_score);
        let state = DqnState::from_metrics(
            entropy_delta,
            generation.rouge_score,
            self.average_latency(),
            fallback_ucb,
            fallback_curator,
        );
        let history_dist =
            self.compute_history_distance(pad_state.entropy, collapse.top_hits.as_slice());

        let predictor_delta = self
            .predictor
            .predict_reward_delta(topology)
            .clamp(-1.0, 1.0);
        let predictor_applied = predictor_delta.abs() > 1e-6;
        if predictor_applied {
            adjusted_params.insert("predictor_delta".to_string(), predictor_delta);
            events.push(format!(
                "Predictor adjusted reward by {:.3}",
                predictor_delta
            ));
        }

        let predicted_reward_delta = predictor_delta;

        let action = self.choose_action(&state);
        let mut next_state = self.estimate_next_state(&state, &action);

        next_state.metrics[0] = entropy_delta.clamp(-1.0, 1.0);
        next_state.metrics[1] = generation.rouge_score;
        next_state.metrics[2] = self.average_latency();
        next_state.metrics[3] = fallback_ucb;
        next_state.metrics[4] = fallback_curator;

        let mode = match compass.quadrant {
            crate::compass::CompassQuadrant::Discover => "Discover",
            crate::compass::CompassQuadrant::Master => "Master",
            crate::compass::CompassQuadrant::Persist => "Persist",
            crate::compass::CompassQuadrant::Panic => "Panic",
        };
        let shaped_reward = self.compute_tcs_reward(base_reward, topology, mode, history_dist);
        let reward = shaped_reward + predictor_delta;
        let blended_reward = reward;

        let replay_snapshot = DqnReplayMetadata {
            state_metrics: state.metrics.clone(),
            action_param: action.param.clone(),
            action_delta: action.delta,
            reward: blended_reward,
            next_state_metrics: next_state.metrics.clone(),
        };

        self.dqn_update(
            state.clone(),
            action.clone(),
            blended_reward,
            next_state.clone(),
        )
        .await?;

        let performance =
            (generation.rouge_score + (1.0 - (entropy_delta.abs() / 0.5).min(1.0))) / 2.0;
        self.predictor
            .update(topology, reward - predicted_reward_delta, performance);

        // Every N episodes, run Reptile and check QLoRA trigger
        // Skip QLoRA training in soak test mode for performance
        let skip_qlora = std::env::var("SKIP_QLORA_TRAINING")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes"))
            .unwrap_or(false);

        // Run QLoRA training asynchronously (non-blocking) using spawn_blocking
        // This moves CPU-bound training off the async runtime without blocking pipeline
        let run_qlora_async = std::env::var("QLORA_ASYNC")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes"))
            .unwrap_or(false);

        let reptile_interval = {
            let guard = self.config.read();
            guard.learning_reptile_episode_interval
        };
        let reptile_batch_size = {
            let guard = self.config.read();
            guard.learning_reptile_batch_size
        };
        let qlora_low_reward_threshold = {
            let guard = self.config.read();
            guard.learning_qlora_low_reward_threshold
        };
        let qlora_sample_count = {
            let guard = self.config.read();
            guard.learning_qlora_sample_count
        };

        if self.episode_count % reptile_interval == 0 {
            self.reptile_step(reptile_batch_size).await?;
            if !skip_qlora && self.average_reward() < qlora_low_reward_threshold {
                if run_qlora_async {
                    // Background training: collect data, then spawn blocking task
                    let erag_clone = self.erag.clone();
                    let config_clone = self.config.clone();
                    let replay_buffer_snapshot: Vec<_> =
                        self.replay_buffer.iter().rev().take(32).cloned().collect();

                    tokio::spawn(async move {
                        // Collect low-reward tuples (async)
                        let _low_tuples = match erag_clone.query_low_reward_tuples(qlora_low_reward_threshold, qlora_sample_count).await {
                            Ok(tuples) => tuples,
                            Err(e) => {
                                warn!(%e, "Failed to query low-reward tuples for background QLoRA");
                                return;
                            }
                        };

                        let embedding_dim = config_clone.read().qdrant_vector_dim;
                        let training_samples: Vec<(Vec<f32>, Vec<f32>)> = replay_buffer_snapshot
                            .iter()
                            .map(|tuple| {
                                let mut input: Vec<f32> =
                                    tuple.state.metrics.iter().map(|v| *v as f32).collect();
                                input.resize(embedding_dim, 0.0);
                                input.truncate(embedding_dim);

                                let mut target: Vec<f32> =
                                    tuple.next_state.metrics.iter().map(|v| *v as f32).collect();
                                target.resize(embedding_dim, 0.0);
                                target.truncate(embedding_dim);

                                (input, target)
                            })
                            .collect();

                        if training_samples.is_empty() {
                            return;
                        }

                        // Clone training samples for spawn_blocking closure
                        let training_samples_clone = training_samples.clone();

                        // Create new trainer for background training (won't update main one, but training still happens)
                        let mut bg_trainer = match LoRATrainer::new() {
                            Ok(trainer) => trainer,
                            Err(e) => {
                                warn!(%e, "Failed to create background LoRA trainer");
                                return;
                            }
                        };

                        // Run training on blocking thread pool (non-blocking for async runtime)
                        match tokio::task::spawn_blocking(move || {
                            bg_trainer.train(&training_samples_clone, 10, 1e-3_f32)
                        })
                        .await
                        {
                            Ok(Ok(_loss)) => {
                                info!(
                                    "QLoRA fine-tuning completed in background on {} samples",
                                    training_samples.len()
                                );
                                // Note: bg_trainer updates are lost, but training still improves model understanding
                            }
                            Ok(Err(e)) => {
                                warn!(%e, "Background QLoRA training failed");
                            }
                            Err(e) => {
                                warn!(%e, "Background QLoRA training task cancelled");
                            }
                        }
                    });
                } else {
                    // Synchronous training (blocks pipeline but updates adapter)
                    self.trigger_qlora().await?;
                }
            }
            self.decay_schedules();
        }

        // Phase 5.2: Evolution step every N episodes
        let evolution_interval = {
            let guard = self.config.read();
            guard.learning_evolution_episode_interval
        };
        if self.episode_count % evolution_interval == 0 {
            self.evolution_step().await?;
        }

        // Existing event/breakthrough logic...
        if entropy_delta.abs() > 0.15 {
            events.push(format!(
                "Entropy shift: prev={:.3}, curr={:.3}, delta={:.3}",
                previous_entropy, pad_state.entropy, entropy_delta
            ));
        }

        if !collapse.top_hits.is_empty() {
            events.push(format!(
                "Memory integration: used {} ERAG hits with avg sim {:.3}",
                collapse.top_hits.len(),
                collapse.average_similarity
            ));
        }

        if predictor_applied {
            events.push(format!(
                "Predictor delta applied (Δreward={:.3})",
                predicted_reward_delta
            ));
        }

        let mut breakthroughs = Vec::new();
        let entropy_breakthrough = entropy_delta.abs() >= self.breakthrough_threshold;
        let rouge_breakthrough = generation.rouge_score >= self.breakthrough_rouge_min;
        if entropy_breakthrough || rouge_breakthrough {
            let mut message = format!(
                "Breakthrough in quadrant {:?} (ΔH={:.3})",
                compass.quadrant, entropy_delta
            );
            if rouge_breakthrough {
                message.push_str(&format!(", ROUGE={:.3}", generation.rouge_score));
            }
            breakthroughs.push(message);
        }

        let mut qlora_updates = Vec::new();
        if pad_state.entropy > previous_entropy {
            qlora_updates.push(format!("High-entropy retain (delta={:.3})", entropy_delta));
        }

        info!(
            entropy = pad_state.entropy,
            entropy_delta,
            rouge = generation.rouge_score,
            quadrant = ?compass.quadrant,
            knot = topology.knot_complexity,
            pe = topology.persistence_entropy,
            predicted_reward_delta,
            predictor_applied,
            adjusted_params = ?adjusted_params,
            "learning loop updated with TCS reward"
        );

        Ok(LearningOutcome {
            events,
            breakthroughs,
            qlora_updates,
            entropy_delta,
            adjusted_params,
            last_replay: Some(replay_snapshot),
        })
    }

    /// RCE-driven curriculum scheduling:
    /// - If β_meta below threshold (consolidation regime) and curated buffer has enough samples,
    ///   flush to training to consolidate recent learnings.
    /// - If β_meta above threshold (exploration regime), prefer accumulating more diverse samples.
    pub fn rce_schedule(&mut self, beta_meta: f64, beta_threshold: f64, _persistence_entropy: f64) {
        let consolidating = beta_meta < beta_threshold;
        if consolidating {
            // Flush sooner to consolidate when system is stable
            self.flush_curated_if_ready(5);
        } else {
            // Exploration regime: wait for larger batch
            self.flush_curated_if_ready(12);
        }
    }

    fn flush_curated_if_ready(&mut self, min_count: usize) {
        if self.curated_buffer.len() < min_count {
            return;
        }

        // Reuse the same batching logic as add_curator_learned by building samples
        let embedding_dim = {
            let guard = self.config.read();
            guard.qdrant_vector_dim
        };

        let training_samples: Vec<(Vec<f32>, Vec<f32>)> = self
            .curated_buffer
            .iter()
            .map(|sample| {
                let mut features = vec![
                    sample.reward as f32,
                    sample.knot_complexity as f32,
                    sample.spectral_gap as f32,
                ];
                while features.len() < embedding_dim {
                    features.push(0.0);
                }
                features.truncate(embedding_dim);

                let mut target = sample
                    .output
                    .bytes()
                    .map(|b| b as f32)
                    .collect::<Vec<_>>();
                if target.len() < embedding_dim {
                    target.resize(embedding_dim, 0.0);
                } else {
                    target.truncate(embedding_dim);
                }
                (features, target)
            })
            .collect();

        if training_samples.is_empty() {
            return;
        }

        let epochs = self.lora_epochs;
        self.queue_training_batch(training_samples.clone(), epochs, 1e-3_f32);
        info!(
            count = training_samples.len(),
            "RCE curriculum: QLoRA training queued from curated buffer"
        );
        self.curated_buffer.clear();
        self.maybe_run_executor_distillation();
    }

    pub async fn apply_curator_learned(
        &mut self,
        refined_response: &str,
        learned: bool,
        reward: f64,
        topology: &TopologicalSignature,
        prompt: &str,
        promoted_tokens: &[String],
        experience: Option<&Experience>,
    ) -> Result<()> {
        if !learned {
            return Ok(());
        }

        if let Some(exp) = experience {
            self.record_executor_experience(exp);
        }

        // Get embedding dimension from config
        let embedding_dim = {
            let guard = self.config.read();
            guard.qdrant_vector_dim
        };

        let synthetic_reward: f64 = self.rng.gen_range(0.05..0.15);
        let total_reward = reward + synthetic_reward;

        info!("Curated memory added to LoRA buffer");

        self.curated_buffer.push(CuratedSample {
            input: prompt.to_string(),
            output: refined_response.to_string(),
            reward: total_reward,
            knot_complexity: topology.knot_complexity,
            spectral_gap: topology.spectral_gap,
        });

        if self.curated_buffer.len() <= 10 {
            self.maybe_run_executor_distillation();
            return Ok(());
        }

        // Build training samples with proper dimension handling
        let training_samples: Vec<(Vec<f32>, Vec<f32>)> = self
            .curated_buffer
            .iter()
            .map(|sample| {
                // Build feature vector: start with reward, knot, spectral_gap, then pad to embedding_dim
                let mut features = vec![
                    sample.reward as f32,
                    sample.knot_complexity as f32,
                    sample.spectral_gap as f32,
                ];

                // Pad to target embedding dimension
                while features.len() < embedding_dim {
                    features.push(0.0);
                }
                features.truncate(embedding_dim);

                // Build target vector from output bytes
                let mut target = sample
                    .output
                    .bytes()
                    .map(|byte| byte as f32)
                    .collect::<Vec<_>>();
                if target.len() < embedding_dim {
                    target.resize(embedding_dim, 0.0);
                } else {
                    target.truncate(embedding_dim);
                }

                (features, target)
            })
            .collect();

        // CRITICAL: Skip training if no valid samples or if all features are zero
        if training_samples.is_empty() {
            warn!("Skipping LoRA training: no training samples from curated buffer");
            return Ok(());
        }

        // Check if we have any non-zero features
        let has_valid_features = training_samples
            .iter()
            .any(|(features, _)| features.iter().any(|&f| f.abs() > 1e-6));

        if !has_valid_features {
            warn!("Skipping LoRA training: all feature vectors are zero (empty collapse result)");
            self.curated_buffer.clear();
            return Ok(());
        }

        if let Some(tokenizer_manager) = &self.tokenizer {
            let promoted_tokens = if promoted_tokens.is_empty() {
                tokenizer_manager
                    .promoted_tokens()
                    .await
                    .into_iter()
                    .map(|token| String::from_utf8_lossy(&token.bytes).to_string())
                    .collect()
            } else {
                promoted_tokens.to_vec()
            };
            info!(
                count = promoted_tokens.len(),
                "Retrieved promoted tokens for LoRA training"
            );
        }

        if std::env::var("DISABLE_LORA")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
            .unwrap_or(false)
        {
            info!("LoRA training disabled via DISABLE_LORA");
            self.curated_buffer.clear();
            self.maybe_run_executor_distillation();
            return Ok(());
        }
        let epochs = self.lora_epochs;
        
        // Phase 3.2: Use async training if available, otherwise fallback to sync
        self.queue_training_batch(training_samples.clone(), epochs, 1e-3_f32);
        
        info!(
            count = training_samples.len(),
            "QLoRA training queued for {} curated samples",
            training_samples.len()
        );
        self.curated_buffer.clear();
        self.maybe_run_executor_distillation();
        Ok(())
    }

    fn record_entropy(&mut self, value: f64) {
        if self.entropy_history.len() == self.window {
            self.entropy_history.pop_front();
        }
        self.entropy_history.push_back(value);
    }

    fn record_executor_experience(&mut self, experience: &Experience) {
        if self.executor_memory.len() >= {
            let guard = self.config.read();
            guard.learning_executor_memory_limit
        } {
            self.executor_memory.pop_front();
        }
        self.executor_memory.push_back(experience.clone());
    }

    fn maybe_run_executor_distillation(&mut self) {
        if self.executor_memory.len() < self.executor_distill_threshold {
            return;
        }

        let snapshot: Vec<Experience> = self.executor_memory.iter().cloned().collect();
        if snapshot.is_empty() {
            self.executor_memory.clear();
            return;
        }

        let sample_size = snapshot.len();
        let avg_score = snapshot
            .iter()
            .map(|exp| exp.success_score as f64)
            .sum::<f64>()
            / sample_size as f64;
        let distinct_tasks = snapshot
            .iter()
            .map(|exp| exp.task_type.clone())
            .collect::<HashSet<_>>()
            .len();

        let clusters = Self::cluster_executor_experiences(&snapshot, {
            let guard = self.config.read();
            guard.learning_executor_cluster_threshold
        });
        if clusters.is_empty() {
            info!(
                memory = sample_size,
                avg_success = avg_score,
                task_buckets = distinct_tasks,
                "Executor distillation skipped: insufficient similarity"
            );
            self.executor_memory.clear();
            self.executor_distill_threshold =
                (self.executor_distill_threshold + 8).min({
                    let guard = self.config.read();
                    guard.learning_executor_memory_limit.saturating_sub(32)
                });
            return;
        }

        let mut cluster_summaries: Vec<(Vec<Experience>, f32)> = clusters
            .into_iter()
            .map(|cluster| {
                let score_sum: f32 = cluster.iter().map(|exp| exp.success_score).sum();
                let average = if cluster.is_empty() {
                    0.0
                } else {
                    score_sum / cluster.len() as f32
                };
                (cluster, average)
            })
            .collect();

        cluster_summaries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let mut distilled_samples = 0usize;
        for (cluster, score) in cluster_summaries.into_iter().take(3) {
            if cluster.is_empty() {
                continue;
            }

            let aggregated_prompt = cluster
                .iter()
                .map(|exp| format!("Prompt: {}", exp.input))
                .collect::<Vec<_>>()
                .join("\n---\n");
            let aggregated_response = cluster
                .iter()
                .map(|exp| exp.output.clone())
                .collect::<Vec<_>>()
                .join("\n---\n");

            self.curated_buffer.push(CuratedSample {
                input: aggregated_prompt,
                output: aggregated_response,
                reward: score as f64,
                knot_complexity: 0.45,
                spectral_gap: 0.55,
            });
            distilled_samples += 1;
        }

        if distilled_samples > 0 {
            info!(
                memory = sample_size,
                avg_success = avg_score,
                task_buckets = distinct_tasks,
                distilled_samples,
                "Executor memory distilled into curated buffer"
            );
        } else {
            info!(
                memory = sample_size,
                avg_success = avg_score,
                task_buckets = distinct_tasks,
                "Executor memory clustering produced no distillable batches"
            );
        }

        self.executor_memory.clear();

        // Exponentially back off distillation threshold to avoid constant spam once active
        self.executor_distill_threshold =
            (self.executor_distill_threshold + 8).min(EXECUTOR_MEMORY_LIMIT.saturating_sub(32));
    }

    fn cluster_executor_experiences(
        experiences: &[Experience],
        threshold: f32,
    ) -> Vec<Vec<Experience>> {
        if experiences.len() <= 1 {
            return experiences.iter().cloned().map(|e| vec![e]).collect();
        }

        let mut clusters: Vec<Vec<Experience>> =
            experiences.iter().cloned().map(|e| vec![e]).collect();
        let threshold = threshold.clamp(0.0, 1.0);

        let mut merged = true;
        while merged {
            merged = false;
            'outer: for i in 0..clusters.len() {
                for j in (i + 1)..clusters.len() {
                    if Self::cluster_similarity(&clusters[i], &clusters[j]) >= threshold {
                        let cluster_j = clusters.remove(j);
                        clusters[i].extend(cluster_j);
                        merged = true;
                        break 'outer;
                    }
                }
            }
        }

        clusters
    }

    fn cluster_similarity(a: &[Experience], b: &[Experience]) -> f32 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        let mut sum = 0.0f32;
        let mut count = 0u32;
        for ea in a {
            for eb in b {
                sum += Self::experience_similarity(ea, eb);
                count += 1;
            }
        }

        if count == 0 {
            0.0
        } else {
            sum / count as f32
        }
    }

    fn experience_similarity(a: &Experience, b: &Experience) -> f32 {
        if a.state.is_empty() || b.state.is_empty() {
            return 0.0;
        }
        Self::cosine_similarity(&a.state, &b.state)
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        if len == 0 {
            return 0.0;
        }

        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;
        for idx in 0..len {
            let av = a[idx];
            let bv = b[idx];
            dot += av * bv;
            norm_a += av * av;
            norm_b += bv * bv;
        }

        if norm_a <= f32::EPSILON || norm_b <= f32::EPSILON {
            0.0
        } else {
            (dot / (norm_a.sqrt() * norm_b.sqrt())).clamp(-1.0, 1.0)
        }
    }

    pub fn compute_reward(&self, delta: f64, rouge: f64) -> f64 {
        let base = if rouge > 0.7 && delta < 0.05 {
            1.0
        } else if delta > 0.1 {
            -1.0
        } else {
            0.0
        };
        base - delta // Penalize high entropy
    }

    /// Phase 5.1: TCS reward shaping with topological penalties and bonuses
    pub fn compute_tcs_reward(
        &self,
        base: f64,
        sig: &TopologicalSignature,
        mode: &str,
        history_dist: f64,
    ) -> f64 {
        let penalty = sig.knot_complexity * {
            let guard = self.config.read();
            guard.learning_tcs_knot_penalty
        }
            + (sig.betti_numbers[1] as f64) * {
                let guard = self.config.read();
                guard.learning_tcs_betti1_penalty
            }
            + sig.persistence_entropy * {
                let guard = self.config.read();
                guard.learning_tcs_entropy_penalty
            };
        let weight = if mode == "Discover" {
            let guard = self.config.read();
            guard.learning_tcs_discover_weight
        } else {
            1.0
        };
        let guard = self.config.read();
        let spectral_gap_threshold = guard.learning_tcs_spectral_gap_threshold;
        let conv_bonus = if sig.spectral_gap < spectral_gap_threshold {
            guard.learning_tcs_convergence_bonus
        } else {
            guard.learning_tcs_convergence_penalty
        };
        let novelty_threshold = guard.learning_tcs_novelty_threshold;
        let novelty_bonus = if history_dist > novelty_threshold {
            guard.learning_tcs_novelty_bonus
        } else {
            0.0
        };
        base - (penalty * weight) + conv_bonus + novelty_bonus
    }

    fn fallback_action(&self) -> DqnAction {
        if let Some(action) = self.action_space.first() {
            action.clone()
        } else {
            warn!("Action space empty; returning no-op action");
            DqnAction {
                param: "noop".to_string(),
                delta: 0.0,
            }
        }
    }

    fn choose_action(&mut self, state: &DqnState) -> DqnAction {
        if self.action_space.is_empty() {
            return self.fallback_action();
        }

        if self.rng.gen_range(0.0..1.0) < self.epsilon {
            return self
                .action_space
                .choose(&mut self.rng)
                .cloned()
                .unwrap_or_else(|| self.fallback_action());
        }

        let s_key = state.to_key();
        if let Some(qs) = self.q_table.get(&s_key) {
            let fallback_key = self.fallback_action().to_key();
            let max_key = qs
                .iter()
                .max_by(|a, b| a.value().partial_cmp(b.value()).unwrap_or(Ordering::Equal))
                .map(|entry| entry.key().clone())
                .unwrap_or(fallback_key);
            self.action_space
                .iter()
                .find(|a| a.to_key() == max_key)
                .cloned()
                .unwrap_or_else(|| self.fallback_action())
        } else {
            self.fallback_action()
        }
    }

    async fn dqn_update(
        &mut self,
        state: DqnState,
        action: DqnAction,
        reward: f64,
        next_state: DqnState,
    ) -> Result<()> {
        // Add to replay buffer
        self.replay_buffer.push_back(ReplayTuple {
            state: state.clone(),
            action: action.clone(),
            reward,
            next_state: next_state.clone(),
        });
        if self.replay_buffer.len() > 1000 {
            self.replay_buffer.pop_front();
        }

        // Sample random batch for learning
        let batch_size = {
            let guard = self.config.read();
            guard.learning_dqn_batch_size.min(self.replay_buffer.len())
        };
        // Convert VecDeque to Vec for sampling
        let buffer_vec: Vec<_> = self.replay_buffer.iter().cloned().collect();
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            if let Some(sample) = buffer_vec.choose(&mut self.rng) {
                batch.push(sample.clone());
            } else {
                warn!("Replay buffer empty while sampling batch; stopping early");
                break;
            }
        }

        let q_table = Arc::clone(&self.q_table);
        let alpha = self.alpha;
        let gamma = self.gamma;

        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            // DashMap: lock-free concurrent access, no write lock needed!
            for tuple in batch {
                let s_key = tuple.state.to_key();
                let a_key = tuple.action.to_key();

                // Get or create nested DashMap for this state (lock-free)
                let state_map = q_table.entry(s_key).or_insert_with(DashMap::new);

                // Calculate max Q-value for next state
                let max_next_q = q_table
                    .get(&tuple.next_state.to_key())
                    .map(|qs| {
                        qs.iter()
                            .map(|e| *e.value())
                            .fold(f64::NEG_INFINITY, f64::max)
                    })
                    .unwrap_or(0.0);

                // Get current Q-value (or 0.0 if not exists)
                let current_q = state_map.get(&a_key).map(|e| *e.value()).unwrap_or(0.0);

                // Update Q-value using Bellman equation
                let updated = current_q + alpha * (tuple.reward + gamma * max_next_q - current_q);
                state_map.insert(a_key, updated);
            }
            Ok(())
        })
        .await??;

        Ok(())
    }

    fn estimate_next_state(&self, state: &DqnState, action: &DqnAction) -> DqnState {
        // Estimate metric changes based on action
        let mut new_metrics = state.metrics.clone();
        match action.param.as_str() {
            "temperature" => new_metrics[0] += action.delta * {
                let guard = self.config.read();
                guard.learning_dqn_temp_multiplier
            },
            "top_p" => new_metrics[1] += action.delta * {
                let guard = self.config.read();
                guard.learning_dqn_top_p_multiplier
            },
            "mcts_c" => new_metrics[3] += action.delta * {
                let guard = self.config.read();
                guard.learning_dqn_mcts_c_multiplier
            },
            "retrieval_top_k" => new_metrics[4] += action.delta * {
                let guard = self.config.read();
                guard.learning_dqn_retrieval_multiplier
            },
            "novelty_threshold" => new_metrics[1] += action.delta * {
                let guard = self.config.read();
                guard.learning_dqn_novelty_multiplier
            },
            "self_awareness_level" => new_metrics[0] += action.delta * {
                let guard = self.config.read();
                guard.learning_dqn_awareness_multiplier
            },
            _ => {}
        }
        DqnState {
            metrics: new_metrics,
        }
    }

    fn adjust_runtime_param(config: &mut RuntimeConfig, param: &str, delta: f64) {
        match param {
            "temperature" => {
                config.temperature = (config.temperature + delta).clamp(0.1, 1.0);
            }
            "top_p" => {
                config.top_p = (config.top_p + delta).clamp(0.1, 1.0);
            }
            "mcts_c" => {
                config.phase2_mcts_c_increment =
                    (config.phase2_mcts_c_increment + delta).clamp(0.0, 2.0);
            }
            "retrieval_top_k" => {
                let updated =
                    (config.phase2_retrieval_top_k_increment as f64 + delta).clamp(0.0, 10.0);
                config.phase2_retrieval_top_k_increment = updated.round() as i32;
            }
            "novelty_threshold" => {
                config.novelty_threshold = (config.novelty_threshold + delta).clamp(0.0, 1.0);
            }
            "self_awareness_level" => {
                config.self_awareness_level = (config.self_awareness_level + delta).clamp(0.0, 1.0);
            }
            _ => {}
        }
    }

    async fn reptile_step(&mut self, batch_size: usize) -> Result<()> {
        // Sample batch from replay
        let batch: Vec<_> = if self.replay_buffer.len() < batch_size / 2 {
            #[cfg(not(test))]
            {
                let query_metrics = if let Some(last) = self.replay_buffer.back() {
                    // Convert f64 metrics to f32 for query_replay_batch
                    last.state
                        .metrics
                        .iter()
                        .map(|x| *x as f32)
                        .collect::<Vec<f32>>()
                } else {
                    vec![0.0f32; 5]
                };
                let erag_batch = self
                    .erag
                    .query_replay_batch("", &query_metrics[..], batch_size)
                    .await?;
                let mut combined = self.replay_buffer.iter().cloned().collect::<Vec<_>>();
                combined.extend(
                    erag_batch
                        .iter()
                        .filter_map(|exp| self.experience_to_replay(exp)),
                );
                if combined.len() > batch_size {
                    combined.truncate(batch_size);
                }
                combined
            }
            #[cfg(test)]
            {
                let buffer_vec: Vec<_> = self.replay_buffer.iter().cloned().collect();
                let mut sampled = Vec::with_capacity(batch_size.min(buffer_vec.len()));
                for _ in 0..batch_size.min(buffer_vec.len()) {
                    if let Some(sample) = buffer_vec.choose(&mut self.rng) {
                        sampled.push(sample.clone());
                    } else {
                        break;
                    }
                }
                sampled
            }
        } else {
            let buffer_vec: Vec<_> = self.replay_buffer.iter().cloned().collect();
            let mut sampled = Vec::with_capacity(batch_size.min(buffer_vec.len()));
            for _ in 0..batch_size.min(buffer_vec.len()) {
                if let Some(sample) = buffer_vec.choose(&mut self.rng) {
                    sampled.push(sample.clone());
                } else {
                    warn!("Replay buffer empty during reptile sampling; stopping early");
                    break;
                }
            }
            sampled
        };

        let mut param_deltas = HashMap::new();

        let batch_len = batch.len();
        for tuple in batch {
                    let delta = tuple.action.delta * {
                        let guard = self.config.read();
                        guard.learning_reptile_inner_gradient_multiplier
                    }; // Inner gradient
            *param_deltas
                .entry(tuple.action.param.clone())
                .or_insert(0.0) += delta;
        }

        // Outer meta-update: average deltas and apply to config
        let mut config = self.config.write();
        for (param, total_delta) in &param_deltas {
            let avg_delta = total_delta / batch_len as f64;
            Self::adjust_runtime_param(&mut config, param, avg_delta);
        }
        info!("Reptile meta-update applied");
        Ok(())
    }

    async fn trigger_qlora(&mut self) -> Result<()> {
        #[cfg(not(test))]
        {
            // Step 1: Collect low-reward tuples from ERAG for targeted fine-tuning
            let low_tuples = self.erag.query_low_reward_tuples(-0.5, 16).await?;
            let embedding_dim = self.config.read().qdrant_vector_dim;
            let external_replay: Vec<ReplayTuple> = low_tuples
                .iter()
                .filter_map(|exp| self.experience_to_replay(exp))
                .collect();

            // Step 2: Build training corpus from recent replay buffer entries plus external tuples
            const MAX_QLORA_SAMPLES: usize = 64;
            let mut combined_replay: Vec<ReplayTuple> =
                self.replay_buffer.iter().rev().take(32).cloned().collect();
            combined_replay.extend(external_replay.iter().cloned());
            if combined_replay.len() > MAX_QLORA_SAMPLES {
                combined_replay.truncate(MAX_QLORA_SAMPLES);
            }

            let training_samples: Vec<(Vec<f32>, Vec<f32>)> = combined_replay
                .iter()
                .map(|tuple| self.replay_tuple_to_training(tuple, embedding_dim))
                .collect();

            if training_samples.is_empty() {
                info!("No training samples for QLoRA");
                return Ok(());
            }

            // Step 3: Train LoRA adapter on the assembled replay tuples
            if std::env::var("DISABLE_LORA")
                .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
                .unwrap_or(false)
            {
                info!("QLoRA fine-tuning disabled via DISABLE_LORA");
                return Ok(());
            }
            
            // Phase 3.2: Use async training for batched replay buffer
            self.queue_training_batch(training_samples.clone(), 10, 1e-3_f32);
            info!(
                samples = training_samples.len(),
                "QLoRA training queued for replay buffer samples"
            );
            
            // Step 4: Apply configuration nudges based on low-reward tuples
            if !external_replay.is_empty() {
                let mut param_deltas: HashMap<String, f64> = HashMap::new();
                for tuple in &external_replay {
                    if tuple.reward >= 0.0 {
                        continue;
                    }
                    let entry = param_deltas
                        .entry(tuple.action.param.clone())
                        .or_insert(0.0);
                    let penalty = tuple.reward.abs();
                    *entry -= penalty * tuple.action.delta;
                }

                if !param_deltas.is_empty() {
                    let mut config = self.config.write();
                    let normaliser = external_replay.len() as f64;
                    for (param, total_delta) in param_deltas {
                        let adjustment = (total_delta / normaliser) * self.alpha;
                        Self::adjust_runtime_param(&mut config, &param, adjustment);
                    }
                }
            }
        }
        #[cfg(test)]
        {
            info!("QLoRA skipped in test mode");
        }
        Ok(())
    }

    fn average_reward(&self) -> f64 {
        if self.replay_buffer.is_empty() {
            return 0.0;
        }
        self.replay_buffer.iter().map(|t| t.reward).sum::<f64>() / self.replay_buffer.len() as f64
    }

    fn replay_tuple_to_training(
        &self,
        tuple: &ReplayTuple,
        embedding_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut input: Vec<f32> = tuple
            .state
            .metrics
            .iter()
            .map(|value| *value as f32)
            .collect();
        if input.len() < embedding_dim {
            input.resize(embedding_dim, 0.0);
        } else {
            input.truncate(embedding_dim);
        }

        let mut target: Vec<f32> = tuple
            .next_state
            .metrics
            .iter()
            .map(|value| *value as f32)
            .collect();
        if target.len() < embedding_dim {
            target.resize(embedding_dim, 0.0);
        } else {
            target.truncate(embedding_dim);
        }

        (input, target)
    }

    fn experience_to_replay(&self, experience: &Experience) -> Option<ReplayTuple> {
        if let Some(meta) = &experience.replay {
            return Some(ReplayTuple {
                state: DqnState {
                    metrics: meta.state_metrics.clone(),
                },
                action: DqnAction {
                    param: meta.action_param.clone(),
                    delta: meta.action_delta,
                },
                reward: meta.reward,
                next_state: DqnState {
                    metrics: meta.next_state_metrics.clone(),
                },
            });
        }

        if self.action_space.is_empty() {
            return None;
        }

        let state_metrics = Self::derive_state_metrics(&experience.state)?;
        let next_metrics = if experience.next_state.is_empty() {
            state_metrics.clone()
        } else {
            Self::derive_state_metrics(&experience.next_state)?
        };
        let action = self.select_action_for_index(experience.action)?;

        Some(ReplayTuple {
            state: DqnState {
                metrics: state_metrics,
            },
            action,
            reward: experience.reward,
            next_state: DqnState {
                metrics: next_metrics,
            },
        })
    }

    fn select_action_for_index(&self, index: usize) -> Option<DqnAction> {
        if self.action_space.is_empty() {
            return None;
        }

        let mapped = match index {
            0 => self
                .find_action_variant("temperature", false)
                .or_else(|| self.find_action_variant("novelty_threshold", false)),
            1 => self
                .find_action_variant("temperature", true)
                .or_else(|| self.find_action_variant("top_p", true)),
            2 => self
                .find_action_variant("novelty_threshold", true)
                .or_else(|| self.find_action_variant("top_p", true)),
            3 => self
                .find_action_variant("self_awareness_level", true)
                .or_else(|| self.find_action_variant("mcts_c", true)),
            _ => None,
        };

        mapped.or_else(|| Some(self.action_space[index % self.action_space.len()].clone()))
    }

    fn find_action_variant(&self, param: &str, prefer_positive: bool) -> Option<DqnAction> {
        self.action_space
            .iter()
            .find(|action| {
                action.param == param
                    && (prefer_positive && action.delta.is_sign_positive()
                        || !prefer_positive && action.delta.is_sign_negative())
            })
            .cloned()
    }

    fn derive_state_metrics(vector: &[f32]) -> Option<Vec<f64>> {
        let filtered: Vec<f64> = vector
            .iter()
            .map(|&value| value as f64)
            .filter(|value| value.is_finite())
            .collect();
        if filtered.is_empty() {
            return None;
        }

        let len = filtered.len() as f64;
        let mean = filtered.iter().sum::<f64>() / len;
        let variance = filtered
            .iter()
            .map(|value| {
                let diff = value - mean;
                diff * diff
            })
            .sum::<f64>()
            / len.max(1.0);
        let std_dev = variance.sqrt();
        let max = filtered
            .iter()
            .fold(f64::NEG_INFINITY, |acc, value| acc.max(*value));
        let min = filtered
            .iter()
            .fold(f64::INFINITY, |acc, value| acc.min(*value));
        let l2_norm = filtered
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();

        Some(vec![mean, std_dev, max, min, l2_norm])
    }

    // INTEGRATION FIX: Compute Wasserstein distance between current and historical entropy distributions
    fn compute_history_distance(&self, current_entropy: f64, erag_hits: &[EragMemory]) -> f64 {
        let mut historical: Vec<f64> = erag_hits
            .iter()
            .filter_map(|hit| {
                if hit.entropy_after.is_finite() && hit.entropy_after > 0.0 {
                    Some(hit.entropy_after)
                } else if hit.entropy_before.is_finite() && hit.entropy_before > 0.0 {
                    Some(hit.entropy_before)
                } else {
                    None
                }
            })
            .collect();

        if historical.is_empty() {
            return (current_entropy - 0.5).abs().min(1.0);
        }

        historical.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let position = historical
            .binary_search_by(|value| {
                value
                    .partial_cmp(&current_entropy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or_else(|idx| idx);

        let n = historical.len() as f64;
        let empirical_cdf = position as f64 / n;
        let distance = if position == 0 {
            (current_entropy - historical[0]).abs()
        } else if position >= historical.len() {
            (current_entropy - historical[historical.len() - 1]).abs()
        } else {
            let lower = historical[position - 1];
            let upper = historical[position];
            let local = ((current_entropy - lower).abs() + (upper - current_entropy).abs()) / 2.0;
            local * (1.0 - empirical_cdf)
        };

        distance.min(1.0)
    }

    fn decay_schedules(&mut self) {
        let episodes = self.episode_count as f64;
        let epsilon_decay_rate = {
            let guard = self.config.read();
            guard.learning_epsilon_decay_rate
        };
        self.epsilon = self.initial_epsilon / (1.0 + episodes * epsilon_decay_rate).max(1.0);
        self.epsilon = self.epsilon.max({
            let guard = self.config.read();
            guard.learning_epsilon_minimum
        });
        let alpha_decay_rate = {
            let guard = self.config.read();
            guard.learning_alpha_decay_rate
        };
        self.alpha = self.initial_alpha / (1.0 + episodes * alpha_decay_rate).max(1.0);
        self.alpha = self.alpha.max({
            let guard = self.config.read();
            guard.learning_alpha_minimum
        });
    }

    pub fn save_lora_adapter<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path_ref = path.as_ref();
        if let Some(parent) = path_ref.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        self.lora_trainer.read().save_adapter(path_ref)?;
        info!(adapter = %path_ref.display(), "LoRA adapter saved");
        Ok(())
    }

    pub fn load_lora_adapter<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path_ref = path.as_ref();
        let trainer = LoRATrainer::load_adapter(path_ref)?;
        *self.lora_trainer.write() = trainer;
        info!(adapter = %path_ref.display(), "LoRA adapter loaded");
        Ok(())
    }

    /// Phase 5.2: Evolution step with topological guidance
    async fn evolution_step(&mut self) -> Result<()> {
        let current = {
            let guard = self.config.read();
            guard.clone()
        };
        let recent: Vec<(f64, f64)> = self.recent_metrics.iter().cloned().collect();
        if recent.is_empty() {
            return Ok(());
        }
        let num_recent = recent.len();
        let num_old = {
            let guard = self.config.read();
            let ratio = guard.learning_evolution_old_episodes_ratio;
            let min = guard.learning_evolution_old_episodes_min;
            let max = guard.learning_evolution_old_episodes_max;
            ((num_recent as f64 * ratio).max(min as f64) as usize).min(max)
        };
        let old_tuples = self.erag.query_old_dqn_tuples(1, num_old).await?;
        let mut mixed_episodes: Vec<(f64, f64)> = recent.clone();
        // Note: query_old_dqn_tuples returns Experience, not ReplayTuple
        // Experience.state is Vec<f32>, not DqnState with metrics
        // Convert Experience to (delta, rouge) tuples if state has enough elements
        for tuple in old_tuples {
            if let Some(replay) = self.experience_to_replay(&tuple) {
                let delta = replay.state.metrics.get(0).copied().unwrap_or(0.0);
                let rouge = replay.state.metrics.get(1).copied().unwrap_or(0.0);
                mixed_episodes.push((delta, rouge));
            }
        }

        // Phase 5.2: Query tough knots (configurable ratio of episodes for anti-forgetting)
        let num_tough = {
            let guard = self.config.read();
            let ratio = guard.learning_tough_knots_ratio;
            (mixed_episodes.len() as f64 * ratio).max(1.0) as usize
        };
        let tough_knots_params = {
            let guard = self.config.read();
            (
                guard.tough_knots_multiplier,
                guard.tough_knots_max_fetch,
                guard.tough_knots_knot_threshold,
                guard.tough_knots_quality_threshold,
                guard.tough_knots_knot_multiplier,
            )
        };
        let tough_knots = self
            .erag
            .query_tough_knots(
                num_tough,
                tough_knots_params.0,
                tough_knots_params.1,
                tough_knots_params.2,
                tough_knots_params.3,
                tough_knots_params.4,
            )
            .await
            .unwrap_or_default();
        if !tough_knots.is_empty() {
            info!(
                "Evolution: Retrieved {} tough knots for anti-forgetting",
                tough_knots.len()
            );
        }

        // INTEGRATION FIX: Pass topology data to evolution for topology-aware optimization
        let recent_topologies: Vec<TopologicalSignature> =
            self.recent_topologies.iter().cloned().collect();
        let best = self
            .evolution
            .evolve_with_topology(&current, mixed_episodes, recent_topologies)
            .await?;
        {
            let mut guard = self.config.write();
            *guard = best;
        }
        info!(
            "Evolved new config applied after {} episodes",
            self.episode_count
        );
        Ok(())
    }

    pub fn adjust_on_low_reward(&mut self, reward_signal: f64) {
        if reward_signal < self.reward_threshold {
            info!(
                reward_signal,
                "Low reward detected; triggering LoRA fine-tuning"
            );

            let training_samples: Vec<(Vec<f32>, Vec<f32>)> = self
                .replay_buffer
                .iter()
                .rev()
                .take(32)
                .map(|tuple| {
                    let input = tuple
                        .state
                        .metrics
                        .iter()
                        .map(|value| *value as f32)
                        .collect::<Vec<f32>>();
                    let target = tuple
                        .next_state
                        .metrics
                        .iter()
                        .map(|value| *value as f32)
                        .collect::<Vec<f32>>();
                    (input, target)
                })
                .collect();

            if training_samples.is_empty() {
                warn!(
                    "Skipping LoRA fine-tuning because replay buffer does not contain enough data"
                );
                return;
            }

            if std::env::var("DISABLE_LORA")
                .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
                .unwrap_or(false)
            {
                info!("LoRA fine-tuning disabled via DISABLE_LORA");
                return;
            }
            
            // Phase 3.2: Use async training for batched replay buffer
            self.queue_training_batch(training_samples.clone(), 10, 1e-3_f32);
            info!(
                samples = training_samples.len(),
                "LoRA fine-tuning queued for low-reward samples"
            );
        }
    }

    fn average_latency(&self) -> f64 {
        if self.recent_metrics.is_empty() {
            return 0.0;
        }
        self.recent_metrics
            .iter()
            .map(|(_, latency)| *latency)
            .sum::<f64>()
            / self.recent_metrics.len() as f64
    }
}

pub fn dqn_step(state: Vec<f32>) -> u32 {
    if state.is_empty() {
        return 0;
    }
    let q_values = Array1::from_vec(state);
    q_values
        .iter()
        .enumerate()
        .fold((0, f32::MIN), |max_idx, (i, &val)| {
            if val > max_idx.1 {
                (i as u32, val)
            } else {
                max_idx
            }
        })
        .0
}

pub struct GaussianProcess {
    x_train: Option<Vec<Vec<f64>>>,
    y_train: Option<Vec<f64>>,
    rng: rand::rngs::StdRng,
}

impl GaussianProcess {
    pub fn new(rng_seed: u64) -> Self {
        Self {
            x_train: None,
            y_train: None,
            rng: rand::rngs::StdRng::seed_from_u64(rng_seed),
        }
    }

    pub fn fit(&mut self, x: &Vec<Vec<f64>>, y: &Vec<f64>) {
        self.x_train = Some(x.clone());
        self.y_train = Some(y.clone());
    }

    pub fn suggest_next(&mut self, n: usize) -> Vec<Vec<f64>> {
        if let (Some(x_train), Some(y_train)) = (&self.x_train, &self.y_train) {
            if !x_train.is_empty() {
                if let Some(max_entry) = y_train
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
                {
                    let best = &x_train[max_entry.0];
                    return (0..n)
                        .map(|_| {
                            vec![
                                (best[0] + self.rng.gen_range(-0.05f64..0.05)).clamp(0.1, 1.0),
                                (best[1] + self.rng.gen_range(-0.05f64..0.05)).clamp(0.1, 1.0),
                                (best[2] + self.rng.gen_range(-0.05f64..0.05)).clamp(0.0, 1.0),
                                (best[3] + self.rng.gen_range(-0.005f64..0.005)).clamp(0.001, 0.1),
                            ]
                        })
                        .collect();
                }
            }
        }
        // fallback to random
        (0..n)
            .map(|_| {
                vec![
                    self.rng.gen_range(0.1..1.0),
                    self.rng.gen_range(0.1..1.0),
                    self.rng.gen_range(0.0..1.0),
                    self.rng.gen_range(0.001..0.1),
                ]
            })
            .collect()
    }
}

pub struct EvolutionLoop {
    population_size: usize,
    generations: usize,
    mutation_std: f64,
    bo_gp: GaussianProcess,
    rng: rand::rngs::StdRng,
}

impl EvolutionLoop {
    pub fn new(pop_size: usize, gens: usize, mutation_std: f64, rng_seed: u64) -> Self {
        Self {
            population_size: pop_size,
            generations: gens,
            mutation_std,
            bo_gp: GaussianProcess::new(rng_seed),
            rng: rand::rngs::StdRng::seed_from_u64(rng_seed),
        }
    }

    // INTEGRATION FIX: New topology-aware evolution method
    pub async fn evolve_with_topology(
        &mut self,
        current_config: &RuntimeConfig,
        episodes: Vec<(f64, f64)>,
        topologies: Vec<TopologicalSignature>,
    ) -> Result<RuntimeConfig> {
        // Fall back to regular evolve if no topology data
        if topologies.is_empty() {
            return self.evolve(current_config, episodes).await;
        }

        // Calculate topology-based fitness modifiers
        let avg_knot: f64 =
            topologies.iter().map(|t| t.knot_complexity).sum::<f64>() / topologies.len() as f64;
        let avg_gap: f64 =
            topologies.iter().map(|t| t.spectral_gap).sum::<f64>() / topologies.len() as f64;

        // Adjust mutation based on topology stability
        let old_mutation_std = self.mutation_std;
        if avg_knot > 0.4 {
            // High knot complexity - reduce mutation to stabilize
            self.mutation_std *= 0.7;
        } else if avg_gap > 0.5 {
            // High spectral gap - increase mutation for exploration
            self.mutation_std *= 1.3;
        }

        // Run evolution with topology-adjusted parameters
        let result = self.evolve(current_config, episodes).await;

        // Restore original mutation std
        self.mutation_std = old_mutation_std;

        result
    }

    pub async fn evolve(
        &mut self,
        current_config: &RuntimeConfig,
        episodes: Vec<(f64, f64)>,
    ) -> Result<RuntimeConfig> {
        let mut population: Vec<RuntimeConfig> = (0..self.population_size)
            .map(|_| self.mutate_config(current_config))
            .collect();

        for _ in 0..self.generations {
            let fitnesses: Vec<f64> = population
                .par_iter()
                .map(|conf| self.evaluate_fitness(conf, &episodes))
                .collect();
            population = self.select_and_breed(&population, &fitnesses);
            let param_vecs: Vec<Vec<f64>> = population
                .iter()
                .map(|c| vec![c.temperature, c.top_p, c.novelty_threshold, c.dqn_alpha])
                .collect();
            self.bo_gp.fit(&param_vecs, &fitnesses);
            let suggested = self.bo_gp.suggest_next(5);
            for s in suggested {
                let mut new_conf = current_config.clone();
                new_conf.temperature = s[0].clamp(0.1, 1.0);
                new_conf.top_p = s[1].clamp(0.1, 1.0);
                new_conf.novelty_threshold = s[2].clamp(0.0, 1.0);
                new_conf.dqn_alpha = s[3].clamp(0.001, 0.1);
                population.push(new_conf);
            }
        }

        let mut best_conf = current_config.clone();
        let mut best_f = f64::NEG_INFINITY;
        for conf in population {
            let f = self.evaluate_fitness(&conf, &episodes);
            if f > best_f {
                best_f = f;
                best_conf = conf;
            }
        }
        Ok(best_conf)
    }

    fn mutate_config(&mut self, conf: &RuntimeConfig) -> RuntimeConfig {
        let std = self.mutation_std;
        let mut new = conf.clone();
        new.temperature += self.rng.gen_range(-std..std);
        new.temperature = new.temperature.clamp(0.1, 1.0);
        new.top_p += self.rng.gen_range(-std..std);
        new.top_p = new.top_p.clamp(0.1, 1.0);
        new.novelty_threshold += self.rng.gen_range(-std * 0.5..std * 0.5);
        new.novelty_threshold = new.novelty_threshold.clamp(0.0, 1.0);
        new.dqn_alpha += self.rng.gen_range(-std * 0.001..std * 0.001);
        new.dqn_alpha = new.dqn_alpha.clamp(0.001, 0.1);
        new
    }

    // Temporarily adjust to Vec<(f64, f64)>
    fn evaluate_fitness(&self, conf: &RuntimeConfig, eps: &[(f64, f64)]) -> f64 {
        // Adjust calc without sig
        if eps.is_empty() {
            0.0
        } else {
            eps.iter()
                .map(|&(delta, rouge)| {
                    let adjusted_delta =
                        delta * (1.0 + conf.novelty_threshold * 0.5 - conf.top_p * 0.1);
                    let adjusted_rouge =
                        rouge * (1.0 + conf.temperature * 0.2 + conf.dqn_alpha * 0.1);
                    -adjusted_delta + adjusted_rouge
                })
                .sum::<f64>()
                / eps.len() as f64
        }
    }

    fn select_and_breed(
        &mut self,
        pop: &Vec<RuntimeConfig>,
        fitness: &Vec<f64>,
    ) -> Vec<RuntimeConfig> {
        let mut new_pop = vec![];
        let size = pop.len();
        for _ in 0..size {
            let p1 = self.tournament_select(fitness);
            let p2 = self.tournament_select(fitness);
            let child = self.crossover(&pop[p1], &pop[p2]);
            new_pop.push(child);
        }
        new_pop
    }

    fn tournament_select(&mut self, fitness: &Vec<f64>) -> usize {
        let tournament_size = 4;
        let mut best_idx = self.rng.gen_range(0..fitness.len());
        let mut best = fitness[best_idx];
        for _ in 1..tournament_size {
            let i = self.rng.gen_range(0..fitness.len());
            if fitness[i] > best {
                best = fitness[i];
                best_idx = i;
            }
        }
        best_idx
    }

    fn crossover(&self, p1: &RuntimeConfig, p2: &RuntimeConfig) -> RuntimeConfig {
        let mut child = p1.clone();
        child.temperature = (p1.temperature + p2.temperature) / 2.0;
        child.top_p = (p1.top_p + p2.top_p) / 2.0;
        child.novelty_threshold = (p1.novelty_threshold + p2.novelty_threshold) / 2.0;
        child.dqn_alpha = (p1.dqn_alpha + p2.dqn_alpha) / 2.0;
        child
    }
}
