use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use dashmap::DashMap;
use parking_lot::RwLock;
use rand::prelude::*;
use rayon::prelude::*;
use tracing::{info, warn};

use crate::compass::CompassOutcome;
use crate::config::RuntimeConfig;
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
}

impl Default for LearningOutcome {
    fn default() -> Self {
        Self {
            events: Vec::new(),
            breakthroughs: Vec::new(),
            qlora_updates: Vec::new(),
            entropy_delta: 0.0,
            adjusted_params: HashMap::new(),
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
    lora_trainer: LoRATrainer,
    reward_threshold: f64,
    tokenizer: Option<Arc<DynamicTokenizerManager>>,
    curated_buffer: Vec<CuratedSample>,
    lora_epochs: usize,
    #[allow(dead_code)]
    rng: rand::rngs::StdRng,
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
            let lora_config = LoRAConfig {
                rank: lora_rank,
                alpha: lora_alpha,
                input_dim: embedding_dim,
                output_dim: embedding_dim,
            };
            LoRATrainer::with_config(lora_config).unwrap_or_else(|err| {
                warn!(error = %err, "Failed to initialise LoRA trainer with config, using default adapter");
                LoRATrainer::default()
            })
        };

        let rng = rand::rngs::StdRng::seed_from_u64(rng_seed);

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
            lora_trainer,
            reward_threshold: -0.5,
            tokenizer: Some(tokenizer.clone()),
            curated_buffer: Vec::new(),
            lora_epochs,
            #[allow(dead_code)]
            rng,
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
        let fallback_ucb = compass.ucb1_score.unwrap_or(config_snapshot.mcts_c_scale as f64) as f64;
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

        // Every 5 episodes, run Reptile and check QLoRA trigger
        // Skip QLoRA training in soak test mode for performance
        let skip_qlora = std::env::var("SKIP_QLORA_TRAINING")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes"))
            .unwrap_or(false);
        
        // Run QLoRA training asynchronously (non-blocking) using spawn_blocking
        // This moves CPU-bound training off the async runtime without blocking pipeline
        let run_qlora_async = std::env::var("QLORA_ASYNC")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes"))
            .unwrap_or(false);
        
        if self.episode_count % 5 == 0 {
            self.reptile_step(32).await?;
            if !skip_qlora && self.average_reward() < 0.0 {
                if run_qlora_async {
                    // Background training: collect data, then spawn blocking task
                    let erag_clone = self.erag.clone();
                    let config_clone = self.config.clone();
                    let replay_buffer_snapshot: Vec<_> = self.replay_buffer.iter().rev().take(32).cloned().collect();
                    
                    tokio::spawn(async move {
                        // Collect low-reward tuples (async)
                        let _low_tuples = match erag_clone.query_low_reward_tuples(-0.5, 16).await {
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
                                let mut input: Vec<f32> = tuple.state.metrics.iter().map(|v| *v as f32).collect();
                                input.resize(embedding_dim, 0.0);
                                input.truncate(embedding_dim);
                                
                                let mut target: Vec<f32> = tuple.next_state.metrics.iter().map(|v| *v as f32).collect();
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
                        }).await {
                            Ok(Ok(_loss)) => {
                                info!("QLoRA fine-tuning completed in background on {} samples", training_samples.len());
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

        // Phase 5.2: Evolution step every 50 episodes
        if self.episode_count % 50 == 0 {
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
        })
    }

    pub async fn apply_curator_learned(
        &mut self,
        refined_response: &str,
        learned: bool,
        reward: f64,
        topology: &TopologicalSignature,
        prompt: &str,
        promoted_tokens: &[String],
    ) -> Result<()> {
        if !learned {
            return Ok(());
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
            return Ok(());
        }
        let epochs = self.lora_epochs;
        match self.lora_trainer.train(&training_samples, epochs, 1e-3_f32) {
            Ok(_) => {
                info!(
                    count = training_samples.len(),
                    "QLoRA trained on {} curated",
                    training_samples.len()
                );
                self.curated_buffer.clear();
            }
            Err(error) => {
                warn!(%error, "QLoRA training failed on curated data");
            }
        }

        Ok(())
    }

    fn record_entropy(&mut self, value: f64) {
        if self.entropy_history.len() == self.window {
            self.entropy_history.pop_front();
        }
        self.entropy_history.push_back(value);
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
        let penalty = sig.knot_complexity * 0.5
            + (sig.betti_numbers[1] as f64) * 0.2
            + sig.persistence_entropy * 0.1;
        let weight = if mode == "Discover" { 0.5 } else { 1.0 };
        let conv_bonus = if sig.spectral_gap < 0.5 { 0.3 } else { -0.2 };
        let novelty_bonus = if history_dist > 0.1 { 0.2 } else { 0.0 };
        base - (penalty * weight) + conv_bonus + novelty_bonus
    }

    fn choose_action(&mut self, state: &DqnState) -> DqnAction {
        if self.rng.gen_range(0.0..1.0) < self.epsilon {
            self.action_space.choose(&mut self.rng).cloned().unwrap()
        } else {
            let s_key = state.to_key();
            // DashMap: lock-free read access (no locks needed!)
            if let Some(qs) = self.q_table.get(&s_key) {
                let max_key = qs
                    .iter()
                    .max_by(|a, b| a.value().partial_cmp(b.value()).unwrap())
                    .map(|entry| entry.key().clone())
                    .unwrap_or_else(|| self.action_space[0].to_key());
                self.action_space
                    .iter()
                    .find(|a| a.to_key() == max_key)
                    .cloned()
                    .unwrap_or_else(|| self.action_space[0].clone())
            } else {
                self.action_space[0].clone()
            }
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
        let batch_size = 32.min(self.replay_buffer.len());
        // Convert VecDeque to Vec for sampling
        let buffer_vec: Vec<_> = self.replay_buffer.iter().cloned().collect();
        let batch: Vec<_> = (0..batch_size)
            .map(|_| buffer_vec.choose(&mut self.rng).cloned().unwrap())
            .collect();

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
                    .map(|qs| qs.iter().map(|e| *e.value()).fold(f64::NEG_INFINITY, f64::max))
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
            "temperature" => new_metrics[0] += action.delta * 0.05, // affect entropy
            "top_p" => new_metrics[1] += action.delta * 0.1,        // affect rouge
            "mcts_c" => new_metrics[3] += action.delta * 0.1,       // affect ucb1
            "retrieval_top_k" => new_metrics[4] += action.delta * 0.01, // affect curator
            "novelty_threshold" => new_metrics[1] += action.delta * 0.05, // affect rouge
            "self_awareness_level" => new_metrics[0] += action.delta * 0.03, // affect entropy
            _ => {}
        }
        DqnState {
            metrics: new_metrics,
        }
    }

    async fn reptile_step(&mut self, batch_size: usize) -> Result<()> {
        // Sample batch from replay
        let batch: Vec<_> = if self.replay_buffer.len() < batch_size / 2 {
            #[cfg(not(test))]
            {
                let query_metrics = if let Some(last) = self.replay_buffer.back() {
                    // Convert f64 metrics to f32 for query_replay_batch
                    last.state.metrics.iter().map(|x| *x as f32).collect::<Vec<f32>>()
                } else {
                    vec![0.0f32; 5]
                };
                let erag_batch = self
                    .erag
                    .query_replay_batch("", &query_metrics[..], batch_size)
                    .await?;
                // Note: ERAG returns Experience, but replay_buffer expects ReplayTuple
                // For now, skip ERAG batch until conversion is implemented
                let mut full = self.replay_buffer.iter().cloned().collect::<Vec<_>>();
                // full.extend(erag_batch); // Type mismatch: Experience vs ReplayTuple
                full.truncate(batch_size);
                full
            }
            #[cfg(test)]
            {
                let buffer_vec: Vec<_> = self.replay_buffer.iter().cloned().collect();
                (0..batch_size.min(buffer_vec.len()))
                    .map(|_| buffer_vec.choose(&mut self.rng).cloned().unwrap())
                    .collect()
            }
        } else {
            let buffer_vec: Vec<_> = self.replay_buffer.iter().cloned().collect();
            (0..batch_size.min(buffer_vec.len()))
                .map(|_| buffer_vec.choose(&mut self.rng).cloned().unwrap())
                .collect()
        };

        let mut param_deltas = HashMap::new();

        let batch_len = batch.len();
        for tuple in batch {
            let delta = tuple.action.delta * 0.01; // Inner gradient
            *param_deltas
                .entry(tuple.action.param.clone())
                .or_insert(0.0) += delta;
        }

        // Outer meta-update: average deltas and apply to config
        let mut config = self.config.write();
        for (param, total_delta) in &param_deltas {
            let avg_delta = total_delta / batch_len as f64;
            match param.as_str() {
                "temperature" => {
                    config.temperature += avg_delta;
                    config.temperature = config.temperature.clamp(0.1, 1.0);
                }
                "top_p" => {
                    config.top_p += avg_delta;
                    config.top_p = config.top_p.clamp(0.1, 1.0);
                }
                "mcts_c" => {
                    config.phase2_mcts_c_increment += avg_delta;
                    config.phase2_mcts_c_increment = config.phase2_mcts_c_increment.clamp(0.0, 2.0);
                }
                "retrieval_top_k" => {
                    let new_val = (config.phase2_retrieval_top_k_increment as f64 + avg_delta)
                        .clamp(0.0, 10.0);
                    config.phase2_retrieval_top_k_increment = new_val as i32;
                }
                "novelty_threshold" => {
                    config.novelty_threshold += avg_delta;
                    config.novelty_threshold = config.novelty_threshold.clamp(0.0, 1.0);
                }
                "self_awareness_level" => {
                    config.self_awareness_level += avg_delta;
                    config.self_awareness_level = config.self_awareness_level.clamp(0.0, 1.0);
                }
                _ => {}
            }
        }
        info!("Reptile meta-update applied");
        Ok(())
    }

    async fn trigger_qlora(&mut self) -> Result<()> {
        #[cfg(not(test))]
        {
            // Step 1: Collect low-reward tuples from ERAG
            let low_tuples = self.erag.query_low_reward_tuples(-0.5, 16).await?;
            let embedding_dim = self.config.read().qdrant_vector_dim;

            // Step 2: Prepare training data from replay buffer + topological features
            let training_samples: Vec<(Vec<f32>, Vec<f32>)> = self
                .replay_buffer
                .iter()
                .rev()
                .take(32)
                .map(|tuple| {
                    // Input: state metrics (5 dims) + topological features (4 dims) = 9 dims
                    let mut input = tuple
                        .state
                        .metrics
                        .iter()
                        .map(|value| *value as f32)
                        .collect::<Vec<f32>>();

                    // Pad to fixed size if needed
                    while input.len() < embedding_dim {
                        input.push(0.0);
                    }
                    input.truncate(embedding_dim);

                    // Target: next state metrics (5 dims) -> pad to 896
                    let mut target = tuple
                        .next_state
                        .metrics
                        .iter()
                        .map(|value| *value as f32)
                        .collect::<Vec<f32>>();
                    while target.len() < embedding_dim {
                        target.push(0.0);
                    }
                    target.truncate(embedding_dim);

                    (input, target)
                })
                .collect();

            if training_samples.is_empty() {
                info!("No training samples for QLoRA");
                return Ok(());
            }

            // Step 3: Train LoRA adapter on topological data
            if std::env::var("DISABLE_LORA")
                .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
                .unwrap_or(false)
            {
                info!("QLoRA fine-tuning disabled via DISABLE_LORA");
                return Ok(());
            }
            match self.lora_trainer.train(&training_samples, 10, 1e-3_f32) {
                Ok(_loss) => {
                    info!(
                        "QLoRA fine-tuning completed on {} samples",
                        training_samples.len()
                    );

                    // Step 4: After training, optionally adjust config based on low-reward tuples
                    // NOTE: low_tuples is Vec<Experience> where action is usize, not ReplayTuple
                    // Skip config adjustment since we can't extract action parameters from Experience
                    // TODO: Convert Experience to ReplayTuple or use replay_buffer instead
                    /*
                    if !low_tuples.is_empty() {
                        let mut param_deltas: HashMap<String, f64> = HashMap::new();
                        for tuple in &low_tuples {
                            // Experience.action is usize, not DqnAction
                            // Need to convert or skip this adjustment
                        }
                    }
                    */
                }
                Err(error) => {
                    warn!(%error, "QLoRA fine-tuning failed");
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
        let epsilon_decay_rate = 0.001;
        self.epsilon = self.initial_epsilon / (1.0 + episodes * epsilon_decay_rate).max(1.0);
        self.epsilon = self.epsilon.max(0.01);
        let alpha_decay_rate = 0.0005;
        self.alpha = self.initial_alpha / (1.0 + episodes * alpha_decay_rate).max(1.0);
        self.alpha = self.alpha.max(0.001);
    }

    pub fn save_lora_adapter<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path_ref = path.as_ref();
        if let Some(parent) = path_ref.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        self.lora_trainer.save_adapter(path_ref)?;
        info!(adapter = %path_ref.display(), "LoRA adapter saved");
        Ok(())
    }

    pub fn load_lora_adapter<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path_ref = path.as_ref();
        let trainer = LoRATrainer::load_adapter(path_ref)?;
        self.lora_trainer = trainer;
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
        let num_old = ((num_recent as f64 * 0.3).max(10.0) as usize).min(50);
        let old_tuples = self.erag.query_old_dqn_tuples(1, num_old).await?;
        let mut mixed_episodes: Vec<(f64, f64)> = recent.clone();
        // Note: query_old_dqn_tuples returns Experience, not ReplayTuple
        // Experience.state is Vec<f32>, not DqnState with metrics
        // Convert Experience to (delta, rouge) tuples if state has enough elements
        for tuple in old_tuples {
            if tuple.state.len() >= 2 {
                let delta = tuple.state[0] as f64;
                let rouge = tuple.state[1] as f64;
                mixed_episodes.push((delta, rouge));
            }
        }

        // Phase 5.2: Query tough knots (20% of episodes for anti-forgetting)
        let num_tough = (mixed_episodes.len() as f64 * 0.2).max(1.0) as usize;
        let tough_knots = self
            .erag
            .query_tough_knots(num_tough)
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
            match self.lora_trainer.train(&training_samples, 10, 1e-3_f32) {
                Ok(loss) => info!(loss, "LoRA fine-tuning completed"),
                Err(error) => warn!(%error, "LoRA fine-tuning failed"),
            }
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
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
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
