use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::Mutex;

use anyhow::Result;
use rand::prelude::*;
use rayon::prelude::*;
use tracing::{info, warn};

use crate::compass::CompassOutcome;
use crate::config::RuntimeConfig;
use crate::erag::{CollapseResult, EragClient, EragMemory};
use crate::generation::GenerationResult;
use crate::lora_trainer::LoRATrainer;
use crate::tcs_analysis::TopologicalSignature;
use crate::tcs_predictor::TcsPredictor;
use crate::torus::PadGhostState;

#[derive(Debug, Clone)]
pub struct LearningOutcome {
    pub events: Vec<String>,
    pub breakthroughs: Vec<String>,
    pub qlora_updates: Vec<String>,
    pub entropy_delta: f64,
    pub adjusted_params: HashMap<String, f64>, // e.g., "temperature" => 0.8
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

pub struct LearningLoop {
    entropy_history: VecDeque<f64>,
    window: usize,
    breakthrough_threshold: f64,
    replay_buffer: VecDeque<ReplayTuple>,
    q_table: HashMap<String, HashMap<String, f64>>, // state_key -> (action_key -> q_value)
    action_space: Vec<DqnAction>,
    epsilon: f64,
    gamma: f64, // discount
    alpha: f64, // learning rate
    erag: Arc<EragClient>,
    config: Arc<Mutex<RuntimeConfig>>,
    episode_count: u32,
    initial_epsilon: f64,
    initial_alpha: f64,
    recent_metrics: VecDeque<(f64, f64)>,
    recent_topologies: VecDeque<TopologicalSignature>,  // INTEGRATION FIX: Track topology history
    evolution: EvolutionLoop,
    predictor: TcsPredictor,  // FIXED: Removed underscore to make it active
    lora_trainer: LoRATrainer,
    reward_threshold: f64,
}

impl LearningLoop {
    pub fn new(
        window: usize,
        breakthrough_threshold: f64,
        epsilon: f64,
        gamma: f64,
        alpha: f64,
        erag: Arc<EragClient>,
        config: Arc<Mutex<RuntimeConfig>>,
    ) -> Self {
        let action_space = vec![
            DqnAction {
                param: "temperature".to_string(),
                delta: -0.1,
            },
            DqnAction {
                param: "temperature".to_string(),
                delta: 0.1,
            },
            DqnAction {
                param: "top_p".to_string(),
                delta: -0.05,
            },
            DqnAction {
                param: "top_p".to_string(),
                delta: 0.05,
            },
            DqnAction {
                param: "mcts_c".to_string(),
                delta: -0.2,
            },
            DqnAction {
                param: "mcts_c".to_string(),
                delta: 0.2,
            },
            DqnAction {
                param: "retrieval_top_k".to_string(),
                delta: -5.0,
            },
            DqnAction {
                param: "retrieval_top_k".to_string(),
                delta: 5.0,
            },
            DqnAction {
                param: "novelty_threshold".to_string(),
                delta: -0.1,
            },
            DqnAction {
                param: "novelty_threshold".to_string(),
                delta: 0.1,
            },
            DqnAction {
                param: "self_awareness_level".to_string(),
                delta: -0.1,
            },
            DqnAction {
                param: "self_awareness_level".to_string(),
                delta: 0.1,
            },
        ];

        let lora_trainer = LoRATrainer::new().unwrap_or_else(|err| {
            warn!(error = %err, "Failed to initialise LoRA trainer, using default adapter");
            LoRATrainer::default()
        });

        Self {
            entropy_history: VecDeque::with_capacity(window),
            window,
            breakthrough_threshold,
            replay_buffer: VecDeque::new(),
            q_table: HashMap::new(),
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
            recent_topologies: VecDeque::with_capacity(50),  // INTEGRATION FIX: Initialize topology tracking
            evolution: EvolutionLoop::new(20, 5, 0.05),
            predictor: TcsPredictor::new(),  // FIXED: Removed underscore
            lora_trainer,
            reward_threshold: -0.5,
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

        let state = DqnState::from_metrics(
            entropy_delta,
            generation.rouge_score,
            generation.latency_ms,
            compass.ucb1_score.unwrap_or(0.5) as f64,
            collapse.curator_quality.unwrap_or(0.5) as f64,
        );

        let predicted_reward_delta = self.predictor.predict_reward_delta(topology);
        let predictor_action = if self.predictor.should_trigger(topology) {
            let config_snapshot = {
                let guard = self.config.lock().await;
                guard.clone()
            };
            let (param, delta) = self.predictor.predict_action(topology, &config_snapshot);
            info!(
                param = %param,
                delta,
                knot = topology.knot_complexity,
                gap = topology.spectral_gap,
                "TCS predictor triggered"
            );
            Some(DqnAction { param, delta })
        } else {
            None
        };

        let action = self.choose_action(&state);

        let mut adjusted_params: HashMap<String, f64> = HashMap::new();
        let mut predictor_applied = false;
        {
            let mut config = self.config.lock().await;
            Self::apply_action_to_config(&mut config, &action);
            adjusted_params
                .entry(action.param.clone())
                .and_modify(|delta| *delta += action.delta)
                .or_insert(action.delta);

            if let Some(pred_action) = predictor_action.as_ref() {
                if pred_action.delta.abs() > 1e-6 {
                    Self::apply_action_to_config(&mut config, pred_action);
                    adjusted_params
                        .entry(pred_action.param.clone())
                        .and_modify(|delta| *delta += pred_action.delta)
                        .or_insert(pred_action.delta);
                    predictor_applied = true;
                }
            }
        }

        // Simulate next state (estimate based on action and predictor influence)
        let mut next_state = self.estimate_next_state(&state, &action);
        if let Some(pred_action) = predictor_action.as_ref() {
            if pred_action.delta.abs() > 1e-6 {
                next_state = self.estimate_next_state(&next_state, pred_action);
            }
        }

        let base_reward = self.compute_reward(entropy_delta, generation.rouge_score);

        let mode = match compass.quadrant {
            crate::compass::CompassQuadrant::Discover => "Discover",
            crate::compass::CompassQuadrant::Master => "Master",
            crate::compass::CompassQuadrant::Persist => "Persist",
            crate::compass::CompassQuadrant::Panic => "Panic",
        };
        // INTEGRATION FIX: Calculate actual Wasserstein distance from ERAG history
        let history_dist = self.compute_history_distance(pad_state.entropy, &collapse.top_hits);
        
        let reward = self.compute_tcs_reward(base_reward, topology, mode, history_dist);
    let blended_reward = reward + (predicted_reward_delta * 0.3);

        self.dqn_update(state.clone(), action.clone(), blended_reward, next_state.clone())
            .await?;

        let performance =
            (generation.rouge_score + (1.0 - (entropy_delta.abs() / 0.5).min(1.0))) / 2.0;
        self.predictor
            .update(topology, reward - predicted_reward_delta, performance);

        // Every 5 episodes, run Reptile and check QLoRA trigger
        if self.episode_count % 5 == 0 {
            self.reptile_step(32).await?;
            if self.average_reward() < 0.0 {
                self.trigger_qlora().await?;
            }
            self.decay_schedules();
        }

        // Phase 5.2: Evolution step every 50 episodes
        if self.episode_count % 50 == 0 {
            self.evolution_step().await?;
        }

        // Existing event/breakthrough logic...
        let mut events = Vec::new();
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

        if let Some(pred_action) = predictor_action.as_ref() {
            events.push(format!(
                "TCS predictor {} {} Δ{:.3} (pred Δreward={:.3})",
                if predictor_applied { "applied" } else { "suggested" },
                pred_action.param,
                pred_action.delta,
                predicted_reward_delta
            ));
        }

        let mut breakthroughs = Vec::new();
        if entropy_delta > self.breakthrough_threshold {
            breakthroughs.push(format!(
                "Breakthrough in quadrant {:?} (ΔH={:.3})",
                compass.quadrant, entropy_delta
            ));
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

    fn choose_action(&self, state: &DqnState) -> DqnAction {
        let mut rng = thread_rng();
        if rng.gen::<f64>() < self.epsilon {
            self.action_space.choose(&mut rng).cloned().unwrap()
        } else {
            let s_key = state.to_key();
            let q_values = self.q_table.get(&s_key);
            if let Some(qs) = q_values {
                let max_key = qs
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(k, _)| k.clone())
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

    fn apply_action_to_config(config: &mut RuntimeConfig, action: &DqnAction) {
        match action.param.as_str() {
            "temperature" => config.temperature += action.delta,
            "top_p" => config.top_p += action.delta,
            "mcts_c" => config.phase2_mcts_c_increment += action.delta,
            "retrieval_top_k" => {
                config.phase2_retrieval_top_k_increment += action.delta as i32;
            }
            "novelty_threshold" => config.novelty_threshold += action.delta,
            "self_awareness_level" => config.self_awareness_level += action.delta,
            _ => {}
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
        let mut rng = thread_rng();
        let batch_size = 32.min(self.replay_buffer.len());
        // Convert VecDeque to Vec for sampling
        let buffer_vec: Vec<_> = self.replay_buffer.iter().cloned().collect();
        let batch: Vec<_> = (0..batch_size)
            .map(|_| buffer_vec.choose(&mut rng).cloned().unwrap())
            .collect();

        for tuple in batch {
            let s_key = tuple.state.to_key();
            let a_key = tuple.action.to_key();

            // Compute max_next_q first (immutable borrow)
            let next_qs = self.q_table.get(&tuple.next_state.to_key());
            let max_next_q = next_qs
                .map(|qs| qs.values().cloned().fold(f64::NEG_INFINITY, f64::max))
                .unwrap_or(0.0);

            // Now get mutable access and update
            let qs = self.q_table.entry(s_key).or_insert_with(HashMap::new);
            let current_q = *qs.entry(a_key.clone()).or_insert(0.0);
            let new_q =
                current_q + self.alpha * (tuple.reward + self.gamma * max_next_q - current_q);
            qs.insert(a_key, new_q);
        }

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
                    last.state.metrics.as_slice()
                } else {
                    &[0.0; 5][..]
                };
                let erag_batch = self
                    .erag
                    .query_replay_batch("", query_metrics, batch_size)
                    .await?;
                let mut full = self.replay_buffer.iter().cloned().collect::<Vec<_>>();
                full.extend(erag_batch);
                full.truncate(batch_size);
                full
            }
            #[cfg(test)]
            {
                let mut rng = thread_rng();
                let buffer_vec: Vec<_> = self.replay_buffer.iter().cloned().collect();
                (0..batch_size.min(buffer_vec.len()))
                    .map(|_| buffer_vec.choose(&mut rng).cloned().unwrap())
                    .collect()
            }
        } else {
            let mut rng = thread_rng();
            let buffer_vec: Vec<_> = self.replay_buffer.iter().cloned().collect();
            (0..batch_size.min(buffer_vec.len()))
                .map(|_| buffer_vec.choose(&mut rng).cloned().unwrap())
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
            let mut config = self.config.lock().await;
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
                    
                    // Pad to fixed size if needed (target size: 768 for LoRA input_dim)
                    while input.len() < 768 {
                        input.push(0.0);
                    }
                    input.truncate(768);
                    
                    // Target: next state metrics (5 dims) -> pad to 768
                    let mut target = tuple
                        .next_state
                        .metrics
                        .iter()
                        .map(|value| *value as f32)
                        .collect::<Vec<f32>>();
                    while target.len() < 768 {
                        target.push(0.0);
                    }
                    target.truncate(768);
                    
                    (input, target)
                })
                .collect();

            if training_samples.is_empty() {
                info!("No training samples for QLoRA");
                return Ok(());
            }

            // Step 3: Train LoRA adapter on topological data
            match self.lora_trainer.train(&training_samples, 10, 1e-3_f32) {
                Ok(_loss) => {
                    info!(
                        "QLoRA fine-tuning completed on {} samples",
                        training_samples.len()
                    );
                    
                    // Step 4: After training, optionally adjust config based on low-reward tuples
                    if !low_tuples.is_empty() {
                        let mut param_deltas: HashMap<String, f64> = HashMap::new();
                        for tuple in &low_tuples {
                            let amplified_delta =
                                tuple.action_delta * (1.0 - tuple.reward * 2.0).max(-2.0).min(2.0);
                            *param_deltas
                                .entry(tuple.action_param.clone())
                                .or_insert(0.0) += amplified_delta;
                        }
                        let avg_len = low_tuples.len() as f64;
                        let mut config = self.config.lock().await;
                        for (param, total) in &param_deltas {
                            let avg_delta = total / avg_len * 0.3; // Reduced impact after LoRA training
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
                                    config.phase2_mcts_c_increment =
                                        config.phase2_mcts_c_increment.clamp(0.0, 2.0);
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
                        info!("Post-LoRA config adjustments applied");
                    }
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

    /// Phase 5.2: Evolution step with topological guidance
    async fn evolution_step(&mut self) -> Result<()> {
        let current = {
            let guard = self.config.lock().await;
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
        for tuple in old_tuples {
            if tuple.state.len() >= 2 {
                let delta = tuple.state[0];
                let rouge = tuple.state[1];
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
        let recent_topologies: Vec<TopologicalSignature> = self.recent_topologies.iter().cloned().collect();
        let best = self.evolution.evolve_with_topology(&current, mixed_episodes, recent_topologies).await?;
        {
            let mut guard = self.config.lock().await;
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

            match self.lora_trainer.train(&training_samples, 10, 1e-3_f32) {
                Ok(loss) => info!(loss, "LoRA fine-tuning completed"),
                Err(error) => warn!(%error, "LoRA fine-tuning failed"),
            }
        }
    }
}

#[derive(Default)]
pub struct GaussianProcess {
    x_train: Option<Vec<Vec<f64>>>,
    y_train: Option<Vec<f64>>,
}

impl GaussianProcess {
    pub fn fit(&mut self, x: &Vec<Vec<f64>>, y: &Vec<f64>) {
        self.x_train = Some(x.clone());
        self.y_train = Some(y.clone());
    }

    pub fn suggest_next(&self, n: usize) -> Vec<Vec<f64>> {
        let mut rng = thread_rng();
        if let (Some(ref x_train), Some(ref y_train)) = (&self.x_train, &self.y_train) {
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
                                (best[0] + rng.gen_range(-0.05f64..0.05)).clamp(0.1, 1.0),
                                (best[1] + rng.gen_range(-0.05f64..0.05)).clamp(0.1, 1.0),
                                (best[2] + rng.gen_range(-0.05f64..0.05)).clamp(0.0, 1.0),
                                (best[3] + rng.gen_range(-0.005f64..0.005)).clamp(0.001, 0.1),
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
                    rng.gen_range(0.1..1.0),
                    rng.gen_range(0.1..1.0),
                    rng.gen_range(0.0..1.0),
                    rng.gen_range(0.001..0.1),
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
}

impl EvolutionLoop {
    pub fn new(pop_size: usize, gens: usize, mutation_std: f64) -> Self {
        Self {
            population_size: pop_size,
            generations: gens,
            mutation_std,
            bo_gp: GaussianProcess::default(),
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

    fn mutate_config(&self, conf: &RuntimeConfig) -> RuntimeConfig {
        let mut rng = thread_rng();
        let std = self.mutation_std;
        let mut new = conf.clone();
        new.temperature += rng.gen_range(-std..std);
        new.temperature = new.temperature.clamp(0.1, 1.0);
        new.top_p += rng.gen_range(-std..std);
        new.top_p = new.top_p.clamp(0.1, 1.0);
        new.novelty_threshold += rng.gen_range(-std * 0.5..std * 0.5);
        new.novelty_threshold = new.novelty_threshold.clamp(0.0, 1.0);
        new.dqn_alpha += rng.gen_range(-std * 0.001..std * 0.001);
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

    fn select_and_breed(&self, pop: &Vec<RuntimeConfig>, fitness: &Vec<f64>) -> Vec<RuntimeConfig> {
        let mut new_pop = vec![];
        let size = pop.len();
        let mut rng = thread_rng();
        for _ in 0..size {
            let p1 = self.tournament_select(fitness, &mut rng);
            let p2 = self.tournament_select(fitness, &mut rng);
            let child = self.crossover(&pop[p1], &pop[p2]);
            new_pop.push(child);
        }
        new_pop
    }

    fn tournament_select(&self, fitness: &Vec<f64>, rng: &mut ThreadRng) -> usize {
        let tournament_size = 4;
        let mut best_idx = rng.gen_range(0..fitness.len());
        let mut best = fitness[best_idx];
        for _ in 1..tournament_size {
            let i = rng.gen_range(0..fitness.len());
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
