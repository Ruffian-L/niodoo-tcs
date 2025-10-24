use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use rand::prelude::*;
use tracing::info;

use crate::compass::CompassOutcome;
use crate::config::RuntimeConfig;
use crate::erag::{CollapseResult, EragClient};
use crate::generation::GenerationResult;
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

impl std::hash::Hash for DqnState {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.to_key().hash(state);
    }
}

impl PartialEq for DqnState {
    fn eq(&self, other: &Self) -> bool {
        self.to_key() == other.to_key()
    }
}

impl Eq for DqnState {}

#[derive(Clone, Debug)]
pub struct DqnAction {
    pub param: String,  // e.g., "temperature"
    pub delta: f64,     // e.g., 0.1 or -0.1
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
    replay_buffer: Vec<ReplayTuple>,
    q_table: HashMap<String, HashMap<String, f64>>, // state_key -> (action_key -> q_value)
    action_space: Vec<DqnAction>,
    epsilon: f64,
    gamma: f64, // discount
    alpha: f64, // learning rate
    erag: Arc<EragClient>,
    config: Arc<Mutex<RuntimeConfig>>,
    episode_count: u32,
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
        ];
        Self {
            entropy_history: VecDeque::with_capacity(window),
            window,
            breakthrough_threshold,
            replay_buffer: Vec::new(),
            q_table: HashMap::new(),
            action_space,
            epsilon,
            gamma,
            alpha,
            erag,
            config,
            episode_count: 0,
        }
    }

    pub fn update(
        &mut self,
        pad_state: &PadGhostState,
        compass: &CompassOutcome,
        collapse: &CollapseResult,
        generation: &GenerationResult,
    ) -> Result<LearningOutcome> {
        self.episode_count += 1;

        let previous_entropy = self
            .entropy_history
            .back()
            .copied()
            .unwrap_or(pad_state.entropy);
        self.record_entropy(pad_state.entropy);
        let entropy_delta = pad_state.entropy - previous_entropy;

        let state = DqnState::from_metrics(
            entropy_delta,
            generation.rouge_score,
            generation.latency_ms,
            compass.ucb1_score.unwrap_or(0.5) as f64,
            collapse.curator_quality.unwrap_or(0.5) as f64,
        );

        let action = self.choose_action(&state);

        // Apply action to config (mutable access)
        {
            let mut config = self.config.lock().unwrap();
            match action.param.as_str() {
                "temperature" => config.temperature += action.delta,
                "top_p" => config.top_p += action.delta,
                "mcts_c" => config.phase2_mcts_c_increment += action.delta,
                "retrieval_top_k" => {
                    config.phase2_retrieval_top_k_increment += action.delta as i32
                }
                _ => {}
            }
        }

        // Simulate next state (estimate based on action)
        let next_state = self.estimate_next_state(&state, &action);

        let reward = self.compute_reward(entropy_delta, generation.rouge_score);

        self.dqn_update(state.clone(), action.clone(), reward, next_state.clone());

        // Every 5 episodes, run Reptile and check QLoRA trigger
        if self.episode_count % 5 == 0 {
            self.reptile_step(32); // Batch size
            if self.average_reward() < 0.0 {
                self.trigger_qlora();
            }
            self.decay_epsilon();
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

        let mut breakthroughs = Vec::new();
        if entropy_delta > self.breakthrough_threshold {
            breakthroughs.push(format!(
                "Breakthrough in quadrant {:?} (ΔH={:.3})",
                compass.quadrant, entropy_delta
            ));
        }

        let mut qlora_updates = Vec::new();
        if pad_state.entropy > previous_entropy {
            qlora_updates.push(format!(
                "High-entropy retain (delta={:.3})",
                entropy_delta
            ));
        }

        info!(
            entropy = pad_state.entropy,
            entropy_delta,
            rouge = generation.rouge_score,
            quadrant = ?compass.quadrant,
            "learning loop updated"
        );

        Ok(LearningOutcome {
            events,
            breakthroughs,
            qlora_updates,
            entropy_delta,
            adjusted_params: HashMap::from([(action.param, action.delta)]),
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
                    .map(|(k, _)| k)
                    .unwrap();
                self.action_space
                    .iter()
                    .find(|a| a.param == *max_key)
                    .cloned()
                    .unwrap_or_else(|| self.action_space[0].clone())
            } else {
                self.action_space[0].clone()
            }
        }
    }

    fn dqn_update(
        &mut self,
        state: DqnState,
        action: DqnAction,
        reward: f64,
        next_state: DqnState,
    ) {
        // Add to replay buffer
        self.replay_buffer.push(ReplayTuple {
            state: state.clone(),
            action: action.clone(),
            reward,
            next_state: next_state.clone(),
        });
        if self.replay_buffer.len() > 1000 {
            self.replay_buffer.remove(0);
        }

        // Sample random batch for learning
        let mut rng = thread_rng();
        let batch_size = 32.min(self.replay_buffer.len());
        let batch: Vec<_> = (0..batch_size)
            .map(|_| {
                self.replay_buffer
                    .choose(&mut rng)
                    .cloned()
                    .unwrap()
            })
            .collect();

        for tuple in batch {
            let s_key = tuple.state.to_key();
            let a_key = tuple.action.param.clone();
            
            // Compute max_next_q first (immutable borrow)
            let next_qs = self.q_table.get(&tuple.next_state.to_key());
            let max_next_q = next_qs
                .map(|qs| qs.values().cloned().fold(f64::NEG_INFINITY, f64::max))
                .unwrap_or(0.0);
            
            // Now get mutable access and update
            let qs = self.q_table.entry(s_key).or_insert_with(HashMap::new);
            let current_q = *qs.entry(a_key.clone()).or_insert(0.0);
            let new_q = current_q + self.alpha * (tuple.reward + self.gamma * max_next_q - current_q);
            qs.insert(a_key, new_q);
        }

        // Store in ERAG for long-term (async stub)
        // Note: In real impl, this would await erag.store_dqn_tuple
        info!("DQN tuple stored in replay buffer");
    }

    fn estimate_next_state(&self, state: &DqnState, action: &DqnAction) -> DqnState {
        // Estimate metric changes based on action
        let mut new_metrics = state.metrics.clone();
        match action.param.as_str() {
            "temperature" => new_metrics[0] += action.delta * 0.05, // affect entropy
            "top_p" => new_metrics[1] += action.delta * 0.1,       // affect rouge
            "mcts_c" => new_metrics[3] += action.delta * 0.1,     // affect ucb1
            "retrieval_top_k" => new_metrics[4] += action.delta * 0.01, // affect curator
            _ => {}
        }
        DqnState {
            metrics: new_metrics,
        }
    }

    fn reptile_step(&mut self, batch_size: usize) {
        // Sample batch from replay
        let mut rng = thread_rng();
        let batch: Vec<_> = (0..batch_size.min(self.replay_buffer.len()))
            .map(|_| self.replay_buffer.choose(&mut rng).cloned().unwrap())
            .collect();

        let mut param_deltas = HashMap::new();

        for tuple in batch {
            // Inner loop simulation (compute delta)
            let delta = tuple.action.delta * 0.01; // Inner gradient
            *param_deltas.entry(tuple.action.param).or_insert(0.0) += delta;
        }

        // Outer meta-update: average deltas and apply to config
        let mut config = self.config.lock().unwrap();
        for (param, total_delta) in param_deltas {
            let avg_delta = total_delta / batch_size as f64;
            match param.as_str() {
                "temperature" => config.temperature += avg_delta,
                "top_p" => config.top_p += avg_delta,
                "mcts_c" => config.phase2_mcts_c_increment += avg_delta,
                "retrieval_top_k" => {
                    config.phase2_retrieval_top_k_increment += avg_delta as i32
                }
                _ => {}
            }
        }
        info!("Reptile meta-update applied");
    }

    fn trigger_qlora(&self) {
        // Stub: Trigger QLoRA fine-tuning
        info!("QLoRA triggered due to low reward - fine-tuning model");
        // In full impl: call QLoRA on recent low-reward episodes from ERAG
    }

    fn average_reward(&self) -> f64 {
        if self.replay_buffer.is_empty() {
            return 0.0;
        }
        self.replay_buffer
            .iter()
            .map(|t| t.reward)
            .sum::<f64>()
            / self.replay_buffer.len() as f64
    }

    fn decay_epsilon(&mut self) {
        self.epsilon *= 0.995;
        self.epsilon = self.epsilon.max(0.01);
    }
}
