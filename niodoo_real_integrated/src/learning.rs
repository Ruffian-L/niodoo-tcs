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
    pub param: String,  // e.g., "temperature"
    pub delta: f64,     // e.g., 0.1 or -0.1
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
    replay_buffer: Vec<ReplayTuple>,
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
            DqnAction { param: "novelty_threshold".to_string(), delta: -0.1 },
            DqnAction { param: "novelty_threshold".to_string(), delta: 0.1 },
            DqnAction { param: "self_awareness_level".to_string(), delta: -0.1 },
            DqnAction { param: "self_awareness_level".to_string(), delta: 0.1 },
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
            initial_epsilon: epsilon,
            initial_alpha: alpha,
        }
    }

    pub async fn update(
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
                "novelty_threshold" => config.novelty_threshold += action.delta,
                "self_awareness_level" => config.self_awareness_level += action.delta,
                _ => {}
            }
        }

        // Simulate next state (estimate based on action)
        let next_state = self.estimate_next_state(&state, &action);

        let reward = self.compute_reward(entropy_delta, generation.rouge_score);

        self.dqn_update(state.clone(), action.clone(), reward, next_state.clone()).await?;

        // Every 5 episodes, run Reptile and check QLoRA trigger
        if self.episode_count % 5 == 0 {
            self.reptile_step(32).await?;
            if self.average_reward() < 0.0 {
                self.trigger_qlora().await?;
            }
            self.decay_schedules();
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

    async fn dqn_update(
        &mut self,
        state: DqnState,
        action: DqnAction,
        reward: f64,
        next_state: DqnState,
    ) -> Result<()> {
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
            let a_key = tuple.action.to_key();
            
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

        #[cfg(not(test))]
        {
            let tuple = DqnTuple {
                state: state.metrics.clone(),
                action_param: action.param.clone(),
                action_delta: action.delta,
                reward,
                next_state: next_state.metrics.clone(),
            };
            if let Err(e) = self.erag.store_dqn_tuple(&tuple).await {
                tracing::warn!("Failed to store DQN tuple: {}", e);
            }
        }
        Ok(())
    }

    fn estimate_next_state(&self, state: &DqnState, action: &DqnAction) -> DqnState {
        // Estimate metric changes based on action
        let mut new_metrics = state.metrics.clone();
        match action.param.as_str() {
            "temperature" => new_metrics[0] += action.delta * 0.05, // affect entropy
            "top_p" => new_metrics[1] += action.delta * 0.1,       // affect rouge
            "mcts_c" => new_metrics[3] += action.delta * 0.1,     // affect ucb1
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
                let query_metrics = if let Some(last) = self.replay_buffer.last() {
                    last.state.metrics.as_slice()
                } else {
                    &[0.0; 5][..]
                };
                let erag_batch = self.erag.query_replay_batch("", query_metrics, batch_size).await?;
                let mut full = self.replay_buffer.iter().cloned().collect::<Vec<_>>();
                full.extend(erag_batch);
                full.truncate(batch_size);
                full
            }
            #[cfg(test)]
            {
                let mut rng = thread_rng();
                (0..batch_size.min(self.replay_buffer.len())).map(|_| self.replay_buffer.choose(&mut rng).cloned().unwrap()).collect()
            }
        } else {
            let mut rng = thread_rng();
            (0..batch_size.min(self.replay_buffer.len())).map(|_| self.replay_buffer.choose(&mut rng).cloned().unwrap()).collect()
        };

        let mut param_deltas = HashMap::new();

        for tuple in batch {
            let delta = tuple.action.delta * 0.01; // Inner gradient
            *param_deltas.entry(tuple.action.param.clone()).or_insert(0.0) += delta;
        }

        // Outer meta-update: average deltas and apply to config
        let mut config = self.config.lock().unwrap();
        for (param, total_delta) in param_deltas {
            let avg_delta = total_delta / batch.len() as f64;
            match param.as_str() {
                "temperature" => config.temperature += avg_delta; config.temperature = config.temperature.clamp(0.1, 1.0),
                "top_p" => config.top_p += avg_delta; config.top_p = config.top_p.clamp(0.1, 1.0),
                "mcts_c" => config.phase2_mcts_c_increment += avg_delta; config.phase2_mcts_c_increment = config.phase2_mcts_c_increment.clamp(0.0, 2.0),
                "retrieval_top_k" => config.phase2_retrieval_top_k_increment += avg_delta as i32; let new_k = (config.phase2_retrieval_top_k_increment as f64 + avg_delta).clamp(0.0, 10.0) as i32; config.phase2_retrieval_top_k_increment = new_k,
                "novelty_threshold" => config.novelty_threshold += avg_delta; config.novelty_threshold = config.novelty_threshold.clamp(0.0, 1.0),
                "self_awareness_level" => config.self_awareness_level += avg_delta; config.self_awareness_level = config.self_awareness_level.clamp(0.0, 1.0),
                _ => {}
            }
        }
        info!("Reptile meta-update applied");
        Ok(())
    }

    async fn trigger_qlora(&mut self) -> Result<()> {
        #[cfg(not(test))]
        {
            let low_tuples = self.erag.query_low_reward_tuples(-0.5, 16).await?;
            if low_tuples.is_empty() {
                info!("No low-reward tuples for QLoRA");
                return Ok(());
            }
            let mut param_deltas: HashMap<String, f64> = HashMap::new();
            for tuple in &low_tuples {
                let amplified_delta = tuple.action_delta * (1.0 - tuple.reward * 2.0).max(-2.0).min(2.0);
                *param_deltas.entry(tuple.action_param.clone()).or_insert(0.0) += amplified_delta;
            }
            let avg_len = low_tuples.len() as f64;
            let mut config = self.config.lock().unwrap();
            for (param, total) in param_deltas {
                let avg_delta = total / avg_len * 0.5; // conservative
                match param.as_str() {
                    "temperature" => config.temperature += avg_delta; config.temperature = config.temperature.clamp(0.1, 1.0),
                    "top_p" => config.top_p += avg_delta; config.top_p = config.top_p.clamp(0.1, 1.0),
                    "mcts_c" => config.phase2_mcts_c_increment += avg_delta; config.phase2_mcts_c_increment = config.phase2_mcts_c_increment.clamp(0.0, 2.0),
                    "retrieval_top_k" => config.phase2_retrieval_top_k_increment += (avg_delta * 0.5) as i32; let new_k = (config.phase2_retrieval_top_k_increment as f64 + avg_delta).clamp(0.0, 10.0) as i32; config.phase2_retrieval_top_k_increment = new_k,
                    "novelty_threshold" => config.novelty_threshold += avg_delta; config.novelty_threshold = config.novelty_threshold.clamp(0.0, 1.0),
                    "self_awareness_level" => config.self_awareness_level += avg_delta; config.self_awareness_level = config.self_awareness_level.clamp(0.0, 1.0),
                    _ => {},
                }
            }
            info!("QLoRA fine-tuning simulated: adjusted {} params from {} low-reward tuples", param_deltas.len(), low_tuples.len());
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
        self.replay_buffer
            .iter()
            .map(|t| t.reward)
            .sum::<f64>()
            / self.replay_buffer.len() as f64
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
}
