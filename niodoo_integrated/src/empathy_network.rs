use actix::{Actor, Context, Handler, Message, Arbiter, SyncArbiter, SyncContext};
use anyhow::Result;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing;
use crate::emotional_mapping::EmotionalState;
use crate::compass::CompassResult;

// Configuration constants
const PAD_DIMENSION: usize = 7;
const GHOST_INDICES_START: usize = 3;
const GHOST_INDICES_END: usize = 7;
const GHOST_NOISE_RANGE: f64 = 0.2;
const PROBABILITY_NORMALIZATION_OFFSET: f64 = 1.0;
const PROBABILITY_NORMALIZATION_DIVISOR: f64 = 2.0;
const ENTROPY_THRESHOLD: f64 = 1.0;
const MIN_ENTROPY_FOR_SAFETY: f64 = 1e-6;
const ENTROPY_EXPLORATION_RANGE: f64 = 0.1;
const NODE_COUNT: usize = 4;
const DEFAULT_MCTS_C: f64 = 0.5;
const ACTOR_WORKER_COUNT: usize = 1;

// PAD emotional values
const FRUSTRATION_PLEASURE: f64 = -0.5;
const FRUSTRATION_AROUSAL: f64 = 0.7;
const FRUSTRATION_DOMINANCE: f64 = -0.3;
const DESPAIR_PLEASURE: f64 = -0.8;
const DESPAIR_AROUSAL: f64 = 0.2;
const DESPAIR_DOMINANCE: f64 = -0.6;
const AWAKENING_PLEASURE: f64 = 0.6;
const AWAKENING_AROUSAL: f64 = 0.8;
const AWAKENING_DOMINANCE: f64 = 0.4;

#[derive(Message)]
#[rtype(result = "Result<EmotionalState>")]
pub struct ProcessSocialInput {
    pub prompt: Arc<str>,
    pub embedding: Arc<Vec<f64>>,
}

#[derive(Message)]
#[rtype(result = "Result<CompassResult>")]
pub struct ProcessCognitiveInput {
    pub emotional_state: Arc<EmotionalState>,
}

pub struct HeartNode {
    node_id: String,
}

impl HeartNode {
    pub fn new(node_id: String) -> Self {
        Self { node_id }
    }
}

impl Actor for HeartNode {
    type Context = Context<Self>;
}

impl Handler<ProcessSocialInput> for HeartNode {
    type Result = Result<EmotionalState>;

    fn handle(&mut self, msg: ProcessSocialInput, _ctx: &mut Context<Self>) -> Self::Result {
        // Simulate emotional processing with some async work
        // In real implementation, this would do actual emotional mapping
        let mut pad = [0.0; PAD_DIMENSION];

        // Simple emotional analysis based on prompt keywords
        let lower_prompt = msg.prompt.to_lowercase();
        if lower_prompt.contains("frustration") || lower_prompt.contains("grind") {
            pad[0] = FRUSTRATION_PLEASURE;
            pad[1] = FRUSTRATION_AROUSAL;
            pad[2] = FRUSTRATION_DOMINANCE;
        } else if lower_prompt.contains("despair") {
            pad[0] = DESPAIR_PLEASURE;
            pad[1] = DESPAIR_AROUSAL;
            pad[2] = DESPAIR_DOMINANCE;
        } else if lower_prompt.contains("awakening") || lower_prompt.contains("transcendence") {
            pad[0] = AWAKENING_PLEASURE;
            pad[1] = AWAKENING_AROUSAL;
            pad[2] = AWAKENING_DOMINANCE;
        }

        // Add ghosts (VAE-style perturbations)
        use rand::prelude::*;
        let mut rng = thread_rng();
        for i in GHOST_INDICES_START..GHOST_INDICES_END {
            pad[i] = rng.gen_range(-GHOST_NOISE_RANGE..GHOST_NOISE_RANGE);
        }

        // Compute entropy
        let entropy = -pad.iter().map(|&p| {
            let prob: f64 = (p + PROBABILITY_NORMALIZATION_OFFSET) / PROBABILITY_NORMALIZATION_DIVISOR;
            if prob > 0.0 { prob * prob.ln() } else { 0.0 }
        }).sum::<f64>();

        Ok(EmotionalState {
            pad_vector: pad,
            entropy,
        })
    }
}

pub struct CognitiveNode {
    node_id: String,
    mcts_c: f64,
}

impl CognitiveNode {
    pub fn new(node_id: String, mcts_c: f64) -> Self {
        Self { node_id, mcts_c }
    }
}

impl Actor for CognitiveNode {
    type Context = SyncContext<Self>;
}

impl Handler<ProcessCognitiveInput> for CognitiveNode {
    type Result = Result<CompassResult>;

    fn handle(&mut self, msg: ProcessCognitiveInput, _ctx: &mut SyncContext<Self>) -> Self::Result {
        let emotional_state = &*msg.emotional_state;

        // 2-bit state encoding from PAD
        let pleasure = emotional_state.pad_vector[0];
        let arousal = emotional_state.pad_vector[1];

        // Threat/healing detection
        let is_threat = pleasure < 0.0 && arousal > 0.0 && emotional_state.entropy > ENTROPY_THRESHOLD;
        let is_healing = pleasure > 0.0 && arousal < 0.0 && emotional_state.entropy < ENTROPY_THRESHOLD;

        let state_bits = match (is_threat, is_healing) {
            (true, false) => "10",
            (false, true) => "01",
            (true, true) => "11",
            (false, false) => "00",
        };

        // MCTS exploration
        let mut branches = Vec::new();
        use rand::prelude::*;
        let mut rng = thread_rng();

        let safe_entropy = emotional_state.entropy.max(MIN_ENTROPY_FOR_SAFETY);
        if emotional_state.entropy <= 0.0 {
            tracing::warn!("Non-positive entropy detected in empathy network: {}", emotional_state.entropy);
        }
        let exploration_bonus = self.mcts_c * (safe_entropy.ln() / (1.0 + rng.gen::<f64>())).sqrt();
        let branch = format!("explore_{:.2}_{:.2}",
            emotional_state.pad_vector[0] + exploration_bonus,
            emotional_state.entropy + rng.gen_range(-ENTROPY_EXPLORATION_RANGE..ENTROPY_EXPLORATION_RANGE)
        );
        branches.push(branch);

        Ok(CompassResult {
            state: state_bits.to_string(),
            is_threat,
            is_healing,
            mcts_branches: branches,
        })
    }
}

#[derive(Debug)]
pub struct CompleteEmpathyNetwork {
    heart_nodes: Vec<actix::Addr<HeartNode>>,
    cognitive_nodes: Vec<actix::Addr<CognitiveNode>>,
    current_heart: AtomicUsize,
    current_cognitive: AtomicUsize,
}

impl CompleteEmpathyNetwork {
    pub fn new() -> Result<Self> {
        let mut heart_nodes = Vec::new();
        let mut cognitive_nodes = Vec::new();

        // Spawn heart nodes using Arbiter
        for i in 0..NODE_COUNT {
            let node_id = format!("heart_{}", i);
            let addr = HeartNode::new(node_id).start();
            heart_nodes.push(addr);
        }

        // Spawn cognitive nodes with SyncArbiter for CPU-bound tasks
        for i in 0..NODE_COUNT {
            let node_id = format!("cognitive_{}", i);
            let addr = actix::SyncArbiter::start(ACTOR_WORKER_COUNT, move || CognitiveNode::new(node_id.clone(), DEFAULT_MCTS_C));
            cognitive_nodes.push(addr);
        }

        Ok(Self {
            heart_nodes,
            cognitive_nodes,
            current_heart: AtomicUsize::new(0),
            current_cognitive: AtomicUsize::new(0),
        })
    }

    pub async fn process_social_input(&self, prompt: String, embedding: Vec<f64>) -> Result<EmotionalState> {
        let msg = ProcessSocialInput { prompt: Arc::from(prompt), embedding: Arc::new(embedding) };

        // Round-robin load balancing across heart nodes
        let prev = self.current_heart.fetch_add(1, Ordering::Relaxed);
        let idx = prev % self.heart_nodes.len();
        let addr = &self.heart_nodes[idx];

        addr.send(msg).await?
    }

    pub async fn process_cognitive_input(&self, emotional_state: EmotionalState) -> Result<CompassResult> {
        let msg = ProcessCognitiveInput { emotional_state: Arc::new(emotional_state) };

        // Round-robin load balancing across cognitive nodes
        let prev = self.current_cognitive.fetch_add(1, Ordering::Relaxed);
        let idx = prev % self.cognitive_nodes.len();
        let addr = &self.cognitive_nodes[idx];

        addr.send(msg).await?
    }

    pub fn get_node_count(&self) -> (usize, usize) {
        (self.heart_nodes.len(), self.cognitive_nodes.len())
    }
}

impl Drop for CompleteEmpathyNetwork {
    fn drop(&mut self) {
        // Graceful shutdown of actors would happen here
        // In Actix, actors are automatically stopped when Addr is dropped
    }
}