use actix::{Actor, Context, Handler, Message, Arbiter, SyncArbiter, SyncContext};
use anyhow::Result;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing;
use crate::emotional_mapping::EmotionalState;
use crate::compass::CompassResult;

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
        let mut pad = [0.0; 7];

        // Simple emotional analysis based on prompt keywords
        let lower_prompt = msg.prompt.to_lowercase();
        if lower_prompt.contains("frustration") || lower_prompt.contains("grind") {
            pad[0] = -0.5; // Negative pleasure
            pad[1] = 0.7;  // High arousal
            pad[2] = -0.3; // Low dominance
        } else if lower_prompt.contains("despair") {
            pad[0] = -0.8; // Very negative pleasure
            pad[1] = 0.2;  // Low arousal
            pad[2] = -0.6; // Low dominance
        } else if lower_prompt.contains("awakening") || lower_prompt.contains("transcendence") {
            pad[0] = 0.6;  // Positive pleasure
            pad[1] = 0.8;  // High arousal
            pad[2] = 0.4;  // Moderate dominance
        }

        // Add ghosts (VAE-style perturbations)
        use rand::prelude::*;
        let mut rng = thread_rng();
        for i in 3..7 {
            pad[i] = rng.gen_range(-0.2..0.2);
        }

        // Compute entropy
        let entropy = -pad.iter().map(|&p| {
            let prob: f64 = (p + 1.0) / 2.0;
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
        let is_threat = pleasure < 0.0 && arousal > 0.0 && emotional_state.entropy > 1.0;
        let is_healing = pleasure > 0.0 && arousal < 0.0 && emotional_state.entropy < 1.0;

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

        let safe_entropy = emotional_state.entropy.max(1e-6);
        if emotional_state.entropy <= 0.0 {
            tracing::warn!("Non-positive entropy detected in empathy network: {}", emotional_state.entropy);
        }
        let exploration_bonus = self.mcts_c * (safe_entropy.ln() / (1.0 + rng.gen::<f64>())).sqrt();
        let branch = format!("explore_{:.2}_{:.2}",
            emotional_state.pad_vector[0] + exploration_bonus,
            emotional_state.entropy + rng.gen_range(-0.1..0.1)
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
        for i in 0..4 {
            let node_id = format!("heart_{}", i);
            let addr = HeartNode::new(node_id).start();
            heart_nodes.push(addr);
        }

        // Spawn cognitive nodes with SyncArbiter for CPU-bound tasks
        for i in 0..4 {
            let node_id = format!("cognitive_{}", i);
            let mcts_c = 0.5; // Reduced exploration constant for faster computation
            let addr = actix::SyncArbiter::start(1, move || CognitiveNode::new(node_id.clone(), mcts_c));
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