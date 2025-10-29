use anyhow::Result;
use std::collections::HashMap;
use rand::prelude::*;
use tokio::time::{sleep, Duration};
use tracing;
use crate::emotional_mapping::EmotionalState;

// Configuration constants
const COMPASS_PROCESSING_DELAY_MS: u64 = 4;
const MCTS_DELAY_MS: u64 = 3;
const MCTS_BRANCHES: usize = 3;
const THREAT_PLEASURE_THRESHOLD: f64 = -0.2;
const THREAT_AROUSAL_THRESHOLD: f64 = 0.2;
const THREAT_ENTROPY_THRESHOLD: f64 = 2.0;
const HEALING_PLEASURE_THRESHOLD: f64 = 0.2;
const HEALING_AROUSAL_THRESHOLD: f64 = -0.2;
const HEALING_ENTROPY_THRESHOLD: f64 = 3.0;
const MIN_ENTROPY_FOR_SAFETY: f64 = 1e-6;
const ENTROPY_EXPLORATION_RANGE: f64 = 0.1;

#[derive(Debug, Clone)]
pub struct CompassResult {
    pub state: String, // 2-bit encoded state
    pub is_threat: bool,
    pub is_healing: bool,
    pub mcts_branches: Vec<String>,
}

#[derive(Debug)]
pub struct CompassEngine {
    mcts_c: f64, // Exploration constant
}

impl CompassEngine {
    pub fn new(mcts_c: f64) -> Result<Self> {
        Ok(Self { mcts_c })
    }

    pub async fn process_state(&self, emotional_state: &EmotionalState) -> Result<CompassResult> {
        // Simulate compass processing time
        sleep(Duration::from_millis(COMPASS_PROCESSING_DELAY_MS)).await;

        // 2-bit state encoding from PAD
        let pleasure = emotional_state.pad_vector[0];
        let arousal = emotional_state.pad_vector[1];
        let dominance = emotional_state.pad_vector[2];

        // Dynamic threat/healing detection based on entropy and PAD
        let is_threat = pleasure < THREAT_PLEASURE_THRESHOLD 
            && arousal > THREAT_AROUSAL_THRESHOLD 
            && emotional_state.entropy > THREAT_ENTROPY_THRESHOLD;
        let is_healing = pleasure > HEALING_PLEASURE_THRESHOLD 
            && arousal < HEALING_AROUSAL_THRESHOLD 
            && emotional_state.entropy < HEALING_ENTROPY_THRESHOLD;

        // 2-bit encoding: [threat_bit][healing_bit]
        let state_bits = match (is_threat, is_healing) {
            (true, false) => "10",  // Threat
            (false, true) => "01",  // Healing
            (true, true) => "11",   // Mixed
            (false, false) => "00", // Neutral
        };

        // MCTS exploration with configured number of branches
        let branches = self.mcts_explore(emotional_state, MCTS_BRANCHES).await?;

        Ok(CompassResult {
            state: state_bits.to_string(),
            is_threat,
            is_healing,
            mcts_branches: branches,
        })
    }

    async fn mcts_explore(&self, state: &EmotionalState, num_branches: usize) -> Result<Vec<String>> {
        // Simulate MCTS computation time
        sleep(Duration::from_millis(MCTS_DELAY_MS)).await;

        let mut branches = Vec::new();
        let mut rng = thread_rng();

        for _ in 0..num_branches {
            // UCB1 selection for exploration
            let safe_entropy = state.entropy.max(MIN_ENTROPY_FOR_SAFETY);
            if state.entropy <= 0.0 {
                tracing::warn!("Non-positive entropy detected in mcts_explore: {}", state.entropy);
            }
            let exploration_bonus = self.mcts_c * (safe_entropy.ln() / (1.0 + rng.r#gen::<f64>())).sqrt();

            // Generate branch based on emotional state + exploration
            let branch = format!("explore_{:.2}_{:.2}",
                state.pad_vector[0] + exploration_bonus,
                state.entropy + rng.gen_range(-ENTROPY_EXPLORATION_RANGE..ENTROPY_EXPLORATION_RANGE)
            );
            branches.push(branch);
        }

        Ok(branches)
    }
}
