use anyhow::Result;
use std::collections::HashMap;
use rand::prelude::*;
use crate::emotional_mapping::EmotionalState;

#[derive(Debug, Clone)]
pub struct CompassResult {
    pub state: String, // 2-bit encoded state
    pub is_threat: bool,
    pub is_healing: bool,
    pub mcts_branches: Vec<String>,
}

pub struct CompassEngine {
    mcts_c: f64, // Exploration constant
}

impl CompassEngine {
    pub fn new(mcts_c: f64) -> Result<Self> {
        Ok(Self { mcts_c })
    }

    pub async fn process_state(&self, emotional_state: &EmotionalState) -> Result<CompassResult> {
        // 2-bit state encoding from PAD
        let pleasure = emotional_state.pad_vector[0];
        let arousal = emotional_state.pad_vector[1];
        let dominance = emotional_state.pad_vector[2];

        // Thresholds based on emotional quadrants
        let is_threat = pleasure < -0.3 && arousal > 0.3;
        let is_healing = pleasure > 0.3 && arousal < -0.3;

        // 2-bit encoding: [threat_bit][healing_bit]
        let state_bits = match (is_threat, is_healing) {
            (true, false) => "10",  // Threat
            (false, true) => "01",  // Healing
            (true, true) => "11",   // Mixed
            (false, false) => "00", // Neutral
        };

        // MCTS exploration with 3 branches
        let branches = self.mcts_explore(emotional_state, 3).await?;

        Ok(CompassResult {
            state: state_bits.to_string(),
            is_threat,
            is_healing,
            mcts_branches: branches,
        })
    }

    async fn mcts_explore(&self, state: &EmotionalState, num_branches: usize) -> Result<Vec<String>> {
        let mut branches = Vec::new();
        let mut rng = thread_rng();

        for _ in 0..num_branches {
            // UCB1 selection for exploration
            let exploration_bonus = self.mcts_c * (state.entropy.ln() / (1.0 + rng.gen::<f64>())).sqrt();

            // Generate branch based on emotional state + exploration
            let branch = format!("explore_{:.2}_{:.2}",
                state.pad_vector[0] + exploration_bonus,
                state.entropy + rng.gen_range(-0.1..0.1)
            );
            branches.push(branch);
        }

        Ok(branches)
    }
}
