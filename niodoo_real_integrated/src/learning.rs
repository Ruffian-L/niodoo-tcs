use std::collections::VecDeque;
use std::time::SystemTime;

use anyhow::Result;
use tracing::info;

use crate::compass::CompassOutcome;
use crate::erag::CollapseResult;
use crate::generation::GenerationResult;
use crate::torus::PadGhostState;

#[derive(Debug, Clone)]
pub struct LearningOutcome {
    pub events: Vec<String>,
    pub breakthroughs: Vec<String>,
    pub qlora_updates: Vec<String>,
    pub entropy_delta: f64,
}

pub struct LearningLoop {
    entropy_history: VecDeque<f64>,
    window: usize,
    breakthrough_threshold: f64,
}

impl LearningLoop {
    pub fn new(window: usize, breakthrough_threshold: f64) -> Self {
        Self {
            entropy_history: VecDeque::with_capacity(window),
            window,
            breakthrough_threshold,
        }
    }

    pub fn update(
        &mut self,
        pad_state: &PadGhostState,
        compass: &CompassOutcome,
        collapse: &CollapseResult,
        generation: &GenerationResult,
    ) -> Result<LearningOutcome> {
        let previous_entropy = self
            .entropy_history
            .back()
            .copied()
            .unwrap_or(pad_state.entropy);
        self.record_entropy(pad_state.entropy);
        let entropy_delta = pad_state.entropy - previous_entropy;

        let mut events = Vec::new();
        if entropy_delta.abs() > 0.15 {
            events.push(format!(
                "Entropy shift detected: previous={:.3} current={:.3} delta={:.3}",
                previous_entropy, pad_state.entropy, entropy_delta
            ));
        }

        let mut qlora_updates = Vec::new();
        if pad_state.entropy > previous_entropy {
            qlora_updates.push(format!(
                "Retain high-entropy trajectory via QLoRA (delta={:.3})",
                entropy_delta
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
                "Breakthrough recognised in quadrant {:?} at {:?} (ΔH={:.3})",
                compass.quadrant,
                SystemTime::now(),
                entropy_delta
            ));
        }

        info!(
            entropy = pad_state.entropy,
            entropy_delta,
            rouge = generation.rouge_to_baseline,
            quadrant = ?compass.quadrant,
            "learning loop updated",
        );

        Ok(LearningOutcome {
            events,
            breakthroughs,
            qlora_updates,
            entropy_delta,
        })
    }

    fn record_entropy(&mut self, value: f64) {
        if self.entropy_history.len() == self.window {
            self.entropy_history.pop_front();
        }
        self.entropy_history.push_back(value);
    }
}

#[derive(Clone)]
pub struct DqnState {
    pub params: Vec<f64>, // e.g., [novelty_threshold, self_awareness_level]
}

#[derive(Clone)]
pub struct DqnAction {
    pub adjustments: Vec<f64>, // deltas for each param
}

impl LearningLoop {
    pub fn compute_reward(&self, delta: f64, rouge: f64) -> f64 {
        -delta + rouge
    }
    // Stub for DQN update
    pub fn dqn_update(
        &mut self,
        _state: DqnState,
        _action: DqnAction,
        _reward: f64,
        _next_state: DqnState,
    ) {
        // TODO: Replay buffer push, target net, etc.
    }
}
