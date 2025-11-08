//! Reward computation types for RL Execution Harness

use serde::{Deserialize, Serialize};

/// Execution reward breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionReward {
    /// R_correct: Functional correctness reward (1.0 if tests pass, 0.0 if fail)
    pub functional: f64,
    /// R_CQS: Code Quality Score reward (normalized to [0, 1], higher = better quality)
    pub cqs: f64,
    /// R_topo: Topological quality reward (normalized to [0, 1], higher = better topology)
    pub topological: f64,
    /// R_total: Composite reward = w1·functional + w2·cqs + w3·topological
    pub total: f64,
}

impl ExecutionReward {
    /// Create a reward with all components zero
    pub fn zero() -> Self {
        Self {
            functional: 0.0,
            cqs: 0.0,
            topological: 0.0,
            total: 0.0,
        }
    }
}


