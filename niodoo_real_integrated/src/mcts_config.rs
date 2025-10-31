//! MCTS Configuration and Tuning Parameters
//!
//! This module provides configuration structures for tuning MCTS search behavior,
//! including iteration caps, time budgets, exploration parameters, and reward shaping.

use serde::{Deserialize, Serialize};

/// Configuration for MCTS search parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MctsConfig {
    /// Maximum number of simulations per search
    pub max_simulations: usize,

    /// Maximum time budget in milliseconds
    pub max_time_ms: u64,

    /// Exploration constant for UCB1 (typically sqrt(2) ≈ 1.414)
    pub exploration_c: f64,

    /// Reward shaping parameters
    pub reward_shaping: RewardShaping,

    /// Depth limits for search tree
    pub depth_limits: DepthLimits,
}

/// Reward shaping configuration for tuning MCTS exploration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardShaping {
    /// Bonus reward for exploration actions
    pub exploration_bonus: f64,

    /// Penalty for excessive depth
    pub depth_penalty: f64,

    /// Reward scaling factor
    pub reward_scale: f64,

    /// Discount factor for future rewards
    pub discount_factor: f64,
}

/// Depth limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthLimits {
    /// Maximum allowed depth
    pub max_depth: usize,

    /// Depth at which to start pruning
    pub prune_depth: usize,

    /// Whether to use iterative deepening
    pub iterative_deepening: bool,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            max_simulations: 100,
            max_time_ms: 50,
            exploration_c: 1.414, // sqrt(2)
            reward_shaping: RewardShaping::default(),
            depth_limits: DepthLimits::default(),
        }
    }
}

impl Default for RewardShaping {
    fn default() -> Self {
        Self {
            exploration_bonus: 0.1,
            depth_penalty: 0.05,
            reward_scale: 1.0,
            discount_factor: 0.9,
        }
    }
}

impl Default for DepthLimits {
    fn default() -> Self {
        Self {
            max_depth: 10,
            prune_depth: 8,
            iterative_deepening: false,
        }
    }
}

/// Adaptive MCTS configuration that adjusts based on runtime conditions
#[derive(Debug, Clone)]
pub struct AdaptiveMctsConfig {
    base_config: MctsConfig,
    current_iterations: usize,
    avg_time_ms: f64,
}

impl AdaptiveMctsConfig {
    pub fn new(base_config: MctsConfig) -> Self {
        Self {
            base_config,
            current_iterations: 0,
            avg_time_ms: 0.0,
        }
    }

    /// Update configuration based on recent performance
    pub fn adapt(&mut self, elapsed_ms: u64, iterations: usize) {
        self.current_iterations = iterations;
        self.avg_time_ms = (self.avg_time_ms * 0.9) + (elapsed_ms as f64 * 0.1);

        // If average time is too high, reduce max simulations
        if self.avg_time_ms > self.base_config.max_time_ms as f64 * 0.8 {
            self.base_config.max_simulations =
                (self.base_config.max_simulations as f64 * 0.9) as usize;
        }

        // If average time is low, we can afford more simulations
        if self.avg_time_ms < self.base_config.max_time_ms as f64 * 0.3 {
            self.base_config.max_simulations =
                ((self.base_config.max_simulations as f64 * 1.1) as usize).min(200);
        }
    }

    pub fn config(&self) -> &MctsConfig {
        &self.base_config
    }
}

/// Performance profile for MCTS tuning
#[derive(Debug, Clone, Copy)]
pub enum MctsProfile {
    /// Fast profile: low iterations, short time budget
    Fast,

    /// Balanced profile: moderate iterations and time
    Balanced,

    /// Thorough profile: high iterations, longer time budget
    Thorough,
}

impl MctsProfile {
    pub fn to_config(self) -> MctsConfig {
        match self {
            MctsProfile::Fast => MctsConfig {
                max_simulations: 25,
                max_time_ms: 20,
                exploration_c: 1.2,
                reward_shaping: RewardShaping::default(),
                depth_limits: DepthLimits {
                    max_depth: 5,
                    prune_depth: 4,
                    iterative_deepening: false,
                },
            },
            MctsProfile::Balanced => MctsConfig::default(),
            MctsProfile::Thorough => MctsConfig {
                max_simulations: 200,
                max_time_ms: 100,
                exploration_c: 1.414,
                reward_shaping: RewardShaping {
                    exploration_bonus: 0.15,
                    depth_penalty: 0.03,
                    reward_scale: 1.0,
                    discount_factor: 0.95,
                },
                depth_limits: DepthLimits {
                    max_depth: 15,
                    prune_depth: 12,
                    iterative_deepening: true,
                },
            },
        }
    }
}

