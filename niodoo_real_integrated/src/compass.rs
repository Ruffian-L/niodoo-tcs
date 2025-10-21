use anyhow::Result;
use rand::prelude::*;
use tracing::instrument;

use crate::torus::PadGhostState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompassQuadrant {
    Panic,
    Persist,
    Discover,
    Master,
}

#[derive(Debug, Clone)]
pub struct CompassOutcome {
    pub quadrant: CompassQuadrant,
    pub is_threat: bool,
    pub is_healing: bool,
    pub mcts_branches: Vec<MctsBranch>,
    pub intrinsic_reward: f64,
}

#[derive(Debug, Clone)]
pub struct MctsBranch {
    pub label: String,
    pub ucb_score: f64,
    pub entropy_projection: f64,
}

#[derive(Debug, Clone)]
pub struct CompassEngine {
    pub exploration_c: f64,
    variance_spike: f64,
    variance_stagnation: f64,
    _rng: StdRng,
    last_quadrant: Option<CompassQuadrant>,
    last_entropy: Option<f64>,
    last_variance: Option<f64>,
}

impl CompassEngine {
    pub fn new(exploration_c: f64, variance_spike: f64, variance_stagnation: f64) -> Self {
        Self {
            exploration_c,
            variance_spike,
            variance_stagnation,
            _rng: StdRng::seed_from_u64(42),
            last_quadrant: None,
            last_entropy: None,
            last_variance: None,
        }
    }

    #[instrument(skip_all)]
    pub fn evaluate(&mut self, state: &PadGhostState) -> Result<CompassOutcome> {
        let mut pleasure = state.pad[0];
        let mut arousal = state.pad[1];
        let mut dominance = state.pad[2];

        pleasure = (pleasure + self._rng.gen_range(-0.4..0.4)).clamp(-1.0, 1.0);
        arousal = (arousal + self._rng.gen_range(-0.4..0.4)).clamp(-1.0, 1.0);
        dominance = (dominance + self._rng.gen_range(-0.4..0.4)).clamp(-1.0, 1.0);

        if self._rng.gen_bool(0.15) {
            pleasure = (pleasure * 1.1).clamp(-1.0, 1.0);
        }

        let variance = state.sigma.iter().map(|v| v.abs()).sum::<f64>() / state.sigma.len() as f64;

        let base_threat = pleasure < 0.0 && arousal > 0.05;
        let variance_spike = variance > self.variance_spike;
        let variance_stall = variance < self.variance_stagnation;

        let mut is_threat = base_threat || variance_spike || variance_stall;

        if !is_threat {
            if let Some(prev_var) = self.last_variance {
                if variance > prev_var * 1.2 {
                    is_threat = true;
                }
            }
        }

        if !is_threat {
            if let Some(prev_entropy) = self.last_entropy {
                if state.entropy < prev_entropy && variance_stall {
                    is_threat = true;
                }
            }
        }

        if !is_threat && self._rng.gen_bool(0.45) {
            if arousal > -0.2 && pleasure < 0.35 {
                is_threat = true;
            }
        }

        let is_healing = pleasure > 0.25 && dominance > 0.05;

        let quadrant = match (pleasure, arousal) {
            (p, a) if p < -0.1 && a > 0.2 => CompassQuadrant::Panic,
            (p, a) if p < 0.0 && a <= 0.2 => CompassQuadrant::Persist,
            (p, a) if p >= 0.0 && a >= 0.0 => CompassQuadrant::Discover,
            _ => CompassQuadrant::Master,
        };

        if !is_threat && matches!(quadrant, CompassQuadrant::Panic | CompassQuadrant::Persist) {
            is_threat = true;
        }

        let mcts_branches = self.expand_mcts(state);

        let intrinsic_reward = self.compute_intrinsic_reward(quadrant, state.entropy);
        self.last_quadrant = Some(quadrant);
        self.last_entropy = Some(state.entropy);
        self.last_variance = Some(variance);

        Ok(CompassOutcome {
            quadrant,
            is_threat,
            is_healing,
            mcts_branches,
            intrinsic_reward,
        })
    }

    fn compute_intrinsic_reward(&self, quadrant: CompassQuadrant, entropy: f64) -> f64 {
        match (self.last_quadrant, self.last_entropy) {
            (Some(prev), Some(prev_entropy)) => {
                let entropy_delta = prev_entropy - entropy;
                let base = match (prev, quadrant) {
                    (CompassQuadrant::Panic, CompassQuadrant::Discover)
                    | (CompassQuadrant::Persist, CompassQuadrant::Master)
                    | (CompassQuadrant::Panic, CompassQuadrant::Master) => 10.0,
                    (CompassQuadrant::Panic, CompassQuadrant::Persist) => -1.0,
                    (CompassQuadrant::Master, CompassQuadrant::Panic) => -5.0,
                    _ => 1.0,
                };
                base + entropy_delta * 5.0
            }
            _ => 0.0,
        }
    }

    fn expand_mcts(&mut self, state: &PadGhostState) -> Vec<MctsBranch> {
        let mut branches = Vec::with_capacity(3);
        let priors = [0.5 + state.pad[0], 0.5 + state.pad[1], 0.5 + state.pad[2]];
        let mut visit_counts = [1usize; 3];
        let mut total_visits = 3usize;

        for idx in 0..3 {
            let reward_estimate = priors[idx].tanh() as f64;
            let exploration =
                self.exploration_c * ((total_visits as f64).ln() / visit_counts[idx] as f64).sqrt();
            let score = reward_estimate + exploration;
            branches.push(MctsBranch {
                label: format!("branch_{idx}"),
                ucb_score: score,
                entropy_projection: state.entropy + reward_estimate,
            });
            visit_counts[idx] += 1;
            total_visits += 1;
        }

        branches.sort_by(|a, b| {
            b.ucb_score
                .partial_cmp(&a.ucb_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        branches
    }
}
