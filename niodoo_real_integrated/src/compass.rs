use anyhow::Result;
use rand::prelude::*;
use tracing::instrument;

use crate::torus::PadGhostState;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CompassQuadrant {
    Panic,
    Persist,
    Discover,
    Master,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompassOutcome {
    pub quadrant: CompassQuadrant,
    pub is_threat: bool,
    pub is_healing: bool,
    pub mcts_branches: Vec<MctsBranch>,
    pub intrinsic_reward: f64,
    pub ucb1_score: Option<f64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MctsBranch {
    pub label: String,
    pub ucb_score: f64,
    pub entropy_projection: f64,
}

#[derive(Debug)]
pub struct CompassEngine {
    pub exploration_c: f64,
    pub variance_spike: f64,
    pub variance_stagnation: f64,
    _rng: StdRng,
    last_quadrant: Option<CompassQuadrant>,
    last_entropy: Option<f64>,
    last_variance: Option<f64>,
    recent_window: Vec<CompassOutcomeSnapshot>,
    window_max: usize,
}

#[derive(Debug, Clone, Copy)]
struct CompassOutcomeSnapshot {
    entropy: f64,
    variance: f64,
    is_threat: bool,
    is_healing: bool,
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
            recent_window: Vec::with_capacity(64),
            window_max: 64,
        }
    }

    #[instrument(skip_all)]
    pub fn evaluate(
        &mut self,
        state: &PadGhostState,
        topology: Option<&crate::tcs_analysis::TopologicalSignature>,
    ) -> Result<CompassOutcome> {
        let mut pleasure = state.pad[0];
        let mut arousal = state.pad[1];
        let mut dominance = state.pad[2];

        pleasure = (pleasure + self._rng.gen_range(-0.4..0.4)).clamp(-1.0, 1.0);
        arousal = (arousal + self._rng.gen_range(-0.4..0.4)).clamp(-1.0, 1.0);
        dominance = (dominance + self._rng.gen_range(-0.4..0.4)).clamp(-1.0, 1.0);

        if self._rng.gen_bool(0.15) {
            pleasure = (pleasure * 1.1).clamp(-1.0, 1.0);
        }

        let mut variance =
            state.sigma.iter().map(|v| v.abs()).sum::<f64>() / state.sigma.len() as f64;

        // Integrate topology analysis into threat detection
        if let Some(topo) = topology {
            // Knot complexity amplifies variance for threat detection
            if topo.knot_complexity > 0.7 {
                variance *= 1.3; // Boost variance when topology is complex
            }
            // High Betti numbers (H1) indicate loops/cycles - potential threat patterns
            if topo.betti_numbers[1] > 2 {
                variance *= 1.2;
            }
        }

        let (threat_floor, healing_floor) = self.compute_dynamic_thresholds();

        let base_threat = pleasure < threat_floor.0 && arousal > threat_floor.1;
        let variance_spike = variance > threat_floor.2;
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

        let is_healing = pleasure > healing_floor.0 && dominance > healing_floor.1;

        let quadrant = match (pleasure, arousal) {
            (p, a) if p < -0.1 && a > 0.2 => CompassQuadrant::Panic,
            (p, a) if p < 0.0 && a <= 0.2 => CompassQuadrant::Persist,
            (p, a) if p >= 0.0 && a >= 0.0 => CompassQuadrant::Discover,
            _ => CompassQuadrant::Master,
        };

        if !is_threat && matches!(quadrant, CompassQuadrant::Panic | CompassQuadrant::Persist) {
            is_threat = true;
        }

        // Perform MCTS search and extract branches
        let mcts_branches = self.perform_mcts_search(state);

        let intrinsic_reward = self.compute_intrinsic_reward(quadrant, state.entropy);
        self.last_quadrant = Some(quadrant);
        self.last_entropy = Some(state.entropy);
        self.last_variance = Some(variance);

        let ucb1_score = mcts_branches.first().map(|b| b.ucb_score);
        
        let outcome = CompassOutcome {
            quadrant,
            is_threat,
            is_healing,
            mcts_branches,
            intrinsic_reward,
            ucb1_score: ucb1_score,
        };

        self.ingest_outcome(state, &outcome, variance);

        Ok(outcome)
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

    fn ingest_outcome(&mut self, state: &PadGhostState, outcome: &CompassOutcome, variance: f64) {
        if self.recent_window.len() == self.window_max {
            self.recent_window.remove(0);
        }
        self.recent_window.push(CompassOutcomeSnapshot {
            entropy: state.entropy,
            variance,
            is_threat: outcome.is_threat,
            is_healing: outcome.is_healing,
        });
    }

    fn compute_dynamic_thresholds(&mut self) -> ((f64, f64, f64), (f64, f64)) {
        if self.recent_window.len() < 8 {
            return ((0.0, 0.05, self.variance_spike), (0.25, 0.05));
        }

        let recent_threat_rate = self
            .recent_window
            .iter()
            .rev()
            .take(32)
            .filter(|snapshot| snapshot.is_threat)
            .count() as f64
            / 32.0;
        let variance_avg = self
            .recent_window
            .iter()
            .rev()
            .take(32)
            .map(|snapshot| snapshot.variance)
            .sum::<f64>()
            / 32.0;

        let target_threat = 0.4;
        let threat_delta = recent_threat_rate - target_threat;
        let pleasure_floor = (0.0 - threat_delta * 0.5).clamp(-0.3, 0.2);
        let arousal_floor = (0.05 + threat_delta * 0.3).clamp(-0.1, 0.3);
        let var_floor = (self.variance_spike + threat_delta * variance_avg * 0.2)
            .clamp(self.variance_stagnation * 1.2, self.variance_spike * 1.5);

        let recent_healing = self
            .recent_window
            .iter()
            .rev()
            .take(32)
            .filter(|snapshot| snapshot.is_healing)
            .count() as f64
            / 32.0;
        let target_healing = 1.0;
        let healing_delta = target_healing - recent_healing;
        let heal_pleasure = (0.25 - healing_delta * 0.3).clamp(0.1, 0.4);
        let heal_dominance = (0.05 - healing_delta * 0.2).clamp(-0.1, 0.3);

        (
            (pleasure_floor, arousal_floor, var_floor),
            (heal_pleasure, heal_dominance),
        )
    }

    /// Perform MCTS search and convert results to MctsBranch objects
    fn perform_mcts_search(&mut self, state: &PadGhostState) -> Vec<MctsBranch> {
        // MCTS engine implementation pending - use fallback heuristic for now
        self.fallback_mcts_heuristic(state)
    }

    /// Fallback heuristic if MCTS search fails
    fn fallback_mcts_heuristic(&mut self, state: &PadGhostState) -> Vec<MctsBranch> {
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
