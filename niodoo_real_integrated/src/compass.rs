use anyhow::Result;
use rand::prelude::*;
use std::collections::VecDeque;
use std::time::Instant;
use tracing::instrument;

use crate::torus::PadGhostState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompassQuadrant {
    Panic,
    Persist,
    Discover,
    Master,
}

/// Emotional cascade stages: Recognition → Satisfaction → Calm → Motivation
/// Maps to the cognitive progression from breakthrough to integration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum CascadeStage {
    Recognition,    // "Oh. This is TRUE." - Initial breakthrough
    Satisfaction,   // "This is elegant/correct" - Validation
    Calm,          // "I can trust this, it's solid" - Stability
    Motivation,    // "I want to build on this" - Expansion
}

impl CascadeStage {
    /// Map compass quadrant to initial cascade stage
    pub fn from_quadrant(quadrant: CompassQuadrant) -> Self {
        match quadrant {
            CompassQuadrant::Discover => CascadeStage::Recognition,
            CompassQuadrant::Master => CascadeStage::Satisfaction,
            CompassQuadrant::Persist => CascadeStage::Calm,
            CompassQuadrant::Panic => CascadeStage::Recognition, // Panic can lead to recognition
        }
    }

    /// Get next stage in cascade progression
    pub fn next(self) -> Self {
        match self {
            CascadeStage::Recognition => CascadeStage::Satisfaction,
            CascadeStage::Satisfaction => CascadeStage::Calm,
            CascadeStage::Calm => CascadeStage::Motivation,
            CascadeStage::Motivation => CascadeStage::Recognition, // Cycle back
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            CascadeStage::Recognition => "Recognition",
            CascadeStage::Satisfaction => "Satisfaction",
            CascadeStage::Calm => "Calm",
            CascadeStage::Motivation => "Motivation",
        }
    }
}

/// Complete cascade cycle from Recognition to Motivation
#[derive(Debug, Clone)]
pub struct FullCascade {
    pub start_time: Instant,
    pub completion_time: Instant,
    pub stages: Vec<(CascadeStage, Instant)>,
    pub peak_consonance: f64,
}

/// Cascade transition event
#[derive(Debug, Clone)]
pub struct CascadeTransition {
    pub from: CascadeStage,
    pub to: CascadeStage,
    pub timestamp: Instant,
    pub consonance: f64,
    pub compass_quadrant: CompassQuadrant,
}

#[derive(Debug, Clone)]
pub struct CompassOutcome {
    pub quadrant: CompassQuadrant,
    pub is_threat: bool,
    pub is_healing: bool,
    pub mcts_branches: Vec<MctsBranch>,
    pub intrinsic_reward: f64,
    pub cascade_stage: Option<CascadeStage>, // Current cascade stage if tracked
    pub ucb1_score: Option<f64>, // Add missing field
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
            cascade_stage: None, // Set by CascadeTracker
            ucb1_score: None,
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

    /// Evaluate with custom RNG
    pub fn evaluate_with_rng(
        &mut self,
        state: &PadGhostState,
        _topology: Option<&crate::tcs_analysis::TopologicalSignature>,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<CompassOutcome> {
        // Use provided RNG for evaluation
        let mut pleasure = state.pad[0];
        let mut arousal = state.pad[1];
        let mut dominance = state.pad[2];

        pleasure = (pleasure + rng.gen_range(-0.4..0.4)).clamp(-1.0, 1.0);
        arousal = (arousal + rng.gen_range(-0.4..0.4)).clamp(-1.0, 1.0);
        dominance = (dominance + rng.gen_range(-0.4..0.4)).clamp(-1.0, 1.0);

        let variance = state.sigma.iter().map(|v| v.abs()).sum::<f64>() / state.sigma.len() as f64;

        let base_threat = pleasure < 0.0 && arousal > 0.05;
        let variance_spike = variance > self.variance_spike;
        let variance_stall = variance < self.variance_stagnation;

        let mut is_threat = base_threat || variance_spike || variance_stall;

        let is_healing = pleasure > 0.25 && dominance > 0.05;

        let quadrant = match (pleasure, arousal) {
            (p, a) if p < -0.1 && a > 0.2 => CompassQuadrant::Panic,
            (p, a) if p < 0.0 && a <= 0.2 => CompassQuadrant::Persist,
            (p, a) if p >= 0.0 && a >= 0.0 => CompassQuadrant::Discover,
            _ => CompassQuadrant::Master,
        };

        let mcts_branches = self.expand_mcts(state);
        let intrinsic_reward = self.compute_intrinsic_reward(quadrant, state.entropy);

        Ok(CompassOutcome {
            quadrant,
            is_threat,
            is_healing,
            mcts_branches,
            intrinsic_reward,
            cascade_stage: None,
            ucb1_score: None,
        })
    }
}
pub struct CascadeTracker {
    current_stage: Option<CascadeStage>,
    stage_history: VecDeque<(CascadeStage, Instant)>,
    full_cascades: Vec<FullCascade>,
    current_cascade_start: Option<Instant>,
}

impl CascadeTracker {
    pub fn new() -> Self {
        Self {
            current_stage: None,
            stage_history: VecDeque::with_capacity(100),
            full_cascades: Vec::new(),
            current_cascade_start: None,
        }
    }

    /// Detect cascade transition based on compass outcome and consonance
    pub fn detect_transition(
        &mut self,
        compass: &CompassOutcome,
        consonance: f64,
    ) -> Option<CascadeTransition> {
        let proposed_stage = CascadeStage::from_quadrant(compass.quadrant);
        
        // Initial stage assignment
        let Some(current) = self.current_stage else {
            // Start new cascade
            self.current_stage = Some(proposed_stage);
            self.current_cascade_start = Some(Instant::now());
            self.stage_history.push_back((proposed_stage, Instant::now()));
            return None; // No transition yet, just initialization
        };

        // Check if we should transition
        let should_transition = self.should_transition(current, proposed_stage, compass, consonance);

        if should_transition {
            let transition = CascadeTransition {
                from: current,
                to: proposed_stage,
                timestamp: Instant::now(),
                consonance,
                compass_quadrant: compass.quadrant,
            };

            self.current_stage = Some(proposed_stage);
            self.stage_history.push_back((proposed_stage, Instant::now()));

            // Check if we completed a full cascade (Recognition → Motivation)
            if self.check_full_cascade() {
                if let Some(start) = self.current_cascade_start {
                    let full_cascade = FullCascade {
                        start_time: start,
                        completion_time: Instant::now(),
                        stages: self.stage_history.iter().cloned().collect(),
                        peak_consonance: consonance,
                    };
                    self.full_cascades.push(full_cascade);
                    self.current_cascade_start = Some(Instant::now()); // Start new cascade
                }
            }

            Some(transition)
        } else {
            None
        }
    }

    /// Determine if we should transition based on cascade progression rules
    fn should_transition(
        &self,
        current: CascadeStage,
        proposed: CascadeStage,
        compass: &CompassOutcome,
        consonance: f64,
    ) -> bool {
        // High consonance required for transitions
        if consonance < 0.7 {
            return false; // Not aligned enough
        }

        // Allow forward progression (Recognition → Satisfaction → Calm → Motivation)
        if proposed == current.next() {
            return true;
        }

        // Allow Recognition → Satisfaction if high consonance and Master quadrant
        if current == CascadeStage::Recognition 
            && proposed == CascadeStage::Satisfaction 
            && compass.quadrant == CompassQuadrant::Master
            && consonance > 0.8 {
            return true;
        }

        // Allow Satisfaction → Calm if stable and Persist quadrant
        if current == CascadeStage::Satisfaction 
            && proposed == CascadeStage::Calm 
            && compass.quadrant == CompassQuadrant::Persist
            && !compass.is_threat {
            return true;
        }

        // Allow Calm → Motivation if new Discover triggered
        if current == CascadeStage::Calm 
            && proposed == CascadeStage::Motivation 
            && compass.quadrant == CompassQuadrant::Discover
            && consonance > 0.75 {
            return true;
        }

        // Allow restart from Motivation back to Recognition (new cycle)
        if current == CascadeStage::Motivation 
            && proposed == CascadeStage::Recognition 
            && compass.quadrant == CompassQuadrant::Discover {
            return true;
        }

        false
    }

    /// Check if we've completed a full cascade cycle
    fn check_full_cascade(&self) -> bool {
        if self.stage_history.len() < 4 {
            return false;
        }

        // Check if we have all 4 stages in order
        let stages: Vec<CascadeStage> = self.stage_history.iter().map(|(s, _)| *s).collect();
        
        // Look for Recognition → Satisfaction → Calm → Motivation pattern
        for i in 0..=stages.len().saturating_sub(4) {
            if stages[i] == CascadeStage::Recognition
                && stages.get(i + 1).copied() == Some(CascadeStage::Satisfaction)
                && stages.get(i + 2).copied() == Some(CascadeStage::Calm)
                && stages.get(i + 3).copied() == Some(CascadeStage::Motivation)
            {
                return true;
            }
        }

        false
    }

    pub fn current_stage(&self) -> Option<CascadeStage> {
        self.current_stage
    }

    pub fn full_cascades_count(&self) -> usize {
        self.full_cascades.len()
    }

    pub fn get_full_cascades(&self) -> &[FullCascade] {
        &self.full_cascades
    }
}

impl Default for CascadeTracker {
    fn default() -> Self {
        Self::new()
    }
}
