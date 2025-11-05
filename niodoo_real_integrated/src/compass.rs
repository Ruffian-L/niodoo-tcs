use anyhow::Result;
use rand::prelude::*;
use std::collections::VecDeque;
use std::time::Instant;
use tracing::instrument;

use crate::torus::PadGhostState;
use crate::ntoken_client::NTokenFeatures;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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
    Recognition,  // "Oh. This is TRUE." - Initial breakthrough
    Satisfaction, // "This is elegant/correct" - Validation
    Calm,         // "I can trust this, it's solid" - Stability
    Motivation,   // "I want to build on this" - Expansion
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
    pub ucb1_score: Option<f64>,             // Add missing field
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
    /// Compass configuration values (optional, falls back to defaults if None)
    config: Option<CompassConfig>,
}

/// Compass configuration values extracted from RuntimeConfig
#[derive(Debug, Clone)]
pub struct CompassConfig {
    pub h1_persistence_divisor: f64,
    pub h1_penalty_scale: f64,
    pub sheaf_energy_threshold: f64,
    pub sheaf_boost_multiplier: f64,
    pub dominance_penalty_multiplier: f64,
    pub dominance_boost_multiplier: f64,
    pub arousal_penalty_multiplier: f64,
    pub random_noise_range: f64,
    pub pleasure_boost_probability: f64,
    pub pleasure_boost_multiplier: f64,
    pub base_threat_arousal_threshold: f64,
    pub variance_spike_multiplier: f64,
    pub random_threat_probability: f64,
    pub random_threat_arousal_threshold: f64,
    pub random_threat_pleasure_threshold: f64,
    pub healing_pleasure_threshold: f64,
    pub healing_dominance_threshold: f64,
    pub quadrant_panic_pleasure_threshold: f64,
    pub quadrant_panic_arousal_threshold: f64,
    pub quadrant_persist_arousal_threshold: f64,
    pub reward_panic_to_discover: f64,
    pub reward_panic_to_persist: f64,
    pub reward_panic_to_master: f64,
    pub reward_master_to_panic: f64,
    pub reward_default: f64,
    pub reward_entropy_multiplier: f64,
    pub mcts_h1_bonus_cap: f64,
    pub mcts_h1_bonus_multiplier: f64,
    pub mcts_persistence_divisor: f64,
    pub mcts_persistence_multiplier: f64,
    pub mcts_knot_multiplier: f64,
    pub mcts_knot_multiplier_cap: f64,
    pub mcts_knot_weight: f64,
    pub mcts_gap_multiplier: f64,
    pub mcts_entropy_multiplier: f64,
    pub mcts_entropy_multiplier_cap: f64,
    pub mcts_entropy_weight: f64,
    pub mcts_h0_bonus_cap: f64,
    pub mcts_h0_bonus_multiplier: f64,
    pub mcts_default_exploration_base: f64,
    pub mcts_default_exploration_divisor: f64,
    pub cascade_min_consonance: f64,
    pub cascade_recognition_satisfaction_consonance: f64,
    pub cascade_calm_motivation_consonance: f64,
}

impl Default for CompassConfig {
    fn default() -> Self {
        Self {
            h1_persistence_divisor: 2.5,
            h1_penalty_scale: 0.3,
            sheaf_energy_threshold: 0.3,
            sheaf_boost_multiplier: 0.5,
            dominance_penalty_multiplier: 0.7,
            dominance_boost_multiplier: 0.8,
            arousal_penalty_multiplier: 0.5,
            random_noise_range: 0.4,
            pleasure_boost_probability: 0.15,
            pleasure_boost_multiplier: 1.1,
            base_threat_arousal_threshold: 0.05,
            variance_spike_multiplier: 1.2,
            random_threat_probability: 0.45,
            random_threat_arousal_threshold: -0.2,
            random_threat_pleasure_threshold: 0.35,
            healing_pleasure_threshold: 0.25,
            healing_dominance_threshold: 0.05,
            quadrant_panic_pleasure_threshold: -0.1,
            quadrant_panic_arousal_threshold: 0.2,
            quadrant_persist_arousal_threshold: 0.2,
            reward_panic_to_discover: 10.0,
            reward_panic_to_persist: -1.0,
            reward_panic_to_master: 10.0,
            reward_master_to_panic: -5.0,
            reward_default: 1.0,
            reward_entropy_multiplier: 5.0,
            mcts_h1_bonus_cap: 5.0,
            mcts_h1_bonus_multiplier: 0.1,
            mcts_persistence_divisor: 3.0,
            mcts_persistence_multiplier: 0.15,
            mcts_knot_multiplier: 2.0,
            mcts_knot_multiplier_cap: 1.0,
            mcts_knot_weight: 0.2,
            mcts_gap_multiplier: 0.15,
            mcts_entropy_multiplier: 1.5,
            mcts_entropy_multiplier_cap: 1.0,
            mcts_entropy_weight: 0.12,
            mcts_h0_bonus_cap: 5.0,
            mcts_h0_bonus_multiplier: 0.1,
            mcts_default_exploration_base: 0.05,
            mcts_default_exploration_divisor: 3.0,
            cascade_min_consonance: 0.7,
            cascade_recognition_satisfaction_consonance: 0.8,
            cascade_calm_motivation_consonance: 0.75,
        }
    }
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
            config: None,
        }
    }

    /// Create with config for configurable thresholds
    pub fn new_with_config(exploration_c: f64, variance_spike: f64, variance_stagnation: f64, config: CompassConfig) -> Self {
        Self {
            exploration_c,
            variance_spike,
            variance_stagnation,
            _rng: StdRng::seed_from_u64(42),
            last_quadrant: None,
            last_entropy: None,
            last_variance: None,
            config: Some(config),
        }
    }

    /// Get config value or default
    fn get_config(&self) -> &CompassConfig {
        self.config.as_ref().unwrap_or(&CompassConfig::default())
    }

    #[instrument(skip_all)]
    pub fn evaluate(&mut self, state: &PadGhostState) -> Result<CompassOutcome> {
        self.evaluate_with_ntoken(state, None, None)
    }

    /// Evaluate with optional nToken features
    pub fn evaluate_with_ntoken(
        &mut self,
        state: &PadGhostState,
        ntoken_features: Option<&NTokenFeatures>,
        topology: Option<&crate::tcs_analysis::TopologicalSignature>,
    ) -> Result<CompassOutcome> {
        let mut pleasure = state.pad[0];
        let mut arousal = state.pad[1];
        let mut dominance = state.pad[2];

        // Adjust PAD state based on nToken metrics:
        // - High H₁ persistence → low PAD (frustrated): unresolved loops, tension building
        // - Low sheaf energy → high PAD (relieved): system found consistent story
        let cfg = self.get_config();
        if let Some(ntoken) = ntoken_features {
            // Normalize H₁ persistence (typical range: 0.0-5.0, use sigmoid to map to [-1, 1])
            // High persistence (e.g., >2.0) → reduce pleasure/dominance (frustrated)
            let h1_persistence_norm = (ntoken.h1_total_persistence / cfg.h1_persistence_divisor).tanh(); // Maps to [-1, 1]
            let h1_penalty = h1_persistence_norm * cfg.h1_penalty_scale; // Scale impact
            
            // Normalize sheaf energy (typical range: 0.0-1.0, already normalized)
            // Low energy (<threshold) → increase pleasure/dominance (relieved)
            let sheaf_boost = if ntoken.sheaf_energy < cfg.sheaf_energy_threshold {
                (cfg.sheaf_energy_threshold - ntoken.sheaf_energy) * cfg.sheaf_boost_multiplier // Stronger boost for lower energy
            } else {
                0.0
            };
            
            // Apply adjustments: high persistence reduces PAD, low sheaf energy increases it
            pleasure = (pleasure - h1_penalty + sheaf_boost).clamp(-1.0, 1.0);
            dominance = (dominance - h1_penalty * cfg.dominance_penalty_multiplier + sheaf_boost * cfg.dominance_boost_multiplier).clamp(-1.0, 1.0);
            // Arousal increases with unresolved loops (tension building)
            arousal = (arousal + h1_penalty * cfg.arousal_penalty_multiplier).clamp(-1.0, 1.0);
        }

        pleasure = (pleasure + self._rng.gen_range(-cfg.random_noise_range..cfg.random_noise_range)).clamp(-1.0, 1.0);
        arousal = (arousal + self._rng.gen_range(-cfg.random_noise_range..cfg.random_noise_range)).clamp(-1.0, 1.0);
        dominance = (dominance + self._rng.gen_range(-cfg.random_noise_range..cfg.random_noise_range)).clamp(-1.0, 1.0);

        if self._rng.gen_bool(cfg.pleasure_boost_probability) {
            pleasure = (pleasure * cfg.pleasure_boost_multiplier).clamp(-1.0, 1.0);
        }

        let variance = state.sigma.iter().map(|v| v.abs()).sum::<f64>() / state.sigma.len() as f64;

        let base_threat = pleasure < 0.0 && arousal > cfg.base_threat_arousal_threshold;
        let variance_spike = variance > self.variance_spike;
        let variance_stall = variance < self.variance_stagnation;

        let mut is_threat = base_threat || variance_spike || variance_stall;

        if !is_threat {
            if let Some(prev_var) = self.last_variance {
                if variance > prev_var * cfg.variance_spike_multiplier {
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

        if !is_threat && self._rng.gen_bool(cfg.random_threat_probability) {
            if arousal > cfg.random_threat_arousal_threshold && pleasure < cfg.random_threat_pleasure_threshold {
                is_threat = true;
            }
        }

        let is_healing = pleasure > cfg.healing_pleasure_threshold && dominance > cfg.healing_dominance_threshold;

        let quadrant = match (pleasure, arousal) {
            (p, a) if p < cfg.quadrant_panic_pleasure_threshold && a > cfg.quadrant_panic_arousal_threshold => CompassQuadrant::Panic,
            (p, a) if p < 0.0 && a <= cfg.quadrant_persist_arousal_threshold => CompassQuadrant::Persist,
            (p, a) if p >= 0.0 && a >= 0.0 => CompassQuadrant::Discover,
            _ => CompassQuadrant::Master,
        };

        if !is_threat && matches!(quadrant, CompassQuadrant::Panic | CompassQuadrant::Persist) {
            is_threat = true;
        }

        let mcts_branches = self.expand_mcts(state, topology);

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
        let cfg = self.get_config();
        match (self.last_quadrant, self.last_entropy) {
            (Some(prev), Some(prev_entropy)) => {
                let entropy_delta = prev_entropy - entropy;
                let base = match (prev, quadrant) {
                    (CompassQuadrant::Panic, CompassQuadrant::Discover)
                    | (CompassQuadrant::Persist, CompassQuadrant::Master)
                    | (CompassQuadrant::Panic, CompassQuadrant::Master) => cfg.reward_panic_to_discover,
                    (CompassQuadrant::Panic, CompassQuadrant::Persist) => cfg.reward_panic_to_persist,
                    (CompassQuadrant::Master, CompassQuadrant::Panic) => cfg.reward_master_to_panic,
                    _ => cfg.reward_default,
                };
                base + entropy_delta * cfg.reward_entropy_multiplier
            }
            _ => 0.0,
        }
    }

    /// Expand MCTS branches using topology-aware strategies
    /// Problem structure informs solution shape:
    /// - High H₁ persistence → "unwind loops" branches
    /// - High knot complexity → "simplify structure" branches
    /// - Low spectral gap → "stabilize" branches
    /// - High persistence entropy → "structure" branches
    fn expand_mcts(
        &mut self,
        state: &PadGhostState,
        topology: Option<&crate::tcs_analysis::TopologicalSignature>,
    ) -> Vec<MctsBranch> {
        let mut branches = Vec::with_capacity(3);
        let priors = [0.5 + state.pad[0], 0.5 + state.pad[1], 0.5 + state.pad[2]];
        let mut visit_counts = [1usize; 3];
        let mut total_visits = 3usize;

        // Determine branch strategies based on topology
        let branch_strategies = if let Some(topology) = topology {
            self.compute_topology_strategies(topology, state)
        } else {
            // Fallback to PAD-based strategies when topology unavailable
            vec![
                (0, "emotional_prior".to_string()),
                (1, "emotional_prior".to_string()),
                (2, "emotional_prior".to_string()),
            ]
        };

        for (idx, (pad_idx, strategy)) in branch_strategies.iter().enumerate() {
            let pad_prior = priors[*pad_idx];
            let reward_estimate = pad_prior.tanh() as f64;

            // Apply topology-based adjustments to reward
            let topology_bonus = if let Some(topology) = topology {
                self.compute_topology_bonus(topology, strategy, idx)
            } else {
                0.0
            };

            let exploration =
                self.exploration_c * ((total_visits as f64).ln() / visit_counts[idx] as f64).sqrt();
            let score = reward_estimate + exploration + topology_bonus;
            
            branches.push(MctsBranch {
                label: format!("{}_{}", strategy, idx),
                ucb_score: score,
                entropy_projection: state.entropy + reward_estimate + topology_bonus,
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

    /// Compute branch strategies based on topology
    /// Maps problem structure to solution approaches
    fn compute_topology_strategies(
        &self,
        topology: &crate::tcs_analysis::TopologicalSignature,
        state: &PadGhostState,
    ) -> Vec<(usize, String)> {
        let mut strategies = Vec::new();

        // Strategy 1: High H₁ persistence → "unwind loops" (resolve cyclical patterns)
        if topology.betti_numbers[1] > 2 || topology.total_persistence > 2.0 {
            strategies.push((0, "unwind_loops".to_string()));
        } else {
            strategies.push((0, "explore".to_string()));
        }

        // Strategy 2: High knot complexity → "simplify structure" (reduce tangling)
        if topology.knot_complexity > 0.4 {
            strategies.push((1, "simplify_structure".to_string()));
        } else if topology.spectral_gap < 0.3 {
            // Low spectral gap → "stabilize" (increase structural stability)
            strategies.push((1, "stabilize".to_string()));
        } else {
            strategies.push((1, "exploit".to_string()));
        }

        // Strategy 3: High persistence entropy → "structure" (organize information)
        if topology.persistence_entropy > 0.6 {
            strategies.push((2, "structure".to_string()));
        } else if topology.betti_numbers[0] > 3 {
            // High H₀ → "connect" (link disconnected components)
            strategies.push((2, "connect".to_string()));
        } else {
            strategies.push((2, "refine".to_string()));
        }

        strategies
    }

    /// Compute topology-based bonus for branch selection
    fn compute_topology_bonus(
        &self,
        topology: &crate::tcs_analysis::TopologicalSignature,
        strategy: &str,
        branch_idx: usize,
    ) -> f64 {
        let cfg = self.get_config();
        match strategy {
            "unwind_loops" => {
                // Stronger bonus for high H₁ persistence
                let h1_bonus = (topology.betti_numbers[1] as f64).min(cfg.mcts_h1_bonus_cap) * cfg.mcts_h1_bonus_multiplier;
                let persistence_bonus = (topology.total_persistence / cfg.mcts_persistence_divisor).min(1.0) * cfg.mcts_persistence_multiplier;
                h1_bonus + persistence_bonus
            }
            "simplify_structure" => {
                // Bonus for high knot complexity
                let knot_bonus = (topology.knot_complexity * cfg.mcts_knot_multiplier).min(cfg.mcts_knot_multiplier_cap) * cfg.mcts_knot_weight;
                knot_bonus
            }
            "stabilize" => {
                // Bonus when spectral gap is low (needs stabilization)
                let gap_bonus = (1.0 - topology.spectral_gap).max(0.0) * cfg.mcts_gap_multiplier;
                gap_bonus
            }
            "structure" => {
                // Bonus for high persistence entropy (needs organization)
                let entropy_bonus = (topology.persistence_entropy * cfg.mcts_entropy_multiplier).min(cfg.mcts_entropy_multiplier_cap) * cfg.mcts_entropy_weight;
                entropy_bonus
            }
            "connect" => {
                // Bonus for high H₀ (many disconnected components)
                let h0_bonus = (topology.betti_numbers[0] as f64).min(cfg.mcts_h0_bonus_cap) * cfg.mcts_h0_bonus_multiplier;
                h0_bonus
            }
            _ => {
                // Default: small exploration bonus
                cfg.mcts_default_exploration_base * (branch_idx as f64 + 1.0) / cfg.mcts_default_exploration_divisor
            }
        }
    }

    /// Evaluate with custom RNG and optional nToken features
    pub fn evaluate_with_rng(
        &mut self,
        state: &PadGhostState,
        topology: Option<&crate::tcs_analysis::TopologicalSignature>,
        rng: &mut rand::rngs::StdRng,
        ntoken_features: Option<&NTokenFeatures>,
    ) -> Result<CompassOutcome> {
        // Use provided RNG for evaluation
        let mut pleasure = state.pad[0];
        let mut arousal = state.pad[1];
        let mut dominance = state.pad[2];

        // Adjust PAD state based on nToken metrics:
        // - High H₁ persistence → low PAD (frustrated): unresolved loops, tension building
        // - Low sheaf energy → high PAD (relieved): system found consistent story
        let cfg = self.get_config();
        if let Some(ntoken) = ntoken_features {
            // Normalize H₁ persistence (typical range: 0.0-5.0, use sigmoid to map to [-1, 1])
            // High persistence (e.g., >2.0) → reduce pleasure/dominance (frustrated)
            let h1_persistence_norm = (ntoken.h1_total_persistence / cfg.h1_persistence_divisor).tanh(); // Maps to [-1, 1]
            let h1_penalty = h1_persistence_norm * cfg.h1_penalty_scale; // Scale impact
            
            // Normalize sheaf energy (typical range: 0.0-1.0, already normalized)
            // Low energy (<threshold) → increase pleasure/dominance (relieved)
            let sheaf_boost = if ntoken.sheaf_energy < cfg.sheaf_energy_threshold {
                (cfg.sheaf_energy_threshold - ntoken.sheaf_energy) * cfg.sheaf_boost_multiplier // Stronger boost for lower energy
            } else {
                0.0
            };
            
            // Apply adjustments: high persistence reduces PAD, low sheaf energy increases it
            pleasure = (pleasure - h1_penalty + sheaf_boost).clamp(-1.0, 1.0);
            dominance = (dominance - h1_penalty * cfg.dominance_penalty_multiplier + sheaf_boost * cfg.dominance_boost_multiplier).clamp(-1.0, 1.0);
            // Arousal increases with unresolved loops (tension building)
            arousal = (arousal + h1_penalty * cfg.arousal_penalty_multiplier).clamp(-1.0, 1.0);
        }

        pleasure = (pleasure + rng.gen_range(-cfg.random_noise_range..cfg.random_noise_range)).clamp(-1.0, 1.0);
        arousal = (arousal + rng.gen_range(-cfg.random_noise_range..cfg.random_noise_range)).clamp(-1.0, 1.0);
        dominance = (dominance + rng.gen_range(-cfg.random_noise_range..cfg.random_noise_range)).clamp(-1.0, 1.0);

        let variance = state.sigma.iter().map(|v| v.abs()).sum::<f64>() / state.sigma.len() as f64;

        let base_threat = pleasure < 0.0 && arousal > cfg.base_threat_arousal_threshold;
        let variance_spike = variance > self.variance_spike;
        let variance_stall = variance < self.variance_stagnation;

        let mut is_threat = base_threat || variance_spike || variance_stall;

        let is_healing = pleasure > cfg.healing_pleasure_threshold && dominance > cfg.healing_dominance_threshold;

        let quadrant = match (pleasure, arousal) {
            (p, a) if p < cfg.quadrant_panic_pleasure_threshold && a > cfg.quadrant_panic_arousal_threshold => CompassQuadrant::Panic,
            (p, a) if p < 0.0 && a <= cfg.quadrant_persist_arousal_threshold => CompassQuadrant::Persist,
            (p, a) if p >= 0.0 && a >= 0.0 => CompassQuadrant::Discover,
            _ => CompassQuadrant::Master,
        };

        let mcts_branches = self.expand_mcts(state, topology);
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
            self.stage_history
                .push_back((proposed_stage, Instant::now()));
            return None; // No transition yet, just initialization
        };

        // Check if we should transition
        let should_transition =
            self.should_transition(current, proposed_stage, compass, consonance);

        if should_transition {
            let transition = CascadeTransition {
                from: current,
                to: proposed_stage,
                timestamp: Instant::now(),
                consonance,
                compass_quadrant: compass.quadrant,
            };

            self.current_stage = Some(proposed_stage);
            self.stage_history
                .push_back((proposed_stage, Instant::now()));

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
        // Use default config values for cascade thresholds (CascadeTracker doesn't have direct config access)
        // These match the defaults in RuntimeConfig
        let min_consonance = 0.7;
        let recognition_satisfaction_consonance = 0.8;
        let calm_motivation_consonance = 0.75;
        
        // High consonance required for transitions
        if consonance < min_consonance {
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
            && consonance > recognition_satisfaction_consonance
        {
            return true;
        }

        // Allow Satisfaction → Calm if stable and Persist quadrant
        if current == CascadeStage::Satisfaction
            && proposed == CascadeStage::Calm
            && compass.quadrant == CompassQuadrant::Persist
            && !compass.is_threat
        {
            return true;
        }

        // Allow Calm → Motivation if new Discover triggered
        if current == CascadeStage::Calm
            && proposed == CascadeStage::Motivation
            && compass.quadrant == CompassQuadrant::Discover
            && consonance > calm_motivation_consonance
        {
            return true;
        }

        // Allow restart from Motivation back to Recognition (new cycle)
        if current == CascadeStage::Motivation
            && proposed == CascadeStage::Recognition
            && compass.quadrant == CompassQuadrant::Discover
        {
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
