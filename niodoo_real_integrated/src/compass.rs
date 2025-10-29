#![allow(dead_code)]

use anyhow::Result;
use blake3::Hasher;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::StandardNormal;
use std::time::Duration;
use tracing::instrument;

use crate::mcts::MctsAction;
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

const MCTS_ACTIONS: [MctsAction; 4] = [
    MctsAction::Retrieve,
    MctsAction::Decompose,
    MctsAction::DirectAnswer,
    MctsAction::Explore,
];

const MCTS_MAX_ITERATIONS: usize = 256;
const MCTS_MAX_DURATION: Duration = Duration::from_millis(120);
const MIN_ROLLOUT_DEPTH: usize = 3;
const MAX_ROLLOUT_DEPTH: usize = 9;
const PAD_CORE_DIMS: usize = 3;

#[derive(Debug, Clone, Copy)]
pub struct CompassRuntimeParams {
    pub exploration_c: f64,
    pub variance_spike: f64,
    pub variance_stagnation: f64,
}

impl CompassRuntimeParams {
    pub fn new(exploration_c: f64, variance_spike: f64, variance_stagnation: f64) -> Self {
        Self {
            exploration_c,
            variance_spike,
            variance_stagnation,
        }
    }
}

#[derive(Debug)]
pub struct CompassEngine {
    pub exploration_c: f64,
    pub variance_spike: f64,
    pub variance_stagnation: f64,
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
            last_quadrant: None,
            last_entropy: None,
            last_variance: None,
            recent_window: Vec::with_capacity(64),
            window_max: 64,
        }
    }

    /// Update compass parameters from RuntimeConfig (called before each cycle)
    pub fn update_params(
        &mut self,
        exploration_c: f64,
        variance_spike: f64,
        variance_stagnation: f64,
    ) {
        self.exploration_c = exploration_c.max(0.0);
        self.variance_spike = variance_spike.max(0.0);
        self.variance_stagnation = variance_stagnation.max(0.0);
    }

    pub fn evaluate_with_params(
        &mut self,
        params: CompassRuntimeParams,
        state: &PadGhostState,
        topology: Option<&crate::tcs_analysis::TopologicalSignature>,
    ) -> Result<CompassOutcome> {
        self.update_params(
            params.exploration_c,
            params.variance_spike,
            params.variance_stagnation,
        );
        self.evaluate(state, topology)
    }

    #[instrument(skip_all)]
    pub fn evaluate_with_rng(
        &mut self,
        state: &PadGhostState,
        topology: Option<&crate::tcs_analysis::TopologicalSignature>,
        rng: &mut StdRng,
    ) -> Result<CompassOutcome> {
        let mut pleasure = state.pad[0];
        let mut arousal = state.pad[1];
        let mut dominance = state.pad[2];

        pleasure = (pleasure + rng.gen_range(-0.4..0.4)).clamp(-1.0, 1.0);
        arousal = (arousal + rng.gen_range(-0.4..0.4)).clamp(-1.0, 1.0);
        dominance = (dominance + rng.gen_range(-0.4..0.4)).clamp(-1.0, 1.0);

        if rng.gen_bool(0.15) {
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

        if !is_threat && rng.gen_bool(0.45) {
            if arousal > -0.2 && pleasure < 0.35 {
                is_threat = true;
            }
        }

        // INTEGRATION FIX: Make healing detection topology-aware
        let mut is_healing = pleasure > healing_floor.0 && dominance > healing_floor.1;

        // Enhance healing detection with topology signals
        if let Some(topo) = topology {
            // Low knot complexity indicates untangled, clear reasoning - healing state
            if topo.knot_complexity < 0.3 && pleasure > 0.2 {
                is_healing = true;
            }
            // High spectral gap with good emotional state is healing
            if topo.spectral_gap > 0.7 && pleasure > 0.0 && dominance > 0.0 {
                is_healing = true;
            }
            // Low persistence entropy indicates stable structure - healing
            if topo.persistence_entropy < 0.3 && !is_threat {
                is_healing = true;
            }
        }

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

    #[instrument(skip_all)]
    pub fn evaluate(
        &mut self,
        state: &PadGhostState,
        topology: Option<&crate::tcs_analysis::TopologicalSignature>,
    ) -> Result<CompassOutcome> {
        // Fallback: deterministic default seed when external RNG isn't provided
        let mut rng = StdRng::seed_from_u64(42);
        self.evaluate_with_rng(state, topology, &mut rng)
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

        let entropy_avg = self
            .recent_window
            .iter()
            .rev()
            .take(32)
            .map(|snapshot| snapshot.entropy)
            .sum::<f64>()
            / 32.0;

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
        let pleasure_floor = (entropy_avg * 0.1 - threat_delta * 0.5).clamp(-0.3, 0.2);
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
        // Try using the new adaptive MCTS implementation first
        use crate::mcts::MctsNode;

        let mut root = MctsNode::new(MctsAction::Retrieve, state.clone(), None);

        // Run adaptive search with reward function
        let stats = root.search_adaptive(
            MCTS_MAX_DURATION.as_millis() as u64,
            self.exploration_c,
            |node| {
                // Reward function: higher entropy stability = better
                let stability = (1.0 - node.state.entropy).clamp(0.0, 1.0);
                let variance_mean =
                    node.state.sigma.iter().copied().sum::<f64>() / node.state.sigma.len() as f64;
                stability - variance_mean * 0.5
            },
        );

        // Convert adaptive search results to MctsBranch objects
        let mut branches = Vec::new();
        for child in root.children.iter() {
            branches.push(MctsBranch {
                label: child.action.to_string(),
                ucb_score: child.ucb1(self.exploration_c),
                entropy_projection: child.state.entropy,
            });
        }

        // Sort by UCB score descending
        branches.sort_by(|a, b| {
            b.ucb_score
                .partial_cmp(&a.ucb_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Fallback to heuristic if no branches found
        if branches.is_empty() {
            branches = self.fallback_mcts_heuristic(state);
        }

        tracing::info!(
            simulations = stats.total_simulations,
            depth = stats.max_depth,
            best_action = branches.first().map(|b| b.label.as_str()).unwrap_or("none"),
            "MCTS adaptive search completed"
        );

        branches
    }

    /// Fallback heuristic if MCTS search fails
    fn fallback_mcts_heuristic(&mut self, state: &PadGhostState) -> Vec<MctsBranch> {
        let mut branches = Vec::with_capacity(3);
        let priors = [0.5 + state.pad[0], 0.5 + state.pad[1], 0.5 + state.pad[2]];
        let mut visit_counts = [1usize; 3];
        let parent_visits = 10usize; // Fixed parent visit count for heuristic

        for idx in 0..3 {
            let reward_estimate = priors[idx].tanh() as f64;
            // Fixed UCB1: c * sqrt(ln(N(parent)) / N(n))
            let exploration = self.exploration_c
                * ((parent_visits as f64).ln() / visit_counts[idx] as f64).sqrt();
            let score = reward_estimate + exploration;
            branches.push(MctsBranch {
                label: format!("branch_{idx}"),
                ucb_score: score,
                entropy_projection: state.entropy + reward_estimate,
            });
            visit_counts[idx] += 1;
        }

        branches.sort_by(|a, b| {
            b.ucb_score
                .partial_cmp(&a.ucb_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        branches
    }

    fn mcts_select_path(&self, arena: &[SearchNode]) -> Vec<usize> {
        let mut path = Vec::new();
        if arena.is_empty() {
            return path;
        }

        let mut current_idx = 0usize;
        path.push(current_idx);
        loop {
            let node = &arena[current_idx];
            if node.children.is_empty() || !node.is_fully_expanded() {
                break;
            }

            let parent_visits = node.visits.max(1);
            let next = node
                .children
                .iter()
                .copied()
                .max_by(|&a, &b| {
                    let score_a = ucb_score(&arena[a], parent_visits, self.exploration_c);
                    let score_b = ucb_score(&arena[b], parent_visits, self.exploration_c);
                    score_a
                        .partial_cmp(&score_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("fully expanded node must have children");
            current_idx = next;
            path.push(current_idx);
        }

        path
    }

    fn mcts_expand(&self, arena: &mut Vec<SearchNode>, node_idx: usize, rng: &mut StdRng) -> usize {
        if arena.is_empty() {
            return node_idx;
        }

        if arena[node_idx].untried_actions.is_empty() {
            return node_idx;
        }

        let action_idx = rng.gen_range(0..arena[node_idx].untried_actions.len());
        let action = arena[node_idx].untried_actions.swap_remove(action_idx);
        let next_state = self.transition_state(&arena[node_idx].state, action, rng);
        let child_idx = arena.len();
        arena.push(SearchNode::new_child(action, next_state));
        arena[node_idx].children.push(child_idx);
        child_idx
    }

    fn mcts_rollout(&self, rng: &mut StdRng, state: &PadGhostState) -> f64 {
        let mut cursor = state.clone();
        let depth = Self::determine_rollout_depth(state);
        for _ in 0..depth {
            let action = MCTS_ACTIONS[rng.gen_range(0..MCTS_ACTIONS.len())];
            cursor = self.transition_state(&cursor, action, rng);
        }
        self.evaluate_state_reward(&cursor)
    }

    fn mcts_branches_from_root(&self, arena: &[SearchNode]) -> Vec<MctsBranch> {
        if arena.is_empty() {
            return Vec::new();
        }

        let root = &arena[0];
        let parent_visits = root.visits.max(1);
        let mut branches = Vec::with_capacity(root.children.len());
        for &child_idx in &root.children {
            let child = &arena[child_idx];
            let score = ucb_score(child, parent_visits, self.exploration_c);
            branches.push(MctsBranch {
                label: child.action.to_string(),
                ucb_score: score,
                entropy_projection: child.state.entropy,
            });
        }

        branches.sort_by(|a, b| {
            b.ucb_score
                .partial_cmp(&a.ucb_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        branches
    }

    fn transition_state(
        &self,
        base: &PadGhostState,
        action: MctsAction,
        rng: &mut StdRng,
    ) -> PadGhostState {
        let mut next = base.clone();
        let pad_dims = base.pad.len() as f64;
        let sigma_mean = base.sigma.iter().copied().sum::<f64>() / base.sigma.len() as f64;
        let mu_mean = base.mu.iter().copied().sum::<f64>() / base.mu.len() as f64;
        let entropy_delta = (sigma_mean + mu_mean.abs() + self.variance_spike) / pad_dims;

        next.entropy = match action {
            MctsAction::Retrieve => (base.entropy - entropy_delta).clamp(0.0, 1.0),
            MctsAction::Decompose => (base.entropy - entropy_delta * 0.5).clamp(0.0, 1.0),
            MctsAction::DirectAnswer => (base.entropy - entropy_delta / 3.0).clamp(0.0, 1.0),
            MctsAction::Explore => (base.entropy + entropy_delta).clamp(0.0, 1.0),
        };

        let noise_scale = (sigma_mean + self.variance_spike).max(f64::EPSILON);
        let mut noise_l1 = 0.0;
        for (idx, value) in next.pad.iter_mut().enumerate() {
            let component_weight = 1.0 / (idx as f64 + 1.0);
            let noise = rng.sample::<f64, _>(StandardNormal) * noise_scale * component_weight;
            noise_l1 += noise.abs();

            *value = (*value + noise).clamp(-1.0, 1.0);
            next.mu[idx] = (next.mu[idx] + noise * component_weight).clamp(-1.0, 1.0);

            let sigma_adjustment = 1.0 + noise.abs() * component_weight;
            next.sigma[idx] = (next.sigma[idx] * sigma_adjustment).max(self.variance_stagnation);
        }

        if !next.raw_stds.is_empty() {
            let adjustment = 1.0 + noise_l1 / (next.raw_stds.len() as f64 + 1.0);
            for value in &mut next.raw_stds {
                *value = (*value * adjustment).max(self.variance_stagnation.min(1.0));
            }
        }

        next
    }

    fn evaluate_state_reward(&self, state: &PadGhostState) -> f64 {
        let stability = (1.0 - state.entropy).clamp(0.0, 1.0);
        let affect_balance =
            state.pad[..PAD_CORE_DIMS].iter().copied().sum::<f64>() / PAD_CORE_DIMS as f64;

        let variance_mean = state.sigma.iter().copied().sum::<f64>() / state.sigma.len() as f64;
        let raw_std_mean = if state.raw_stds.is_empty() {
            0.0
        } else {
            state.raw_stds.iter().copied().sum::<f64>() / state.raw_stds.len() as f64
        };

        let penalty = (variance_mean + raw_std_mean) / 2.0;
        let raw_score = stability + affect_balance - penalty;
        ((raw_score + 1.0) / 2.0).clamp(0.0, 1.0)
    }

    fn mcts_backpropagate(&self, arena: &mut [SearchNode], path: &[usize], reward: f64) {
        for &idx in path.iter().rev() {
            if let Some(node) = arena.get_mut(idx) {
                node.visits += 1;
                node.total_reward += reward;
            }
        }
    }

    fn determine_rollout_depth(state: &PadGhostState) -> usize {
        let entropy_scaled = (state.entropy * state.pad.len() as f64).round() as usize;
        entropy_scaled
            .clamp(MIN_ROLLOUT_DEPTH, MAX_ROLLOUT_DEPTH)
            .min(100)
    }

    fn derive_mcts_seed(state: &PadGhostState) -> u64 {
        let mut hasher = Hasher::new();
        hasher.update(&state.entropy.to_le_bytes());
        for value in &state.pad {
            hasher.update(&value.to_le_bytes());
        }
        for value in &state.mu {
            hasher.update(&value.to_le_bytes());
        }
        for value in &state.sigma {
            hasher.update(&value.to_le_bytes());
        }
        for value in &state.raw_stds {
            hasher.update(&value.to_le_bytes());
        }

        let digest = hasher.finalize();
        let mut seed_bytes = [0u8; 8];
        seed_bytes.copy_from_slice(&digest.as_bytes()[..8]);
        u64::from_le_bytes(seed_bytes)
    }
}

#[derive(Clone)]
struct SearchNode {
    action: MctsAction,
    state: PadGhostState,
    children: Vec<usize>,
    untried_actions: Vec<MctsAction>,
    visits: usize,
    total_reward: f64,
}

impl SearchNode {
    fn new_root(state: PadGhostState) -> Self {
        Self {
            action: MctsAction::Retrieve,
            state,
            children: Vec::new(),
            untried_actions: MCTS_ACTIONS.to_vec(),
            visits: 0,
            total_reward: 0.0,
        }
    }

    fn new_child(action: MctsAction, state: PadGhostState) -> Self {
        Self {
            action,
            state,
            children: Vec::new(),
            untried_actions: MCTS_ACTIONS.to_vec(),
            visits: 0,
            total_reward: 0.0,
        }
    }

    fn is_fully_expanded(&self) -> bool {
        self.untried_actions.is_empty()
    }

    fn avg_reward(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.total_reward / self.visits as f64
        }
    }
}

fn ucb_score(node: &SearchNode, parent_visits: usize, exploration_c: f64) -> f64 {
    if node.visits == 0 {
        return f64::INFINITY;
    }

    let exploitation = node.avg_reward();
    let exploration = if parent_visits > 0 {
        exploration_c * ((parent_visits as f64).ln() / node.visits as f64).sqrt()
    } else {
        0.0
    };

    exploitation + exploration
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_state() -> PadGhostState {
        PadGhostState {
            pad: [0.1, -0.2, 0.3, 0.05, -0.1, 0.08, 0.0],
            entropy: 0.42,
            mu: [0.0; 7],
            sigma: [0.18; 7],
            raw_stds: vec![0.14; 7],
        }
    }

    #[test]
    fn mcts_search_produces_branches() {
        let mut engine = CompassEngine::new(1.2, 0.35, 0.05);
        let state = sample_state();
        let branches = engine.perform_mcts_search(&state);
        assert!(!branches.is_empty());
        assert!(branches.iter().all(|b| b.ucb_score.is_finite()));
    }
}
