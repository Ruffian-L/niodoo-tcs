//! Phase 5.3: TCS Topological Predictor
//! Predicts reward deltas and optimal actions based on topological features

use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use tracing::{debug, info, warn};

use crate::config::RuntimeConfig;
use crate::tcs_analysis::TopologicalSignature;

/// Phase 5.3: Topological predictor for reward/action forecasting
pub struct TcsPredictor {
    feature_weights: HashMap<String, f64>,
    history: Vec<(TopologicalSignature, f64, f64)>, // (sig, reward_delta, performance)
    capacity: usize,
    ridge_lambda: f64,
}

const FEATURE_COLUMNS: [&str; 8] = [
    "knot_complexity",
    "betti0",
    "betti1",
    "betti2",
    "persistence_entropy",
    "spectral_gap",
    "total_persistence",
    "mean_persistence",
];

const FEATURE_LEN: usize = FEATURE_COLUMNS.len();

impl TcsPredictor {
    pub fn new() -> Self {
        let mut weights = HashMap::new();
        weights.insert("knot_complexity".to_string(), -0.8);
        weights.insert("betti0".to_string(), 0.1);
        weights.insert("betti1".to_string(), -0.3);
        weights.insert("betti2".to_string(), 0.05);
        weights.insert("persistence_entropy".to_string(), -0.2);
        weights.insert("spectral_gap".to_string(), 0.8);
        weights.insert("total_persistence".to_string(), 0.15);
        weights.insert("mean_persistence".to_string(), 0.05);

        Self {
            feature_weights: weights,
            history: Vec::new(),
            capacity: 100,
            ridge_lambda: 1e-3,
        }
    }

    pub fn set_capacity(&mut self, capacity: usize) {
        self.capacity = capacity.max(FEATURE_LEN);
        if self.history.len() > self.capacity {
            let start = self.history.len() - self.capacity;
            self.history = self.history.split_off(start);
        }
    }

    /// Phase 5.3: Predict reward delta based on topological signature
    pub fn predict_reward_delta(&self, sig: &TopologicalSignature) -> f64 {
        self.score_signature(sig)
    }

    fn feature_vector(sig: &TopologicalSignature) -> [f64; FEATURE_LEN] {
        [
            sig.knot_complexity,
            sig.betti_numbers.get(0).copied().unwrap_or(0) as f64,
            sig.betti_numbers.get(1).copied().unwrap_or(0) as f64,
            sig.betti_numbers.get(2).copied().unwrap_or(0) as f64,
            sig.persistence_entropy,
            sig.spectral_gap,
            sig.total_persistence,
            sig.mean_persistence,
        ]
    }

    pub fn score_signature(&self, sig: &TopologicalSignature) -> f64 {
        let features = Self::feature_vector(sig);
        features
            .iter()
            .zip(FEATURE_COLUMNS.iter())
            .map(|(value, name)| *value * self.feature_weights.get(*name).copied().unwrap_or(0.0))
            .sum()
    }

    pub fn fit_weights(&mut self) {
        if self.history.len() < FEATURE_LEN {
            return;
        }

        let rows = self.history.len();
        let mut design = Vec::with_capacity(rows * FEATURE_LEN);
        let mut targets = Vec::with_capacity(rows);

        for (signature, reward_delta, _) in &self.history {
            let vector = Self::feature_vector(signature);
            design.extend_from_slice(&vector);
            targets.push(*reward_delta);
        }

        let x = DMatrix::from_row_slice(rows, FEATURE_LEN, &design);
        let y = DVector::from_vec(targets);

        let xtx =
            &x.transpose() * &x + DMatrix::identity(FEATURE_LEN, FEATURE_LEN) * self.ridge_lambda;
        let xty = &x.transpose() * &y;

        match xtx.cholesky() {
            Some(cholesky) => {
                let solution = cholesky.solve(&xty);
                for (idx, name) in FEATURE_COLUMNS.iter().enumerate() {
                    let weight = solution[idx];
                    self.feature_weights.insert((*name).to_string(), weight);
                }
                info!(
                    "TCS Predictor weights updated using {} samples (ridge λ={:.4})",
                    rows, self.ridge_lambda
                );
            }
            None => {
                warn!(
                    "Unable to invert topology feature matrix (rows={}, cols={})",
                    rows, FEATURE_LEN
                );
            }
        }
    }

    /// Phase 5.3: Predict optimal action based on topological features
    pub fn predict_action(
        &self,
        sig: &TopologicalSignature,
        _config: &RuntimeConfig,
    ) -> (String, f64) {
        // If knot complexity is high, suggest reducing temperature/entropy
        if sig.knot_complexity > 0.4 {
            debug!(
                "High knot complexity {:.3}, suggesting temperature reduction",
                sig.knot_complexity
            );
            return ("temperature".to_string(), -0.1);
        }

        // If spectral gap is high (unstable), suggest parameter stabilization
        if sig.spectral_gap > 0.5 {
            debug!(
                "High spectral gap {:.3}, suggesting stabilization",
                sig.spectral_gap
            );
            return ("top_p".to_string(), 0.05);
        }

        // If betti numbers indicate complexity, adjust novelty threshold
        if sig.betti_numbers[1] > 2 {
            debug!(
                "High H1 betti {}, suggesting novelty increase",
                sig.betti_numbers[1]
            );
            return ("novelty_threshold".to_string(), 0.1);
        }

        // Default: no action
        ("temperature".to_string(), 0.0)
    }

    /// Phase 5.3: Update predictor with new experience
    pub fn update(&mut self, sig: &TopologicalSignature, reward_delta: f64, performance: f64) {
        self.history.push((sig.clone(), reward_delta, performance));
        if self.history.len() > self.capacity {
            self.history.remove(0);
        }

        if self.history.len() >= FEATURE_LEN && self.history.len() % FEATURE_LEN == 0 {
            self.fit_weights();
        }

        // Adaptive learning: adjust weights based on recent performance
        if self.history.len() >= 10 {
            self.adapt_weights();
        }
    }

    /// Phase 5.3: Adapt feature weights based on correlation with performance
    fn adapt_weights(&mut self) {
        let recent = &self.history[self.history.len().saturating_sub(20)..];
        if recent.is_empty() {
            return;
        }

        // Simple correlation-based adaptation
        let avg_perf: f64 = recent.iter().map(|(_, _, p)| p).sum::<f64>() / recent.len() as f64;

        for (sig, _, _) in recent {
            if sig.knot_complexity > 0.4 && avg_perf < 0.5 {
                // High knot correlates with low performance - strengthen penalty
                if let Some(weight) = self.feature_weights.get_mut("knot_complexity") {
                    *weight *= 1.05;
                } else {
                    tracing::warn!("feature_weights missing 'knot_complexity' key");
                }
            }
            if sig.spectral_gap < 0.3 && avg_perf > 0.7 {
                // Low gap correlates with high performance - strengthen bonus
                if let Some(weight) = self.feature_weights.get_mut("spectral_gap") {
                    *weight *= 1.05;
                } else {
                    tracing::warn!("feature_weights missing 'spectral_gap' key");
                }
            }
        }

        // Clamp weights to reasonable ranges
        for weight in self.feature_weights.values_mut() {
            *weight = weight.clamp(-1.0, 1.0);
        }

        info!("TCS Predictor weights adapted: {:?}", self.feature_weights);
    }

    /// Phase 5.3: Check if predictor should trigger (knot > 0.4)
    pub fn should_trigger(&self, sig: &TopologicalSignature) -> bool {
        sig.knot_complexity > 0.4 || sig.spectral_gap > 0.5
    }

    /// Phase 5.3: Get recent performance statistics
    pub fn get_stats(&self) -> (f64, f64) {
        if self.history.is_empty() {
            return (0.0, 0.0);
        }
        let avgs: (f64, f64) = self
            .history
            .iter()
            .map(|(_, rd, p)| (*rd, *p))
            .fold((0.0, 0.0), |acc, x| (acc.0 + x.0, acc.1 + x.1));
        let len = self.history.len() as f64;
        (avgs.0 / len, avgs.1 / len)
    }
}

impl Default for TcsPredictor {
    fn default() -> Self {
        Self::new()
    }
}
