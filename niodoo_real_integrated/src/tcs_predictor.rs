//! Phase 5.3: TCS Topological Predictor
//! Predicts reward deltas and optimal actions based on topological features

use std::collections::HashMap;
use tracing::{debug, info};

use crate::config::RuntimeConfig;
use crate::tcs_analysis::TopologicalSignature;

/// Phase 5.3: Topological predictor for reward/action forecasting
pub struct TcsPredictor {
    feature_weights: HashMap<String, f64>,
    history: Vec<(TopologicalSignature, f64, f64)>, // (sig, reward_delta, performance)
    capacity: usize,
}

impl TcsPredictor {
    pub fn new() -> Self {
        let mut weights = HashMap::new();
        weights.insert("knot_complexity".to_string(), -0.5);
        weights.insert("betti1".to_string(), -0.2);
        weights.insert("persistence_entropy".to_string(), -0.1);
        weights.insert("spectral_gap".to_string(), 0.4); // Low gap is good
        weights.insert("betti0".to_string(), 0.3);

        Self {
            feature_weights: weights,
            history: Vec::new(),
            capacity: 100,
        }
    }

    /// Phase 5.3: Predict reward delta based on topological signature
    pub fn predict_reward_delta(&self, sig: &TopologicalSignature) -> f64 {
        let knot_contrib = sig.knot_complexity as f64
            * self.feature_weights.get("knot_complexity").unwrap_or(&0.0);
        let betti1_contrib =
            sig.betti_numbers[1] as f64 * self.feature_weights.get("betti1").unwrap_or(&0.0);
        let pe_contrib = sig.persistence_entropy
            * self
                .feature_weights
                .get("persistence_entropy")
                .unwrap_or(&0.0);
        let gap_contrib =
            sig.spectral_gap * self.feature_weights.get("spectral_gap").unwrap_or(&0.0);
        let betti0_contrib =
            sig.betti_numbers[0] as f64 * self.feature_weights.get("betti0").unwrap_or(&0.0);

        knot_contrib + betti1_contrib + pe_contrib + gap_contrib + betti0_contrib
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
                *self.feature_weights.get_mut("knot_complexity").unwrap() *= 1.05;
            }
            if sig.spectral_gap < 0.3 && avg_perf > 0.7 {
                // Low gap correlates with high performance - strengthen bonus
                *self.feature_weights.get_mut("spectral_gap").unwrap() *= 1.05;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RuntimeConfig;
    use crate::tcs_analysis::TopologicalSignature;
    use uuid::Uuid;

    fn create_test_sig(knot: f64, gap: f64) -> TopologicalSignature {
        TopologicalSignature {
            id: Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            persistence_features: vec![],
            betti_numbers: [1, 1, 0],
            knot_complexity: knot,
            knot_polynomial: "test".to_string(),
            tqft_dimension: 2,
            cobordism_type: None,
            persistence_entropy: 0.5,
            spectral_gap: gap,
            computation_time_ms: 1.0,
        }
    }

    #[test]
    fn test_predictor_creation() {
        let predictor = TcsPredictor::new();
        assert!(!predictor.feature_weights.is_empty());
    }

    #[test]
    fn test_reward_prediction() {
        let predictor = TcsPredictor::new();
        let sig = create_test_sig(0.5, 0.3);
        let pred = predictor.predict_reward_delta(&sig);
        assert!(pred < 0.0); // High knot should predict negative
    }

    #[test]
    fn test_action_prediction() {
        let predictor = TcsPredictor::new();
        let args = crate::config::CliArgs::default();
        let config = RuntimeConfig::load(&args).expect("failed to load runtime config for test");
        let sig = create_test_sig(0.5, 0.3);
        let (param, delta) = predictor.predict_action(&sig, &config);
        assert_eq!(param, "temperature");
        assert!(delta < 0.0);
    }

    #[test]
    fn test_trigger_logic() {
        let predictor = TcsPredictor::new();
        let high_knot = create_test_sig(0.5, 0.3);
        let low_knot = create_test_sig(0.2, 0.3);

        assert!(predictor.should_trigger(&high_knot));
        assert!(!predictor.should_trigger(&low_knot));
    }
}
