//! Hyperfocus Detection Module
//! Detects when all parallel threads find consonance and collapse into coherent action
//! This represents "zero internal conflict, pure aligned momentum"

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use crate::consonance::ConsonanceMetrics;

/// Hyperfocus event - when all systems align
#[derive(Debug, Clone, Serialize)]
pub struct HyperfocusEvent {
    pub overall_consonance: f64,
    pub aligned_signals: Vec<String>,
    pub coherent_action: CoherentAction,
    #[serde(skip)]
    pub timestamp: std::time::Instant,
}

/// Coherent actions to take when hyperfocus is detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoherentAction {
    StoreBreakthrough,    // Store as aligned breakthrough memory
    PromoteToken,         // Promote token to vocabulary
    ConsolidateMemory,    // Consolidate memory
    ReduceExploration,    // Reduce exploration noise
    All,                 // Do all actions
}

impl CoherentAction {
    pub fn name(&self) -> &'static str {
        match self {
            CoherentAction::StoreBreakthrough => "store_breakthrough",
            CoherentAction::PromoteToken => "promote_token",
            CoherentAction::ConsolidateMemory => "consolidate_memory",
            CoherentAction::ReduceExploration => "reduce_exploration",
            CoherentAction::All => "all",
        }
    }
}

/// Hyperfocus detector that monitors multiple parallel signals
pub struct HyperfocusDetector {
    signal_weights: HashMap<String, f64>,
    threshold: f64,  // Default: 0.85 = hyperfocus trigger
    min_signal_threshold: f64,  // Each signal must be > this (default: 0.7)
}

impl HyperfocusDetector {
    pub fn new() -> Self {
        let mut signal_weights = HashMap::new();
        // Default weights for different signals
        signal_weights.insert("compass".to_string(), 0.3);
        signal_weights.insert("erag".to_string(), 0.25);
        signal_weights.insert("topology".to_string(), 0.20);
        signal_weights.insert("generation".to_string(), 0.15);
        signal_weights.insert("curator".to_string(), 0.10);

        Self {
            signal_weights,
            threshold: 0.85,
            min_signal_threshold: 0.7,
        }
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold.clamp(0.0, 1.0);
        self
    }

    pub fn with_min_signal_threshold(mut self, min: f64) -> Self {
        self.min_signal_threshold = min.clamp(0.0, 1.0);
        self
    }

    pub fn with_weights(mut self, weights: HashMap<String, f64>) -> Self {
        self.signal_weights = weights;
        self
    }

    /// Detect hyperfocus from multiple consonance signals
    /// Returns Some(HyperfocusEvent) if all signals align above threshold
    pub fn detect(&self, signals: &HashMap<String, ConsonanceMetrics>) -> Option<HyperfocusEvent> {
        if signals.is_empty() {
            return None;
        }

        // Check that all signals meet minimum threshold
        let all_signals_above_threshold = signals.iter().all(|(_, metrics)| {
            metrics.score >= self.min_signal_threshold
        });

        if !all_signals_above_threshold {
            return None;
        }

        // Compute weighted average of all consonance scores
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        let mut aligned_signals = Vec::new();

        for (signal_name, metrics) in signals.iter() {
            let weight = self.signal_weights.get(signal_name).copied().unwrap_or(0.1);
            weighted_sum += metrics.score * weight;
            total_weight += weight;
            
            if metrics.score >= self.min_signal_threshold {
                aligned_signals.push(signal_name.clone());
            }
        }

        if total_weight == 0.0 {
            return None;
        }

        let overall_consonance = weighted_sum / total_weight;

        // Check if we exceed threshold
        if overall_consonance >= self.threshold {
            // Determine coherent action based on signal strengths
            let coherent_action = self.determine_coherent_action(signals, overall_consonance);

            Some(HyperfocusEvent {
                overall_consonance,
                aligned_signals,
                coherent_action,
                timestamp: std::time::Instant::now(),
            })
        } else {
            None
        }
    }

    /// Determine which coherent action to take based on signal strengths
    fn determine_coherent_action(
        &self,
        signals: &HashMap<String, ConsonanceMetrics>,
        overall_consonance: f64,
    ) -> CoherentAction {
        // If extremely high consonance (>0.95), do all actions
        if overall_consonance > 0.95 {
            return CoherentAction::All;
        }

        // Check individual signal strengths to determine best action
        let compass_score = signals.get("compass").map(|m| m.score).unwrap_or(0.0);
        let erag_score = signals.get("erag").map(|m| m.score).unwrap_or(0.0);
        let topology_score = signals.get("topology").map(|m| m.score).unwrap_or(0.0);
        let curator_score = signals.get("curator").map(|m| m.score).unwrap_or(0.0);

        // If curator is very high, focus on storing breakthrough
        if curator_score > 0.9 {
            return CoherentAction::StoreBreakthrough;
        }

        // If topology is very high, consider token promotion
        if topology_score > 0.9 {
            return CoherentAction::PromoteToken;
        }

        // If ERAG is very high, consolidate memory
        if erag_score > 0.9 {
            return CoherentAction::ConsolidateMemory;
        }

        // If compass is very high but others moderate, reduce exploration
        if compass_score > 0.9 && overall_consonance > 0.85 {
            return CoherentAction::ReduceExploration;
        }

        // Default: store breakthrough
        CoherentAction::StoreBreakthrough
    }

    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    pub fn min_signal_threshold(&self) -> f64 {
        self.min_signal_threshold
    }
}

impl Default for HyperfocusDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consonance::{ConsonanceMetrics, ConsonanceSource};

    fn create_test_consonance(score: f64) -> ConsonanceMetrics {
        ConsonanceMetrics {
            score,
            sources: vec![ConsonanceSource::EmotionalCoherence(score)],
            confidence: 0.9,
            dissonance_score: 1.0 - score,
        }
    }

    #[test]
    fn test_hyperfocus_detection() {
        let detector = HyperfocusDetector::new();
        let mut signals = HashMap::new();
        
        signals.insert("compass".to_string(), create_test_consonance(0.9));
        signals.insert("erag".to_string(), create_test_consonance(0.85));
        signals.insert("topology".to_string(), create_test_consonance(0.88));

        let event = detector.detect(&signals);
        assert!(event.is_some());
        
        let event = event.unwrap();
        assert!(event.overall_consonance >= 0.85);
        assert_eq!(event.aligned_signals.len(), 3);
    }

    #[test]
    fn test_hyperfocus_no_detection() {
        let detector = HyperfocusDetector::new();
        let mut signals = HashMap::new();
        
        signals.insert("compass".to_string(), create_test_consonance(0.6)); // Too low
        signals.insert("erag".to_string(), create_test_consonance(0.65));

        let event = detector.detect(&signals);
        assert!(event.is_none()); // Should not trigger
    }

    #[test]
    fn test_coherent_action_determination() {
        let detector = HyperfocusDetector::new();
        let mut signals = HashMap::new();
        
        signals.insert("curator".to_string(), create_test_consonance(0.95));
        signals.insert("compass".to_string(), create_test_consonance(0.85));
        signals.insert("erag".to_string(), create_test_consonance(0.85));

        let event = detector.detect(&signals);
        assert!(event.is_some());
        
        let event = event.unwrap();
        // Should choose StoreBreakthrough because curator is high
        assert_eq!(event.coherent_action, CoherentAction::StoreBreakthrough);
    }
}

