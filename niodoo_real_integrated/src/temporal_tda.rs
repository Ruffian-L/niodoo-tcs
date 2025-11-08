//! Temporal Topological Data Analysis for Failure Chain Detection
//!
//! This module implements temporal TDA to detect failure patterns using persistent homology
//! on time-series topological data. It identifies repeating failure loops and danger signatures
//! before they cascade into system failures.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, SystemTime};
use tracing::{debug, warn};

use crate::compass::CompassQuadrant;
use crate::tcs_analysis::TopologicalSignature;

/// Topological snapshot captured at a point in time
/// Used to track system state evolution and detect failure patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalSnapshot {
    /// Betti number β₁ (loops/entanglement)
    pub beta_1: f32,
    /// Betti number β₂ (voids/gaps)
    pub beta_2: f32,
    /// Current compass quadrant (Panic/Persist/Discover/Master)
    pub compass_state: CompassQuadrant,
    /// Token count at snapshot time
    pub token_count: usize,
    /// Timestamp when snapshot was captured
    pub timestamp: DateTime<Utc>,
    /// Full topological signature for detailed analysis
    pub topology: TopologicalSignature,
    /// Arousal level from PAD state
    pub arousal: f32,
    /// Entropy value
    pub entropy: f64,
}

impl TopologicalSnapshot {
    /// Create a new topological snapshot
    pub fn new(
        topology: TopologicalSignature,
        compass_state: CompassQuadrant,
        token_count: usize,
        arousal: f32,
        entropy: f64,
    ) -> Self {
        Self {
            beta_1: topology.betti_numbers[1] as f32,
            beta_2: topology.betti_numbers[2] as f32,
            compass_state,
            token_count,
            timestamp: Utc::now(),
            topology,
            arousal,
            entropy,
        }
    }

    /// Create a persistence barcode representation for this snapshot
    /// Returns (birth, death) pairs for β₁ features
    pub fn persistence_barcode(&self) -> Vec<(f32, f32)> {
        self.topology
            .persistence_features
            .iter()
            .filter(|f| f.dimension == 1)
            .map(|f| (f.birth as f32, f.death as f32))
            .collect()
    }
}

/// Failure chain: sequence of topological states leading to failure
/// Detected using temporal TDA pattern matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureChain {
    /// Sequence of topological snapshots in the chain
    pub snapshots: Vec<TopologicalSnapshot>,
    /// Detected pattern type
    pub pattern_type: FailurePatternType,
    /// Severity score (higher = more severe)
    pub severity: f32,
    /// Whether this chain represents a repeating loop
    pub is_loop: bool,
    /// First detection timestamp
    pub first_detected: DateTime<Utc>,
    /// Number of times this pattern has been observed
    pub occurrence_count: usize,
}

/// Types of failure patterns detected via temporal TDA
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailurePatternType {
    /// Rate limit barcode pattern (β₁=3-4, β₂≈0)
    RateLimitBarcode,
    /// Overload barcode pattern (β₁>5, β₂>2)
    OverloadBarcode,
    /// Entropy divergence pattern (unstable entropy with high β₁)
    EntropyDivergence,
    /// Compass oscillation pattern (rapid quadrant changes)
    CompassOscillation,
    /// Token velocity acceleration (rapid token count growth)
    TokenVelocityAcceleration,
    /// Unknown pattern
    Unknown,
}

impl FailurePatternType {
    /// Determine pattern type from topological signature
    pub fn from_snapshot(snapshot: &TopologicalSnapshot) -> Self {
        let beta_1 = snapshot.beta_1;
        let beta_2 = snapshot.beta_2;
        let arousal = snapshot.arousal.abs();

        // Rate limit pattern: moderate β₁, low β₂
        if beta_1 >= 3.0 && beta_1 <= 4.0 && beta_2 < 0.5 {
            return FailurePatternType::RateLimitBarcode;
        }

        // Overload pattern: high β₁, high β₂
        if beta_1 > 5.0 && beta_2 > 2.0 {
            return FailurePatternType::OverloadBarcode;
        }

        // Entropy divergence: high β₁ with unstable entropy
        if beta_1 > 4.0 && (snapshot.entropy > 2.5 || snapshot.entropy < 1.0) {
            return FailurePatternType::EntropyDivergence;
        }

        FailurePatternType::Unknown
    }

    /// Get pattern name
    pub fn name(&self) -> &'static str {
        match self {
            FailurePatternType::RateLimitBarcode => "RateLimitBarcode",
            FailurePatternType::OverloadBarcode => "OverloadBarcode",
            FailurePatternType::EntropyDivergence => "EntropyDivergence",
            FailurePatternType::CompassOscillation => "CompassOscillation",
            FailurePatternType::TokenVelocityAcceleration => "TokenVelocityAcceleration",
            FailurePatternType::Unknown => "Unknown",
        }
    }
}

/// Danger signature: precursor patterns that predict failures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DangerSignature {
    /// β₁ trend (Rising/Falling/Stable)
    pub beta_1_trend: Trend,
    /// Arousal level
    pub arousal: f32,
    /// Token velocity (tokens per second)
    pub token_velocity: f32,
    /// Whether entropy is diverging
    pub entropy_divergence: bool,
    /// Dominance trend (Rising/Falling/Stable)
    pub dominance_trend: Trend,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
}

/// Trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Trend {
    Rising,
    Falling,
    Stable,
}

impl DangerSignature {
    /// Check if this signature matches dangerous patterns
    pub fn is_dangerous(&self) -> bool {
        // High arousal (>0.3)
        if self.arousal > 0.3 {
            return true;
        }

        // Rising β₁ with entropy divergence
        if self.beta_1_trend == Trend::Rising && self.entropy_divergence {
            return true;
        }

        // Accelerating token velocity (>100 tokens/sec)
        if self.token_velocity > 100.0 {
            return true;
        }

        // Dropping dominance with high arousal
        if self.dominance_trend == Trend::Falling && self.arousal > 0.2 {
            return true;
        }

        false
    }
}

/// Temporal TDA detector for failure chain analysis
pub struct TemporalTDADetector {
    /// History of topological snapshots (rolling window)
    pub history: VecDeque<TopologicalSnapshot>,
    /// Maximum history window size
    pub window_size: usize,
    /// Wasserstein distance threshold for detecting transitions
    pub wasserstein_threshold: f32,
    /// Failure chains detected
    pub failure_chains: VecDeque<FailureChain>,
    /// Maximum number of failure chains to track
    pub max_chains: usize,
}

impl TemporalTDADetector {
    /// Create a new temporal TDA detector
    pub fn new(window_size: usize, wasserstein_threshold: f32) -> Self {
        Self {
            history: VecDeque::with_capacity(window_size),
            window_size,
            wasserstein_threshold,
            failure_chains: VecDeque::with_capacity(10),
            max_chains: 10,
        }
    }

    /// Add a new topological snapshot to history
    pub fn add_snapshot(&mut self, snapshot: TopologicalSnapshot) {
        // Maintain window size
        if self.history.len() >= self.window_size {
            self.history.pop_front();
        }
        self.history.push_back(snapshot);
    }

    /// Calculate Wasserstein distance between two persistence diagrams
    /// Simplified 1-Wasserstein distance for (birth, death) pairs
    pub fn wasserstein_distance(barcode_a: &[(f32, f32)], barcode_b: &[(f32, f32)]) -> f32 {
        if barcode_a.is_empty() && barcode_b.is_empty() {
            return 0.0;
        }

        if barcode_a.is_empty() || barcode_b.is_empty() {
            // One is empty, return infinity distance
            return f32::INFINITY;
        }

        // Simplified Wasserstein-1 distance: sum of point-wise distances
        // For production, use proper optimal transport algorithm
        let mut distances = Vec::new();

        for (birth_a, death_a) in barcode_a {
            let mut min_dist = f32::INFINITY;
            for (birth_b, death_b) in barcode_b {
                let dist = ((birth_a - birth_b).powi(2) + (death_a - death_b).powi(2)).sqrt();
                min_dist = min_dist.min(dist);
            }
            distances.push(min_dist);
        }

        // Average distance
        distances.iter().sum::<f32>() / distances.len() as f32
    }

    /// Detect failure loops by analyzing temporal patterns
    pub fn detect_failure_loop(&self) -> Option<FailureChain> {
        if self.history.len() < 3 {
            return None;
        }

        let snapshots: Vec<_> = self.history.iter().collect();

        // Look for repeating patterns
        for window_size in 2..=snapshots.len().min(5) {
            for start_idx in 0..=snapshots.len() - window_size * 2 {
                let pattern: Vec<_> = snapshots[start_idx..start_idx + window_size]
                    .iter()
                    .cloned()
                    .cloned()
                    .collect();

                // Check if pattern repeats
                let next_start = start_idx + window_size;
                if next_start + window_size <= snapshots.len() {
                    let next_pattern: Vec<_> = snapshots[next_start..next_start + window_size]
                        .iter()
                        .cloned()
                        .cloned()
                        .collect();

                    // Compare patterns using Wasserstein distance
                    let mut total_distance = 0.0;
                    let mut count = 0;

                    for (a, b) in pattern.iter().zip(next_pattern.iter()) {
                        let barcode_a = a.persistence_barcode();
                        let barcode_b = b.persistence_barcode();
                        let dist = Self::wasserstein_distance(&barcode_a, &barcode_b);
                        total_distance += dist;
                        count += 1;
                    }

                    let avg_distance = if count > 0 {
                        total_distance / count as f32
                    } else {
                        f32::INFINITY
                    };

                    // If patterns are similar (low Wasserstein distance), it's a loop
                    if avg_distance < self.wasserstein_threshold {
                        let pattern_type = FailurePatternType::from_snapshot(&pattern[0]);
                        let severity = self.calculate_severity(&pattern);

                        debug!(
                            pattern_type = ?pattern_type,
                            severity = severity,
                            window_size = window_size,
                            "Detected failure loop pattern"
                        );

                        return Some(FailureChain {
                            snapshots: pattern.iter().cloned().collect(),
                            pattern_type,
                            severity,
                            is_loop: true,
                            first_detected: Utc::now(),
                            occurrence_count: 1,
                        });
                    }
                }
            }
        }

        None
    }

    /// Calculate severity score for a failure chain
    fn calculate_severity(&self, snapshots: &[TopologicalSnapshot]) -> f32 {
        if snapshots.is_empty() {
            return 0.0;
        }

        let mut severity = 0.0;

        // High β₁ contributes to severity
        let avg_beta_1: f32 =
            snapshots.iter().map(|s| s.beta_1).sum::<f32>() / snapshots.len() as f32;
        severity += avg_beta_1 * 0.3;

        // High arousal contributes
        let avg_arousal: f32 =
            snapshots.iter().map(|s| s.arousal.abs()).sum::<f32>() / snapshots.len() as f32;
        severity += avg_arousal * 0.2;

        // Panic state contributes
        let panic_count = snapshots
            .iter()
            .filter(|s| s.compass_state == CompassQuadrant::Panic)
            .count();
        severity += (panic_count as f32 / snapshots.len() as f32) * 0.3;

        // Entropy divergence contributes
        let entropy_variance: f32 = {
            let mean =
                snapshots.iter().map(|s| s.entropy as f32).sum::<f32>() / snapshots.len() as f32;
            let variance = snapshots
                .iter()
                .map(|s| ((s.entropy as f32) - mean).powi(2))
                .sum::<f32>()
                / snapshots.len() as f32;
            variance.sqrt()
        };
        if entropy_variance > 0.5 {
            severity += 0.2;
        }

        severity.min(10.0) // Cap at 10.0
    }

    /// Analyze sequence and detect danger signatures
    pub fn analyze_sequence(&self) -> Option<DangerSignature> {
        if self.history.len() < 3 {
            return None;
        }

        let recent: Vec<_> = self.history.iter().rev().take(5).collect();

        // Calculate trends
        let beta_1_values: Vec<f32> = recent.iter().map(|s| s.beta_1).collect();
        let beta_1_trend = Self::calculate_trend(&beta_1_values);

        let arousal_values: Vec<f32> = recent.iter().map(|s| s.arousal.abs()).collect();
        // Guard against division by zero (recent.len() >= 2 checked above, but be safe)
        let avg_arousal = if arousal_values.is_empty() {
            0.0
        } else {
            arousal_values.iter().sum::<f32>() / arousal_values.len() as f32
        };

        // Token velocity (tokens per second approximation)
        let token_velocity = if recent.len() >= 2 {
            let time_diff = (recent[0].timestamp - recent[recent.len() - 1].timestamp)
                .num_seconds()
                .max(1) as f32;
            let token_diff = (recent[0].token_count as i32
                - recent[recent.len() - 1].token_count as i32)
                .abs() as f32;
            token_diff / time_diff
        } else {
            0.0
        };

        // Entropy divergence
        let entropy_values: Vec<f64> = recent.iter().map(|s| s.entropy).collect();
        // Guard against division by zero (recent.len() >= 2 checked above, but be safe)
        let (entropy_mean, entropy_variance) = if entropy_values.is_empty() {
            (0.0, 0.0)
        } else {
            let mean = entropy_values.iter().sum::<f64>() / entropy_values.len() as f64;
            let variance = entropy_values
                .iter()
                .map(|&e| (e - mean).powi(2))
                .sum::<f64>()
                / entropy_values.len() as f64;
            (mean, variance)
        };
        let entropy_divergence = entropy_variance.sqrt() > 0.5;

        // Dominance trend (placeholder - would need dominance from PAD state)
        let dominance_trend = Trend::Stable;

        let confidence = if recent.len() >= 3 { 0.7 } else { 0.4 };

        let signature = DangerSignature {
            beta_1_trend,
            arousal: avg_arousal,
            token_velocity,
            entropy_divergence,
            dominance_trend,
            confidence,
        };

        if signature.is_dangerous() {
            Some(signature)
        } else {
            None
        }
    }

    /// Calculate trend from a sequence of values
    fn calculate_trend(values: &[f32]) -> Trend {
        if values.len() < 2 {
            return Trend::Stable;
        }

        let first_half: f32 =
            values[..values.len() / 2].iter().sum::<f32>() / (values.len() / 2) as f32;
        let second_half: f32 = values[values.len() / 2..].iter().sum::<f32>()
            / (values.len() - values.len() / 2) as f32;

        let diff = second_half - first_half;

        if diff > 0.1 {
            Trend::Rising
        } else if diff < -0.1 {
            Trend::Falling
        } else {
            Trend::Stable
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tcs_analysis::TopologicalSignature;
    use uuid::Uuid;

    fn create_test_topology(beta_1: usize, beta_2: usize) -> TopologicalSignature {
        TopologicalSignature {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            persistence_features: vec![],
            betti_numbers: [1, beta_1, beta_2],
            knot_complexity: 0.5,
            knot_polynomial: "".to_string(),
            tqft_dimension: 2,
            cobordism_type: None,
            persistence_entropy: 1.5,
            spectral_gap: 0.3,
            euler_characteristic: 1.0,
            total_persistence: 1.0,
            max_persistence: 1.0,
            mean_persistence: 1.0,
            laplacian_spectral_radius: 0.3,
            computation_time_ms: 10.0,
        }
    }

    #[test]
    fn test_failure_pattern_detection() {
        let topology = create_test_topology(4, 0);
        let snapshot = TopologicalSnapshot::new(topology, CompassQuadrant::Panic, 1000, 0.5, 2.5);

        let pattern = FailurePatternType::from_snapshot(&snapshot);
        assert_eq!(pattern, FailurePatternType::RateLimitBarcode);
    }

    #[test]
    fn test_wasserstein_distance() {
        let barcode_a = vec![(0.0, 1.0), (0.5, 2.0)];
        let barcode_b = vec![(0.1, 1.1), (0.6, 2.1)];

        let distance = TemporalTDADetector::wasserstein_distance(&barcode_a, &barcode_b);
        assert!(distance < 1.0); // Should be small for similar barcodes
    }

    #[test]
    fn test_danger_signature() {
        let signature = DangerSignature {
            beta_1_trend: Trend::Rising,
            arousal: 0.4,
            token_velocity: 150.0,
            entropy_divergence: true,
            dominance_trend: Trend::Falling,
            confidence: 0.8,
        };

        assert!(signature.is_dangerous());
    }
}
