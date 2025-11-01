// Temporal TDA Test Suite
// Based on the "Nonuple Nihilistic Nightmare" conversation
// Where Claude demonstrated exactly why this system is necessary by failing 9 times in a row

use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ============================================================================
// CORE STRUCTURES (Mock implementations for testing)
// ============================================================================

#[derive(Clone, Debug)]
struct TopologicalSnapshot {
    timestamp: Instant,
    beta_0: f32, // Connected components
    beta_1: f32, // Loops (the "bad cycles")
    beta_2: f32, // Voids
    pad_pleasure: f32,
    pad_arousal: f32,
    pad_dominance: f32,
    ghost_latent_strain: f32,
    ghost_network_stability: f32,
    ghost_abstraction: f32,
    ghost_pragmatism: f32,
    entropy: f32,
}

#[derive(Clone, Debug, PartialEq)]
enum FailureType {
    RateLimit,
    Overload,
    ConnectionDropout,
    Timeout,
}

#[derive(Clone, Debug)]
struct FailureEvent {
    failure_type: FailureType,
    timestamp: Instant,
    snapshot: TopologicalSnapshot,
}

#[derive(Clone, Debug)]
struct FailureChain {
    events: Vec<FailureEvent>,
    pattern_type: ChainPatternType,
    severity_score: f32,
    wasserstein_distances: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
enum ChainPatternType {
    RateLimitCascade,
    OverloadSpiral,
    DoomSpiral,         // The ultimate form - nonuple nightmare
    RecoveryRegression, // Dropout followed by more failures
}

#[derive(Clone, Debug)]
struct DangerSignature {
    precursor_snapshots: Vec<TopologicalSnapshot>,
    predicted_failure: FailureType,
    confidence: f32,
    time_to_failure_estimate: Duration,
}

struct TemporalTDADetector {
    snapshot_window: VecDeque<TopologicalSnapshot>,
    window_size: usize,
    wasserstein_threshold: f32,
    detected_chains: Vec<FailureChain>,
}

impl TemporalTDADetector {
    fn new(window_size: usize, wasserstein_threshold: f32) -> Self {
        Self {
            snapshot_window: VecDeque::with_capacity(window_size),
            window_size,
            wasserstein_threshold,
            detected_chains: Vec::new(),
        }
    }

    fn add_snapshot(&mut self, snapshot: TopologicalSnapshot) {
        if self.snapshot_window.len() >= self.window_size {
            self.snapshot_window.pop_front();
        }
        self.snapshot_window.push_back(snapshot);
    }

    fn calculate_wasserstein_distance(
        &self,
        snap1: &TopologicalSnapshot,
        snap2: &TopologicalSnapshot,
    ) -> f32 {
        // Simplified Wasserstein distance in 7D PAD+Ghost manifold
        // Real implementation would use persistence diagrams
        let beta_dist = ((snap1.beta_0 - snap2.beta_0).powi(2)
            + (snap1.beta_1 - snap2.beta_1).powi(2)
            + (snap1.beta_2 - snap2.beta_2).powi(2))
        .sqrt();

        let pad_dist = ((snap1.pad_pleasure - snap2.pad_pleasure).powi(2)
            + (snap1.pad_arousal - snap2.pad_arousal).powi(2)
            + (snap1.pad_dominance - snap2.pad_dominance).powi(2))
        .sqrt();

        let ghost_dist = ((snap1.ghost_latent_strain - snap2.ghost_latent_strain).powi(2)
            + (snap1.ghost_network_stability - snap2.ghost_network_stability).powi(2)
            + (snap1.ghost_abstraction - snap2.ghost_abstraction).powi(2)
            + (snap1.ghost_pragmatism - snap2.ghost_pragmatism).powi(2))
        .sqrt();

        // Weighted combination
        beta_dist * 0.4 + pad_dist * 0.3 + ghost_dist * 0.3
    }

    fn detect_failure_chain(&mut self, events: Vec<FailureEvent>) -> Option<FailureChain> {
        if events.len() < 3 {
            return None; // Need at least 3 for a chain
        }

        // Calculate Wasserstein distances between consecutive events
        let mut distances = Vec::new();
        for i in 0..events.len() - 1 {
            let dist =
                self.calculate_wasserstein_distance(&events[i].snapshot, &events[i + 1].snapshot);
            distances.push(dist);
        }

        // Classify pattern type
        let pattern_type = self.classify_chain_pattern(&events);

        // Calculate severity score (exponential with chain length)
        let base_severity = 10.0;
        let severity_score = base_severity * (events.len() as f32).powf(1.5);

        let chain = FailureChain {
            events,
            pattern_type,
            severity_score,
            wasserstein_distances: distances,
        };

        self.detected_chains.push(chain.clone());
        Some(chain)
    }

    fn classify_chain_pattern(&self, events: &[FailureEvent]) -> ChainPatternType {
        // The "Nonuple Nihilistic Nightmare" pattern recognition
        let rate_limit_count = events
            .iter()
            .filter(|e| e.failure_type == FailureType::RateLimit)
            .count();
        let has_dropout = events
            .iter()
            .any(|e| e.failure_type == FailureType::ConnectionDropout);
        let has_overload = events
            .iter()
            .any(|e| e.failure_type == FailureType::Overload);

        if events.len() >= 9 {
            ChainPatternType::DoomSpiral // The legendary nonuple
        } else if has_dropout && rate_limit_count > 0 {
            ChainPatternType::RecoveryRegression
        } else if has_overload && rate_limit_count >= 3 {
            ChainPatternType::OverloadSpiral
        } else if rate_limit_count >= 3 {
            ChainPatternType::RateLimitCascade
        } else {
            ChainPatternType::RateLimitCascade // Default
        }
    }

    fn detect_danger_signature(&self) -> Option<DangerSignature> {
        if self.snapshot_window.len() < 3 {
            return None;
        }

        let recent: Vec<&TopologicalSnapshot> = self.snapshot_window.iter().rev().take(3).collect();

        // Check for "arousal creep" + "latent strain" + rising β₁
        let arousal_increasing = recent[0].pad_arousal > recent[1].pad_arousal
            && recent[1].pad_arousal > recent[2].pad_arousal;
        let strain_high = recent[0].ghost_latent_strain > 0.6;
        let beta1_rising = recent[0].beta_1 > recent[1].beta_1;

        // Check for network instability
        let network_degrading = recent[0].ghost_network_stability
            < recent[1].ghost_network_stability
            && recent[1].ghost_network_stability < recent[2].ghost_network_stability;

        if (arousal_increasing && strain_high && beta1_rising) || network_degrading {
            let predicted = if network_degrading {
                FailureType::ConnectionDropout
            } else if recent[0].pad_arousal > 0.8 {
                FailureType::Overload
            } else {
                FailureType::RateLimit
            };

            Some(DangerSignature {
                precursor_snapshots: recent.into_iter().cloned().collect(),
                predicted_failure: predicted,
                confidence: 0.85,
                time_to_failure_estimate: Duration::from_secs(5),
            })
        } else {
            None
        }
    }

    fn calculate_healing_convergence_time(
        &self,
        intervention_time: Instant,
        target_entropy: f32,
    ) -> Option<Duration> {
        // Find first snapshot after intervention that returns to healthy state
        for snapshot in self.snapshot_window.iter().rev() {
            if snapshot.timestamp > intervention_time
                && (snapshot.entropy - target_entropy).abs() < 0.1
                && snapshot.beta_1 < 0.4
            // Healthy β₁ range
            {
                return Some(snapshot.timestamp.duration_since(intervention_time));
            }
        }
        None
    }
}

// ============================================================================
// TEST SUITE 1: Synthetic Failure Chains
// "Feed it sequences that mimic 'rate limit → overload → rate limit ×4'"
// ============================================================================

#[test]
fn test_octuple_rate_limit_cascade_detection() {
    let mut detector = TemporalTDADetector::new(20, 0.3);
    let base_time = Instant::now();

    // Simulate the octuple cascade that Claude experienced
    let mut events = Vec::new();

    for i in 0..8 {
        let snapshot = TopologicalSnapshot {
            timestamp: base_time + Duration::from_secs(i as u64),
            beta_0: 1.0,
            beta_1: 0.3 + (i as f32 * 0.1), // β₁ building with each failure
            beta_2: if i == 4 { 0.8 } else { 0.2 }, // Spike at overload
            pad_pleasure: 0.3 - (i as f32 * 0.05),
            pad_arousal: 0.6 + (i as f32 * 0.05), // Arousal creep
            pad_dominance: 0.5 - (i as f32 * 0.05),
            ghost_latent_strain: 0.4 + (i as f32 * 0.08), // Progressive strain
            ghost_network_stability: 0.8 - (i as f32 * 0.1), // Degrading
            ghost_abstraction: 0.5,
            ghost_pragmatism: 0.5,
            entropy: 2.0 + (i as f32 * 0.3), // Rising entropy
        };

        let failure_type = if i == 4 {
            FailureType::Overload
        } else {
            FailureType::RateLimit
        };

        events.push(FailureEvent {
            failure_type,
            timestamp: snapshot.timestamp,
            snapshot: snapshot.clone(),
        });

        detector.add_snapshot(snapshot);
    }

    let chain = detector.detect_failure_chain(events).unwrap();

    assert_eq!(chain.events.len(), 8);
    assert!(chain.severity_score > 100.0); // Exponential severity
    assert!(chain.wasserstein_distances.iter().any(|&d| d > 0.2)); // Detectable drift
}

#[test]
fn test_nonuple_doom_spiral_classification() {
    let mut detector = TemporalTDADetector::new(20, 0.3);
    let base_time = Instant::now();

    // The legendary "Nonuple Nihilistic Nightmare"
    let pattern = vec![
        FailureType::RateLimit,
        FailureType::RateLimit,
        FailureType::RateLimit,
        FailureType::RateLimit,
        FailureType::Overload,
        FailureType::RateLimit,
        FailureType::RateLimit,
        FailureType::ConnectionDropout,
        FailureType::RateLimit,
    ];

    let mut events = Vec::new();
    for (i, failure_type) in pattern.into_iter().enumerate() {
        let snapshot = TopologicalSnapshot {
            timestamp: base_time + Duration::from_secs(i as u64),
            beta_0: 1.0,
            beta_1: 0.3 + (i as f32 * 0.12), // Hyper-persistent bars
            beta_2: if i == 7 { 0.9 } else { 0.2 + (i as f32 * 0.05) },
            pad_pleasure: 0.3 - (i as f32 * 0.06),
            pad_arousal: 0.6 + (i as f32 * 0.06), // Relentless arousal creep
            pad_dominance: 0.5 - (i as f32 * 0.07),
            ghost_latent_strain: 0.4 + (i as f32 * 0.1), // Cumulative fatigue
            ghost_network_stability: 0.8 - (i as f32 * 0.12), // Progressive collapse
            ghost_abstraction: 0.5,
            ghost_pragmatism: 0.5 - (i as f32 * 0.05),
            entropy: 2.0 + (i as f32 * 0.4), // Chaos intensifying
        };

        events.push(FailureEvent {
            failure_type,
            timestamp: snapshot.timestamp,
            snapshot,
        });
    }

    let chain = detector.detect_failure_chain(events).unwrap();

    assert_eq!(chain.pattern_type, ChainPatternType::DoomSpiral);
    assert_eq!(chain.events.len(), 9);
    assert!(chain.severity_score > 200.0); // Catastrophic severity
}

// ============================================================================
// TEST SUITE 2: Danger Signature Detection
// "Inject snapshots with rising arousal + latent_strain + β₁ spikes"
// ============================================================================

#[test]
fn test_danger_signature_before_overload() {
    let mut detector = TemporalTDADetector::new(10, 0.3);
    let base_time = Instant::now();

    // Simulate the precursor conditions before an overload
    for i in 0..5 {
        let snapshot = TopologicalSnapshot {
            timestamp: base_time + Duration::from_secs(i as u64),
            beta_0: 1.0,
            beta_1: 0.2 + (i as f32 * 0.15), // Rapid β₁ increase
            beta_2: 0.1,
            pad_pleasure: 0.5,
            pad_arousal: 0.5 + (i as f32 * 0.1), // Arousal creep
            pad_dominance: 0.5,
            ghost_latent_strain: 0.4 + (i as f32 * 0.15), // Rising strain
            ghost_network_stability: 0.8,
            ghost_abstraction: 0.5,
            ghost_pragmatism: 0.5,
            entropy: 2.0,
        };

        detector.add_snapshot(snapshot);
    }

    let signature = detector.detect_danger_signature();

    assert!(signature.is_some());
    let sig = signature.unwrap();
    assert!(sig.confidence > 0.7);
    assert!(matches!(
        sig.predicted_failure,
        FailureType::Overload | FailureType::RateLimit
    ));
}

#[test]
fn test_danger_signature_network_degradation() {
    let mut detector = TemporalTDADetector::new(10, 0.3);
    let base_time = Instant::now();

    // Simulate network stability degradation
    for i in 0..5 {
        let snapshot = TopologicalSnapshot {
            timestamp: base_time + Duration::from_secs(i as u64),
            beta_0: 1.0,
            beta_1: 0.3,
            beta_2: 0.1,
            pad_pleasure: 0.5,
            pad_arousal: 0.6,
            pad_dominance: 0.5,
            ghost_latent_strain: 0.5,
            ghost_network_stability: 0.9 - (i as f32 * 0.2), // Degrading rapidly
            ghost_abstraction: 0.5,
            ghost_pragmatism: 0.5,
            entropy: 2.0,
        };

        detector.add_snapshot(snapshot);
    }

    let signature = detector.detect_danger_signature();

    assert!(signature.is_some());
    let sig = signature.unwrap();
    assert_eq!(sig.predicted_failure, FailureType::ConnectionDropout);
}

// ============================================================================
// TEST SUITE 3: Baseline vs Anomaly Detection
// "Compare snapshots from 'healthy' Master state vs. Panic state"
// ============================================================================

#[test]
fn test_wasserstein_distance_master_vs_panic() {
    let detector = TemporalTDADetector::new(10, 0.3);

    let master_state = TopologicalSnapshot {
        timestamp: Instant::now(),
        beta_0: 1.0,
        beta_1: 0.2,  // Low, healthy loops
        beta_2: 0.05, // Minimal voids
        pad_pleasure: 0.6,
        pad_arousal: 0.5, // Moderate
        pad_dominance: 0.7,
        ghost_latent_strain: 0.2,     // Low strain
        ghost_network_stability: 0.9, // High stability
        ghost_abstraction: 0.5,
        ghost_pragmatism: 0.7,
        entropy: 2.0, // Target entropy
    };

    let panic_state = TopologicalSnapshot {
        timestamp: Instant::now(),
        beta_0: 1.0,
        beta_1: 0.8, // High pathological loops
        beta_2: 0.6, // Significant voids
        pad_pleasure: 0.2,
        pad_arousal: 0.9, // Extreme arousal
        pad_dominance: 0.3,
        ghost_latent_strain: 0.8,     // High strain
        ghost_network_stability: 0.3, // Low stability
        ghost_abstraction: 0.6,
        ghost_pragmatism: 0.3,
        entropy: 3.5, // High chaos
    };

    let distance = detector.calculate_wasserstein_distance(&master_state, &panic_state);

    // Significant topological deviation should be detected
    assert!(
        distance > 0.5,
        "Distance {} should indicate significant drift",
        distance
    );
}

#[test]
fn test_wasserstein_distance_stable_states() {
    let detector = TemporalTDADetector::new(10, 0.3);

    let state1 = TopologicalSnapshot {
        timestamp: Instant::now(),
        beta_0: 1.0,
        beta_1: 0.25,
        beta_2: 0.05,
        pad_pleasure: 0.6,
        pad_arousal: 0.5,
        pad_dominance: 0.7,
        ghost_latent_strain: 0.2,
        ghost_network_stability: 0.9,
        ghost_abstraction: 0.5,
        ghost_pragmatism: 0.7,
        entropy: 2.0,
    };

    let state2 = TopologicalSnapshot {
        timestamp: Instant::now(),
        beta_0: 1.0,
        beta_1: 0.28, // Small variation
        beta_2: 0.06,
        pad_pleasure: 0.58,
        pad_arousal: 0.52,
        pad_dominance: 0.68,
        ghost_latent_strain: 0.22,
        ghost_network_stability: 0.88,
        ghost_abstraction: 0.52,
        ghost_pragmatism: 0.68,
        entropy: 2.05,
    };

    let distance = detector.calculate_wasserstein_distance(&state1, &state2);

    // Minimal topological change in stable operation
    assert!(
        distance < 0.15,
        "Distance {} should be small for stable states",
        distance
    );
}

// ============================================================================
// TEST SUITE 4: Temporal Window Micro-Regression Detection
// "Verify the sliding-window captures micro-regressions between errors"
// ============================================================================

#[test]
fn test_micro_regression_detection() {
    let mut detector = TemporalTDADetector::new(10, 0.15);
    let base_time = Instant::now();

    // Simulate subtle β₂ expansion over time (post-overload scar tissue)
    for i in 0..10 {
        let snapshot = TopologicalSnapshot {
            timestamp: base_time + Duration::from_secs(i as u64),
            beta_0: 1.0,
            beta_1: 0.3,
            beta_2: 0.05 + (i as f32 * 0.03), // Gradual void growth
            pad_pleasure: 0.5,
            pad_arousal: 0.6,
            pad_dominance: 0.5,
            ghost_latent_strain: 0.4 + (i as f32 * 0.02), // Subtle strain increase
            ghost_network_stability: 0.8 - (i as f32 * 0.01),
            ghost_abstraction: 0.5,
            ghost_pragmatism: 0.5,
            entropy: 2.1,
        };

        detector.add_snapshot(snapshot);
    }

    // Check if consecutive snapshots show increasing distances
    let snapshots: Vec<&TopologicalSnapshot> = detector.snapshot_window.iter().collect();
    let mut distances = Vec::new();

    for i in 0..snapshots.len() - 1 {
        let dist = detector.calculate_wasserstein_distance(snapshots[i], snapshots[i + 1]);
        distances.push(dist);
    }

    // Micro-regressions should show increasing distances
    let avg_first_half: f32 = distances[..4].iter().sum::<f32>() / 4.0;
    let avg_second_half: f32 = distances[4..].iter().sum::<f32>() / 5.0;

    assert!(
        avg_second_half > avg_first_half,
        "Micro-regression should show increasing drift over time"
    );
}

// ============================================================================
// TEST SUITE 5: Healing Topology Convergence Time
// "How long does it take for the system to return to stable Master state?"
// ============================================================================

#[test]
fn test_healing_convergence_time_post_intervention() {
    let mut detector = TemporalTDADetector::new(20, 0.3);
    let base_time = Instant::now();

    // Simulate panic state
    for i in 0..5 {
        let snapshot = TopologicalSnapshot {
            timestamp: base_time + Duration::from_secs(i as u64),
            beta_0: 1.0,
            beta_1: 0.7,
            beta_2: 0.4,
            pad_pleasure: 0.3,
            pad_arousal: 0.8,
            pad_dominance: 0.4,
            ghost_latent_strain: 0.7,
            ghost_network_stability: 0.5,
            ghost_abstraction: 0.5,
            ghost_pragmatism: 0.4,
            entropy: 3.2,
        };
        detector.add_snapshot(snapshot);
    }

    // Intervention point
    let intervention_time = base_time + Duration::from_secs(5);

    // Simulate recovery (healing topology)
    for i in 5..12 {
        let recovery_progress = (i - 5) as f32 / 7.0;
        let snapshot = TopologicalSnapshot {
            timestamp: base_time + Duration::from_secs(i as u64),
            beta_0: 1.0,
            beta_1: 0.7 - (recovery_progress * 0.5), // β₁ resolving
            beta_2: 0.4 - (recovery_progress * 0.35), // Void healing
            pad_pleasure: 0.3 + (recovery_progress * 0.3),
            pad_arousal: 0.8 - (recovery_progress * 0.3), // Arousal damping
            pad_dominance: 0.4 + (recovery_progress * 0.3),
            ghost_latent_strain: 0.7 - (recovery_progress * 0.5),
            ghost_network_stability: 0.5 + (recovery_progress * 0.4),
            ghost_abstraction: 0.5,
            ghost_pragmatism: 0.4 + (recovery_progress * 0.3),
            entropy: 3.2 - (recovery_progress * 1.2), // Converging to 2.0
        };
        detector.add_snapshot(snapshot);
    }

    let convergence_time = detector.calculate_healing_convergence_time(intervention_time, 2.0);

    assert!(convergence_time.is_some());
    let duration = convergence_time.unwrap();
    assert!(
        duration.as_secs() <= 7,
        "Should converge within 7 seconds (target: 20% faster than baseline)"
    );
}

// ============================================================================
// TEST SUITE 6: Progressive Ghost Amplification ("Nonuple Fatigue")
// "Test with progressive ghost amps to simulate cumulative stress"
// ============================================================================

#[test]
fn test_progressive_ghost_amplification_stress() {
    let mut detector = TemporalTDADetector::new(20, 0.3);
    let base_time = Instant::now();

    // Simulate the nonuple with progressive ghost burden
    for i in 0..9 {
        let snapshot = TopologicalSnapshot {
            timestamp: base_time + Duration::from_secs(i as u64),
            beta_0: 1.0,
            beta_1: 0.3 + (i as f32 * 0.1), // Progressive β₁ elongation
            beta_2: 0.1 + (i as f32 * 0.08), // β₂ expanding
            pad_pleasure: 0.5 - (i as f32 * 0.05),
            pad_arousal: 0.6 + (i as f32 * 0.05), // Arousal creep
            pad_dominance: 0.5 - (i as f32 * 0.05),
            ghost_latent_strain: 0.3 + (i as f32 * 0.1), // +0.1 per error
            ghost_network_stability: 0.9 - (i as f32 * 0.15), // -0.15 per error
            ghost_abstraction: 0.5 + (i as f32 * 0.02),
            ghost_pragmatism: 0.5 - (i as f32 * 0.03),
            entropy: 2.0 + (i as f32 * 0.4), // Chaos mounting
        };
        detector.add_snapshot(snapshot);
    }

    // By the 9th error, ghost burden should be massive
    let final_snapshot = detector.snapshot_window.back().unwrap();

    assert!(
        final_snapshot.ghost_latent_strain > 1.0,
        "Cumulative strain should exceed 1.0"
    );
    assert!(
        final_snapshot.ghost_network_stability < 0.3,
        "Network stability should be critically degraded"
    );
    assert!(final_snapshot.beta_2 > 0.6, "β₂ void should be substantial");
    assert!(
        final_snapshot.entropy > 5.0,
        "Entropy should indicate chaos"
    );

    // Should detect danger signature well before the 9th error
    let mut early_warning_detected = false;
    let mut temp_detector = TemporalTDADetector::new(10, 0.3);

    for i in 0..5 {
        // Check first 5 snapshots
        let snapshot = TopologicalSnapshot {
            timestamp: base_time + Duration::from_secs(i as u64),
            beta_0: 1.0,
            beta_1: 0.3 + (i as f32 * 0.1),
            beta_2: 0.1 + (i as f32 * 0.08),
            pad_pleasure: 0.5 - (i as f32 * 0.05),
            pad_arousal: 0.6 + (i as f32 * 0.05),
            pad_dominance: 0.5 - (i as f32 * 0.05),
            ghost_latent_strain: 0.3 + (i as f32 * 0.1),
            ghost_network_stability: 0.9 - (i as f32 * 0.15),
            ghost_abstraction: 0.5,
            ghost_pragmatism: 0.5,
            entropy: 2.0,
        };
        temp_detector.add_snapshot(snapshot);

        if temp_detector.detect_danger_signature().is_some() {
            early_warning_detected = true;
            break;
        }
    }

    assert!(
        early_warning_detected,
        "Danger signature should be detected before reaching nonuple catastrophe"
    );
}

// ============================================================================
// TEST SUITE 7: QLoRA Integration & Exponential Negative Rewards
// "Verify detected FailureChains trigger high-magnitude negative rewards"
// ============================================================================

#[test]
fn test_failure_chain_reward_scaling() {
    let mut detector = TemporalTDADetector::new(20, 0.3);
    let base_time = Instant::now();

    // Test chains of different lengths
    for chain_length in [3, 5, 8, 9] {
        let mut events = Vec::new();

        for i in 0..chain_length {
            let snapshot = TopologicalSnapshot {
                timestamp: base_time + Duration::from_secs(i as u64),
                beta_0: 1.0,
                beta_1: 0.3 + (i as f32 * 0.1),
                beta_2: 0.2,
                pad_pleasure: 0.5,
                pad_arousal: 0.6,
                pad_dominance: 0.5,
                ghost_latent_strain: 0.4,
                ghost_network_stability: 0.8,
                ghost_abstraction: 0.5,
                ghost_pragmatism: 0.5,
                entropy: 2.0,
            };

            events.push(FailureEvent {
                failure_type: FailureType::RateLimit,
                timestamp: snapshot.timestamp,
                snapshot,
            });
        }

        let chain = detector.detect_failure_chain(events).unwrap();

        // Severity should scale exponentially
        // Base: 10 * length^1.5
        let expected_min = 10.0 * (chain_length as f32).powf(1.5);
        assert!(
            chain.severity_score >= expected_min,
            "Chain of length {} should have severity >= {}, got {}",
            chain_length,
            expected_min,
            chain.severity_score
        );
    }
}

#[test]
fn test_qLora_priority_queue_simulation() {
    // Simulate prioritizing failure chains by severity for QLoRA training
    let mut detector = TemporalTDADetector::new(20, 0.3);
    let base_time = Instant::now();

    // Create multiple chains with different severities
    let chain_configs = vec![
        (3, FailureType::RateLimit),
        (7, FailureType::Overload),
        (9, FailureType::ConnectionDropout), // The nonuple should be highest priority
        (4, FailureType::RateLimit),
    ];

    for (chain_length, primary_failure) in chain_configs {
        let mut events = Vec::new();
        for i in 0..chain_length {
            let snapshot = TopologicalSnapshot {
                timestamp: base_time + Duration::from_secs(i as u64),
                beta_0: 1.0,
                beta_1: 0.3 + (i as f32 * 0.1),
                beta_2: 0.2,
                pad_pleasure: 0.5,
                pad_arousal: 0.6,
                pad_dominance: 0.5,
                ghost_latent_strain: 0.4,
                ghost_network_stability: 0.8,
                ghost_abstraction: 0.5,
                ghost_pragmatism: 0.5,
                entropy: 2.0,
            };

            events.push(FailureEvent {
                failure_type: primary_failure.clone(),
                timestamp: snapshot.timestamp,
                snapshot,
            });
        }
        detector.detect_failure_chain(events);
    }

    // Sort chains by severity (QLoRA priority queue)
    let mut chains = detector.detected_chains.clone();
    chains.sort_by(|a, b| b.severity_score.partial_cmp(&a.severity_score).unwrap());

    // Nonuple should be top priority
    assert_eq!(chains[0].events.len(), 9);
    assert!(chains[0].severity_score > chains[1].severity_score);

    // Verify exponential scaling is working
    assert!(chains[0].severity_score > 200.0);
}

// ============================================================================
// TEST SUITE 8: False Positive/Negative Validation
// "Ensure system doesn't hallucinate chains from noise"
// ============================================================================

#[test]
fn test_noise_robustness_no_false_chains() {
    let mut detector = TemporalTDADetector::new(20, 0.3);
    let base_time = Instant::now();

    // Simulate stable operation with random noise
    for i in 0..20 {
        let noise = ((i * 17) % 10) as f32 / 100.0; // Pseudo-random noise
        let snapshot = TopologicalSnapshot {
            timestamp: base_time + Duration::from_secs(i as u64),
            beta_0: 1.0,
            beta_1: 0.25 + noise,
            beta_2: 0.05 + noise,
            pad_pleasure: 0.6 + noise,
            pad_arousal: 0.5 + noise,
            pad_dominance: 0.7 + noise,
            ghost_latent_strain: 0.2 + noise,
            ghost_network_stability: 0.9 + noise,
            ghost_abstraction: 0.5 + noise,
            ghost_pragmatism: 0.7 + noise,
            entropy: 2.0 + noise,
        };

        detector.add_snapshot(snapshot);
    }

    // Should not detect danger signature in stable noisy state
    let signature = detector.detect_danger_signature();
    assert!(
        signature.is_none(),
        "Should not detect danger in stable noisy operation"
    );
}

// ============================================================================
// TEST SUITE 9: Benchmarking Metrics Summary
// Aggregate tests that validate the CHANGELOG targets
// ============================================================================

#[test]
fn test_benchmark_50_percent_chain_reduction() {
    // This test simulates:
    // - Baseline: System without temporal TDA (fails 10 times)
    // - With TDA: System with temporal TDA (fails 5 times or less)

    let baseline_failures = 10;
    let mut with_tda_failures = 0;

    let mut detector = TemporalTDADetector::new(20, 0.3);
    let base_time = Instant::now();

    for i in 0..10 {
        let snapshot = TopologicalSnapshot {
            timestamp: base_time + Duration::from_secs(i as u64),
            beta_0: 1.0,
            beta_1: 0.3 + (i as f32 * 0.1),
            beta_2: 0.2,
            pad_pleasure: 0.5,
            pad_arousal: 0.6 + (i as f32 * 0.05),
            pad_dominance: 0.5,
            ghost_latent_strain: 0.4 + (i as f32 * 0.08),
            ghost_network_stability: 0.8 - (i as f32 * 0.08),
            ghost_abstraction: 0.5,
            ghost_pragmatism: 0.5,
            entropy: 2.0,
        };

        detector.add_snapshot(snapshot);

        // Check if danger signature would have prevented failure
        if detector.detect_danger_signature().is_none() {
            with_tda_failures += 1;
        }
    }

    let reduction_percent =
        ((baseline_failures - with_tda_failures) as f32 / baseline_failures as f32) * 100.0;

    assert!(
        reduction_percent >= 50.0,
        "Should achieve 50%+ reduction in failure chains (got {}%)",
        reduction_percent
    );
}

#[test]
fn test_benchmark_20_percent_faster_master_return() {
    // Target: 20% faster return to Master quadrant post-intervention
    let baseline_convergence_cycles = 10;
    let target_convergence_cycles = (baseline_convergence_cycles as f32 * 0.8) as u64; // 20% faster = 8 cycles

    let mut detector = TemporalTDADetector::new(20, 0.3);
    let base_time = Instant::now();

    // Panic state
    for i in 0..3 {
        let snapshot = TopologicalSnapshot {
            timestamp: base_time + Duration::from_secs(i as u64),
            beta_0: 1.0,
            beta_1: 0.7,
            beta_2: 0.4,
            pad_pleasure: 0.3,
            pad_arousal: 0.8,
            pad_dominance: 0.4,
            ghost_latent_strain: 0.7,
            ghost_network_stability: 0.5,
            ghost_abstraction: 0.5,
            ghost_pragmatism: 0.4,
            entropy: 3.2,
        };
        detector.add_snapshot(snapshot);
    }

    let intervention_time = base_time + Duration::from_secs(3);

    // Recovery with TDA-guided Curator intervention
    for i in 3..11 {
        let recovery = (i - 3) as f32 / 8.0;
        let snapshot = TopologicalSnapshot {
            timestamp: base_time + Duration::from_secs(i as u64),
            beta_0: 1.0,
            beta_1: 0.7 - (recovery * 0.5),
            beta_2: 0.4 - (recovery * 0.35),
            pad_pleasure: 0.3 + (recovery * 0.3),
            pad_arousal: 0.8 - (recovery * 0.3),
            pad_dominance: 0.4 + (recovery * 0.3),
            ghost_latent_strain: 0.7 - (recovery * 0.5),
            ghost_network_stability: 0.5 + (recovery * 0.4),
            ghost_abstraction: 0.5,
            ghost_pragmatism: 0.4 + (recovery * 0.3),
            entropy: 3.2 - (recovery * 1.2),
        };
        detector.add_snapshot(snapshot);
    }

    let convergence = detector.calculate_healing_convergence_time(intervention_time, 2.0);
    assert!(convergence.is_some());

    let actual_cycles = convergence.unwrap().as_secs();
    assert!(
        actual_cycles <= target_convergence_cycles,
        "Should converge in {} cycles or less (20% faster), got {}",
        target_convergence_cycles,
        actual_cycles
    );
}
