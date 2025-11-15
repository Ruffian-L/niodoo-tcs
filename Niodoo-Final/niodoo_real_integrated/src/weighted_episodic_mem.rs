//! Weighted Episodic Memory system with multi-factor fitness scoring
//!
//! Implements neuroscience-inspired episodic memory with:
//! - Multi-factor fitness function: temporal decay, PAD emotional weighting,
//!   topological connectivity (Betti β₁), retrieval count, consonance
//! - Three-phase temporal decay dynamics
//! - PAD emotional salience calculation
//! - Integration with ERAG and MCTS systems

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::torus::PadGhostState;

/// Default fitness weights: [temporal, pad_salience, beta1_connectivity, retrieval_count, consonance, resource_penalty]
/// Resource penalty weight (w₆) is subtracted, so it reduces fitness when resources are constrained
pub const DEFAULT_FITNESS_WEIGHTS: [f32; 6] = [0.20, 0.18, 0.18, 0.13, 0.18, 0.13];
/// Legacy 5-weight default (for backward compatibility)
pub const DEFAULT_FITNESS_WEIGHTS_LEGACY: [f32; 5] = [0.25, 0.20, 0.20, 0.15, 0.20];

/// Temporal decay phase constants (tau values in days)
#[derive(Debug, Clone, Copy)]
pub struct TemporalDecayConfig {
    /// Phase 1 (0-1 days): Rapid initial forgetting during active consolidation
    pub phase1_tau: f64, // 0.3 days
    /// Phase 2 (1-9 days): Stable retention during systems consolidation
    pub phase2_tau: f64, // 5.0 days
    /// Phase 3 (9+ days): Schema-dependent neocortical storage
    pub phase3_tau: f64, // 2.0 days
    /// Persistence entropy scaling factor for adaptive decay
    /// Higher values = more impact of entropy on decay rate
    pub persistence_entropy_alpha: f64, // 0.3 default
}

impl Default for TemporalDecayConfig {
    fn default() -> Self {
        Self {
            phase1_tau: 0.3,
            phase2_tau: 5.0,
            phase3_tau: 2.0,
            persistence_entropy_alpha: 0.3,
        }
    }
}

/// Weighted episodic memory metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedMemoryMetadata {
    /// Overall fitness score (0.0-1.0)
    pub fitness_score: f32,
    /// Number of times this memory has been retrieved
    pub retrieval_count: u32,
    /// Last access timestamp
    pub last_accessed: DateTime<Utc>,
    /// Consolidation level (0.0-1.0), higher = more consolidated
    pub consolidation_level: f32,
    /// Betti β₁ connectivity score (topological feature)
    pub beta_1_connectivity: f32,
    /// Consonance score (graph-theoretic coherence)
    pub consonance_score: f32,
    /// Community ID from graph clustering
    pub community_id: Option<u32>,
    /// H1 trust score from persistent homology (loops indicate consistency)
    #[serde(default)]
    pub h1_trust_score: f32,
    /// H2 anomaly score from persistent homology (voids indicate gaps)
    #[serde(default)]
    pub h2_anomaly_score: f32,
    /// Persistence entropy for adaptive decay
    #[serde(default)]
    pub persistence_entropy: f64,
}

impl Default for WeightedMemoryMetadata {
    fn default() -> Self {
        Self {
            fitness_score: 0.5,
            retrieval_count: 0,
            last_accessed: Utc::now(),
            consolidation_level: 0.0,
            beta_1_connectivity: 0.0,
            consonance_score: 0.0,
            community_id: None,
            h1_trust_score: 0.5,
            h2_anomaly_score: 0.0,
            persistence_entropy: 0.0,
        }
    }
}

/// Calculate PAD emotional salience
///
/// Formula: (2×arousal + |pleasure| + 0.5×normalized_dominance) / 3.5
///
/// Arousal most strongly predicts encoding strength (flashbulb memory effect),
/// pleasure biases retrieval direction, dominance modulates confidence.
pub fn calculate_pad_salience(pad_state: &PadGhostState) -> f32 {
    // Extract PAD dimensions from PadGhostState (7D torus)
    // Assume: pad[0] = pleasure, pad[1] = arousal, pad[2] = dominance (normalized)
    let pleasure = pad_state.pad[0] as f32;
    let arousal = pad_state.pad[1] as f32;
    let dominance = pad_state.pad[2] as f32;

    // Normalize dominance to [0, 1] range (assuming [-1, 1] input)
    let normalized_dominance = (dominance + 1.0) / 2.0;

    // Calculate salience
    let salience = (2.0 * arousal.abs() + pleasure.abs() + 0.5 * normalized_dominance) / 3.5;

    // Clamp to [0, 1]
    salience.clamp(0.0, 1.0)
}

/// Calculate adaptive decay factor from persistence entropy
///
/// Low entropy (stable agents) = slower decay
/// Formula: factor = 1 + alpha * (1 - normalized_entropy)
/// where normalized_entropy is clamped to [0, 1]
pub fn calculate_adaptive_decay_factor(entropy: f64) -> f64 {
    // Normalize entropy: assume max entropy around 5.0 (log of reasonable number of features)
    let normalized_entropy = (entropy / 5.0).min(1.0).max(0.0);
    // Low entropy = stable = slower decay
    1.0 + 0.3 * (1.0 - normalized_entropy)
}

/// Calculate temporal decay component with three-phase dynamics
///
/// Phase 1 (0-1 days): τ = 0.3 for rapid initial forgetting
/// Phase 2 (1-9 days): τ = 5.0 for stable retention
/// Phase 3 (9+ days): τ = 2.0 for schema-dependent storage
///
/// Consolidation extends time constants: τ_effective = τ × (1 + 0.5 × consolidation_level)
/// Persistence entropy extends time constants: τ_effective = τ × (1 + alpha × (1 - normalized_entropy))
pub fn calculate_temporal_decay(
    age_days: f64,
    consolidation_level: f32,
    config: &TemporalDecayConfig,
    persistence_entropy: Option<f64>,
) -> f32 {
    let tau = if age_days < 1.0 {
        config.phase1_tau
    } else if age_days < 9.0 {
        config.phase2_tau
    } else {
        config.phase3_tau
    };

    // Apply consolidation extension
    let mut tau_effective = tau * (1.0 + 0.5 * consolidation_level as f64);

    // Apply persistence entropy extension (stable agents decay slower)
    if let Some(entropy) = persistence_entropy {
        let entropy_factor = calculate_adaptive_decay_factor(entropy);
        tau_effective *= entropy_factor;
    }

    // Exponential decay: e^(-age/tau)
    let decay = (-age_days / tau_effective).exp();

    decay as f32
}

/// Calculate retrieval count component with logarithmic spacing effect
///
/// Formula: log(1 + retrieval_count)
/// Implements spacing effect - one of cognitive psychology's most robust findings
pub fn calculate_retrieval_weight(retrieval_count: u32) -> f32 {
    (1.0 + retrieval_count as f32).ln()
}

/// Normalize retrieval weight to [0, 1] range
///
/// Typical max retrieval count ~100, so log(101) ≈ 4.6
/// Normalize by dividing by 5.0 (slightly above max expected)
pub fn normalized_retrieval_weight(retrieval_count: u32) -> f32 {
    calculate_retrieval_weight(retrieval_count) / 5.0
}

/// Calculate resource penalty Res(m) from resource availability
///
/// Returns a penalty value (0.0 = no penalty, 1.0 = maximum penalty) based on resource constraints.
/// Lower resource availability = higher penalty.
pub fn calculate_resource_penalty(
    resource_availability: Option<&crate::resource_budget::ResourceAvailability>,
) -> f32 {
    match resource_availability {
        Some(avail) => {
            // Use minimum availability as the bottleneck
            let min_avail = avail.minimum();
            // Convert availability to penalty: 1.0 = no penalty, 0.0 = max penalty
            1.0 - min_avail
        }
        None => 0.0, // No resource tracking = no penalty
    }
}

/// Calculate multi-factor fitness score
///
/// Formula: F(m) = w₁·T(m) + w₂·PAD(m) + w₃·β₁(m) + w₄·R(m) + w₅·C(m) - w₆·Res(m)
///
/// Where:
/// - T(m): Temporal decay (with adaptive persistence entropy scaling)
/// - PAD(m): Emotional salience
/// - β₁(m): Topological connectivity (uses H1 trust score if available, else graph-based β₁)
/// - R(m): Retrieval frequency
/// - C(m): Consonance
/// - Res(m): Resource penalty (0.0 = no penalty, 1.0 = max penalty)
///
/// Weights should sum to 1.0 for proper normalization (w₆ is subtracted, so it reduces fitness)
pub fn calculate_fitness_score(
    age_days: f64,
    pad_state: &PadGhostState,
    retrieval_count: u32,
    beta_1_connectivity: f32,
    consonance_score: f32,
    consolidation_level: f32,
    weights: &[f32; 6],
    temporal_config: &TemporalDecayConfig,
    resource_availability: Option<&crate::resource_budget::ResourceAvailability>,
    persistence_entropy: Option<f64>,
    h1_trust_score: Option<f32>,
    h2_anomaly_score: Option<f32>,
) -> f32 {
    // Use adaptive temporal decay with persistence entropy
    let temporal = calculate_temporal_decay(
        age_days,
        consolidation_level,
        temporal_config,
        persistence_entropy,
    );
    let pad_salience = calculate_pad_salience(pad_state);
    let retrieval_weight = normalized_retrieval_weight(retrieval_count);

    // Use H1 trust score if available, otherwise fall back to graph-based β₁
    let beta1_normalized = if let Some(h1_score) = h1_trust_score {
        h1_score.clamp(0.0, 1.0)
    } else {
        beta_1_connectivity.clamp(0.0, 1.0)
    };

    let consonance_normalized = consonance_score.clamp(0.0, 1.0);

    // Calculate resource penalty
    let resource_penalty = calculate_resource_penalty(resource_availability);

    // Weighted combination: positive terms - resource penalty
    let mut fitness = weights[0] * temporal
        + weights[1] * pad_salience
        + weights[2] * beta1_normalized
        + weights[3] * retrieval_weight
        + weights[4] * consonance_normalized
        - weights[5] * resource_penalty;

    // Subtract H2 anomaly score as penalty if available
    if let Some(h2_score) = h2_anomaly_score {
        fitness -= 0.1 * h2_score; // Small penalty for anomalies
    }

    fitness.clamp(0.0, 1.0)
}

/// Legacy fitness function with 5 weights (for backward compatibility)
pub fn calculate_fitness_score_legacy(
    age_days: f64,
    pad_state: &PadGhostState,
    retrieval_count: u32,
    beta_1_connectivity: f32,
    consonance_score: f32,
    consolidation_level: f32,
    weights: &[f32; 5],
    temporal_config: &TemporalDecayConfig,
) -> f32 {
    // Convert 5 weights to 6 weights (w₆ = 0.0)
    let weights_6 = [
        weights[0], weights[1], weights[2], weights[3], weights[4], 0.0,
    ];
    calculate_fitness_score(
        age_days,
        pad_state,
        retrieval_count,
        beta_1_connectivity,
        consonance_score,
        consolidation_level,
        &weights_6,
        temporal_config,
        None, // resource_availability
        None, // persistence_entropy
        None, // h1_trust_score
        None, // h2_anomaly_score
    )
}

/// Calculate age in days from timestamp
pub fn age_in_days(timestamp: &DateTime<Utc>) -> f64 {
    let now = Utc::now();
    let duration = now.signed_duration_since(*timestamp);
    duration.num_seconds() as f64 / 86400.0 // Convert to days
}

/// Update memory metadata after retrieval
pub fn update_retrieval_stats(metadata: &mut WeightedMemoryMetadata) {
    metadata.retrieval_count += 1;
    metadata.last_accessed = Utc::now();
}

/// Initialize memory metadata for new memory
pub fn initialize_memory_metadata(
    pad_state: &PadGhostState,
    consolidation_level: f32,
) -> WeightedMemoryMetadata {
    WeightedMemoryMetadata {
        fitness_score: 0.5, // Initial neutral score
        retrieval_count: 0,
        last_accessed: Utc::now(),
        consolidation_level,
        beta_1_connectivity: 0.0, // Will be computed later
        consonance_score: 0.0,    // Will be computed later
        community_id: None,
        h1_trust_score: 0.5,      // Initial neutral trust
        h2_anomaly_score: 0.0,    // No anomalies initially
        persistence_entropy: 0.0, // Will be computed later
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::torus::PadGhostState;

    fn create_test_pad_state() -> PadGhostState {
        PadGhostState {
            pad: [0.5, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0], // High arousal (pad[1]=0.8)
            entropy: 0.5,
            mu: [0.5, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0],
            sigma: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        }
    }

    #[test]
    fn test_pad_salience_high_arousal() {
        let pad_state = create_test_pad_state();
        let salience = calculate_pad_salience(&pad_state);
        // High arousal should produce high salience
        assert!(salience > 0.4);
    }

    #[test]
    fn test_temporal_decay_phase1() {
        let config = TemporalDecayConfig::default();
        let decay = calculate_temporal_decay(0.5, 0.0, &config, None); // 0.5 days, no consolidation
                                                                       // Should decay significantly in phase 1
        assert!(decay < 0.5);
    }

    #[test]
    fn test_temporal_decay_consolidation() {
        let config = TemporalDecayConfig::default();
        let decay_no_consolidation = calculate_temporal_decay(1.0, 0.0, &config, None);
        let decay_with_consolidation = calculate_temporal_decay(1.0, 0.5, &config, None);
        // Consolidation should slow decay
        assert!(decay_with_consolidation > decay_no_consolidation);
    }

    #[test]
    fn test_temporal_decay_persistence_entropy() {
        let config = TemporalDecayConfig::default();
        let decay_no_entropy = calculate_temporal_decay(1.0, 0.0, &config, None);
        let decay_low_entropy = calculate_temporal_decay(1.0, 0.0, &config, Some(0.5)); // Low entropy = stable
                                                                                        // Low entropy should slow decay (stable agents decay slower)
        assert!(decay_low_entropy > decay_no_entropy);
    }

    #[test]
    fn test_retrieval_weight() {
        let weight_0 = calculate_retrieval_weight(0);
        let weight_10 = calculate_retrieval_weight(10);
        assert_eq!(weight_0, 0.0);
        assert!(weight_10 > weight_0);
    }

    #[test]
    fn test_fitness_calculation() {
        let pad_state = create_test_pad_state();
        let weights = DEFAULT_FITNESS_WEIGHTS;
        let config = TemporalDecayConfig::default();

        let fitness = calculate_fitness_score(
            1.0, // 1 day old
            &pad_state, 5,   // 5 retrievals
            0.6, // beta_1 connectivity
            0.7, // consonance
            0.2, // consolidation level
            &weights, &config, None, // resource_availability
            None, // persistence_entropy
            None, // h1_trust_score
            None, // h2_anomaly_score
        );

        assert!(fitness >= 0.0 && fitness <= 1.0);
    }
}
