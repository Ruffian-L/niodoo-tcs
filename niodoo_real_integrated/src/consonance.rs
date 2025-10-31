//! Consonance/Dissonance Detection Module
//! Computes alignment metrics from multiple system signals to detect
//! when ideas resonate (consonance) vs clash (dissonance)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::compass::{CompassOutcome, CompassQuadrant};
use crate::curator::CuratedResponse;
use crate::erag::CollapseResult;
use crate::tcs_analysis::TopologicalSignature;
use crate::torus::PadGhostState;

/// Consonance score from 0.0 (dissonant) to 1.0 (consonant)
/// Represents how aligned multiple signals are - high consonance = "this is right"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsonanceMetrics {
    pub score: f64,              // 0.0 (dissonant) → 1.0 (consonant)
    pub sources: Vec<ConsonanceSource>,  // Which signals contributed
    pub confidence: f64,          // How certain we are (0.0-1.0)
    pub dissonance_score: f64,    // Explicit dissonance (0.0-1.0, inverse of score)
}

/// Individual signal sources contributing to consonance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsonanceSource {
    EmotionalCoherence(f64),      // PAD stability
    TopologicalConsistency(f64),  // Knot complexity alignment
    ERAGRelevance(f64),           // Memory retrieval similarity
    CompassTransition(f64),       // Smooth state transitions
    CuratorQuality(f64),         // Quality score from curator
}

impl ConsonanceSource {
    pub fn score(&self) -> f64 {
        match self {
            ConsonanceSource::EmotionalCoherence(s) => *s,
            ConsonanceSource::TopologicalConsistency(s) => *s,
            ConsonanceSource::ERAGRelevance(s) => *s,
            ConsonanceSource::CompassTransition(s) => *s,
            ConsonanceSource::CuratorQuality(s) => *s,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            ConsonanceSource::EmotionalCoherence(_) => "emotional_coherence",
            ConsonanceSource::TopologicalConsistency(_) => "topological_consistency",
            ConsonanceSource::ERAGRelevance(_) => "erag_relevance",
            ConsonanceSource::CompassTransition(_) => "compass_transition",
            ConsonanceSource::CuratorQuality(_) => "curator_quality",
        }
    }
}

/// Compute consonance metrics from all available signals
pub fn compute_consonance(
    pad_state: &PadGhostState,
    compass: &CompassOutcome,
    erag_collapse: &CollapseResult,
    topology: &TopologicalSignature,
    curator: Option<&CuratedResponse>,
    last_compass: Option<&CompassOutcome>,
) -> ConsonanceMetrics {
    let mut sources = Vec::new();

    // 1. Emotional Coherence: PAD stability
    // High variance = dissonance, low variance = consonance
    let pad_variance = compute_pad_variance(pad_state);
    let emotional_coherence = (1.0 - pad_variance.min(1.0)).max(0.0);
    sources.push(ConsonanceSource::EmotionalCoherence(emotional_coherence));

    // 2. Topological Consistency: Knot complexity alignment
    // Stable knot complexity = consonance, spikes = dissonance
    let topological_consistency = compute_topological_consistency(topology, pad_state);
    sources.push(ConsonanceSource::TopologicalConsistency(topological_consistency));

    // 3. ERAG Relevance: Memory retrieval quality
    // High similarity scores = relevant memories = consonance
    let erag_relevance = compute_erag_relevance(erag_collapse);
    sources.push(ConsonanceSource::ERAGRelevance(erag_relevance));

    // 4. Compass Transition: Smooth state changes
    // Smooth transitions = consonance, chaotic = dissonance
    let compass_transition = compute_compass_transition(compass, last_compass);
    sources.push(ConsonanceSource::CompassTransition(compass_transition));

    // 5. Curator Quality: Response quality score
    // High quality = consonance, low quality = dissonance
    let curator_quality = if let Some(cur) = curator {
        // Curator quality is implicit in learned flag and reason
        // If learned=true and refined_response is longer than original, high quality
        let quality_score = if cur.learned {
            // Boost quality if curator learned something
            0.85 + (cur.refined_response.len() as f64 / 1000.0).min(0.15)
        } else {
            // Base quality on refined response length (heuristic)
            (cur.refined_response.len() as f64 / 500.0).min(0.7)
        };
        quality_score.min(1.0).max(0.0)
    } else {
        0.5 // Neutral if no curator
    };
    sources.push(ConsonanceSource::CuratorQuality(curator_quality));

    // Weighted average of all sources
    // Weights: emotional=0.25, topological=0.20, ERAG=0.25, compass=0.20, curator=0.10
    let weights = [0.25, 0.20, 0.25, 0.20, 0.10];
    let weighted_sum: f64 = sources
        .iter()
        .zip(weights.iter())
        .map(|(src, w)| src.score() * w)
        .sum();

    let score = weighted_sum.clamp(0.0, 1.0);
    let dissonance_score = (1.0 - score).clamp(0.0, 1.0);

    // Confidence based on how many sources we have and their agreement
    let confidence = compute_confidence(&sources);

    ConsonanceMetrics {
        score,
        sources,
        confidence,
        dissonance_score,
    }
}

/// Compute PAD variance as measure of emotional instability
fn compute_pad_variance(pad_state: &PadGhostState) -> f64 {
    // Compute variance of PAD values
    let pad_mean = (pad_state.pad[0] + pad_state.pad[1] + pad_state.pad[2]) / 3.0;
    let variance = ((pad_state.pad[0] - pad_mean).powi(2)
        + (pad_state.pad[1] - pad_mean).powi(2)
        + (pad_state.pad[2] - pad_mean).powi(2))
        / 3.0;
    variance.sqrt() // Standard deviation
}

/// Compute topological consistency based on knot complexity and entropy alignment
fn compute_topological_consistency(topology: &TopologicalSignature, pad_state: &PadGhostState) -> f64 {
    // Stable entropy + reasonable knot complexity = consonance
    let entropy_stable = if pad_state.entropy >= 1.8 && pad_state.entropy <= 2.2 {
        1.0 // Target entropy range
    } else {
        // Distance from target
        1.0 - (pad_state.entropy - 2.0).abs() / 2.0
    };

    // Knot complexity should be moderate (not too high, not too low)
    let knot_consistency = if topology.knot_complexity >= 1.0 && topology.knot_complexity <= 5.0 {
        1.0
    } else {
        1.0 - (topology.knot_complexity - 3.0).abs() / 10.0
    };

    // Spectral gap should be positive (topological stability)
    let spectral_consistency = if topology.spectral_gap > 0.0 {
        (topology.spectral_gap / 2.0).min(1.0)
    } else {
        0.0
    };

    // Average of all consistency measures
    ((entropy_stable + knot_consistency + spectral_consistency) / 3.0).clamp(0.0, 1.0)
}

/// Compute ERAG relevance based on retrieval similarity scores
fn compute_erag_relevance(collapse: &CollapseResult) -> f64 {
    // High average similarity = good retrieval = consonance
    // If no hits, low relevance
    if collapse.top_hits.is_empty() {
        return 0.3; // Low but not zero
    }

    // Average similarity score indicates relevance
    let relevance = collapse.average_similarity as f64;
    let relevance = relevance.clamp(0.0, 1.0);
    
    // Boost if we have multiple relevant hits
    let hit_bonus = (collapse.top_hits.len() as f64 / 3.0).min(0.2);
    
    (relevance + hit_bonus).min(1.0)
}

/// Compute compass transition smoothness
fn compute_compass_transition(compass: &CompassOutcome, last_compass: Option<&CompassOutcome>) -> f64 {
    // If no previous state, assume neutral
    let Some(last) = last_compass else {
        return 0.5;
    };

    // Smooth transitions = consonance
    // Chaotic transitions = dissonance
    
    match (last.quadrant, compass.quadrant) {
        // Good transitions (consonance)
        (CompassQuadrant::Panic, CompassQuadrant::Discover) => 0.9,
        (CompassQuadrant::Persist, CompassQuadrant::Master) => 0.9,
        (CompassQuadrant::Discover, CompassQuadrant::Master) => 0.95,
        (CompassQuadrant::Master, CompassQuadrant::Discover) => 0.85,
        
        // Bad transitions (dissonance)
        (CompassQuadrant::Master, CompassQuadrant::Panic) => 0.2,
        (CompassQuadrant::Discover, CompassQuadrant::Panic) => 0.3,
        
        // Same state (neutral/slightly positive)
        (a, b) if a == b => {
            if compass.is_threat {
                0.4 // Threat = dissonance
            } else if compass.is_healing {
                0.9 // Healing = consonance
            } else {
                0.6 // Neutral
            }
        }
        
        // Other transitions (neutral)
        _ => 0.5,
    }
}

/// Compute confidence in consonance score based on source agreement
fn compute_confidence(sources: &[ConsonanceSource]) -> f64 {
    if sources.is_empty() {
        return 0.0;
    }

    let scores: Vec<f64> = sources.iter().map(|s| s.score()).collect();
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    
    // Standard deviation - lower = more agreement = higher confidence
    let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
    let std_dev = variance.sqrt();
    
    // Confidence decreases with higher variance
    // Max variance for 0-1 scores is ~0.5, so normalize
    let confidence = 1.0 - (std_dev / 0.5).min(1.0);
    
    // Also boost confidence if we have all sources
    let completeness_bonus = if sources.len() >= 5 { 0.1 } else { 0.0 };
    
    (confidence + completeness_bonus).min(1.0)
}

/// Helper to get consonance score as percentage
impl ConsonanceMetrics {
    pub fn as_percent(&self) -> f64 {
        self.score * 100.0
    }

    pub fn is_consonant(&self, threshold: f64) -> bool {
        self.score >= threshold
    }

    pub fn is_dissonant(&self, threshold: f64) -> bool {
        self.dissonance_score >= threshold
    }

    /// Get source breakdown as a map for logging
    pub fn source_map(&self) -> HashMap<String, f64> {
        self.sources
            .iter()
            .map(|s| (s.name().to_string(), s.score()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compass::CompassOutcome;
    use crate::curator::CuratedResponse;
    use crate::erag::{CollapseResult, EragMemory};
    use crate::tcs_analysis::TopologicalSignature;
    use crate::torus::PadGhostState;
    use std::time::SystemTime;

    fn create_test_pad_state(entropy: f64) -> PadGhostState {
        PadGhostState {
            pad: [0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0],
            mu: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            sigma: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            entropy,
        }
    }

    fn create_test_compass(quadrant: CompassQuadrant) -> CompassOutcome {
        CompassOutcome {
            quadrant,
            is_threat: false,
            is_healing: quadrant == CompassQuadrant::Master,
            mcts_branches: vec![],
            intrinsic_reward: 1.0,
            cascade_stage: None,
            ucb1_score: None,
        }
    }

    fn create_test_topology() -> TopologicalSignature {
        TopologicalSignature {
            id: uuid::Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            persistence_features: vec![],
            betti_numbers: [1, 0, 0],
            knot_complexity: 3.0,
            knot_polynomial: "test".to_string(),
            tqft_dimension: 2,
            cobordism_type: None,
            persistence_entropy: 1.0,
            spectral_gap: 0.5,
            computation_time_ms: 10.0,
        }
    }

    fn create_test_collapse() -> CollapseResult {
        CollapseResult {
            top_hits: vec![EragMemory {
                input: "test".to_string(),
                output: "test".to_string(),
                emotional_vector: crate::erag::EmotionalVector::default(),
                erag_context: vec![],
                entropy_before: 1.5,
                entropy_after: 2.0,
                timestamp: "2025-01-01T00:00:00Z".to_string(),
                compass_state: None,
                cascade_stage: None,
                weighted_metadata: None,
            }],
            aggregated_context: "test context".to_string(),
            average_similarity: 0.8,
            curator_quality: None,
        }
    }

    #[test]
    fn test_consonance_computation() {
        let pad_state = create_test_pad_state(2.0);
        let compass = create_test_compass(CompassQuadrant::Master);
        let collapse = create_test_collapse();
        let topology = create_test_topology();

        let metrics = compute_consonance(&pad_state, &compass, &collapse, &topology, None, None);

        assert!(metrics.score >= 0.0 && metrics.score <= 1.0);
        assert!(metrics.confidence >= 0.0 && metrics.confidence <= 1.0);
        assert_eq!(metrics.sources.len(), 5);
    }

    #[test]
    fn test_consonance_with_curator() {
        let pad_state = create_test_pad_state(2.0);
        let compass = create_test_compass(CompassQuadrant::Master);
        let collapse = create_test_collapse();
        let topology = create_test_topology();
        let curator = CuratedResponse {
            refined_response: "This is a refined response".to_string(),
            learned: true,
            reason: "High quality".to_string(),
            processing_time_ms: 100.0,
            consonance_score: 0.85,
        };

        let metrics = compute_consonance(&pad_state, &compass, &collapse, &topology, Some(&curator), None);

        assert!(metrics.score > 0.0);
        // Should have curator quality source
        assert!(metrics.sources.iter().any(|s| matches!(s, ConsonanceSource::CuratorQuality(_))));
    }

    #[test]
    fn test_consonance_transitions() {
        let pad_state = create_test_pad_state(2.0);
        let last_compass = create_test_compass(CompassQuadrant::Panic);
        let compass = create_test_compass(CompassQuadrant::Discover);
        let collapse = create_test_collapse();
        let topology = create_test_topology();

        let metrics = compute_consonance(&pad_state, &compass, &collapse, &topology, None, Some(&last_compass));

        // Panic→Discover should be high consonance
        assert!(metrics.score > 0.6);
    }
}

