//! TCS Topology Analysis Layer
//! Computes persistent homology, knot invariants, and TQFT signatures on every state

use anyhow::Result;
use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{debug, info, instrument};
use uuid::Uuid;

use crate::compass::CompassOutcome;
use crate::torus::PadGhostState;
use tcs_knot::{CognitiveKnot, JonesPolynomial, KnotDiagram};
use tcs_tda::{PersistenceFeature, PersistentHomology, TakensEmbedding};
use tcs_tqft::{Cobordism, FrobeniusAlgebra, TQFTEngine};

/// Topological signature computed for a state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalSignature {
    pub id: Uuid,
    pub timestamp: chrono::DateTime<chrono::Utc>,

    // Persistent homology features
    #[serde(skip)]
    pub persistence_features: Vec<PersistenceFeature>,
    pub betti_numbers: [usize; 3], // H0, H1, H2

    // Knot invariants
    pub knot_complexity: f32,
    pub knot_polynomial: String,

    // TQFT invariants
    pub tqft_dimension: usize,
    pub cobordism_type: Option<Cobordism>,

    // Performance metrics
    pub computation_time_ms: f64,
}

impl TopologicalSignature {
    pub fn new(
        persistence_features: Vec<PersistenceFeature>,
        betti_numbers: [usize; 3],
        knot_complexity: f32,
        knot_polynomial: String,
        tqft_dimension: usize,
        cobordism_type: Option<Cobordism>,
        computation_time_ms: f64,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            persistence_features,
            betti_numbers,
            knot_complexity,
            knot_polynomial,
            tqft_dimension,
            cobordism_type,
            computation_time_ms,
        }
    }
}

/// TCS Analysis Engine
pub struct TCSAnalyzer {
    homology: PersistentHomology,
    knot_analyzer: JonesPolynomial,
    tqft_engine: TQFTEngine,
    takens: TakensEmbedding,
}

impl TCSAnalyzer {
    /// Initialize TCS analyzer
    pub fn new() -> Result<Self> {
        let homology = PersistentHomology::new(2, 1.0); // Max dimension 2, max edge length 1.0
        let knot_analyzer = JonesPolynomial::new(64);
        let tqft_engine = TQFTEngine::new(2)
            .map_err(|e| anyhow::anyhow!("Failed to initialize TQFT engine: {}", e))?;
        let takens = TakensEmbedding::new(3, 1, 7); // dim=3, delay=1, data_dim=7 (PAD+ghost)

        info!("TCS Analyzer initialized");
        Ok(Self {
            homology,
            knot_analyzer,
            tqft_engine,
            takens,
        })
    }

    /// Analyze topological structure of a state
    #[instrument(skip(self), fields(entropy = pad_state.entropy))]
    pub fn analyze_state(&mut self, pad_state: &PadGhostState) -> Result<TopologicalSignature> {
        let start = Instant::now();

        // Convert PAD state to point cloud representation
        let points = self.pad_to_points(pad_state);

        // Compute persistent homology
        let persistence_features = self.homology.compute(&points);
        let betti_numbers = self.compute_betti_numbers(&persistence_features);

        // Extract knot invariants (simplified - treat PAD as knot diagram)
        let knot_diagram = self.pad_to_knot_diagram(pad_state);
        let knot_analysis = self.knot_analyzer.analyze(&knot_diagram);

        // Infer cobordism from Betti number changes
        let cobordism_type = self.infer_cobordism(&betti_numbers);

        let computation_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        debug!(
            "Topological analysis: Betti={:?}, Knot complexity={:.3}, Cobordism={:?}",
            betti_numbers, knot_analysis.complexity_score, cobordism_type
        );

        Ok(TopologicalSignature::new(
            persistence_features,
            betti_numbers,
            knot_analysis.complexity_score,
            knot_analysis.polynomial,
            self.tqft_engine.dimension,
            cobordism_type,
            computation_time_ms,
        ))
    }

    /// Convert PAD state to point cloud for homology computation
    fn pad_to_points(&self, pad_state: &PadGhostState) -> Vec<DVector<f32>> {
        // Use Takens embedding to create point cloud from PAD coordinates
        let pad_as_time_series: Vec<Vec<f32>> =
            vec![pad_state.pad.iter().map(|&x| x as f32).collect()];

        let mut points = Vec::new();
        for i in 0..7 {
            // Create point from PAD coordinates with mu/sigma as extra dimensions
            let mut coords = Vec::with_capacity(7);
            coords.push(pad_state.pad[i] as f32);
            coords.push(pad_state.mu[i] as f32);
            coords.push(pad_state.sigma[i] as f32);
            // Pad to 7D
            while coords.len() < 7 {
                coords.push(0.0);
            }
            points.push(DVector::from_vec(coords));
        }

        points
    }

    /// Compute Betti numbers from persistence features
    fn compute_betti_numbers(&self, features: &[PersistenceFeature]) -> [usize; 3] {
        let mut betti = [0usize; 3];

        for feature in features {
            if feature.dimension < 3 {
                betti[feature.dimension] += 1;
            }
        }

        betti
    }

    /// Convert PAD state to simplified knot diagram
    fn pad_to_knot_diagram(&self, pad_state: &PadGhostState) -> KnotDiagram {
        // Map PAD values to crossings (over/under crossings)
        let crossings: Vec<i32> = pad_state
            .pad
            .iter()
            .map(|&val| {
                if val > 0.5 {
                    1 // Over-crossing
                } else if val < -0.5 {
                    -1 // Under-crossing
                } else {
                    0 // No crossing
                }
            })
            .filter(|&x| x != 0)
            .collect();

        KnotDiagram { crossings }
    }

    /// Infer cobordism type from Betti number changes
    fn infer_cobordism(&self, betti: &[usize; 3]) -> Option<Cobordism> {
        // Simplified inference based on Betti numbers
        // TODO: Phase 3 - Store previous Betti numbers and compare
        if betti[0] > 1 {
            Some(Cobordism::Split)
        } else if betti[1] > 0 {
            Some(Cobordism::Birth)
        } else {
            Some(Cobordism::Identity)
        }
    }

    /// Analyze transition between two states
    pub fn analyze_transition(
        &mut self,
        before: &PadGhostState,
        after: &PadGhostState,
    ) -> Result<TransitionAnalysis> {
        let before_signature = self.analyze_state(before)?;
        let after_signature = self.analyze_state(after)?;

        // Compute Betti changes
        let betti_delta = [
            after_signature.betti_numbers[0] as i32 - before_signature.betti_numbers[0] as i32,
            after_signature.betti_numbers[1] as i32 - before_signature.betti_numbers[1] as i32,
            after_signature.betti_numbers[2] as i32 - before_signature.betti_numbers[2] as i32,
        ];

        // Infer cobordism from Betti changes
        let inferred_cobordism = TQFTEngine::infer_cobordism_from_betti(
            &before_signature.betti_numbers,
            &after_signature.betti_numbers,
        );

        Ok(TransitionAnalysis {
            before: before_signature,
            after: after_signature,
            betti_delta,
            inferred_cobordism,
        })
    }
}

/// Analysis of transition between two states
#[derive(Debug, Clone)]
pub struct TransitionAnalysis {
    pub before: TopologicalSignature,
    pub after: TopologicalSignature,
    pub betti_delta: [i32; 3],
    pub inferred_cobordism: Option<Cobordism>,
}

impl Default for TCSAnalyzer {
    fn default() -> Self {
        Self::new().expect("Failed to initialize TCS analyzer")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tcs_analyzer_initialization() {
        let analyzer = TCSAnalyzer::new();
        assert!(analyzer.is_ok());
    }

    #[test]
    fn test_pad_to_knot_diagram() {
        let analyzer = TCSAnalyzer::new().unwrap();
        let pad_state = PadGhostState {
            pad: [0.8, -0.3, 0.6, -0.2, 0.4, 0.0, 0.1],
            entropy: 1.98,
            mu: [0.0; 7],
            sigma: [0.5; 7],
        };

        let diagram = analyzer.pad_to_knot_diagram(&pad_state);
        assert!(!diagram.crossings.is_empty());
    }
}
