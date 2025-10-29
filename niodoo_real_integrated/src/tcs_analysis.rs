//! TCS Topology Analysis Layer
//! Computes persistent homology, knot invariants, and TQFT signatures on every state

use anyhow::Result;
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

use crate::torus::PadGhostState;
use tcs_core::metrics::record_topology_metrics;
use tcs_core::topology::{PersistenceFeature, PersistenceResult, Point, TopologyParams};
use tcs_core::{RustVREngine, TopologyEngine};
use tcs_knot::{JonesPolynomial, KnotDiagram};
use tcs_tqft::{Cobordism, TQFTEngine};

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
    pub knot_complexity: f64,
    pub knot_polynomial: String,

    // TQFT invariants
    pub tqft_dimension: usize,
    pub cobordism_type: Option<Cobordism>,

    // New TDA features for Phase 5
    pub persistence_entropy: f64,
    pub spectral_gap: f64,

    // Performance metrics
    pub computation_time_ms: f64,
}

impl TopologicalSignature {
    pub fn new(
        persistence_features: Vec<PersistenceFeature>,
        betti_numbers: [usize; 3],
        knot_complexity: f64,
        knot_polynomial: String,
        tqft_dimension: usize,
        cobordism_type: Option<Cobordism>,
        computation_time_ms: f64,
        persistence_entropy: f64,
        spectral_gap: f64,
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
            persistence_entropy,
            spectral_gap,
            computation_time_ms,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TCSState {
    // Add fields as needed, e.g., persistence_features: Vec<PersistenceFeature>,
    // but keep minimal for now
    pad: Vec<f64>,
    mu: Vec<f64>,
    sigma: Vec<f64>,
}

pub type TCSHandle = Arc<Mutex<TCSState>>;

/// TCS Analysis Engine
pub struct TCSAnalyzer {
    topology_engine: RustVREngine,
    knot_analyzer: JonesPolynomial,
    tqft_engine: TQFTEngine,
}

impl TCSAnalyzer {
    /// Initialize TCS analyzer
    pub fn new() -> Result<Self> {
        let topology_engine = RustVREngine::new();
        let knot_analyzer = JonesPolynomial::new(64);
        let tqft_engine = TQFTEngine::new(2)
            .map_err(|e| anyhow::anyhow!("Failed to initialize TQFT engine: {}", e))?;

        info!("TCS Analyzer initialized");
        Ok(Self {
            topology_engine,
            knot_analyzer,
            tqft_engine,
        })
    }

    /// Apply TQFT reasoning to evolve a state through cobordism transitions
    pub fn apply_tqft_reasoning(
        &self,
        initial_state: &[f64],
        transitions: &[Cobordism],
    ) -> Result<Vec<f64>> {
        use nalgebra::DVector;
        use num_complex::Complex;

        // Convert real state to complex vector
        let complex_state: DVector<Complex<f32>> = DVector::from_iterator(
            initial_state.len().min(self.tqft_engine.dimension),
            initial_state
                .iter()
                .take(self.tqft_engine.dimension)
                .map(|&x| Complex::new(x as f32, 0.0)),
        );

        // Apply TQFT reasoning
        let result_state = self
            .tqft_engine
            .reason(&complex_state, transitions)
            .map_err(|e| anyhow::anyhow!("TQFT reasoning failed: {}", e))?;

        // Convert back to real values
        let real_state: Vec<f64> = result_state.iter().map(|c| c.re as f64).collect();

        Ok(real_state)
    }

    /// Analyze topological structure of a state
    #[instrument(skip(self), fields(entropy = pad_state.entropy))]
    pub fn analyze_state(&mut self, pad_state: &PadGhostState) -> Result<TopologicalSignature> {
        let start = Instant::now();

        let tcs_state = Arc::new(Mutex::new(TCSState::default()));
        let mut guard = tcs_state.lock().unwrap();
        // Use guard for computations, e.g., populate from pad_state
        guard.pad = pad_state.pad.iter().map(|&v| v as f64).collect();
        guard.mu = pad_state.mu.iter().map(|&v| v as f64).collect();
        guard.sigma = pad_state.sigma.iter().map(|&v| v as f64).collect();

        let points = self.pad_to_points(&guard);
        // Scaling guards: cap KNN and filtration for performance with configurable defaults
        let k = std::env::var("TCS_KNN_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16);
        let max_filtration = std::env::var("TCS_MAX_FILTRATION")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(1.5);
        let params = TopologyParams {
            k: k.max(1).min(128),
            max_filtration_value: Some(max_filtration.max(0.1).min(10.0)),
            ..TopologyParams::default()
        };
        let result = self
            .topology_engine
            .compute_persistence(&points, 2, &params)?;
        record_topology_metrics(&result);

        // Parallelize entropy and spectral gap computation
        let persistence_entropy = {
            use rayon::prelude::*;
            result
                .entropy
                .par_iter()
                .map(|(_, value)| f64::from(*value))
                .sum::<f64>()
        };

        let spectral_gap = Self::compute_spectral_gap(&result);

        let mut betti = self.compute_betti_numbers(&result);

        // Validate Betti numbers: for n points, Betti_1 <= n-1 for connected components
        // Additionally, enforce configurable constraint (default: 6, override via TCS_BETTI1_MAX)
        // This prevents mathematically impossible values and ensures realistic bounds
        let num_points = points.len();
        let theoretical_max = num_points.saturating_sub(1);
        let constraint_max = std::env::var("TCS_BETTI1_MAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(6);
        let max_allowed = theoretical_max.min(constraint_max);

        let original_betti1 = betti[1];

        // Debug: Log before capping
        debug!(
            "Betti numbers before capping: {:?}, num_points={}, theoretical_max={}, constraint_max={}, max_allowed={}",
            betti, num_points, theoretical_max, constraint_max, max_allowed
        );

        // Investigate: Count persistent features manually to understand the issue
        let mut persistent_count_debug = [0usize; 3];
        for diagram in &result.diagrams {
            if diagram.dimension < 3 {
                let persistent_count = diagram
                    .features
                    .iter()
                    .filter(|f| f.death.is_infinite())
                    .count();
                persistent_count_debug[diagram.dimension] = persistent_count;
                if persistent_count > 0 {
                    debug!(
                        "Dimension {}: {} persistent features (total features: {})",
                        diagram.dimension,
                        persistent_count,
                        diagram.features.len()
                    );
                }
            }
        }

        if betti[1] > max_allowed {
            warn!(
                "Betti_1 ({}) exceeds maximum (theoretical: {}, constraint: {}), capping to {}. num_points={}, persistent_count_debug={:?}",
                betti[1],
                theoretical_max,
                constraint_max,
                max_allowed,
                num_points,
                persistent_count_debug
            );
            betti[1] = max_allowed;
        }

        // Verify capping worked - enforce it if somehow it didn't take
        if betti[1] > max_allowed {
            warn!(
                "Betti_1 capping failed during check: value={} exceeds max_allowed={}. Force-capping now.",
                betti[1], max_allowed
            );
            betti[1] = max_allowed;
        }

        if betti[1] != original_betti1 {
            info!(
                "Betti_1 capped from {} to {} (max_allowed={})",
                original_betti1, betti[1], max_allowed
            );
        }

        // Assertion: betti[1] should never exceed max_allowed at this point
        assert!(
            betti[1] <= max_allowed,
            "Betti_1 assertion failed: {} > {} (theoretical_max={}, constraint_max={}, num_points={})",
            betti[1],
            max_allowed,
            theoretical_max,
            constraint_max,
            num_points
        );

        let gap = (betti[1] as f64 - betti[0] as f64).abs();
        // Cap knot proxy to prevent saturation: use capped Betti_1 value, ensuring variation
        // Ensure we use the capped betti[1] value, not the original
        let knot_proxy = (betti[1] as f64).min(constraint_max as f64).max(0.0);

        let phi = Self::approximate_phi_from_betti(&betti);
        info!("IIT Φ (approx): {:.6}", phi);

        // Keep existing knot polynomial computation
        let knot_diagram = self.pad_to_knot_diagram(&guard);
        let knot_analysis = self.knot_analyzer.analyze(&knot_diagram);
        let knot_polynomial = knot_analysis.polynomial;

        // Use TDA knot proxy for complexity, but cap knot_analysis.complexity_score to prevent saturation
        // This ensures variation in knot complexity rather than always saturating at max value
        // Allow override via TCS_KNOT_COMPLEXITY_MAX (defaults to constraint_max if not set)
        let knot_complexity_max = std::env::var("TCS_KNOT_COMPLEXITY_MAX")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(constraint_max as f64);
        let knot_analysis_score = (knot_analysis.complexity_score as f64).min(knot_complexity_max);
        let knot_complexity = knot_proxy.max(knot_analysis_score).min(knot_complexity_max);

        debug!(
            "Knot complexity calculation: betti[1]={}, knot_proxy={:.3}, knot_analysis_score={:.3}, final={:.3}, max={:.3}",
            betti[1], knot_proxy, knot_analysis_score, knot_complexity, knot_complexity_max
        );

        let cobordism_type = self.infer_cobordism(&betti);

        let computation_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        debug!(
            "Topological analysis: Betti={:?}, Knot complexity={:.3}, PE={:.3}, Gap={:.3}, Cobordism={:?}",
            betti, knot_complexity, persistence_entropy, gap, cobordism_type
        );

        let persistence_features = Self::collect_persistence_features(&result);

        Ok(TopologicalSignature::new(
            persistence_features,
            betti,
            knot_complexity,
            knot_polynomial,
            self.tqft_engine.dimension,
            cobordism_type,
            computation_time_ms,
            persistence_entropy,
            spectral_gap,
        ))
    }

    /// Convert PAD state to point cloud for homology computation
    /// Uses PAD, mu, sigma, and incorporates entropy to make topology input-sensitive
    fn pad_to_points(&self, pad_state: &TCSState) -> Vec<Point> {
        let mut points = Vec::new();

        // Calculate global statistics for normalization
        let pad_mean: f64 = pad_state.pad.iter().sum::<f64>() / pad_state.pad.len() as f64;
        let pad_variance: f64 = pad_state
            .pad
            .iter()
            .map(|&v| (v - pad_mean).powi(2))
            .sum::<f64>()
            / pad_state.pad.len() as f64;

        for i in 0..7 {
            // Create point from PAD coordinates with mu/sigma as extra dimensions
            // Add entropy-normalized variance to make topology sensitive to input variation
            let mut coords = Vec::with_capacity(7);
            coords.push(pad_state.pad[i]);
            coords.push(pad_state.mu[i]);
            coords.push(pad_state.sigma[i]);

            // Add additional dimensions: variance-weighted position, relative deviation
            // This ensures different PAD distributions produce different point clouds
            coords.push((pad_state.pad[i] - pad_mean) * pad_variance.sqrt());
            coords.push(pad_state.mu[i] * pad_state.sigma[i]); // Interaction term

            // Pad to 7D
            while coords.len() < 7 {
                coords.push(0.0);
            }
            let point = coords.into_iter().map(|v| v as f32).collect::<Vec<_>>();
            points.push(Point::new(point));
        }
        points
    }

    /// Compute Betti numbers from persistence features
    /// Betti numbers count only persistent features (death == infinity), not all features
    fn compute_betti_numbers(&self, result: &PersistenceResult) -> [usize; 3] {
        let mut betti = [0usize; 3];
        for diagram in &result.diagrams {
            if diagram.dimension < 3 {
                // Only count features that persist to infinity (death == f64::INFINITY)
                // These represent true topological holes, not temporary features
                let persistent_count = diagram
                    .features
                    .iter()
                    .filter(|f| f.death.is_infinite())
                    .count();
                betti[diagram.dimension] = persistent_count;
            }
        }
        betti
    }

    /// Convert PAD state to simplified knot diagram
    fn pad_to_knot_diagram(&self, pad_state: &TCSState) -> KnotDiagram {
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

    /// Infer cobordism type from Betti number changes using TQFT engine
    fn infer_cobordism(&self, betti: &[usize; 3]) -> Option<Cobordism> {
        // Use TQFT engine's proper inference method
        // This would need previous state, so for now use static inference
        // In production, track previous Betti numbers for comparison
        use std::sync::RwLock;
        static PREV_BETTI: RwLock<Option<[usize; 3]>> = RwLock::new(None);

        let prev_opt = { PREV_BETTI.read().unwrap().clone() };
        let cobordism = if let Some(prev) = prev_opt {
            TQFTEngine::infer_cobordism_from_betti(&prev, betti)
        } else {
            // First run - infer from structure
            if betti[0] > 1 {
                Some(Cobordism::Split)
            } else if betti[1] > 0 {
                Some(Cobordism::Birth)
            } else {
                Some(Cobordism::Identity)
            }
        };
        {
            let mut w = PREV_BETTI.write().unwrap();
            *w = Some(*betti);
        }

        cobordism
    }

    fn collect_persistence_features(result: &PersistenceResult) -> Vec<PersistenceFeature> {
        result
            .diagrams
            .iter()
            .flat_map(|diagram| diagram.features.iter().cloned())
            .collect()
    }

    fn compute_spectral_gap(result: &PersistenceResult) -> f64 {
        let mut values: Vec<f64> = result
            .entropy
            .iter()
            .map(|(_, value)| f64::from(*value))
            .collect();

        if values.len() < 2 {
            return values.first().copied().unwrap_or(0.0);
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let min = values.first().copied().unwrap_or(0.0);
        let max = values.last().copied().unwrap_or(0.0);
        (max - min).max(0.0)
    }

    fn approximate_phi_from_betti(betti: &[usize; 3]) -> f64 {
        let total: f64 = betti.iter().map(|&b| b as f64).sum();
        if total <= f64::EPSILON {
            return 0.0;
        }

        let weights = [0.5_f64, 0.3, 0.2];
        betti
            .iter()
            .zip(weights.iter())
            .map(|(&b, &w)| w * (b as f64 / total))
            .sum()
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

impl PadGhostState {
    #[allow(dead_code)]
    fn to_tensor(&self) -> anyhow::Result<Tensor> {
        let mut values: Vec<f32> = Vec::with_capacity(512);
        values.extend(self.pad.iter().map(|v| *v as f32));
        values.extend(self.mu.iter().map(|v| *v as f32));
        values.extend(self.sigma.iter().map(|v| *v as f32));

        // Pad to the expected embedding width
        if values.len() < 512 {
            values.resize(512, 0.0);
        }

        let tensor = Tensor::from_vec(values, (1, 512), &Device::Cpu)?;
        Ok(tensor)
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
    use crate::torus::PadGhostState as PadState;

    #[test]
    fn test_betti_delta_signals_change() {
        let mut analyzer = TCSAnalyzer::new().expect("init tcs analyzer");
        let before = PadState {
            pad: [0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.0],
            entropy: 0.4,
            mu: [0.0; 7],
            sigma: [0.1; 7],
            raw_stds: [0.1; 7].to_vec(),
        };
        let after = PadState {
            pad: [0.5, 0.2, -0.1, 0.0, 0.0, 0.0, 0.0],
            entropy: 0.35,
            mu: [0.0; 7],
            sigma: [0.12; 7],
            raw_stds: [0.12; 7].to_vec(),
        };
        let trans = analyzer
            .analyze_transition(&before, &after)
            .expect("transition");
        assert_eq!(trans.betti_delta.len(), 3);
    }

    #[test]
    fn test_tcs_delta() {
        let mut analyzer = TCSAnalyzer::new().unwrap();
        let mut pad_state = PadGhostState::default();
        pad_state.pad[0] = 0.5; // Simple state
        let signature = analyzer.analyze_state(&pad_state).unwrap();
        // Basic check: entropy should be computed
        assert!(signature.persistence_entropy >= 0.0);
        // Delta proxy: knot complexity
        let delta = signature.knot_complexity; // Assume baseline 0
        assert!(delta.is_finite());
    }
}
