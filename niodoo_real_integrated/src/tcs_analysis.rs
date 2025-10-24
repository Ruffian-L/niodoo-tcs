//! TCS Topology Analysis Layer
//! Computes persistent homology, knot invariants, and TQFT signatures on every state

use anyhow::Result;
use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{debug, info, instrument};
use uuid::Uuid;

use crate::torus::PadGhostState;
use tcs_knot::{JonesPolynomial, KnotDiagram};
use tcs_tda::{PersistenceFeature, PersistentHomology, TakensEmbedding};
use tcs_tqft::{Cobordism, TQFTEngine};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

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
        let points: Vec<Vec<f64>> = self.pad_to_points(pad_state);

        // Compute persistent homology using Gudhi via pyo3
        let (betti_numbers, persistence_features, pe, gap, knot_proxy) = Python::with_gil(|py| {
            let locals = PyDict::new(py);
            let code = r#"
import gudhi
from math import log

def compute_tda_features(points, max_edge=2.0, max_dim=2):
    if not points:
        return {
            'betti_numbers': [1, 0, 0],
            'persistence_features': [],
            'persistence_entropy': 0.0,
            'spectral_gap': 0.0,
            'knot_complexity': 0.0
        }
    rc = gudhi.RipsComplex(points=points, max_edge_length=max_edge)
    st = rc.create_simplex_tree(max_dimension=max_dim)
    betti = st.betti_numbers()
    pers = st.persistence()
    # Compute lifetimes for finite intervals
    lifetime = 0.0
    intervals = []
    for dim, (birth, death) in pers:
        if death == float('inf'):
            continue
        l = death - birth
        lifetime += l
        intervals.append((dim, birth, death))
    if lifetime > 0:
        pe = 0.0
        for dim, birth, death in intervals:
            p = (death - birth) / lifetime
            pe -= p * log(p)
    else:
        pe = 0.0
    # Pad betti to 3
    betti_padded = betti + [0] * (3 - len(betti))
    gap = abs(betti_padded[1] - betti_padded[0])
    knot = betti_padded[1]
    # Persistence features: all intervals, inf as None
    pers_features = [(dim, birth, None if death == float('inf') else death) for dim, (birth, death) in pers]
    return {
        'betti_numbers': betti_padded[:3],
        'persistence_features': pers_features,
        'persistence_entropy': pe,
        'spectral_gap': gap,
        'knot_complexity': float(knot)
    }
"#;
            py.run(code, None, Some(locals)).map_err(|e| anyhow::anyhow!("Python code execution failed: {}", e))?;
            let compute_fn = locals.get_item("compute_tda_features")
                .and_then(|f| f.extract::<&PyAny>(py))
                .ok_or_else(|| anyhow::anyhow!("Failed to extract compute function"))?;
            let points_py_lists: Vec<Py<PyList>> = points.iter().map(|point| {
                PyList::new(py, point.iter().cloned()).into()
            }).collect();
            let points_py = PyList::new(py, &points_py_lists);
            let result = compute_fn.call1((points_py,), ()).map_err(|e| anyhow::anyhow!("Python function call failed: {}", e))?;
            let dict = result.downcast::<PyDict>(py).map_err(|e| anyhow::anyhow!("Result is not a dict: {}", e))?;
            let betti_vec: Vec<usize> = dict.get_item("betti_numbers")
                .and_then(|b| b.extract(py))
                .unwrap_or(vec![1, 0, 0]);
            let betti_arr = [betti_vec.get(0).copied().unwrap_or(1), betti_vec.get(1).copied().unwrap_or(0), betti_vec.get(2).copied().unwrap_or(0)];
            let pers_features_raw: Vec<(usize, f64, Option<f64>)> = dict.get_item("persistence_features")
                .and_then(|p| p.extract(py))
                .unwrap_or_default();
            let persistence_features: Vec<PersistenceFeature> = pers_features_raw.into_iter().map(|(dim, birth, death)| PersistenceFeature {
                dimension: dim,
                birth,
                death: death.unwrap_or(f64::INFINITY),
            }).collect();
            let pe = dict.get_item("persistence_entropy").and_then(|p| p.extract::<f64>(py)).unwrap_or(0.0);
            let gap = dict.get_item("spectral_gap").and_then(|g| g.extract::<f64>(py)).unwrap_or(0.0);
            let knot = dict.get_item("knot_complexity").and_then(|k| k.extract::<f64>(py)).unwrap_or(0.0);
            Ok((betti_arr, persistence_features, pe, gap, knot))
        })?;

        // Keep existing knot polynomial computation
        let knot_diagram = self.pad_to_knot_diagram(pad_state);
        let knot_analysis = self.knot_analyzer.analyze(&knot_diagram);
        let knot_polynomial = knot_analysis.polynomial;

        // Use TDA knot proxy for complexity
        let knot_complexity = knot_proxy.max(knot_analysis.complexity_score as f64);

        let cobordism_type = self.infer_cobordism(&betti_numbers);

        let computation_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        debug!(
            "Topological analysis: Betti={:?}, Knot complexity={:.3}, PE={:.3}, Gap={:.3}, Cobordism={:?}",
            betti_numbers, knot_complexity, pe, gap, cobordism_type
        );

        Ok(TopologicalSignature::new(
            persistence_features,
            betti_numbers,
            knot_complexity,
            knot_polynomial,
            self.tqft_engine.dimension,
            cobordism_type,
            computation_time_ms,
            pe,
            gap,
        ))
    }

    /// Convert PAD state to point cloud for homology computation
    fn pad_to_points(&self, pad_state: &PadGhostState) -> Vec<Vec<f64>> {
        let mut points = Vec::new();
        for i in 0..7 {
            // Create point from PAD coordinates with mu/sigma as extra dimensions
            let mut coords = Vec::with_capacity(7);
            coords.push(pad_state.pad[i] as f64);
            coords.push(pad_state.mu[i] as f64);
            coords.push(pad_state.sigma[i] as f64);
            // Pad to 7D
            while coords.len() < 7 {
                coords.push(0.0);
            }
            points.push(coords);
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
