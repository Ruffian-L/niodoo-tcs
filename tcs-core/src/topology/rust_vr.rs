use anyhow::Result;
use nalgebra::DVector;

use super::{
    BettiCurve, PersistenceDiagram, PersistenceFeature, PersistenceResult, Point, TopologyEngine,
    TopologyParams,
};

/// Deterministic stub implementation that mirrors the Rust VR engine shape without
/// relying on heavy external crates. The goal is to keep downstream code exercising
/// the same data paths while returning predictable features.
#[derive(Debug, Default)]
pub struct RustVREngine;

impl RustVREngine {
    pub fn new() -> Self {
        Self
    }
}

impl TopologyEngine for RustVREngine {
    fn compute_persistence(
        &self,
        points: &[Point],
        max_dim: u8,
        _params: &TopologyParams,
    ) -> Result<PersistenceResult> {
        let stats = PersistenceStats::from_points(points);
        let target_dim = max_dim.min(2) as usize;

        let mut diagrams = Vec::with_capacity(target_dim + 1);
        let mut betti_curves = Vec::with_capacity(target_dim + 1);
        let mut entropy = Vec::with_capacity(target_dim + 1);

        for dim in 0..=target_dim {
            let feature = PersistenceFeature {
                birth: 0.0,
                death: 2.0,
                dimension: dim,
            };

            let mut diagram = PersistenceDiagram::new(dim);
            diagram.features.push(feature.clone());
            diagrams.push(diagram);

            betti_curves.push(BettiCurve::new(dim, vec![(0.0, 1)]));

            let dim_entropy = (stats.mean_norm + dim as f32 * 0.1).ln_1p() + stats.variance;
            entropy.push((dim, dim_entropy.max(0.0)));
        }

        let result = PersistenceResult {
            diagrams,
            betti_curves,
            entropy,
        };

        crate::metrics::record_topology_metrics(&result);
        Ok(result)
    }
}

#[derive(Debug, Clone, Copy)]
struct PersistenceStats {
    mean_norm: f32,
    variance: f32,
}

impl PersistenceStats {
    fn from_points(points: &[Point]) -> Self {
        if points.is_empty() {
            return Self {
                mean_norm: 0.0,
                variance: 0.0,
            };
        }

        let mut norms = Vec::with_capacity(points.len());
        for point in points {
            let vector = DVector::from_vec(point.coords.clone());
            norms.push(vector.norm());
        }

        let count = norms.len() as f32;
        let mean_norm = norms.iter().copied().sum::<f32>() / count.max(1.0);
        let variance = norms
            .iter()
            .copied()
            .map(|norm| {
                let delta = norm - mean_norm;
                delta * delta
            })
            .sum::<f32>()
            / count.max(1.0);

        Self {
            mean_norm,
            variance,
        }
    }
}
