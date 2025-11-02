use serde::{Deserialize, Serialize};

/// Minimal, read-only sheaf descriptors suitable for ΔS_sheaf computation.
/// This avoids runtime modification until Phase 3 wiring in the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SheafDescriptors {
    /// Optional per-module/stalk norms or capacities
    pub stalk_norms: Vec<f64>,
    /// Optional restriction map residual norms between modules
    pub restriction_residuals: Vec<f64>,
}

impl SheafDescriptors {
    /// Compute a simple divergence score from residuals; callers may plug in
    /// more sophisticated geometry-aware distances once available.
    pub fn divergence_score(&self) -> f64 {
        self.restriction_residuals.iter().copied().map(f64::abs).sum()
    }
}


