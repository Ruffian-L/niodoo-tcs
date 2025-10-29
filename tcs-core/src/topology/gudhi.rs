use anyhow::{Result, anyhow};

use super::{PersistenceResult, Point, TopologyEngine, TopologyParams};

/// Placeholder for the forthcoming Gudhi-backed topology engine.
pub struct GudhiEngine;

impl GudhiEngine {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self
    }
}

impl TopologyEngine for GudhiEngine {
    fn compute_persistence(
        &self,
        _points: &[Point],
        _max_dim: u8,
        _params: &TopologyParams,
    ) -> Result<PersistenceResult> {
        Err(anyhow!(
            "GudhiEngine is not implemented yet; enable the `tda_rust` feature"
        ))
    }
}
