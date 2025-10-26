//! Topology engine abstractions and implementations for persistence computation.

#[cfg(feature = "tda_gudhi")]
mod gudhi;
#[cfg(feature = "tda_rust")]
mod rust_vr;
mod types;

pub use types::*;

/// Core trait implemented by all topology engines.
pub trait TopologyEngine {
    /// Compute persistence information up to the requested dimension using the
    /// configured filtration strategy.
    fn compute_persistence(
        &self,
        points: &[Point],
        max_dim: u8,
        params: &TopologyParams,
    ) -> anyhow::Result<PersistenceResult>;
}

#[cfg(feature = "tda_gudhi")]
pub use gudhi::GudhiEngine;
#[cfg(feature = "tda_rust")]
pub use rust_vr::RustVREngine;
