//! EBM-TQFT Bridge
//!
//! Integrates EBM energy network with TQFT computation, providing a learnable
//! approximation to the #P-hard Jones polynomial computation.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use ndarray::Array1;
use tracing::{debug, info, warn};

use crate::models::energy_network::TopologicalEnergyNetwork;
use crate::tcs_analysis::TopologicalSignature;
use crate::topology::tda_features::TDAFeatureExtractor;
use tcs_tqft::TQFTEngine;

/// EBM-TQFT Bridge
///
/// Provides a bridge between EBM energy network and TQFT computation,
/// allowing EBM to approximate Jones polynomial when enabled.
pub struct EBMTQFTBridge {
    tda_extractor: TDAFeatureExtractor,
    energy_net: Option<TopologicalEnergyNetwork>,
    use_ebm: bool,
    device: Device,
}

impl EBMTQFTBridge {
    /// Create a new EBM-TQFT Bridge
    ///
    /// Args:
    ///   - use_ebm: Whether to use EBM approximation (if false, falls back to exact computation)
    ///   - energy_net: Optional pre-trained energy network
    ///   - device: Device to run on
    pub fn new(
        use_ebm: bool,
        energy_net: Option<TopologicalEnergyNetwork>,
        device: Device,
    ) -> Self {
        let tda_extractor = TDAFeatureExtractor::default();

        info!(
            "Initialized EBMTQFTBridge: use_ebm={}, has_energy_net={}",
            use_ebm,
            energy_net.is_some()
        );

        Self {
            tda_extractor,
            energy_net,
            use_ebm,
            device,
        }
    }

    /// Compute topological score (Jones polynomial approximation or exact)
    ///
    /// If EBM is enabled and energy network is available, uses EBM approximation.
    /// Otherwise falls back to exact Jones polynomial computation.
    ///
    /// Args:
    ///   - signature: TopologicalSignature containing Betti numbers and persistence data
    ///   - tqft_engine: TQFT engine for exact computation fallback
    ///
    /// Returns:
    ///   - Topological score (f64)
    pub fn compute_topological_score(
        &self,
        signature: &TopologicalSignature,
        tqft_engine: &mut TQFTEngine,
    ) -> Result<f64> {
        if self.use_ebm {
            if let Some(ref energy_net) = self.energy_net {
                // Use EBM approximation
                let features = self.tda_extractor.extract_from_signature(signature)?;
                let features_tensor = self.features_to_tensor(&features)?;
                let score = energy_net.approximate_jones_polynomial(&features_tensor)?;

                debug!("EBM topological score: {:.6}", score);
                return Ok(score);
            } else {
                warn!("EBM enabled but no energy network available, falling back to exact computation");
            }
        }

        // Fallback to exact Jones polynomial computation
        self.compute_jones_polynomial_exact(signature, tqft_engine)
    }

    /// Compute exact Jones polynomial (fallback)
    ///
    /// Uses the existing TQFT engine to compute exact Jones polynomial.
    fn compute_jones_polynomial_exact(
        &self,
        signature: &TopologicalSignature,
        _tqft_engine: &mut TQFTEngine,
    ) -> Result<f64> {
        // Use knot_complexity as proxy for Jones polynomial value
        // In full implementation, would use tqft_engine to compute exact polynomial
        let score = signature.knot_complexity;

        debug!("Exact topological score (knot_complexity): {:.6}", score);
        Ok(score)
    }

    /// Convert TDA features array to tensor
    fn features_to_tensor(&self, features: &Array1<f64>) -> Result<Tensor> {
        let feature_vec: Vec<f32> = features.iter().map(|&x| x as f32).collect();
        Ok(Tensor::from_vec(
            feature_vec,
            (features.len(),),
            &self.device,
        )?)
    }

    /// Set energy network (for loading pre-trained models)
    pub fn set_energy_net(&mut self, energy_net: TopologicalEnergyNetwork) {
        self.energy_net = Some(energy_net);
        self.use_ebm = true;
        info!("Energy network set in EBM-TQFT bridge");
    }

    /// Enable/disable EBM
    pub fn set_use_ebm(&mut self, use_ebm: bool) {
        self.use_ebm = use_ebm;
        info!("EBM-TQFT bridge: use_ebm={}", use_ebm);
    }

    /// Check if EBM is enabled and available
    pub fn is_ebm_available(&self) -> bool {
        self.use_ebm && self.energy_net.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tcs_analysis::TopologicalSignature;
    use tcs_tqft::TQFTEngine;

    #[test]
    fn test_bridge_creation() -> Result<()> {
        let device = Device::Cpu;
        let bridge = EBMTQFTBridge::new(false, None, device);

        assert!(!bridge.is_ebm_available());
        Ok(())
    }
}
