use serde::{Deserialize, Serialize};

/// Weights used for β_meta aggregation
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BetaMetaWeights {
    pub alpha_betti: f64,
    pub alpha_meta: f64,
    pub alpha_motif: f64,
    pub alpha_sheaf: f64,
}

impl Default for BetaMetaWeights {
    fn default() -> Self {
        Self {
            alpha_betti: 1.0,
            alpha_meta: 1.0,
            alpha_motif: 1.0,
            alpha_sheaf: 1.0,
        }
    }
}

/// Inputs required to compute β_meta at a given time window
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BetaMetaInputs {
    /// ||dβ/dt||: norm of Betti number derivatives across dimensions
    pub d_betti_norm: f64,
    /// σ_R: metastability (e.g., Kuramoto order parameter variance)
    pub metastability_sigma_r: f64,
    /// H_topo: persistence entropy (or related topological entropy measure)
    pub persistence_entropy: f64,
    /// Σ w_m |d/dt[n_m(t)]|: higher-order motif flux
    pub motif_flux: f64,
    /// ΔS_sheaf: sheaf geometry divergence across the window
    pub sheaf_divergence: f64,
}

impl BetaMetaInputs {
    pub fn new(
        d_betti_norm: f64,
        metastability_sigma_r: f64,
        persistence_entropy: f64,
        motif_flux: f64,
        sheaf_divergence: f64,
    ) -> Self {
        Self {
            d_betti_norm,
            metastability_sigma_r,
            persistence_entropy,
            motif_flux,
            sheaf_divergence,
        }
    }
}

/// Compute β_meta according to the composite formulation:
/// β_meta(t) = α₁·||dβ/dt|| + α₂·σ_R(t)·H_topo(t) + α₃·Σ w_m·|d/dt[n_m]| + α₄·ΔS_sheaf
pub fn compute_beta_meta(weights: BetaMetaWeights, inputs: BetaMetaInputs) -> f64 {
    let term_betti = weights.alpha_betti * inputs.d_betti_norm;
    let term_meta = weights.alpha_meta * (inputs.metastability_sigma_r * inputs.persistence_entropy);
    let term_motif = weights.alpha_motif * inputs.motif_flux;
    let term_sheaf = weights.alpha_sheaf * inputs.sheaf_divergence;
    term_betti + term_meta + term_motif + term_sheaf
}


