use serde::{Deserialize, Serialize};

use crate::beta_meta::{compute_beta_meta, BetaMetaInputs, BetaMetaWeights};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BetaMetaSnapshot {
    pub t_unix_ms: i64,
    pub beta_meta: f64,
    pub d_betti_norm: f64,
    pub metastability_sigma_r: f64,
    pub persistence_entropy: f64,
    pub motif_flux: f64,
    pub sheaf_divergence: f64,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct RceMetricSeries {
    pub snapshots: Vec<BetaMetaSnapshot>,
}

impl RceMetricSeries {
    pub fn push(&mut self, t_unix_ms: i64, weights: BetaMetaWeights, inputs: BetaMetaInputs) {
        let beta = compute_beta_meta(weights, inputs);
        self.snapshots.push(BetaMetaSnapshot {
            t_unix_ms,
            beta_meta: beta,
            d_betti_norm: inputs.d_betti_norm,
            metastability_sigma_r: inputs.metastability_sigma_r,
            persistence_entropy: inputs.persistence_entropy,
            motif_flux: inputs.motif_flux,
            sheaf_divergence: inputs.sheaf_divergence,
        });
    }

    pub fn latest(&self) -> Option<&BetaMetaSnapshot> {
        self.snapshots.last()
    }
}


