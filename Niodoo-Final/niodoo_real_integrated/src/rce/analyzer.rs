use std::collections::VecDeque;
use std::time::Instant;

use crate::metrics::rce_metrics;
use crate::tcs_analysis::TopologicalSignature;
use crate::torus::PadGhostState;
use tcs_rce::beta_meta::{compute_beta_meta, BetaMetaInputs, BetaMetaWeights};
use tracing::{debug, info};

pub struct RceAnalyzer {
    last_betti: Option<[usize; 3]>,
    last_ts: Option<Instant>,
    entropy_window: VecDeque<f64>,
    window_capacity: usize,
    peak_beta_meta: f64,
    last_beta_meta: f64,
    last_sigma_r: f64,
    weights: BetaMetaWeights,
    threshold: f64,
}

impl RceAnalyzer {
    pub fn new(window_capacity: usize, weights: BetaMetaWeights, threshold: f64) -> Self {
        Self {
            last_betti: None,
            last_ts: None,
            entropy_window: VecDeque::with_capacity(window_capacity.max(2)),
            window_capacity: window_capacity.max(2),
            peak_beta_meta: 0.0,
            last_beta_meta: 0.0,
            last_sigma_r: 0.0,
            weights,
            threshold,
        }
    }

    fn record_entropy(&mut self, value: f64) {
        if self.entropy_window.len() == self.window_capacity {
            self.entropy_window.pop_front();
        }
        self.entropy_window.push_back(value);
    }

    fn compute_entropy_std(&self) -> f64 {
        if self.entropy_window.len() < 2 {
            return 0.0;
        }
        let mean =
            self.entropy_window.iter().copied().sum::<f64>() / (self.entropy_window.len() as f64);
        let var = self
            .entropy_window
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f64>()
            / ((self.entropy_window.len() - 1) as f64);
        var.sqrt()
    }

    pub fn update(&mut self, pad: &PadGhostState, topo: &TopologicalSignature) -> f64 {
        self.update_with_prompt_timestamp(pad, topo, None)
    }

    pub fn update_with_prompt_timestamp(
        &mut self,
        pad: &PadGhostState,
        topo: &TopologicalSignature,
        prompt_ts: Option<Instant>,
    ) -> f64 {
        // dBetti/dt
        let now = Instant::now();
        let dt_secs = self
            .last_ts
            .map(|prev| now.duration_since(prev).as_secs_f64())
            .unwrap_or(1.0);
        self.last_ts = Some(now);

        // Record update latency (time between consecutive updates)
        let m = rce_metrics();
        if dt_secs > 0.0 && dt_secs < 1000.0 {
            // Only record reasonable latencies (avoid initial start and outliers)
            m.record_update_latency(dt_secs);
        }

        let d_betti_norm = if let Some(prev) = self.last_betti {
            let diff0 = (topo.betti_numbers[0] as f64 - prev[0] as f64).abs();
            let diff1 = (topo.betti_numbers[1] as f64 - prev[1] as f64).abs();
            let diff2 = (topo.betti_numbers[2] as f64 - prev[2] as f64).abs();
            (diff0 + diff1 + diff2) / dt_secs.max(1e-6)
        } else {
            0.0
        };
        self.last_betti = Some(topo.betti_numbers);

        // Metastability proxy: std-dev of recent entropy values
        self.record_entropy(pad.entropy);
        let sigma_r = self.compute_entropy_std();
        self.last_sigma_r = sigma_r;

        // Persistence entropy directly from topology
        let h_topo = topo.persistence_entropy;

        // Motif flux and sheaf divergence are wired in later phases
        let inputs = BetaMetaInputs::new(d_betti_norm, sigma_r, h_topo, 0.0, 0.0);
        let beta = compute_beta_meta(self.weights, inputs);
        self.last_beta_meta = beta;

        // Metrics
        self.peak_beta_meta = self.peak_beta_meta.max(beta);
        m.record_beta_meta(beta, self.peak_beta_meta);
        m.record_persistence_entropy(h_topo);
        m.record_spectral_gap(topo.spectral_gap);

        let is_spike = beta >= self.threshold;
        if is_spike {
            m.inc_spike();

            // Record prompt-to-spike latency if prompt timestamp is available
            if let Some(prompt_start) = prompt_ts {
                let prompt_to_spike_latency = now.duration_since(prompt_start).as_secs_f64();
                if prompt_to_spike_latency > 0.0 && prompt_to_spike_latency < 60.0 {
                    m.record_prompt_to_spike_latency(prompt_to_spike_latency);
                    info!(
                        beta = beta,
                        threshold = self.threshold,
                        dt_secs = dt_secs,
                        prompt_to_spike_secs = prompt_to_spike_latency,
                        persistence_entropy = h_topo,
                        spectral_gap = topo.spectral_gap,
                        "rce.beta_meta_spike"
                    );
                }
            } else {
                info!(
                    beta = beta,
                    threshold = self.threshold,
                    dt_secs = dt_secs,
                    persistence_entropy = h_topo,
                    spectral_gap = topo.spectral_gap,
                    "rce.beta_meta_spike"
                );
            }
        } else {
            debug!(
                beta = beta,
                threshold = self.threshold,
                dt_secs = dt_secs,
                persistence_entropy = h_topo,
                spectral_gap = topo.spectral_gap,
                "rce.beta_meta_update"
            );
        }

        beta
    }

    pub fn last_beta(&self) -> f64 {
        self.last_beta_meta
    }

    pub fn current_metastability(&self) -> f64 {
        self.last_sigma_r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tcs_analysis::TopologicalSignature;
    use crate::torus::PadGhostState;
    use tcs_core::PersistentFeature;

    #[test]
    fn test_beta_meta_basic() {
        let mut analyzer = RceAnalyzer::new(
            4,
            BetaMetaWeights {
                alpha_betti: 1.0,
                alpha_meta: 1.0,
                alpha_motif: 0.0,
                alpha_sheaf: 0.0,
            },
            0.5,
        );
        let pad = PadGhostState {
            pad: [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0],
            mu: [0.0; 7],
            sigma: [0.1; 7],
            entropy: 2.0,
        };
        let topo = TopologicalSignature::new(
            vec![PersistentFeature {
                birth: 0.0,
                death: 1.0,
                dimension: 0,
            }],
            [1, 0, 0],
            0.1,
            "deprecated".to_string(),
            2,
            None,
            1.0,
            0.5,
            0.2,
            0.0,
            1.0,
            1.0,
            0.5,
            0.5,
            0.2,
            0,
            Vec::new(),
            0.1,
            0.05,
            Vec::new(),
            0.3,
            0.02,
            Vec::new(),
        );
        let beta = analyzer.update(&pad, &topo);
        assert!(beta >= 0.0);
        assert!(analyzer.last_beta() >= 0.0);
    }
}
