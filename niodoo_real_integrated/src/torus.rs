use anyhow::Result;
use nalgebra::SVector;
use rand::Rng;
use rand::{rngs::StdRng, SeedableRng};
use tracing::{debug, instrument};

use crate::util::shannon_entropy;

#[derive(Debug, Clone)]
pub struct PadGhostState {
    pub pad: [f64; 7],
    pub entropy: f64,
    pub mu: [f64; 7],
    pub sigma: [f64; 7],
    pub raw_stds: Vec<f64>,
}

/// Differentiable torus projection approximated by a light-weight VAE head.
/// Deterministic: Uses an owned, seeded StdRng
pub struct TorusPadMapper {
    rng: StdRng,
}

impl TorusPadMapper {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Map an embedding onto the 7D PAD+ghost manifold.
    #[instrument(skip_all)]
    pub fn project(&mut self, embedding: &[f32]) -> Result<PadGhostState> {
        anyhow::ensure!(
            embedding.len() >= 128,
            "embedding must be at least 128 dims"
        );

        let base_radius = 2.2;
        let tube_radius = 0.8;
        let twist_k = 3.5;
        let mut mu = [0.0f64; 7];
        let mut sigma = [0.0f64; 7];
        let mut ghosts = [0.0f64; 7];
        let mut raw_stds: Vec<f64> = Vec::with_capacity(7);

        // Extract mu and sigma from embedding slices - VARIANCE-BASED
        // Each dimension gets a slice of embedding to compute mean/variance
        let slice_size = embedding.len() / 7;
        for i in 0..7 {
            let start = i * slice_size;
            let end = if i == 6 {
                embedding.len()
            } else {
                (i + 1) * slice_size
            };
            let slice = &embedding[start..end];

            // Compute mean (mu) from slice
            mu[i] = slice.iter().map(|&x| x as f64).sum::<f64>() / slice.len() as f64;

            // Compute variance (sigma) from slice
            let variance = slice
                .iter()
                .map(|&x| {
                    let diff = x as f64 - mu[i];
                    diff * diff
                })
                .sum::<f64>()
                / slice.len() as f64;

            // Raw std (before any scaling/clamping) for instrumentation
            let raw_std = variance.sqrt();
            raw_stds.push(raw_std);

            // Dynamically scale std to avoid hard clamp choking 896-d unit-normalized embeds.
            // Conservative amplifier: sqrt(full_embedding_dim)/10.0
            let amp = (embedding.len() as f64).sqrt() / 10.0;
            let scaled = raw_std * amp;

            // Dynamic floor derived from embedding dim to replace fixed 0.05
            let dynamic_floor = (1.0 / (embedding.len() as f64).sqrt()) * 0.5; // tunable

            sigma[i] = scaled.max(dynamic_floor);
        }

        let mut pad = [0.0f64; 7];
        for dim in 0..7 {
            let eps = self.rng.sample::<f64, _>(rand_distr::StandardNormal);
            pad[dim] = mu[dim] + sigma[dim] * eps;
            ghosts[dim] = (embedding[64 + dim] as f64).sin();
        }

        let mut torus_vec: SVector<f64, 7> = SVector::zeros();
        for idx in 0..7 {
            let u = pad[idx].tanh() * std::f64::consts::PI;
            let v = (pad[(idx + 1) % 7] + ghosts[idx]).tanh() * std::f64::consts::PI;
            let radius = base_radius + tube_radius * ((v / 2.0 + twist_k * u).cos());
            torus_vec[idx] = (radius * u.cos()).tanh();
        }

        let mut probs = [0.0f64; 7];
        let mut sum = 0.0;
        for (i, val) in torus_vec.iter().enumerate() {
            let p = ((val + ghosts[i]).tanh() + 1.0) * 0.5;
            probs[i] = p;
            sum += p;
        }
        if sum > 0.0 {
            for p in probs.iter_mut() {
                *p /= sum;
            }
        }

        let entropy = shannon_entropy(&probs);

        // Debug instrumentation: log raw stds, final sigmas, and mu summary at debug level
        debug!(
            ?raw_stds,
            ?sigma,
            ?mu,
            entropy = entropy,
            "torus: computed mu/sigma/raw_stds"
        );

        let mut pad_arr = [0.0f64; 7];
        for (i, val) in torus_vec.iter().enumerate() {
            pad_arr[i] = *val;
        }

        Ok(PadGhostState {
            pad: pad_arr,
            entropy,
            mu,
            sigma,
            raw_stds,
        })
    }
}
