use anyhow::Result;
use nalgebra::SVector;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tracing::instrument;

use crate::util::shannon_entropy;

#[derive(Debug, Clone)]
pub struct PadGhostState {
    pub pad: [f64; 7],
    pub entropy: f64,
    pub mu: [f64; 7],
    pub sigma: [f64; 7],
}

/// Differentiable torus projection approximated by a light-weight VAE head.
pub struct TorusPadMapper {
    latent_rng: StdRng,
}

impl TorusPadMapper {
    pub fn new(seed: u64) -> Self {
        Self {
            latent_rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Map a 896-dimensional embedding onto the 7D PAD+ghost manifold.
    #[instrument(skip_all)]
    pub fn project(&mut self, embedding: &[f32]) -> Result<PadGhostState> {
        anyhow::ensure!(embedding.len() >= 128, "embedding must be at least 128 dims");

        let base_radius = 2.2;
        let tube_radius = 0.8;
        let twist_k = 3.5;
        let mut mu = [0.0f64; 7];
        let mut sigma = [0.0f64; 7];
        let mut ghosts = [0.0f64; 7];

        for i in 0..7 {
            mu[i] = embedding[i] as f64;
            sigma[i] = (embedding[7 + i] as f64).tanh().abs().max(0.05);
        }

        let mut pad = [0.0f64; 7];
        for dim in 0..7 {
            let eps = self.latent_rng.sample::<f64, _>(rand_distr::StandardNormal);
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

        let mut pad_arr = [0.0f64; 7];
        for (i, val) in torus_vec.iter().enumerate() {
            pad_arr[i] = *val;
        }

        Ok(PadGhostState {
            pad: pad_arr,
            entropy,
            mu,
            sigma,
        })
    }
}
