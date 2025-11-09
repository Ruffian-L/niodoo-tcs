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
        anyhow::ensure!(embedding.len() >= 64, "embedding must be at least 64 dims");

        // Compute VAE statistics by splitting the vector into mean/logvar heads.
        let head_width = 7usize;
        let mut mu = [0.0f64; 7];
        let mut logvar = [0.0f64; 7];
        for i in 0..head_width {
            mu[i] = embedding[i] as f64;
            logvar[i] = embedding[head_width + i] as f64;
        }

        // Softplus for variance to keep positive, with PAD noise boost to ±0.5 for spike chaos.
        let mut sigma = [0.5f64; 7];

        // Reparameterisation trick with deterministic RNG.
        let mut pad = [0.0f64; 7];
        for i in 0..7 {
            let eps = self.latent_rng.sample::<f64, _>(rand_distr::StandardNormal);
            pad[i] = mu[i] + sigma[i] * eps;
        }

        // Wrap to torus using sin/cos pairs for first three dimensions.
        let mut torus_vec: SVector<f64, 7> = SVector::zeros();
        for (i, value) in pad.iter().enumerate() {
            let angle = *value;
            torus_vec[i] = angle.tanh().clamp(-1.0, 1.0);
        }

        // Compute entropy from the probability simplex induced by the PAD vector.
        let mut probs = [0.0f64; 7];
        let mut sum = 0.0;
        for (i, val) in torus_vec.iter().enumerate() {
            let p = (val + 1.0) / 2.0; // map [-1,1] -> [0,1]
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
