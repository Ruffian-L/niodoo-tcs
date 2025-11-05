//! Interfaces between the Topological Cognitive System (TCS) and Niodoo pipelines.

use crate::mobius_labyrinth::{labyrinth_signature, solve_mobius_labyrinth};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// High-level bridge that can inject Möbius mirages for rut prevention.
pub struct NiodooTcsBridge {
    rut_mirage: RutMirage,
}

impl NiodooTcsBridge {
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            rut_mirage: RutMirage::new(embedding_dim),
        }
    }

    /// Generates a mirage embedding and corresponding diagnostics.
    pub fn generate_mirage(&self, embedding: &[f32]) -> MirageOutcome {
        self.rut_mirage.generate(embedding)
    }
}

#[derive(Clone)]
pub struct RutMirage {
    dim: usize,
    eigen_vectors: Vec<Vec<f32>>,
}

impl RutMirage {
    pub fn new(dim: usize) -> Self {
        let mut eigen_vectors = Vec::new();
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..3 {
            let mut vector = vec![0.0f32; dim];
            for value in vector.iter_mut() {
                *value = rng.gen_range(-0.2..0.2);
            }
            normalize(&mut vector);
            eigen_vectors.push(vector);
        }

        Self { dim, eigen_vectors }
    }

    pub fn generate(&self, embedding: &[f32]) -> MirageOutcome {
        let mut mirage = embedding.to_vec();
        if mirage.is_empty() {
            mirage.resize(self.dim.max(4), 0.0);
        }

        let signature = labyrinth_signature(&mirage, 0.4);
        let mut rng = StdRng::seed_from_u64((signature.entropy * 1_000.0) as u64 + 7);

        for eigen in &self.eigen_vectors {
            let intensity = rng.gen_range(0.02..0.08);
            for (value, eigen_component) in mirage.iter_mut().zip(eigen.iter()) {
                *value += intensity * eigen_component;
            }
        }

        normalize(&mut mirage);
        MirageOutcome {
            pad_projection: solve_mobius_labyrinth(&mirage, 0.4),
            entropy: signature.entropy,
            stability: signature.stability,
            mirage_embedding: mirage,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MirageOutcome {
    pub pad_projection: Vec<f32>,
    pub mirage_embedding: Vec<f32>,
    pub entropy: f32,
    pub stability: f32,
}

fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector.iter_mut() {
            *value /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mobius_labyrinth::PAD_SPACE_DIM;

    #[test]
    fn mirage_generation_preserves_dimension() {
        let bridge = NiodooTcsBridge::new(8);
        let base_embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5, -0.2, -0.1, 0.0];
        let outcome = bridge.generate_mirage(&base_embedding);
        assert_eq!(outcome.mirage_embedding.len(), base_embedding.len());
        assert_eq!(outcome.pad_projection.len(), PAD_SPACE_DIM);
    }
}
