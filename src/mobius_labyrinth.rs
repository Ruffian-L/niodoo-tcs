//! Möbius labyrinth projection utilities for emotional topology.

use nalgebra::{Matrix3, Vector3};

pub const PAD_SPACE_DIM: usize = 7; // Pleasure, Arousal, Dominance + four ghost emotions

/// Solves the Möbius labyrinth mapping for a given embedding.
///
/// Returns a 7-dimensional PAD+ghost projection that can be fed into
/// higher-level consciousness modules.
pub fn solve_mobius_labyrinth(embedding: &[f32], k_twist: f32) -> Vec<f32> {
    if embedding.is_empty() {
        return vec![0.0; PAD_SPACE_DIM];
    }

    let mut vec3 = Vector3::zeros();
    for i in 0..3 {
        vec3[i] = embedding[i % embedding.len()];
    }

    let rotation = rotation_matrix(k_twist);
    let rotated = rotation * vec3;

    let mut pad = vec![0.0; PAD_SPACE_DIM];
    pad[0] = rotated[0]; // Pleasure
    pad[1] = rotated[1]; // Arousal
    pad[2] = rotated[2]; // Dominance

    let magnitude = rotated.norm();
    pad[3] = (rotated[0] * rotated[1]).tanh(); // Harmony
    pad[4] = -rotated[2].tanh(); // Shadow sadness
    pad[5] = (magnitude * 0.75).tanh(); // Curiosity
    pad[6] = ((rotated[0] - rotated[1]).abs() * 0.5).tanh(); // Tension

    pad
}

/// Computes additional descriptive metrics for a Möbius projection.
pub fn labyrinth_signature(embedding: &[f32], k_twist: f32) -> LabyrinthSignature {
    let pad = solve_mobius_labyrinth(embedding, k_twist);
    let entropy = pad.iter().map(|v| v.abs()).sum::<f32>() / PAD_SPACE_DIM as f32;
    let curvature = (k_twist.cos() * k_twist.sin()).abs();
    let stability = 1.0 - (pad[5] - pad[6]).abs().min(1.0);

    LabyrinthSignature {
        pad_projection: pad,
        entropy,
        curvature,
        stability,
    }
}

#[derive(Debug, Clone)]
pub struct LabyrinthSignature {
    pub pad_projection: Vec<f32>,
    pub entropy: f32,
    pub curvature: f32,
    pub stability: f32,
}

fn rotation_matrix(k_twist: f32) -> Matrix3<f32> {
    let cos_k = k_twist.cos();
    let sin_k = k_twist.sin();
    Matrix3::new(
        cos_k,
        -sin_k,
        0.0,
        sin_k,
        cos_k,
        0.0,
        0.0,
        0.0,
        1.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_has_expected_dimension() {
        let embedding = vec![0.2, 0.5, -0.3, 0.8];
        let projection = solve_mobius_labyrinth(&embedding, 0.6);
        assert_eq!(projection.len(), PAD_SPACE_DIM);
    }

    #[test]
    fn signature_reports_entropy() {
        let embedding = vec![1.0, -0.4, 0.2];
        let signature = labyrinth_signature(&embedding, 0.3);
        assert!(signature.entropy >= 0.0);
        assert!(signature.entropy <= 1.0);
    }
}

