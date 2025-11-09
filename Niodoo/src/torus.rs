//! K-Twisted Torus Projection for PAD Emotional Manifold
//!
//! This module implements the topological manifold that serves as the
//! cognitive-affective substrate for the NIODOO system. The k-twisted torus
//! is the geometric realization of the Pleasure-Arousal-Dominance (PAD) model
//! extended with 4 "ghost" dimensions for additional cognitive state.
//!
//! Mathematical foundation:
//! x(u,v) = (R + v*cos(2ku)) * cos(u)
//! y(u,v) = (R + v*cos(2ku)) * sin(u)
//! z(u,v) = v * sin(2ku)

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Configuration for the k-twisted torus manifold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TorusConfig {
    /// Major radius (R) - overall size of the torus
    pub major_radius: f32,
    /// Strip width - maximum range of v parameter
    pub strip_width: f32,
    /// Number of half-twists (k parameter)
    /// k=1: non-orientable (Möbius-like)
    /// k=2: orientable
    pub twists: i32,
    /// Random seed for deterministic projection
    pub seed: u64,
}

impl Default for TorusConfig {
    fn default() -> Self {
        Self {
            major_radius: 2.0,
            strip_width: 0.5,
            twists: 1,
            seed: 42,
        }
    }
}

impl TorusConfig {
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read torus config from {}", path))?;
        toml::from_str(&content).context("failed to parse torus config")
    }
}

/// PAD (Pleasure-Arousal-Dominance) state on the 7D manifold
///
/// The first 3 dimensions are the classic PAD emotional model:
/// - Pleasure: valence (positive/negative emotion)
/// - Arousal: intensity (calm/excited)
/// - Dominance: control (submissive/dominant)
///
/// The remaining 4 "ghost" dimensions capture additional cognitive state
/// that doesn't fit neatly into the PAD model but influences behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PadState {
    /// 7D coordinates on the manifold: [P, A, D, g1, g2, g3, g4]
    pub coordinates: [f64; 7],
    /// Shannon entropy of the coordinate distribution
    pub entropy: f64,
    /// Variational mean (μ) from embedding projection
    pub mu: [f64; 7],
    /// Variational std dev (σ) from embedding projection
    pub sigma: [f64; 7],
    /// 3D position on the torus surface (for visualization)
    pub surface_position: [f32; 3],
}

impl PadState {
    /// Get Pleasure component (index 0)
    pub fn pleasure(&self) -> f64 {
        self.coordinates[0]
    }

    /// Get Arousal component (index 1)
    pub fn arousal(&self) -> f64 {
        self.coordinates[1]
    }

    /// Get Dominance component (index 2)
    pub fn dominance(&self) -> f64 {
        self.coordinates[2]
    }

    /// Get ghost dimensions (indices 3-6)
    pub fn ghost_dimensions(&self) -> [f64; 4] {
        [
            self.coordinates[3],
            self.coordinates[4],
            self.coordinates[5],
            self.coordinates[6],
        ]
    }
}

/// K-Twisted Torus Projector
///
/// Projects high-dimensional embeddings onto the 7D PAD manifold using
/// a variational approach (reparameterization trick) for smooth, differentiable
/// projections.
pub struct TorusProjector {
    config: TorusConfig,
    rng: rand::rngs::StdRng,
}

impl TorusProjector {
    pub fn new(config: TorusConfig) -> Self {
        use rand::SeedableRng;
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(config.seed),
            config,
        }
    }

    /// Project a high-dimensional embedding onto the 7D PAD manifold
    ///
    /// This implements a lightweight VAE-style projection:
    /// 1. Split embedding into μ (mean) and logvar (log variance) heads
    /// 2. Apply reparameterization trick: z = μ + σ * ε
    /// 3. Map to torus surface using parametric equations
    /// 4. Compute entropy of the resulting distribution
    pub fn project(&mut self, embedding: &[f32]) -> Result<PadState> {
        use rand::Rng;
        
        anyhow::ensure!(
            embedding.len() >= 14,
            "embedding must be at least 14 dims (7 for μ + 7 for logvar)"
        );

        // Extract variational parameters from embedding
        let head_width = 7;
        let mut mu = [0.0f64; 7];
        let mut logvar = [0.0f64; 7];
        
        for i in 0..head_width {
            mu[i] = embedding[i] as f64;
            logvar[i] = embedding[head_width + i] as f64;
        }

        // Compute sigma from logvar (softplus for positivity)
        let mut sigma = [0.0f64; 7];
        for i in 0..7 {
            // σ = softplus(logvar) = log(1 + exp(logvar))
            // Clamped for numerical stability
            let lv = logvar[i].clamp(-10.0, 10.0);
            sigma[i] = (1.0 + lv.exp()).ln().max(0.1);
        }

        // Reparameterization trick: z = μ + σ * ε
        let mut coordinates = [0.0f64; 7];
        for i in 0..7 {
            let eps: f64 = self.rng.sample(rand_distr::StandardNormal);
            coordinates[i] = mu[i] + sigma[i] * eps;
        }

        // Wrap to manifold using tanh (maps to [-1, 1])
        for coord in coordinates.iter_mut() {
            *coord = coord.tanh();
        }

        // Compute Shannon entropy from the coordinate distribution
        let entropy = self.compute_entropy(&coordinates);

        // Map first two PAD dimensions (Pleasure, Arousal) to torus surface
        // using the k-twisted parametric equations
        let surface_position = self.map_to_surface(coordinates[0] as f32, coordinates[1] as f32);

        Ok(PadState {
            coordinates,
            entropy,
            mu,
            sigma,
            surface_position,
        })
    }

    /// Map 2D coordinates to 3D torus surface using k-twisted parametric equations
    fn map_to_surface(&self, pleasure: f32, arousal: f32) -> [f32; 3] {
        // Map [-1, 1] to [0, 2π] for u parameter (toroidal angle)
        let u = (pleasure + 1.0) * PI;
        
        // Map [-1, 1] to [-strip_width/2, strip_width/2] for v parameter
        let v = arousal * self.config.strip_width / 2.0;

        let k = self.config.twists as f32;
        let r = self.config.major_radius;

        // Core k-twisted torus parametric equations:
        let twist_factor = 2.0 * k * u;
        let radius_at_u = r + v * twist_factor.cos();

        let x = radius_at_u * u.cos();
        let y = radius_at_u * u.sin();
        let z = v * twist_factor.sin();

        [x, y, z]
    }

    /// Compute Shannon entropy of the coordinate distribution
    fn compute_entropy(&self, coords: &[f64; 7]) -> f64 {
        // Convert coordinates to probability simplex
        let mut probs = [0.0f64; 7];
        let mut sum = 0.0;
        
        for (i, &coord) in coords.iter().enumerate() {
            // Map [-1, 1] to [0, 1]
            let p = (coord + 1.0) / 2.0;
            probs[i] = p;
            sum += p;
        }

        // Normalize to sum to 1
        if sum > 1e-10 {
            for p in probs.iter_mut() {
                *p /= sum;
            }
        }

        // Compute Shannon entropy: H = -Σ p_i * log(p_i)
        let mut entropy = 0.0;
        for &p in probs.iter() {
            if p > 1e-10 {
                entropy -= p * p.ln();
            }
        }

        entropy
    }
}

/// K-Twisted Torus Generator for mesh export (visualization)
///
/// This generates the actual 3D mesh data for the torus surface,
/// useful for debugging and visualization in tools like Blender.
pub struct KTwistedTorusGenerator {
    pub major_radius: f32,
    pub strip_width: f32,
    pub twists: i32,
    pub u_steps: usize,
    pub v_steps: usize,
}

impl KTwistedTorusGenerator {
    pub fn new(
        major_radius: f32,
        strip_width: f32,
        twists: i32,
        u_steps: usize,
        v_steps: usize,
    ) -> Self {
        Self {
            major_radius,
            strip_width,
            twists,
            u_steps,
            v_steps,
        }
    }

    /// Calculate 3D position using k-twisted parametric equations
    pub fn calculate_position(&self, u: f32, v: f32) -> [f32; 3] {
        let k = self.twists as f32;
        let r = self.major_radius;

        let twist_factor = 2.0 * k * u;
        let radius_at_u = r + v * twist_factor.cos();

        let x = radius_at_u * u.cos();
        let y = radius_at_u * u.sin();
        let z = v * twist_factor.sin();

        [x, y, z]
    }

    /// Export mesh to OBJ file for visualization
    pub fn export_to_obj(&self, path: &str) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path)?;

        writeln!(file, "# K-Twisted Toroidal Surface")?;
        writeln!(
            file,
            "# R={}, w={}, k={}",
            self.major_radius, self.strip_width, self.twists
        )?;

        // Generate vertices
        for i in 0..self.u_steps {
            for j in 0..self.v_steps {
                let u = (i as f32 / self.u_steps as f32) * 2.0 * PI;
                let v_norm = (j as f32 / (self.v_steps - 1) as f32) - 0.5;
                let v = v_norm * self.strip_width;

                let pos = self.calculate_position(u, v);
                writeln!(file, "v {} {} {}", pos[0], pos[1], pos[2])?;
            }
        }

        // Generate faces
        for i in 0..(self.u_steps - 1) {
            for j in 0..(self.v_steps - 1) {
                let i0 = (i * self.v_steps + j) + 1; // OBJ is 1-indexed
                let i1 = ((i + 1) * self.v_steps + j) + 1;
                let i2 = (i * self.v_steps + j + 1) + 1;
                let i3 = ((i + 1) * self.v_steps + j + 1) + 1;

                writeln!(file, "f {} {} {}", i0, i1, i2)?;
                writeln!(file, "f {} {} {}", i1, i3, i2)?;
            }
        }

        Ok(())
    }
}

use anyhow::Context;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_torus_projection() {
        let config = TorusConfig::default();
        let mut projector = TorusProjector::new(config);

        // Create a mock embedding (768 dims, but we only use first 14)
        let mut embedding = vec![0.0f32; 768];
        for i in 0..14 {
            embedding[i] = (i as f32) * 0.1;
        }

        let result = projector.project(&embedding);
        assert!(result.is_ok());

        let pad_state = result.unwrap();
        assert_eq!(pad_state.coordinates.len(), 7);
        assert!(pad_state.entropy >= 0.0);
        
        // Coordinates should be in [-1, 1] after tanh
        for &coord in pad_state.coordinates.iter() {
            assert!(coord >= -1.0 && coord <= 1.0);
        }
    }

    #[test]
    fn test_surface_mapping() {
        let config = TorusConfig::default();
        let mut projector = TorusProjector::new(config);

        let embedding = vec![0.5f32; 768];
        let pad_state = projector.project(&embedding).unwrap();

        // Surface position should be non-zero
        let pos = pad_state.surface_position;
        let magnitude = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
        assert!(magnitude > 0.0);
    }

    #[test]
    fn test_obj_export() {
        let generator = KTwistedTorusGenerator::new(2.0, 0.5, 1, 16, 8);
        let result = generator.export_to_obj("/tmp/test_torus.obj");
        assert!(result.is_ok());
    }
}

