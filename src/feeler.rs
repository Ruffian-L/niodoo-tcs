// Copyright (c) 2025 Jason Van Pham (ruffian-l on GitHub) @ The Niodoo Collaborative
// Licensed under the MIT License - See LICENSE file for details
// Attribution required for all derivative works

//! Predictive feeler probes with PAD valence scoring for code review

use rayon::prelude::*;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use anyhow::{Result, anyhow};
use std::f32::consts::PI;
use crate::EmotionalVector;

/// PAD (Pleasure, Arousal, Dominance) valence structure for emotional scoring
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PADValence {
    pub pleasure: f32,    // -1.0 to 1.0 (negative to positive)
    pub arousal: f32,     // -1.0 to 1.0 (calm to excited)
    pub dominance: f32,   // -1.0 to 1.0 (submissive to dominant)
}

/// Phantom Limb - Extended PAD with 2 ghost dimensions for latent affects
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PhantomPAD {
    // Core PAD dimensions (5D)
    pub pleasure: f32,
    pub arousal: f32,
    pub dominance: f32,
    pub joy: f32,         // From EmotionalVector
    pub sadness: f32,     // From EmotionalVector

    // Ghost dimensions (2D) - latent affects via VAE projection
    pub boredom_ghost: f32,    // Slow-drift shadow dimension
    pub curiosity_ghost: f32,  // High-frequency ripple dimension
}

impl PADValence {
    /// Calculate total valence score (pleasure + arousal - dominance)
    pub fn total(&self) -> f32 {
        self.pleasure + self.arousal - self.dominance
    }

    /// Create neutral valence
    pub fn neutral() -> Self {
        Self {
            pleasure: 0.0,
            arousal: 0.0,
            dominance: 0.0,
        }
    }

    /// Create positive valence (good code feel)
    pub fn positive() -> Self {
        Self {
            pleasure: 0.8,
            arousal: 0.3,
            dominance: 0.1,
        }
    }

    /// Create negative valence (bad code feel)
    pub fn negative() -> Self {
        Self {
            pleasure: -0.8,
            arousal: 0.6,
            dominance: 0.3,
        }
    }
}

impl PhantomPAD {
    /// Create from EmotionalVector (5D core)
    pub fn from_emotion_vector(ev: &EmotionalVector) -> Self {
        Self {
            pleasure: 0.0, // Will be inferred from context
            arousal: 0.0,  // Will be inferred from context
            dominance: 0.0,// Will be inferred from context
            joy: ev.joy,
            sadness: ev.sadness,
            boredom_ghost: 0.0,
            curiosity_ghost: 0.0,
        }
    }

    /// Project ghost dimensions using VAE-like approach on embed tail
    pub fn project_ghosts(&mut self, embed_tail: &[f32]) -> Result<()> {
        if embed_tail.len() < 10 {
            return Ok(()); // Not enough data for projection
        }

        // Variational projection: μ and σ from embed tail
        let (mu, sigma) = self.variational_projection(embed_tail)?;

        // Ghost dimensions: μ * sin(σ * u) on torus param u
        let u = self.calculate_torus_parameter();
        self.boredom_ghost = mu[0] * (sigma[0] * u).sin();
        self.curiosity_ghost = mu[1] * (sigma[1] * u).cos();

        // Clamp to reasonable ranges
        self.boredom_ghost = self.boredom_ghost.clamp(-1.0, 1.0);
        self.curiosity_ghost = self.curiosity_ghost.clamp(-1.0, 1.0);

        Ok(())
    }

    /// Variational autoencoder-style projection (simplified)
    fn variational_projection(&self, embed_tail: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        // Simple VAE projection using mean and std of embed tail
        let n = embed_tail.len() as f32;
        let mu_boredom = embed_tail.iter().take(5).sum::<f32>() / 5.0;
        let mu_curiosity = embed_tail.iter().skip(5).take(5).sum::<f32>() / 5.0;

        let sigma_boredom = (embed_tail.iter().take(5)
            .map(|x| (x - mu_boredom).powi(2))
            .sum::<f32>() / 5.0).sqrt();
        let sigma_curiosity = (embed_tail.iter().skip(5).take(5)
            .map(|x| (x - mu_curiosity).powi(2))
            .sum::<f32>() / 5.0).sqrt();

        Ok((vec![mu_boredom, mu_curiosity], vec![sigma_boredom, sigma_curiosity]))
    }

    /// Calculate torus parameter u for ghost projection
    fn calculate_torus_parameter(&self) -> f32 {
        // Use core emotions to parameterize the torus
        let base_u = (self.joy - self.sadness) * PI; // Emotional balance
        let arousal_mod = self.arousal * 0.5; // Arousal modulation

        (base_u + arousal_mod).rem_euclid(2.0 * PI)
    }

    /// Check if ghosts should promote to full PAD dimensions
    pub fn should_promote_ghosts(&self) -> (bool, bool) {
        let promote_boredom = self.boredom_ghost.abs() > 0.3;
        let promote_curiosity = self.curiosity_ghost.abs() > 0.3;

        (promote_boredom, promote_curiosity)
    }

    /// Promote ghost to full dimension (e.g., boredom → persistent state)
    pub fn promote_boredom_ghost(&mut self) {
        // Boredom torques PERSIST: increase dominance, decrease arousal
        self.dominance += self.boredom_ghost * 0.2;
        self.arousal -= self.boredom_ghost.abs() * 0.1;
        self.boredom_ghost *= 0.5; // Reduce ghost influence after promotion
    }

    /// Promote curiosity ghost to full dimension
    pub fn promote_curiosity_ghost(&mut self) {
        // Curiosity torques DISCOVER: increase pleasure, arousal
        self.pleasure += self.curiosity_ghost * 0.3;
        self.arousal += self.curiosity_ghost * 0.2;
        self.curiosity_ghost *= 0.5; // Reduce ghost influence after promotion
    }

    /// Calculate total valence including ghost influence
    pub fn total_phantom_valence(&self) -> f32 {
        let core_valence = self.pleasure + self.arousal - self.dominance;
        let ghost_influence = self.boredom_ghost * 0.1 - self.curiosity_ghost * 0.1;

        core_valence + ghost_influence
    }

    /// Get 7D vector representation
    pub fn as_vec(&self) -> [f32; 7] {
        [
            self.pleasure,
            self.arousal,
            self.dominance,
            self.joy,
            self.sadness,
            self.boredom_ghost,
            self.curiosity_ghost,
        ]
    }
}

/// Color classes for the 8-way probe system
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Color {
    Red,     // Aggression/anger
    Orange,  // Frustration/annoyance
    Yellow,  // Anxiety/fear
    Green,   // Calm/relief
    Cyan,    // Sadness/disappointment
    Blue,    // Trust/confidence
    Violet,  // Surprise/amazement
    Magenta, // Intuition/insight
}

impl Color {
    /// Get all 8 colors for probe spawning
    pub fn all_colors() -> [Self; 8] {
        [
            Self::Red,
            Self::Orange,
            Self::Yellow,
            Self::Green,
            Self::Cyan,
            Self::Blue,
            Self::Violet,
            Self::Magenta,
        ]
    }

    /// Get color index (0-7)
    pub fn index(&self) -> usize {
        match self {
            Self::Red => 0,
            Self::Orange => 1,
            Self::Yellow => 2,
            Self::Green => 3,
            Self::Cyan => 4,
            Self::Blue => 5,
            Self::Violet => 6,
            Self::Magenta => 7,
        }
    }
}

/// Predictive probe with trajectory and valence scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Probe {
    pub color: Color,
    pub trajectory: Vec<Vec<f32>>,  // Sequence of embedding points
    pub valence: PADValence,
    pub variance: f32,              // Trajectory variance for retraction logic
}

impl Probe {
    /// Create a new probe with initial state
    pub fn new(color: Color, initial_embedding: Vec<f32>) -> Self {
        let mut trajectory = Vec::new();
        trajectory.push(initial_embedding);

        Self {
            color,
            trajectory,
            valence: PADValence::neutral(),
            variance: 0.0,
        }
    }

    /// Add a point to the trajectory
    pub fn add_point(&mut self, embedding: Vec<f32>) {
        self.trajectory.push(embedding);
        self.update_variance();
    }

    /// Update trajectory variance for retraction logic
    fn update_variance(&mut self) {
        if self.trajectory.len() < 2 {
            self.variance = 0.0;
            return;
        }

        // Calculate variance of trajectory points
        let mean: Vec<f32> = self.trajectory.iter()
            .fold(vec![0.0; self.trajectory[0].len()], |acc, point| {
                acc.iter().zip(point.iter()).map(|(a, b)| a + b).collect()
            })
            .iter()
            .map(|&sum| sum / self.trajectory.len() as f32)
            .collect();

        let variance: f32 = self.trajectory.iter()
            .map(|point| {
                point.iter().zip(mean.iter())
                    .map(|(p, m)| (p - m).powi(2))
                    .sum::<f32>()
            })
            .sum::<f32>() / self.trajectory.len() as f32;

        self.variance = variance.sqrt();
    }

    /// Check if probe should be retracted (high variance = unstable)
    pub fn should_retract(&self, threshold: f32) -> bool {
        self.variance > threshold
    }
}

/// Spawn 8 parallel probes from input embedding
pub fn spawn_probes(input_emb: &[f32]) -> Vec<Probe> {
    Color::all_colors()
        .into_par_iter()
        .map(|color| {
            // Add color-specific perturbation to input embedding
            let mut perturbed_emb = input_emb.to_vec();
            let perturbation = match color {
                Color::Red => vec![0.1, 0.05, -0.05],      // Aggression boost
                Color::Orange => vec![0.05, 0.1, 0.0],     // Frustration
                Color::Yellow => vec![-0.05, 0.1, 0.05],   // Anxiety
                Color::Green => vec![0.0, -0.1, 0.0],      // Calm
                Color::Cyan => vec![0.05, -0.05, 0.1],     // Sadness
                Color::Blue => vec![0.0, 0.0, -0.1],       // Trust
                Color::Violet => vec![-0.1, 0.05, 0.05],   // Surprise
                Color::Magenta => vec![0.05, 0.05, 0.1],   // Intuition
            };

            // Apply perturbation (ensure bounds)
            for (i, &p) in perturbation.iter().enumerate() {
                if i < perturbed_emb.len() {
                    perturbed_emb[i] = (perturbed_emb[i] + p).clamp(-1.0, 1.0);
                }
            }

            Probe::new(color, perturbed_emb)
        })
        .collect()
}

/// Simulate trajectory evolution using RBF Gaussian processes + Möbius topology
pub fn simulate_trajectory(probe: &mut Probe, steps: usize) -> Result<()> {
    if probe.trajectory.is_empty() {
        return Err(anyhow!("Probe has no initial trajectory"));
    }

    let current_emb = probe.trajectory.last().unwrap().clone();
    let phi = (1.0_f32 + 5.0_f32.sqrt()) / 2.0; // Golden ratio for Möbius

    for step in 0..steps {
        // RBF forecast: Gaussian kernel on Möbius surface
        let next_emb = rbf_forecast(&current_emb, step as f32, phi)?;

        // Apply Möbius twist (rotation around golden ratio axis)
        let twisted_emb = apply_mobius_twist(&next_emb, phi, step as f32)?;

        probe.add_point(twisted_emb);
    }

    Ok(())
}

/// RBF forecast using Gaussian kernel on Möbius surface
fn rbf_forecast(current_emb: &[f32], step: f32, phi: f32) -> Result<Vec<f32>> {
    let sigma = 0.5_f32; // Kernel bandwidth
    let mut next_emb = vec![0.0; current_emb.len()];

    // Simple RBF: exp(-||x||^2 / (2σ^2)) + sinusoidal component for exploration
    for (i, &x) in current_emb.iter().enumerate() {
        let gaussian = (-x.powi(2) / (2.0 * sigma.powi(2))).exp();
        let oscillation = (step * phi * std::f32::consts::PI).sin();
        next_emb[i] = (gaussian + oscillation * 0.1).clamp(-1.0, 1.0);
    }

    Ok(next_emb)
}

/// Apply Möbius twist (rotation around golden ratio axis)
fn apply_mobius_twist(emb: &[f32], phi: f32, step: f32) -> Result<Vec<f32>> {
    let mut twisted = emb.to_vec();

    // Möbius twist: rotation around axis defined by golden ratio
    let angle = step * phi * std::f32::consts::PI / 180.0; // Convert to radians
    let cos_angle = angle.cos();
    let sin_angle = angle.sin();

    // Apply 3D rotation (simplified - would need proper quaternion/matrix)
    for i in 0..twisted.len().min(3) {
        let temp = twisted[i];
        twisted[i] = temp * cos_angle - twisted[(i + 1) % 3] * sin_angle;
        twisted[(i + 1) % 3] = temp * sin_angle + twisted[(i + 1) % 3] * cos_angle;
    }

    Ok(twisted)
}

/// Score trajectories and filter out unstable probes
pub fn score_and_filter(probes: &mut Vec<Probe>) -> Vec<Probe> {
    let variance_threshold = 0.3; // Retract if variance > threshold

    probes.retain(|probe| !probe.should_retract(variance_threshold));

    // Sort by valence total score
    probes.sort_by(|a, b| b.valence.total().partial_cmp(&a.valence.total()).unwrap());

    probes.clone()
}

/// Fuse top 3 probes into weighted vector
pub fn fuse_top_three(probes: &[Probe]) -> Vec<f32> {
    if probes.is_empty() {
        return vec![];
    }

    let top_three = &probes[0..probes.len().min(3)];
    let embedding_dim = if !probes.is_empty() && !probes[0].trajectory.is_empty() {
        probes[0].trajectory[0].len()
    } else {
        return vec![];
    };

    let mut fused = vec![0.0; embedding_dim];
    let mut total_weight = 0.0;

    for (i, probe) in top_three.iter().enumerate() {
        let weight = 1.0 / (i + 1) as f32; // Higher weight for better probes
        total_weight += weight;

        if let Some(last_point) = probe.trajectory.last() {
            for (j, &val) in last_point.iter().enumerate() {
                fused[j] += val * weight;
            }
        }
    }

    // Normalize
    if total_weight > 0.0 {
        fused.iter_mut().for_each(|x| *x /= total_weight);
    }

    fused
}

/// Score a probe's trajectory and update its valence
pub fn score_trajectory(probe: &mut Probe, code_context: &str) -> Result<()> {
    if probe.trajectory.len() < 2 {
        return Err(anyhow!("Probe needs at least 2 trajectory points for scoring"));
    }

    // Analyze trajectory stability and code sentiment
    let avg_valence = calculate_trajectory_valence(&probe.trajectory, code_context);

    // Update probe valence based on trajectory analysis
    probe.valence = avg_valence;

    Ok(())
}

/// Calculate PAD valence from trajectory and code context
fn calculate_trajectory_valence(trajectory: &[Vec<f32>], code_context: &str) -> PADValence {
    // Simple sentiment analysis based on code patterns
    let mut pleasure = 0.0_f32;
    let mut arousal = 0.0_f32;
    let mut dominance = 0.0_f32;

    // Check for positive patterns (good code indicators)
    let positive_patterns = ["good", "excellent", "clean", "efficient", "well-structured"];
    let negative_patterns = ["bad", "terrible", "messy", "inefficient", "buggy"];

    let context_lower = code_context.to_lowercase();

    for pattern in positive_patterns {
        if context_lower.contains(pattern) {
            pleasure += 0.2;
        }
    }

    for pattern in negative_patterns {
        if context_lower.contains(pattern) {
            pleasure -= 0.2;
        }
    }

    // Trajectory stability contributes to arousal
    let trajectory_stability = if trajectory.len() >= 2 {
        let first = &trajectory[0];
        let last = trajectory.last().unwrap();

        let distance: f32 = first.iter().zip(last.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        1.0 - (distance / first.len() as f32).min(1.0)
    } else {
        0.0
    };

    arousal = trajectory_stability;

    // Dominance based on trajectory confidence (lower variance = higher dominance)
    let trajectory_variance = calculate_trajectory_variance(trajectory);
    dominance = (1.0 - trajectory_variance).max(0.0);

    PADValence {
        pleasure: pleasure.clamp(-1.0, 1.0),
        arousal: arousal.clamp(-1.0, 1.0),
        dominance: dominance.clamp(-1.0, 1.0),
    }
}

/// Calculate variance of trajectory points
fn calculate_trajectory_variance(trajectory: &[Vec<f32>]) -> f32 {
    if trajectory.len() < 2 {
        return 0.0;
    }

    // Calculate mean for each dimension
    let dim = trajectory[0].len();
    let mut means = vec![0.0_f32; dim];

    for point in trajectory {
        for (i, &val) in point.iter().enumerate() {
            means[i] += val;
        }
    }

    for mean in means.iter_mut() {
        *mean /= trajectory.len() as f32;
    }

    // Calculate variance
    let mut variance = 0.0_f32;
    for point in trajectory {
        for (i, &val) in point.iter().enumerate() {
            variance += (val - means[i]).powi(2);
        }
    }

    variance / (trajectory.len() * dim) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_valence() {
        let neutral = PADValence::neutral();
        assert_eq!(neutral.total(), 0.0);

        let positive = PADValence::positive();
        assert!(positive.total() > 0.5);

        let negative = PADValence::negative();
        assert!(negative.total() < -0.5);
    }

    #[test]
    fn test_color_system() {
        let colors = Color::all_colors();
        assert_eq!(colors.len(), 8);
        assert_eq!(Color::Red.index(), 0);
        assert_eq!(Color::Magenta.index(), 7);
    }

    #[test]
    fn test_probe_creation() {
        let embedding = vec![0.1, 0.2, 0.3];
        let probe = Probe::new(Color::Red, embedding.clone());

        assert_eq!(probe.color, Color::Red);
        assert_eq!(probe.trajectory.len(), 1);
        assert_eq!(probe.trajectory[0], embedding);
        assert_eq!(probe.valence.total(), 0.0);
    }

    #[test]
    fn test_spawn_probes() {
        let input_emb = vec![0.5, 0.5, 0.5];
        let probes = spawn_probes(&input_emb);

        assert_eq!(probes.len(), 8);

        // Each probe should have different colors
        let colors: Vec<_> = probes.iter().map(|p| p.color).collect();
        assert_eq!(colors.len(), 8);
        assert!(colors.contains(&Color::Red));
        assert!(colors.contains(&Color::Magenta));
    }

    #[test]
    fn test_trajectory_variance() {
        let trajectory = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![0.5, 0.5, 0.5],
        ];

        let variance = calculate_trajectory_variance(&trajectory);
        assert!(variance > 0.0 && variance < 1.0);
    }
}






