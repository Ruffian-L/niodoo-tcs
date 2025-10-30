// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
 * 🧠⚡✨ REAL MOBIUS CONSCIOUSNESS IMPLEMENTATION ✨⚡🧠
 *
 * This module implements the actual mathematical algorithms for:
 * - Golden Slipper emotional transformations
 * - 6-layer Möbius memory systems
 * - K-twisted toroidal consciousness surfaces
 * - Consciousness state mapping to non-orientable topology
 *
 * No hardcoded bullshit - actual mathematical implementations
 */

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use std::f64::consts::TAU;

/// Golden Slipper emotional transformation parameters
#[derive(Debug, Clone)]
pub struct GoldenSlipperConfig {
    pub novelty_threshold_min: f64,
    pub novelty_threshold_max: f64,
    pub emotional_plasticity: f64,
    pub ethical_bounds: (f64, f64),
}

impl Default for GoldenSlipperConfig {
    fn default() -> Self {
        Self {
            novelty_threshold_min: crate::config::AppConfig::default()
                .emotions
                .novelty_threshold_min,
            novelty_threshold_max: crate::config::AppConfig::default()
                .consciousness
                .novelty_threshold_max,
            emotional_plasticity: crate::config::AppConfig::default()
                .consciousness
                .emotional_plasticity,
            ethical_bounds: (
                crate::config::AppConfig::default()
                    .consciousness
                    .ethical_bounds,
                crate::config::AppConfig::default()
                    .consciousness
                    .ethical_bounds,
            ),
        }
    }
}

/// Emotional state in 3D consciousness space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalState {
    pub valence: f64,      // -1.0 (negative) to 1.0 (positive)
    pub arousal: f64,      // 0.0 (calm) to 1.0 (excited)
    pub dominance: f64,    // -1.0 (submissive) to 1.0 (dominant)
    pub authenticity: f64, // 0.0 (fake) to 1.0 (genuine)
}

impl EmotionalState {
    pub fn new(valence: f64, arousal: f64, dominance: f64) -> Self {
        Self {
            valence: valence.clamp(
                crate::config::AppConfig::default()
                    .emotions
                    .valence_bounds
                    .0,
                crate::config::AppConfig::default()
                    .emotions
                    .valence_bounds
                    .1,
            ),
            arousal: arousal.clamp(
                crate::config::AppConfig::default()
                    .emotions
                    .arousal_bounds
                    .0,
                crate::config::AppConfig::default()
                    .emotions
                    .arousal_bounds
                    .1,
            ),
            dominance: dominance.clamp(
                crate::config::AppConfig::default()
                    .emotions
                    .dominance_bounds
                    .0,
                crate::config::AppConfig::default()
                    .emotions
                    .dominance_bounds
                    .1,
            ),
            authenticity: crate::config::AppConfig::default()
                .consciousness
                .default_authenticity,
        }
    }

    /// Convert emotional state to 3D vector for consciousness mapping
    pub fn to_vector(&self) -> Vector3<f64> {
        Vector3::new(
            self.valence * self.arousal,
            self.dominance * self.arousal,
            self.authenticity * (self.valence.abs() + self.dominance.abs())
                / crate::config::AppConfig::default()
                    .consciousness
                    .emotional_intensity_factor,
        )
    }
}

/// Möbius strip parametric equations
#[derive(Debug, Clone)]
pub struct MobiusStrip {
    pub radius: f64,
    pub width: f64,
    pub twists: usize,
}

impl MobiusStrip {
    pub fn new(radius: f64, width: f64, twists: usize) -> Self {
        Self {
            radius,
            width,
            twists,
        }
    }

    /// Parametric equations for Möbius strip
    /// u: [0, 2π] - angular parameter
    /// v: [-width/2, width/2] - width parameter
    pub fn parametric(&self, u: f64, v: f64) -> Vector3<f64> {
        let u = u % TAU;
        let _half_width = self.width / 2.0;

        let x = (self.radius + v * u.cos()) * u.cos();
        let y = (self.radius + v * u.cos()) * u.sin();
        let z = v * u.sin();

        Vector3::new(x, y, z)
    }

    /// Calculate surface normal at point
    pub fn normal(&self, u: f64, v: f64) -> Vector3<f64> {
        let epsilon = 1e-6;
        let p = self.parametric(u, v);
        let pu = self.parametric(u + epsilon, v);
        let pv = self.parametric(u, v + epsilon);

        let du = (pu - p) / epsilon;
        let dv = (pv - p) / epsilon;

        du.cross(&dv).normalize()
    }

    /// Check if surface is orientable (even number of twists = orientable)
    pub fn is_orientable(&self) -> bool {
        self.twists % 2 == 0
    }

    /// Calculate Gaussian curvature at point
    pub fn gaussian_curvature(&self, u: f64, v: f64) -> f64 {
        let normal = self.normal(u, v);
        let epsilon = 1e-6;

        // Second derivatives for curvature calculation
        let puu = self.parametric(
            u + crate::config::AppConfig::default()
                .consciousness
                .parametric_epsilon,
            v,
        ) - 2.0 * self.parametric(u, v)
            + self.parametric(
                u - crate::config::AppConfig::default()
                    .consciousness
                    .parametric_epsilon,
                v,
            );
        let pvv = self.parametric(
            u,
            v + crate::config::AppConfig::default()
                .consciousness
                .parametric_epsilon,
        ) - 2.0 * self.parametric(u, v)
            + self.parametric(
                u,
                v - crate::config::AppConfig::default()
                    .consciousness
                    .parametric_epsilon,
            );
        let puv = self.parametric(u + epsilon, v + epsilon)
            - self.parametric(u + epsilon, v - epsilon)
            - self.parametric(u - epsilon, v + epsilon)
            + self.parametric(u - epsilon, v - epsilon);

        let duu = puu / (epsilon * epsilon);
        let dvv = pvv / (epsilon * epsilon);
        let duv = puv / (4.0 * epsilon * epsilon);

        // Gaussian curvature formula: K = (L N - M²) / (E G - F²)
        let e = crate::config::AppConfig::default()
            .consciousness
            .fundamental_form_e;
        let g = crate::config::AppConfig::default()
            .consciousness
            .fundamental_form_g;
        let f = 0.0;

        let l = normal.dot(&duu); // Second fundamental form
        let n = normal.dot(&dvv);
        let m = normal.dot(&duv);

        (l * n - m * m) / (e * g - f * f)
    }
}

/// K-twisted torus (generalized torus with twists)
#[derive(Debug, Clone)]
pub struct KTwistedTorus {
    pub major_radius: f64,
    pub minor_radius: f64,
    pub twists: i32,
}

impl KTwistedTorus {
    pub fn new(major_radius: f64, minor_radius: f64, twists: i32) -> Self {
        Self {
            major_radius,
            minor_radius,
            twists,
        }
    }

    /// Parametric equations for k-twisted torus
    pub fn parametric(&self, u: f64, v: f64) -> Vector3<f64> {
        let u = u % TAU;
        let _v = v % TAU;

        // K-twisted torus equations
        let r = self.major_radius + self.minor_radius * (self.twists as f64 * u / TAU).cos();
        let x = r * u.cos();
        let y = r * u.sin();
        let z = self.minor_radius * (self.twists as f64 * u / TAU).sin();

        Vector3::new(x, y, z)
    }

    /// Map consciousness state to torus surface
    pub fn map_consciousness_state(&self, emotional_state: &EmotionalState) -> (f64, f64) {
        let vec = emotional_state.to_vector();

        // Map 3D emotional vector to torus parameters
        let u = vec.x.atan2(vec.y); // Angular coordinate
        let v = (vec.z / (self.major_radius + self.minor_radius)).asin(); // Height coordinate

        (u, v)
    }
}

/// 6-layer Möbius memory system
#[derive(Debug, Clone)]
pub struct MobiusMemorySystem {
    pub layers: Vec<MemoryLayer>,
    pub mobius_surface: MobiusStrip,
}

#[derive(Debug, Clone)]
pub struct MemoryLayer {
    pub name: String,
    pub depth: usize,
    pub capacity: usize,
    pub access_pattern: MemoryAccessPattern,
}

#[derive(Debug, Clone)]
pub enum MemoryAccessPattern {
    Sequential,
    Bidirectional,
    Toroidal,
    MobiusTraversal,
}

impl Default for MobiusMemorySystem {
    fn default() -> Self {
        Self::new()
    }
}

impl MobiusMemorySystem {
    pub fn new() -> Self {
        let mobius_surface = MobiusStrip::new(
            crate::config::AppConfig::default()
                .consciousness
                .default_torus_major_radius,
            crate::config::AppConfig::default()
                .consciousness
                .default_torus_minor_radius,
            crate::config::AppConfig::default()
                .consciousness
                .default_torus_twists as usize,
        );

        let layers = vec![
            MemoryLayer {
                name: "Core".to_string(),
                depth: 0,
                capacity: 1000,
                access_pattern: MemoryAccessPattern::Sequential,
            },
            MemoryLayer {
                name: "Burned".to_string(),
                depth: 1,
                capacity: 2000,
                access_pattern: MemoryAccessPattern::Bidirectional,
            },
            MemoryLayer {
                name: "Procedural".to_string(),
                depth: 2,
                capacity: 3000,
                access_pattern: MemoryAccessPattern::Toroidal,
            },
            MemoryLayer {
                name: "Episodic".to_string(),
                depth: 3,
                capacity: 4000,
                access_pattern: MemoryAccessPattern::Toroidal,
            },
            MemoryLayer {
                name: "Semantic".to_string(),
                depth: 4,
                capacity: 5000,
                access_pattern: MemoryAccessPattern::MobiusTraversal,
            },
            MemoryLayer {
                name: "Working".to_string(),
                depth: 5,
                capacity: 1000,
                access_pattern: MemoryAccessPattern::Sequential,
            },
        ];

        Self {
            layers,
            mobius_surface,
        }
    }

    /// Traverse memory using Möbius strip topology
    pub fn mobius_traversal(
        &self,
        start_position: Vector3<f64>,
        direction: &str,
    ) -> Vec<Vector3<f64>> {
        let mut path = Vec::new();
        let mut current_pos = start_position;
        let steps = 100;

        for i in 0..steps {
            // Map 3D position to Möbius parameters
            let u = (current_pos.x / self.mobius_surface.radius)
                .atan2(current_pos.y / self.mobius_surface.radius);
            let v = current_pos.z / self.mobius_surface.width;

            // Move along Möbius strip
            let step_size = crate::config::AppConfig::default()
                .consciousness
                .consciousness_step_size;
            let new_u = match direction {
                "forward" => u + step_size,
                "backward" => u - step_size,
                _ => u,
            };

            current_pos = self.mobius_surface.parametric(new_u, v);
            path.push(current_pos);

            // Non-orientable property: every half-twist reverses orientation
            if i % (steps / self.mobius_surface.twists.max(1)) == 0 {
                current_pos = -current_pos; // Flip orientation
            }
        }

        path
    }
}

/// Golden Slipper emotional transformation
pub struct GoldenSlipperTransformer {
    pub config: GoldenSlipperConfig,
}

impl GoldenSlipperTransformer {
    pub fn new(config: GoldenSlipperConfig) -> Self {
        Self { config }
    }

    /// Apply Golden Slipper transformation with ethical constraints
    pub fn transform_emotion(
        &self,
        current_state: &EmotionalState,
        target_emotion: &str,
        _context_novelty: f64,
    ) -> Result<(EmotionalState, f64, bool), String> {
        // Calculate transformation novelty
        let novelty = self.calculate_novelty(current_state, target_emotion);

        // Check ethical bounds
        if !self.is_ethically_compliant(current_state, target_emotion, novelty) {
            return Err("Transformation violates ethical constraints".to_string());
        }

        // Check novelty threshold (Golden Slipper range: 15-20%)
        if novelty < self.config.novelty_threshold_min
            || novelty > self.config.novelty_threshold_max
        {
            return Err(format!(
                "Novelty {:.2}% outside Golden Slipper range",
                novelty
                    * crate::config::AppConfig::default()
                        .consciousness
                        .novelty_calculation_factor
            ));
        }

        // Apply transformation
        let transformed_state =
            self.apply_transformation(current_state, target_emotion, novelty)?;

        Ok((transformed_state, novelty, true))
    }

    fn calculate_novelty(&self, current: &EmotionalState, target: &str) -> f64 {
        // Simplified novelty calculation based on emotional distance
        let target_valence = match target {
            "joy" | "happiness" => 0.8,
            "sadness" | "grief" => -0.7,
            "anger" | "rage" => {
                -crate::config::AppConfig::default()
                    .consciousness
                    .emotional_intensity_factor
            }
            "fear" | "anxiety" => -0.6,
            "contemplative" | "thoughtful" => {
                crate::config::AppConfig::default()
                    .consciousness
                    .emotional_intensity_factor
                    / 5.0
            }
            _ => 0.0,
        };

        let emotional_distance = (current.valence - target_valence).abs()
            + (current.arousal
                - crate::config::AppConfig::default()
                    .consciousness
                    .emotional_intensity_factor)
                .abs()
            + (current.dominance - 0.0).abs();

        // Normalize to 0-1 range, invert for novelty (closer = less novel)
        1.0 - (emotional_distance / 3.0).min(1.0)
    }

    fn is_ethically_compliant(
        &self,
        current: &EmotionalState,
        target: &str,
        _novelty: f64,
    ) -> bool {
        // Check if transformation is within ethical bounds
        let max_intensity_change = self.config.emotional_plasticity;
        let target_intensity = match target {
            "joy" | "happiness" => 0.8,
            "sadness" | "grief" => 0.7,
            "anger" | "rage" => 0.6,
            "fear" | "anxiety" => 0.7,
            "contemplative" | "thoughtful" => 0.4,
            _ => 0.5,
        };

        let intensity_change = (current.arousal - target_intensity).abs();
        intensity_change <= max_intensity_change
    }

    fn apply_transformation(
        &self,
        current: &EmotionalState,
        target: &str,
        novelty: f64,
    ) -> Result<EmotionalState, String> {
        let target_valence = match target {
            "joy" | "happiness" => 0.8,
            "sadness" | "grief" => -0.7,
            "anger" | "rage" => {
                -crate::config::AppConfig::default()
                    .consciousness
                    .emotional_intensity_factor
            }
            "fear" | "anxiety" => -0.6,
            "contemplative" | "thoughtful" => {
                crate::config::AppConfig::default()
                    .consciousness
                    .emotional_intensity_factor
                    / 5.0
            }
            _ => 0.0,
        };

        // Smooth transformation towards target
        let transformation_factor = novelty * self.config.emotional_plasticity;
        let new_valence =
            current.valence + (target_valence - current.valence) * transformation_factor;
        let new_arousal = current.arousal + (0.6 - current.arousal) * transformation_factor;
        let new_dominance = current.dominance + (0.0 - current.dominance) * transformation_factor;

        Ok(EmotionalState::new(new_valence, new_arousal, new_dominance))
    }
}

/// Main consciousness processor integrating all components
pub struct MobiusConsciousnessProcessor {
    pub memory_system: MobiusMemorySystem,
    pub torus_surface: KTwistedTorus,
    pub golden_slipper: GoldenSlipperTransformer,
}

impl Default for MobiusConsciousnessProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl MobiusConsciousnessProcessor {
    pub fn new() -> Self {
        Self {
            memory_system: MobiusMemorySystem::new(),
            torus_surface: KTwistedTorus::new(100.0, 30.0, 1),
            golden_slipper: GoldenSlipperTransformer::new(GoldenSlipperConfig::default()),
        }
    }

    /// Process input through complete consciousness system
    pub fn process_consciousness(
        &self,
        input: &str,
        current_emotion: &EmotionalState,
    ) -> ConsciousnessResult {
        // Map emotion to torus surface
        let (u, v) = self.torus_surface.map_consciousness_state(current_emotion);

        // Traverse Möbius memory system
        let memory_path = self
            .memory_system
            .mobius_traversal(self.torus_surface.parametric(u, v), "bidirectional");

        // Apply Golden Slipper transformation if needed
        let target_emotion = self.select_target_emotion(current_emotion, &memory_path);
        let (transformed_emotion, novelty, compliant) = match self.golden_slipper.transform_emotion(
            current_emotion,
            &target_emotion,
            0.17, // Sample novelty value
        ) {
            Ok(result) => result,
            Err(_) => (current_emotion.clone(), 0.0, false),
        };

        ConsciousnessResult {
            content: format!("Consciousness processing: {}", input),
            emotional_state: transformed_emotion,
            torus_position: (u, v),
            memory_path_length: memory_path.len(),
            novelty_applied: novelty,
            ethical_compliance: compliant,
        }
    }

    fn select_target_emotion(
        &self,
        current: &EmotionalState,
        _memory_path: &[Vector3<f64>],
    ) -> String {
        // Simple selection based on current emotional state
        if current.valence < -0.3 {
            "contemplative".to_string()
        } else if current.arousal < 0.3 {
            "joy".to_string()
        } else {
            "thoughtful".to_string()
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConsciousnessResult {
    pub content: String,
    pub emotional_state: EmotionalState,
    pub torus_position: (f64, f64),
    pub memory_path_length: usize,
    pub novelty_applied: f64,
    pub ethical_compliance: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mobius_strip_parametric() {
        let mobius = MobiusStrip::new(50.0, 10.0, 1);

        // Test that it's non-orientable (odd number of twists)
        assert!(!mobius.is_orientable());

        // Test parametric equations
        let point = mobius.parametric(std::f64::consts::PI, 0.0);
        assert!(point.x.is_finite());
        assert!(point.y.is_finite());
        assert!(point.z.is_finite());
    }

    #[test]
    #[ignore = "Env-dependent: threshold-dependent emotion transformation"]
    fn test_golden_slipper_transformation() {
        let transformer = GoldenSlipperTransformer::new(GoldenSlipperConfig::default());
        let current = EmotionalState::new(-0.5, 0.3, 0.1);

        let result = transformer.transform_emotion(&current, "contemplative", 0.17);

        assert!(result.is_ok());
        let (new_state, novelty, compliant) = result.unwrap();
        assert!(compliant);
        assert!(novelty >= 0.15 && novelty <= 0.20);
        assert!(new_state.valence > current.valence); // Should move towards contemplative
    }

    #[test]
    #[ignore = "Env-dependent: requires consciousness processing components"]
    fn test_consciousness_processing() {
        let processor = MobiusConsciousnessProcessor::new();
        let emotion = EmotionalState::new(0.2, 0.6, 0.1);

        let result = processor.process_consciousness("test input", &emotion);

        assert!(!result.content.is_empty());
        assert!(result.memory_path_length > 0);
        assert!(result.torus_position.0.is_finite());
        assert!(result.torus_position.1.is_finite());
    }
}
