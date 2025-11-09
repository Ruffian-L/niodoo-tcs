//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
 * 🌀 MÖBIUS FLIP INTEGRATION - Low-Confidence Emotional Transformation
 *
 * Agent 4: FlipIntegrator Implementation
 *
 * Applies z-twist + jitter to boost joy when Qwen confidence < 0.5
 * Connects Möbius topology with Qwen inference for ethical nurturing
 *
 * Mathematical Foundation:
 * - Möbius strip: z-twist by π (180°) inverts emotional polarity
 * - Jitter: 15-20% random perturbation (noveltyVariance)
 * - Joy boost: joyIntensity = 1.0 - sadnessIntensity after flip
 */

use anyhow::{anyhow, Result};
use rand::Rng;
use std::f32::consts::PI;
use tracing::{info, warn};

/// Emotional vector representation in 3D space
#[derive(Debug, Clone)]
pub struct EmotionalVector {
    pub x: f32, // Sadness ← → Joy axis
    pub y: f32, // Fear ← → Calm axis
    pub z: f32, // Anger ← → Peace axis
}

impl EmotionalVector {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Calculate magnitude for normalization
    pub fn magnitude(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Normalize vector to unit length
    pub fn normalize(&mut self) {
        let mag = self.magnitude();
        if mag > 0.0 {
            self.x /= mag;
            self.y /= mag;
            self.z /= mag;
        }
    }

    /// Calculate dot product with another vector
    pub fn dot(&self, other: &EmotionalVector) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

/// Möbius flip configuration
#[derive(Debug, Clone)]
pub struct MobiusFlipConfig {
    /// Confidence threshold for triggering flip (default: 0.5)
    pub confidence_threshold: f32,
    /// Novelty variance for jitter (default: 0.175 = 17.5%)
    pub novelty_variance: f32,
    /// Joy boost multiplier after flip (default: 1.0)
    pub joy_boost_multiplier: f32,
    /// Enable visual proof logging
    pub visual_proof_enabled: bool,
}

impl Default for MobiusFlipConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            novelty_variance: 0.175, // 17.5% middle of 15-20% range
            joy_boost_multiplier: 1.0,
            visual_proof_enabled: true,
        }
    }
}

/// Möbius flip processor - integrates with Qwen inference
pub struct MobiusFlipProcessor {
    config: MobiusFlipConfig,
}

impl MobiusFlipProcessor {
    pub fn new(config: MobiusFlipConfig) -> Self {
        Self { config }
    }

    /// Main integration point: Apply Möbius flip on low-confidence Qwen outputs
    ///
    /// # Arguments
    /// * `emotion_vec` - Input emotional vector (sadness, fear, anger)
    /// * `confidence` - Qwen confidence score [0, 1]
    ///
    /// # Returns
    /// * Transformed emotional vector with joy boost and jitter
    pub fn mobius_flip_on_low_confidence(
        &self,
        emotion_vec: Vec<f32>,
        confidence: f32,
    ) -> Result<Vec<f32>> {
        // Validation
        if emotion_vec.len() != 3 {
            return Err(anyhow!(
                "Invalid emotion vector length: expected 3, got {}",
                emotion_vec.len()
            ));
        }

        if !(0.0..=1.0).contains(&confidence) {
            warn!(
                "Confidence out of bounds [0, 1]: {:.3}, clamping...",
                confidence
            );
        }

        let confidence = confidence.clamp(0.0, 1.0);

        // Convert to EmotionalVector
        let emotional_state = EmotionalVector::new(
            emotion_vec[0], // sadness
            emotion_vec[1], // fear
            emotion_vec[2], // anger
        );

        // Check if flip should be applied
        if confidence >= self.config.confidence_threshold {
            info!(
                "✅ Confidence {:.3} >= threshold {:.3}, no flip needed",
                confidence, self.config.confidence_threshold
            );
            return Ok(emotion_vec);
        }

        // Apply Möbius flip transformation
        info!(
            "🌀 Low confidence {:.3} < {:.3}, applying Möbius flip...",
            confidence, self.config.confidence_threshold
        );

        let flipped = self.apply_mobius_flip(&emotional_state, confidence)?;

        // Visual proof output
        if self.config.visual_proof_enabled {
            self.log_visual_proof(&emotional_state, &flipped, confidence);
        }

        Ok(vec![flipped.x, flipped.y, flipped.z])
    }

    /// Apply the core Möbius flip transformation
    ///
    /// Steps:
    /// 1. Z-twist: Rotate emotional vector by π on Möbius strip
    /// 2. Jitter: Add novelty variance (15-20%) random perturbation
    /// 3. Joy boost: Transform sadness → joy (joyIntensity = 1.0 - sadnessIntensity)
    fn apply_mobius_flip(
        &self,
        input: &EmotionalVector,
        confidence: f32,
    ) -> Result<EmotionalVector> {
        let mut rng = rand::thread_rng();

        // Step 1: Z-twist by π (180° rotation on Möbius strip)
        // This inverts the emotional polarity: sadness ↔ joy
        let twist_angle = PI; // 180 degrees
        let cos_twist = twist_angle.cos();
        let sin_twist = twist_angle.sin();

        // Rotate around Z-axis (preserves z component, flips x-y plane)
        let mut flipped = EmotionalVector::new(
            input.x * cos_twist - input.y * sin_twist, // Rotate X
            input.x * sin_twist + input.y * cos_twist, // Rotate Y
            -input.z,                                  // Invert Z (Möbius characteristic)
        );

        // Step 2: Add novelty jitter (15-20% perturbation)
        let jitter_strength = (1.0 - confidence) * self.config.novelty_variance;
        let jitter = EmotionalVector::new(
            (rng.r#gen::<f32>() - 0.5) * jitter_strength,
            (rng.r#gen::<f32>() - 0.5) * jitter_strength,
            (rng.r#gen::<f32>() - 0.5) * jitter_strength,
        );

        flipped.x += jitter.x;
        flipped.y += jitter.y;
        flipped.z += jitter.z;

        // Step 3: Joy boost transformation
        // Map sadness axis (negative x) to joy axis (positive x)
        let sadness_intensity = (-input.x).max(0.0); // Extract sadness component
        let joy_boost = sadness_intensity * self.config.joy_boost_multiplier;

        // Flip sadness to joy on x-axis
        flipped.x = joy_boost - input.x; // Transform: sadness → joy

        // Normalize to maintain emotional intensity bounds
        flipped.normalize();

        info!(
            "🔄 Möbius flip complete: sadness {:.3} → joy {:.3}, jitter ±{:.3}",
            sadness_intensity, joy_boost, jitter_strength
        );

        Ok(flipped)
    }

    /// Log visual proof of transformation for debugging
    fn log_visual_proof(&self, before: &EmotionalVector, after: &EmotionalVector, confidence: f32) {
        let sadness_before = (-before.x).max(0.0);
        let joy_after = after.x.max(0.0);
        let joy_delta = joy_after - (-before.x);

        info!("┌─────────────────────────────────────────────┐");
        info!("│ 🌀 MÖBIUS FLIP VISUAL PROOF                │");
        info!("├─────────────────────────────────────────────┤");
        info!(
            "│ Low conf: {:.2}                             │",
            confidence
        );
        info!("│ → Flip applied                              │");
        info!("│ → Joy boost: +{:.2}                        │", joy_delta);
        info!("├─────────────────────────────────────────────┤");
        info!("│ BEFORE:                                     │");
        info!(
            "│   Sadness: {:.3}                           │",
            sadness_before
        );
        info!("│   Fear:    {:.3}                           │", before.y);
        info!("│   Anger:   {:.3}                           │", before.z);
        info!("├─────────────────────────────────────────────┤");
        info!("│ AFTER:                                      │");
        info!("│   Joy:     {:.3}                           │", joy_after);
        info!("│   Calm:    {:.3}                           │", after.y);
        info!("│   Peace:   {:.3}                           │", after.z);
        info!("└─────────────────────────────────────────────┘");
    }

    /// Integration helper: Extract confidence from Qwen response
    ///
    /// This assumes Qwen returns confidence as part of metadata
    /// If not available, use temperature as proxy: low temp → low confidence
    pub fn extract_confidence_from_qwen(temperature: f32) -> f32 {
        // Inverse relationship: low temperature → low confidence
        // Temperature range [0.1, 1.0] → Confidence range [0.3, 0.9]
        let confidence = 0.3 + (temperature - 0.1) * 0.67;
        confidence.clamp(0.0, 1.0)
    }

    /// Update memory spheres with flipped positions for visualization
    ///
    /// Connects to viz_standalone.qml lines 86-91
    pub fn update_memory_sphere_positions(
        &self,
        sphere_count: usize,
        resonance: f32,
        confidence: f32,
    ) -> Vec<(f32, f32, f32)> {
        let mut rng = rand::thread_rng();
        let mut positions = Vec::with_capacity(sphere_count);

        let jitter_strength = (1.0 - confidence) * self.config.novelty_variance;

        for i in 0..sphere_count {
            // Base Möbius torus position (matching viz_standalone.qml line 89-91)
            let angle = (i as f32 / sphere_count as f32) * 2.0 * PI;
            let base_x = angle.sin() * 200.0 * (1.0 + resonance);
            let base_y = angle.cos() * 200.0 * (1.0 + resonance);

            // Apply jitter with novelty variance (line 86)
            let jitter = (rng.r#gen::<f32>() - 0.5) * self.config.novelty_variance;
            let z = jitter * 50.0 * (1.0 + resonance);

            positions.push((base_x, base_y, z));
        }

        info!(
            "🎨 Updated {} memory spheres with jitter ±{:.3}",
            sphere_count, jitter_strength
        );

        positions
    }
}

/// QwenBridge integration wrapper
///
/// Connects Möbius flip processor to Qwen inference pipeline
pub struct QwenMobiusIntegration {
    flip_processor: MobiusFlipProcessor,
}

impl QwenMobiusIntegration {
    pub fn new(config: MobiusFlipConfig) -> Self {
        Self {
            flip_processor: MobiusFlipProcessor::new(config),
        }
    }

    /// Process Qwen output with Möbius flip if confidence is low
    ///
    /// # Arguments
    /// * `emotional_weights` - Raw emotional weights from Qwen [sadness, fear, anger]
    /// * `temperature` - Qwen generation temperature (proxy for confidence)
    ///
    /// # Returns
    /// * Transformed weights with potential flip applied
    pub fn process_qwen_emotional_output(
        &self,
        emotional_weights: Vec<f32>,
        temperature: f32,
    ) -> Result<Vec<f32>> {
        // Extract confidence from temperature
        let confidence = MobiusFlipProcessor::extract_confidence_from_qwen(temperature);

        // Apply Möbius flip on low confidence
        self.flip_processor
            .mobius_flip_on_low_confidence(emotional_weights, confidence)
    }

    /// Get visual flip proof string for UI display
    pub fn get_flip_proof(&self, confidence: f32) -> String {
        if confidence < self.flip_processor.config.confidence_threshold {
            format!(
                "Low conf {:.2} → Flip applied → Joy +{:.2}",
                confidence,
                (1.0 - confidence) * 0.35 // Joy boost estimate
            )
        } else {
            format!("Conf {:.2} OK, no flip needed", confidence)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mobius_flip_on_low_confidence() {
        let config = MobiusFlipConfig::default();
        let processor = MobiusFlipProcessor::new(config);

        // Test case: high sadness, low confidence
        let emotion_vec = vec![-0.8, -0.3, -0.2]; // Sadness, fear, anger
        let confidence = 0.42; // Low confidence

        let result = processor
            .mobius_flip_on_low_confidence(emotion_vec.clone(), confidence)
            .unwrap();

        // After flip, sadness should become joy (positive x)
        assert!(
            result[0] > emotion_vec[0],
            "Expected joy boost after flip, got x: {} vs {}",
            result[0],
            emotion_vec[0]
        );

        tracing::info!(
            "✅ Test passed: Low conf {:.2} → Flip applied → Joy +{:.2}",
            confidence,
            result[0] - emotion_vec[0]
        );
    }

    #[test]
    fn test_no_flip_on_high_confidence() {
        let config = MobiusFlipConfig::default();
        let processor = MobiusFlipProcessor::new(config);

        let emotion_vec = vec![-0.8, -0.3, -0.2];
        let confidence = 0.75; // High confidence

        let result = processor
            .mobius_flip_on_low_confidence(emotion_vec.clone(), confidence)
            .unwrap();

        // High confidence should not trigger flip
        assert_eq!(result, emotion_vec, "Expected no flip for high confidence");
    }

    #[test]
    fn test_confidence_extraction() {
        let temp_low = 0.2;
        let conf_low = MobiusFlipProcessor::extract_confidence_from_qwen(temp_low);
        assert!(
            conf_low < 0.5,
            "Low temperature should yield low confidence"
        );

        let temp_high = 0.8;
        let conf_high = MobiusFlipProcessor::extract_confidence_from_qwen(temp_high);
        assert!(
            conf_high > 0.5,
            "High temperature should yield high confidence"
        );
    }

    #[test]
    fn test_memory_sphere_update() {
        let config = MobiusFlipConfig::default();
        let processor = MobiusFlipProcessor::new(config);

        let positions = processor.update_memory_sphere_positions(8, 0.5, 0.42);

        assert_eq!(positions.len(), 8, "Expected 8 sphere positions");

        // Check that positions have jitter applied
        for (i, (x, y, z)) in positions.iter().enumerate() {
            tracing::info!("Sphere {}: ({:.2}, {:.2}, {:.2})", i, x, y, z);
            assert!(z.abs() > 0.0, "Expected non-zero z jitter");
        }
    }
}
