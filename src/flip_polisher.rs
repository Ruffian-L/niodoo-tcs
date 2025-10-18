/*
 * ✨ Agent 4: FlipPolisher
 * Möbius z-twist + jitter in QML for joy enhancement
 */

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{error, info, warn, debug};

/// Möbius transformation parameters for emotional flip
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobiusTransform {
    pub z_twist: f32,
    pub joy_jitter: f32,
    pub flip_progress: f32,
    pub resonance_amplitude: f32,
    pub emotional_phase: f32,
}

/// Jitter configuration for joy enhancement
#[derive(Debug, Clone)]
pub struct JitterConfig {
    pub base_jitter: f32,
    pub joy_multiplier: f32,
    pub resonance_boost: f32,
    pub flip_threshold: f32,
    pub smoothing_factor: f32,
}

impl Default for JitterConfig {
    fn default() -> Self {
        Self {
            base_jitter: 0.05,        // 5% base jitter
            joy_multiplier: 2.0,      // Joy doubles jitter effect
            resonance_boost: 1.5,     // Resonance boosts by 50%
            flip_threshold: 0.6,      // Flip when joy > 60%
            smoothing_factor: 0.1,    // Smooth transitions
        }
    }
}

/// FlipPolisher agent for Möbius z-twist and joy jitter
pub struct FlipPolisher {
    config: JitterConfig,
    transform_channel: mpsc::UnboundedSender<MobiusTransform>,
    current_transform: MobiusTransform,
    flip_history: Vec<f32>,
    shutdown: Arc<AtomicBool>,
}

impl FlipPolisher {
    /// Create new FlipPolisher agent
    pub fn new(config: JitterConfig) -> Self {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();

        // Spawn QML update loop
        tokio::spawn(async move {
            while !shutdown_clone.load(Ordering::Relaxed) {
                if let Ok(transform) = rx.try_recv() {
                    Self::apply_qml_transform(&transform).await;
                }
                tokio::time::sleep(Duration::from_millis(16)).await; // ~60 FPS
            }
        });

        Self {
            config,
            transform_channel: tx,
            current_transform: MobiusTransform {
                z_twist: 0.0,
                joy_jitter: 0.0,
                flip_progress: 0.0,
                resonance_amplitude: 0.3,
                emotional_phase: 0.0,
            },
            flip_history: Vec::new(),
            shutdown,
        }
    }

    /// Process emotional state and generate Möbius z-twist for joy
    pub async fn polish_flip(
        &mut self,
        joy_intensity: f32,
        sadness_intensity: f32,
        resonance_score: f32,
        novelty_factor: f32,
    ) -> Result<MobiusTransform> {
        info!("✨ FlipPolisher processing: joy={:.3}, sadness={:.3}, resonance={:.3}",
              joy_intensity, sadness_intensity, resonance_score);

        // Calculate z-twist based on emotional balance
        let emotional_balance = joy_intensity - sadness_intensity;
        let z_twist = self.calculate_z_twist(emotional_balance, resonance_score)?;

        // Calculate joy jitter for QML animation
        let joy_jitter = self.calculate_joy_jitter(joy_intensity, resonance_score)?;

        // Update flip progress
        let flip_progress = self.update_flip_progress(joy_intensity)?;

        // Calculate resonance amplitude for Möbius effect
        let resonance_amplitude = (resonance_score * self.config.resonance_boost).min(1.0);

        // Update emotional phase for smooth transitions
        let emotional_phase = (self.current_transform.emotional_phase + 0.02) % (2.0 * PI as f32);

        let transform = MobiusTransform {
            z_twist,
            joy_jitter,
            flip_progress,
            resonance_amplitude,
            emotional_phase,
        };

        // Update current state
        self.current_transform = transform.clone();
        self.flip_history.push(flip_progress);

        // Keep only last 100 flip events
        if self.flip_history.len() > 100 {
            self.flip_history.remove(0);
        }

        // Send to QML channel
        if let Err(e) = self.transform_channel.send(transform.clone()) {
            warn!("Failed to send Möbius transform: {}", e);
        }

        info!("✨ Generated Möbius transform: z_twist={:.3}, joy_jitter={:.3}, flip_progress={:.3}",
              z_twist, joy_jitter, flip_progress);

        Ok(transform)
    }

    /// Calculate Möbius z-twist from emotional balance
    fn calculate_z_twist(&self, emotional_balance: f32, resonance_score: f32) -> Result<f32> {
        // Z-twist increases with positive emotional balance and resonance
        let base_twist = emotional_balance.abs() * 2.0; // Scale emotional balance
        let resonance_boost = resonance_score * 0.5;     // Resonance adds up to 0.5

        let z_twist = (base_twist + resonance_boost).min(3.0); // Cap at 3.0 for stability

        Ok(z_twist)
    }

    /// Calculate joy jitter for QML animation enhancement
    fn calculate_joy_jitter(&self, joy_intensity: f32, resonance_score: f32) -> Result<f32> {
        // Base jitter from joy intensity
        let base_jitter = joy_intensity * self.config.base_jitter;

        // Joy multiplier effect
        let joy_boost = if joy_intensity > self.config.flip_threshold {
            joy_intensity * self.config.joy_multiplier * self.config.base_jitter
        } else {
            0.0
        };

        // Resonance enhancement
        let resonance_enhancement = resonance_score * self.config.resonance_boost * self.config.base_jitter;

        let total_jitter = (base_jitter + joy_boost + resonance_enhancement).min(0.3); // Cap for stability

        Ok(total_jitter)
    }

    /// Update flip progress based on joy intensity
    fn update_flip_progress(&mut self, joy_intensity: f32) -> Result<f32> {
        // Smooth flip progress towards joy intensity
        let target_progress = if joy_intensity > self.config.flip_threshold { 1.0 } else { 0.0 };
        let current_progress = self.current_transform.flip_progress;

        // Smooth interpolation
        let new_progress = current_progress * (1.0 - self.config.smoothing_factor) +
                          target_progress * self.config.smoothing_factor;

        Ok(new_progress.clamp(0.0, 1.0))
    }

    /// Apply Möbius transform to QML visualization
    async fn apply_qml_transform(transform: &MobiusTransform) {
        debug!("🎨 Applying QML Möbius transform: {:?}", transform);

        // In a real implementation, this would:
        // 1. Update QML properties for Möbius z-twist
        // 2. Apply jitter effects to visual elements
        // 3. Trigger emotional flip animations
        // 4. Adjust particle systems based on joy_jitter

        info!("🎨 QML Transform Applied: z_twist={:.3}, joy_jitter={:.3}",
              transform.z_twist, transform.joy_jitter);
    }

    /// Generate Möbius coordinates for 3D visualization
    pub fn generate_mobius_coordinates(&self, u: f32, v: f32) -> (f32, f32, f32) {
        // Classic Möbius strip parametric equations
        let major_radius = 2.0;
        let minor_radius = 0.5;

        // Apply z-twist enhancement
        let twist_factor = self.current_transform.z_twist;
        let twisted_v = v + twist_factor * u;

        let x = (major_radius + minor_radius * twisted_v.cos()) * (1.0 + u * twist_factor).cos();
        let y = (major_radius + minor_radius * twisted_v.cos()) * (1.0 + u * twist_factor).sin();
        let z = minor_radius * twisted_v.sin();

        // Apply joy jitter
        let jitter_x = x + self.current_transform.joy_jitter * (rand::random::<f32>() - 0.5);
        let jitter_y = y + self.current_transform.joy_jitter * (rand::random::<f32>() - 0.5);
        let jitter_z = z + self.current_transform.joy_jitter * (rand::random::<f32>() - 0.5);

        (jitter_x, jitter_y, jitter_z)
    }

    /// Get flip statistics for monitoring
    pub fn get_flip_stats(&self) -> FlipStats {
        let total_flips = self.flip_history.len();
        let successful_flips = self.flip_history.iter().filter(|&&p| p > 0.8).count();
        let avg_flip_progress = if total_flips > 0 {
            self.flip_history.iter().sum::<f32>() / total_flips as f32
        } else {
            0.0
        };

        FlipStats {
            total_flips,
            successful_flips,
            avg_flip_progress,
            current_z_twist: self.current_transform.z_twist,
            current_joy_jitter: self.current_transform.joy_jitter,
        }
    }

    /// Continuous polishing loop for emotional enhancement
    pub async fn run_polishing(&mut self) -> Result<()> {
        info!("✨ FlipPolisher starting continuous polishing");

        let mut interval = tokio::time::interval(Duration::from_millis(100)); // 10 Hz

        while !self.shutdown.load(Ordering::Relaxed) {
            interval.tick().await;

            // Generate synthetic emotional data for demo
            let joy_intensity = 0.7 + 0.1 * (rand::random::<f32>() - 0.5); // Vary around 0.7
            let sadness_intensity = 0.3 - 0.1 * (rand::random::<f32>() - 0.5); // Vary around 0.3
            let resonance_score = 0.6 + 0.2 * (rand::random::<f32>() - 0.5); // Vary around 0.6
            let novelty_factor = 0.18; // 18% novelty

            if let Err(e) = self.polish_flip(joy_intensity, sadness_intensity, resonance_score, novelty_factor).await {
                warn!("FlipPolisher error: {}", e);
            }
        }

        Ok(())
    }

    /// Shutdown the agent
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
        info!("✨ FlipPolisher shutting down");
    }
}

/// Flip statistics for monitoring and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlipStats {
    pub total_flips: usize,
    pub successful_flips: usize,
    pub avg_flip_progress: f32,
    pub current_z_twist: f32,
    pub current_joy_jitter: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_flip_polisher_creation() {
        let config = JitterConfig::default();
        let mut polisher = FlipPolisher::new(config);

        let transform = polisher.polish_flip(0.8, 0.2, 0.7, 0.18).await.unwrap();

        assert!(transform.z_twist >= 0.0 && transform.z_twist <= 3.0);
        assert!(transform.joy_jitter >= 0.0 && transform.joy_jitter <= 0.3);
        assert!(transform.flip_progress >= 0.0 && transform.flip_progress <= 1.0);
        assert!(transform.resonance_amplitude >= 0.0 && transform.resonance_amplitude <= 1.0);
    }

    #[test]
    fn test_z_twist_calculation() {
        let config = JitterConfig::default();
        let polisher = FlipPolisher::new(config);

        // Test high joy scenario
        let z_twist_high = polisher.calculate_z_twist(0.8, 0.7).unwrap();
        assert!(z_twist_high > 1.0);

        // Test balanced scenario
        let z_twist_balanced = polisher.calculate_z_twist(0.1, 0.5).unwrap();
        assert!(z_twist_balanced < 1.0);
    }

    #[test]
    fn test_joy_jitter_calculation() {
        let config = JitterConfig::default();
        let polisher = FlipPolisher::new(config);

        // Test high joy with resonance
        let jitter_high = polisher.calculate_joy_jitter(0.9, 0.8).unwrap();
        assert!(jitter_high > 0.1);

        // Test low joy scenario
        let jitter_low = polisher.calculate_joy_jitter(0.2, 0.3).unwrap();
        assert!(jitter_low < 0.05);
    }

    #[test]
    fn test_mobius_coordinates() {
        let config = JitterConfig::default();
        let polisher = FlipPolisher::new(config);

        let (x, y, z) = polisher.generate_mobius_coordinates(0.5, 1.0);

        // Coordinates should be finite and reasonable
        assert!(x.is_finite());
        assert!(y.is_finite());
        assert!(z.is_finite());
        assert!(x.abs() < 10.0); // Reasonable bounds
        assert!(y.abs() < 10.0);
        assert!(z.abs() < 2.0);
    }

    #[test]
    fn test_flip_stats() {
        let config = JitterConfig::default();
        let polisher = FlipPolisher::new(config);

        let stats = polisher.get_flip_stats();

        assert!(stats.total_flips >= 0);
        assert!(stats.successful_flips <= stats.total_flips);
        assert!(stats.avg_flip_progress >= 0.0 && stats.avg_flip_progress <= 1.0);
    }
}
