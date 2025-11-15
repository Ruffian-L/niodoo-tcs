//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Rust-Qt Bridge for K-Twist Topology Visualization
//!
//! This module provides the bridge between Rust topology computations
//! and Qt visualization components.

use crate::gaussian_process::KTwistGaussianProcess;
use crate::mobius_memory::{MemoryLayer, MemorySystemState, MobiusMemorySystem};
use crate::topology::mobius_torus_k_twist::{
    KTwistParameters, KTwistTopologyBridge, TopologyState,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete system state for Qt visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationState {
    pub topology_state: TopologyState,
    pub memory_state: MemorySystemState,
    pub gaussian_points: Vec<GaussianSphere>,
    pub consciousness_streams: Vec<ConsciousnessStream>,
    pub current_emotion: EmotionalState,
    pub system_metrics: SystemMetrics,
}

/// Gaussian sphere for uncertainty visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianSphere {
    pub id: String,
    pub position: (f64, f64, f64),
    pub radius: f64,
    pub color: (f64, f64, f64, f64), // RGBA
    pub uncertainty: f64,
    pub confidence: f64,
}

/// Consciousness stream for real-time data flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessStream {
    pub id: String,
    pub current_position: (f64, f64, f64),
    pub velocity: (f64, f64, f64),
    pub intensity: f64,
    pub color: (f64, f64, f64, f64),
    pub data_type: String,
}

/// Current emotional state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalState {
    pub arousal: f64,   // Red component
    pub valence: f64,   // Green component
    pub dominance: f64, // Blue component
    pub intensity: f64, // Overall intensity
    pub timestamp: u64,
}

/// System performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub memory_stability: f64,
    pub topology_coherence: f64,
    pub gaussian_novelty: f64,
    pub processing_fps: f64,
    pub system_load: f64,
    pub timestamp: u64,
}

/// Main bridge between Rust backend and Qt frontend
#[derive(Debug)]
pub struct KTwistQtBridge {
    topology_bridge: KTwistTopologyBridge,
    memory_system: MobiusMemorySystem,
    gaussian_process: KTwistGaussianProcess,
    visualization_state: VisualizationState,
    update_counter: u64,
}

impl Default for KTwistQtBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl KTwistQtBridge {
    /// Create a new Qt bridge
    pub fn new() -> Self {
        let topology_bridge = KTwistTopologyBridge::new();
        let memory_system = MobiusMemorySystem::new(Default::default());
        let gaussian_process = KTwistGaussianProcess::new(Default::default());

        let visualization_state = VisualizationState {
            topology_state: TopologyState {
                vertices: Vec::new(),
                indices: Vec::new(),
                consciousness_points: Vec::new(),
                parameters: KTwistParameters::default(),
            },
            memory_state: MemorySystemState {
                layers: HashMap::new(),
                params: Default::default(),
                total_memories: 0,
                stability_metrics: Default::default(),
            },
            gaussian_points: Vec::new(),
            consciousness_streams: Vec::new(),
            current_emotion: EmotionalState {
                arousal: 0.5,
                valence: 0.5,
                dominance: 0.5,
                intensity: 0.0,
                timestamp: 0,
            },
            system_metrics: SystemMetrics {
                memory_stability: 0.0,
                topology_coherence: 0.0,
                gaussian_novelty: 0.0,
                processing_fps: 0.0,
                system_load: 0.0,
                timestamp: 0,
            },
        };

        Self {
            topology_bridge,
            memory_system,
            gaussian_process,
            visualization_state,
            update_counter: 0,
        }
    }

    /// Update the entire system with new consciousness data
    pub fn update_system(&mut self, consciousness_data: &str) -> Result<()> {
        self.update_counter += 1;

        // Update topology
        self.topology_bridge.update_topology(consciousness_data)?;

        // Add memory entry
        let _memory_id = self.memory_system.add_memory(
            consciousness_data.to_string(),
            MemoryLayer::Working,
            0.5,
        )?;

        // Apply emotional transformation
        self.memory_system.apply_emotional_transformation()?;

        // Update Gaussian process
        self.update_gaussian_process()?;

        // Generate visualization state
        self.generate_visualization_state()?;

        Ok(())
    }

    /// Update Gaussian process with current system state
    fn update_gaussian_process(&mut self) -> Result<()> {
        // Get consciousness points from topology
        let consciousness_points = self.topology_bridge.get_consciousness_points().to_vec();

        // Update Gaussian process topology
        self.gaussian_process.update_topology(consciousness_points);

        Ok(())
    }

    /// Generate complete visualization state
    fn generate_visualization_state(&mut self) -> Result<()> {
        // Update topology state
        self.visualization_state.topology_state = self.topology_bridge.export_state()?;

        // Update memory state
        self.visualization_state.memory_state = self.memory_system.export_state()?;

        // Generate Gaussian spheres
        self.visualization_state.gaussian_points = self.generate_gaussian_spheres();

        // Generate consciousness streams
        self.visualization_state.consciousness_streams = self.generate_consciousness_streams();

        // Update current emotion
        self.visualization_state.current_emotion = self.calculate_current_emotion();

        // Update system metrics
        self.visualization_state.system_metrics = self.calculate_system_metrics();

        Ok(())
    }

    /// Generate Gaussian spheres for uncertainty visualization
    fn generate_gaussian_spheres(&self) -> Vec<GaussianSphere> {
        let mut spheres = Vec::new();

        // Get consciousness points
        let consciousness_points = self.topology_bridge.get_consciousness_points();

        for (i, &(x, y, z)) in consciousness_points.iter().enumerate() {
            // Calculate uncertainty based on Gaussian process
            let uncertainty = self.calculate_point_uncertainty(x, y, z);
            let confidence = 1.0 - uncertainty;

            // Calculate sphere radius based on uncertainty
            let radius = uncertainty * 2.0 + 0.1; // Minimum radius of 0.1

            // Calculate color based on uncertainty and position
            let color = self.calculate_sphere_color(uncertainty, x, y, z);

            spheres.push(GaussianSphere {
                id: format!("sphere_{}", i),
                position: (x, y, z),
                radius,
                color,
                uncertainty,
                confidence,
            });
        }

        spheres
    }

    /// Calculate uncertainty for a specific point
    fn calculate_point_uncertainty(&self, x: f64, y: f64, z: f64) -> f64 {
        // Simple uncertainty calculation based on distance from center
        let distance_from_center = (x * x + y * y + z * z).sqrt();
        let max_distance = 10.0; // Assume max distance of 10

        // Uncertainty increases with distance from center
        let distance_uncertainty = (distance_from_center / max_distance).min(1.0);

        // Add some noise based on update counter for dynamic behavior
        let time_factor = (self.update_counter as f64 * 0.01).sin() * 0.1;

        (distance_uncertainty + time_factor).max(0.0).min(1.0)
    }

    /// Calculate sphere color based on uncertainty and position
    fn calculate_sphere_color(
        &self,
        uncertainty: f64,
        x: f64,
        y: f64,
        z: f64,
    ) -> (f64, f64, f64, f64) {
        // Base color based on position
        let r = ((x + 5.0) / 10.0).max(0.0).min(1.0);
        let g = ((y + 5.0) / 10.0).max(0.0).min(1.0);
        let b = ((z + 5.0) / 10.0).max(0.0).min(1.0);

        // Adjust alpha based on uncertainty (higher uncertainty = more transparent)
        let alpha = 0.3 + uncertainty * 0.7;

        (r, g, b, alpha)
    }

    /// Generate consciousness streams for real-time data flow
    fn generate_consciousness_streams(&self) -> Vec<ConsciousnessStream> {
        let mut streams = Vec::new();

        // Get memory entries from different layers
        for layer in [
            MemoryLayer::CoreBurned,
            MemoryLayer::Procedural,
            MemoryLayer::Episodic,
            MemoryLayer::Semantic,
            MemoryLayer::Somatic,
            MemoryLayer::Working,
        ] {
            let memories = self.memory_system.get_layer_memories(&layer);

            if !memories.is_empty() {
                // Create stream for this layer
                let memory = &memories[0]; // Use first memory as representative
                let (x, y, z) = memory.topology_position;

                // Calculate velocity based on emotional vector
                let (er, eg, eb) = memory.emotional_vector;
                let velocity = (er * 0.1, eg * 0.1, eb * 0.1);

                // Calculate intensity based on stability and emotional weight
                let intensity = memory.stability * memory.emotional_weight;

                // Calculate color based on emotional vector
                let color = (er, eg, eb, 0.8);

                streams.push(ConsciousnessStream {
                    id: format!("stream_{:?}", layer),
                    current_position: (x, y, z),
                    velocity,
                    intensity,
                    color,
                    data_type: format!("{:?}", layer),
                });
            }
        }

        streams
    }

    /// Calculate current emotional state from memory system
    fn calculate_current_emotion(&self) -> EmotionalState {
        let mut total_arousal = 0.0;
        let mut total_valence = 0.0;
        let mut total_dominance = 0.0;
        let mut total_weight = 0.0;

        // Aggregate emotional vectors from all memories
        for layer in [
            MemoryLayer::CoreBurned,
            MemoryLayer::Procedural,
            MemoryLayer::Episodic,
            MemoryLayer::Semantic,
            MemoryLayer::Somatic,
            MemoryLayer::Working,
        ] {
            let memories = self.memory_system.get_layer_memories(&layer);

            for memory in memories {
                let (r, g, b) = memory.emotional_vector;
                let weight = memory.emotional_weight;

                total_arousal += r * weight;
                total_valence += g * weight;
                total_dominance += b * weight;
                total_weight += weight;
            }
        }

        // Normalize by total weight
        let arousal: f64 = if total_weight > 0.0 {
            total_arousal / total_weight
        } else {
            0.5
        };
        let valence: f64 = if total_weight > 0.0 {
            total_valence / total_weight
        } else {
            0.5
        };
        let dominance: f64 = if total_weight > 0.0 {
            total_dominance / total_weight
        } else {
            0.5
        };

        // Calculate overall intensity
        let intensity =
            ((arousal - 0.5).powi(2) + (valence - 0.5).powi(2) + (dominance - 0.5).powi(2)).sqrt();

        EmotionalState {
            arousal: arousal.max(0.0).min(1.0),
            valence: valence.max(0.0).min(1.0),
            dominance: dominance.max(0.0).min(1.0),
            intensity: intensity.max(0.0).min(1.0),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    /// Calculate system performance metrics
    fn calculate_system_metrics(&self) -> SystemMetrics {
        let memory_metrics = self.memory_system.get_stability_metrics();

        // Calculate topology coherence
        let topology_coherence = self.calculate_topology_coherence();

        // Calculate Gaussian novelty
        let gaussian_novelty = self.calculate_gaussian_novelty();

        // Estimate processing FPS (simplified)
        let processing_fps = 60.0; // Assume 60 FPS for now

        // Estimate system load (simplified)
        let system_load = (memory_metrics.overall_stability * 0.3
            + topology_coherence * 0.3
            + gaussian_novelty * 0.4)
            .min(1.0);

        SystemMetrics {
            memory_stability: memory_metrics.overall_stability,
            topology_coherence,
            gaussian_novelty,
            processing_fps,
            system_load,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    /// Calculate topology coherence
    fn calculate_topology_coherence(&self) -> f64 {
        let consciousness_points = self.topology_bridge.get_consciousness_points();

        if consciousness_points.len() < 2 {
            return 1.0;
        }

        // Calculate average distance between points
        let mut total_distance = 0.0;
        let mut count = 0;

        for i in 0..consciousness_points.len() {
            for j in (i + 1)..consciousness_points.len() {
                let (x1, y1, z1) = consciousness_points[i];
                let (x2, y2, z2) = consciousness_points[j];

                let distance = ((x2 - x1).powi(2) + (y2 - y1).powi(2) + (z2 - z1).powi(2)).sqrt();
                total_distance += distance;
                count += 1;
            }
        }

        if count > 0 {
            let avg_distance = total_distance / count as f64;
            // Coherence is inverse of average distance (normalized)
            (-avg_distance / 10.0).exp()
        } else {
            1.0
        }
    }

    /// Calculate Gaussian novelty
    fn calculate_gaussian_novelty(&self) -> f64 {
        let consciousness_points = self.topology_bridge.get_consciousness_points();

        if consciousness_points.is_empty() {
            return 0.0;
        }

        // Calculate spread of consciousness points
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        let mut min_z = f64::INFINITY;
        let mut max_z = f64::NEG_INFINITY;

        for &(x, y, z) in consciousness_points {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            min_z = min_z.min(z);
            max_z = max_z.max(z);
        }

        let spread_x = max_x - min_x;
        let spread_y = max_y - min_y;
        let spread_z = max_z - min_z;

        let total_spread = spread_x + spread_y + spread_z;

        // Normalize novelty to 0-1 range
        (total_spread / 30.0).min(1.0)
    }

    /// Get current visualization state
    pub fn get_visualization_state(&self) -> &VisualizationState {
        &self.visualization_state
    }

    /// Export visualization state as JSON
    pub fn export_json(&self) -> Result<String> {
        let json = serde_json::to_string_pretty(&self.visualization_state)?;
        Ok(json)
    }

    /// Update topology parameters
    pub fn update_topology_parameters(&mut self, params: KTwistParameters) -> Result<()> {
        // Update topology bridge parameters
        self.topology_bridge = KTwistTopologyBridge::new();
        // Note: In a real implementation, we'd need a way to update parameters

        Ok(())
    }

    /// Get system statistics
    pub fn get_system_stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();

        let memory_metrics = self.memory_system.get_stability_metrics();
        stats.insert(
            "memory_stability".to_string(),
            memory_metrics.overall_stability,
        );
        stats.insert(
            "emotional_coherence".to_string(),
            memory_metrics.emotional_coherence,
        );
        stats.insert(
            "consolidation_rate".to_string(),
            memory_metrics.consolidation_rate,
        );

        let system_metrics = &self.visualization_state.system_metrics;
        stats.insert(
            "topology_coherence".to_string(),
            system_metrics.topology_coherence,
        );
        stats.insert(
            "gaussian_novelty".to_string(),
            system_metrics.gaussian_novelty,
        );
        stats.insert("processing_fps".to_string(), system_metrics.processing_fps);
        stats.insert("system_load".to_string(), system_metrics.system_load);

        stats.insert(
            "total_memories".to_string(),
            self.memory_system.get_total_memories() as f64,
        );
        stats.insert(
            "consciousness_points".to_string(),
            self.topology_bridge.get_consciousness_points().len() as f64,
        );

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let bridge = KTwistQtBridge::new();
        assert_eq!(bridge.update_counter, 0);
    }

    #[test]
    fn test_system_update() {
        let mut bridge = KTwistQtBridge::new();
        let test_data = "test consciousness data";

        assert!(bridge.update_system(test_data).is_ok());
        assert_eq!(bridge.update_counter, 1);
    }

    #[test]
    fn test_json_export() {
        let bridge = KTwistQtBridge::new();
        let json = bridge.export_json().unwrap();
        assert!(!json.is_empty());
        assert!(json.contains("topology_state"));
        assert!(json.contains("memory_state"));
    }

    #[test]
    fn test_system_stats() {
        let bridge = KTwistQtBridge::new();
        let stats = bridge.get_system_stats();

        assert!(stats.contains_key("memory_stability"));
        assert!(stats.contains_key("topology_coherence"));
        assert!(stats.contains_key("total_memories"));
    }
}
