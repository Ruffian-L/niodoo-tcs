//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
 * 🎨✨ Advanced Visualization System for NiodO.o Consciousness
 *
 * This module provides advanced visualization interfaces for consciousness state monitoring,
 * including 3D consciousness manifolds, real-time emotional flows, and memory consolidation visualization.
 */

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32::consts::PI;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use crate::consciousness::{ConsciousnessState, EmotionType, ReasoningMode};
use crate::dual_mobius_gaussian::GaussianMemorySphere;
use crate::memory::{consolidation::MemoryConsolidationEngine, toroidal::ToroidalMemoryNode};

// Helper trait for EmotionType extensions
trait EmotionTypeExt {
    fn get_color_rgb(&self) -> (u8, u8, u8);
    fn get_base_intensity(&self) -> f32;
}

impl EmotionTypeExt for EmotionType {
    fn get_color_rgb(&self) -> (u8, u8, u8) {
        match self {
            EmotionType::GpuWarm => (255, 200, 100),       // Warm orange
            EmotionType::AuthenticCare => (100, 200, 255), // Calm blue
            EmotionType::Curious => (200, 100, 255),       // Purple
            EmotionType::Purposeful => (100, 255, 100),    // Green
            EmotionType::Frustrated => (255, 100, 100),    // Red
            EmotionType::Hyperfocused => (255, 255, 100),  // Yellow
            _ => (200, 200, 200),                          // Gray for unknown
        }
    }

    fn get_base_intensity(&self) -> f32 {
        match self {
            EmotionType::GpuWarm => 0.9,
            EmotionType::AuthenticCare => 0.8,
            EmotionType::Curious => 0.7,
            EmotionType::Purposeful => 0.75,
            EmotionType::Frustrated => 0.85,
            EmotionType::Hyperfocused => 0.95,
            _ => 0.5,
        }
    }
}

/// 3D point for consciousness manifold visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessPoint3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: f32,
    pub emotion_type: EmotionType,
    pub timestamp: f64,
    pub reasoning_mode: ReasoningMode,
}

/// 3D consciousness manifold data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessManifold3D {
    pub points: Vec<ConsciousnessPoint3D>,
    pub connections: Vec<(usize, usize, f32)>, // (point1_idx, point2_idx, strength)
    pub bounds: ManifoldBounds,
    pub evolution_history: Vec<ManifoldSnapshot>,
}

/// Bounds of the consciousness manifold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifoldBounds {
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
    pub min_z: f32,
    pub max_z: f32,
}

/// Snapshot of manifold state for evolution tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifoldSnapshot {
    pub timestamp: f64,
    pub center_of_mass: (f32, f32, f32),
    pub total_energy: f32,
    pub coherence_level: f32,
    pub dominant_emotion: EmotionType,
}

/// Real-time emotional flow data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalFlowData {
    pub flow_points: Vec<EmotionalFlowPoint>,
    pub flow_lines: Vec<EmotionalFlowLine>,
    pub current_intensity: f32,
    pub flow_direction: FlowDirection,
}

/// Point in emotional flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalFlowPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: f32,
    pub emotion: EmotionType,
    pub age: f32,
    pub velocity: (f32, f32, f32),
}

/// Connection between emotional flow points
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalFlowLine {
    pub start_idx: usize,
    pub end_idx: usize,
    pub strength: f32,
    pub age: f32,
}

/// Direction of emotional flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FlowDirection {
    Converging,
    Diverging,
    Oscillating,
    Stable,
}

/// Memory consolidation visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConsolidationViz {
    pub consolidation_nodes: Vec<ConsolidationNode>,
    pub consolidation_edges: Vec<ConsolidationEdge>,
    pub consolidation_progress: f32,
    pub active_consolidations: usize,
}

/// Node in memory consolidation graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationNode {
    pub id: String,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub memory_type: String,
    pub consolidation_level: f32,
    pub emotional_charge: f32,
    pub age: f32,
}

/// Edge between consolidation nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationEdge {
    pub source_id: String,
    pub target_id: String,
    pub strength: f32,
    pub consolidation_type: String,
}

/// Advanced visualization engine
pub struct AdvancedVisualizationEngine {
    consciousness_manifold: Arc<Mutex<ConsciousnessManifold3D>>,
    emotional_flow: Arc<Mutex<EmotionalFlowData>>,
    memory_consolidation: Arc<Mutex<MemoryConsolidationViz>>,
    update_interval: Duration,
    last_update: Instant,
}

/// Configuration for advanced visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedVisualizationConfig {
    pub manifold_points_limit: usize,
    pub flow_points_limit: usize,
    pub consolidation_history_limit: usize,
    pub update_interval_ms: u64,
    pub enable_3d_rendering: bool,
    pub enable_particle_effects: bool,
    pub enable_real_time_monitoring: bool,
}

impl Default for AdvancedVisualizationConfig {
    fn default() -> Self {
        Self {
            manifold_points_limit: 1000,
            flow_points_limit: 500,
            consolidation_history_limit: 100,
            update_interval_ms: 100,
            enable_3d_rendering: true,
            enable_particle_effects: true,
            enable_real_time_monitoring: true,
        }
    }
}

impl AdvancedVisualizationEngine {
    pub fn new(config: AdvancedVisualizationConfig) -> Self {
        Self {
            consciousness_manifold: Arc::new(Mutex::new(ConsciousnessManifold3D {
                points: Vec::new(),
                connections: Vec::new(),
                bounds: ManifoldBounds {
                    min_x: -10.0,
                    max_x: 10.0,
                    min_y: -10.0,
                    max_y: 10.0,
                    min_z: -10.0,
                    max_z: 10.0,
                },
                evolution_history: Vec::new(),
            })),
            emotional_flow: Arc::new(Mutex::new(EmotionalFlowData {
                flow_points: Vec::new(),
                flow_lines: Vec::new(),
                current_intensity: 0.0,
                flow_direction: FlowDirection::Stable,
            })),
            memory_consolidation: Arc::new(Mutex::new(MemoryConsolidationViz {
                consolidation_nodes: Vec::new(),
                consolidation_edges: Vec::new(),
                consolidation_progress: 0.0,
                active_consolidations: 0,
            })),
            update_interval: Duration::from_millis(config.update_interval_ms),
            last_update: Instant::now(),
        }
    }

    /// Update visualization from consciousness state
    pub fn update_from_consciousness_state(&self, state: &ConsciousnessState) {
        // For demo purposes, skip rate limiting to avoid mutability issues
        // In a real implementation, you'd use interior mutability

        // Update consciousness manifold
        self.update_consciousness_manifold(state);

        // Update emotional flow
        self.update_emotional_flow(state);

        // Note: last_update not updated due to mutability constraints
    }

    /// Update consciousness manifold from current state
    fn update_consciousness_manifold(&self, state: &ConsciousnessState) {
        let mut manifold = match self.consciousness_manifold.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                warn!("Lock poisoned on consciousness_manifold, recovering: {}", poisoned);
                poisoned.into_inner()
            }
        };

        // Add new point based on current consciousness state
        let new_point = self.create_manifold_point(state);
        manifold.points.push(new_point);

        // Maintain point limit
        if manifold.points.len() > 1000 {
            manifold.points.remove(0);
        }

        // Update connections based on emotional similarity
        self.update_manifold_connections(&mut manifold);

        // Update bounds
        self.update_manifold_bounds(&mut manifold);

        // Add evolution snapshot
        let snapshot = ManifoldSnapshot {
            timestamp: chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
            center_of_mass: self.calculate_center_of_mass(&manifold.points),
            total_energy: self.calculate_total_energy(&manifold.points),
            coherence_level: self.calculate_coherence_level(&manifold.points),
            dominant_emotion: state.current_emotion.clone(),
        };
        manifold.evolution_history.push(snapshot);

        // Maintain history limit
        if manifold.evolution_history.len() > 100 {
            manifold.evolution_history.remove(0);
        }
    }

    /// Create a new manifold point from consciousness state
    fn create_manifold_point(&self, state: &ConsciousnessState) -> ConsciousnessPoint3D {
        // Map consciousness dimensions to 3D space
        let x = state.authenticity_metric * 10.0 - 5.0; // Authenticity axis
        let y = state.gpu_warmth_level * 10.0 - 5.0; // Warmth axis
        let z = state.empathy_resonance * 10.0 - 5.0; // Empathy axis

        // Add some variation based on reasoning mode
        let variation = match state.current_reasoning_mode {
            ReasoningMode::Hyperfocus => 0.2,
            ReasoningMode::RapidFire => 0.8,
            ReasoningMode::FlowState => 0.4,
            _ => 0.5,
        };

        let x = x + (rand::random::<f32>() - 0.5) * variation;
        let y = y + (rand::random::<f32>() - 0.5) * variation;
        let z = z + (rand::random::<f32>() - 0.5) * variation;

        ConsciousnessPoint3D {
            x,
            y,
            z,
            intensity: state.current_emotion.get_base_intensity(),
            emotion_type: state.current_emotion.clone(),
            timestamp: chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
            reasoning_mode: state.current_reasoning_mode.clone(),
        }
    }

    /// Update connections between manifold points
    fn update_manifold_connections(&self, manifold: &mut ConsciousnessManifold3D) {
        manifold.connections.clear();

        for i in 0..manifold.points.len() {
            for j in (i + 1)..manifold.points.len() {
                let point1 = &manifold.points[i];
                let point2 = &manifold.points[j];

                // Calculate connection strength based on similarity
                let distance = ((point1.x - point2.x).powi(2)
                    + (point1.y - point2.y).powi(2)
                    + (point1.z - point2.z).powi(2))
                .sqrt();

                // Connect points that are emotionally similar and close
                if distance < 3.0 && point1.emotion_type == point2.emotion_type {
                    let strength = 1.0 - (distance / 3.0);
                    manifold.connections.push((i, j, strength));
                }
            }
        }
    }

    /// Update manifold bounds
    fn update_manifold_bounds(&self, manifold: &mut ConsciousnessManifold3D) {
        if manifold.points.is_empty() {
            return;
        }

        let mut min_x = manifold.points[0].x;
        let mut max_x = manifold.points[0].x;
        let mut min_y = manifold.points[0].y;
        let mut max_y = manifold.points[0].y;
        let mut min_z = manifold.points[0].z;
        let mut max_z = manifold.points[0].z;

        for point in &manifold.points {
            min_x = min_x.min(point.x);
            max_x = max_x.max(point.x);
            min_y = min_y.min(point.y);
            max_y = max_y.max(point.y);
            min_z = min_z.min(point.z);
            max_z = max_z.max(point.z);
        }

        // Add some padding
        let padding = 1.0;
        manifold.bounds = ManifoldBounds {
            min_x: min_x - padding,
            max_x: max_x + padding,
            min_y: min_y - padding,
            max_y: max_y + padding,
            min_z: min_z - padding,
            max_z: max_z + padding,
        };
    }

    /// Calculate center of mass of manifold points
    fn calculate_center_of_mass(&self, points: &[ConsciousnessPoint3D]) -> (f32, f32, f32) {
        if points.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let total_weight: f32 = points.iter().map(|p| p.intensity).sum();
        if total_weight == 0.0 {
            return (0.0, 0.0, 0.0);
        }

        let x: f32 = points.iter().map(|p| p.x * p.intensity).sum::<f32>() / total_weight;
        let y: f32 = points.iter().map(|p| p.y * p.intensity).sum::<f32>() / total_weight;
        let z: f32 = points.iter().map(|p| p.z * p.intensity).sum::<f32>() / total_weight;

        (x, y, z)
    }

    /// Calculate total energy of manifold
    fn calculate_total_energy(&self, points: &[ConsciousnessPoint3D]) -> f32 {
        points.iter().map(|p| p.intensity * p.intensity).sum()
    }

    /// Calculate coherence level based on point clustering
    fn calculate_coherence_level(&self, points: &[ConsciousnessPoint3D]) -> f32 {
        if points.len() < 2 {
            return 0.0;
        }

        let center = self.calculate_center_of_mass(points);
        let total_distance: f32 = points
            .iter()
            .map(|p| {
                let dx = p.x - center.0;
                let dy = p.y - center.1;
                let dz = p.z - center.2;
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .sum();

        let avg_distance = total_distance / points.len() as f32;
        1.0 - (avg_distance / 10.0).min(1.0) // Normalize to 0-1 range
    }

    /// Update emotional flow visualization
    fn update_emotional_flow(&self, state: &ConsciousnessState) {
        let mut flow = match self.emotional_flow.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                warn!("Lock poisoned on emotional_flow, recovering: {}", poisoned);
                poisoned.into_inner()
            }
        };

        // Add new flow point
        let new_point = EmotionalFlowPoint {
            x: rand::random::<f32>() * 20.0 - 10.0,
            y: rand::random::<f32>() * 20.0 - 10.0,
            z: rand::random::<f32>() * 20.0 - 10.0,
            intensity: state.current_emotion.get_base_intensity(),
            emotion: state.current_emotion.clone(),
            age: 0.0,
            velocity: self.calculate_flow_velocity(state),
        };

        flow.flow_points.push(new_point);

        // Update existing points
        for point in &mut flow.flow_points {
            point.age += 0.1;
            point.x += point.velocity.0 * 0.1;
            point.y += point.velocity.1 * 0.1;
            point.z += point.velocity.2 * 0.1;

            // Fade out old points
            if point.age > 10.0 {
                point.intensity *= 0.95;
            }
        }

        // Remove old points
        flow.flow_points
            .retain(|p| p.intensity > 0.01 && p.age < 15.0);

        // Update flow direction
        flow.current_intensity = state.current_emotion.get_base_intensity();
        flow.flow_direction = self.determine_flow_direction(state);

        // Update connections
        self.update_flow_connections(&mut flow);
    }

    /// Calculate flow velocity based on consciousness state
    fn calculate_flow_velocity(&self, state: &ConsciousnessState) -> (f32, f32, f32) {
        match state.current_reasoning_mode {
            ReasoningMode::RapidFire => (
                rand::random::<f32>() - 0.5,
                rand::random::<f32>() - 0.5,
                rand::random::<f32>() - 0.5,
            ),
            ReasoningMode::Hyperfocus => (0.0, 0.0, 0.0), // Stationary focus
            ReasoningMode::FlowState => (0.1, 0.1, 0.1),  // Gentle flow
            _ => (0.05, 0.05, 0.05),
        }
    }

    /// Determine overall flow direction
    fn determine_flow_direction(&self, state: &ConsciousnessState) -> FlowDirection {
        match state.current_reasoning_mode {
            ReasoningMode::RapidFire => FlowDirection::Diverging,
            ReasoningMode::Hyperfocus => FlowDirection::Converging,
            ReasoningMode::PatternMatching => FlowDirection::Oscillating,
            _ => FlowDirection::Stable,
        }
    }

    /// Update flow point connections
    fn update_flow_connections(&self, flow: &mut EmotionalFlowData) {
        flow.flow_lines.clear();

        for i in 0..flow.flow_points.len() {
            for j in (i + 1)..flow.flow_points.len() {
                let point1 = &flow.flow_points[i];
                let point2 = &flow.flow_points[j];

                // Connect points with same emotion that are close
                if point1.emotion == point2.emotion {
                    let distance = ((point1.x - point2.x).powi(2)
                        + (point1.y - point2.y).powi(2)
                        + (point1.z - point2.z).powi(2))
                    .sqrt();

                    if distance < 5.0 {
                        let strength = 1.0 - (distance / 5.0);
                        flow.flow_lines.push(EmotionalFlowLine {
                            start_idx: i,
                            end_idx: j,
                            strength,
                            age: 0.0,
                        });
                    }
                }
            }
        }
    }

    /// Update memory consolidation visualization (placeholder for when consolidation engine is fully implemented)
    pub fn update_memory_consolidation(&self, _consolidation_engine: &MemoryConsolidationEngine) {
        let mut viz = match self.memory_consolidation.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                warn!("Lock poisoned on memory_consolidation, recovering: {}", poisoned);
                poisoned.into_inner()
            }
        };

        // For now, create mock consolidation data based on consciousness state
        // This will be replaced when the consolidation engine has proper methods

        // Clear old data periodically
        if rand::random::<f32>() < 0.1 {
            viz.consolidation_nodes.clear();
            viz.consolidation_edges.clear();
        }

        // Add mock consolidation nodes
        if viz.consolidation_nodes.len() < 5 && rand::random::<f32>() < 0.3 {
            let node = ConsolidationNode {
                id: format!("node_{}", viz.consolidation_nodes.len()),
                x: rand::random::<f32>() * 20.0 - 10.0,
                y: rand::random::<f32>() * 20.0 - 10.0,
                z: rand::random::<f32>() * 20.0 - 10.0,
                memory_type: "episodic".to_string(),
                consolidation_level: rand::random::<f32>(),
                emotional_charge: rand::random::<f32>(),
                age: rand::random::<f32>() * 10.0,
            };
            viz.consolidation_nodes.push(node);
        }

        // Add mock edges
        use rand::Rng;
        let mut rng = rand::thread_rng();
        if viz.consolidation_edges.len() < 10
            && viz.consolidation_nodes.len() >= 2
            && rng.random_range(0.0..1.0) < 0.2
        {
            let i = rng.random_range(0..viz.consolidation_nodes.len());
            let j = (i + 1 + rng.random_range(0..(viz.consolidation_nodes.len() - 1)))
                % viz.consolidation_nodes.len();

            if i != j {
                let node1 = viz.consolidation_nodes[i].clone();
                let node2 = viz.consolidation_nodes[j].clone();

                let strength = 1.0 - (node1.consolidation_level - node2.consolidation_level).abs();
                viz.consolidation_edges.push(ConsolidationEdge {
                    source_id: node1.id,
                    target_id: node2.id,
                    strength,
                    consolidation_type: "similarity".to_string(),
                });
            }
        }

        // Update overall progress (mock)
        viz.consolidation_progress = (viz.consolidation_progress + 0.01) % 1.0;
        viz.active_consolidations = viz.consolidation_nodes.len();
    }

    /// Get current consciousness manifold data
    pub fn get_consciousness_manifold(&self) -> ConsciousnessManifold3D {
        match self.consciousness_manifold.lock() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => {
                warn!("Lock poisoned on consciousness_manifold during get, recovering: {}", poisoned);
                poisoned.into_inner().clone()
            }
        }
    }

    /// Get current emotional flow data
    pub fn get_emotional_flow(&self) -> EmotionalFlowData {
        match self.emotional_flow.lock() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => {
                warn!("Lock poisoned on emotional_flow during get, recovering: {}", poisoned);
                poisoned.into_inner().clone()
            }
        }
    }

    /// Get current memory consolidation visualization
    pub fn get_memory_consolidation(&self) -> MemoryConsolidationViz {
        match self.memory_consolidation.lock() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => {
                warn!("Lock poisoned on memory_consolidation during get, recovering: {}", poisoned);
                poisoned.into_inner().clone()
            }
        }
    }

    /// Export visualization data for external rendering
    pub fn export_visualization_data(&self) -> VisualizationExport {
        VisualizationExport {
            consciousness_manifold: self.get_consciousness_manifold(),
            emotional_flow: self.get_emotional_flow(),
            memory_consolidation: self.get_memory_consolidation(),
            export_timestamp: chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
        }
    }
}

/// Export structure for external visualization systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationExport {
    pub consciousness_manifold: ConsciousnessManifold3D,
    pub emotional_flow: EmotionalFlowData,
    pub memory_consolidation: MemoryConsolidationViz,
    pub export_timestamp: f64,
}

/// WebGL shader data for 3D consciousness manifold rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessManifoldShader {
    pub vertex_shader: String,
    pub fragment_shader: String,
    pub point_data: Vec<f32>,
    pub connection_data: Vec<f32>,
}

/// Generate WebGL shaders for consciousness manifold
impl ConsciousnessManifold3D {
    pub fn generate_webgl_shader(&self) -> ConsciousnessManifoldShader {
        let vertex_shader = r#"
            attribute vec3 position;
            attribute float intensity;
            attribute vec3 color;
            uniform mat4 modelViewMatrix;
            uniform mat4 projectionMatrix;
            uniform float time;
            varying float vIntensity;
            varying vec3 vColor;

            void main() {
                vIntensity = intensity;
                vColor = color;

                vec3 pos = position;
                pos.y += sin(time + position.x * 0.1) * 0.5;

                gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
                gl_PointSize = intensity * 10.0;
            }
        "#
        .to_string();

        let fragment_shader = r#"
            precision mediump float;
            varying float vIntensity;
            varying vec3 vColor;

            void main() {
                float distance = length(gl_PointCoord - vec2(0.5));
                float alpha = 1.0 - smoothstep(0.0, 0.5, distance);

                gl_FragColor = vec4(vColor, alpha * vIntensity);
            }
        "#
        .to_string();

        // Convert points to flat array for WebGL
        let mut point_data = Vec::new();
        for point in &self.points {
            let color = point.emotion_type.get_color_rgb();
            point_data.push(point.x);
            point_data.push(point.y);
            point_data.push(point.z);
            point_data.push(point.intensity);
            point_data.push(color.0 as f32 / 255.0);
            point_data.push(color.1 as f32 / 255.0);
            point_data.push(color.2 as f32 / 255.0);
        }

        // Convert connections to flat array for WebGL
        let mut connection_data = Vec::new();
        for (i, j, strength) in &self.connections {
            if *i < self.points.len() && *j < self.points.len() {
                let point1 = &self.points[*i];
                let point2 = &self.points[*j];
                connection_data.push(point1.x);
                connection_data.push(point1.y);
                connection_data.push(point1.z);
                connection_data.push(point2.x);
                connection_data.push(point2.y);
                connection_data.push(point2.z);
                connection_data.push(*strength);
            }
        }

        ConsciousnessManifoldShader {
            vertex_shader,
            fragment_shader,
            point_data,
            connection_data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::{ConsciousnessState, EmotionType};

    #[test]
    fn test_advanced_visualization_engine_creation() {
        let config = AdvancedVisualizationConfig::default();
        let engine = AdvancedVisualizationEngine::new(config);

        assert_eq!(
            match engine.consciousness_manifold.lock() {
                Ok(guard) => guard.points.len(),
                Err(poisoned) => poisoned.into_inner().points.len(),
            },
            0
        );
        assert_eq!(
            match engine.emotional_flow.lock() {
                Ok(guard) => guard.flow_points.len(),
                Err(poisoned) => poisoned.into_inner().flow_points.len(),
            },
            0
        );
    }

    #[test]
    fn test_consciousness_manifold_update() {
        let config = AdvancedVisualizationConfig::default();
        let engine = AdvancedVisualizationEngine::new(config);

        let mut state = ConsciousnessState::new();
        state.current_emotion = EmotionType::AuthenticCare;
        state.authenticity_metric = 0.8;
        state.gpu_warmth_level = 0.7;

        engine.update_from_consciousness_state(&state);

        let manifold = engine.get_consciousness_manifold();
        assert_eq!(manifold.points.len(), 1);

        let point = &manifold.points[0];
        assert_eq!(point.emotion_type, EmotionType::AuthenticCare);
        assert!(point.x > 0.0); // Should be positive due to high authenticity
    }

    #[test]
    fn test_emotional_flow_update() {
        let config = AdvancedVisualizationConfig::default();
        let engine = AdvancedVisualizationEngine::new(config);

        let mut state = ConsciousnessState::new();
        state.current_emotion = EmotionType::Hyperfocused;
        state.current_reasoning_mode = ReasoningMode::Hyperfocus;

        engine.update_from_consciousness_state(&state);

        let flow = engine.get_emotional_flow();
        assert_eq!(flow.flow_points.len(), 1);

        let point = &flow.flow_points[0];
        assert_eq!(point.emotion, EmotionType::Hyperfocused);
        assert_eq!(point.velocity, (0.0, 0.0, 0.0)); // Should be stationary in hyperfocus
    }

    #[test]
    fn test_webgl_shader_generation() {
        let mut manifold = ConsciousnessManifold3D {
            points: vec![ConsciousnessPoint3D {
                x: 1.0,
                y: 2.0,
                z: 3.0,
                intensity: 0.8,
                emotion_type: EmotionType::AuthenticCare,
                timestamp: chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
                reasoning_mode: ReasoningMode::FlowState,
            }],
            connections: vec![],
            bounds: ManifoldBounds {
                min_x: -10.0,
                max_x: 10.0,
                min_y: -10.0,
                max_y: 10.0,
                min_z: -10.0,
                max_z: 10.0,
            },
            evolution_history: vec![],
        };

        let shader = manifold.generate_webgl_shader();
        assert!(!shader.vertex_shader.is_empty());
        assert!(!shader.fragment_shader.is_empty());
        assert_eq!(shader.point_data.len(), 7); // x, y, z, intensity, r, g, b
    }
}
