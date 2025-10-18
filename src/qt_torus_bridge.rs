//! Qt Torus Bridge
//! 
//! This module provides a bridge between Rust torus implementations and Qt visualization system.
//! It handles real-time data streaming, mesh generation, and consciousness state updates.

use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use anyhow::Result;

use crate::k_twisted_torus::KTwistedTorusGenerator;
use crate::memory::toroidal::{ToroidalConsciousnessSystem, ToroidalCoordinate};
use crate::torus_enhancements::EnhancedTorus;
use crate::torus_performance::PerformanceTorus;

/// Qt bridge for torus visualization and consciousness data streaming
#[derive(Debug)]
pub struct QtTorusBridge {
    /// Enhanced torus for mathematical operations
    enhanced_torus: Arc<EnhancedTorus>,
    /// Performance torus for real-time mesh generation
    performance_torus: Arc<PerformanceTorus>,
    /// Consciousness system for real-time data
    consciousness_system: Arc<ToroidalConsciousnessSystem>,
    /// Current mesh data cache
    mesh_cache: Arc<Mutex<MeshCache>>,
    /// Streaming configuration
    streaming_config: Arc<RwLock<StreamingConfig>>,
    /// Last update time
    last_update: Arc<Mutex<Instant>>,
}

/// Cached mesh data for Qt rendering
#[derive(Debug, Clone)]
pub struct MeshCache {
    pub vertices: Vec<f32>,
    pub indices: Vec<u32>,
    pub normals: Vec<f32>,
    pub uv_coords: Vec<f32>,
    pub last_generation: Instant,
    pub resolution: (usize, usize),
    pub k_twist: f64,
}

/// Streaming configuration for real-time updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    pub enabled: bool,
    pub update_interval_ms: u64,
    pub max_fps: u32,
    pub adaptive_resolution: bool,
    pub target_error: f64,
    pub consciousness_streaming: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            update_interval_ms: 16, // ~60 FPS
            max_fps: 60,
            adaptive_resolution: true,
            target_error: 0.01,
            consciousness_streaming: true,
        }
    }
}

/// Consciousness data structure for Qt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessData {
    pub memory_nodes: Vec<MemoryNodeData>,
    pub consciousness_streams: Vec<ConsciousnessStreamData>,
    pub current_emotion: EmotionData,
    pub consciousness_activity: f64,
    pub timestamp: u64,
}

/// Memory node data for Qt visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNodeData {
    pub id: String,
    pub coordinate: ToroidalCoordinate,
    pub content: String,
    pub activation_strength: f64,
    pub color: [f32; 3],
    pub size: f32,
}

/// Consciousness stream data for Qt visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessStreamData {
    pub stream_id: String,
    pub current_position: ToroidalCoordinate,
    pub velocity: [f64; 2],
    pub processing_buffer: Vec<MemoryNodeData>,
    pub color: [f32; 3],
    pub intensity: f32,
}

/// Emotion data for Qt visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionData {
    pub emotion_type: String,
    pub intensity: f64,
    pub valence: f64,
    pub arousal: f64,
    pub color: [f32; 3],
}

impl QtTorusBridge {
    /// Create a new Qt torus bridge
    pub fn new(
        major_radius: f64,
        minor_radius: f64,
        k_twist: f64,
        resolution: (usize, usize),
    ) -> Result<Self> {
        let enhanced_torus = Arc::new(EnhancedTorus::new(
            major_radius,
            minor_radius,
            k_twist,
            resolution,
        ));

        let performance_torus = Arc::new(PerformanceTorus::new(
            major_radius,
            minor_radius,
            k_twist,
            resolution,
        ));

        let consciousness_system = Arc::new(ToroidalConsciousnessSystem::new(
            major_radius,
            minor_radius,
            100, // max_nodes
        ));

        let mesh_cache = Arc::new(Mutex::new(MeshCache {
            vertices: Vec::new(),
            indices: Vec::new(),
            normals: Vec::new(),
            uv_coords: Vec::new(),
            last_generation: Instant::now(),
            resolution,
            k_twist,
        }));

        let streaming_config = Arc::new(RwLock::new(StreamingConfig::default()));

        Ok(Self {
            enhanced_torus,
            performance_torus,
            consciousness_system,
            mesh_cache,
            streaming_config,
            last_update: Arc::new(Mutex::new(Instant::now())),
        })
    }

    /// Generate mesh data for Qt rendering
    pub async fn generate_mesh_data(&self, resolution: Option<(usize, usize)>) -> Result<MeshCache> {
        let target_resolution = resolution.unwrap_or(self.enhanced_torus.resolution);

        // Check if we need to regenerate mesh
        let mut cache = self.mesh_cache.lock()
            .map_err(|e| anyhow::anyhow!("Failed to acquire mesh cache lock: {}", e))?;
        let needs_regeneration = cache.resolution != target_resolution ||
            cache.k_twist != self.enhanced_torus.k_twist ||
            cache.last_generation.elapsed() > Duration::from_millis(100);

        if needs_regeneration {
            // Generate new mesh using performance-optimized torus
            let (vertices, indices) = self.performance_torus.generate_lod_mesh(0)?;
            
            // Convert f64 to f32 for Qt compatibility
            let vertices_f32: Vec<f32> = vertices.iter().map(|&v| v as f32).collect();
            let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
            
            // Extract normals and UV coordinates
            let mut normals = Vec::new();
            let mut uv_coords = Vec::new();
            
            for chunk in vertices_f32.chunks(8) {
                if chunk.len() >= 8 {
                    // Position: chunk[0..3], Normal: chunk[3..6], UV: chunk[6..8]
                    normals.extend_from_slice(&chunk[3..6]);
                    uv_coords.extend_from_slice(&chunk[6..8]);
                }
            }
            
            *cache = MeshCache {
                vertices: vertices_f32,
                indices: indices_u32,
                normals,
                uv_coords,
                last_generation: Instant::now(),
                resolution: target_resolution,
                k_twist: self.enhanced_torus.k_twist,
            };
        }

        Ok(cache.clone())
    }

    /// Get current consciousness data for Qt visualization
    pub async fn get_consciousness_data(&self) -> Result<ConsciousnessData> {
        let nodes = self.consciousness_system.memory_nodes.read().await;
        let streams = self.consciousness_system.streams.read().await;

        // Convert memory nodes to Qt format
        let memory_nodes: Vec<MemoryNodeData> = nodes
            .values()
            .map(|node| MemoryNodeData {
                id: node.id.clone(),
                coordinate: node.coordinate.clone(),
                content: node.content.clone(),
                activation_strength: node.activation_strength,
                color: self.calculate_node_color(node.activation_strength),
                size: (node.activation_strength * 10.0) as f32,
            })
            .collect();

        // Convert consciousness streams to Qt format
        let consciousness_streams: Vec<ConsciousnessStreamData> = streams
            .iter()
            .map(|stream| ConsciousnessStreamData {
                stream_id: stream.stream_id.clone(),
                current_position: stream.current_position.clone(),
                velocity: stream.velocity,
                processing_buffer: stream.processing_buffer
                    .iter()
                    .map(|node| MemoryNodeData {
                        id: node.id.clone(),
                        coordinate: node.coordinate.clone(),
                        content: node.content.clone(),
                        activation_strength: node.activation_strength,
                        color: self.calculate_node_color(node.activation_strength),
                        size: (node.activation_strength * 10.0) as f32,
                    })
                    .collect(),
                color: self.calculate_stream_color(&stream.stream_id),
                intensity: 0.8, // Default intensity
            })
            .collect();

        // Get current emotion (placeholder for now)
        let current_emotion = EmotionData {
            emotion_type: "Neutral".to_string(),
            intensity: 0.5,
            valence: 0.0,
            arousal: 0.0,
            color: [0.5, 0.5, 0.5],
        };

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Ok(ConsciousnessData {
            memory_nodes,
            consciousness_streams,
            current_emotion,
            consciousness_activity: 0.7, // Placeholder
            timestamp,
        })
    }

    /// Update torus parameters
    pub async fn update_torus_parameters(
        &self,
        major_radius: Option<f64>,
        minor_radius: Option<f64>,
        k_twist: Option<f64>,
        resolution: Option<(usize, usize)>,
    ) -> Result<()> {
        if let Some(radius) = major_radius {
            // Update enhanced torus
            let new_enhanced = EnhancedTorus::new(
                radius,
                minor_radius.unwrap_or(self.enhanced_torus.minor_radius),
                k_twist.unwrap_or(self.enhanced_torus.k_twist),
                resolution.unwrap_or(self.enhanced_torus.resolution),
            );
            // Note: In a real implementation, you'd need to replace the Arc contents
        }

        if let Some(radius) = minor_radius {
            // Update minor radius
        }

        if let Some(k) = k_twist {
            // Update k-twist parameter
        }

        if let Some(res) = resolution {
            // Update resolution
        }

        // Invalidate mesh cache
        let mut cache = self.mesh_cache.lock()
            .map_err(|e| anyhow::anyhow!("Failed to acquire mesh cache lock: {}", e))?;
        cache.last_generation = Instant::now() - Duration::from_secs(1);

        Ok(())
    }

    /// Start real-time streaming to Qt
    pub async fn start_streaming(&self) -> Result<()> {
        let mut config = self.streaming_config.write().await;
        config.enabled = true;
        config.consciousness_streaming = true;
        Ok(())
    }

    /// Stop real-time streaming
    pub async fn stop_streaming(&self) -> Result<()> {
        let mut config = self.streaming_config.write().await;
        config.enabled = false;
        config.consciousness_streaming = false;
        Ok(())
    }

    /// Get streaming status
    pub async fn is_streaming(&self) -> bool {
        let config = self.streaming_config.read().await;
        config.enabled && config.consciousness_streaming
    }

    /// Calculate node color based on activation strength
    fn calculate_node_color(&self, activation: f64) -> [f32; 3] {
        // Color mapping: low activation = blue, high activation = red
        let intensity = activation.clamp(0.0, 1.0) as f32;
        [intensity, 0.2, 1.0 - intensity]
    }

    /// Calculate stream color based on stream ID
    fn calculate_stream_color(&self, stream_id: &str) -> [f32; 3] {
        // Simple hash-based color generation
        let hash = stream_id.chars().fold(0u32, |acc, c| acc.wrapping_add(c as u32));
        let r = ((hash & 0xFF) as f32) / 255.0;
        let g = (((hash >> 8) & 0xFF) as f32) / 255.0;
        let b = (((hash >> 16) & 0xFF) as f32) / 255.0;
        [r, g, b]
    }

    /// Export mesh to OBJ format for Qt
    pub async fn export_mesh_to_obj(&self) -> Result<String> {
        let mesh_cache = self.generate_mesh_data(None).await?;
        
        let mut obj_content = String::new();
        obj_content.push_str("# K-Twisted Torus Mesh for Qt Visualization\n");
        obj_content.push_str("# Generated by Rust QtTorusBridge\n\n");

        // Write vertices
        for chunk in mesh_cache.vertices.chunks(8) {
            if chunk.len() >= 3 {
                obj_content.push_str(&format!("v {} {} {}\n", chunk[0], chunk[1], chunk[2]));
            }
        }

        // Write normals
        for chunk in mesh_cache.normals.chunks(3) {
            if chunk.len() >= 3 {
                obj_content.push_str(&format!("vn {} {} {}\n", chunk[0], chunk[1], chunk[2]));
            }
        }

        // Write UV coordinates
        for chunk in mesh_cache.uv_coords.chunks(2) {
            if chunk.len() >= 2 {
                obj_content.push_str(&format!("vt {} {}\n", chunk[0], chunk[1]));
            }
        }

        // Write faces
        for chunk in mesh_cache.indices.chunks(3) {
            if chunk.len() >= 3 {
                // OBJ indices are 1-based
                let v1 = chunk[0] + 1;
                let v2 = chunk[1] + 1;
                let v3 = chunk[2] + 1;
                obj_content.push_str(&format!("f {} {} {}\n", v1, v2, v3));
            }
        }

        Ok(obj_content)
    }

    /// Get performance metrics for Qt monitoring
    pub async fn get_performance_metrics(&self) -> Result<PerformanceMetrics> {
        let mesh_cache = self.mesh_cache.lock()
            .map_err(|e| anyhow::anyhow!("Failed to acquire mesh cache lock: {}", e))?;
        let config = self.streaming_config.read().await;

        Ok(PerformanceMetrics {
            mesh_generation_time: mesh_cache.last_generation.elapsed(),
            vertex_count: mesh_cache.vertices.len() / 8,
            triangle_count: mesh_cache.indices.len() / 3,
            memory_usage_mb: (mesh_cache.vertices.len() * 4 + mesh_cache.indices.len() * 4) as f64 / (1024.0 * 1024.0),
            streaming_enabled: config.enabled,
            target_fps: config.max_fps,
            adaptive_resolution: config.adaptive_resolution,
        })
    }
}

/// Performance metrics for Qt monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub mesh_generation_time: Duration,
    pub vertex_count: usize,
    pub triangle_count: usize,
    pub memory_usage_mb: f64,
    pub streaming_enabled: bool,
    pub target_fps: u32,
    pub adaptive_resolution: bool,
}

/// Qt bridge manager for multiple torus instances
#[derive(Debug)]
pub struct QtTorusBridgeManager {
    bridges: HashMap<String, Arc<QtTorusBridge>>,
    active_bridge: Option<String>,
}

impl QtTorusBridgeManager {
    /// Create a new bridge manager
    pub fn new() -> Self {
        Self {
            bridges: HashMap::new(),
            active_bridge: None,
        }
    }

    /// Add a new torus bridge
    pub async fn add_bridge(
        &mut self,
        name: String,
        major_radius: f64,
        minor_radius: f64,
        k_twist: f64,
        resolution: (usize, usize),
    ) -> Result<()> {
        let bridge = QtTorusBridge::new(major_radius, minor_radius, k_twist, resolution)?;
        self.bridges.insert(name.clone(), Arc::new(bridge));
        
        if self.active_bridge.is_none() {
            self.active_bridge = Some(name);
        }
        
        Ok(())
    }

    /// Get the active bridge
    pub fn get_active_bridge(&self) -> Option<&Arc<QtTorusBridge>> {
        self.active_bridge.as_ref().and_then(|name| self.bridges.get(name))
    }

    /// Set the active bridge
    pub fn set_active_bridge(&mut self, name: String) -> Result<()> {
        if self.bridges.contains_key(&name) {
            self.active_bridge = Some(name);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Bridge '{}' not found", name))
        }
    }

    /// Get all bridge names
    pub fn get_bridge_names(&self) -> Vec<String> {
        self.bridges.keys().cloned().collect()
    }

    /// Remove a bridge
    pub fn remove_bridge(&mut self, name: &str) -> Option<Arc<QtTorusBridge>> {
        if self.active_bridge.as_ref() == Some(&name.to_string()) {
            self.active_bridge = None;
        }
        self.bridges.remove(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_qt_torus_bridge_creation() {
        let bridge = QtTorusBridge::new(100.0, 20.0, 1.0, (64, 32));
        assert!(bridge.is_ok());
    }

    #[tokio::test]
    async fn test_mesh_generation() {
        let bridge = QtTorusBridge::new(100.0, 20.0, 1.0, (32, 16))
            .expect("Failed to create bridge");
        let mesh_cache = bridge.generate_mesh_data(None).await
            .expect("Failed to generate mesh data");
        
        assert!(!mesh_cache.vertices.is_empty());
        assert!(!mesh_cache.indices.is_empty());
        assert_eq!(mesh_cache.resolution, (32, 16));
    }

    #[tokio::test]
    async fn test_consciousness_data() {
        let bridge = QtTorusBridge::new(100.0, 20.0, 1.0, (32, 16))
            .expect("Failed to create bridge");
        let consciousness_data = bridge.get_consciousness_data().await
            .expect("Failed to get consciousness data");
        
        assert!(consciousness_data.timestamp > 0);
        assert_eq!(consciousness_data.current_emotion.emotion_type, "Neutral");
    }

    #[tokio::test]
    async fn test_export_to_obj() {
        let bridge = QtTorusBridge::new(100.0, 20.0, 1.0, (16, 8))
            .expect("Failed to create bridge");
        let obj_content = bridge.export_mesh_to_obj().await
            .expect("Failed to export to OBJ");
        
        assert!(obj_content.contains("# K-Twisted Torus Mesh"));
        assert!(obj_content.contains("v "));
        assert!(obj_content.contains("f "));
    }

    #[tokio::test]
    async fn test_bridge_manager() {
        let mut manager = QtTorusBridgeManager::new();

        manager.add_bridge("test1".to_string(), 100.0, 20.0, 1.0, (32, 16)).await
            .expect("Failed to add bridge");
        manager.add_bridge("test2".to_string(), 200.0, 40.0, 2.0, (64, 32)).await
            .expect("Failed to add bridge");

        assert_eq!(manager.get_bridge_names().len(), 2);
        assert!(manager.get_active_bridge().is_some());

        manager.set_active_bridge("test2".to_string())
            .expect("Failed to set active bridge");
        assert_eq!(manager.active_bridge, Some("test2".to_string()));
    }
}


