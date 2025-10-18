//! Qt Data Bridge
use tracing::{info, error, warn};
//! 
//! This module provides a JSON-based data bridge between Rust torus implementations
//! and Qt visualization system. It handles real-time data serialization and file-based communication.

use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use anyhow::Result;
use tokio::fs as async_fs;
use tokio::time::sleep;

use crate::qt_torus_bridge::{QtTorusBridge, ConsciousnessData, PerformanceMetrics};

/// JSON data bridge for Qt communication
#[derive(Debug)]
pub struct QtDataBridge {
    /// Path to consciousness data JSON file
    consciousness_data_path: String,
    /// Path to mesh data JSON file
    mesh_data_path: String,
    /// Path to performance metrics JSON file
    performance_metrics_path: String,
    /// Update interval for data streaming
    update_interval: Duration,
    /// Last update time
    last_update: Instant,
    /// Streaming active flag
    streaming_active: bool,
}

/// Mesh data structure for JSON serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshData {
    pub vertices: Vec<f32>,
    pub indices: Vec<u32>,
    pub normals: Vec<f32>,
    pub uv_coords: Vec<f32>,
    pub resolution: (usize, usize),
    pub k_twist: f64,
    pub major_radius: f64,
    pub minor_radius: f64,
    pub timestamp: u64,
}

/// Real-time update data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeUpdate {
    pub consciousness_data: ConsciousnessData,
    pub mesh_data: Option<MeshData>,
    pub performance_metrics: PerformanceMetrics,
    pub update_type: String,
    pub timestamp: u64,
}

impl QtDataBridge {
    /// Create a new Qt data bridge
    pub fn new(data_directory: &str) -> Self {
        let consciousness_data_path = format!("{}/consciousness_data.json", data_directory);
        let mesh_data_path = format!("{}/mesh_data.json", data_directory);
        let performance_metrics_path = format!("{}/performance_metrics.json", data_directory);

        // Ensure directory exists
        if let Err(e) = fs::create_dir_all(data_directory) {
            tracing::info!("Warning: Could not create data directory {}: {}", data_directory, e);
        }

        Self {
            consciousness_data_path,
            mesh_data_path,
            performance_metrics_path,
            update_interval: Duration::from_millis(16), // ~60 FPS
            last_update: Instant::now(),
            streaming_active: false,
        }
    }

    /// Write consciousness data to JSON file
    pub async fn write_consciousness_data(&self, data: &ConsciousnessData) -> Result<()> {
        let json_content = serde_json::to_string_pretty(data)?;
        async_fs::write(&self.consciousness_data_path, json_content).await?;
        Ok(())
    }

    /// Write mesh data to JSON file
    pub async fn write_mesh_data(&self, data: &MeshData) -> Result<()> {
        let json_content = serde_json::to_string_pretty(data)?;
        async_fs::write(&self.mesh_data_path, json_content).await?;
        Ok(())
    }

    /// Write performance metrics to JSON file
    pub async fn write_performance_metrics(&self, metrics: &PerformanceMetrics) -> Result<()> {
        let json_content = serde_json::to_string_pretty(metrics)?;
        async_fs::write(&self.performance_metrics_path, json_content).await?;
        Ok(())
    }

    /// Write complete real-time update
    pub async fn write_realtime_update(&self, update: &RealtimeUpdate) -> Result<()> {
        // Write individual components
        self.write_consciousness_data(&update.consciousness_data).await?;
        if let Some(ref mesh_data) = update.mesh_data {
            self.write_mesh_data(mesh_data).await?;
        }
        self.write_performance_metrics(&update.performance_metrics).await?;

        // Write combined update
        let combined_path = format!("{}/realtime_update.json", 
            Path::new(&self.consciousness_data_path).parent().unwrap().to_str().unwrap());
        let json_content = serde_json::to_string_pretty(update)?;
        async_fs::write(combined_path, json_content).await?;

        Ok(())
    }

    /// Read consciousness data from JSON file
    pub async fn read_consciousness_data(&self) -> Result<Option<ConsciousnessData>> {
        if !Path::new(&self.consciousness_data_path).exists() {
            return Ok(None);
        }

        let content = async_fs::read_to_string(&self.consciousness_data_path).await?;
        let data: ConsciousnessData = serde_json::from_str(&content)?;
        Ok(Some(data))
    }

    /// Read mesh data from JSON file
    pub async fn read_mesh_data(&self) -> Result<Option<MeshData>> {
        if !Path::new(&self.mesh_data_path).exists() {
            return Ok(None);
        }

        let content = async_fs::read_to_string(&self.mesh_data_path).await?;
        let data: MeshData = serde_json::from_str(&content)?;
        Ok(Some(data))
    }

    /// Read performance metrics from JSON file
    pub async fn read_performance_metrics(&self) -> Result<Option<PerformanceMetrics>> {
        if !Path::new(&self.performance_metrics_path).exists() {
            return Ok(None);
        }

        let content = async_fs::read_to_string(&self.performance_metrics_path).await?;
        let metrics: PerformanceMetrics = serde_json::from_str(&content)?;
        Ok(Some(metrics))
    }

    /// Start real-time streaming to Qt
    pub async fn start_streaming(&mut self, torus_bridge: &QtTorusBridge) -> Result<()> {
        if self.streaming_active {
            return Ok(());
        }

        self.streaming_active = true;
        self.last_update = Instant::now();

        // Start streaming task
        let bridge_clone = torus_bridge.clone();
        let data_bridge_clone = self.clone();
        
        tokio::spawn(async move {
            data_bridge_clone.streaming_loop(bridge_clone).await;
        });

        Ok(())
    }

    /// Stop real-time streaming
    pub fn stop_streaming(&mut self) {
        self.streaming_active = false;
    }

    /// Check if streaming is active
    pub fn is_streaming(&self) -> bool {
        self.streaming_active
    }

    /// Set update interval
    pub fn set_update_interval(&mut self, interval: Duration) {
        self.update_interval = interval;
    }

    /// Get update interval
    pub fn get_update_interval(&self) -> Duration {
        self.update_interval
    }

    /// Main streaming loop
    async fn streaming_loop(&self, torus_bridge: QtTorusBridge) {
        let mut frame_count = 0u64;
        let mut last_mesh_update = Instant::now();

        while self.streaming_active {
            let start_time = Instant::now();

            // Get consciousness data
            let consciousness_data = match torus_bridge.get_consciousness_data().await {
                Ok(data) => data,
                Err(e) => {
                    info!("Error getting consciousness data: {}", e);
                    continue;
                }
            };

            // Get performance metrics
            let performance_metrics = torus_bridge.get_performance_metrics().await;

            // Update mesh data periodically (every 100ms to avoid excessive updates)
            let mesh_data = if last_mesh_update.elapsed() > Duration::from_millis(100) {
                match torus_bridge.generate_mesh_data(None).await {
                    Ok(mesh_cache) => {
                        last_mesh_update = Instant::now();
                        Some(MeshData {
                            vertices: mesh_cache.vertices,
                            indices: mesh_cache.indices,
                            normals: mesh_cache.normals,
                            uv_coords: mesh_cache.uv_coords,
                            resolution: mesh_cache.resolution,
                            k_twist: mesh_cache.k_twist,
                            major_radius: 100.0, // TODO: Get from torus bridge
                            minor_radius: 20.0,  // TODO: Get from torus bridge
                            timestamp: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs(),
                        })
                    }
                    Err(e) => {
                        info!("Error generating mesh data: {}", e);
                        None
                    }
                }
            } else {
                None
            };

            // Create real-time update
            let update = RealtimeUpdate {
                consciousness_data,
                mesh_data,
                performance_metrics,
                update_type: "frame_update".to_string(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            };

            // Write update to files
            if let Err(e) = self.write_realtime_update(&update).await {
                info!("Error writing real-time update: {}", e);
            }

            frame_count += 1;
            if frame_count % 100 == 0 {
                info!("📡 Qt Data Bridge: {} frames streamed", frame_count);
            }

            // Calculate sleep time to maintain target FPS
            let elapsed = start_time.elapsed();
            if elapsed < self.update_interval {
                sleep(self.update_interval - elapsed).await;
            }
        }

        info!("📡 Qt Data Bridge: Streaming stopped after {} frames", frame_count);
    }

    /// Generate test data for Qt visualization
    pub async fn generate_test_data(&self) -> Result<()> {
        // Generate test consciousness data
        let test_consciousness_data = ConsciousnessData {
            memory_nodes: vec![
                crate::qt_torus_bridge::MemoryNodeData {
                    id: "test_node_1".to_string(),
                    coordinate: crate::memory::toroidal::ToroidalCoordinate::new(0.0, 0.0),
                    content: "Test memory content".to_string(),
                    activation_strength: 0.8,
                    color: [1.0, 0.0, 0.0],
                    size: 8.0,
                },
                crate::qt_torus_bridge::MemoryNodeData {
                    id: "test_node_2".to_string(),
                    coordinate: crate::memory::toroidal::ToroidalCoordinate::new(3.14159, 1.5708),
                    content: "Another test memory".to_string(),
                    activation_strength: 0.6,
                    color: [0.0, 1.0, 0.0],
                    size: 6.0,
                },
            ],
            consciousness_streams: vec![
                crate::qt_torus_bridge::ConsciousnessStreamData {
                    stream_id: "test_stream_1".to_string(),
                    current_position: crate::memory::toroidal::ToroidalCoordinate::new(1.0, 0.5),
                    velocity: [0.1, 0.05],
                    processing_buffer: vec![],
                    color: [0.0, 0.0, 1.0],
                    intensity: 0.7,
                },
            ],
            current_emotion: crate::qt_torus_bridge::EmotionData {
                emotion_type: "Curiosity".to_string(),
                intensity: 0.7,
                valence: 0.3,
                arousal: 0.5,
                color: [0.7, 0.3, 0.5],
            },
            consciousness_activity: 0.75,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        // Generate test mesh data
        let test_mesh_data = MeshData {
            vertices: vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], // Single vertex
            indices: vec![0], // Single index
            normals: vec![0.0, 0.0, 1.0],
            uv_coords: vec![0.0, 0.0],
            resolution: (32, 16),
            k_twist: 1.0,
            major_radius: 100.0,
            minor_radius: 20.0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        // Generate test performance metrics
        let test_performance_metrics = PerformanceMetrics {
            mesh_generation_time: Duration::from_millis(5),
            vertex_count: 512,
            triangle_count: 930,
            memory_usage_mb: 0.5,
            streaming_enabled: true,
            target_fps: 60,
            adaptive_resolution: true,
        };

        // Write test data
        self.write_consciousness_data(&test_consciousness_data).await?;
        self.write_mesh_data(&test_mesh_data).await?;
        self.write_performance_metrics(&test_performance_metrics).await?;

        info!("✅ Test data generated for Qt visualization");
        Ok(())
    }

    /// Validate JSON files
    pub async fn validate_json_files(&self) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();

        // Validate consciousness data
        match self.read_consciousness_data().await {
            Ok(Some(data)) => {
                report.consciousness_data_valid = true;
                report.consciousness_data_timestamp = data.timestamp;
                report.memory_nodes_count = data.memory_nodes.len();
                report.streams_count = data.consciousness_streams.len();
            }
            Ok(None) => {
                report.consciousness_data_valid = false;
                report.errors.push("Consciousness data file not found".to_string());
            }
            Err(e) => {
                report.consciousness_data_valid = false;
                report.errors.push(format!("Consciousness data validation error: {}", e));
            }
        }

        // Validate mesh data
        match self.read_mesh_data().await {
            Ok(Some(data)) => {
                report.mesh_data_valid = true;
                report.mesh_data_timestamp = data.timestamp;
                report.vertex_count = data.vertices.len() / 8; // 8 components per vertex
                report.triangle_count = data.indices.len() / 3;
            }
            Ok(None) => {
                report.mesh_data_valid = false;
                report.errors.push("Mesh data file not found".to_string());
            }
            Err(e) => {
                report.mesh_data_valid = false;
                report.errors.push(format!("Mesh data validation error: {}", e));
            }
        }

        // Validate performance metrics
        match self.read_performance_metrics().await {
            Ok(Some(metrics)) => {
                report.performance_metrics_valid = true;
                report.memory_usage_mb = metrics.memory_usage_mb;
                report.target_fps = metrics.target_fps;
            }
            Ok(None) => {
                report.performance_metrics_valid = false;
                report.errors.push("Performance metrics file not found".to_string());
            }
            Err(e) => {
                report.performance_metrics_valid = false;
                report.errors.push(format!("Performance metrics validation error: {}", e));
            }
        }

        Ok(report)
    }
}

impl Clone for QtDataBridge {
    fn clone(&self) -> Self {
        Self {
            consciousness_data_path: self.consciousness_data_path.clone(),
            mesh_data_path: self.mesh_data_path.clone(),
            performance_metrics_path: self.performance_metrics_path.clone(),
            update_interval: self.update_interval,
            last_update: self.last_update,
            streaming_active: self.streaming_active,
        }
    }
}

/// Validation report for JSON files
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub consciousness_data_valid: bool,
    pub mesh_data_valid: bool,
    pub performance_metrics_valid: bool,
    pub consciousness_data_timestamp: u64,
    pub mesh_data_timestamp: u64,
    pub memory_nodes_count: usize,
    pub streams_count: usize,
    pub vertex_count: usize,
    pub triangle_count: usize,
    pub memory_usage_mb: f64,
    pub target_fps: u32,
    pub errors: Vec<String>,
}

impl ValidationReport {
    fn new() -> Self {
        Self {
            consciousness_data_valid: false,
            mesh_data_valid: false,
            performance_metrics_valid: false,
            consciousness_data_timestamp: 0,
            mesh_data_timestamp: 0,
            memory_nodes_count: 0,
            streams_count: 0,
            vertex_count: 0,
            triangle_count: 0,
            memory_usage_mb: 0.0,
            target_fps: 0,
            errors: Vec::new(),
        }
    }

    /// Check if all data is valid
    pub fn is_valid(&self) -> bool {
        self.consciousness_data_valid && self.mesh_data_valid && self.performance_metrics_valid
    }

    /// Print validation report
    pub fn print_report(&self) {
        info!("📊 Qt Data Bridge Validation Report");
        info!("==================================");
        info!("Consciousness Data: {}", if self.consciousness_data_valid { "✅ Valid" } else { "❌ Invalid" });
        info!("Mesh Data: {}", if self.mesh_data_valid { "✅ Valid" } else { "❌ Invalid" });
        info!("Performance Metrics: {}", if self.performance_metrics_valid { "✅ Valid" } else { "❌ Invalid" });
        
        if self.consciousness_data_valid {
            info!("  Memory Nodes: {}", self.memory_nodes_count);
            info!("  Streams: {}", self.streams_count);
            info!("  Timestamp: {}", self.consciousness_data_timestamp);
        }
        
        if self.mesh_data_valid {
            info!("  Vertices: {}", self.vertex_count);
            info!("  Triangles: {}", self.triangle_count);
            info!("  Timestamp: {}", self.mesh_data_timestamp);
        }
        
        if self.performance_metrics_valid {
            info!("  Memory Usage: {:.2} MB", self.memory_usage_mb);
            info!("  Target FPS: {}", self.target_fps);
        }
        
        if !self.errors.is_empty() {
            info!("\n❌ Errors:");
            for error in &self.errors {
                info!("  - {}", error);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_qt_data_bridge_creation() {
        let temp_dir = tempdir().unwrap();
        let bridge = QtDataBridge::new(temp_dir.path().to_str().unwrap());
        
        assert!(!bridge.is_streaming());
        assert_eq!(bridge.get_update_interval(), Duration::from_millis(16));
    }

    #[tokio::test]
    async fn test_test_data_generation() {
        let temp_dir = tempdir().unwrap();
        let bridge = QtDataBridge::new(temp_dir.path().to_str().unwrap());
        
        let result = bridge.generate_test_data().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_json_validation() {
        let temp_dir = tempdir().unwrap();
        let bridge = QtDataBridge::new(temp_dir.path().to_str().unwrap());
        
        // Generate test data first
        bridge.generate_test_data().await.unwrap();
        
        // Validate
        let report = bridge.validate_json_files().await.unwrap();
        assert!(report.is_valid());
        assert!(report.memory_nodes_count > 0);
        assert!(report.streams_count > 0);
    }

    #[tokio::test]
    async fn test_streaming_control() {
        let temp_dir = tempdir().unwrap();
        let mut bridge = QtDataBridge::new(temp_dir.path().to_str().unwrap());
        
        assert!(!bridge.is_streaming());
        
        bridge.set_update_interval(Duration::from_millis(100));
        assert_eq!(bridge.get_update_interval(), Duration::from_millis(100));
        
        bridge.stop_streaming();
        assert!(!bridge.is_streaming());
    }
}

#[no_mangle]
pub extern "C" fn stop_streaming(bridge: &mut QtDataBridge) {
    bridge.stop_streaming();
}
#[no_mangle]
pub extern "C" fn set_update_interval(bridge: &mut QtDataBridge, interval_ms: u64) {
    bridge.set_update_interval(Duration::from_millis(interval_ms));
}
#[no_mangle]
pub extern "C" fn get_update_interval(bridge: &QtDataBridge) -> u64 {
    bridge.get_update_interval().as_millis() as u64
}
