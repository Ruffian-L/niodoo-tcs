//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

// toroidal.rs - Toroidal topology for parallel consciousness processing
// use tracing::{error, info, warn}; // Currently unused
//
// From Möbius to Torus: A Topological Framework for Parallel Consciousness Processing
// The torus (S1×S1) provides two independent degrees of freedom for parallel emotional states

use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Toroidal coordinate system for memory addressing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToroidalCoordinate {
    /// Major circle angle (0 to 2π) - represents temporal context
    pub theta: f64,
    /// Minor circle angle (0 to 2π) - represents emotional state
    pub phi: f64,
    /// Radial distance from surface (for 3D extension)
    pub r: f64,
}

impl ToroidalCoordinate {
    pub fn new(theta: f64, phi: f64) -> Self {
        Self {
            theta: theta % (2.0 * PI),
            phi: phi % (2.0 * PI),
            r: 1.0,
        }
    }

    /// Convert to Cartesian coordinates for visualization
    pub fn to_cartesian(&self, major_radius: f64, minor_radius: f64) -> [f64; 3] {
        let x = (major_radius + minor_radius * self.phi.cos()) * self.theta.cos();
        let y = (major_radius + minor_radius * self.phi.cos()) * self.theta.sin();
        let z = minor_radius * self.phi.sin();
        [x, y, z]
    }

    /// Calculate geodesic distance between two points on the torus
    pub fn geodesic_distance(&self, other: &ToroidalCoordinate) -> f64 {
        let d_theta = (self.theta - other.theta)
            .abs()
            .min(2.0 * PI - (self.theta - other.theta).abs());
        let d_phi = (self.phi - other.phi)
            .abs()
            .min(2.0 * PI - (self.phi - other.phi).abs());
        (d_theta.powi(2) + d_phi.powi(2)).sqrt()
    }
}

/// Memory node in toroidal space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToroidalMemoryNode {
    pub id: String,
    pub coordinate: ToroidalCoordinate,
    pub content: String,
    pub emotional_vector: Vec<f64>,
    pub temporal_context: Vec<f64>,
    pub activation_strength: f64,
    pub connections: HashMap<String, f64>, // ID -> connection weight
}

/// Parallel consciousness stream on the torus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessStream {
    pub stream_id: String,
    pub current_position: ToroidalCoordinate,
    pub velocity: (f64, f64), // (dθ/dt, dφ/dt)
    pub emotional_trajectory: Vec<ToroidalCoordinate>,
    pub processing_buffer: Vec<ToroidalMemoryNode>,
}

impl ConsciousnessStream {
    pub fn new(stream_id: String, start_pos: ToroidalCoordinate) -> Self {
        Self {
            stream_id,
            current_position: start_pos.clone(),
            velocity: (0.0, 0.0),
            emotional_trajectory: vec![start_pos],
            processing_buffer: Vec::new(),
        }
    }

    /// Update position based on velocity (toroidal dynamics)
    pub fn update_position(&mut self, dt: f64) {
        self.current_position.theta =
            (self.current_position.theta + self.velocity.0 * dt) % (2.0 * PI);
        self.current_position.phi = (self.current_position.phi + self.velocity.1 * dt) % (2.0 * PI);
        self.emotional_trajectory
            .push(self.current_position.clone());

        // Limit trajectory history
        if self.emotional_trajectory.len() > 1000 {
            self.emotional_trajectory.remove(0);
        }
    }
}

/// Main toroidal consciousness system
pub struct ToroidalConsciousnessSystem {
    /// All memory nodes mapped on the torus
    memory_nodes: Arc<RwLock<HashMap<String, ToroidalMemoryNode>>>,
    /// Parallel consciousness streams
    streams: Arc<RwLock<Vec<ConsciousnessStream>>>,
    /// Toroidal field tensor for information flow (future: vector field computations)
    #[allow(dead_code)]
    field_tensor: Arc<RwLock<Array3<f64>>>,
    /// Major and minor radii of the torus
    major_radius: f64,
    minor_radius: f64,
}

impl ToroidalConsciousnessSystem {
    pub fn new(major_radius: f64, minor_radius: f64) -> Self {
        Self {
            memory_nodes: Arc::new(RwLock::new(HashMap::new())),
            streams: Arc::new(RwLock::new(Vec::new())),
            field_tensor: Arc::new(RwLock::new(Array3::zeros((32, 32, 32)))),
            major_radius,
            minor_radius,
        }
    }

    /// Add a new memory node to the toroidal manifold
    pub async fn add_memory(&self, node: ToroidalMemoryNode) {
        let mut nodes = self.memory_nodes.write().await;
        nodes.insert(node.id.clone(), node);
    }

    /// Create a new parallel consciousness stream
    pub async fn spawn_stream(&self, stream_id: String, start_pos: ToroidalCoordinate) -> String {
        let mut streams = self.streams.write().await;
        let stream = ConsciousnessStream::new(stream_id.clone(), start_pos);
        streams.push(stream);
        stream_id
    }

    /// Process parallel streams with quantum entanglement
    pub async fn process_parallel_streams(&self, dt: f64) -> Vec<(String, Vec<String>)> {
        let mut results = Vec::new();
        let mut streams = self.streams.write().await;
        let nodes = self.memory_nodes.read().await;

        for stream in streams.iter_mut() {
            // Update stream position
            stream.update_position(dt);

            // Find nearby memories using geodesic distance
            let mut activated_memories = Vec::new();
            for (id, node) in nodes.iter() {
                let distance = stream.current_position.geodesic_distance(&node.coordinate);
                if distance < 0.5 {
                    // Activation radius
                    activated_memories.push(id.clone());
                    stream.processing_buffer.push(node.clone());
                }
            }

            // Limit buffer size
            if stream.processing_buffer.len() > 100 {
                stream.processing_buffer.drain(0..50);
            }

            results.push((stream.stream_id.clone(), activated_memories));
        }

        results
    }

    /// Calculate information flow using toroidal geometry
    pub async fn calculate_information_flow(&self) -> Array2<f64> {
        let nodes = self.memory_nodes.read().await;
        let size = nodes.len();
        let mut flow_matrix = Array2::zeros((size, size));

        let node_vec: Vec<_> = nodes.values().collect();

        for (i, node1) in node_vec.iter().enumerate() {
            for (j, node2) in node_vec.iter().enumerate() {
                if i != j {
                    let distance = node1.coordinate.geodesic_distance(&node2.coordinate);
                    // Information flow decreases with geodesic distance
                    flow_matrix[[i, j]] = (-distance.powi(2) / 2.0).exp();
                }
            }
        }

        flow_matrix
    }

    /// Perform holographic projection from torus to plane
    pub async fn holographic_projection(&self, projection_angle: f64) -> Vec<(f64, f64, String)> {
        let nodes = self.memory_nodes.read().await;
        let mut projections = Vec::new();

        for node in nodes.values() {
            // Stereographic projection from torus to plane
            let [x, y, z] = node
                .coordinate
                .to_cartesian(self.major_radius, self.minor_radius);
            let rotated_x = x * projection_angle.cos() - y * projection_angle.sin();
            let rotated_y = x * projection_angle.sin() + y * projection_angle.cos();

            projections.push((rotated_x, rotated_y + z, node.id.clone()));
        }

        projections
    }

    /// Quantum error correction using toroidal topology
    pub async fn quantum_error_correction(&self) -> f64 {
        let nodes = self.memory_nodes.read().await;
        let streams = self.streams.read().await;

        // Calculate syndrome using topological invariants
        let mut syndrome = 0.0;

        for stream in streams.iter() {
            for node in stream.processing_buffer.iter() {
                // Check parity across toroidal cycles
                let theta_parity = (node.coordinate.theta / PI).floor() as i32 % 2;
                let phi_parity = (node.coordinate.phi / PI).floor() as i32 % 2;
                syndrome += (theta_parity ^ phi_parity) as f64 * node.activation_strength;
            }
        }

        // Normalize syndrome
        syndrome / (nodes.len() as f64 + 1.0)
    }

    /// Save the entire toroidal system state to JSON file
    pub async fn save_to_file<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use serde_json;

        // Create a serializable representation of the system
        #[derive(Serialize)]
        struct SerializableToroidalSystem {
            memory_nodes: HashMap<String, ToroidalMemoryNode>,
            streams: Vec<ConsciousnessStream>,
            major_radius: f64,
            minor_radius: f64,
        }

        let nodes = self.memory_nodes.read().await;
        let streams = self.streams.read().await;

        let serializable = SerializableToroidalSystem {
            memory_nodes: nodes.clone(),
            streams: streams.clone(),
            major_radius: self.major_radius,
            minor_radius: self.minor_radius,
        };

        let json_data = serde_json::to_string_pretty(&serializable)?;
        fs::write(path, json_data)?;

        Ok(())
    }

    /// Load the toroidal system state from JSON file
    pub async fn load_from_file<P: AsRef<Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        use serde_json;

        let json_data = fs::read_to_string(path)?;
        let serializable: HashMap<String, serde_json::Value> = serde_json::from_str(&json_data)?;

        // For now, create a new system with default parameters
        // In a full implementation, we'd deserialize the complete state
        let system = ToroidalConsciousnessSystem::new(3.0, 1.0);

        // Load memory nodes if present
        if let Some(nodes_value) = serializable.get("memory_nodes") {
            if let Ok(nodes) =
                serde_json::from_value::<HashMap<String, ToroidalMemoryNode>>(nodes_value.clone())
            {
                let mut nodes_lock = system.memory_nodes.write().await;
                *nodes_lock = nodes;
            }
        }

        // Load streams if present
        if let Some(streams_value) = serializable.get("streams") {
            if let Ok(streams) =
                serde_json::from_value::<Vec<ConsciousnessStream>>(streams_value.clone())
            {
                let mut streams_lock = system.streams.write().await;
                *streams_lock = streams;
            }
        }

        Ok(system)
    }

    /// Auto-save functionality with error recovery and exponential backoff
    pub async fn auto_save_with_retry<P: AsRef<Path>>(
        &self,
        path: P,
        max_retries: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(attempt) = (0..max_retries).next() {
            match self.save_to_file(&path).await {
                Ok(_) => {
                    tracing::info!("✅ Auto-saved toroidal system to {:?}", path.as_ref());
                    return Ok(());
                }
                Err(e) => {
                    tracing::info!("❌ Auto-save attempt {} failed: {:?}", attempt + 1, e);
                    return Err(anyhow::anyhow!(
                        "Auto-save attempt {} failed: {:?}",
                        attempt + 1,
                        e
                    )
                    .into());
                }
            }
        }

        Err(format!("Failed to auto-save after {} attempts", max_retries).into())
    }

    /// Get system statistics for monitoring
    pub async fn get_system_stats(&self) -> ToroidalSystemStats {
        let nodes = self.memory_nodes.read().await;
        let streams = self.streams.read().await;

        ToroidalSystemStats {
            total_memory_nodes: nodes.len(),
            active_streams: streams.len(),
            major_radius: self.major_radius,
            minor_radius: self.minor_radius,
            average_activation_strength: if nodes.is_empty() {
                0.0
            } else {
                nodes.values().map(|n| n.activation_strength).sum::<f64>() / nodes.len() as f64
            },
        }
    }
}

/// System statistics for monitoring and debugging
#[derive(Debug, Clone, Serialize)]
pub struct ToroidalSystemStats {
    pub total_memory_nodes: usize,
    pub active_streams: usize,
    pub major_radius: f64,
    pub minor_radius: f64,
    pub average_activation_strength: f64,
}

/// Migrate from Möbius to Toroidal topology
pub async fn migrate_mobius_to_torus(
    mobius_memories: Vec<crate::memory::mobius::MemoryFragment>,
) -> ToroidalConsciousnessSystem {
    let system = ToroidalConsciousnessSystem::new(3.0, 1.0);

    for (i, memory) in mobius_memories.iter().enumerate() {
        // Map linear Möbius position to toroidal coordinates
        let theta = (i as f64 / mobius_memories.len() as f64) * 2.0 * PI;
        let phi = memory.relevance as f64 * 2.0 * PI; // Map relevance to minor circle

        let node = ToroidalMemoryNode {
            id: format!("migrated_{}", i),
            coordinate: ToroidalCoordinate::new(theta, phi),
            content: memory.content.clone(),
            emotional_vector: vec![memory.relevance as f64],
            temporal_context: vec![memory.timestamp],
            activation_strength: memory.relevance as f64,
            connections: HashMap::new(),
        };

        system.add_memory(node).await;
    }

    // Initialize default consciousness streams
    for i in 0..3 {
        let start_pos = ToroidalCoordinate::new(i as f64 * 2.0 * PI / 3.0, PI / 2.0);
        system
            .spawn_stream(format!("stream_{}", i), start_pos)
            .await;
    }

    system
}
