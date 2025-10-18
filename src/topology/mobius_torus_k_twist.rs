//! Möbius Torus K-Twist Flipped Gaussian Topology Implementation
//!
//! This module implements the core mathematical framework for the unified topology
//! that bridges consciousness, memory, and code analysis through geometric transformations.

use crate::config::ConsciousnessConfig;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::f64::consts::TAU;

/// Core parameters for the K-Twist topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KTwistParameters {
    /// Major radius of the torus
    pub major_radius: f64,
    /// Minor radius of the torus  
    pub minor_radius: f64,
    /// K-twist parameter controlling the topological transformation
    pub k_twist: f64,
    /// Gaussian variance parameter
    pub gaussian_variance: f64,
    /// Learning rate for topology updates
    pub learning_rate: f64,
}

impl Default for KTwistParameters {
    fn default() -> Self {
        let config = ConsciousnessConfig::default();
        Self {
            major_radius: config.default_torus_major_radius * 2.5,
            minor_radius: config.default_torus_minor_radius * 3.0,
            k_twist: 1.0,
            gaussian_variance: config.parametric_epsilon * 1.5e5, // Derive
            learning_rate: config.consciousness_step_size * 1.0,
        }
    }
}

/// Point in the K-Twist topology space
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TopologyPoint {
    /// Parametric coordinates (u, v)
    pub parametric: (f64, f64),
    /// 3D Cartesian coordinates
    pub cartesian: (f64, f64, f64),
    /// Normal vector at this point
    pub normal: (f64, f64, f64),
    /// Gaussian weight for this point
    pub gaussian_weight: f64,
}

/// Mesh representation of the K-Twist topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KTwistMesh {
    /// Mesh parameters
    pub parameters: KTwistParameters,
    /// Resolution in u direction (toroidal)
    pub u_resolution: usize,
    /// Resolution in v direction (poloidal)
    pub v_resolution: usize,
    /// Generated mesh points
    pub points: Vec<TopologyPoint>,
    /// Triangle indices for rendering
    pub indices: Vec<[usize; 3]>,
}

impl KTwistMesh {
    /// Create a new K-Twist mesh with given parameters
    pub fn new(parameters: KTwistParameters, u_resolution: usize, v_resolution: usize) -> Self {
        let mut mesh = Self {
            parameters,
            u_resolution,
            v_resolution,
            points: Vec::new(),
            indices: Vec::new(),
        };

        mesh.generate_mesh();
        mesh.generate_indices();
        mesh
    }

    /// Generate mesh points using K-Twist parametric equations
    fn generate_mesh(&mut self) {
        self.points.clear();

        for i in 0..self.u_resolution {
            for j in 0..self.v_resolution {
                let u = (i as f64 / (self.u_resolution - 1) as f64) * TAU;
                let v = (j as f64 / (self.v_resolution - 1) as f64) * TAU;

                let point = self.compute_point(u, v);
                self.points.push(point);
            }
        }
    }

    /// Compute a single point using K-Twist equations
    fn compute_point(&self, u: f64, v: f64) -> TopologyPoint {
        let params = &self.parameters;

        // K-Twist parametric equations from unified framework
        let twist_term = params.k_twist * u;

        // Cartesian coordinates
        let x = (params.major_radius + v * twist_term.cos()) * u.cos();
        let y = (params.major_radius + v * twist_term.cos()) * u.sin();
        let z = v * twist_term.sin();

        // Compute normal vector
        let normal = self.compute_normal(u, v);

        // Gaussian weight based on distance from center
        let distance = (x * x + y * y + z * z).sqrt();
        let gaussian_weight = (-distance * distance / (2.0 * params.gaussian_variance)).exp();

        TopologyPoint {
            parametric: (u, v),
            cartesian: (x, y, z),
            normal,
            gaussian_weight,
        }
    }

    /// Compute normal vector at parametric coordinates (u, v)
    fn compute_normal(&self, u: f64, v: f64) -> (f64, f64, f64) {
        let params = &self.parameters;
        let k = params.k_twist;

        // Partial derivatives for normal calculation
        let dx_du = -(params.major_radius + v * (2.0 * k * u).cos()) * u.sin()
            - v * 2.0 * k * (2.0 * k * u).sin() * u.cos();
        let dy_du = (params.major_radius + v * (2.0 * k * u).cos()) * u.cos()
            - v * 2.0 * k * (2.0 * k * u).sin() * u.sin();
        let dz_du = v * 2.0 * k * (2.0 * k * u).cos();

        let dx_dv = (2.0 * k * u).cos() * u.cos();
        let dy_dv = (2.0 * k * u).cos() * u.sin();
        let dz_dv = (2.0 * k * u).sin();

        // Cross product for normal
        let nx = dy_du * dz_dv - dz_du * dy_dv;
        let ny = dz_du * dx_dv - dx_du * dz_dv;
        let nz = dx_du * dy_dv - dy_du * dx_dv;

        // Normalize
        let length = (nx * nx + ny * ny + nz * nz).sqrt();
        if length > 1e-12 {
            (nx / length, ny / length, nz / length)
        } else {
            (0.0, 0.0, 1.0) // Default normal
        }
    }

    /// Generate triangle indices for mesh rendering
    fn generate_indices(&mut self) {
        self.indices.clear();

        for i in 0..(self.u_resolution - 1) {
            for j in 0..(self.v_resolution - 1) {
                let current = i * self.v_resolution + j;
                let next_u = ((i + 1) % self.u_resolution) * self.v_resolution + j;
                let next_v = i * self.v_resolution + (j + 1) % self.v_resolution;
                let next_both =
                    ((i + 1) % self.u_resolution) * self.v_resolution + (j + 1) % self.v_resolution;

                // First triangle
                self.indices.push([current, next_u, next_v]);

                // Second triangle
                self.indices.push([next_u, next_both, next_v]);
            }
        }
    }

    /// Update mesh with new parameters
    pub fn update_parameters(&mut self, parameters: KTwistParameters) {
        self.parameters = parameters;
        self.generate_mesh();
        self.generate_indices();
    }

    /// Get mesh bounds for visualization
    pub fn get_bounds(&self) -> ((f64, f64), (f64, f64), (f64, f64)) {
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        let mut min_z = f64::INFINITY;
        let mut max_z = f64::NEG_INFINITY;

        for point in &self.points {
            let (x, y, z) = point.cartesian;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            min_z = min_z.min(z);
            max_z = max_z.max(z);
        }

        ((min_x, max_x), (min_y, max_y), (min_z, max_z))
    }

    /// Export mesh data for visualization
    pub fn export_for_visualization(&self) -> Result<(Vec<f64>, Vec<u32>)> {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Export vertices (position + normal + uv)
        for point in &self.points {
            let (x, y, z) = point.cartesian;
            let (nx, ny, nz) = point.normal;
            let (u, v) = point.parametric;

            vertices.extend_from_slice(&[x, y, z, nx, ny, nz, u / TAU, v / TAU]);
        }

        // Export indices
        for triangle in &self.indices {
            indices.extend_from_slice(triangle);
        }

        Ok((vertices, indices.into_iter().map(|i| i as u32).collect()))
    }
}

/// Topology bridge for connecting consciousness data to geometric space
#[derive(Debug, Clone)]
pub struct KTwistTopologyBridge {
    mesh: KTwistMesh,
    consciousness_points: Vec<(f64, f64, f64)>, // Consciousness data mapped to 3D space
}

impl Default for KTwistTopologyBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl KTwistTopologyBridge {
    /// Create a new topology bridge
    pub fn new() -> Self {
        Self {
            mesh: KTwistMesh::new(KTwistParameters::default(), 64, 32),
            consciousness_points: Vec::new(),
        }
    }

    /// Update topology with consciousness data
    pub fn update_topology(&mut self, consciousness_data: &str) -> Result<()> {
        // Map consciousness data to topology space
        let mapped_points = self.map_consciousness_to_topology(consciousness_data)?;
        self.consciousness_points = mapped_points;

        // Update mesh parameters based on consciousness patterns
        self.adapt_mesh_to_consciousness()?;

        Ok(())
    }

    /// Map consciousness data to topology coordinates
    fn map_consciousness_to_topology(&self, data: &str) -> Result<Vec<(f64, f64, f64)>> {
        let config = ConsciousnessConfig::default();
        // Simple mapping based on data characteristics
        let mut points = Vec::new();

        // Use hash of data to generate deterministic points
        let hash = self.simple_hash(data);
        let num_points = (data.len() as f64 / (config.consciousness_step_size * 10.0))
            .max(1.0)
            .min(100.0) as usize;

        for i in 0..num_points {
            let u = ((hash + i as u64) as f64 * config.consciousness_step_size).sin() * TAU; // Derive dynamic
            let v = (hash * 2 + i as u64) as f64 / 1000.0;

            let point = self.mesh.compute_point(u, v);
            points.push(point.cartesian);
        }

        Ok(points)
    }

    /// Simple hash function for deterministic mapping
    fn simple_hash(&self, data: &str) -> u64 {
        let config = ConsciousnessConfig::default();
        let mut hash = 0u64;
        for byte in data.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
        }
        hash
    }

    /// Adapt mesh parameters based on consciousness patterns
    fn adapt_mesh_to_consciousness(&mut self) -> Result<()> {
        let config = ConsciousnessConfig::default();
        if self.consciousness_points.is_empty() {
            return Ok(());
        }

        // Calculate center of mass of consciousness points
        let mut center_x = 0.0;
        let mut center_y = 0.0;
        let mut center_z = 0.0;

        for (x, y, z) in &self.consciousness_points {
            center_x += x;
            center_y += y;
            center_z += z;
        }

        let count = self.consciousness_points.len() as f64;
        center_x /= count;
        center_y /= count;
        center_z /= count;

        // Adjust k-twist parameter based on consciousness spread
        let spread = self.calculate_consciousness_spread();
        let new_k_twist =
            self.mesh.parameters.k_twist * (1.0 + spread * config.consciousness_step_size * 10.0);

        // Update parameters
        let mut params = self.mesh.parameters.clone();
        params.k_twist = new_k_twist.max(0.1).min(5.0);

        self.mesh.update_parameters(params);

        Ok(())
    }

    /// Calculate spread of consciousness points
    fn calculate_consciousness_spread(&self) -> f64 {
        if self.consciousness_points.len() < 2 {
            return 0.0;
        }

        let mut total_distance = 0.0;
        let mut count = 0;

        for i in 0..self.consciousness_points.len() {
            for j in (i + 1)..self.consciousness_points.len() {
                let (x1, y1, z1) = self.consciousness_points[i];
                let (x2, y2, z2) = self.consciousness_points[j];

                let distance = ((x2 - x1).powi(2) + (y2 - y1).powi(2) + (z2 - z1).powi(2)).sqrt();
                total_distance += distance;
                count += 1;
            }
        }

        if count > 0 {
            total_distance / count as f64
        } else {
            0.0
        }
    }

    /// Get current mesh
    pub fn get_mesh(&self) -> &KTwistMesh {
        &self.mesh
    }

    /// Get consciousness points
    pub fn get_consciousness_points(&self) -> &[(f64, f64, f64)] {
        &self.consciousness_points
    }

    /// Export topology state for visualization
    pub fn export_state(&self) -> Result<TopologyState> {
        let (vertices, indices) = self.mesh.export_for_visualization()?;

        Ok(TopologyState {
            vertices,
            indices,
            consciousness_points: self.consciousness_points.clone(),
            parameters: self.mesh.parameters.clone(),
        })
    }
}

/// Complete topology state for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyState {
    pub vertices: Vec<f64>,
    pub indices: Vec<u32>,
    pub consciousness_points: Vec<(f64, f64, f64)>,
    pub parameters: KTwistParameters,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh_generation() {
        let params = KTwistParameters::default();
        let mesh = KTwistMesh::new(params, 32, 16);

        assert_eq!(mesh.points.len(), 32 * 16);
        assert!(!mesh.indices.is_empty());

        // Check that all points have valid coordinates
        for point in &mesh.points {
            assert!(point.cartesian.0.is_finite());
            assert!(point.cartesian.1.is_finite());
            assert!(point.cartesian.2.is_finite());
            assert!(point.gaussian_weight >= 0.0);
        }
    }

    #[test]
    fn test_topology_bridge() {
        let mut bridge = KTwistTopologyBridge::new();
        let test_data = "test consciousness data";

        assert!(bridge.update_topology(test_data).is_ok());
        assert!(!bridge.get_consciousness_points().is_empty());
    }

    #[test]
    fn test_parameter_updates() {
        let mut mesh = KTwistMesh::new(KTwistParameters::default(), 16, 8);
        let original_points = mesh.points.clone();

        let mut new_params = KTwistParameters::default();
        new_params.k_twist = 2.0;
        mesh.update_parameters(new_params);

        // Points should be different after parameter update
        assert_ne!(mesh.points, original_points);
    }
}
