//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

// K-Twisted Toroidal Surface Generator
// Based on mathematical analysis for MOBIUS TORUS GAUSSIAN TOPOLOGY
// Implements proper parametric equations for k-twisted surfaces

use anyhow::Result;
use std::f32::consts::PI;
use std::io::Write;
use crate::config::ConsciousnessConfig;

/// K-Twisted Toroidal Surface Generator
///
/// Generates a toroidal surface with k half-twists, combining:
/// - Circular path of a torus
/// - Characteristic twist of a Möbius strip
///
/// Mathematical foundation from analysis document:
/// x(u,v) = (R + v*cos(2ku)) * cos(u)
/// y(u,v) = (R + v*cos(2ku)) * sin(u)  
/// z(u,v) = v * sin(2ku)
pub struct KTwistedTorusGenerator {
    /// Major radius (distance from center to tube center)
    pub major_radius: f32,
    /// Strip half-width (replaces minor radius)
    pub strip_width: f32,
    /// Number of half-twists (k parameter)
    /// k=1: One half-twist (non-orientable like Möbius strip)
    /// k=2: Two half-twists (orientable)
    /// Odd k: Non-orientable, Even k: Orientable
    pub twists: i32,
    /// Toroidal resolution (u parameter steps)
    pub u_steps: usize,
    /// Poloidal resolution (v parameter steps)
    pub v_steps: usize,
}

/// Vertex data structure for GPU consumption
#[derive(Debug, Clone)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

impl KTwistedTorusGenerator {
    pub fn new(
        major_radius: f32,
        strip_width: f32,
        twists: i32,
        u_steps: usize,
        v_steps: usize,
    ) -> Self {
        Self {
            major_radius,
            strip_width,
            twists,
            u_steps,
            v_steps,
        }
    }

    /// Generate complete mesh data with proper normals
    /// Returns (vertices, indices) for GPU consumption
    pub fn generate_mesh_data(&self) -> Result<(Vec<f32>, Vec<u32>)> {
        let total_vertices = self.u_steps * self.v_steps;
        let total_triangles = (self.u_steps - 1) * (self.v_steps - 1) * 2;
        let mut vertices = Vec::with_capacity(total_vertices * 6); // 6 floats per vertex (pos + normal)
        let mut indices = Vec::with_capacity(total_triangles * 3); // 3 indices per triangle

        // Generate vertices with interleaved data [x,y,z,nx,ny,nz]
        for i in 0..self.u_steps {
            for j in 0..self.v_steps {
                let u = (i as f32 / self.u_steps as f32) * 2.0 * PI;
                let v_norm = (j as f32 / (self.v_steps - 1) as f32) - 0.5; // -0.5 to 0.5
                let v = v_norm * self.strip_width; // Scale by strip width

                // Calculate position using correct parametric equations
                let position = self.calculate_position(u, v);
                let normal = self.calculate_normal(u, v)?;

                // Interleaved vertex data: position + normal
                vertices.extend_from_slice(&position);
                vertices.extend_from_slice(&normal);
            }
        }

        // Generate indices for triangle strips
        for i in 0..(self.u_steps - 1) {
            for j in 0..(self.v_steps - 1) {
                let i0 = (i * self.v_steps + j) as u32;
                let i1 = ((i + 1) * self.v_steps + j) as u32;
                let i2 = (i * self.v_steps + j + 1) as u32;
                let i3 = ((i + 1) * self.v_steps + j + 1) as u32;

                // Two triangles per quad
                indices.extend_from_slice(&[i0, i1, i2]);
                indices.extend_from_slice(&[i1, i3, i2]);
            }
        }

        Ok((vertices, indices))
    }

    /// Calculate 3D position using k-twisted parametric equations
    pub fn calculate_position(&self, u: f32, v: f32) -> [f32; 3] {
        let k = self.twists as f32;
        let r = self.major_radius;

        // Core mathematical formula from analysis:
        // x(u,v) = (R + v*cos(2ku)) * cos(u)
        // y(u,v) = (R + v*cos(2ku)) * sin(u)
        // z(u,v) = v * sin(2ku)

        let twist_factor = 2.0 * k * u;
        let radius_at_u = r + v * twist_factor.cos();

        let x = radius_at_u * u.cos();
        let y = radius_at_u * u.sin();
        let z = v * twist_factor.sin();

        [x, y, z]
    }

    /// Calculate surface normal vector using partial derivatives
    /// Critical for proper lighting and shading
    fn calculate_normal(&self, u: f32, v: f32) -> Result<[f32; 3]> {
        let k = self.twists as f32;
        let r = self.major_radius;

        // Partial derivatives from analysis document
        // ∂P/∂u calculation
        let twist_factor = 2.0 * k * u;
        let cos_u = u.cos();
        let sin_u = u.sin();
        let cos_twist = twist_factor.cos();
        let sin_twist = twist_factor.sin();

        let du_x = -k * v * sin_twist * cos_u - (r + v * cos_twist) * sin_u;
        let du_y = -k * v * sin_twist * sin_u + (r + v * cos_twist) * cos_u;
        let du_z = k * v * cos_twist;

        // ∂P/∂v calculation
        let dv_x = cos_twist * cos_u;
        let dv_y = cos_twist * sin_u;
        let dv_z = sin_twist;

        // Cross product: N = ∂P/∂u × ∂P/∂v
        let normal_x = du_y * dv_z - du_z * dv_y;
        let normal_y = du_z * dv_x - du_x * dv_z;
        let normal_z = du_x * dv_y - du_y * dv_x;

        // Normalize the normal vector
        let length = (normal_x * normal_x + normal_y * normal_y + normal_z * normal_z).sqrt();

        if length < 1e-6 {
            // Fallback for degenerate cases
            return Ok([0.0, 0.0, 1.0]);
        }

        Ok([normal_x / length, normal_y / length, normal_z / length])
    }

    /// Export mesh to OBJ file for Blender validation
    /// Critical debugging step from analysis document
    pub fn export_to_obj<W: Write>(&self, writer: &mut W) -> Result<()> {
        let (vertices, indices) = self.generate_mesh_data()?;

        writeln!(writer, "# K-Twisted Toroidal Surface")?;
        writeln!(
            writer,
            "# R={}, w={}, k={}",
            self.major_radius, self.strip_width, self.twists
        )?;

        // Write vertices (positions only for OBJ)
        for chunk in vertices.chunks(6) {
            // 6 floats per vertex (pos + normal)
            writeln!(writer, "v {} {} {}", chunk[0], chunk[1], chunk[2])?;
        }

        // Write vertex normals
        for chunk in vertices.chunks(6) {
            writeln!(writer, "vn {} {} {}", chunk[3], chunk[4], chunk[5])?;
        }

        // Write faces (1-indexed in OBJ format)
        for chunk in indices.chunks(3) {
            let i1 = chunk[0] + 1;
            let i2 = chunk[1] + 1;
            let i3 = chunk[2] + 1;
            writeln!(writer, "f {}//{} {}//{} {}//{}", i1, i1, i2, i2, i3, i3)?;
        }

        Ok(())
    }

    /// Calculate surface properties for analysis
    pub fn calculate_surface_properties(&self) -> SurfaceProperties {
        let is_orientable = self.twists % 2 == 0;
        let approximate_area = 2.0 * PI * self.major_radius * self.strip_width * 2.0;

        SurfaceProperties {
            is_orientable,
            approximate_surface_area: approximate_area,
            genus: if is_orientable { 1 } else { 0 }, // Topological genus
            euler_characteristic: if is_orientable { 0 } else { 1 },
        }
    }
}

/// Surface mathematical properties
#[derive(Debug)]
pub struct SurfaceProperties {
    pub is_orientable: bool,
    pub approximate_surface_area: f32,
    pub genus: i32,
    pub euler_characteristic: i32,
}

impl Default for KTwistedTorusGenerator {
    fn default() -> Self {
        let config = ConsciousnessConfig::default();
        Self {
            major_radius: config.default_torus_major_radius,
            strip_width: config.default_torus_minor_radius,
            twists: 1,
            u_steps: (config.resolution_factor * 64.0) as usize,
            v_steps: (config.resolution_factor * 32.0) as usize,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_generation() {
        let generator = KTwistedTorusGenerator::default();
        let result = generator.generate_mesh_data();
        assert!(result.is_ok());

        let (vertices, indices) = result.unwrap();
        assert!(!vertices.is_empty());
        assert!(!indices.is_empty());

        // Each vertex should have 6 components (pos + normal)
        assert_eq!(vertices.len() % 6, 0);

        // Indices should be triangles
        assert_eq!(indices.len() % 3, 0);
    }

    #[test]
    fn test_surface_properties() {
        let generator = KTwistedTorusGenerator::new(100.0, 20.0, 1, 64, 16);
        let props = generator.calculate_surface_properties();

        // k=1 should be non-orientable
        assert!(!props.is_orientable);
        assert_eq!(props.genus, 0);
        assert_eq!(props.euler_characteristic, 1);
    }

    #[test]
    fn test_orientability() {
        let non_orientable = KTwistedTorusGenerator::new(100.0, 20.0, 1, 32, 16);
        let orientable = KTwistedTorusGenerator::new(100.0, 20.0, 2, 32, 16);

        assert!(!non_orientable.calculate_surface_properties().is_orientable);
        assert!(orientable.calculate_surface_properties().is_orientable);
    }

    #[test]
    fn test_obj_export() {
        let generator = KTwistedTorusGenerator::new(50.0, 10.0, 1, 16, 8);
        let mut output = Vec::new();

        let result = generator.export_to_obj(&mut output);
        assert!(result.is_ok());

        let obj_content = String::from_utf8(output).unwrap();
        assert!(obj_content.contains("v ")); // Has vertices
        assert!(obj_content.contains("vn ")); // Has normals
        assert!(obj_content.contains("f ")); // Has faces
    }
}
