//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Advanced Torus Enhancements
//! 
//! This module provides enhanced mathematical operations for torus topology,
//! including geodesic calculations, curvature analysis, and optimization techniques.

use std::f64::consts::{PI, TAU};

/// Enhanced torus geometry with advanced mathematical operations
#[derive(Debug, Clone)]
pub struct EnhancedTorus {
    pub major_radius: f64,
    pub minor_radius: f64,
    pub k_twist: f64,
    pub resolution: (usize, usize),
}

impl EnhancedTorus {
    /// Create a new enhanced torus with k-twist topology
    pub fn new(major_radius: f64, minor_radius: f64, k_twist: f64, resolution: (usize, usize)) -> Self {
        Self {
            major_radius,
            minor_radius,
            k_twist,
            resolution,
        }
    }

    /// Calculate geodesic distance between two points on the torus surface
    /// Uses the k-twisted torus metric for accurate distance calculations
    pub fn geodesic_distance(&self, p1: (f64, f64), p2: (f64, f64)) -> f64 {
        let (u1, v1) = p1;
        let (u2, v2) = p2;
        
        // Convert to Cartesian coordinates
        let pos1 = self.parametric_to_cartesian(u1, v1);
        let pos2 = self.parametric_to_cartesian(u2, v2);
        
        // Calculate arc length using the k-twist metric
        let du = u2 - u1;
        let dv = v2 - v1;
        
        // Account for torus periodicity
        let du_wrapped = ((du + PI) % TAU) - PI;
        let dv_wrapped = ((dv + PI) % TAU) - PI;
        
        // Metric tensor components for k-twisted torus
        let g_uu = self.metric_component_uu(u1, v1);
        let g_uv = self.metric_component_uv(u1, v1);
        let g_vv = self.metric_component_vv(u1, v1);
        
        // Geodesic distance formula: ds² = g_uu * du² + 2*g_uv * du*dv + g_vv * dv²
        let ds_squared = g_uu * du_wrapped * du_wrapped + 
                        2.0 * g_uv * du_wrapped * dv_wrapped + 
                        g_vv * dv_wrapped * dv_wrapped;
        
        ds_squared.sqrt()
    }

    /// Convert parametric coordinates to Cartesian coordinates
    pub fn parametric_to_cartesian(&self, u: f64, v: f64) -> (f64, f64, f64) {
        let twist_term = 2.0 * self.k_twist * u;
        
        let x = (self.major_radius + v * twist_term.cos()) * u.cos();
        let y = (self.major_radius + v * twist_term.cos()) * u.sin();
        let z = v * twist_term.sin();
        
        (x, y, z)
    }

    /// Calculate Gaussian curvature at a point on the surface
    pub fn gaussian_curvature(&self, u: f64, v: f64) -> f64 {
        let k = self.k_twist;
        
        // First fundamental form coefficients
        let E = self.first_fundamental_form_e(u, v);
        let F = self.first_fundamental_form_f(u, v);
        let G = self.first_fundamental_form_g(u, v);
        
        // Second fundamental form coefficients
        let L = self.second_fundamental_form_l(u, v);
        let M = self.second_fundamental_form_m(u, v);
        let N = self.second_fundamental_form_n(u, v);
        
        // Gaussian curvature: K = (LN - M²) / (EG - F²)
        let numerator = L * N - M * M;
        let denominator = E * G - F * F;
        
        if denominator.abs() < 1e-12 {
            0.0 // Avoid division by zero
        } else {
            numerator / denominator
        }
    }

    /// Calculate mean curvature at a point on the surface
    pub fn mean_curvature(&self, u: f64, v: f64) -> f64 {
        let E = self.first_fundamental_form_e(u, v);
        let F = self.first_fundamental_form_f(u, v);
        let G = self.first_fundamental_form_g(u, v);
        let L = self.second_fundamental_form_l(u, v);
        let M = self.second_fundamental_form_m(u, v);
        let N = self.second_fundamental_form_n(u, v);
        
        // Mean curvature: H = (EN - 2FM + GL) / (2(EG - F²))
        let numerator = E * N - 2.0 * F * M + G * L;
        let denominator = 2.0 * (E * G - F * F);
        
        if denominator.abs() < 1e-12 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Calculate surface area of the k-twisted torus
    pub fn surface_area(&self) -> f64 {
        let mut total_area = 0.0;
        let du = TAU / self.resolution.0 as f64;
        let dv = TAU / self.resolution.1 as f64;
        
        for i in 0..self.resolution.0 {
            for j in 0..self.resolution.1 {
                let u = i as f64 * du;
                let v = j as f64 * dv;
                
                let E = self.first_fundamental_form_e(u, v);
                let F = self.first_fundamental_form_f(u, v);
                let G = self.first_fundamental_form_g(u, v);
                
                // Area element: dA = √(EG - F²) du dv
                let area_element = (E * G - F * F).sqrt() * du * dv;
                total_area += area_element;
            }
        }
        
        total_area
    }

    /// Find optimal mesh resolution based on curvature variation
    pub fn adaptive_resolution(&self, target_error: f64) -> (usize, usize) {
        let mut u_res = 32;
        let mut v_res = 16;
        
        // Iteratively refine resolution until curvature error is below threshold
        for _ in 0..5 {
            let max_curvature_error = self.estimate_curvature_error(u_res, v_res);
            
            if max_curvature_error < target_error {
                break;
            }
            
            // Increase resolution
            u_res = (u_res as f64 * 1.5) as usize;
            v_res = (v_res as f64 * 1.5) as usize;
            
            // Cap resolution to prevent excessive computation
            u_res = u_res.min(512);
            v_res = v_res.min(256);
        }
        
        (u_res, v_res)
    }

    /// Estimate maximum curvature error for given resolution
    fn estimate_curvature_error(&self, u_res: usize, v_res: usize) -> f64 {
        let mut max_error: f64 = 0.0;
        let du = TAU / u_res as f64;
        let dv = TAU / v_res as f64;
        
        for i in 0..(u_res - 1) {
            for j in 0..(v_res - 1) {
                let u = i as f64 * du;
                let v = j as f64 * dv;
                
                let k1 = self.gaussian_curvature(u, v);
                let k2 = self.gaussian_curvature(u + du, v);
                let k3 = self.gaussian_curvature(u, v + dv);
                
                let error: f64 = ((k2 - k1).abs() + (k3 - k1).abs()) / 2.0;
                max_error = max_error.max(error);
            }
        }
        
        max_error
    }

    // Metric tensor components for k-twisted torus
    fn metric_component_uu(&self, u: f64, v: f64) -> f64 {
        let k = self.k_twist;
        let twist_term = 2.0 * k * u;
        
        let du_x = -(self.major_radius + v * twist_term.cos()) * u.sin() 
                  - v * 2.0 * k * twist_term.sin() * u.cos();
        let du_y = (self.major_radius + v * twist_term.cos()) * u.cos() 
                  - v * 2.0 * k * twist_term.sin() * u.sin();
        let du_z = v * 2.0 * k * twist_term.cos();
        
        du_x * du_x + du_y * du_y + du_z * du_z
    }

    fn metric_component_uv(&self, u: f64, v: f64) -> f64 {
        let k = self.k_twist;
        let twist_term = 2.0 * k * u;
        
        let du_x = -(self.major_radius + v * twist_term.cos()) * u.sin() 
                  - v * 2.0 * k * twist_term.sin() * u.cos();
        let du_y = (self.major_radius + v * twist_term.cos()) * u.cos() 
                  - v * 2.0 * k * twist_term.sin() * u.sin();
        let du_z = v * 2.0 * k * twist_term.cos();
        
        let dv_x = twist_term.cos() * u.cos();
        let dv_y = twist_term.cos() * u.sin();
        let dv_z = twist_term.sin();
        
        du_x * dv_x + du_y * dv_y + du_z * dv_z
    }

    fn metric_component_vv(&self, u: f64, v: f64) -> f64 {
        let k = self.k_twist;
        let twist_term = 2.0 * k * u;
        
        let dv_x = twist_term.cos() * u.cos();
        let dv_y = twist_term.cos() * u.sin();
        let dv_z = twist_term.sin();
        
        dv_x * dv_x + dv_y * dv_y + dv_z * dv_z
    }

    // First fundamental form coefficients
    fn first_fundamental_form_e(&self, u: f64, v: f64) -> f64 {
        self.metric_component_uu(u, v)
    }

    fn first_fundamental_form_f(&self, u: f64, v: f64) -> f64 {
        self.metric_component_uv(u, v)
    }

    fn first_fundamental_form_g(&self, u: f64, v: f64) -> f64 {
        self.metric_component_vv(u, v)
    }

    // Second fundamental form coefficients
    fn second_fundamental_form_l(&self, u: f64, v: f64) -> f64 {
        let k = self.k_twist;
        let twist_term = 2.0 * k * u;
        
        // Second partial derivatives
        let d2u_x = -(self.major_radius + v * twist_term.cos()) * u.cos() 
                    + v * 4.0 * k * k * twist_term.cos() * u.cos()
                    - v * 2.0 * k * twist_term.sin() * u.sin();
        let d2u_y = -(self.major_radius + v * twist_term.cos()) * u.sin() 
                    + v * 4.0 * k * k * twist_term.cos() * u.sin()
                    + v * 2.0 * k * twist_term.sin() * u.cos();
        let d2u_z = -v * 4.0 * k * k * twist_term.sin();
        
        // Normal vector components
        let normal = self.surface_normal(u, v);
        
        d2u_x * normal.0 + d2u_y * normal.1 + d2u_z * normal.2
    }

    fn second_fundamental_form_m(&self, u: f64, v: f64) -> f64 {
        let k = self.k_twist;
        let twist_term = 2.0 * k * u;
        
        // Mixed partial derivatives
        let d2uv_x = -2.0 * k * twist_term.sin() * u.cos();
        let d2uv_y = -2.0 * k * twist_term.sin() * u.sin();
        let d2uv_z = 2.0 * k * twist_term.cos();
        
        let normal = self.surface_normal(u, v);
        
        d2uv_x * normal.0 + d2uv_y * normal.1 + d2uv_z * normal.2
    }

    fn second_fundamental_form_n(&self, u: f64, v: f64) -> f64 {
        // Second partial derivative with respect to v is zero for this parametrization
        0.0
    }

    /// Calculate surface normal vector
    fn surface_normal(&self, u: f64, v: f64) -> (f64, f64, f64) {
        let k = self.k_twist;
        let twist_term = 2.0 * k * u;
        
        // Partial derivatives
        let du_x = -(self.major_radius + v * twist_term.cos()) * u.sin() 
                  - v * 2.0 * k * twist_term.sin() * u.cos();
        let du_y = (self.major_radius + v * twist_term.cos()) * u.cos() 
                  - v * 2.0 * k * twist_term.sin() * u.sin();
        let du_z = v * 2.0 * k * twist_term.cos();
        
        let dv_x = twist_term.cos() * u.cos();
        let dv_y = twist_term.cos() * u.sin();
        let dv_z = twist_term.sin();
        
        // Cross product for normal
        let nx = du_y * dv_z - du_z * dv_y;
        let ny = du_z * dv_x - du_x * dv_z;
        let nz = du_x * dv_y - du_y * dv_x;
        
        // Normalize
        let length = (nx * nx + ny * ny + nz * nz).sqrt();
        if length > 1e-12 {
            (nx / length, ny / length, nz / length)
        } else {
            (0.0, 0.0, 1.0)
        }
    }

    /// Generate optimized mesh with curvature-based refinement
    pub fn generate_optimized_mesh(&self, target_error: f64) -> Result<(Vec<f64>, Vec<u32>), String> {
        let (u_res, v_res) = self.adaptive_resolution(target_error);
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        
        // Generate vertices
        for i in 0..u_res {
            for j in 0..v_res {
                let u = (i as f64 / (u_res - 1) as f64) * TAU;
                let v = (j as f64 / (v_res - 1) as f64) * TAU;
                
                let (x, y, z) = self.parametric_to_cartesian(u, v);
                let (nx, ny, nz) = self.surface_normal(u, v);
                
                // Store vertex data: position + normal + uv
                vertices.extend_from_slice(&[x, y, z, nx, ny, nz, u / TAU, v / TAU]);
            }
        }
        
        // Generate indices
        for i in 0..(u_res - 1) {
            for j in 0..(v_res - 1) {
                let current = i * v_res + j;
                let next_u = ((i + 1) % u_res) * v_res + j;
                let next_v = i * v_res + (j + 1) % v_res;
                let next_both = ((i + 1) % u_res) * v_res + (j + 1) % v_res;
                
                // Two triangles per quad
                indices.extend_from_slice(&[current as u32, next_u as u32, next_v as u32]);
                indices.extend_from_slice(&[next_u as u32, next_both as u32, next_v as u32]);
            }
        }
        
        Ok((vertices, indices))
    }

    /// Calculate topological invariants
    pub fn topological_invariants(&self) -> TopologicalInvariants {
        let is_orientable = (self.k_twist as i32) % 2 == 0;
        let genus = if is_orientable { 1 } else { 0 };
        let euler_characteristic = if is_orientable { 0 } else { 1 };
        
        TopologicalInvariants {
            is_orientable,
            genus,
            euler_characteristic,
            k_twist: self.k_twist,
        }
    }
}

/// Topological invariants of the k-twisted torus
#[derive(Debug, Clone)]
pub struct TopologicalInvariants {
    pub is_orientable: bool,
    pub genus: i32,
    pub euler_characteristic: i32,
    pub k_twist: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geodesic_distance() {
        let torus = EnhancedTorus::new(100.0, 20.0, 1.0, (32, 16));
        
        let p1 = (0.0, 0.0);
        let p2 = (PI, 0.0);
        
        let distance = torus.geodesic_distance(p1, p2);
        assert!(distance > 0.0);
        assert!(distance.is_finite());
    }

    #[test]
    fn test_gaussian_curvature() {
        let torus = EnhancedTorus::new(100.0, 20.0, 1.0, (32, 16));
        
        let curvature = torus.gaussian_curvature(0.0, 0.0);
        assert!(curvature.is_finite());
    }

    #[test]
    fn test_surface_area() {
        let torus = EnhancedTorus::new(100.0, 20.0, 1.0, (64, 32));
        
        let area = torus.surface_area();
        assert!(area > 0.0);
        assert!(area.is_finite());
    }

    #[test]
    fn test_adaptive_resolution() {
        let torus = EnhancedTorus::new(100.0, 20.0, 1.0, (32, 16));
        
        let (u_res, v_res) = torus.adaptive_resolution(0.01);
        assert!(u_res >= 32);
        assert!(v_res >= 16);
    }

    #[test]
    fn test_optimized_mesh_generation() {
        let torus = EnhancedTorus::new(100.0, 20.0, 1.0, (32, 16));
        
        let result = torus.generate_optimized_mesh(0.01);
        assert!(result.is_ok());
        
        let (vertices, indices) = result.unwrap();
        assert!(!vertices.is_empty());
        assert!(!indices.is_empty());
        assert_eq!(vertices.len() % 8, 0); // 8 components per vertex
        assert_eq!(indices.len() % 3, 0); // 3 indices per triangle
    }

    #[test]
    fn test_topological_invariants() {
        let torus_non_orientable = EnhancedTorus::new(100.0, 20.0, 1.0, (32, 16));
        let torus_orientable = EnhancedTorus::new(100.0, 20.0, 2.0, (32, 16));
        
        let inv1 = torus_non_orientable.topological_invariants();
        let inv2 = torus_orientable.topological_invariants();
        
        assert!(!inv1.is_orientable);
        assert!(inv2.is_orientable);
        assert_eq!(inv1.genus, 0);
        assert_eq!(inv2.genus, 1);
    }
}

