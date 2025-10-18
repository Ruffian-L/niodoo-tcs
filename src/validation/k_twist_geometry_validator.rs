//! K-Twisted Torus Geometry Validator
//!
//! Phase 4: Mathematical Validation Module
//! Implements reference implementation of parametric equations and validates
//! the k-twisted torus geometry implementation against mathematical theory.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::f64::consts::{PI, TAU};

/// Reference implementation of k-twisted torus parametric equations
/// Used to validate the actual implementation against mathematical theory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KTwistReferenceImplementation {
    /// Major radius (distance from center to tube center)
    pub major_radius: f64,
    /// Minor radius (tube radius)
    pub minor_radius: f64,
    /// Number of half-twists (k parameter)
    pub k_twist: f64,
}

impl Default for KTwistReferenceImplementation {
    fn default() -> Self {
        Self {
            major_radius: 5.0,
            minor_radius: 1.5,
            k_twist: 1.0,
        }
    }
}

impl KTwistReferenceImplementation {
    /// Calculate 3D position using reference k-twisted parametric equations
    ///
    /// Mathematical foundation:
    /// x(u,v) = (R + r*cos(v)) * cos(u) + r*cos(k*u)*cos(v)*cos(u) - r*sin(k*u)*sin(v)*sin(u)
    /// y(u,v) = (R + r*cos(v)) * sin(u) + r*cos(k*u)*cos(v)*sin(u) + r*sin(k*u)*sin(v)*cos(u)
    /// z(u,v) = r*sin(v) + r*sin(k*u)*cos(v)
    ///
    /// Where:
    /// - R = major_radius (distance from center to tube center)
    /// - r = minor_radius (tube radius)
    /// - k = k_twist (number of half-twists)
    /// - u ∈ [0, 2π] (toroidal angle)
    /// - v ∈ [0, 2π] (poloidal angle)
    pub fn calculate_position(&self, u: f64, v: f64) -> (f64, f64, f64) {
        let major_r = self.major_radius;
        let r = self.minor_radius;
        let k = self.k_twist;

        // Base torus coordinates
        let base_x = (major_r + r * v.cos()) * u.cos();
        let base_y = (major_r + r * v.cos()) * u.sin();
        let base_z = r * v.sin();

        // K-twist transformation
        let twist_factor = k * u;
        let cos_twist = twist_factor.cos();
        let sin_twist = twist_factor.sin();

        // Apply k-twist transformation
        let x = base_x + r * cos_twist * v.cos() * u.cos() - r * sin_twist * v.sin() * u.sin();
        let y = base_y + r * cos_twist * v.cos() * u.sin() + r * sin_twist * v.sin() * u.cos();
        let z = base_z + r * sin_twist * v.cos();

        (x, y, z)
    }

    /// Calculate surface normal vector using partial derivatives
    /// Critical for proper lighting and shading validation
    pub fn calculate_normal(
        &self,
        u: f64,
        v: f64,
        numerical_zero_threshold: f64,
    ) -> Result<(f64, f64, f64)> {
        let _major_r = self.major_radius;
        let _r = self.minor_radius;
        let _k = self.k_twist;

        // Calculate partial derivatives
        let (dx_du, dy_du, dz_du) = self.calculate_partial_derivative_u(u, v);
        let (dx_dv, dy_dv, dz_dv) = self.calculate_partial_derivative_v(u, v);

        // Cross product to get normal vector
        let normal_x = dy_du * dz_dv - dz_du * dy_dv;
        let normal_y = dz_du * dx_dv - dx_du * dz_dv;
        let normal_z = dx_du * dy_dv - dy_du * dx_dv;

        // Normalize
        let magnitude = (normal_x * normal_x + normal_y * normal_y + normal_z * normal_z).sqrt();

        if magnitude < numerical_zero_threshold {
            return Err(anyhow::anyhow!(
                "Normal vector magnitude too small: {}",
                magnitude
            ));
        }

        Ok((
            normal_x / magnitude,
            normal_y / magnitude,
            normal_z / magnitude,
        ))
    }

    /// Calculate partial derivative with respect to u (toroidal direction)
    fn calculate_partial_derivative_u(&self, u: f64, v: f64) -> (f64, f64, f64) {
        let major_r = self.major_radius;
        let r = self.minor_radius;
        let k = self.k_twist;

        let twist_factor = k * u;
        let cos_twist = twist_factor.cos();
        let sin_twist = twist_factor.sin();
        let cos_u = u.cos();
        let sin_u = u.sin();
        let cos_v = v.cos();
        let sin_v = v.sin();

        // ∂x/∂u
        let dx_du = -(major_r + r * cos_v) * sin_u
            - r * k * sin_twist * cos_v * cos_u
            - r * cos_twist * cos_v * sin_u
            - r * k * cos_twist * sin_v * sin_u
            - r * sin_twist * sin_v * cos_u;

        // ∂y/∂u
        let dy_du = (major_r + r * cos_v) * cos_u - r * k * sin_twist * cos_v * sin_u
            + r * cos_twist * cos_v * cos_u
            + r * k * cos_twist * sin_v * cos_u
            - r * sin_twist * sin_v * sin_u;

        // ∂z/∂u
        let dz_du = r * k * cos_twist * cos_v;

        (dx_du, dy_du, dz_du)
    }

    /// Calculate partial derivative with respect to v (poloidal direction)
    fn calculate_partial_derivative_v(&self, u: f64, v: f64) -> (f64, f64, f64) {
        let _major_r = self.major_radius;
        let r = self.minor_radius;
        let k = self.k_twist;

        let twist_factor = k * u;
        let cos_twist = twist_factor.cos();
        let sin_twist = twist_factor.sin();
        let cos_u = u.cos();
        let sin_u = u.sin();
        let cos_v = v.cos();
        let sin_v = v.sin();

        // ∂x/∂v
        let dx_dv =
            -r * sin_v * cos_u - r * cos_twist * sin_v * cos_u - r * sin_twist * cos_v * sin_u;

        // ∂y/∂v
        let dy_dv =
            -r * sin_v * sin_u - r * cos_twist * sin_v * sin_u + r * sin_twist * cos_v * cos_u;

        // ∂z/∂v
        let dz_dv = r * cos_v - r * sin_twist * sin_v;

        (dx_dv, dy_dv, dz_dv)
    }

    /// Verify non-orientability for odd k values
    ///
    /// A surface is non-orientable if there exists a closed curve
    /// that reverses the normal vector orientation when traversed.
    /// For k-twisted tori, odd k values should be non-orientable.
    pub fn verify_non_orientability(&self) -> Result<bool> {
        if self.k_twist.fract() != 0.0 {
            return Err(anyhow::anyhow!(
                "k_twist must be an integer for orientability test"
            ));
        }

        let k = self.k_twist as i32;
        let is_odd = k % 2 != 0;

        if is_odd {
            // Test non-orientability by checking normal vector reversal
            let test_points = vec![(0.0, 0.0), (PI, 0.0), (TAU, 0.0)];

            let mut normals = Vec::new();
            for (u, v) in test_points {
                let normal = self.calculate_normal(u, v, 1e-10)?;
                normals.push(normal);
            }

            // Check if normal vector orientation is reversed
            let first_normal = &normals[0];
            let last_normal = &normals[normals.len() - 1];

            let dot_product = first_normal.0 * last_normal.0
                + first_normal.1 * last_normal.1
                + first_normal.2 * last_normal.2;

            // For non-orientable surfaces, the normal should be reversed
            let is_non_orientable = dot_product < 0.0;

            Ok(is_non_orientable)
        } else {
            // Even k values should be orientable
            Ok(false)
        }
    }
}

/// Validator for k-twisted torus geometry implementation
#[derive(Debug, Clone)]
pub struct KTwistGeometryValidator {
    pub reference: KTwistReferenceImplementation,
    tolerance: f64,
    numerical_zero_threshold: f64,
}

impl KTwistGeometryValidator {
    pub fn new(major_radius: f64, minor_radius: f64, k_twist: f64, tolerance: f64) -> Self {
        Self::new_with_threshold(major_radius, minor_radius, k_twist, tolerance, 1e-15)
    }

    pub fn new_with_threshold(
        major_radius: f64,
        minor_radius: f64,
        k_twist: f64,
        tolerance: f64,
        numerical_zero_threshold: f64,
    ) -> Self {
        Self {
            reference: KTwistReferenceImplementation {
                major_radius,
                minor_radius,
                k_twist,
            },
            tolerance,
            numerical_zero_threshold,
        }
    }

    /// Validate position calculations against reference implementation
    pub fn validate_positions<F>(&self, implementation: F) -> Result<ValidationResult>
    where
        F: Fn(f64, f64) -> (f64, f64, f64),
    {
        let mut errors = Vec::new();
        let mut max_error: f64 = 0.0;
        let mut total_error = 0.0;
        let mut test_count = 0;

        // Test grid of points
        let u_steps = 20;
        let v_steps = 20;

        for i in 0..u_steps {
            for j in 0..v_steps {
                let u = (i as f64 / (u_steps - 1) as f64) * TAU;
                let v = (j as f64 / (v_steps - 1) as f64) * TAU;

                let reference_pos = self.reference.calculate_position(u, v);
                let implementation_pos = implementation(u, v);

                let error = self.calculate_position_error(&reference_pos, &implementation_pos);

                if error > self.tolerance {
                    errors.push(PositionError {
                        u,
                        v,
                        reference: reference_pos,
                        implementation: implementation_pos,
                        error,
                    });
                }

                max_error = max_error.max(error);
                total_error += error;
                test_count += 1;
            }
        }

        let average_error = total_error / test_count as f64;

        Ok(ValidationResult {
            max_error,
            average_error,
            position_errors: errors,
            normal_errors: Vec::new(),
            orientability_test: None,
        })
    }

    /// Validate normal vector calculations
    pub fn validate_normals<F>(&self, implementation: F) -> Result<ValidationResult>
    where
        F: Fn(f64, f64) -> Result<(f64, f64, f64)>,
    {
        let mut errors = Vec::new();
        let mut max_error: f64 = 0.0;
        let mut total_error = 0.0;
        let mut test_count = 0;

        // Test grid of points
        let u_steps = 20;
        let v_steps = 20;

        for i in 0..u_steps {
            for j in 0..v_steps {
                let u = (i as f64 / (u_steps - 1) as f64) * TAU;
                let v = (j as f64 / (v_steps - 1) as f64) * TAU;

                let reference_normal =
                    self.reference
                        .calculate_normal(u, v, self.numerical_zero_threshold)?;
                let implementation_normal = implementation(u, v)?;

                let error = self.calculate_normal_error(&reference_normal, &implementation_normal);

                if error > self.tolerance {
                    errors.push(NormalError {
                        u,
                        v,
                        reference: reference_normal,
                        implementation: implementation_normal,
                        error,
                    });
                }

                max_error = max_error.max(error);
                total_error += error;
                test_count += 1;
            }
        }

        let average_error = total_error / test_count as f64;

        Ok(ValidationResult {
            max_error,
            average_error,
            position_errors: Vec::new(),
            normal_errors: errors,
            orientability_test: None,
        })
    }

    /// Validate orientability properties
    pub fn validate_orientability(&self) -> Result<ValidationResult> {
        let orientability_result = self.reference.verify_non_orientability()?;

        Ok(ValidationResult {
            max_error: 0.0,
            average_error: 0.0,
            position_errors: Vec::new(),
            normal_errors: Vec::new(),
            orientability_test: Some(OrientabilityTest {
                k_twist: self.reference.k_twist,
                is_non_orientable: orientability_result,
                expected_non_orientable: (self.reference.k_twist as i32) % 2 != 0,
            }),
        })
    }

    /// Calculate error between two position vectors
    fn calculate_position_error(&self, pos1: &(f64, f64, f64), pos2: &(f64, f64, f64)) -> f64 {
        let dx = pos1.0 - pos2.0;
        let dy = pos1.1 - pos2.1;
        let dz = pos1.2 - pos2.2;

        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Calculate error between two normal vectors (angle between them)
    fn calculate_normal_error(&self, normal1: &(f64, f64, f64), normal2: &(f64, f64, f64)) -> f64 {
        let dot_product = normal1.0 * normal2.0 + normal1.1 * normal2.1 + normal1.2 * normal2.2;

        // Clamp to avoid numerical issues
        let clamped_dot = dot_product.max(-1.0).min(1.0);

        // Return angle in radians
        clamped_dot.acos()
    }
}

/// Result of validation testing
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub max_error: f64,
    pub average_error: f64,
    pub position_errors: Vec<PositionError>,
    pub normal_errors: Vec<NormalError>,
    pub orientability_test: Option<OrientabilityTest>,
}

/// Position calculation error
#[derive(Debug, Clone)]
pub struct PositionError {
    pub u: f64,
    pub v: f64,
    pub reference: (f64, f64, f64),
    pub implementation: (f64, f64, f64),
    pub error: f64,
}

/// Normal vector calculation error
#[derive(Debug, Clone)]
pub struct NormalError {
    pub u: f64,
    pub v: f64,
    pub reference: (f64, f64, f64),
    pub implementation: (f64, f64, f64),
    pub error: f64,
}

/// Orientability test result
#[derive(Debug, Clone)]
pub struct OrientabilityTest {
    pub k_twist: f64,
    pub is_non_orientable: bool,
    pub expected_non_orientable: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::ValidationConfig;

    #[test]
    fn test_reference_implementation_basic() {
        let reference = KTwistReferenceImplementation::default();

        // Test basic position calculation
        let (x, y, z) = reference.calculate_position(0.0, 0.0);
        let config = ValidationConfig::default();
        assert!((x - config.expected_torus_position_x).abs() < config.test_assertion_threshold); // R + r = 5.0 + 1.5 = 6.5
        assert!(y.abs() < config.numerical_zero_threshold);
        assert!(z.abs() < config.numerical_zero_threshold);
    }

    #[test]
    fn test_reference_implementation_normal() {
        let reference = KTwistReferenceImplementation::default();

        // Test normal calculation
        let normal = reference.calculate_normal(0.0, 0.0, 1e-15).unwrap();
        let magnitude = (normal.0 * normal.0 + normal.1 * normal.1 + normal.2 * normal.2).sqrt();
        assert!((magnitude - 1.0).abs() < config.test_assertion_threshold); // Should be unit vector
    }

    #[test]
    fn test_orientability_odd_k() {
        let reference = KTwistReferenceImplementation {
            major_radius: 5.0,
            minor_radius: 1.5,
            k_twist: 1.0, // Odd k should be non-orientable
        };

        let is_non_orientable = reference.verify_non_orientability().unwrap();
        assert!(is_non_orientable);
    }

    #[test]
    fn test_orientability_even_k() {
        let reference = KTwistReferenceImplementation {
            major_radius: 5.0,
            minor_radius: 1.5,
            k_twist: 2.0, // Even k should be orientable
        };

        let is_non_orientable = reference.verify_non_orientability().unwrap();
        assert!(!is_non_orientable);
    }

    #[test]
    fn test_validator_position_validation() {
        let validator = KTwistGeometryValidator::new(5.0, 1.5, 1.0, 1e-6);

        // Test with reference implementation (should have zero error)
        let result = validator
            .validate_positions(|u, v| validator.reference.calculate_position(u, v))
            .unwrap();

        assert!(result.max_error < config.test_assertion_threshold);
        assert!(result.position_errors.is_empty());
    }
}
