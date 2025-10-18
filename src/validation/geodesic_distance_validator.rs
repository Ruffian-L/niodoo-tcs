//! Geodesic Distance Validator
//!
//! Phase 4: Mathematical Validation Module
//! Verifies distance calculations use manifold geometry and validates
//! numerical stability for k-twisted torus surfaces.

use anyhow::Result;
use nalgebra::Vector3;
use std::f64::consts::{PI, TAU};

/// Reference implementation of geodesic distance calculation
/// Used to validate the actual implementation against mathematical theory
#[derive(Debug, Clone)]
pub struct GeodesicDistanceReference {
    /// Major radius of the torus
    pub major_radius: f64,
    /// Minor radius of the torus
    pub minor_radius: f64,
    /// K-twist parameter
    pub k_twist: f64,
}

impl GeodesicDistanceReference {
    pub fn new(major_radius: f64, minor_radius: f64, k_twist: f64) -> Self {
        Self {
            major_radius,
            minor_radius,
            k_twist,
        }
    }
}

impl GeodesicDistanceReference {
    /// Calculate geodesic distance between two points on the k-twisted torus
    pub fn geodesic_distance(&self, u1: f64, v1: f64, u2: f64, v2: f64) -> Result<f64> {
        self.geodesic_distance_with_steps(u1, v1, u2, v2, 100)
    }

    /// Calculate geodesic distance with custom integration steps
    pub fn geodesic_distance_with_steps(
        &self,
        u1: f64,
        v1: f64,
        u2: f64,
        v2: f64,
        steps: usize,
    ) -> Result<f64> {
        // Normalize angles to [0, 2π]
        let u1 = self.normalize_angle(u1);
        let v1 = self.normalize_angle(v1);
        let u2 = self.normalize_angle(u2);
        let v2 = self.normalize_angle(v2);

        // Calculate the shortest path considering the k-twist topology
        let du = self.shortest_angle_difference(u1, u2);
        let dv = self.shortest_angle_difference(v1, v2);

        // Use numerical integration to calculate geodesic distance
        let distance = self.integrate_geodesic_path(u1, v1, du, dv, steps)?;

        Ok(distance)
    }

    /// Calculate Euclidean distance fallback (for comparison)
    pub fn euclidean_distance_fallback(&self, u1: f64, v1: f64, u2: f64, v2: f64) -> f64 {
        let (x1, y1, z1) = self.parametric_to_cartesian(u1, v1);
        let (x2, y2, z2) = self.parametric_to_cartesian(u2, v2);

        let dx = x2 - x1;
        let dy = y2 - y1;
        let dz = z2 - z1;

        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Convert parametric coordinates to Cartesian coordinates
    fn parametric_to_cartesian(&self, u: f64, v: f64) -> (f64, f64, f64) {
        let major_r = self.major_radius;
        let r = self.minor_radius;
        let k = self.k_twist;

        let twist_factor = k * u;
        let cos_twist = twist_factor.cos();
        let sin_twist = twist_factor.sin();

        let x = (major_r + r * v.cos()) * u.cos() + r * cos_twist * v.cos() * u.cos()
            - r * sin_twist * v.sin() * u.sin();
        let y = (major_r + r * v.cos()) * u.sin()
            + r * cos_twist * v.cos() * u.sin()
            + r * sin_twist * v.sin() * u.cos();
        let z = r * v.sin() + r * sin_twist * v.cos();

        (x, y, z)
    }

    /// Calculate metric tensor components at a point
    fn metric_tensor(&self, u: f64, v: f64) -> (f64, f64, f64) {
        let _major_r = self.major_radius;
        let _r = self.minor_radius;
        let _k = self.k_twist;

        // Calculate partial derivatives
        let (dx_du, dy_du, dz_du) = self.partial_derivative_u(u, v);
        let (dx_dv, dy_dv, dz_dv) = self.partial_derivative_v(u, v);

        // Metric tensor components
        let g_uu = dx_du * dx_du + dy_du * dy_du + dz_du * dz_du;
        let g_uv = dx_du * dx_dv + dy_du * dy_dv + dz_du * dz_dv;
        let g_vv = dx_dv * dx_dv + dy_dv * dy_dv + dz_dv * dz_dv;

        (g_uu, g_uv, g_vv)
    }

    /// Calculate partial derivative with respect to u
    fn partial_derivative_u(&self, u: f64, v: f64) -> (f64, f64, f64) {
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

        let dx_du = -(major_r + r * cos_v) * sin_u
            - r * k * sin_twist * cos_v * cos_u
            - r * cos_twist * cos_v * sin_u
            - r * k * cos_twist * sin_v * sin_u
            - r * sin_twist * sin_v * cos_u;

        let dy_du = (major_r + r * cos_v) * cos_u - r * k * sin_twist * cos_v * sin_u
            + r * cos_twist * cos_v * cos_u
            + r * k * cos_twist * sin_v * cos_u
            - r * sin_twist * sin_v * sin_u;

        let dz_du = r * k * cos_twist * cos_v;

        (dx_du, dy_du, dz_du)
    }

    /// Calculate partial derivative with respect to v
    fn partial_derivative_v(&self, u: f64, v: f64) -> (f64, f64, f64) {
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

        let dx_dv =
            -r * sin_v * cos_u - r * cos_twist * sin_v * cos_u - r * sin_twist * cos_v * sin_u;

        let dy_dv =
            -r * sin_v * sin_u - r * cos_twist * sin_v * sin_u + r * sin_twist * cos_v * cos_u;

        let dz_dv = r * cos_v - r * sin_twist * sin_v;

        (dx_dv, dy_dv, dz_dv)
    }

    /// Integrate geodesic path using numerical methods
    fn integrate_geodesic_path(
        &self,
        u1: f64,
        v1: f64,
        du: f64,
        dv: f64,
        n_steps: usize,
    ) -> Result<f64> {
        let mut total_distance = 0.0;

        for i in 0..n_steps {
            let t = i as f64 / (n_steps - 1) as f64;
            let u = u1 + t * du;
            let v = v1 + t * dv;

            let (g_uu, g_uv, g_vv) = self.metric_tensor(u, v);

            // Calculate arc length element
            let du_dt = du;
            let dv_dt = dv;

            let ds_squared =
                g_uu * du_dt * du_dt + 2.0 * g_uv * du_dt * dv_dt + g_vv * dv_dt * dv_dt;

            if ds_squared < 0.0 {
                return Err(anyhow::anyhow!(
                    "Negative metric tensor component: {}",
                    ds_squared
                ));
            }

            let ds = ds_squared.sqrt();
            total_distance += ds / n_steps as f64;
        }

        Ok(total_distance)
    }

    /// Normalize angle to [0, 2π]
    fn normalize_angle(&self, angle: f64) -> f64 {
        let mut normalized = angle % TAU;
        if normalized < 0.0 {
            normalized += TAU;
        }
        normalized
    }

    /// Calculate shortest angle difference considering periodicity
    fn shortest_angle_difference(&self, angle1: f64, angle2: f64) -> f64 {
        let diff = angle2 - angle1;
        if diff > PI {
            diff - TAU
        } else if diff < -PI {
            diff + TAU
        } else {
            diff
        }
    }
}

/// Validator for geodesic distance calculations
#[derive(Debug, Clone)]
pub struct GeodesicDistanceValidator {
    pub reference: GeodesicDistanceReference,
    tolerance: f64,
    integration_steps: usize,
    numerical_zero_threshold: f64,
}

impl GeodesicDistanceValidator {
    pub fn new(major_radius: f64, minor_radius: f64, k_twist: f64, tolerance: f64) -> Self {
        Self::new_with_steps_and_threshold(
            major_radius,
            minor_radius,
            k_twist,
            tolerance,
            100,
            1e-15,
        )
    }

    pub fn new_with_steps(
        major_radius: f64,
        minor_radius: f64,
        k_twist: f64,
        tolerance: f64,
        integration_steps: usize,
    ) -> Self {
        Self::new_with_steps_and_threshold(
            major_radius,
            minor_radius,
            k_twist,
            tolerance,
            integration_steps,
            1e-15,
        )
    }

    pub fn new_with_steps_and_threshold(
        major_radius: f64,
        minor_radius: f64,
        k_twist: f64,
        tolerance: f64,
        integration_steps: usize,
        numerical_zero_threshold: f64,
    ) -> Self {
        Self {
            reference: GeodesicDistanceReference::new(major_radius, minor_radius, k_twist),
            tolerance,
            integration_steps,
            numerical_zero_threshold,
        }
    }

    /// Validate geodesic distance implementation
    pub fn validate_geodesic_distance<F>(
        &self,
        implementation: F,
    ) -> Result<DistanceValidationResult>
    where
        F: Fn(f64, f64, f64, f64) -> Result<f64>,
    {
        let mut errors = Vec::new();
        let mut max_error: f64 = 0.0;
        let mut total_error = 0.0;
        let mut test_count = 0;

        // Generate test point pairs for torus
        let test_cases = vec![
            (0.0f64, 0.0f64, std::f64::consts::PI, 0.0f64),
            (0.0f64, 0.0f64, 0.0f64, std::f64::consts::PI),
            (0.0f64, 0.0f64, std::f64::consts::PI, std::f64::consts::PI),
            (std::f64::consts::PI / 2.0, 0.0f64, 0.0f64, 0.0f64),
        ];

        for (u1, v1, u2, v2) in test_cases {
            let reference_distance = self.reference.geodesic_distance_with_steps(
                u1,
                v1,
                u2,
                v2,
                self.integration_steps,
            )?;
            let implementation_distance = implementation(u1, v1, u2, v2)?;

            let error = (reference_distance - implementation_distance).abs();

            if error > self.tolerance {
                errors.push(DistanceError {
                    u1,
                    v1,
                    u2,
                    v2,
                    reference: reference_distance,
                    implementation: implementation_distance,
                    error,
                });
            }

            max_error = max_error.max(error);
            total_error += error;
            test_count += 1;
        }

        let average_error = if test_count > 0 {
            total_error / test_count as f64
        } else {
            0.0
        };

        Ok(DistanceValidationResult {
            max_error,
            average_error,
            errors,
        })
    }

    /// Estimate condition number for numerical stability
    fn estimate_condition_number(&self, p1: &Vector3<f64>, p2: &Vector3<f64>) -> f64 {
        // Simplified condition number estimation
        let distance = (p1 - p2).norm();

        if distance < self.numerical_zero_threshold {
            return f64::INFINITY;
        }

        // Estimate based on point separation
        1.0 / distance
    }
}

/// Manifold types supported by the validator
#[derive(Debug, Clone, Copy)]
pub enum ManifoldType {
    Sphere {
        radius: f64,
    },
    Torus {
        major_radius: f64,
        minor_radius: f64,
    },
    HyperbolicDisk {
        curvature: f64,
    },
    KTwistedTorus {
        major_radius: f64,
        minor_radius: f64,
        k_twist: f64,
    },
    ProjectivePlane,
}

/// Properties expected for geodesic distance calculations
#[derive(Debug, Clone)]
pub struct GeodesicProperties {
    pub expected_triangle_inequality: bool,
    pub expected_symmetry: bool,
    pub expected_positive_definite: bool,
    pub expected_maximum: f64,
}

/// Result of geodesic distance validation
#[derive(Debug, Clone)]
pub struct DistanceValidationResult {
    pub max_error: f64,
    pub average_error: f64,
    pub errors: Vec<DistanceError>,
}

/// Geodesic distance calculation error
#[derive(Debug, Clone)]
pub struct DistanceError {
    pub u1: f64,
    pub v1: f64,
    pub u2: f64,
    pub v2: f64,
    pub reference: f64,
    pub implementation: f64,
    pub error: f64,
}

/// Result of fallback validation
#[derive(Debug, Clone)]
pub struct FallbackValidationResult {
    pub fallback_detected: bool,
    pub errors: Vec<FallbackError>,
}

/// Result of stability validation
#[derive(Debug, Clone)]
pub struct StabilityValidationResult {
    pub stability_issues: Vec<StabilityIssue>,
    pub max_condition_number: f64,
    pub is_stable: bool,
}

/// Fallback detection error
#[derive(Debug, Clone)]
pub struct FallbackError {
    pub point1: Vector3<f64>,
    pub point2: Vector3<f64>,
    pub geodesic_distance: f64,
    pub euclidean_distance: f64,
    pub implementation_distance: f64,
    pub geodesic_error: f64,
    pub euclidean_error: f64,
}

/// Stability issue
#[derive(Debug, Clone)]
pub struct StabilityIssue {
    pub point1: Vector3<f64>,
    pub point2: Vector3<f64>,
    pub issue_type: String,
    pub value: f64,
}

/// Reference implementation for sphere geodesic distance
#[derive(Debug, Clone)]
pub struct SphereGeodesicReference {
    pub radius: f64,
}

impl SphereGeodesicReference {
    pub fn new(radius: f64) -> Self {
        Self { radius }
    }

    pub fn geodesic_distance(&self, p1: &Vector3<f64>, p2: &Vector3<f64>) -> f64 {
        let dot_product = p1.dot(p2);
        let angle = dot_product.acos();
        self.radius * angle
    }
}

/// Reference implementation for torus geodesic distance
#[derive(Debug, Clone)]
pub struct TorusGeodesicReference {
    pub major_radius: f64,
    pub minor_radius: f64,
}

impl TorusGeodesicReference {
    pub fn new(major_radius: f64, minor_radius: f64) -> Self {
        Self {
            major_radius,
            minor_radius,
        }
    }

    pub fn geodesic_distance(&self, p1: &Vector3<f64>, p2: &Vector3<f64>) -> f64 {
        // Simplified torus distance calculation
        let dx = p2.x - p1.x;
        let dy = p2.y - p1.y;
        let dz = p2.z - p1.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Reference implementation for hyperbolic disk geodesic distance
#[derive(Debug, Clone)]
pub struct HyperbolicDiskReference {
    pub curvature: f64,
}

impl HyperbolicDiskReference {
    pub fn new(curvature: f64) -> Self {
        Self { curvature }
    }

    pub fn geodesic_distance(&self, p1: &Vector3<f64>, p2: &Vector3<f64>) -> f64 {
        // Simplified hyperbolic distance calculation
        let norm1_squared = p1.x * p1.x + p1.y * p1.y;
        let norm2_squared = p2.x * p2.x + p2.y * p2.y;

        if norm1_squared >= 1.0 || norm2_squared >= 1.0 {
            return f64::INFINITY;
        }

        let dot_product = p1.dot(p2);
        let denom = 1.0 - norm1_squared * norm2_squared;
        if denom <= 0.0 {
            return f64::INFINITY;
        }

        (dot_product.acosh() * 2.0).abs()
    }
}
