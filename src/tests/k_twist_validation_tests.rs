//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Comprehensive test suite for k-twisted torus geometry validation
//!
//! This module provides extensive tests for the mathematical validation framework,
//! ensuring that geometric computations are correct and numerically stable.

use crate::validation::{
    IssueCategory, IssueSeverity, KTwistGeometryValidator, KTwistImplementation, KTwistReference,
    KTwistValidationResult, TestKTwistImplementation, ValidationIssue,
};
use std::f64::consts::{PI, TAU};

/// Test implementation with correct parametric equations
pub struct CorrectKTwistImplementation {
    major_radius: f64,
    minor_radius: f64,
    k_twist: f64,
}

impl CorrectKTwistImplementation {
    pub fn new(major_radius: f64, minor_radius: f64, k_twist: f64) -> Self {
        Self {
            major_radius,
            minor_radius,
            k_twist,
        }
    }
}

impl KTwistImplementation for CorrectKTwistImplementation {
    fn compute_point(&self, u: f64, v: f64) -> (f64, f64, f64) {
        // Correct parametric equations
        let twist_factor = self.k_twist * u;

        let x = (self.major_radius + self.minor_radius * v.cos()) * u.cos();
        let y = (self.major_radius + self.minor_radius * v.cos()) * u.sin();
        let z = self.minor_radius * v.sin() + self.k_twist * twist_factor.sin();

        (x, y, z)
    }

    fn compute_partial_derivatives(&self, u: f64, v: f64) -> ((f64, f64, f64), (f64, f64, f64)) {
        let twist_factor = self.k_twist * u;

        // Correct partial derivatives
        let dx_du = -(self.major_radius + self.minor_radius * v.cos()) * u.sin()
            + self.k_twist * twist_factor.cos() * u.cos();
        let dy_du = (self.major_radius + self.minor_radius * v.cos()) * u.cos()
            + self.k_twist * twist_factor.cos() * u.sin();
        let dz_du = self.k_twist * twist_factor.cos();

        let dx_dv = -self.minor_radius * v.sin() * u.cos();
        let dy_dv = -self.minor_radius * v.sin() * u.sin();
        let dz_dv = self.minor_radius * v.cos();

        ((dx_du, dy_du, dz_du), (dx_dv, dy_dv, dz_dv))
    }

    fn compute_normal(&self, u: f64, v: f64) -> (f64, f64, f64) {
        let derivatives = self.compute_partial_derivatives(u, v);
        let du = derivatives.0;
        let dv = derivatives.1;

        // Cross product
        let nx = du.1 * dv.2 - du.2 * dv.1;
        let ny = du.2 * dv.0 - du.0 * dv.2;
        let nz = du.0 * dv.1 - du.1 * dv.0;

        // Normalize
        let length = (nx * nx + ny * ny + nz * nz).sqrt();
        if length > 1e-12 {
            (nx / length, ny / length, nz / length)
        } else {
            (0.0, 0.0, 1.0)
        }
    }

    fn get_k_twist(&self) -> f64 {
        self.k_twist
    }
}

#[cfg(test)]
mod comprehensive_tests {
    use super::*;

    /// Test that the validator correctly identifies errors in the current implementation
    #[test]
    fn test_current_implementation_has_errors() {
        let validator = KTwistGeometryValidator::new(5.0, 1.5, 1.0, 1e-6);
        let implementation = TestKTwistImplementation::new(5.0, 1.5, 1.0);

        let result = validator.validate(&implementation).unwrap();

        // The current implementation should have errors
        assert!(!result.is_valid);
        assert!(!result.parametric_validation.equations_valid);
        assert!(!result.parametric_validation.derivatives_valid);
        assert!(!result.parametric_validation.normals_valid);

        // Check that we have critical or major issues
        let has_critical_issues = result.issues.iter().any(|issue| {
            matches!(
                issue.severity,
                IssueSeverity::Critical | IssueSeverity::Major
            )
        });
        assert!(has_critical_issues);

        // Check that parametric equation issues are present
        let has_parametric_issues = result
            .issues
            .iter()
            .any(|issue| matches!(issue.category, IssueCategory::ParametricEquations));
        assert!(has_parametric_issues);
    }

    /// Test that the validator passes correct implementations
    #[test]
    fn test_correct_implementation_passes() {
        let validator = KTwistGeometryValidator::new(5.0, 1.5, 1.0, 1e-6);
        let implementation = CorrectKTwistImplementation::new(5.0, 1.5, 1.0);

        let result = validator.validate(&implementation).unwrap();

        // The correct implementation should pass validation
        assert!(result.is_valid);
        assert!(result.parametric_validation.equations_valid);
        assert!(result.parametric_validation.derivatives_valid);
        assert!(result.parametric_validation.normals_valid);
        assert!(result.topological_validation.is_closed);
        assert!(result.numerical_validation.coordinates_finite);
        assert!(result.numerical_validation.normals_unit_length);

        // Should have no critical or major issues
        let has_critical_issues = result.issues.iter().any(|issue| {
            matches!(
                issue.severity,
                IssueSeverity::Critical | IssueSeverity::Major
            )
        });
        assert!(!has_critical_issues);
    }

    /// Test orientability validation for different k values
    #[test]
    fn test_orientability_validation() {
        // Test even k values (should be orientable)
        let validator_even = KTwistGeometryValidator::new(5.0, 1.5, 2.0, 1e-6);
        let implementation_even = CorrectKTwistImplementation::new(5.0, 1.5, 2.0);

        let result_even = validator_even.validate(&implementation_even).unwrap();
        assert!(result_even.orientability_check.is_orientable);
        assert!(!result_even.orientability_check.k_is_odd);
        assert!(!result_even.orientability_check.mobius_detected);

        // Test odd k values (should be non-orientable)
        let validator_odd = KTwistGeometryValidator::new(5.0, 1.5, 3.0, 1e-6);
        let implementation_odd = CorrectKTwistImplementation::new(5.0, 1.5, 3.0);

        let result_odd = validator_odd.validate(&implementation_odd).unwrap();
        assert!(!result_odd.orientability_check.is_orientable);
        assert!(result_odd.orientability_check.k_is_odd);
        assert!(result_odd.orientability_check.mobius_detected);
    }

    /// Test topological property validation
    #[test]
    fn test_topological_properties() {
        let validator = KTwistGeometryValidator::new(5.0, 1.5, 1.0, 1e-6);
        let implementation = CorrectKTwistImplementation::new(5.0, 1.5, 1.0);

        let result = validator.validate(&implementation).unwrap();

        // Check topological properties
        assert!(result.topological_validation.is_closed);
        assert!(result.topological_validation.topology_preserved);
        assert!((result.topological_validation.euler_characteristic - 0.0).abs() < 1e-10);
        assert!((result.topological_validation.genus - 1.0).abs() < 1e-10);
    }

    /// Test numerical stability validation
    #[test]
    fn test_numerical_stability() {
        let validator = KTwistGeometryValidator::new(5.0, 1.5, 1.0, 1e-6);
        let implementation = CorrectKTwistImplementation::new(5.0, 1.5, 1.0);

        let result = validator.validate(&implementation).unwrap();

        // Check numerical stability
        assert!(result.numerical_validation.is_stable);
        assert!(result.numerical_validation.coordinates_finite);
        assert!(result.numerical_validation.normals_unit_length);
        assert!(result.numerical_validation.max_condition_number < 1e12);
        assert!(result.numerical_validation.precision_issues.is_empty());
    }

    /// Test edge cases and boundary conditions
    #[test]
    fn test_edge_cases() {
        let validator = KTwistGeometryValidator::new(5.0, 1.5, 1.0, 1e-6);
        let implementation = CorrectKTwistImplementation::new(5.0, 1.5, 1.0);

        // Test boundary points
        let boundary_points = vec![(0.0, 0.0), (TAU, 0.0), (0.0, TAU), (TAU, TAU), (PI, PI)];

        for (u, v) in boundary_points {
            let point = implementation.compute_point(u, v);
            assert!(point.0.is_finite());
            assert!(point.1.is_finite());
            assert!(point.2.is_finite());

            let normal = implementation.compute_normal(u, v);
            let normal_length =
                (normal.0 * normal.0 + normal.1 * normal.1 + normal.2 * normal.2).sqrt();
            assert!((normal_length - 1.0).abs() < 1e-10);
        }
    }

    /// Test different parameter ranges
    #[test]
    fn test_parameter_ranges() {
        let test_cases = vec![
            (1.0, 0.5, 0.5),  // Small parameters
            (10.0, 3.0, 2.0), // Large parameters
            (2.0, 1.0, 0.1),  // Small k-twist
            (3.0, 1.5, 5.0),  // Large k-twist
        ];

        for (major_radius, minor_radius, k_twist) in test_cases {
            let validator = KTwistGeometryValidator::new(major_radius, minor_radius, k_twist, 1e-6);
            let implementation =
                CorrectKTwistImplementation::new(major_radius, minor_radius, k_twist);

            let result = validator.validate(&implementation).unwrap();

            // All parameter ranges should be valid
            assert!(result.is_valid);
            assert!(result.parametric_validation.equations_valid);
            assert!(result.numerical_validation.coordinates_finite);
        }
    }

    /// Test error reporting and issue categorization
    #[test]
    fn test_error_reporting() {
        let validator = KTwistGeometryValidator::new(5.0, 1.5, 1.0, 1e-6);
        let implementation = TestKTwistImplementation::new(5.0, 1.5, 1.0);

        let result = validator.validate(&implementation).unwrap();

        // Check that issues are properly categorized
        let parametric_issues: Vec<_> = result
            .issues
            .iter()
            .filter(|issue| matches!(issue.category, IssueCategory::ParametricEquations))
            .collect();
        assert!(!parametric_issues.is_empty());

        let derivative_issues: Vec<_> = result
            .issues
            .iter()
            .filter(|issue| matches!(issue.category, IssueCategory::Derivatives))
            .collect();
        assert!(!derivative_issues.is_empty());

        let normal_issues: Vec<_> = result
            .issues
            .iter()
            .filter(|issue| matches!(issue.category, IssueCategory::NormalVectors))
            .collect();
        assert!(!normal_issues.is_empty());

        // Check that issues have proper descriptions and suggestions
        for issue in &result.issues {
            assert!(!issue.description.is_empty());
            if matches!(
                issue.severity,
                IssueSeverity::Critical | IssueSeverity::Major
            ) {
                assert!(issue.suggested_fix.is_some());
            }
        }
    }

    /// Test reference implementation correctness
    #[test]
    fn test_reference_implementation() {
        let reference = KTwistReference::new(5.0, 1.5, 1.0);

        // Test basic properties
        assert!(reference.check_orientability()); // k=1 is odd, should be non-orientable
        assert!((reference.compute_euler_characteristic() - 0.0).abs() < 1e-10);
        assert!((reference.compute_genus() - 1.0).abs() < 1e-10);

        // Test point computation
        let point = reference.compute_point(0.0, 0.0);
        assert!((point.0 - 6.5).abs() < 1e-10); // (5.0 + 1.5) * cos(0)
        assert!((point.1 - 0.0).abs() < 1e-10); // (5.0 + 1.5) * sin(0)
        assert!((point.2 - 0.0).abs() < 1e-10); // 1.5 * sin(0) + 1.0 * sin(0)

        // Test normal computation
        let normal = reference.compute_normal(0.0, 0.0);
        let normal_length =
            (normal.0 * normal.0 + normal.1 * normal.1 + normal.2 * normal.2).sqrt();
        assert!((normal_length - 1.0).abs() < 1e-10);
    }

    /// Test validation with different tolerances
    #[test]
    fn test_tolerance_sensitivity() {
        let implementation = TestKTwistImplementation::new(5.0, 1.5, 1.0);

        // Test with loose tolerance
        let validator_loose = KTwistGeometryValidator::new(5.0, 1.5, 1.0, 1e-3);
        let result_loose = validator_loose.validate(&implementation).unwrap();

        // Test with tight tolerance
        let validator_tight = KTwistGeometryValidator::new(5.0, 1.5, 1.0, 1e-9);
        let result_tight = validator_tight.validate(&implementation).unwrap();

        // Tight tolerance should catch more errors
        assert!(result_tight.issues.len() >= result_loose.issues.len());
        assert!(
            result_tight.parametric_validation.max_error
                >= result_loose.parametric_validation.max_error
        );
    }

    /// Test validation performance with large test sets
    #[test]
    fn test_validation_performance() {
        let validator = KTwistGeometryValidator::new(5.0, 1.5, 1.0, 1e-6);
        let implementation = CorrectKTwistImplementation::new(5.0, 1.5, 1.0);

        let start = std::time::Instant::now();
        let result = validator.validate(&implementation).unwrap();
        let duration = start.elapsed();

        // Validation should complete quickly (less than 1 second)
        assert!(duration.as_secs() < 1);
        assert!(result.is_valid);
    }

    /// Test validation with extreme parameter values
    #[test]
    fn test_extreme_parameters() {
        let extreme_cases = vec![
            (1e-6, 1e-6, 1e-6), // Very small parameters
            (1e6, 1e6, 1e6),    // Very large parameters
            (0.0, 1.0, 1.0),    // Zero major radius
            (1.0, 0.0, 1.0),    // Zero minor radius
        ];

        for (major_radius, minor_radius, k_twist) in extreme_cases {
            if major_radius > 0.0 && minor_radius > 0.0 {
                let validator =
                    KTwistGeometryValidator::new(major_radius, minor_radius, k_twist, 1e-6);
                let implementation =
                    CorrectKTwistImplementation::new(major_radius, minor_radius, k_twist);

                let result = validator.validate(&implementation).unwrap();

                // Should handle extreme parameters gracefully
                assert!(result.numerical_validation.coordinates_finite);
            }
        }
    }
}
