//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Comprehensive test suite for geodesic distance validation
//!
//! This module provides extensive tests for the geodesic distance validation framework,
//! ensuring that distance calculations on various manifolds are mathematically correct
//! and numerically stable.

use crate::validation::{
    GeodesicDistanceReference, GeodesicDistanceValidator, GeodesicProperties,
    GeodesicValidationResult, HyperbolicDiskReference, ManifoldType, MultiManifoldValidationResult,
    SphereGeodesicReference, TorusGeodesicReference,
};
use nalgebra::Vector3;

/// Test implementation with correct geodesic distance calculation for sphere
pub struct CorrectSphereImplementation;

impl CorrectSphereImplementation {
    pub fn new() -> Self {
        Self
    }

    pub fn geodesic_distance(&self, p1: &Vector3<f64>, p2: &Vector3<f64>) -> f64 {
        // Correct sphere geodesic distance
        let p1_norm = p1 / p1.norm();
        let p2_norm = p2 / p2.norm();

        let cos_angle = p1_norm.dot(&p2_norm).min(1.0).max(-1.0);
        let angle = cos_angle.acos();

        angle // For unit sphere, radius = 1
    }
}

/// Test implementation with correct geodesic distance calculation for torus
pub struct CorrectTorusImplementation;

impl CorrectTorusImplementation {
    pub fn new() -> Self {
        Self
    }

    pub fn geodesic_distance(&self, p1: &Vector3<f64>, p2: &Vector3<f64>) -> f64 {
        // Simplified torus geodesic distance
        let torus_ref = TorusGeodesicReference::new(2.0, 1.0);
        torus_ref.geodesic_distance(p1, p2)
    }
}

/// Test implementation with correct geodesic distance calculation for hyperbolic disk
pub struct CorrectHyperbolicImplementation;

impl CorrectHyperbolicImplementation {
    pub fn new() -> Self {
        Self
    }

    pub fn geodesic_distance(&self, p1: &Vector3<f64>, p2: &Vector3<f64>) -> f64 {
        // Correct hyperbolic distance
        let hyp_ref = HyperbolicDiskReference::new(-1.0);
        hyp_ref.geodesic_distance(p1, p2)
    }
}

/// Test implementation with incorrect geodesic distance (for testing validation)
pub struct IncorrectGeodesicImplementation;

impl IncorrectGeodesicImplementation {
    pub fn new() -> Self {
        Self
    }

    pub fn geodesic_distance(&self, p1: &Vector3<f64>, p2: &Vector3<f64>) -> f64 {
        // Incorrect implementation - uses Euclidean distance instead of geodesic
        (p1 - p2).norm()
    }
}

#[cfg(test)]
mod comprehensive_tests {
    use super::*;

    /// Test that the validator correctly identifies errors in incorrect implementations
    #[test]
    fn test_incorrect_implementation_has_errors() {
        let validator = GeodesicDistanceValidator::for_sphere(1.0, 1e-6);
        let implementation = IncorrectGeodesicImplementation::new();

        let result = validator
            .validate_sphere_geodesic(|p1, p2| implementation.geodesic_distance(p1, p2))
            .unwrap();

        // The incorrect implementation should have errors
        assert!(!result.convergence_achieved);
        assert!(result.max_error > 1e-6);

        // Check that we have errors
        assert!(!result.errors.is_empty());
    }

    /// Test that the validator passes correct implementations
    #[test]
    fn test_correct_implementation_passes() {
        let validator = GeodesicDistanceValidator::for_sphere(1.0, 1e-6);
        let implementation = CorrectSphereImplementation::new();

        let result = validator
            .validate_sphere_geodesic(|p1, p2| implementation.geodesic_distance(p1, p2))
            .unwrap();

        // The correct implementation should pass validation
        assert!(result.convergence_achieved);
        assert!(result.numerical_stability);
        assert!(result.max_error < 1e-10);
        assert!(result.errors.is_empty());
    }

    /// Test sphere geodesic properties validation
    #[test]
    fn test_sphere_geodesic_properties() {
        let validator = GeodesicDistanceValidator::for_sphere(1.0, 1e-6);
        let implementation = CorrectSphereImplementation::new();

        let result = validator
            .validate_sphere_geodesic(|p1, p2| implementation.geodesic_distance(p1, p2))
            .unwrap();

        // Check geodesic properties
        assert!(result.convergence_achieved);
        assert!(result.numerical_stability);

        // Test triangle inequality for a few points
        let p1 = Vector3::new(1.0, 0.0, 0.0);
        let p2 = Vector3::new(0.0, 1.0, 0.0);
        let p3 = Vector3::new(0.0, 0.0, 1.0);

        let d12 = implementation.geodesic_distance(&p1, &p2);
        let d13 = implementation.geodesic_distance(&p1, &p3);
        let d23 = implementation.geodesic_distance(&p2, &p3);

        // Triangle inequality should hold
        assert!(d12 + d13 >= d23);
        assert!(d12 + d23 >= d13);
        assert!(d13 + d23 >= d12);
    }

    /// Test torus geodesic properties validation
    #[test]
    fn test_torus_geodesic_properties() {
        let validator = GeodesicDistanceValidator::for_torus(2.0, 1.0, 1e-6);
        let implementation = CorrectTorusImplementation::new();

        let result = validator
            .validate_torus_geodesic(|p1, p2| implementation.geodesic_distance(p1, p2))
            .unwrap();

        // Check torus properties
        assert!(result.convergence_achieved);
        assert!(result.numerical_stability);

        // Test symmetry
        let p1 = Vector3::new(3.0, 0.0, 0.0);
        let p2 = Vector3::new(-3.0, 0.0, 0.0);

        let d12 = implementation.geodesic_distance(&p1, &p2);
        let d21 = implementation.geodesic_distance(&p2, &p1);

        assert!((d12 - d21).abs() < 1e-10);
    }

    /// Test hyperbolic geodesic properties validation
    #[test]
    fn test_hyperbolic_geodesic_properties() {
        let validator = GeodesicDistanceValidator::for_hyperbolic_disk(-1.0, 1e-6);
        let implementation = CorrectHyperbolicImplementation::new();

        let result = validator
            .validate_hyperbolic_geodesic(|p1, p2| implementation.geodesic_distance(p1, p2))
            .unwrap();

        // Check hyperbolic properties
        assert!(result.convergence_achieved);
        assert!(result.numerical_stability);

        // Test that center to boundary is finite
        let center = Vector3::new(0.0, 0.0, 0.0);
        let boundary = Vector3::new(0.5, 0.0, 0.0);

        let distance = implementation.geodesic_distance(&center, &boundary);
        assert!(distance.is_finite());
        assert!(distance > 0.0);
    }

    /// Test numerical stability validation
    #[test]
    fn test_numerical_stability_validation() {
        let validator = GeodesicDistanceValidator::for_sphere(1.0, 1e-6);
        let implementation = CorrectSphereImplementation::new();

        let result = validator
            .validate_numerical_stability(|p1, p2| Ok(implementation.geodesic_distance(p1, p2)))
            .unwrap();

        // Should be numerically stable
        assert!(result.is_stable);
        assert!(result.stability_issues.is_empty());
        assert!(result.max_condition_number.is_finite());
    }

    /// Test Euclidean fallback detection
    #[test]
    fn test_euclidean_fallback_detection() {
        let validator = GeodesicDistanceValidator::for_sphere(1.0, 1e-6);
        let implementation = IncorrectGeodesicImplementation::new();

        let result = validator
            .check_euclidean_fallback(|p1, p2| implementation.geodesic_distance(p1, p2))
            .unwrap();

        // Should detect Euclidean fallback
        assert!(result.fallback_detected);
        assert!(!result.errors.is_empty());
    }

    /// Test multi-manifold validation
    #[test]
    fn test_multi_manifold_validation() {
        let validator = GeodesicDistanceValidator::for_sphere(1.0, 1e-6);

        let result = validator
            .validate_geodesic_distance(|p1, p2| validator.reference.geodesic_distance(p1, p2))
            .unwrap();

        // Should validate all manifolds
        assert!(result.convergence_achieved);
        assert!(result.all_manifolds_valid);
        assert!(result.total_errors == 0);
        assert!(result.manifold_results.len() > 0);
    }

    /// Test different manifold types
    #[test]
    fn test_different_manifold_types() {
        let test_cases = vec![
            (ManifoldType::Sphere { radius: 1.0 }, "Sphere"),
            (
                ManifoldType::Torus {
                    major_radius: 2.0,
                    minor_radius: 1.0,
                },
                "Torus",
            ),
            (
                ManifoldType::HyperbolicDisk { curvature: -1.0 },
                "HyperbolicDisk",
            ),
            (ManifoldType::ProjectivePlane, "ProjectivePlane"),
            (
                ManifoldType::KTwistedTorus {
                    major_radius: 5.0,
                    minor_radius: 1.5,
                    k_twist: 1.0,
                },
                "KTwistedTorus",
            ),
        ];

        for (manifold_type, name) in test_cases {
            let validator = GeodesicDistanceValidator::new(manifold_type, 1e-6);

            // Test that the validator can be created and used
            let result = validator
                .validate_geodesic_distance(|p1, p2| validator.reference.geodesic_distance(p1, p2))
                .unwrap();

            assert!(result.convergence_achieved);
            assert!(result.all_manifolds_valid);
        }
    }

    /// Test edge cases and boundary conditions
    #[test]
    fn test_edge_cases() {
        let validator = GeodesicDistanceValidator::for_sphere(1.0, 1e-6);
        let implementation = CorrectSphereImplementation::new();

        // Test with very close points
        let p1 = Vector3::new(1.0, 0.0, 0.001);
        let p2 = Vector3::new(1.0, 0.0, 0.002);

        let distance = implementation.geodesic_distance(&p1, &p2);
        assert!(distance > 0.0);
        assert!(distance < 0.01);

        // Test with antipodal points
        let p1 = Vector3::new(1.0, 0.0, 0.0);
        let p2 = Vector3::new(-1.0, 0.0, 0.0);

        let distance = implementation.geodesic_distance(&p1, &p2);
        assert!((distance - std::f64::consts::PI).abs() < 1e-10);

        // Test with same point
        let p1 = Vector3::new(1.0, 0.0, 0.0);
        let p2 = Vector3::new(1.0, 0.0, 0.0);

        let distance = implementation.geodesic_distance(&p1, &p2);
        assert!(distance < 1e-10);
    }

    /// Test reference implementation correctness
    #[test]
    fn test_reference_implementation() {
        let sphere_ref = SphereGeodesicReference::new(1.0);

        let x1 = Vector3::new(1.0, 0.0, 0.0);
        let x2 = Vector3::new(0.0, 1.0, 0.0);

        // Test geodesic distance
        let distance = sphere_ref.geodesic_distance(&x1, &x2);
        let expected = std::f64::consts::FRAC_PI_2; // 90 degrees

        assert!((distance - expected).abs() < 1e-10);

        // Test vector formula
        let distance_vector = sphere_ref.geodesic_distance_vector(&x1, &x2);
        assert!((distance_vector - expected).abs() < 1e-10);

        // Test same point
        let same_distance = sphere_ref.geodesic_distance(&x1, &x1);
        assert!(same_distance < 1e-10);
    }

    /// Test torus reference implementation
    #[test]
    fn test_torus_reference_implementation() {
        let torus_ref = TorusGeodesicReference::new(2.0, 1.0);

        let p1 = Vector3::new(3.0, 0.0, 0.0); // Major radius point
        let p2 = Vector3::new(-3.0, 0.0, 0.0); // Opposite major radius point

        let distance = torus_ref.geodesic_distance(&p1, &p2);

        // Should be finite and positive
        assert!(distance.is_finite());
        assert!(distance > 0.0);

        // Should be symmetric
        let distance_reverse = torus_ref.geodesic_distance(&p2, &p1);
        assert!((distance - distance_reverse).abs() < 1e-10);
    }

    /// Test hyperbolic reference implementation
    #[test]
    fn test_hyperbolic_reference_implementation() {
        let hyp_ref = HyperbolicDiskReference::new(-1.0);

        let center = Vector3::new(0.0, 0.0, 0.0);
        let boundary = Vector3::new(0.5, 0.0, 0.0);

        let distance = hyp_ref.geodesic_distance(&center, &boundary);

        // Should be finite and positive
        assert!(distance.is_finite());
        assert!(distance > 0.0);

        // Test points outside disk should return infinity
        let outside = Vector3::new(1.5, 0.0, 0.0);
        let outside_distance = hyp_ref.geodesic_distance(&center, &outside);
        assert!(outside_distance.is_infinite());
    }

    /// Test validation with different tolerances
    #[test]
    fn test_tolerance_sensitivity() {
        let validator = GeodesicDistanceValidator::for_sphere(1.0, 1e-3);
        let implementation = IncorrectGeodesicImplementation::new();

        let result_loose = validator
            .validate_sphere_geodesic(|p1, p2| implementation.geodesic_distance(p1, p2))
            .unwrap();

        let validator_tight = GeodesicDistanceValidator::for_sphere(1.0, 1e-9);
        let result_tight = validator_tight
            .validate_sphere_geodesic(|p1, p2| implementation.geodesic_distance(p1, p2))
            .unwrap();

        // Tight tolerance should catch more errors or have higher error
        assert!(result_tight.max_error >= result_loose.max_error);
    }

    /// Test validation performance with large test sets
    #[test]
    fn test_validation_performance() {
        let validator = GeodesicDistanceValidator::for_sphere(1.0, 1e-6);
        let implementation = CorrectSphereImplementation::new();

        let start = std::time::Instant::now();
        let result = validator
            .validate_sphere_geodesic(|p1, p2| implementation.geodesic_distance(p1, p2))
            .unwrap();
        let duration = start.elapsed();

        // Validation should complete quickly (less than 1 second)
        assert!(duration.as_secs() < 1);
        assert!(result.convergence_achieved);
    }

    /// Test with extreme parameter values
    #[test]
    fn test_extreme_parameters() {
        let validator = GeodesicDistanceValidator::for_sphere(1.0, 1e-6);
        let implementation = CorrectSphereImplementation::new();

        // Test with very small sphere
        let small_sphere_validator = GeodesicDistanceValidator::for_sphere(0.1, 1e-6);
        let result_small = small_sphere_validator
            .validate_sphere_geodesic(|p1, p2| implementation.geodesic_distance(p1, p2))
            .unwrap();

        assert!(result_small.convergence_achieved);

        // Test with very large sphere
        let large_sphere_validator = GeodesicDistanceValidator::for_sphere(100.0, 1e-6);
        let result_large = large_sphere_validator
            .validate_sphere_geodesic(|p1, p2| implementation.geodesic_distance(p1, p2))
            .unwrap();

        assert!(result_large.convergence_achieved);
    }

    /// Test error reporting and issue categorization
    #[test]
    fn test_error_reporting() {
        let validator = GeodesicDistanceValidator::for_sphere(1.0, 1e-6);
        let implementation = IncorrectGeodesicImplementation::new();

        let result = validator
            .validate_sphere_geodesic(|p1, p2| implementation.geodesic_distance(p1, p2))
            .unwrap();

        // Check that errors are properly reported
        assert!(!result.errors.is_empty());

        for error in &result.errors {
            assert!(error.error > 0.0);
            assert!(error.reference_distance >= 0.0);
            assert!(error.implementation_distance >= 0.0);
            assert!(!error.manifold_type.is_empty());
        }
    }

    /// Test geodesic properties validation
    #[test]
    fn test_geodesic_properties_validation() {
        let validator = GeodesicDistanceValidator::for_sphere(1.0, 1e-6);
        let implementation = CorrectSphereImplementation::new();

        let result = validator
            .validate_sphere_geodesic(|p1, p2| implementation.geodesic_distance(p1, p2))
            .unwrap();

        // Check that geodesic properties are validated
        assert!(result.convergence_achieved);

        // Test triangle inequality with specific points
        let p1 = Vector3::new(1.0, 0.0, 0.0);
        let p2 = Vector3::new(0.707, 0.707, 0.0); // 45 degrees
        let p3 = Vector3::new(0.0, 0.0, 1.0);

        let d12 = implementation.geodesic_distance(&p1, &p2);
        let d13 = implementation.geodesic_distance(&p1, &p3);
        let d23 = implementation.geodesic_distance(&p2, &p3);

        // Triangle inequality should hold for geodesic distances
        assert!(d12 + d13 >= d23);
        assert!(d12 + d23 >= d13);
        assert!(d13 + d23 >= d12);
    }

    /// Test manifold-specific validations
    #[test]
    fn test_manifold_specific_validations() {
        // Test sphere-specific validation
        let sphere_validator = GeodesicDistanceValidator::for_sphere(1.0, 1e-6);
        let sphere_result = sphere_validator
            .validate_geodesic_distance(|p1, p2| {
                sphere_validator.reference.geodesic_distance(p1, p2)
            })
            .unwrap();

        assert!(sphere_result.convergence_achieved);

        // Test torus-specific validation
        let torus_validator = GeodesicDistanceValidator::for_torus(2.0, 1.0, 1e-6);
        let torus_result = torus_validator
            .validate_geodesic_distance(|p1, p2| {
                torus_validator.reference.geodesic_distance(p1, p2)
            })
            .unwrap();

        assert!(torus_result.convergence_achieved);

        // Test hyperbolic-specific validation
        let hyp_validator = GeodesicDistanceValidator::for_hyperbolic_disk(-1.0, 1e-6);
        let hyp_result = hyp_validator
            .validate_geodesic_distance(|p1, p2| hyp_validator.reference.geodesic_distance(p1, p2))
            .unwrap();

        assert!(hyp_result.convergence_achieved);
    }

    /// Test with degenerate cases
    #[test]
    fn test_degenerate_cases() {
        let validator = GeodesicDistanceValidator::for_sphere(1.0, 1e-6);
        let implementation = CorrectSphereImplementation::new();

        // Test with points that might cause numerical issues
        let problematic_cases = vec![
            (Vector3::new(1.0, 0.0, 1e-15), Vector3::new(1.0, 0.0, 2e-15)), // Very close points
            (Vector3::new(1.0, 0.0, 0.0), Vector3::new(1.0, 1e-15, 0.0)), // Near coordinate singularity
        ];

        for (p1, p2) in problematic_cases {
            let distance = implementation.geodesic_distance(&p1, &p2);
            assert!(distance.is_finite());
            assert!(distance >= 0.0);
        }
    }

    /// Test validation framework extensibility
    #[test]
    fn test_validation_framework_extensibility() {
        // Test that new manifold types can be easily added
        let custom_manifold = ManifoldType::Sphere { radius: 2.0 };
        let validator = GeodesicDistanceValidator::new(custom_manifold, 1e-6);

        let result = validator
            .validate_geodesic_distance(|p1, p2| validator.reference.geodesic_distance(p1, p2))
            .unwrap();

        assert!(result.convergence_achieved);
        assert!(result.all_manifolds_valid);
    }
}
