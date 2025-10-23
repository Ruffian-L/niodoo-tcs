//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Mathematical Validation Module
//!
//! This module provides validation for the mathematical components
//! of the Niodoo-Feeling consciousness framework.

use serde::{Deserialize, Serialize};

/// Comprehensive configuration for mathematical validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    // Default torus parameters
    pub torus_major_radius: f64,
    pub torus_minor_radius: f64,
    pub torus_k_twist: f64,

    // All tolerances now configurable
    pub tolerance: f64,
    pub stability_threshold: f64,
    pub numerical_zero_threshold: f64,
    pub test_assertion_threshold: f64,

    // Integration parameters
    pub geodesic_integration_steps: usize,

    // Test parameters (all vectors)
    pub test_k_values: Vec<f64>,
    pub test_length_scales: Vec<f64>,
    pub test_signal_variances: Vec<f64>,
    pub test_major_radii: Vec<f64>,
    pub test_minor_radii: Vec<f64>,
    pub test_k_twists: Vec<f64>,

    // Edge case parameters
    pub small_radius_major: f64,
    pub small_radius_minor: f64,
    pub large_radius_major: f64,
    pub large_radius_minor: f64,
    pub large_k_value: f64,
    pub small_length_scale: f64,
    pub large_length_scale: f64,
    pub small_signal_variance: f64,

    // Test coordinate parameters
    pub test_coordinate_range: f64,
    pub test_coordinate_offset: f64,

    // Expected test values
    pub expected_torus_position_x: f64,
    pub expected_rbf_covariance: f64,
    pub expected_matern_covariance: f64,

    // Novelty detection parameters
    pub novelty_stability_threshold: f64,
    pub novelty_min_value: f64,
    pub novelty_max_value: f64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            // Default torus parameters
            torus_major_radius: 5.0,
            torus_minor_radius: 1.5,
            torus_k_twist: 3.0,

            // All tolerances now configurable
            tolerance: 1e-6,
            stability_threshold: 0.15,
            numerical_zero_threshold: 1e-15,
            test_assertion_threshold: 1e-10,

            // Integration parameters
            geodesic_integration_steps: 100,

            // Test parameters (all vectors)
            test_k_values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            test_length_scales: vec![0.5, 1.0, 2.0, 4.0],
            test_signal_variances: vec![0.5, 1.0, 2.0, 4.0],
            test_major_radii: vec![3.0, 5.0, 7.0, 9.0],
            test_minor_radii: vec![1.0, 1.5, 2.0, 2.5],
            test_k_twists: vec![1.0, 2.0, 3.0, 4.0, 5.0],

            // Edge case parameters
            small_radius_major: 0.1,
            small_radius_minor: 0.01,
            large_radius_major: 1000.0,
            large_radius_minor: 100.0,
            large_k_value: 1000.0,
            small_length_scale: 0.001,
            large_length_scale: 1000.0,
            small_signal_variance: 0.001,

            // Test coordinate parameters
            test_coordinate_range: 4.0,
            test_coordinate_offset: -2.0,

            // Expected test values
            expected_torus_position_x: 6.5, // R + r = 5.0 + 1.5
            expected_rbf_covariance: (-0.5_f64).exp(),
            expected_matern_covariance: 1.0,

            // Novelty detection parameters
            novelty_stability_threshold: 0.15,
            novelty_min_value: 0.0,
            novelty_max_value: 1.0,
        }
    }
}

pub mod gaussian_process_validator;
pub mod geodesic_distance_validator;
pub mod k_twist_geometry_validator;
pub mod novelty_detection_validator;

// Re-export main validator types
pub use k_twist_geometry_validator::{
    KTwistGeometryValidator, ValidationResult as KTwistValidationResult,
};

pub use gaussian_process_validator::{
    GaussianProcessValidator, KernelValidationResult as GPValidationResult,
};

pub use novelty_detection_validator::{
    DistanceBasedReference, InformationTheoreticReference, NoveltyAlgorithm,
    NoveltyDetectionReference, NoveltyDetectionValidator, PredictiveNoveltyReference,
    StatisticalOutlierReference,
};

pub use geodesic_distance_validator::{
    DistanceValidationResult, GeodesicDistanceReference, GeodesicDistanceValidator,
    GeodesicProperties, HyperbolicDiskReference, ManifoldType, SphereGeodesicReference,
    TorusGeodesicReference,
};
