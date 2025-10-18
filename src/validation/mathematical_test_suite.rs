//! Mathematical Test Suite
//! 
//! Phase 4: Mathematical Validation Module
//! Comprehensive test cases for each validator including edge cases,
//! boundary conditions, and regression tests.

use anyhow::Result;
use std::f64::consts::{PI, TAU};

use crate::validation::k_twist_geometry_validator::*;
use crate::validation::gaussian_process_validator::*;
use crate::validation::geodesic_distance_validator::*;
use crate::validation::novelty_detection_validator::*;

/// Comprehensive test suite for mathematical validation
#[derive(Debug, Clone)]
pub struct MathematicalTestSuite {
    /// Validation configuration containing all configurable parameters
    pub config: ValidationConfig,
    /// Number of test points for grid-based tests
    pub grid_resolution: usize,
    /// Enable edge case testing
    pub test_edge_cases: bool,
    /// Enable boundary condition testing
    pub test_boundary_conditions: bool,
    /// Enable regression testing
    pub test_regression: bool,
}

impl Default for MathematicalTestSuite {
    fn default() -> Self {
        Self {
            config: ValidationConfig::default(),
            grid_resolution: 20,
            test_edge_cases: true,
            test_boundary_conditions: true,
            test_regression: true,
        }
    }
}

impl MathematicalTestSuite {
    /// Run all mathematical validation tests
    pub fn run_all_tests(&self) -> Result<TestSuiteResult> {
        let mut results = TestSuiteResult::new();
        
        // Run K-Twist geometry tests
        results.k_twist_results = self.run_k_twist_tests()?;
        
        // Run Gaussian process tests
        results.gp_results = self.run_gaussian_process_tests()?;
        
        // Run geodesic distance tests
        results.geodesic_results = self.run_geodesic_distance_tests()?;
        
        // Run novelty detection tests
        results.novelty_results = self.run_novelty_detection_tests()?;
        
        // Calculate overall results
        results.calculate_overall_results();
        
        Ok(results)
    }
    
    /// Run K-Twist geometry validation tests
    fn run_k_twist_tests(&self) -> Result<KTwistTestResults> {
        let mut results = KTwistTestResults::new();
        
        // Test different k values
        let k_values = self.config.test_k_values.clone();
        
        for k in k_values {
            let validator = KTwistGeometryValidator::new_with_threshold(
                self.config.torus_major_radius,
                self.config.torus_minor_radius,
                k,
                self.config.tolerance,
                self.config.numerical_zero_threshold,
            );
            
            // Test position validation
            let position_result = validator.validate_positions(|u, v| {
                validator.reference.calculate_position(u, v)
            })?;
            results.position_tests.push(PositionTestResult {
                k_twist: k,
                max_error: position_result.max_error,
                average_error: position_result.average_error,
                error_count: position_result.position_errors.len(),
                passed: position_result.max_error < self.config.tolerance,
            });
            
            // Test normal validation
            let normal_result = validator.validate_normals(|u, v| {
                validator.reference.calculate_normal(u, v)
            })?;
            results.normal_tests.push(NormalTestResult {
                k_twist: k,
                max_error: normal_result.max_error,
                average_error: normal_result.average_error,
                error_count: normal_result.normal_errors.len(),
                passed: normal_result.max_error < self.config.tolerance,
            });
            
            // Test orientability
            let orientability_result = validator.validate_orientability()?;
            results.orientability_tests.push(OrientabilityTestResult {
                k_twist: k,
                is_non_orientable: orientability_result.orientability_test
                    .as_ref()
                    .map(|t| t.is_non_orientable)
                    .unwrap_or(false),
                expected_non_orientable: (k as i32) % 2 != 0,
                passed: orientability_result.orientability_test
                    .as_ref()
                    .map(|t| t.is_non_orientable == ((k as i32) % 2 != 0))
                    .unwrap_or(false),
            });
        }
        
        // Test edge cases if enabled
        if self.test_edge_cases {
            results.edge_case_tests = self.run_k_twist_edge_cases()?;
        }
        
        // Test boundary conditions if enabled
        if self.test_boundary_conditions {
            results.boundary_tests = self.run_k_twist_boundary_cases()?;
        }
        
        Ok(results)
    }
    
    /// Run Gaussian process validation tests
    fn run_gaussian_process_tests(&self) -> Result<GaussianProcessTestResults> {
        let mut results = GaussianProcessTestResults::new();
        
        // Test different hyperparameter combinations
        let length_scales = self.config.test_length_scales.clone();
        let signal_variances = self.config.test_signal_variances.clone();
        
        for length_scale in &length_scales {
            for signal_variance in &signal_variances {
                let validator = GaussianProcessValidator::new(
                    *length_scale,
                    *signal_variance,
                    1.5, // Matérn-3/2
                    self.config.tolerance,
                );
                
                // Test RBF kernel
                let rbf_result = validator.validate_rbf_kernel(|x1, x2| {
                    validator.rbf_reference.covariance(x1, x2)
                })?;
                results.rbf_tests.push(RBFTestResult {
                    length_scale: *length_scale,
                    signal_variance: *signal_variance,
                    max_error: rbf_result.max_error,
                    average_error: rbf_result.average_error,
                    error_count: rbf_result.errors.len(),
                    passed: rbf_result.max_error < self.config.tolerance,
                });
                
                // Test Matérn kernel
                let matern_result = validator.validate_matern_kernel(|x1, x2| {
                    validator.matern_reference.covariance(x1, x2)
                })?;
                results.matern_tests.push(MaternTestResult {
                    length_scale: *length_scale,
                    signal_variance: *signal_variance,
                    max_error: matern_result.max_error,
                    average_error: matern_result.average_error,
                    error_count: matern_result.errors.len(),
                    passed: matern_result.max_error < self.config.tolerance,
                });
                
                // Test covariance matrix
                let cov_result = validator.validate_covariance_matrix(|matrix| {
                    matrix.to_owned()
                })?;
                results.covariance_tests.push(CovarianceTestResult {
                    length_scale: *length_scale,
                    signal_variance: *signal_variance,
                    is_symmetric: cov_result.is_symmetric,
                    is_positive_definite: cov_result.is_positive_definite,
                    diagonal_valid: cov_result.diagonal_elements_valid,
                    max_error: cov_result.max_error,
                    passed: cov_result.is_symmetric && cov_result.is_positive_definite && cov_result.diagonal_elements_valid,
                });
            }
        }
        
        // Test edge cases if enabled
        if self.test_edge_cases {
            results.edge_case_tests = self.run_gp_edge_cases()?;
        }
        
        Ok(results)
    }
    
    /// Run geodesic distance validation tests
    fn run_geodesic_distance_tests(&self) -> Result<GeodesicDistanceTestResults> {
        let mut results = GeodesicDistanceTestResults::new();
        
        // Test different torus configurations
        let major_radii = self.config.test_major_radii.clone();
        let minor_radii = self.config.test_minor_radii.clone();
        let k_twists = self.config.test_k_twists.clone();
        
        for major_radius in &major_radii {
            for minor_radius in &minor_radii {
                for k_twist in &k_twists {
                    let validator = GeodesicDistanceValidator::new_with_steps_and_threshold(
                        *major_radius,
                        *minor_radius,
                        *k_twist,
                        self.config.tolerance,
                        self.config.geodesic_integration_steps,
                        self.config.numerical_zero_threshold,
                    );
                    
                    // Test geodesic distance
                    let distance_result = validator.validate_geodesic_distance(|u1, v1, u2, v2| {
                        validator.reference.geodesic_distance_with_steps(u1, v1, u2, v2, self.config.geodesic_integration_steps)
                    })?;
                    results.distance_tests.push(DistanceTestResult {
                        major_radius: *major_radius,
                        minor_radius: *minor_radius,
                        k_twist: *k_twist,
                        max_error: distance_result.max_error,
                        average_error: distance_result.average_error,
                        error_count: distance_result.errors.len(),
                        passed: distance_result.max_error < self.config.tolerance,
                    });
                    
                    // Test fallback detection
                    let fallback_result = validator.check_euclidean_fallback(|u1, v1, u2, v2| {
                        Ok(validator.reference.euclidean_distance_fallback(u1, v1, u2, v2))
                    })?;
                    results.fallback_tests.push(FallbackTestResult {
                        major_radius: *major_radius,
                        minor_radius: *minor_radius,
                        k_twist: *k_twist,
                        fallback_detected: fallback_result.fallback_detected,
                        expected_detected: true, // We're using Euclidean fallback
                        passed: fallback_result.fallback_detected,
                    });
                    
                    // Test numerical stability
                    let stability_result = validator.validate_numerical_stability(|u1, v1, u2, v2| {
                        validator.reference.geodesic_distance(u1, v1, u2, v2)
                    })?;
                    results.stability_tests.push(StabilityTestResult {
                        major_radius: *major_radius,
                        minor_radius: *minor_radius,
                        k_twist: *k_twist,
                        max_condition_number: stability_result.max_condition_number,
                        stability_issue_count: stability_result.stability_issues.len(),
                        is_stable: stability_result.is_stable,
                        passed: stability_result.is_stable,
                    });
                }
            }
        }
        
        Ok(results)
    }
    
    /// Run novelty detection validation tests
    fn run_novelty_detection_tests(&self) -> Result<NoveltyDetectionTestResults> {
        let mut results = NoveltyDetectionTestResults::new();
        
        // Test different stability thresholds
        let stability_thresholds = vec![0.15, 0.175, 0.20];
        
        for threshold in stability_thresholds {
            let validator = NoveltyDetectionValidator::new(threshold, self.config.tolerance);
            
            // Test cosine similarity
            let similarity_result = validator.validate_cosine_similarity(|a, b| {
                validator.reference.cosine_similarity(a, b)
            })?;
            results.similarity_tests.push(SimilarityTestResult {
                stability_threshold: threshold,
                max_error: similarity_result.max_error,
                average_error: similarity_result.average_error,
                error_count: similarity_result.errors.len(),
                passed: similarity_result.max_error < self.config.tolerance,
            });
            
            // Test novelty transformation
            let novelty_result = validator.validate_novelty_transformation(|a, b| {
                validator.reference.bounded_novelty_transformation(a, b)
            })?;
            results.novelty_tests.push(NoveltyTestResult {
                stability_threshold: threshold,
                max_error: novelty_result.max_error,
                average_error: novelty_result.average_error,
                error_count: novelty_result.errors.len(),
                boundedness_violations: novelty_result.boundedness_violations.len(),
                passed: novelty_result.max_error < self.config.tolerance && novelty_result.is_bounded,
            });
            
            // Test stability constraint
            let stability_result = validator.validate_stability_constraint(|prev, curr| {
                Ok(validator.reference.validate_stability_constraint(prev, curr)?)
            })?;
            results.stability_tests.push(StabilityConstraintTestResult {
                stability_threshold: threshold,
                violation_count: stability_result.stability_violations.len(),
                is_stable: stability_result.is_stable,
                passed: stability_result.is_stable,
            });
        }
        
        Ok(results)
    }
    
    /// Run K-Twist edge cases
    fn run_k_twist_edge_cases(&self) -> Result<Vec<EdgeCaseTestResult>> {
        let mut results = Vec::new();
        
        // Test with very small radii
        let validator = KTwistGeometryValidator::new_with_threshold(
            self.config.small_radius_major,
            self.config.small_radius_minor,
            1.0,
            self.config.tolerance,
            self.config.numerical_zero_threshold,
        );
        let result = validator.validate_positions(|u, v| {
            validator.reference.calculate_position(u, v)
        })?;
        results.push(EdgeCaseTestResult {
            test_name: "Very small radii".to_string(),
            passed: result.max_error < self.config.tolerance,
            max_error: result.max_error,
        });
        
        // Test with very large radii
        let validator = KTwistGeometryValidator::new_with_threshold(
            self.config.large_radius_major,
            self.config.large_radius_minor,
            1.0,
            self.config.tolerance,
            self.config.numerical_zero_threshold,
        );
        let result = validator.validate_positions(|u, v| {
            validator.reference.calculate_position(u, v)
        })?;
        results.push(EdgeCaseTestResult {
            test_name: "Very large radii".to_string(),
            passed: result.max_error < self.config.tolerance,
            max_error: result.max_error,
        });
        
        // Test with very large k values
        let validator = KTwistGeometryValidator::new_with_threshold(
            self.config.torus_major_radius,
            self.config.torus_minor_radius,
            self.config.large_k_value,
            self.config.tolerance,
            self.config.numerical_zero_threshold,
        );
        let result = validator.validate_positions(|u, v| {
            validator.reference.calculate_position(u, v)
        })?;
        results.push(EdgeCaseTestResult {
            test_name: "Very large k".to_string(),
            passed: result.max_error < self.config.tolerance,
            max_error: result.max_error,
        });
        
        Ok(results)
    }
    
    /// Run K-Twist boundary cases
    fn run_k_twist_boundary_cases(&self) -> Result<Vec<BoundaryTestResult>> {
        let mut results = Vec::new();
        
        let validator = KTwistGeometryValidator::new_with_threshold(
            self.config.torus_major_radius,
            self.config.torus_minor_radius,
            1.0,
            self.config.tolerance,
            self.config.numerical_zero_threshold,
        );
        
        // Test boundary points
        let boundary_points = vec![
            (0.0, 0.0),
            (TAU, 0.0),
            (0.0, TAU),
            (TAU, TAU),
            (PI, PI),
        ];
        
        for (u, v) in boundary_points {
            let result = validator.validate_positions(|u_test, v_test| {
                if (u_test - u).abs() < self.config.test_assertion_threshold && (v_test - v).abs() < self.config.test_assertion_threshold {
                    validator.reference.calculate_position(u_test, v_test)
                } else {
                    (0.0, 0.0, 0.0)
                }
            })?;
            
            results.push(BoundaryTestResult {
                boundary_point: (u, v),
                passed: result.max_error < self.config.tolerance,
                max_error: result.max_error,
            });
        }
        
        Ok(results)
    }
    
    /// Run Gaussian process edge cases
    fn run_gp_edge_cases(&self) -> Result<Vec<EdgeCaseTestResult>> {
        let mut results = Vec::new();
        
        // Test with very small length scale
        let validator = GaussianProcessValidator::new(
            self.config.small_length_scale,
            1.0,
            1.5,
            self.config.tolerance,
        );
        let result = validator.validate_rbf_kernel(|x1, x2| {
            validator.rbf_reference.covariance(x1, x2)
        })?;
        results.push(EdgeCaseTestResult {
            test_name: "Very small length scale".to_string(),
            passed: result.max_error < self.config.tolerance,
            max_error: result.max_error,
        });
        
        // Test with very large length scale
        let validator = GaussianProcessValidator::new(
            self.config.large_length_scale,
            1.0,
            1.5,
            self.config.tolerance,
        );
        let result = validator.validate_rbf_kernel(|x1, x2| {
            validator.rbf_reference.covariance(x1, x2)
        })?;
        results.push(EdgeCaseTestResult {
            test_name: "Very large length scale".to_string(),
            passed: result.max_error < self.config.tolerance,
            max_error: result.max_error,
        });
        
        // Test with very small signal variance
        let validator = GaussianProcessValidator::new(
            1.0,
            self.config.small_signal_variance,
            1.5,
            self.config.tolerance,
        );
        let result = validator.validate_rbf_kernel(|x1, x2| {
            validator.rbf_reference.covariance(x1, x2)
        })?;
        results.push(EdgeCaseTestResult {
            test_name: "Very small signal variance".to_string(),
            passed: result.max_error < self.config.tolerance,
            max_error: result.max_error,
        });
        
        Ok(results)
    }
}

/// Overall test suite results
#[derive(Debug, Clone)]
pub struct TestSuiteResult {
    pub k_twist_results: KTwistTestResults,
    pub gp_results: GaussianProcessTestResults,
    pub geodesic_results: GeodesicDistanceTestResults,
    pub novelty_results: NoveltyDetectionTestResults,
    pub overall_passed: bool,
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
}

impl TestSuiteResult {
    fn new() -> Self {
        Self {
            k_twist_results: KTwistTestResults::new(),
            gp_results: GaussianProcessTestResults::new(),
            geodesic_results: GeodesicDistanceTestResults::new(),
            novelty_results: NoveltyDetectionTestResults::new(),
            overall_passed: false,
            total_tests: 0,
            passed_tests: 0,
            failed_tests: 0,
        }
    }
    
    fn calculate_overall_results(&mut self) {
        let mut total_tests = 0;
        let mut passed_tests = 0;
        
        // Count K-Twist tests
        total_tests += self.k_twist_results.position_tests.len();
        total_tests += self.k_twist_results.normal_tests.len();
        total_tests += self.k_twist_results.orientability_tests.len();
        total_tests += self.k_twist_results.edge_case_tests.len();
        total_tests += self.k_twist_results.boundary_tests.len();
        
        for test in &self.k_twist_results.position_tests {
            if test.passed { passed_tests += 1; }
        }
        for test in &self.k_twist_results.normal_tests {
            if test.passed { passed_tests += 1; }
        }
        for test in &self.k_twist_results.orientability_tests {
            if test.passed { passed_tests += 1; }
        }
        for test in &self.k_twist_results.edge_case_tests {
            if test.passed { passed_tests += 1; }
        }
        for test in &self.k_twist_results.boundary_tests {
            if test.passed { passed_tests += 1; }
        }
        
        // Count Gaussian process tests
        total_tests += self.gp_results.rbf_tests.len();
        total_tests += self.gp_results.matern_tests.len();
        total_tests += self.gp_results.covariance_tests.len();
        total_tests += self.gp_results.edge_case_tests.len();
        
        for test in &self.gp_results.rbf_tests {
            if test.passed { passed_tests += 1; }
        }
        for test in &self.gp_results.matern_tests {
            if test.passed { passed_tests += 1; }
        }
        for test in &self.gp_results.covariance_tests {
            if test.passed { passed_tests += 1; }
        }
        for test in &self.gp_results.edge_case_tests {
            if test.passed { passed_tests += 1; }
        }
        
        // Count geodesic distance tests
        total_tests += self.geodesic_results.distance_tests.len();
        total_tests += self.geodesic_results.fallback_tests.len();
        total_tests += self.geodesic_results.stability_tests.len();
        
        for test in &self.geodesic_results.distance_tests {
            if test.passed { passed_tests += 1; }
        }
        for test in &self.geodesic_results.fallback_tests {
            if test.passed { passed_tests += 1; }
        }
        for test in &self.geodesic_results.stability_tests {
            if test.passed { passed_tests += 1; }
        }
        
        // Count novelty detection tests
        total_tests += self.novelty_results.similarity_tests.len();
        total_tests += self.novelty_results.novelty_tests.len();
        total_tests += self.novelty_results.stability_tests.len();
        
        for test in &self.novelty_results.similarity_tests {
            if test.passed { passed_tests += 1; }
        }
        for test in &self.novelty_results.novelty_tests {
            if test.passed { passed_tests += 1; }
        }
        for test in &self.novelty_results.stability_tests {
            if test.passed { passed_tests += 1; }
        }
        
        self.total_tests = total_tests;
        self.passed_tests = passed_tests;
        self.failed_tests = total_tests - passed_tests;
        self.overall_passed = self.failed_tests == 0;
    }
}

// Test result structures
#[derive(Debug, Clone)]
pub struct KTwistTestResults {
    pub position_tests: Vec<PositionTestResult>,
    pub normal_tests: Vec<NormalTestResult>,
    pub orientability_tests: Vec<OrientabilityTestResult>,
    pub edge_case_tests: Vec<EdgeCaseTestResult>,
    pub boundary_tests: Vec<BoundaryTestResult>,
}

impl KTwistTestResults {
    fn new() -> Self {
        Self {
            position_tests: Vec::new(),
            normal_tests: Vec::new(),
            orientability_tests: Vec::new(),
            edge_case_tests: Vec::new(),
            boundary_tests: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GaussianProcessTestResults {
    pub rbf_tests: Vec<RBFTestResult>,
    pub matern_tests: Vec<MaternTestResult>,
    pub covariance_tests: Vec<CovarianceTestResult>,
    pub edge_case_tests: Vec<EdgeCaseTestResult>,
}

impl GaussianProcessTestResults {
    fn new() -> Self {
        Self {
            rbf_tests: Vec::new(),
            matern_tests: Vec::new(),
            covariance_tests: Vec::new(),
            edge_case_tests: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GeodesicDistanceTestResults {
    pub distance_tests: Vec<DistanceTestResult>,
    pub fallback_tests: Vec<FallbackTestResult>,
    pub stability_tests: Vec<StabilityTestResult>,
}

impl GeodesicDistanceTestResults {
    fn new() -> Self {
        Self {
            distance_tests: Vec::new(),
            fallback_tests: Vec::new(),
            stability_tests: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NoveltyDetectionTestResults {
    pub similarity_tests: Vec<SimilarityTestResult>,
    pub novelty_tests: Vec<NoveltyTestResult>,
    pub stability_tests: Vec<StabilityConstraintTestResult>,
}

impl NoveltyDetectionTestResults {
    fn new() -> Self {
        Self {
            similarity_tests: Vec::new(),
            novelty_tests: Vec::new(),
            stability_tests: Vec::new(),
        }
    }
}

// Individual test result structures
#[derive(Debug, Clone)]
pub struct PositionTestResult {
    pub k_twist: f64,
    pub max_error: f64,
    pub average_error: f64,
    pub error_count: usize,
    pub passed: bool,
}

#[derive(Debug, Clone)]
pub struct NormalTestResult {
    pub k_twist: f64,
    pub max_error: f64,
    pub average_error: f64,
    pub error_count: usize,
    pub passed: bool,
}

#[derive(Debug, Clone)]
pub struct OrientabilityTestResult {
    pub k_twist: f64,
    pub is_non_orientable: bool,
    pub expected_non_orientable: bool,
    pub passed: bool,
}

#[derive(Debug, Clone)]
pub struct EdgeCaseTestResult {
    pub test_name: String,
    pub passed: bool,
    pub max_error: f64,
}

#[derive(Debug, Clone)]
pub struct BoundaryTestResult {
    pub boundary_point: (f64, f64),
    pub passed: bool,
    pub max_error: f64,
}

#[derive(Debug, Clone)]
pub struct RBFTestResult {
    pub length_scale: f64,
    pub signal_variance: f64,
    pub max_error: f64,
    pub average_error: f64,
    pub error_count: usize,
    pub passed: bool,
}

#[derive(Debug, Clone)]
pub struct MaternTestResult {
    pub length_scale: f64,
    pub signal_variance: f64,
    pub max_error: f64,
    pub average_error: f64,
    pub error_count: usize,
    pub passed: bool,
}

#[derive(Debug, Clone)]
pub struct CovarianceTestResult {
    pub length_scale: f64,
    pub signal_variance: f64,
    pub is_symmetric: bool,
    pub is_positive_definite: bool,
    pub diagonal_valid: bool,
    pub max_error: f64,
    pub passed: bool,
}

#[derive(Debug, Clone)]
pub struct DistanceTestResult {
    pub major_radius: f64,
    pub minor_radius: f64,
    pub k_twist: f64,
    pub max_error: f64,
    pub average_error: f64,
    pub error_count: usize,
    pub passed: bool,
}

#[derive(Debug, Clone)]
pub struct FallbackTestResult {
    pub major_radius: f64,
    pub minor_radius: f64,
    pub k_twist: f64,
    pub fallback_detected: bool,
    pub expected_detected: bool,
    pub passed: bool,
}

#[derive(Debug, Clone)]
pub struct StabilityTestResult {
    pub major_radius: f64,
    pub minor_radius: f64,
    pub k_twist: f64,
    pub max_condition_number: f64,
    pub stability_issue_count: usize,
    pub is_stable: bool,
    pub passed: bool,
}

#[derive(Debug, Clone)]
pub struct SimilarityTestResult {
    pub stability_threshold: f64,
    pub max_error: f64,
    pub average_error: f64,
    pub error_count: usize,
    pub passed: bool,
}

#[derive(Debug, Clone)]
pub struct NoveltyTestResult {
    pub stability_threshold: f64,
    pub max_error: f64,
    pub average_error: f64,
    pub error_count: usize,
    pub boundedness_violations: usize,
    pub passed: bool,
}

#[derive(Debug, Clone)]
pub struct StabilityConstraintTestResult {
    pub stability_threshold: f64,
    pub violation_count: usize,
    pub is_stable: bool,
    pub passed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mathematical_test_suite() {
        let test_suite = MathematicalTestSuite::default();
        let results = test_suite.run_all_tests().unwrap();
        
        // All tests should pass since we're using reference implementations
        assert!(results.overall_passed);
        assert_eq!(results.failed_tests, 0);
    }
    
    #[test]
    fn test_k_twist_tests() {
        let test_suite = MathematicalTestSuite::default();
        let results = test_suite.run_k_twist_tests().unwrap();
        
        // All position tests should pass
        for test in &results.position_tests {
            assert!(test.passed);
        }
        
        // All normal tests should pass
        for test in &results.normal_tests {
            assert!(test.passed);
        }
        
        // Orientability tests should match expectations
        for test in &results.orientability_tests {
            assert!(test.passed);
        }
    }
    
    #[test]
    fn test_gaussian_process_tests() {
        let test_suite = MathematicalTestSuite::default();
        let results = test_suite.run_gaussian_process_tests().unwrap();
        
        // All RBF tests should pass
        for test in &results.rbf_tests {
            assert!(test.passed);
        }
        
        // All Matérn tests should pass
        for test in &results.matern_tests {
            assert!(test.passed);
        }
        
        // All covariance tests should pass
        for test in &results.covariance_tests {
            assert!(test.passed);
        }
    }
    
    #[test]
    fn test_geodesic_distance_tests() {
        let test_suite = MathematicalTestSuite::default();
        let results = test_suite.run_geodesic_distance_tests().unwrap();
        
        // All distance tests should pass
        for test in &results.distance_tests {
            assert!(test.passed);
        }
        
        // All fallback tests should pass
        for test in &results.fallback_tests {
            assert!(test.passed);
        }
        
        // All stability tests should pass
        for test in &results.stability_tests {
            assert!(test.passed);
        }
    }
    
    #[test]
    fn test_novelty_detection_tests() {
        let test_suite = MathematicalTestSuite::default();
        let results = test_suite.run_novelty_detection_tests().unwrap();
        
        // All similarity tests should pass
        for test in &results.similarity_tests {
            assert!(test.passed);
        }
        
        // All novelty tests should pass
        for test in &results.novelty_tests {
            assert!(test.passed);
        }
        
        // All stability tests should pass
        for test in &results.stability_tests {
            assert!(test.passed);
        }
    }
}
