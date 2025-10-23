//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Comprehensive test suite for Gaussian process validation
//!
//! This module provides extensive tests for the GP validation framework,
//! ensuring that kernel functions, matrix operations, and approximations
//! are mathematically correct and numerically stable.

use crate::validation::{
    GPImplementation, GPValidationResult, GPValidator, KernelHyperparameters, ReferenceRBFKernel,
    TestGPImplementation,
};
use nalgebra::{DMatrix, Vector3};

/// Test implementation with correct RBF kernel
pub struct CorrectGPImplementation {
    hyperparameters: KernelHyperparameters,
    inducing_points: Vec<Vector3<f32>>,
}

impl CorrectGPImplementation {
    pub fn new(hyperparameters: KernelHyperparameters, inducing_points: Vec<Vector3<f32>>) -> Self {
        Self {
            hyperparameters,
            inducing_points,
        }
    }
}

impl GPImplementation for CorrectGPImplementation {
    fn compute_kernel(&self, x1: &Vector3<f32>, x2: &Vector3<f32>) -> f32 {
        // Correct RBF kernel implementation
        let squared_distance = (x1 - x2).norm_squared();
        let length_scale_sq = self.hyperparameters.length_scale * self.hyperparameters.length_scale;

        self.hyperparameters.signal_variance * (-0.5 * squared_distance / length_scale_sq).exp()
    }

    fn compute_kernel_matrix(&self, points: &[Vector3<f32>]) -> DMatrix<f32> {
        let n = points.len();
        let mut matrix = DMatrix::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                matrix[(i, j)] = self.compute_kernel(&points[i], &points[j]);
            }
            // Add noise to diagonal
            matrix[(i, i)] += self.hyperparameters.noise_variance;
        }

        matrix
    }

    fn compute_gradient(&self, x1: &Vector3<f32>, x2: &Vector3<f32>) -> Vec<f32> {
        let squared_distance = (x1 - x2).norm_squared();
        let length_scale_sq = self.hyperparameters.length_scale * self.hyperparameters.length_scale;

        let exp_term = (-0.5 * squared_distance / length_scale_sq).exp();
        let base_kernel = self.hyperparameters.signal_variance * exp_term;

        // Gradient w.r.t. length_scale
        let length_scale_grad = base_kernel
            * (squared_distance / (self.hyperparameters.length_scale * length_scale_sq));

        // Gradient w.r.t. signal_variance
        let signal_grad = exp_term;

        vec![length_scale_grad, signal_grad]
    }

    fn get_hyperparameters(&self) -> KernelHyperparameters {
        self.hyperparameters.clone()
    }

    fn get_inducing_points(&self) -> Vec<Vector3<f32>> {
        self.inducing_points.clone()
    }
}

/// Test implementation with incorrect kernel (for testing validation)
pub struct IncorrectGPImplementation {
    hyperparameters: KernelHyperparameters,
    inducing_points: Vec<Vector3<f32>>,
}

impl IncorrectGPImplementation {
    pub fn new(hyperparameters: KernelHyperparameters, inducing_points: Vec<Vector3<f32>>) -> Self {
        Self {
            hyperparameters,
            inducing_points,
        }
    }
}

impl GPImplementation for IncorrectGPImplementation {
    fn compute_kernel(&self, x1: &Vector3<f32>, x2: &Vector3<f32>) -> f32 {
        // Incorrect kernel implementation - not symmetric
        let squared_distance = (x1 - x2).norm_squared();
        let length_scale_sq = self.hyperparameters.length_scale * self.hyperparameters.length_scale;

        // Add asymmetric term to make it incorrect
        let asymmetric_term = if x1.x > x2.x { 0.1 } else { 0.0 };

        self.hyperparameters.signal_variance * (-0.5 * squared_distance / length_scale_sq).exp()
            + asymmetric_term
    }

    fn compute_kernel_matrix(&self, points: &[Vector3<f32>]) -> DMatrix<f32> {
        let n = points.len();
        let mut matrix = DMatrix::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                matrix[(i, j)] = self.compute_kernel(&points[i], &points[j]);
            }
            // Add noise to diagonal
            matrix[(i, i)] += self.hyperparameters.noise_variance;
        }

        matrix
    }

    fn compute_gradient(&self, x1: &Vector3<f32>, x2: &Vector3<f32>) -> Vec<f32> {
        // Incorrect gradient implementation
        vec![0.0, 0.0] // Always return zero gradient
    }

    fn get_hyperparameters(&self) -> KernelHyperparameters {
        self.hyperparameters.clone()
    }

    fn get_inducing_points(&self) -> Vec<Vector3<f32>> {
        self.inducing_points.clone()
    }
}

#[cfg(test)]
mod comprehensive_tests {
    use super::*;

    /// Test that the validator correctly identifies errors in incorrect implementations
    #[test]
    fn test_incorrect_implementation_has_errors() {
        let validator = GPValidator::new(1e-6);
        let hyperparameters = KernelHyperparameters {
            length_scale: 1.0,
            signal_variance: 1.0,
            noise_variance: 0.1,
            num_inducing: 10,
        };
        let inducing_points = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 1.0)];
        let implementation = IncorrectGPImplementation::new(hyperparameters, inducing_points);

        let result = validator.validate(&implementation).unwrap();

        // The incorrect implementation should have errors
        assert!(!result.is_valid);
        assert!(!result.kernel_validation.is_symmetric);

        // Check that we have critical or major issues
        let has_critical_issues = result.issues.iter().any(|issue| {
            matches!(
                issue.severity,
                crate::validation::GPIssueSeverity::Critical
                    | crate::validation::GPIssueSeverity::Major
            )
        });
        assert!(has_critical_issues);

        // Check that kernel function issues are present
        let has_kernel_issues = result.issues.iter().any(|issue| {
            matches!(
                issue.category,
                crate::validation::GPIssueCategory::KernelFunction
            )
        });
        assert!(has_kernel_issues);
    }

    /// Test that the validator passes correct implementations
    #[test]
    fn test_correct_implementation_passes() {
        let validator = GPValidator::new(1e-6);
        let hyperparameters = KernelHyperparameters {
            length_scale: 1.0,
            signal_variance: 1.0,
            noise_variance: 0.1,
            num_inducing: 10,
        };
        let inducing_points = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(2.0, 2.0, 2.0),
        ];
        let implementation = CorrectGPImplementation::new(hyperparameters, inducing_points);

        let result = validator.validate(&implementation).unwrap();

        // The correct implementation should pass validation
        assert!(result.is_valid);
        assert!(result.kernel_validation.is_positive_definite);
        assert!(result.kernel_validation.is_symmetric);
        assert!(result.kernel_validation.is_bounded);
        assert!(result.matrix_validation.kernel_matrix_pd);
        assert!(result.matrix_validation.cholesky_succeeds);
        assert!(result.numerical_validation.is_stable);

        // Should have no critical or major issues
        let has_critical_issues = result.issues.iter().any(|issue| {
            matches!(
                issue.severity,
                crate::validation::GPIssueSeverity::Critical
                    | crate::validation::GPIssueSeverity::Major
            )
        });
        assert!(!has_critical_issues);
    }

    /// Test kernel function properties validation
    #[test]
    fn test_kernel_properties_validation() {
        let validator = GPValidator::new(1e-6);
        let hyperparameters = KernelHyperparameters {
            length_scale: 1.0,
            signal_variance: 1.0,
            noise_variance: 0.1,
            num_inducing: 10,
        };
        let inducing_points = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 1.0)];
        let implementation = CorrectGPImplementation::new(hyperparameters, inducing_points);

        let result = validator.validate(&implementation).unwrap();

        // Check kernel properties
        assert!(result.kernel_validation.is_positive_definite);
        assert!(result.kernel_validation.satisfies_mercer);
        assert!(result.kernel_validation.is_symmetric);
        assert!(result.kernel_validation.is_bounded);
        assert!(result.kernel_validation.max_kernel_value > 0.0);
        assert!(result.kernel_validation.min_kernel_value >= 0.0);
    }

    /// Test matrix operations validation
    #[test]
    fn test_matrix_operations_validation() {
        let validator = GPValidator::new(1e-6);
        let hyperparameters = KernelHyperparameters {
            length_scale: 1.0,
            signal_variance: 1.0,
            noise_variance: 0.1,
            num_inducing: 10,
        };
        let inducing_points = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(2.0, 2.0, 2.0),
        ];
        let implementation = CorrectGPImplementation::new(hyperparameters, inducing_points);

        let result = validator.validate(&implementation).unwrap();

        // Check matrix operations
        assert!(result.matrix_validation.kernel_matrix_pd);
        assert!(result.matrix_validation.cholesky_succeeds);
        assert!(result.matrix_validation.inversion_stable);
        assert!(result.matrix_validation.eigenvalues_positive);
        assert!(result.matrix_validation.condition_number.is_finite());
        assert!(result.matrix_validation.condition_number > 1.0);
        assert!(result.matrix_validation.matrix_rank > 0);
    }

    /// Test approximation quality validation
    #[test]
    fn test_approximation_quality_validation() {
        let validator = GPValidator::new(1e-6);
        let hyperparameters = KernelHyperparameters {
            length_scale: 1.0,
            signal_variance: 1.0,
            noise_variance: 0.1,
            num_inducing: 10,
        };
        let inducing_points = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(2.0, 2.0, 2.0),
        ];
        let implementation = CorrectGPImplementation::new(hyperparameters, inducing_points);

        let result = validator.validate(&implementation).unwrap();

        // Check approximation quality
        assert!(result.approximation_validation.fitc_valid);
        assert!(result.approximation_validation.vfe_valid);
        assert!(result.approximation_validation.sparse_preserves_properties);
        assert!(result.approximation_validation.approximation_error >= 0.0);
        assert!(result.approximation_validation.inducing_points_distributed);
    }

    /// Test numerical stability validation
    #[test]
    fn test_numerical_stability_validation() {
        let validator = GPValidator::new(1e-6);
        let hyperparameters = KernelHyperparameters {
            length_scale: 1.0,
            signal_variance: 1.0,
            noise_variance: 0.1,
            num_inducing: 10,
        };
        let inducing_points = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(2.0, 2.0, 2.0),
        ];
        let implementation = CorrectGPImplementation::new(hyperparameters, inducing_points);

        let result = validator.validate(&implementation).unwrap();

        // Check numerical stability
        assert!(result.numerical_validation.is_stable);
        assert!(result.numerical_validation.matrix_operations_stable);
        assert!(result.numerical_validation.gradient_stable);
        assert!(result.numerical_validation.max_condition_number.is_finite());
        assert!(result.numerical_validation.precision_issues.is_empty());
    }

    /// Test edge cases and boundary conditions
    #[test]
    fn test_edge_cases() {
        let validator = GPValidator::new(1e-6);

        // Test with very small length scale
        let hyperparameters_small = KernelHyperparameters {
            length_scale: 0.01,
            signal_variance: 1.0,
            noise_variance: 0.1,
            num_inducing: 10,
        };
        let inducing_points = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 1.0)];
        let implementation_small =
            CorrectGPImplementation::new(hyperparameters_small, inducing_points);

        let result_small = validator.validate(&implementation_small).unwrap();
        assert!(result_small.numerical_validation.is_stable);

        // Test with very large length scale
        let hyperparameters_large = KernelHyperparameters {
            length_scale: 100.0,
            signal_variance: 1.0,
            noise_variance: 0.1,
            num_inducing: 10,
        };
        let implementation_large =
            CorrectGPImplementation::new(hyperparameters_large, inducing_points);

        let result_large = validator.validate(&implementation_large).unwrap();
        assert!(result_large.numerical_validation.is_stable);
    }

    /// Test different parameter ranges
    #[test]
    fn test_parameter_ranges() {
        let validator = GPValidator::new(1e-6);
        let inducing_points = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(2.0, 2.0, 2.0),
        ];

        let test_cases = vec![
            (0.1, 0.5, 0.01),  // Small parameters
            (10.0, 2.0, 0.5),  // Large parameters
            (1.0, 0.1, 0.001), // Small signal variance
            (1.0, 5.0, 0.1),   // Large signal variance
        ];

        for (length_scale, signal_variance, noise_variance) in test_cases {
            let hyperparameters = KernelHyperparameters {
                length_scale,
                signal_variance,
                noise_variance,
                num_inducing: 10,
            };
            let implementation =
                CorrectGPImplementation::new(hyperparameters, inducing_points.clone());

            let result = validator.validate(&implementation).unwrap();

            // All parameter ranges should be valid
            assert!(result.is_valid);
            assert!(result.kernel_validation.is_positive_definite);
            assert!(result.numerical_validation.is_stable);
        }
    }

    /// Test error reporting and issue categorization
    #[test]
    fn test_error_reporting() {
        let validator = GPValidator::new(1e-6);
        let hyperparameters = KernelHyperparameters {
            length_scale: 1.0,
            signal_variance: 1.0,
            noise_variance: 0.1,
            num_inducing: 10,
        };
        let inducing_points = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 1.0)];
        let implementation = IncorrectGPImplementation::new(hyperparameters, inducing_points);

        let result = validator.validate(&implementation).unwrap();

        // Check that issues are properly categorized
        let kernel_issues: Vec<_> = result
            .issues
            .iter()
            .filter(|issue| {
                matches!(
                    issue.category,
                    crate::validation::GPIssueCategory::KernelFunction
                )
            })
            .collect();
        assert!(!kernel_issues.is_empty());

        // Check that issues have proper descriptions and suggestions
        for issue in &result.issues {
            assert!(!issue.description.is_empty());
            if matches!(
                issue.severity,
                crate::validation::GPIssueSeverity::Critical
                    | crate::validation::GPIssueSeverity::Major
            ) {
                assert!(issue.suggested_fix.is_some());
            }
        }
    }

    /// Test reference implementation correctness
    #[test]
    fn test_reference_implementation() {
        let reference = ReferenceRBFKernel;

        let x1 = Vector3::new(0.0, 0.0, 0.0);
        let x2 = Vector3::new(1.0, 0.0, 0.0);

        // Test kernel computation
        let kernel_value = reference.kernel(&x1, &x2, 1.0, 1.0);
        assert!(kernel_value > 0.0);
        assert!(kernel_value < 1.0);

        // Test with same points
        let same_kernel_value = reference.kernel(&x1, &x1, 1.0, 1.0);
        assert_eq!(same_kernel_value, 1.0);

        // Test kernel matrix
        let points = vec![x1, x2];
        let matrix = reference.kernel_matrix(&points, 1.0, 1.0, 0.1);
        assert!(reference.is_positive_definite(&matrix));
        assert!(reference.cholesky_succeeds(&matrix));

        // Test condition number
        let condition_number = reference.condition_number(&matrix);
        assert!(condition_number.is_finite());
        assert!(condition_number > 1.0);
    }

    /// Test validation with different tolerances
    #[test]
    fn test_tolerance_sensitivity() {
        let hyperparameters = KernelHyperparameters {
            length_scale: 1.0,
            signal_variance: 1.0,
            noise_variance: 0.1,
            num_inducing: 10,
        };
        let inducing_points = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 1.0)];
        let implementation = IncorrectGPImplementation::new(hyperparameters, inducing_points);

        // Test with loose tolerance
        let validator_loose = GPValidator::new(1e-3);
        let result_loose = validator_loose.validate(&implementation).unwrap();

        // Test with tight tolerance
        let validator_tight = GPValidator::new(1e-9);
        let result_tight = validator_tight.validate(&implementation).unwrap();

        // Tight tolerance should catch more errors
        assert!(result_tight.issues.len() >= result_loose.issues.len());
    }

    /// Test validation performance with large test sets
    #[test]
    fn test_validation_performance() {
        let validator = GPValidator::new(1e-6);
        let hyperparameters = KernelHyperparameters {
            length_scale: 1.0,
            signal_variance: 1.0,
            noise_variance: 0.1,
            num_inducing: 10,
        };
        let inducing_points = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(2.0, 2.0, 2.0),
        ];
        let implementation = CorrectGPImplementation::new(hyperparameters, inducing_points);

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
        let validator = GPValidator::new(1e-6);
        let inducing_points = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 1.0)];

        let extreme_cases = vec![
            (1e-6, 1e-6, 1e-6), // Very small parameters
            (1e6, 1e6, 1e6),    // Very large parameters
            (0.0, 1.0, 0.1),    // Zero length scale
            (1.0, 0.0, 0.1),    // Zero signal variance
        ];

        for (length_scale, signal_variance, noise_variance) in extreme_cases {
            if length_scale > 0.0 && signal_variance > 0.0 {
                let hyperparameters = KernelHyperparameters {
                    length_scale,
                    signal_variance,
                    noise_variance,
                    num_inducing: 10,
                };
                let implementation =
                    CorrectGPImplementation::new(hyperparameters, inducing_points.clone());

                let result = validator.validate(&implementation).unwrap();

                // Should handle extreme parameters gracefully
                assert!(result.numerical_validation.is_stable);
            }
        }
    }
}
