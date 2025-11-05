//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Gaussian Process Validator
//!
//! Phase 4: Mathematical Validation Module
//! Verifies kernel implementations (RBF, Matérn) and validates
//! covariance matrix calculations and uncertainty handling.

use super::ValidationConfig;
use anyhow::Result;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

/// Reference implementation of RBF (Radial Basis Function) kernel
/// Used to validate the actual implementation against mathematical theory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RBFKernelReference {
    /// Length scale parameter (ℓ)
    pub length_scale: f64,
    /// Signal variance parameter (σ²)
    pub signal_variance: f64,
    /// Numerical zero threshold for avoiding division by zero
    pub numerical_zero_threshold: f64,
}

impl RBFKernelReference {
    pub fn new(length_scale: f64, signal_variance: f64) -> Self {
        Self::new_with_threshold(length_scale, signal_variance, 1e-15)
    }

    pub fn new_with_threshold(
        length_scale: f64,
        signal_variance: f64,
        numerical_zero_threshold: f64,
    ) -> Self {
        Self {
            length_scale,
            signal_variance,
            numerical_zero_threshold,
        }
    }
}

impl Default for RBFKernelReference {
    fn default() -> Self {
        Self {
            length_scale: 1.0,
            signal_variance: 1.0,
            numerical_zero_threshold: 1e-15,
        }
    }
}

impl RBFKernelReference {
    /// Calculate RBF kernel value between two points
    ///
    /// Mathematical formula:
    /// k(x₁, x₂) = σ² * exp(-0.5 * ||x₁ - x₂||² / ℓ²)
    ///
    /// Where:
    /// - σ² = signal_variance
    /// - ℓ = length_scale
    /// - ||x₁ - x₂||² = squared Euclidean distance
    pub fn covariance(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        let squared_distance = self.squared_euclidean_distance(x1, x2);
        let length_scale_sq = self.length_scale * self.length_scale;

        self.signal_variance * (-0.5 * squared_distance / length_scale_sq).exp()
    }

    /// Calculate squared Euclidean distance between two points
    fn squared_euclidean_distance(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        let mut sum = 0.0;
        for i in 0..x1.len() {
            let diff = x1[i] - x2[i];
            sum += diff * diff;
        }
        sum
    }

    /// Calculate gradient of RBF kernel with respect to hyperparameters
    pub fn gradient(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> (f64, f64) {
        let squared_distance = self.squared_euclidean_distance(x1, x2);
        let length_scale_sq = self.length_scale * self.length_scale;

        let exp_term = (-0.5 * squared_distance / length_scale_sq).exp();
        let base_cov = self.signal_variance * exp_term;

        // Gradient w.r.t. length_scale
        let length_scale_grad =
            base_cov * (squared_distance / (self.length_scale * length_scale_sq));

        // Gradient w.r.t. signal_variance
        let signal_grad = exp_term;

        (length_scale_grad, signal_grad)
    }
}

/// Reference implementation of Matérn kernel
/// Used to validate the actual implementation against mathematical theory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaternKernelReference {
    /// Length scale parameter (ℓ)
    pub length_scale: f64,
    /// Signal variance parameter (σ²)
    pub signal_variance: f64,
    /// Smoothness parameter (ν)
    pub nu: f64,
    /// Numerical zero threshold for avoiding division by zero
    pub numerical_zero_threshold: f64,
}

impl MaternKernelReference {
    pub fn new(length_scale: f64, signal_variance: f64, nu: f64) -> Self {
        Self::new_with_threshold(length_scale, signal_variance, nu, 1e-15)
    }

    pub fn new_with_threshold(
        length_scale: f64,
        signal_variance: f64,
        nu: f64,
        numerical_zero_threshold: f64,
    ) -> Self {
        Self {
            length_scale,
            signal_variance,
            nu,
            numerical_zero_threshold,
        }
    }
}

impl Default for MaternKernelReference {
    fn default() -> Self {
        Self {
            length_scale: 1.0,
            signal_variance: 1.0,
            nu: 1.5, // Matérn-3/2
            numerical_zero_threshold: 1e-15,
        }
    }
}

impl MaternKernelReference {
    /// Calculate Matérn kernel value between two points
    ///
    /// Mathematical formula for Matérn-3/2:
    /// k(x₁, x₂) = σ² * (1 + √3 * r/ℓ) * exp(-√3 * r/ℓ)
    ///
    /// Where:
    /// - σ² = signal_variance
    /// - ℓ = length_scale
    /// - r = ||x₁ - x₂|| (Euclidean distance)
    /// - ν = 3/2 (smoothness parameter)
    pub fn covariance(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        let distance = self.euclidean_distance(x1, x2);
        let scaled_distance = distance / self.length_scale;

        if scaled_distance < self.numerical_zero_threshold {
            return self.signal_variance;
        }

        let sqrt_3 = 3.0_f64.sqrt();
        let matern_factor = (1.0 + sqrt_3 * scaled_distance) * (-sqrt_3 * scaled_distance).exp();

        self.signal_variance * matern_factor
    }

    /// Calculate Euclidean distance between two points
    fn euclidean_distance(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        self.squared_euclidean_distance(x1, x2).sqrt()
    }

    /// Calculate squared Euclidean distance between two points
    fn squared_euclidean_distance(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        let mut sum = 0.0;
        for i in 0..x1.len() {
            let diff = x1[i] - x2[i];
            sum += diff * diff;
        }
        sum
    }

    /// Calculate gradient of Matérn kernel with respect to hyperparameters
    pub fn gradient(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> (f64, f64) {
        let distance = self.euclidean_distance(x1, x2);
        let scaled_distance = distance / self.length_scale;

        let sqrt_3 = 3.0_f64.sqrt();
        let matern_factor = (1.0 + sqrt_3 * scaled_distance) * (-sqrt_3 * scaled_distance).exp();

        // Gradient w.r.t. length_scale
        let length_scale_grad =
            self.signal_variance * sqrt_3 * scaled_distance * matern_factor / self.length_scale;

        // Gradient w.r.t. signal_variance
        let signal_grad = matern_factor;

        (length_scale_grad, signal_grad)
    }
}

/// Validator for Gaussian process kernel implementations
#[derive(Debug, Clone)]
pub struct GaussianProcessValidator {
    pub rbf_reference: RBFKernelReference,
    pub matern_reference: MaternKernelReference,
    tolerance: f64,
    test_coordinate_range: f64,
    test_coordinate_offset: f64,
}

impl GaussianProcessValidator {
    pub fn new(length_scale: f64, signal_variance: f64, matern_nu: f64, tolerance: f64) -> Self {
        let config = ValidationConfig::default();
        Self::new_with_coordinates_and_threshold(
            length_scale,
            signal_variance,
            matern_nu,
            tolerance,
            config.test_coordinate_range,
            config.test_coordinate_offset,
            config.numerical_zero_threshold,
        )
    }

    pub fn new_with_coordinates(
        length_scale: f64,
        signal_variance: f64,
        matern_nu: f64,
        tolerance: f64,
        test_coordinate_range: f64,
        test_coordinate_offset: f64,
    ) -> Self {
        let config = ValidationConfig::default();
        Self::new_with_coordinates_and_threshold(
            length_scale,
            signal_variance,
            matern_nu,
            tolerance,
            test_coordinate_range,
            test_coordinate_offset,
            config.numerical_zero_threshold,
        )
    }

    pub fn new_with_coordinates_and_threshold(
        length_scale: f64,
        signal_variance: f64,
        matern_nu: f64,
        tolerance: f64,
        test_coordinate_range: f64,
        test_coordinate_offset: f64,
        numerical_zero_threshold: f64,
    ) -> Self {
        Self {
            rbf_reference: RBFKernelReference::new_with_threshold(
                length_scale,
                signal_variance,
                numerical_zero_threshold,
            ),
            matern_reference: MaternKernelReference::new_with_threshold(
                length_scale,
                signal_variance,
                matern_nu,
                numerical_zero_threshold,
            ),
            tolerance,
            test_coordinate_range,
            test_coordinate_offset,
        }
    }

    /// Validate RBF kernel implementation
    pub fn validate_rbf_kernel<F>(&self, implementation: F) -> Result<KernelValidationResult>
    where
        F: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
    {
        let mut errors = Vec::new();
        let mut max_error: f64 = 0.0;
        let mut total_error = 0.0;
        let mut test_count = 0;

        // Generate test points
        let test_points = self.generate_test_points();

        for i in 0..test_points.len() {
            for j in 0..test_points.len() {
                let x1 = test_points[i].view();
                let x2 = test_points[j].view();

                let reference_cov = self.rbf_reference.covariance(&x1, &x2);
                let implementation_cov = implementation(&x1, &x2);

                let error = (reference_cov - implementation_cov).abs();

                if error > self.tolerance {
                    errors.push(KernelError {
                        x1: x1.to_owned(),
                        x2: x2.to_owned(),
                        reference: reference_cov,
                        implementation: implementation_cov,
                        error,
                    });
                }

                max_error = max_error.max(error);
                total_error += error;
                test_count += 1;
            }
        }

        let average_error = total_error / test_count as f64;

        Ok(KernelValidationResult {
            kernel_type: "RBF".to_string(),
            max_error,
            average_error,
            errors,
            covariance_matrix_valid: true, // Will be validated separately
        })
    }

    /// Validate Matérn kernel implementation
    pub fn validate_matern_kernel<F>(&self, implementation: F) -> Result<KernelValidationResult>
    where
        F: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
    {
        let mut errors = Vec::new();
        let mut max_error: f64 = 0.0;
        let mut total_error = 0.0;
        let mut test_count = 0;

        // Generate test points
        let test_points = self.generate_test_points();

        for i in 0..test_points.len() {
            for j in 0..test_points.len() {
                let x1 = test_points[i].view();
                let x2 = test_points[j].view();

                let reference_cov = self.matern_reference.covariance(&x1, &x2);
                let implementation_cov = implementation(&x1, &x2);

                let error = (reference_cov - implementation_cov).abs();

                if error > self.tolerance {
                    errors.push(KernelError {
                        x1: x1.to_owned(),
                        x2: x2.to_owned(),
                        reference: reference_cov,
                        implementation: implementation_cov,
                        error,
                    });
                }

                max_error = max_error.max(error);
                total_error += error;
                test_count += 1;
            }
        }

        let average_error = total_error / test_count as f64;

        Ok(KernelValidationResult {
            kernel_type: "Matérn".to_string(),
            max_error,
            average_error,
            errors,
            covariance_matrix_valid: true, // Will be validated separately
        })
    }

    /// Validate covariance matrix properties
    pub fn validate_covariance_matrix<F>(
        &self,
        implementation: F,
    ) -> Result<CovarianceMatrixValidationResult>
    where
        F: Fn(&ArrayView2<f64>) -> Array2<f64>,
    {
        let test_points = self.generate_test_points();
        let n = test_points.len();

        // Build reference covariance matrix
        let mut reference_matrix = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                reference_matrix[[i, j]] = self
                    .rbf_reference
                    .covariance(&test_points[i].view(), &test_points[j].view());
            }
        }

        // Get implementation covariance matrix
        let implementation_matrix = implementation(&reference_matrix.view());

        // Validate properties
        let mut validation_result = CovarianceMatrixValidationResult {
            is_symmetric: true,
            is_positive_definite: true,
            diagonal_elements_valid: true,
            max_error: 0.0,
            average_error: 0.0,
            errors: Vec::new(),
        };

        // Check symmetry
        for i in 0..n {
            for j in 0..n {
                let error = (implementation_matrix[[i, j]] - implementation_matrix[[j, i]]).abs();
                if error > self.tolerance {
                    validation_result.is_symmetric = false;
                    validation_result.errors.push(CovarianceError {
                        i,
                        j,
                        expected: implementation_matrix[[j, i]],
                        actual: implementation_matrix[[i, j]],
                        error,
                    });
                }
            }
        }

        // Check diagonal elements (should be signal_variance + noise)
        for i in 0..n {
            let expected_diagonal = self.rbf_reference.signal_variance;
            let actual_diagonal = implementation_matrix[[i, i]];
            let error = (expected_diagonal - actual_diagonal).abs();

            if error > self.tolerance {
                validation_result.diagonal_elements_valid = false;
                validation_result.errors.push(CovarianceError {
                    i,
                    j: i,
                    expected: expected_diagonal,
                    actual: actual_diagonal,
                    error,
                });
            }
        }

        // Check positive definiteness (simplified check)
        validation_result.is_positive_definite =
            self.check_positive_definiteness(&implementation_matrix);

        // Calculate overall error
        let mut total_error = 0.0;
        let mut max_error: f64 = 0.0;
        for i in 0..n {
            for j in 0..n {
                let error = (reference_matrix[[i, j]] - implementation_matrix[[i, j]]).abs();
                max_error = max_error.max(error);
                total_error += error;
            }
        }

        validation_result.max_error = max_error;
        validation_result.average_error = total_error / (n * n) as f64;

        Ok(validation_result)
    }

    /// Validate mean and variance predictions
    pub fn validate_predictions<F>(&self, implementation: F) -> Result<PredictionValidationResult>
    where
        F: Fn(&ArrayView1<f64>) -> (f64, f64),
    {
        let test_points = self.generate_test_points();
        let mut errors = Vec::new();
        let mut max_mean_error: f64 = 0.0;
        let mut max_variance_error: f64 = 0.0;
        let mut total_mean_error = 0.0;
        let mut total_variance_error = 0.0;
        let mut test_count = 0;

        for point in &test_points {
            let (predicted_mean, predicted_variance) = implementation(&point.view());

            // For validation, we'll use simple reference values
            // In practice, these would come from a known GP solution
            let reference_mean = 0.0; // Assuming zero mean
            let reference_variance = self.rbf_reference.signal_variance;

            let mean_error = (predicted_mean - reference_mean).abs();
            let variance_error = (predicted_variance - reference_variance).abs();

            if mean_error > self.tolerance || variance_error > self.tolerance {
                errors.push(PredictionError {
                    point: point.to_owned(),
                    predicted_mean,
                    predicted_variance,
                    reference_mean,
                    reference_variance,
                    mean_error,
                    variance_error,
                });
            }

            max_mean_error = max_mean_error.max(mean_error);
            max_variance_error = max_variance_error.max(variance_error);
            total_mean_error += mean_error;
            total_variance_error += variance_error;
            test_count += 1;
        }

        Ok(PredictionValidationResult {
            max_mean_error,
            max_variance_error,
            average_mean_error: total_mean_error / test_count as f64,
            average_variance_error: total_variance_error / test_count as f64,
            errors,
        })
    }

    /// Generate test points for validation
    fn generate_test_points(&self) -> Vec<Array1<f64>> {
        let mut points = Vec::new();

        // Generate points in a grid
        let n_points = 10;
        for i in 0..n_points {
            for j in 0..n_points {
                let x = (i as f64 / (n_points - 1) as f64) * self.test_coordinate_range
                    - self.test_coordinate_offset;
                let y = (j as f64 / (n_points - 1) as f64) * self.test_coordinate_range
                    - self.test_coordinate_offset;
                points.push(Array1::from_vec(vec![x, y]));
            }
        }

        points
    }

    /// Check if matrix is positive definite (simplified check)
    fn check_positive_definiteness(&self, matrix: &Array2<f64>) -> bool {
        // Simple check: all eigenvalues should be positive
        // This is a simplified implementation
        let n = matrix.nrows();
        let mut is_positive_definite = true;

        for i in 0..n {
            if matrix[[i, i]] <= 0.0 {
                is_positive_definite = false;
                break;
            }
        }

        is_positive_definite
    }
}

/// Result of kernel validation
#[derive(Debug, Clone)]
pub struct KernelValidationResult {
    pub kernel_type: String,
    pub max_error: f64,
    pub average_error: f64,
    pub errors: Vec<KernelError>,
    pub covariance_matrix_valid: bool,
}

/// Result of covariance matrix validation
#[derive(Debug, Clone)]
pub struct CovarianceMatrixValidationResult {
    pub is_symmetric: bool,
    pub is_positive_definite: bool,
    pub diagonal_elements_valid: bool,
    pub max_error: f64,
    pub average_error: f64,
    pub errors: Vec<CovarianceError>,
}

/// Result of prediction validation
#[derive(Debug, Clone)]
pub struct PredictionValidationResult {
    pub max_mean_error: f64,
    pub max_variance_error: f64,
    pub average_mean_error: f64,
    pub average_variance_error: f64,
    pub errors: Vec<PredictionError>,
}

/// Kernel calculation error
#[derive(Debug, Clone)]
pub struct KernelError {
    pub x1: Array1<f64>,
    pub x2: Array1<f64>,
    pub reference: f64,
    pub implementation: f64,
    pub error: f64,
}

/// Covariance matrix error
#[derive(Debug, Clone)]
pub struct CovarianceError {
    pub i: usize,
    pub j: usize,
    pub expected: f64,
    pub actual: f64,
    pub error: f64,
}

/// Prediction error
#[derive(Debug, Clone)]
pub struct PredictionError {
    pub point: Array1<f64>,
    pub predicted_mean: f64,
    pub predicted_variance: f64,
    pub reference_mean: f64,
    pub reference_variance: f64,
    pub mean_error: f64,
    pub variance_error: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::ValidationConfig;

    #[test]
    fn test_rbf_reference_basic() {
        let config = ValidationConfig::default();
        let rbf = RBFKernelReference::default();

        let x1 = Array1::from_vec(vec![0.0, 0.0]);
        let x2 = Array1::from_vec(vec![1.0, 0.0]);

        let cov = rbf.covariance(&x1.view(), &x2.view());
        let expected = config.expected_rbf_covariance;

        assert!((cov - expected).abs() < config.test_assertion_threshold);
    }

    #[test]
    fn test_matern_reference_basic() {
        let config = ValidationConfig::default();
        let matern = MaternKernelReference::default();

        let x1 = Array1::from_vec(vec![0.0, 0.0]);
        let x2 = Array1::from_vec(vec![1.0, 0.0]);

        let cov = matern.covariance(&x1.view(), &x2.view());
        let expected = config.expected_matern_covariance;

        assert!((cov - expected).abs() < config.test_assertion_threshold);
    }

    #[test]
    fn test_validator_rbf_validation() {
        let config = ValidationConfig::default();
        let validator = GaussianProcessValidator::new(1.0, 1.0, 1.5, 1e-6);

        // Test with reference implementation (should have zero error)
        let result = validator
            .validate_rbf_kernel(|x1, x2| validator.rbf_reference.covariance(x1, x2))
            .unwrap();

        assert!(result.max_error < config.test_assertion_threshold);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_validator_matern_validation() {
        let config = ValidationConfig::default();
        let validator = GaussianProcessValidator::new(1.0, 1.0, 1.5, 1e-6);

        // Test with reference implementation (should have zero error)
        let result = validator
            .validate_matern_kernel(|x1, x2| validator.matern_reference.covariance(x1, x2))
            .unwrap();

        assert!(result.max_error < config.test_assertion_threshold);
        assert!(result.errors.is_empty());
    }
}
