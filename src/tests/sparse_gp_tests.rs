//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Comprehensive tests for sparse Gaussian processes functionality

#[cfg(test)]
mod tests {
    use super::super::sparse_gaussian_processes::*;
    use anyhow::Result;
    use nalgebra::{DMatrix, DVector, Vector3};

    /// Test sparse Gaussian process creation
    #[test]
    fn test_sparse_gp_creation() {
        let inducing_points = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(2.0, 2.0, 2.0),
        ];

        let hyperparameters = KernelHyperparameters {
            length_scale: 1.0,
            signal_variance: 1.0,
            noise_variance: 0.1,
        };

        let sparse_gp =
            SparseGaussianProcess::new(inducing_points, Box::new(SparseRBFKernel), hyperparameters);

        assert_eq!(sparse_gp.num_inducing_points, 3);
        assert_eq!(sparse_gp.inducing_points.len(), 3);
        assert_eq!(sparse_gp.hyperparameters.length_scale, 1.0);
        assert_eq!(sparse_gp.hyperparameters.signal_variance, 1.0);
        assert_eq!(sparse_gp.hyperparameters.noise_variance, 0.1);
    }

    /// Test RBF kernel function
    #[test]
    fn test_rbf_kernel() {
        let kernel = SparseRBFKernel;
        let params = KernelHyperparameters {
            length_scale: 1.0,
            signal_variance: 1.0,
            noise_variance: 0.1,
        };

        let x1 = Vector3::new(0.0, 0.0, 0.0);
        let x2 = Vector3::new(1.0, 0.0, 0.0);

        let kernel_value = kernel.kernel(&x1, &x2, &params);
        assert!(kernel_value > 0.0);
        assert!(kernel_value < 1.0);

        // Test with same points
        let same_kernel_value = kernel.kernel(&x1, &x1, &params);
        assert_eq!(same_kernel_value, 1.0);
    }

    /// Test kernel matrix computation
    #[test]
    fn test_kernel_matrix_computation() {
        let kernel = SparseRBFKernel;
        let params = KernelHyperparameters {
            length_scale: 1.0,
            signal_variance: 1.0,
            noise_variance: 0.1,
        };

        let points = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ];

        let kernel_matrix = kernel.kernel_matrix(&points, &params);
        assert_eq!(kernel_matrix.nrows(), 3);
        assert_eq!(kernel_matrix.ncols(), 3);

        // Check symmetry
        for i in 0..3 {
            for j in 0..3 {
                assert!((kernel_matrix[(i, j)] - kernel_matrix[(j, i)]).abs() < 1e-6);
            }
        }

        // Check diagonal elements
        for i in 0..3 {
            assert!((kernel_matrix[(i, i)] - 1.0).abs() < 1e-6);
        }
    }

    /// Test FITC Gaussian process
    #[test]
    fn test_fitc_gp_creation() {
        let inducing_points = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 1.0)];

        let hyperparameters = KernelHyperparameters {
            length_scale: 1.0,
            signal_variance: 1.0,
            noise_variance: 0.1,
        };

        let fitc_gp = FITCGaussianProcess::new(inducing_points, hyperparameters);

        assert_eq!(fitc_gp.inducing_points.len(), 2);
        assert_eq!(fitc_gp.hyperparameters.length_scale, 1.0);
    }

    /// Test VFE Gaussian process
    #[test]
    fn test_vfe_gp_creation() {
        let inducing_points = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 1.0)];

        let hyperparameters = KernelHyperparameters {
            length_scale: 1.0,
            signal_variance: 1.0,
            noise_variance: 0.1,
        };

        let vfe_params = VFEParameters {
            variational_mean: DVector::from_vec(vec![0.0, 0.0]),
            variational_covariance: DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0])),
        };

        let vfe_gp = VFEGaussianProcess::new(inducing_points, hyperparameters, vfe_params);

        assert_eq!(vfe_gp.inducing_points.len(), 2);
        assert_eq!(vfe_gp.hyperparameters.length_scale, 1.0);
        assert_eq!(vfe_gp.vfe_params.variational_mean.len(), 2);
    }

    /// Test SO Gaussian process
    #[test]
    fn test_so_gp_creation() {
        let inducing_points = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 1.0)];

        let hyperparameters = KernelHyperparameters {
            length_scale: 1.0,
            signal_variance: 1.0,
            noise_variance: 0.1,
        };

        let so_gp = SOGaussianProcess::new(inducing_points, hyperparameters);

        assert_eq!(so_gp.inducing_points.len(), 2);
        assert_eq!(so_gp.hyperparameters.length_scale, 1.0);
    }

    /// Test scalability comparison
    #[test]
    fn test_scalability_comparison() {
        let comparison = ScalabilityComparison {
            results: vec![
                ScalabilityResult {
                    implementation: "FITC".to_string(),
                    training_size: 100,
                    total_time_ms: 50.0,
                    avg_prediction_time_ms: 5.0,
                },
                ScalabilityResult {
                    implementation: "VFE".to_string(),
                    training_size: 100,
                    total_time_ms: 60.0,
                    avg_prediction_time_ms: 6.0,
                },
            ],
            speedup_analysis: SpeedupAnalysis {
                average_speedup: 1.2,
                best_implementation: "FITC".to_string(),
                scalability_trends: std::collections::HashMap::new(),
            },
        };

        assert_eq!(comparison.results.len(), 2);
        assert_eq!(comparison.speedup_analysis.average_speedup, 1.2);
        assert_eq!(comparison.speedup_analysis.best_implementation, "FITC");
    }

    /// Test scalability result
    #[test]
    fn test_scalability_result() {
        let result = ScalabilityResult {
            implementation: "Test".to_string(),
            training_size: 1000,
            total_time_ms: 100.0,
            avg_prediction_time_ms: 10.0,
        };

        assert_eq!(result.implementation, "Test");
        assert_eq!(result.training_size, 1000);
        assert_eq!(result.total_time_ms, 100.0);
        assert_eq!(result.avg_prediction_time_ms, 10.0);
    }

    /// Test speedup analysis
    #[test]
    fn test_speedup_analysis() {
        let analysis = SpeedupAnalysis {
            average_speedup: 1.5,
            best_implementation: "Best".to_string(),
            scalability_trends: std::collections::HashMap::new(),
        };

        assert_eq!(analysis.average_speedup, 1.5);
        assert_eq!(analysis.best_implementation, "Best");
        assert!(analysis.scalability_trends.is_empty());
    }

    /// Test kernel hyperparameters
    #[test]
    fn test_kernel_hyperparameters() {
        let params = KernelHyperparameters {
            length_scale: 2.0,
            signal_variance: 1.5,
            noise_variance: 0.05,
        };

        assert_eq!(params.length_scale, 2.0);
        assert_eq!(params.signal_variance, 1.5);
        assert_eq!(params.noise_variance, 0.05);
    }

    /// Test VFE parameters
    #[test]
    fn test_vfe_parameters() {
        let variational_mean = DVector::from_vec(vec![0.5, -0.3, 0.8]);
        let variational_covariance =
            DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 0.5, 1.2]));

        let vfe_params = VFEParameters {
            variational_mean,
            variational_covariance,
        };

        assert_eq!(vfe_params.variational_mean.len(), 3);
        assert_eq!(vfe_params.variational_covariance.nrows(), 3);
        assert_eq!(vfe_params.variational_covariance.ncols(), 3);
    }

    /// Test sparse GP cache
    #[test]
    fn test_sparse_gp_cache() {
        let k_mm = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0]));
        let k_mm_inv = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0]));
        let k_mm_chol = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0]));

        let cache = SparseGPCache {
            k_mm,
            k_mm_inv: Some(k_mm_inv),
            k_mm_chol: Some(k_mm_chol),
        };

        assert_eq!(cache.k_mm.nrows(), 2);
        assert_eq!(cache.k_mm.ncols(), 2);
        assert!(cache.k_mm_inv.is_some());
        assert!(cache.k_mm_chol.is_some());
    }

    /// Test consciousness data point
    #[test]
    fn test_consciousness_data_point() {
        let point = ConsciousnessDataPoint {
            position: Vector3::new(1.0, 2.0, 3.0),
            emotional_value: 0.8,
            timestamp: 1234567890.0,
            context: "test context".to_string(),
        };

        assert_eq!(point.position.x, 1.0);
        assert_eq!(point.position.y, 2.0);
        assert_eq!(point.position.z, 3.0);
        assert_eq!(point.emotional_value, 0.8);
        assert_eq!(point.timestamp, 1234567890.0);
        assert_eq!(point.context, "test context");
    }

    /// Test consciousness data point serialization
    #[test]
    fn test_consciousness_data_point_serialization() {
        let point = ConsciousnessDataPoint {
            position: Vector3::new(1.0, 2.0, 3.0),
            emotional_value: 0.8,
            timestamp: 1234567890.0,
            context: "test context".to_string(),
        };

        // Test serialization
        let serialized = serde_json::to_string(&point);
        assert!(serialized.is_ok());

        let serialized = serialized.unwrap();
        assert!(!serialized.is_empty());

        // Test deserialization
        let deserialized: Result<ConsciousnessDataPoint, _> = serde_json::from_str(&serialized);
        assert!(deserialized.is_ok());

        let deserialized = deserialized.unwrap();
        assert_eq!(deserialized.position.x, 1.0);
        assert_eq!(deserialized.position.y, 2.0);
        assert_eq!(deserialized.position.z, 3.0);
        assert_eq!(deserialized.emotional_value, 0.8);
        assert_eq!(deserialized.timestamp, 1234567890.0);
        assert_eq!(deserialized.context, "test context");
    }

    /// Test kernel function trait implementation
    #[test]
    fn test_kernel_function_trait() {
        let kernel = SparseRBFKernel;
        let params = KernelHyperparameters {
            length_scale: 1.0,
            signal_variance: 1.0,
            noise_variance: 0.1,
        };

        let x1 = Vector3::new(0.0, 0.0, 0.0);
        let x2 = Vector3::new(1.0, 1.0, 1.0);

        // Test kernel computation
        let kernel_value = kernel.kernel(&x1, &x2, &params);
        assert!(kernel_value > 0.0);

        // Test gradient computation
        let gradient = kernel.gradient(&x1, &x2, &params);
        assert_eq!(gradient.len(), 3); // Should have 3 components for 3D vector
    }

    /// Test matrix operations
    #[test]
    fn test_matrix_operations() {
        let matrix = DMatrix::from_diagonal(&DVector::from_vec(vec![2.0, 3.0, 4.0]));

        // Test Cholesky decomposition
        let cholesky = matrix.cholesky();
        assert!(cholesky.is_some());

        let chol = cholesky.unwrap();
        let l = chol.l();
        assert_eq!(l.nrows(), 3);
        assert_eq!(l.ncols(), 3);

        // Test pseudoinverse
        let pseudo_inv = matrix.pseudo_inverse(1e-8);
        assert!(pseudo_inv.is_ok());

        let inv = pseudo_inv.unwrap();
        assert_eq!(inv.nrows(), 3);
        assert_eq!(inv.ncols(), 3);
    }

    /// Test vector operations
    #[test]
    fn test_vector_operations() {
        let v1 = Vector3::new(1.0, 2.0, 3.0);
        let v2 = Vector3::new(4.0, 5.0, 6.0);

        // Test dot product
        let dot_product = v1.dot(&v2);
        assert_eq!(dot_product, 32.0); // 1*4 + 2*5 + 3*6 = 32

        // Test norm
        let norm = v1.norm();
        assert!((norm - 3.7416573867739413).abs() < 1e-6);

        // Test distance
        let distance = (v1 - v2).norm();
        assert!((distance - 5.196152422706632).abs() < 1e-6);
    }

    /// Test numerical stability
    #[test]
    fn test_numerical_stability() {
        // Test with very small values
        let small_matrix = DMatrix::from_diagonal(&DVector::from_vec(vec![1e-10, 1e-10]));
        let pseudo_inv = small_matrix.pseudo_inverse(1e-8);
        assert!(pseudo_inv.is_ok());

        // Test with very large values
        let large_matrix = DMatrix::from_diagonal(&DVector::from_vec(vec![1e10, 1e10]));
        let pseudo_inv = large_matrix.pseudo_inverse(1e-8);
        assert!(pseudo_inv.is_ok());
    }

    /// Test edge cases
    #[test]
    fn test_edge_cases() {
        // Test with zero vector
        let zero_vector = Vector3::new(0.0, 0.0, 0.0);
        let normal_vector = Vector3::new(1.0, 0.0, 0.0);

        let kernel = SparseRBFKernel;
        let params = KernelHyperparameters {
            length_scale: 1.0,
            signal_variance: 1.0,
            noise_variance: 0.1,
        };

        let kernel_value = kernel.kernel(&zero_vector, &normal_vector, &params);
        assert!(kernel_value > 0.0);
        assert!(kernel_value < 1.0);

        // Test with identical vectors
        let same_kernel_value = kernel.kernel(&normal_vector, &normal_vector, &params);
        assert_eq!(same_kernel_value, 1.0);
    }

    /// Test performance characteristics
    #[test]
    fn test_performance_characteristics() {
        // Test with different matrix sizes
        for size in [2, 5, 10, 20] {
            let matrix = DMatrix::from_diagonal(&DVector::from_element(size, 1.0));

            // Test Cholesky decomposition
            let start = std::time::Instant::now();
            let cholesky = matrix.cholesky();
            let duration = start.elapsed();

            assert!(cholesky.is_some());
            assert!(duration.as_millis() < 1000); // Should be fast for small matrices
        }
    }
}
