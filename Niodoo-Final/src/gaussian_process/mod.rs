//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Gaussian Process Kernel with K-Twist Topology Influence
//!
//! This module implements RBF kernels that incorporate the K-Twist topology
//! for enhanced pattern recognition and uncertainty quantification.

use anyhow::Result;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Kernel parameters for K-Twist influenced Gaussian processes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KTwistKernelParams {
    /// Base lengthscale for RBF kernel
    pub lengthscale: f64,
    /// Variance parameter
    pub variance: f64,
    /// K-Twist influence factor
    pub k_twist_influence: f64,
    /// Topology adaptation rate
    pub topology_adaptation: f64,
    /// Noise level
    pub noise: f64,
}

impl Default for KTwistKernelParams {
    fn default() -> Self {
        Self {
            lengthscale: 1.0,
            variance: 1.0,
            k_twist_influence: 0.15,
            topology_adaptation: 0.01,
            noise: 0.01,
        }
    }
}

/// K-Twist influenced RBF kernel
#[derive(Debug, Clone)]
pub struct KTwistRBFKernel {
    params: KTwistKernelParams,
    topology_points: Vec<(f64, f64, f64)>, // Topology reference points
}

impl KTwistRBFKernel {
    /// Create a new K-Twist RBF kernel
    pub fn new(params: KTwistKernelParams) -> Self {
        Self {
            params,
            topology_points: Vec::new(),
        }
    }

    /// Update kernel with topology information
    pub fn update_topology(&mut self, topology_points: Vec<(f64, f64, f64)>) {
        self.topology_points = topology_points;
    }

    /// Compute kernel matrix between two sets of points
    pub fn compute_kernel_matrix(&self, x1: &Array2<f64>, x2: &Array2<f64>) -> Array2<f64> {
        let n1 = x1.nrows();
        let n2 = x2.nrows();
        let mut k = Array2::zeros((n1, n2));

        for i in 0..n1 {
            for j in 0..n2 {
                let x1_row = x1.row(i);
                let x2_row = x2.row(j);
                k[[i, j]] = self.compute_kernel_value(&x1_row.to_owned(), &x2_row.to_owned());
            }
        }

        k
    }

    /// Compute kernel value between two points
    pub fn compute_kernel_value(&self, x1: &Array1<f64>, x2: &Array1<f64>) -> f64 {
        // Base RBF kernel
        let base_kernel = self.compute_base_rbf(x1, x2);

        // K-Twist topology influence
        let topology_influence = self.compute_topology_influence(x1, x2);

        // Combine base kernel with topology influence
        base_kernel * (1.0 + self.params.k_twist_influence * topology_influence)
    }

    /// Compute base RBF kernel value
    fn compute_base_rbf(&self, x1: &Array1<f64>, x2: &Array1<f64>) -> f64 {
        let squared_distance = self.squared_distance(x1, x2);
        self.params.variance * (-squared_distance / (2.0 * self.params.lengthscale.powi(2))).exp()
    }

    /// Compute squared distance between two points
    fn squared_distance(&self, x1: &Array1<f64>, x2: &Array1<f64>) -> f64 {
        let diff = x1 - x2;
        diff.iter().map(|&x| x * x).sum()
    }

    /// Compute topology influence factor
    fn compute_topology_influence(&self, x1: &Array1<f64>, x2: &Array1<f64>) -> f64 {
        if self.topology_points.is_empty() {
            return 0.0;
        }

        // Find closest topology points
        let closest_to_x1 = self.find_closest_topology_point(x1);
        let closest_to_x2 = self.find_closest_topology_point(x2);

        // Compute topology-based similarity
        let topology_similarity = self.compute_topology_similarity(closest_to_x1, closest_to_x2);

        // Normalize influence
        topology_similarity.min(1.0).max(0.0)
    }

    /// Find closest topology point to given coordinate
    fn find_closest_topology_point(&self, x: &Array1<f64>) -> usize {
        let mut min_distance = f64::INFINITY;
        let mut closest_idx = 0;

        for (i, topology_point) in self.topology_points.iter().enumerate() {
            let distance = self.distance_to_topology_point(x, topology_point);
            if distance < min_distance {
                min_distance = distance;
                closest_idx = i;
            }
        }

        closest_idx
    }

    /// Compute distance to topology point
    fn distance_to_topology_point(&self, x: &Array1<f64>, topology_point: &(f64, f64, f64)) -> f64 {
        let (tx, ty, tz) = *topology_point;

        // Project topology point to same dimensionality as x
        let topology_array = if x.len() >= 3 {
            Array1::from_vec(vec![tx, ty, tz])
        } else {
            Array1::from_vec(vec![tx, ty])
        };

        let diff = x - &topology_array;
        diff.iter().map(|&val| val * val).sum::<f64>().sqrt()
    }

    /// Compute topology similarity between two topology points
    fn compute_topology_similarity(&self, idx1: usize, idx2: usize) -> f64 {
        if idx1 >= self.topology_points.len() || idx2 >= self.topology_points.len() {
            return 0.0;
        }

        let (x1, y1, z1) = self.topology_points[idx1];
        let (x2, y2, z2) = self.topology_points[idx2];

        // Compute geometric similarity in topology space
        let distance = ((x2 - x1).powi(2) + (y2 - y1).powi(2) + (z2 - z1).powi(2)).sqrt();

        // Convert distance to similarity (closer points = higher similarity)
        (-distance / self.params.lengthscale).exp()
    }

    /// Predict mean and variance for new points
    pub fn predict(
        &self,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
        x_test: &Array2<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        let n_train = x_train.nrows();
        let n_test = x_test.nrows();

        // Compute kernel matrices
        let k_train_train = self.compute_kernel_matrix(x_train, x_train);
        let k_test_train = self.compute_kernel_matrix(x_test, x_train);
        let k_test_test = self.compute_kernel_matrix(x_test, x_test);

        // Add noise to diagonal of training kernel
        let mut k_train_train_noisy = k_train_train.clone();
        for i in 0..n_train {
            k_train_train_noisy[[i, i]] += self.params.noise;
        }

        // Solve for alpha: (K + σ²I)α = y
        let alpha = self.solve_linear_system(&k_train_train_noisy, y_train)?;

        // Predict mean: μ* = K*T α
        let mut mean = Array1::zeros(n_test);
        for i in 0..n_test {
            for j in 0..n_train {
                mean[i] += k_test_train[[i, j]] * alpha[j];
            }
        }

        // Predict variance: σ²* = K** - K*T (K + σ²I)⁻¹ K*
        let mut variance = Array1::zeros(n_test);
        for i in 0..n_test {
            let mut var_sum = 0.0;
            for j in 0..n_train {
                for k in 0..n_train {
                    var_sum += k_test_train[[i, j]]
                        * self.kernel_inverse(&k_train_train_noisy, j, k)
                        * k_test_train[[i, k]];
                }
            }
            variance[i] = k_test_test[[i, i]] - var_sum;
            variance[i] = variance[i].max(0.0); // Ensure non-negative variance
        }

        Ok((mean, variance))
    }

    /// Solve linear system using Cholesky decomposition
    fn solve_linear_system(&self, k: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>> {
        // Simple iterative solver for small systems
        let n = k.nrows();
        let mut alpha = Array1::zeros(n);
        let mut residual = y.clone();

        // Iterative refinement
        for _ in 0..10 {
            let mut new_alpha = Array1::zeros(n);
            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..n {
                    if i != j {
                        sum += k[[i, j]] * alpha[j];
                    }
                }
                // Prevent division by zero
                let diagonal = k[[i, i]];
                if diagonal.abs() < 1e-12 {
                    anyhow::bail!(
                        "Kernel matrix has near-zero diagonal element - matrix is singular"
                    );
                }
                new_alpha[i] = (residual[i] - sum) / diagonal;
            }
            alpha = new_alpha;

            // Update residual
            residual = y.clone();
            for i in 0..n {
                for j in 0..n {
                    residual[i] -= k[[i, j]] * alpha[j];
                }
            }
        }

        Ok(alpha)
    }

    /// Get kernel inverse element (simplified)
    fn kernel_inverse(&self, k: &Array2<f64>, i: usize, j: usize) -> f64 {
        // Simplified inverse computation
        if i == j {
            let diagonal = k[[i, j]];
            if diagonal.abs() < 1e-12 {
                // Return a safe default for near-singular matrices
                1e12 // Large value indicates numerical instability
            } else {
                1.0 / diagonal
            }
        } else {
            0.0
        }
    }

    /// Update kernel parameters
    pub fn update_parameters(&mut self, params: KTwistKernelParams) {
        self.params = params;
    }

    /// Get current parameters
    pub fn get_parameters(&self) -> &KTwistKernelParams {
        &self.params
    }

    /// Compute log marginal likelihood
    pub fn log_marginal_likelihood(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let n = x.nrows();
        let k = self.compute_kernel_matrix(x, x);

        // Add noise to diagonal
        let mut k_noisy = k.clone();
        for i in 0..n {
            k_noisy[[i, i]] += self.params.noise;
        }

        // Compute log determinant (simplified)
        let log_det = self.compute_log_determinant(&k_noisy);

        // Compute quadratic form
        let alpha = self.solve_linear_system(&k_noisy, y)?;
        let quadratic_form = y.dot(&alpha);

        // Log marginal likelihood: -0.5 * (y^T K^-1 y + log|K| + n log(2π))
        let log_likelihood =
            -0.5 * (quadratic_form + log_det + n as f64 * (2.0 * std::f64::consts::PI).ln());

        Ok(log_likelihood)
    }

    /// Compute log determinant (simplified)
    fn compute_log_determinant(&self, k: &Array2<f64>) -> f64 {
        // Simplified log determinant computation
        let mut log_det = 0.0;
        for i in 0..k.nrows() {
            let diagonal = k[[i, i]];
            if diagonal <= 1e-12 {
                // Matrix is singular or near-singular
                // Return large negative value to indicate poor likelihood
                return -1e12;
            }
            log_det += diagonal.ln();
        }
        log_det
    }
}

/// Gaussian Process with K-Twist kernel
#[derive(Debug, Clone)]
pub struct KTwistGaussianProcess {
    kernel: KTwistRBFKernel,
    x_train: Option<Array2<f64>>,
    y_train: Option<Array1<f64>>,
}

impl KTwistGaussianProcess {
    /// Create a new K-Twist Gaussian Process
    pub fn new(kernel_params: KTwistKernelParams) -> Self {
        Self {
            kernel: KTwistRBFKernel::new(kernel_params),
            x_train: None,
            y_train: None,
        }
    }

    /// Fit the Gaussian Process to training data
    pub fn fit(&mut self, x: Array2<f64>, y: Array1<f64>) -> Result<()> {
        self.x_train = Some(x);
        self.y_train = Some(y);
        Ok(())
    }

    /// Predict on new data
    pub fn predict(&self, x_test: &Array2<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        let x_train = self
            .x_train
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model not fitted"))?;
        let y_train = self
            .y_train
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model not fitted"))?;

        self.kernel.predict(x_train, y_train, x_test)
    }

    /// Update topology information
    pub fn update_topology(&mut self, topology_points: Vec<(f64, f64, f64)>) {
        self.kernel.update_topology(topology_points);
    }

    /// Get kernel
    pub fn get_kernel(&self) -> &KTwistRBFKernel {
        &self.kernel
    }

    /// Get kernel mutably
    pub fn get_kernel_mut(&mut self) -> &mut KTwistRBFKernel {
        &mut self.kernel
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_kernel_computation() {
        let params = KTwistKernelParams::default();
        let kernel = KTwistRBFKernel::new(params);

        let x1 = array![[1.0, 2.0], [3.0, 4.0]];
        let x2 = array![[1.1, 2.1], [3.1, 4.1]];

        let k = kernel.compute_kernel_matrix(&x1, &x2);
        assert_eq!(k.shape(), [2, 2]);
        assert!(k[[0, 0]] > 0.0);
    }

    #[test]
    fn test_gaussian_process() {
        let params = KTwistKernelParams::default();
        let mut gp = KTwistGaussianProcess::new(params);

        let x_train = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y_train = array![1.0, 2.0, 3.0];

        assert!(gp.fit(x_train, y_train).is_ok());

        let x_test = array![[2.0, 3.0], [4.0, 5.0]];
        let (mean, variance) = gp.predict(&x_test).unwrap();

        assert_eq!(mean.len(), 2);
        assert_eq!(variance.len(), 2);
        assert!(variance.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_topology_influence() {
        let params = KTwistKernelParams::default();
        let mut kernel = KTwistRBFKernel::new(params);

        let topology_points = vec![(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)];
        kernel.update_topology(topology_points);

        let x1 = array![0.1, 0.1];
        let x2 = array![0.9, 0.9];

        let kernel_value = kernel.compute_kernel_value(&x1, &x2);
        assert!(kernel_value > 0.0);
    }
}
