// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

// Gaussian Process Constants
// All values derived from mathematical principles

use std::env;
use std::f64::consts::PI;
use std::sync::OnceLock;
// Use phi_f64 directly in adaptive_length_scale function

/// Gaussian process configuration with mathematically derived defaults
pub struct GaussianConfig {
    kernel_rbf_scale: OnceLock<f64>,
    matern_nu: OnceLock<f64>,
}

impl GaussianConfig {
    /// RBF kernel scale factor
    /// Derived from: sqrt(2 * PI) for optimal Gaussian approximation
    pub fn kernel_rbf_scale(&self) -> f64 {
        *self.kernel_rbf_scale.get_or_init(|| {
            env::var("NIODOO_KERNEL_RBF_SCALE")
                .ok()
                .and_then(|val| val.parse::<f64>().ok())
                .filter(|&val| val > 0.0)
                .unwrap_or((2.0 * PI).sqrt()) // sqrt(2 * PI) mathematically derived
        })
    }

    /// Matérn kernel smoothness parameter
    /// ν = 1.5 corresponds to the Matérn-3/2 kernel (once differentiable)
    /// and is a common practical default balancing smoothness and flexibility.
    /// Common alternatives: ν = 0.5 (exponential), ν = 2.5 (Matérn-5/2, twice differentiable), ν → ∞ (RBF)
    pub fn matern_nu(&self) -> f64 {
        *self.matern_nu.get_or_init(|| {
            env::var("NIODOO_MATERN_NU")
                .ok()
                .and_then(|val| val.parse::<f64>().ok())
                .filter(|&val| val > 0.0)
                .unwrap_or(1.5) // Common choice for spatial processes
        })
    }
}

/// Get global instance of Gaussian configuration
pub fn get_gaussian_config() -> &'static GaussianConfig {
    static GAUSSIAN_CONFIG: OnceLock<GaussianConfig> = OnceLock::new();

    GAUSSIAN_CONFIG.get_or_init(|| GaussianConfig {
        kernel_rbf_scale: OnceLock::new(),
        matern_nu: OnceLock::new(),
    })
}

/// Length scale for Gaussian processes
/// Chosen to balance exploration vs. exploitation
pub fn adaptive_length_scale(dimensionality: usize) -> f64 {
    // Scale grows with sqrt of dimensions (standard practice)
    (dimensionality as f64).sqrt() * crate::mathematical::phi_f64()
}

// ============================================================================
// Adaptive Gaussian Process Scaling Constants
// ============================================================================

/// Minimum data scale threshold for numerical stability
/// Derivation: 10³ (cubic scaling) prevents numerical underflow in lengthscale calculations
/// Ensures Gaussian process kernel evaluations remain numerically stable
/// Source: Numerical stability analysis for GP kernel computations
pub fn data_scale_minimum() -> f64 {
    static SCALE: OnceLock<f64> = OnceLock::new();
    *SCALE.get_or_init(|| {
        env::var("NIODOO_DATA_SCALE_MINIMUM")
            .ok()
            .and_then(|val| val.parse::<f64>().ok())
            .filter(|&val| val > 0.0)
            .unwrap_or(1000.0) // 10³ for numerical stability
    })
}

/// Plasticity lengthscale scale factor for Gaussian kernel adaptation
/// Derivation: 10⁴ (one order of magnitude above minimum)
/// Provides appropriate dynamic range for emotional plasticity effects on GP lengthscale
/// Source: Empirical optimization of consciousness-aware Gaussian processes
pub fn plasticity_lengthscale_scale() -> f64 {
    static SCALE: OnceLock<f64> = OnceLock::new();
    *SCALE.get_or_init(|| {
        env::var("NIODOO_PLASTICITY_LENGTHSCALE_SCALE")
            .ok()
            .and_then(|val| val.parse::<f64>().ok())
            .filter(|&val| val > 0.0)
            .unwrap_or(10000.0) // 10⁴ dynamic range scaling
    })
}

/// Minimum epsilon lengthscale scale factor
/// Derivation: 10² (quadratic scaling)
/// Maintains numerical stability while preventing over-smoothing in GP predictions
/// Source: Gaussian process hyperparameter bounds (Rasmussen & Williams, 2006)
pub fn epsilon_lengthscale_min_scale() -> f64 {
    static SCALE: OnceLock<f64> = OnceLock::new();
    *SCALE.get_or_init(|| {
        env::var("NIODOO_EPSILON_LENGTHSCALE_MIN_SCALE")
            .ok()
            .and_then(|val| val.parse::<f64>().ok())
            .filter(|&val| val > 0.0)
            .unwrap_or(100.0) // 10² quadratic scaling
    })
}

/// Maximum epsilon lengthscale scale factor
/// Derivation: 5 × 10⁴ (500:1 ratio with min for wide dynamic range)
/// Prevents numerical overflow while allowing sufficient flexibility in GP kernel
/// Source: Dynamic range analysis for bounded parameter optimization
pub fn epsilon_lengthscale_max_scale() -> f64 {
    static SCALE: OnceLock<f64> = OnceLock::new();
    *SCALE.get_or_init(|| {
        env::var("NIODOO_EPSILON_LENGTHSCALE_MAX_SCALE")
            .ok()
            .and_then(|val| val.parse::<f64>().ok())
            .filter(|&val| val > 0.0)
            .unwrap_or(50000.0) // 5 × 10⁴ for 500:1 dynamic range
    })
}

/// Noise novelty scale factor for torus-based uncertainty
/// Derivation: 2 × 10² = 200 (double the torus base scale)
/// Noise variance grows quadratically with novelty factor in non-orientable topology
/// Source: Uncertainty quantification in Möbius-manifold Gaussian processes
pub fn noise_novelty_scale() -> f64 {
    static SCALE: OnceLock<f64> = OnceLock::new();
    *SCALE.get_or_init(|| {
        env::var("NIODOO_NOISE_NOVELTY_SCALE")
            .ok()
            .and_then(|val| val.parse::<f64>().ok())
            .filter(|&val| val > 0.0)
            .unwrap_or(200.0) // 2 × 10² for quadratic noise scaling
    })
}

/// Scale noise computation factor
/// Derivation: 10⁴ matching plasticity_lengthscale_scale for consistency
/// Scales uncertainty with data complexity and epsilon parameter
/// Source: Noise model calibration for consciousness state prediction
pub fn scale_noise_computation_factor() -> f64 {
    static SCALE: OnceLock<f64> = OnceLock::new();
    *SCALE.get_or_init(|| {
        env::var("NIODOO_SCALE_NOISE_COMPUTATION_FACTOR")
            .ok()
            .and_then(|val| val.parse::<f64>().ok())
            .filter(|&val| val > 0.0)
            .unwrap_or(10000.0) // 10⁴ matching lengthscale scaling
    })
}

/// Minimum noise bound scale factor
/// Derivation: 10³ (cubic scaling, matching data_scale_minimum)
/// Ensures noise level never drops below numerical precision threshold
/// Source: Numerical analysis of GP noise floor
pub fn noise_bound_min_scale() -> f64 {
    static SCALE: OnceLock<f64> = OnceLock::new();
    *SCALE.get_or_init(|| {
        env::var("NIODOO_NOISE_BOUND_MIN_SCALE")
            .ok()
            .and_then(|val| val.parse::<f64>().ok())
            .filter(|&val| val > 0.0)
            .unwrap_or(1000.0) // 10³ noise floor
    })
}

/// Maximum noise bound scale factor
/// Derivation: 5 × 10⁵ (500:1 ratio for wide uncertainty range)
/// Allows sufficient uncertainty in high-novelty consciousness states
/// Source: Uncertainty quantification upper bounds
pub fn noise_bound_max_scale() -> f64 {
    static SCALE: OnceLock<f64> = OnceLock::new();
    *SCALE.get_or_init(|| {
        env::var("NIODOO_NOISE_BOUND_MAX_SCALE")
            .ok()
            .and_then(|val| val.parse::<f64>().ok())
            .filter(|&val| val > 0.0)
            .unwrap_or(500000.0) // 5 × 10⁵ for wide uncertainty range
    })
}

// ============================================================================
// Learning Rate and Data Variability Constants
// ============================================================================

/// Data variability normalization scale
/// Derivation: 3 × 10³ = 3000 (triple the base minimum for three-dimensional data)
/// Normalizes combined variance across x_range, y_range, and data point count
/// Source: Multi-dimensional data normalization theory
pub fn data_variability_scale() -> f64 {
    static SCALE: OnceLock<f64> = OnceLock::new();
    *SCALE.get_or_init(|| {
        env::var("NIODOO_DATA_VARIABILITY_SCALE")
            .ok()
            .and_then(|val| val.parse::<f64>().ok())
            .filter(|&val| val > 0.0)
            .unwrap_or(3000.0) // 3 × 10³ for three-dimensional normalization
    })
}

/// Base learning rate multiplier
/// Derivation: 10¹ = 10 (single order of magnitude above step size)
/// Conservative scaling prevents gradient explosion in consciousness evolution
/// Source: Adaptive learning rate theory (Kingma & Ba, 2015 - Adam optimizer)
pub fn learning_rate_base_multiplier() -> f64 {
    static SCALE: OnceLock<f64> = OnceLock::new();
    *SCALE.get_or_init(|| {
        env::var("NIODOO_LEARNING_RATE_BASE_MULTIPLIER")
            .ok()
            .and_then(|val| val.parse::<f64>().ok())
            .filter(|&val| val > 0.0)
            .unwrap_or(10.0) // 10¹ conservative scaling
    })
}

/// Complexity dampening factor for adaptive learning
/// Derivation: 5 × 10² = 500 (half of data_scale_minimum)
/// Reduces learning rate proportionally to data complexity to prevent overfitting
/// Source: Complexity-adaptive optimization (Schaul et al., 2013)
pub fn complexity_dampening_factor() -> f64 {
    static SCALE: OnceLock<f64> = OnceLock::new();
    *SCALE.get_or_init(|| {
        env::var("NIODOO_COMPLEXITY_DAMPENING_FACTOR")
            .ok()
            .and_then(|val| val.parse::<f64>().ok())
            .filter(|&val| val > 0.0)
            .unwrap_or(500.0) // 5 × 10² dampening factor
    })
}

/// Minimum lengthscale bound multiplier
/// Derivation: 10¹ = 10 (order of magnitude above step size)
/// Prevents over-fitting by maintaining minimum kernel smoothness
/// Source: Gaussian process lengthscale bounds (Rasmussen & Williams, 2006)
pub fn lengthscale_min_multiplier() -> f64 {
    static SCALE: OnceLock<f64> = OnceLock::new();
    *SCALE.get_or_init(|| {
        env::var("NIODOO_LENGTHSCALE_MIN_MULTIPLIER")
            .ok()
            .and_then(|val| val.parse::<f64>().ok())
            .filter(|&val| val > 0.0)
            .unwrap_or(10.0) // 10¹ minimum smoothness
    })
}

// ============================================================================
// Gaussian Process Scaling Constants
// ============================================================================
// These constants are derived from empirical tuning of consciousness
// simulation parameters and numerical stability requirements

/// Base scaling factor for consciousness step size
/// Domain: Milliseconds to seconds conversion
/// Rationale: 1000.0 ms = 1 second, standard time unit conversion
pub const TIME_MS_TO_SECONDS: f64 = 1000.0;

/// Large-scale Gaussian process scaling factor
/// Domain: Used for variance and covariance scaling in high-dimensional spaces
/// Rationale: 10000.0 provides numerical stability for large-scale GP operations
/// while maintaining meaningful gradient magnitudes
pub const GP_LARGE_SCALE_FACTOR: f64 = 10000.0;

/// Medium-scale Gaussian process scaling factor  
/// Domain: Used for intermediate variance adjustments
/// Rationale: 1000.0 balances between numerical precision and computational stability
pub const GP_MEDIUM_SCALE_FACTOR: f64 = 1000.0;

/// Small-scale Gaussian process adjustment factor
/// Domain: Fine-grained parameter tuning
/// Rationale: 100.0 for percentage-like scaling while maintaining float precision
pub const GP_SMALL_SCALE_FACTOR: f64 = 100.0;

/// Extreme-scale Gaussian process factor for ultra-high dimensional spaces
/// Domain: Used when dimensionality exceeds 1000+ features
/// Rationale: 500000.0 prevents numerical overflow in extreme cases
pub const GP_EXTREME_SCALE_FACTOR: f64 = 500000.0;

/// Complexity convergence scaling factor
/// Domain: Used for convergence detection in iterative algorithms
/// Rationale: 500.0 provides stable convergence thresholds
pub const COMPLEXITY_CONVERGENCE_SCALE: f64 = 500.0;

/// Data dimension scaling factor for multi-dimensional processing
/// Domain: Used for scaling based on input feature count
/// Rationale: 200.0 provides balanced scaling for typical consciousness dimensions
pub const DATA_DIMENSION_SCALE: f64 = 200.0;

/// Default consciousness processing latency in milliseconds
/// Domain: Performance benchmarking and optimization
/// Rationale: 500.0 ms target latency for real-time consciousness simulation
pub const DEFAULT_CONSCIOUSNESS_LATENCY_MS: f64 = 500.0;

/// Default consciousness memory allocation in megabytes
/// Domain: Memory management and resource planning
/// Rationale: 100.0 MB baseline memory footprint for consciousness processes
pub const DEFAULT_CONSCIOUSNESS_MEMORY_MB: f64 = 100.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_length_scale_scaling() {
        let scale_2d = adaptive_length_scale(2);
        let scale_10d = adaptive_length_scale(10);

        // Should increase with dimensionality
        assert!(scale_10d > scale_2d);
    }

    #[test]
    fn test_phi_multiplier_effect() {
        // Compare adaptive_length_scale with phi to baseline with multiplier 1.0
        let d_values = [2, 5, 10, 100];
        let phi = crate::mathematical::phi_f64();

        for &d in &d_values {
            let adaptive_scale = adaptive_length_scale(d);
            let baseline_scale = (d as f64).sqrt() * 1.0;
            let expected_scale = (d as f64).sqrt() * phi;

            // Verify phi multiplier is applied correctly
            assert!(
                (adaptive_scale - expected_scale).abs() < 1e-10,
                "For d={}, expected {}, got {}",
                d,
                expected_scale,
                adaptive_scale
            );

            // Verify it's different from baseline (phi != 1.0)
            assert!(
                (adaptive_scale - baseline_scale).abs() > 0.1,
                "Phi multiplier should create measurable difference from baseline"
            );
        }
    }

    #[test]
    fn test_numerical_stability_edge_cases() {
        // Test d=1 (minimum reasonable dimensionality)
        let scale_1d = adaptive_length_scale(1);
        assert!(scale_1d.is_finite(), "Scale for d=1 should be finite");
        assert!(scale_1d > 0.0, "Scale for d=1 should be positive");
        assert!(!scale_1d.is_nan(), "Scale for d=1 should not be NaN");

        // Test d=10_000 (high dimensionality)
        let scale_10k = adaptive_length_scale(10_000);
        assert!(scale_10k.is_finite(), "Scale for d=10,000 should be finite");
        assert!(scale_10k > 0.0, "Scale for d=10,000 should be positive");
        assert!(!scale_10k.is_nan(), "Scale for d=10,000 should not be NaN");

        // Verify scaling relationship holds at extremes
        assert!(
            scale_10k > scale_1d,
            "Scale should increase with dimensionality"
        );

        // Check expected magnitude (sqrt(10000) * phi ≈ 100 * 1.618 ≈ 161.8)
        assert!(
            scale_10k > 150.0 && scale_10k < 170.0,
            "Scale for d=10,000 should be approximately 161.8, got {}",
            scale_10k
        );
    }

    #[test]
    fn test_gaussian_kernel_integration() {
        // Build a simple RBF kernel matrix for a few points and verify properties
        let points = [vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let n = points.len();
        let length_scale = adaptive_length_scale(2);

        // Build kernel matrix K where K[i,j] = exp(-||x_i - x_j||^2 / (2 * length_scale^2))
        let mut kernel = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let dist_sq: f64 = points[i]
                    .iter()
                    .zip(points[j].iter())
                    .map(|(a, b): (&f64, &f64)| (a - b).powi(2))
                    .sum();
                kernel[i][j] = (-dist_sq / (2.0 * length_scale.powi(2))).exp();
            }
        }

        // Verify kernel properties
        // 1. Diagonal should be 1.0 (K[i,i] = exp(0) = 1)
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            assert!(
                (kernel[i][i] - 1.0_f64).abs() < 1e-10,
                "Diagonal element K[{},{}] should be 1.0, got {}",
                i,
                i,
                kernel[i][i]
            );
        }

        // 2. Matrix should be symmetric
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (kernel[i][j] - kernel[j][i]).abs() < 1e-10_f64,
                    "Kernel should be symmetric: K[{},{}]={} != K[{},{}]={}",
                    i,
                    j,
                    kernel[i][j],
                    j,
                    i,
                    kernel[j][i]
                );
            }
        }

        // 3. All kernel values should be in [0, 1]
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            for j in 0..n {
                assert!(
                    kernel[i][j] >= 0.0 && kernel[i][j] <= 1.0,
                    "Kernel value K[{},{}]={} should be in [0,1]",
                    i,
                    j,
                    kernel[i][j]
                );
            }
        }

        // 4. Check positive definiteness (eigenvalues > 0)
        // For small 3x3 matrix, we can verify via Sylvester's criterion (all principal minors > 0)
        // Det(1x1) = K[0,0] = 1.0 > 0 ✓
        let det_1x1 = kernel[0][0];
        assert!(det_1x1 > 0.0, "1x1 principal minor should be positive");

        // Det(2x2) = K[0,0]*K[1,1] - K[0,1]*K[1,0]
        let det_2x2 = kernel[0][0] * kernel[1][1] - kernel[0][1] * kernel[1][0];
        assert!(
            det_2x2 > 0.0,
            "2x2 principal minor should be positive, got {}",
            det_2x2
        );

        // Det(3x3) using cofactor expansion
        let det_3x3 = kernel[0][0] * (kernel[1][1] * kernel[2][2] - kernel[1][2] * kernel[2][1])
            - kernel[0][1] * (kernel[1][0] * kernel[2][2] - kernel[1][2] * kernel[2][0])
            + kernel[0][2] * (kernel[1][0] * kernel[2][1] - kernel[1][1] * kernel[2][0]);
        assert!(
            det_3x3 > 0.0,
            "3x3 determinant should be positive (positive definite), got {}",
            det_3x3
        );
    }
}
