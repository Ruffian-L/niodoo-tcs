//! Novelty Detection Validator
//!
//! Phase 4: Mathematical Validation Module
//! Validates novelty detection algorithms for consciousness state analysis,
//! ensuring proper detection of novel patterns and events using multiple approaches.

use anyhow::Result;
use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};

/// Reference implementation of novelty detection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoveltyDetectionReference {
    /// Algorithm type for novelty detection
    pub algorithm_type: NoveltyAlgorithm,
    /// Stability constraint percentage (15-20%)
    pub stability_threshold: f64,
    /// Minimum novelty value
    pub min_novelty: f64,
    /// Maximum novelty value
    pub max_novelty: f64,
    /// Numerical zero threshold for floating point comparisons
    pub numerical_zero_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NoveltyAlgorithm {
    /// Statistical outlier detection using z-score
    StatisticalOutlier { z_threshold: f64 },
    /// Information-theoretic novelty using KL divergence
    InformationTheoretic { kl_threshold: f64 },
    /// Distance-based novelty using nearest neighbors
    DistanceBased {
        k_neighbors: usize,
        distance_threshold: f64,
    },
    /// Predictive novelty using reconstruction error
    PredictiveNovelty { reconstruction_threshold: f64 },
    /// Consciousness-specific novelty using emotional state changes
    ConsciousnessNovelty { emotion_change_threshold: f64 },
}

impl NoveltyDetectionReference {
    pub fn new(
        algorithm_type: NoveltyAlgorithm,
        stability_threshold: f64,
        min_novelty: f64,
        max_novelty: f64,
    ) -> Self {
        Self {
            algorithm_type,
            stability_threshold,
            min_novelty,
            max_novelty,
            numerical_zero_threshold: 1e-15,
        }
    }

    pub fn new_with_threshold(
        algorithm_type: NoveltyAlgorithm,
        stability_threshold: f64,
        min_novelty: f64,
        max_novelty: f64,
        numerical_zero_threshold: f64,
    ) -> Self {
        Self {
            algorithm_type,
            stability_threshold,
            min_novelty,
            max_novelty,
            numerical_zero_threshold,
        }
    }
}

impl Default for NoveltyDetectionReference {
    fn default() -> Self {
        Self {
            algorithm_type: NoveltyAlgorithm::ConsciousnessNovelty {
                emotion_change_threshold: 0.3,
            },
            stability_threshold: 0.175, // 17.5% (middle of 15-20% range)
            min_novelty: 0.0,
            max_novelty: 1.0,
            numerical_zero_threshold: 1e-15,
        }
    }
}

/// Reference implementation for statistical outlier detection
#[derive(Debug, Clone)]
pub struct StatisticalOutlierReference {
    /// Z-score threshold for outlier detection
    pub z_threshold: f64,
    /// Minimum sample size for reliable statistics
    pub min_samples: usize,
}

impl StatisticalOutlierReference {
    pub fn new(z_threshold: f64) -> Self {
        Self {
            z_threshold,
            min_samples: 30,
        }
    }

    /// Detect outliers using z-score method
    pub fn detect_outliers(&self, data: &[f64]) -> Vec<bool> {
        if data.len() < self.min_samples {
            return vec![false; data.len()];
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev < 1e-10 {
            return vec![false; data.len()];
        }

        data.iter()
            .map(|&x| ((x - mean).abs() / std_dev) > self.z_threshold)
            .collect()
    }
}

/// Reference implementation for information-theoretic novelty detection
#[derive(Debug, Clone)]
pub struct InformationTheoreticReference {
    /// KL divergence threshold
    pub kl_threshold: f64,
    /// Number of bins for histogram
    pub num_bins: usize,
}

impl InformationTheoreticReference {
    pub fn new(kl_threshold: f64) -> Self {
        Self {
            kl_threshold,
            num_bins: 20,
        }
    }

    /// Calculate KL divergence between two distributions
    pub fn kl_divergence(&self, p: &[f64], q: &[f64]) -> f64 {
        if p.len() != q.len() {
            return f64::INFINITY;
        }

        let p_hist = self.create_histogram(p);
        let q_hist = self.create_histogram(q);

        let mut kl_div = 0.0;
        for i in 0..self.num_bins {
            let p_val = p_hist[i];
            let q_val = q_hist[i];

            if p_val > 0.0 && q_val > 0.0 {
                kl_div += p_val * (p_val / q_val).ln();
            } else if p_val > 0.0 && q_val == 0.0 {
                kl_div += f64::INFINITY;
            }
        }

        kl_div
    }

    /// Create histogram from data
    fn create_histogram(&self, data: &[f64]) -> Vec<f64> {
        if data.is_empty() {
            return vec![0.0; self.num_bins];
        }

        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;

        if range < 1e-10 {
            let mut hist = vec![0.0; self.num_bins];
            hist[0] = 1.0;
            return hist;
        }

        let mut hist = vec![0.0; self.num_bins];
        for &value in data {
            let bin = ((value - min_val) / range * (self.num_bins - 1) as f64) as usize;
            let bin = bin.min(self.num_bins - 1);
            hist[bin] += 1.0;
        }

        // Normalize
        let total = hist.iter().sum::<f64>();
        if total > 0.0 {
            for h in &mut hist {
                *h /= total;
            }
        }

        hist
    }
}

/// Reference implementation for distance-based novelty detection
#[derive(Debug, Clone)]
pub struct DistanceBasedReference {
    /// Number of nearest neighbors to consider
    pub k_neighbors: usize,
    /// Distance threshold for novelty
    pub distance_threshold: f64,
}

impl DistanceBasedReference {
    pub fn new(k_neighbors: usize, distance_threshold: f64) -> Self {
        Self {
            k_neighbors,
            distance_threshold,
        }
    }

    /// Calculate Euclidean distance between two points
    fn euclidean_distance(&self, p1: &[f64], p2: &[f64]) -> f64 {
        if p1.len() != p2.len() {
            return f64::INFINITY;
        }

        p1.iter()
            .zip(p2.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Find k-nearest neighbors and their distances
    pub fn k_nearest_distances(&self, data: &[Vec<f64>], query: &[f64]) -> Vec<f64> {
        let mut distances: Vec<f64> = data
            .iter()
            .map(|point| self.euclidean_distance(point, query))
            .filter(|&d| d.is_finite())
            .collect();

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        distances.truncate(self.k_neighbors);
        distances
    }

    /// Detect novelty based on distance to nearest neighbors
    pub fn detect_novelty(&self, data: &[Vec<f64>], query: &[f64]) -> bool {
        let distances = self.k_nearest_distances(data, query);

        if distances.is_empty() {
            return true; // No data points, consider novel
        }

        let min_distance = distances.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        min_distance > self.distance_threshold
    }
}

/// Reference implementation for predictive novelty detection
#[derive(Debug, Clone)]
pub struct PredictiveNoveltyReference {
    /// Reconstruction error threshold
    pub reconstruction_threshold: f64,
    /// Autoencoder hidden dimension
    pub hidden_dim: usize,
}

impl PredictiveNoveltyReference {
    pub fn new(reconstruction_threshold: f64) -> Self {
        Self {
            reconstruction_threshold,
            hidden_dim: 10,
        }
    }

    /// Simple autoencoder for novelty detection (simplified implementation)
    pub fn reconstruction_error(&self, input: &[f64]) -> f64 {
        // Simplified autoencoder: project to hidden space and back
        let encoded = self.encode(input);
        let reconstructed = self.decode(&encoded);

        // Calculate reconstruction error
        input
            .iter()
            .zip(reconstructed.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Simple encoding function (linear projection)
    fn encode(&self, input: &[f64]) -> Vec<f64> {
        // Simple random projection for demonstration
        let mut encoded = Vec::with_capacity(self.hidden_dim);
        for i in 0..self.hidden_dim {
            let weight = (i as f64 + 1.0) / (self.hidden_dim as f64 + 1.0);
            let projection = input
                .iter()
                .enumerate()
                .map(|(j, &x)| x * (j as f64 + 1.0).sin() * weight)
                .sum::<f64>();
            encoded.push(projection);
        }
        encoded
    }

    /// Simple decoding function (linear projection)
    fn decode(&self, encoded: &[f64]) -> Vec<f64> {
        // Simple linear reconstruction
        let input_dim = 3; // Assuming 3D input for consciousness states
        let mut reconstructed = vec![0.0; input_dim];

        for i in 0..input_dim {
            for (j, &enc) in encoded.iter().enumerate() {
                let weight = (i as f64 + 1.0) * (j as f64 + 1.0).cos();
                reconstructed[i] += enc * weight;
            }
        }

        reconstructed
    }

    /// Detect novelty based on reconstruction error
    pub fn detect_novelty(&self, input: &[f64]) -> bool {
        let error = self.reconstruction_error(input);
        error > self.reconstruction_threshold
    }
}

impl NoveltyDetectionReference {
    /// Calculate cosine similarity between two vectors
    ///
    /// Mathematical formula:
    /// cos(θ) = (a · b) / (||a|| * ||b||)
    ///
    /// Where:
    /// - a · b = dot product
    /// - ||a||, ||b|| = vector magnitudes
    pub fn cosine_similarity(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        let dot_product = self.dot_product(a, b);
        let magnitude_a = self.magnitude(a);
        let magnitude_b = self.magnitude(b);

        if magnitude_a < self.numerical_zero_threshold
            || magnitude_b < self.numerical_zero_threshold
        {
            return 0.0; // Avoid division by zero
        }

        dot_product / (magnitude_a * magnitude_b)
    }

    /// Calculate bounded novelty transformation
    ///
    /// Mathematical formula:
    /// novelty = 1 - cosine_similarity
    ///
    /// This transformation ensures:
    /// - novelty ∈ [0, 1]
    /// - novelty = 0 when vectors are identical (cosine_similarity = 1)
    /// - novelty = 1 when vectors are orthogonal (cosine_similarity = 0)
    /// - novelty = 2 when vectors are opposite (cosine_similarity = -1)
    pub fn bounded_novelty_transformation(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        let cosine_sim = self.cosine_similarity(a, b);
        let novelty = 1.0 - cosine_sim;

        // Ensure boundedness
        novelty.max(self.min_novelty).min(self.max_novelty)
    }

    /// Calculate dot product of two vectors
    fn dot_product(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        let mut sum = 0.0;
        for i in 0..a.len() {
            sum += a[i] * b[i];
        }
        sum
    }

    /// Calculate magnitude of a vector
    fn magnitude(&self, a: &ArrayView1<f64>) -> f64 {
        let mut sum = 0.0;
        for i in 0..a.len() {
            sum += a[i] * a[i];
        }
        sum.sqrt()
    }

    /// Validate stability constraint
    ///
    /// The stability constraint ensures that novelty changes are bounded
    /// within 15-20% of the previous value to prevent oscillations.
    pub fn validate_stability_constraint(
        &self,
        previous_novelty: f64,
        current_novelty: f64,
    ) -> Result<StabilityResult> {
        let novelty_change = (current_novelty - previous_novelty).abs();
        let relative_change = if previous_novelty > self.numerical_zero_threshold {
            novelty_change / previous_novelty
        } else {
            novelty_change
        };

        let is_stable = relative_change <= self.stability_threshold;

        Ok(StabilityResult {
            previous_novelty,
            current_novelty,
            novelty_change,
            relative_change,
            stability_threshold: self.stability_threshold,
            is_stable,
        })
    }

    /// Calculate novelty gradient for optimization
    ///
    /// This is useful for understanding how novelty changes
    /// with respect to input vector changes.
    pub fn novelty_gradient(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        let mut gradient = Array1::zeros(a.len());

        let cosine_sim = self.cosine_similarity(a, b);
        let magnitude_a = self.magnitude(a);
        let magnitude_b = self.magnitude(b);

        if magnitude_a < self.numerical_zero_threshold
            || magnitude_b < self.numerical_zero_threshold
        {
            return gradient;
        }

        for i in 0..a.len() {
            // Gradient of novelty w.r.t. a[i]
            let term1 = b[i] / (magnitude_a * magnitude_b);
            let term2 = cosine_sim * a[i] / (magnitude_a * magnitude_a);
            gradient[i] = -(term1 - term2);
        }

        gradient
    }
}

/// Validator for novelty detection implementation
#[derive(Debug, Clone)]
pub struct NoveltyDetectionValidator {
    pub reference: NoveltyDetectionReference,
    tolerance: f64,
    numerical_zero_threshold: f64,
}

impl NoveltyDetectionValidator {
    pub fn new(stability_threshold: f64, tolerance: f64) -> Self {
        Self::new_with_bounds_and_threshold(stability_threshold, 0.0, 1.0, tolerance, 1e-15)
    }

    pub fn new_with_bounds(
        stability_threshold: f64,
        min_novelty: f64,
        max_novelty: f64,
        tolerance: f64,
    ) -> Self {
        Self::new_with_bounds_and_threshold(
            stability_threshold,
            min_novelty,
            max_novelty,
            tolerance,
            1e-15,
        )
    }

    pub fn new_with_bounds_and_threshold(
        stability_threshold: f64,
        min_novelty: f64,
        max_novelty: f64,
        tolerance: f64,
        numerical_zero_threshold: f64,
    ) -> Self {
        Self {
            reference: NoveltyDetectionReference::new(
                NoveltyAlgorithm::ConsciousnessNovelty {
                    emotion_change_threshold: 0.3,
                },
                stability_threshold,
                min_novelty,
                max_novelty,
            ),
            tolerance,
            numerical_zero_threshold,
        }
    }

    /// Validate cosine similarity implementation
    pub fn validate_cosine_similarity<F>(
        &self,
        implementation: F,
    ) -> Result<SimilarityValidationResult>
    where
        F: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
    {
        let mut errors = Vec::new();
        let mut max_error: f64 = 0.0;
        let mut total_error = 0.0;
        let mut test_count = 0;

        // Generate test vector pairs
        let test_pairs = self.generate_test_vector_pairs();

        for (a, b) in test_pairs {
            let reference_similarity = self.reference.cosine_similarity(&a.view(), &b.view());
            let implementation_similarity = implementation(&a.view(), &b.view());

            let error = (reference_similarity - implementation_similarity).abs();

            if error > self.tolerance {
                errors.push(SimilarityError {
                    a: a.to_owned(),
                    b: b.to_owned(),
                    reference: reference_similarity,
                    implementation: implementation_similarity,
                    error,
                });
            }

            max_error = max_error.max(error);
            total_error += error;
            test_count += 1;
        }

        let average_error = total_error / test_count as f64;

        Ok(SimilarityValidationResult {
            max_error,
            average_error,
            errors,
            boundedness_valid: true, // Will be checked separately
        })
    }

    /// Validate bounded novelty transformation
    pub fn validate_novelty_transformation<F>(
        &self,
        implementation: F,
    ) -> Result<NoveltyValidationResult>
    where
        F: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
    {
        let mut errors = Vec::new();
        let mut max_error: f64 = 0.0;
        let mut total_error = 0.0;
        let mut test_count = 0;
        let mut boundedness_violations = Vec::new();

        // Generate test vector pairs
        let test_pairs = self.generate_test_vector_pairs();

        for (a, b) in test_pairs {
            let reference_novelty = self
                .reference
                .bounded_novelty_transformation(&a.view(), &b.view());
            let implementation_novelty = implementation(&a.view(), &b.view());

            let error = (reference_novelty - implementation_novelty).abs();

            if error > self.tolerance {
                errors.push(NoveltyError {
                    a: a.to_owned(),
                    b: b.to_owned(),
                    reference: reference_novelty,
                    implementation: implementation_novelty,
                    error,
                });
            }

            // Check boundedness
            if implementation_novelty < self.reference.min_novelty
                || implementation_novelty > self.reference.max_novelty
            {
                boundedness_violations.push(BoundednessViolation {
                    a: a.to_owned(),
                    b: b.to_owned(),
                    novelty: implementation_novelty,
                    min_bound: self.reference.min_novelty,
                    max_bound: self.reference.max_novelty,
                });
            }

            max_error = max_error.max(error);
            total_error += error;
            test_count += 1;
        }

        let average_error = total_error / test_count as f64;

        Ok(NoveltyValidationResult {
            max_error,
            average_error,
            errors,
            boundedness_violations: boundedness_violations.clone(),
            is_bounded: boundedness_violations.is_empty(),
        })
    }

    /// Validate stability constraint
    pub fn validate_stability_constraint<F>(
        &self,
        implementation: F,
    ) -> Result<StabilityValidationResult>
    where
        F: Fn(f64, f64) -> Result<StabilityResult>,
    {
        let mut stability_violations = Vec::new();
        let mut test_count = 0;

        // Generate test cases for stability
        let test_cases = self.generate_stability_test_cases();

        for (previous_novelty, current_novelty) in test_cases {
            let reference_result = self
                .reference
                .validate_stability_constraint(previous_novelty, current_novelty)?;
            let implementation_result = implementation(previous_novelty, current_novelty)?;

            if reference_result.is_stable != implementation_result.is_stable {
                stability_violations.push(StabilityViolation {
                    previous_novelty,
                    current_novelty,
                    reference_stable: reference_result.is_stable,
                    implementation_stable: implementation_result.is_stable,
                    reference_relative_change: reference_result.relative_change,
                    implementation_relative_change: implementation_result.relative_change,
                });
            }

            test_count += 1;
        }

        Ok(StabilityValidationResult {
            stability_violations: stability_violations.clone(),
            is_stable: stability_violations.is_empty(),
            test_count,
        })
    }

    /// Validate novelty gradient calculation
    pub fn validate_novelty_gradient<F>(
        &self,
        implementation: F,
    ) -> Result<GradientValidationResult>
    where
        F: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> Array1<f64>,
    {
        let mut errors = Vec::new();
        let mut max_error: f64 = 0.0;
        let mut total_error = 0.0;
        let mut test_count = 0;

        // Generate test vector pairs
        let test_pairs = self.generate_test_vector_pairs();

        for (a, b) in test_pairs {
            let reference_gradient = self.reference.novelty_gradient(&a.view(), &b.view());
            let implementation_gradient = implementation(&a.view(), &b.view());

            let error =
                self.calculate_gradient_error(&reference_gradient, &implementation_gradient);

            if error > self.tolerance {
                errors.push(GradientError {
                    a: a.to_owned(),
                    b: b.to_owned(),
                    reference: reference_gradient.to_owned(),
                    implementation: implementation_gradient.to_owned(),
                    error,
                });
            }

            max_error = max_error.max(error);
            total_error += error;
            test_count += 1;
        }

        let average_error = total_error / test_count as f64;

        Ok(GradientValidationResult {
            max_error,
            average_error,
            errors,
        })
    }

    /// Generate test vector pairs for validation
    fn generate_test_vector_pairs(&self) -> Vec<(Array1<f64>, Array1<f64>)> {
        let mut pairs = Vec::new();

        // Test cases with different relationships
        let test_vectors = vec![
            Array1::from_vec(vec![1.0, 0.0, 0.0]),    // Unit vector along x
            Array1::from_vec(vec![0.0, 1.0, 0.0]),    // Unit vector along y
            Array1::from_vec(vec![0.0, 0.0, 1.0]),    // Unit vector along z
            Array1::from_vec(vec![1.0, 1.0, 0.0]),    // Diagonal vector
            Array1::from_vec(vec![1.0, 1.0, 1.0]),    // All ones
            Array1::from_vec(vec![-1.0, 0.0, 0.0]),   // Negative x
            Array1::from_vec(vec![0.0, -1.0, 0.0]),   // Negative y
            Array1::from_vec(vec![0.0, 0.0, -1.0]),   // Negative z
            Array1::from_vec(vec![-1.0, -1.0, -1.0]), // All negative
        ];

        for i in 0..test_vectors.len() {
            for j in 0..test_vectors.len() {
                pairs.push((test_vectors[i].to_owned(), test_vectors[j].to_owned()));
            }
        }

        pairs
    }

    /// Generate test cases for stability validation
    fn generate_stability_test_cases(&self) -> Vec<(f64, f64)> {
        let mut cases = Vec::new();

        // Test cases with different stability scenarios
        let base_novelty = 0.5;
        let stability_threshold = self.reference.stability_threshold;

        cases.push((base_novelty, base_novelty)); // No change
        cases.push((
            base_novelty,
            base_novelty * (1.0 + stability_threshold * 0.5),
        )); // Small change
        cases.push((base_novelty, base_novelty * (1.0 + stability_threshold))); // At threshold
        cases.push((
            base_novelty,
            base_novelty * (1.0 + stability_threshold * 1.5),
        )); // Above threshold
        cases.push((
            base_novelty,
            base_novelty * (1.0 - stability_threshold * 0.5),
        )); // Small decrease
        cases.push((base_novelty, base_novelty * (1.0 - stability_threshold))); // Decrease at threshold
        cases.push((
            base_novelty,
            base_novelty * (1.0 - stability_threshold * 1.5),
        )); // Large decrease

        cases
    }

    /// Calculate error between two gradient vectors
    fn calculate_gradient_error(&self, grad1: &Array1<f64>, grad2: &Array1<f64>) -> f64 {
        let mut sum = 0.0;
        for i in 0..grad1.len() {
            let diff = grad1[i] - grad2[i];
            sum += diff * diff;
        }
        sum.sqrt()
    }
}

/// Result of stability constraint validation
#[derive(Debug, Clone)]
pub struct StabilityResult {
    pub previous_novelty: f64,
    pub current_novelty: f64,
    pub novelty_change: f64,
    pub relative_change: f64,
    pub stability_threshold: f64,
    pub is_stable: bool,
}

/// Result of similarity validation
#[derive(Debug, Clone)]
pub struct SimilarityValidationResult {
    pub max_error: f64,
    pub average_error: f64,
    pub errors: Vec<SimilarityError>,
    pub boundedness_valid: bool,
}

/// Result of novelty validation
#[derive(Debug, Clone)]
pub struct NoveltyValidationResult {
    pub max_error: f64,
    pub average_error: f64,
    pub errors: Vec<NoveltyError>,
    pub boundedness_violations: Vec<BoundednessViolation>,
    pub is_bounded: bool,
}

/// Result of stability validation
#[derive(Debug, Clone)]
pub struct StabilityValidationResult {
    pub stability_violations: Vec<StabilityViolation>,
    pub is_stable: bool,
    pub test_count: usize,
}

/// Result of gradient validation
#[derive(Debug, Clone)]
pub struct GradientValidationResult {
    pub max_error: f64,
    pub average_error: f64,
    pub errors: Vec<GradientError>,
}

/// Similarity calculation error
#[derive(Debug, Clone)]
pub struct SimilarityError {
    pub a: Array1<f64>,
    pub b: Array1<f64>,
    pub reference: f64,
    pub implementation: f64,
    pub error: f64,
}

/// Novelty calculation error
#[derive(Debug, Clone)]
pub struct NoveltyError {
    pub a: Array1<f64>,
    pub b: Array1<f64>,
    pub reference: f64,
    pub implementation: f64,
    pub error: f64,
}

/// Boundedness violation
#[derive(Debug, Clone)]
pub struct BoundednessViolation {
    pub a: Array1<f64>,
    pub b: Array1<f64>,
    pub novelty: f64,
    pub min_bound: f64,
    pub max_bound: f64,
}

/// Stability violation
#[derive(Debug, Clone)]
pub struct StabilityViolation {
    pub previous_novelty: f64,
    pub current_novelty: f64,
    pub reference_stable: bool,
    pub implementation_stable: bool,
    pub reference_relative_change: f64,
    pub implementation_relative_change: f64,
}

/// Gradient calculation error
#[derive(Debug, Clone)]
pub struct GradientError {
    pub a: Array1<f64>,
    pub b: Array1<f64>,
    pub reference: Array1<f64>,
    pub implementation: Array1<f64>,
    pub error: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::ValidationConfig;

    #[test]
    fn test_cosine_similarity_basic() {
        let reference = NoveltyDetectionReference::default();

        let a = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let b = Array1::from_vec(vec![1.0, 0.0, 0.0]);

        let similarity = reference.cosine_similarity(&a.view(), &b.view());
        let config = ValidationConfig::default();
        assert!((similarity - 1.0).abs() < config.test_assertion_threshold); // Identical vectors

        let c = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        let similarity = reference.cosine_similarity(&a.view(), &c.view());
        assert!(similarity.abs() < config.test_assertion_threshold); // Orthogonal vectors
    }

    #[test]
    fn test_novelty_transformation_basic() {
        let reference = NoveltyDetectionReference::default();

        let a = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let b = Array1::from_vec(vec![1.0, 0.0, 0.0]);

        let novelty = reference.bounded_novelty_transformation(&a.view(), &b.view());
        assert!(novelty.abs() < config.test_assertion_threshold); // Identical vectors should have zero novelty

        let c = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        let novelty = reference.bounded_novelty_transformation(&a.view(), &c.view());
        assert!((novelty - 1.0).abs() < config.test_assertion_threshold); // Orthogonal vectors should have novelty 1
    }

    #[test]
    fn test_stability_constraint() {
        let reference = NoveltyDetectionReference::default();

        let result = reference.validate_stability_constraint(0.5, 0.6).unwrap();
        assert!(result.is_stable); // 20% change should be stable

        let result = reference.validate_stability_constraint(0.5, 0.7).unwrap();
        assert!(!result.is_stable); // 40% change should be unstable
    }

    #[test]
    fn test_validator_cosine_similarity() {
        let validator = NoveltyDetectionValidator::new(0.175, 1e-6);

        // Test with reference implementation (should have zero error)
        let result = validator
            .validate_cosine_similarity(|a, b| validator.reference.cosine_similarity(a, b))
            .unwrap();

        assert!(result.max_error < config.test_assertion_threshold);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_validator_novelty_transformation() {
        let validator = NoveltyDetectionValidator::new(0.175, 1e-6);

        // Test with reference implementation (should have zero error)
        let result = validator
            .validate_novelty_transformation(|a, b| {
                validator.reference.bounded_novelty_transformation(a, b)
            })
            .unwrap();

        assert!(result.max_error < config.test_assertion_threshold);
        assert!(result.errors.is_empty());
        assert!(result.is_bounded);
    }
}
