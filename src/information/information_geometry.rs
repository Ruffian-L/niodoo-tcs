/*
 * 📊 INFORMATION GEOMETRY IMPLEMENTATION 📊
 *
 * Implements Information Geometry for principled learning and belief updating
 * using Bayesian Surprise and natural gradient optimization.
 *
 * Based on "The Geometry of Thought" framework:
 * - Fisher Information Metric for principled distance measures
 * - Bayesian Surprise (KL divergence) as learning signal
 * - Natural gradient descent on statistical manifolds
 * - Statistical manifolds for probability distributions
 */

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::mem;

/// Statistical manifold for probability distributions
#[derive(Debug, Clone)]
pub struct StatisticalManifold {
    pub distributions: Vec<ProbabilityDistribution>,
    pub fisher_metric: Array2<f64>,
    pub dimension: usize,
}

/// Probability distribution representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilityDistribution {
    pub parameters: Array1<f64>,
    pub distribution_type: DistributionType,
    pub support: (f64, f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionType {
    Gaussian,
    Beta,
    Dirichlet,
    Exponential,
    Poisson,
}

impl ProbabilityDistribution {
    pub fn new_gaussian(mean: f64, variance: f64) -> Self {
        Self {
            parameters: Array1::from_vec(vec![mean, variance]),
            distribution_type: DistributionType::Gaussian,
            support: (f64::NEG_INFINITY, f64::INFINITY),
        }
    }

    pub fn new_beta(alpha: f64, beta: f64) -> Self {
        Self {
            parameters: Array1::from_vec(vec![alpha, beta]),
            distribution_type: DistributionType::Beta,
            support: (0.0, 1.0),
        }
    }

    /// Calculate probability density at point x
    pub fn pdf(&self, x: f64) -> f64 {
        match self.distribution_type {
            DistributionType::Gaussian => {
                let mean = self.parameters[0];
                let variance = self.parameters[1];
                let std_dev = variance.sqrt();
                (1.0 / (std_dev * (2.0 * PI).sqrt()))
                    * (-0.5 * ((x - mean) / std_dev).powi(2)).exp()
            }
            DistributionType::Beta => {
                let alpha = self.parameters[0];
                let beta = self.parameters[1];
                if !(0.0..=1.0).contains(&x) {
                    0.0
                } else {
                    (x.powf(alpha - 1.0) * (1.0 - x).powf(beta - 1.0))
                        / self.beta_function(alpha, beta)
                }
            }
            DistributionType::Exponential => {
                let lambda = self.parameters[0];
                if x < 0.0 {
                    0.0
                } else {
                    lambda * (-lambda * x).exp()
                }
            }
            _ => 0.0, // Simplified for other distributions
        }
    }

    /// Calculate log-likelihood
    pub fn log_likelihood(&self, x: f64) -> f64 {
        self.pdf(x).ln()
    }

    /// Beta function for Beta distribution
    fn beta_function(&self, alpha: f64, beta: f64) -> f64 {
        // Approximation using gamma function
        self.gamma_function(alpha) * self.gamma_function(beta) / self.gamma_function(alpha + beta)
    }

    /// Gamma function approximation
    fn gamma_function(&self, z: f64) -> f64 {
        // Stirling's approximation
        (2.0 * PI / z).sqrt() * (z / std::f64::consts::E).powf(z)
    }
}

/// Fisher Information Metric calculator
pub struct FisherInformationMetric;

impl FisherInformationMetric {
    /// Calculate Fisher information matrix for a distribution
    pub fn calculate_fisher_matrix(&self, distribution: &ProbabilityDistribution) -> Array2<f64> {
        let dim = distribution.parameters.len();
        let mut fisher_matrix = Array2::zeros((dim, dim));

        match distribution.distribution_type {
            DistributionType::Gaussian => {
                let variance = distribution.parameters[1];
                fisher_matrix[[0, 0]] = 1.0 / variance; // I(μ,μ)
                fisher_matrix[[1, 1]] = 1.0 / (2.0 * variance.powi(2)); // I(σ²,σ²)
            }
            DistributionType::Beta => {
                let alpha = distribution.parameters[0];
                let beta = distribution.parameters[1];
                fisher_matrix[[0, 0]] = self.psi_prime(alpha) - self.psi_prime(alpha + beta);
                fisher_matrix[[1, 1]] = self.psi_prime(beta) - self.psi_prime(alpha + beta);
                fisher_matrix[[0, 1]] = -self.psi_prime(alpha + beta);
                fisher_matrix[[1, 0]] = -self.psi_prime(alpha + beta);
            }
            _ => {
                // Default identity matrix
                for i in 0..dim {
                    fisher_matrix[[i, i]] = 1.0;
                }
            }
        }

        fisher_matrix
    }

    /// Calculate distance between two distributions using Fisher metric
    pub fn fisher_distance(
        &self,
        dist1: &ProbabilityDistribution,
        dist2: &ProbabilityDistribution,
    ) -> f64 {
        let fisher_matrix = self.calculate_fisher_matrix(dist1);
        let param_diff = &dist2.parameters - &dist1.parameters;

        // d² = (θ₂ - θ₁)ᵀ G(θ₁) (θ₂ - θ₁)
        let mut distance_squared = 0.0;
        for i in 0..param_diff.len() {
            for j in 0..param_diff.len() {
                distance_squared += param_diff[i] * fisher_matrix[[i, j]] * param_diff[j];
            }
        }

        distance_squared.sqrt()
    }

    /// Digamma function approximation
    fn psi_prime(&self, z: f64) -> f64 {
        // Approximation for ψ'(z)
        1.0 / z + 1.0 / (2.0 * z.powi(2)) + 1.0 / (6.0 * z.powi(3))
    }
}

/// Bayesian Surprise calculator
pub struct BayesianSurprise;

impl BayesianSurprise {
    /// Calculate Bayesian Surprise: S(D,M) = D_KL(P(M|D) || P(M))
    pub fn calculate_surprise(
        &self,
        prior: &ProbabilityDistribution,
        posterior: &ProbabilityDistribution,
    ) -> f64 {
        self.kullback_leibler_divergence(posterior, prior)
    }

    /// Calculate Kullback-Leibler divergence: D_KL(P || Q)
    pub fn kullback_leibler_divergence(
        &self,
        p: &ProbabilityDistribution,
        q: &ProbabilityDistribution,
    ) -> f64 {
        // Numerical integration over support
        let (a, b) = p.support;
        let num_points = 1000;
        let dx = (b - a) / (num_points as f64);

        let mut kl_divergence = 0.0;

        for i in 0..num_points {
            let x = a + i as f64 * dx;
            let p_x = p.pdf(x);
            let q_x = q.pdf(x);

            if p_x > 1e-10 && q_x > 1e-10 {
                kl_divergence += p_x * (p_x / q_x).ln() * dx;
            }
        }

        kl_divergence
    }

    /// Calculate mutual information between two distributions
    pub fn mutual_information(
        &self,
        joint: &ProbabilityDistribution,
        marginal1: &ProbabilityDistribution,
        marginal2: &ProbabilityDistribution,
    ) -> f64 {
        // I(X;Y) = ∫∫ p(x,y) log(p(x,y)/(p(x)p(y))) dx dy
        // Simplified for 1D case
        let (a, b) = joint.support;
        let num_points = 100;
        let dx = (b - a) / (num_points as f64);

        let mut mutual_info = 0.0;

        for i in 0..num_points {
            let x = a + i as f64 * dx;
            let joint_x = joint.pdf(x);
            let marginal1_x = marginal1.pdf(x);
            let marginal2_x = marginal2.pdf(x);

            if joint_x > 1e-10 && marginal1_x > 1e-10 && marginal2_x > 1e-10 {
                mutual_info += joint_x * (joint_x / (marginal1_x * marginal2_x)).ln() * dx;
            }
        }

        mutual_info
    }
}

/// Natural gradient descent optimizer
pub struct NaturalGradientDescent {
    pub learning_rate: f64,
    pub momentum: f64,
    pub fisher_metric: FisherInformationMetric,
    pub velocity: Array1<f64>,
}

impl NaturalGradientDescent {
    pub fn new(learning_rate: f64, momentum: f64, parameter_dim: usize) -> Self {
        Self {
            learning_rate,
            momentum,
            fisher_metric: FisherInformationMetric,
            velocity: Array1::zeros(parameter_dim),
        }
    }

    /// Update parameters using natural gradient
    pub fn update(
        &mut self,
        parameters: &mut Array1<f64>,
        gradient: &Array1<f64>,
        distribution: &ProbabilityDistribution,
    ) -> Result<()> {
        // Calculate Fisher information matrix
        let fisher_matrix = self.fisher_metric.calculate_fisher_matrix(distribution);

        // Calculate natural gradient: G⁻¹ ∇L
        let natural_gradient = self.solve_linear_system(&fisher_matrix, gradient)?;

        // Update velocity with momentum
        self.velocity = &self.velocity * self.momentum + &natural_gradient * self.learning_rate;

        // Update parameters
        *parameters = &*parameters + &self.velocity;

        Ok(())
    }

    /// Solve linear system Gx = b
    fn solve_linear_system(&self, matrix: &Array2<f64>, rhs: &Array1<f64>) -> Result<Array1<f64>> {
        let n = matrix.nrows();
        let mut solution = Array1::zeros(n);

        // Simple Gaussian elimination (for small matrices)
        if n <= 10 {
            let mut augmented = Array2::zeros((n, n + 1));
            for i in 0..n {
                for j in 0..n {
                    augmented[[i, j]] = matrix[[i, j]];
                }
                augmented[[i, n]] = rhs[i];
            }

            // Forward elimination
            for i in 0..n {
                let pivot = augmented[[i, i]];
                if pivot.abs() < 1e-10 {
                    return Err(anyhow!("Singular matrix"));
                }

                for j in i..n {
                    augmented[[i, j]] /= pivot;
                }
                augmented[[i, n]] /= pivot;

                for k in (i + 1)..n {
                    let factor = augmented[[k, i]];
                    for j in i..n {
                        augmented[[k, j]] -= factor * augmented[[i, j]];
                    }
                    augmented[[k, n]] -= factor * augmented[[i, n]];
                }
            }

            // Back substitution
            for i in (0..n).rev() {
                solution[i] = augmented[[i, n]];
                for j in (i + 1)..n {
                    solution[i] -= augmented[[i, j]] * solution[j];
                }
            }
        } else {
            // For larger matrices, use iterative method
            solution = rhs.clone();
            for _ in 0..100 {
                // Max iterations
                let mut new_solution = Array1::zeros(n);
                for i in 0..n {
                    new_solution[i] = rhs[i];
                    for j in 0..n {
                        if i != j {
                            new_solution[i] -= matrix[[i, j]] * solution[j];
                        }
                    }
                    new_solution[i] /= matrix[[i, i]];
                }
                solution = new_solution;
            }
        }

        Ok(solution)
    }
}

/// Information-theoretic learning signal
pub struct InformationLearningSignal {
    pub surprise_calculator: BayesianSurprise,
    pub fisher_metric: FisherInformationMetric,
    pub surprise_threshold: f64,
}

impl InformationLearningSignal {
    pub fn new(surprise_threshold: f64) -> Self {
        Self {
            surprise_calculator: BayesianSurprise,
            fisher_metric: FisherInformationMetric,
            surprise_threshold,
        }
    }

    /// Calculate learning signal from observation
    pub fn calculate_learning_signal(
        &self,
        prior: &ProbabilityDistribution,
        observation: f64,
    ) -> Result<f64> {
        // Create posterior distribution (simplified Bayesian update)
        let posterior = self.update_belief(prior, observation)?;

        // Calculate Bayesian surprise
        let surprise = self
            .surprise_calculator
            .calculate_surprise(prior, &posterior);

        // Return learning signal (surprise above threshold)
        if surprise > self.surprise_threshold {
            Ok(surprise)
        } else {
            Ok(0.0)
        }
    }

    /// Update belief given observation (simplified Bayesian update)
    fn update_belief(
        &self,
        prior: &ProbabilityDistribution,
        observation: f64,
    ) -> Result<ProbabilityDistribution> {
        match prior.distribution_type {
            DistributionType::Gaussian => {
                let prior_mean = prior.parameters[0];
                let prior_var = prior.parameters[1];

                // Simplified update (assumes unit observation variance)
                let posterior_var = 1.0 / (1.0 / prior_var + 1.0);
                let posterior_mean = posterior_var * (prior_mean / prior_var + observation);

                Ok(ProbabilityDistribution::new_gaussian(
                    posterior_mean,
                    posterior_var,
                ))
            }
            DistributionType::Beta => {
                let alpha = prior.parameters[0];
                let beta = prior.parameters[1];

                // Update Beta parameters
                let new_alpha = alpha + observation;
                let new_beta = beta + 1.0 - observation;

                Ok(ProbabilityDistribution::new_beta(new_alpha, new_beta))
            }
            _ => {
                // Default: return prior unchanged
                Ok(prior.clone())
            }
        }
    }

    /// Calculate information length of belief update
    pub fn information_length(
        &self,
        prior: &ProbabilityDistribution,
        posterior: &ProbabilityDistribution,
    ) -> f64 {
        self.fisher_metric.fisher_distance(prior, posterior)
    }
}

/// Qualia space for characterizing conscious experience
pub struct QualiaSpace {
    pub dimensions: usize,
    pub information_measures: Vec<InformationMeasure>,
    pub geometric_structure: Array2<f64>,
}

#[derive(Debug, Clone)]
pub struct InformationMeasure {
    pub name: String,
    pub value: f64,
    pub distribution: ProbabilityDistribution,
}

impl QualiaSpace {
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            information_measures: Vec::new(),
            geometric_structure: Array2::zeros((dimensions, dimensions)),
        }
    }

    /// Add information measure to qualia space
    pub fn add_measure(&mut self, measure: InformationMeasure) {
        self.information_measures.push(measure);
        self.update_geometric_structure();
    }

    /// Update geometric structure based on information measures
    fn update_geometric_structure(&mut self) {
        let fisher_metric = FisherInformationMetric;

        for i in 0..self.information_measures.len() {
            for j in 0..self.information_measures.len() {
                if i != j {
                    let distance = fisher_metric.fisher_distance(
                        &self.information_measures[i].distribution,
                        &self.information_measures[j].distribution,
                    );
                    self.geometric_structure[[i, j]] = distance;
                }
            }
        }
    }

    /// Calculate qualia shape descriptor
    pub fn calculate_qualia_shape(&self) -> Array1<f64> {
        let mut shape_descriptor = Array1::zeros(self.dimensions);

        for (i, measure) in self.information_measures.iter().enumerate() {
            shape_descriptor[i] = measure.value;
        }

        shape_descriptor
    }

    /// Calculate similarity between two qualia shapes
    pub fn qualia_similarity(&self, other: &QualiaSpace) -> f64 {
        let shape1 = self.calculate_qualia_shape();
        let shape2 = other.calculate_qualia_shape();

        let mut similarity = 0.0;
        for i in 0..shape1.len().min(shape2.len()) {
            similarity += (shape1[i] - shape2[i]).powi(2);
        }

        (-similarity).exp() // Gaussian kernel
    }
}

/// Active inference framework
pub struct ActiveInference {
    pub generative_model: ProbabilityDistribution,
    pub recognition_model: ProbabilityDistribution,
    pub surprise_calculator: BayesianSurprise,
    pub free_energy: f64,
}

impl ActiveInference {
    pub fn new(
        generative_model: ProbabilityDistribution,
        recognition_model: ProbabilityDistribution,
    ) -> Self {
        Self {
            generative_model,
            recognition_model,
            surprise_calculator: BayesianSurprise,
            free_energy: 0.0,
        }
    }

    /// Calculate free energy (surprise)
    pub fn calculate_free_energy(&mut self, observation: f64) -> f64 {
        // Free energy = -log p(o) + D_KL(q(s) || p(s|o))
        let log_likelihood = -self.generative_model.log_likelihood(observation);
        let kl_divergence = self
            .surprise_calculator
            .kullback_leibler_divergence(&self.recognition_model, &self.generative_model);

        self.free_energy = log_likelihood + kl_divergence;
        self.free_energy
    }

    /// Update recognition model to minimize free energy
    pub fn update_recognition_model(&mut self, observation: f64, learning_rate: f64) -> Result<()> {
        let mut optimizer = NaturalGradientDescent::new(
            learning_rate,
            0.9,
            self.recognition_model.parameters.len(),
        );

        // Calculate gradient of free energy with respect to recognition model parameters
        let gradient = self.calculate_free_energy_gradient(observation);

        // Update parameters using natural gradient
        let mut parameters = mem::take(&mut self.recognition_model.parameters);
        optimizer.update(&mut parameters, &gradient, &self.recognition_model)?;
        self.recognition_model.parameters = parameters;

        Ok(())
    }

    /// Calculate gradient of free energy
    fn calculate_free_energy_gradient(&self, observation: f64) -> Array1<f64> {
        let mut gradient = Array1::zeros(self.recognition_model.parameters.len());

        // Simplified gradient calculation
        for i in 0..gradient.len() {
            let mut params_plus = self.recognition_model.parameters.clone();
            let mut params_minus = self.recognition_model.parameters.clone();

            params_plus[i] += 1e-6;
            params_minus[i] -= 1e-6;

            let mut model_plus = self.recognition_model.clone();
            let mut model_minus = self.recognition_model.clone();
            model_plus.parameters = params_plus;
            model_minus.parameters = params_minus;

            let free_energy_plus = self.calculate_free_energy_for_model(&model_plus, observation);
            let free_energy_minus = self.calculate_free_energy_for_model(&model_minus, observation);

            gradient[i] = (free_energy_plus - free_energy_minus) / (2.0 * 1e-6);
        }

        gradient
    }

    /// Calculate free energy for a specific model
    fn calculate_free_energy_for_model(
        &self,
        model: &ProbabilityDistribution,
        observation: f64,
    ) -> f64 {
        let log_likelihood = -model.log_likelihood(observation);
        let kl_divergence = self
            .surprise_calculator
            .kullback_leibler_divergence(model, &self.generative_model);
        log_likelihood + kl_divergence
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probability_distribution() {
        let gaussian = ProbabilityDistribution::new_gaussian(0.0, 1.0);
        let pdf = gaussian.pdf(0.0);
        assert!(pdf > 0.0);
    }

    #[test]
    fn test_fisher_information_metric() {
        let fisher_metric = FisherInformationMetric;
        let gaussian = ProbabilityDistribution::new_gaussian(0.0, 1.0);
        let fisher_matrix = fisher_metric.calculate_fisher_matrix(&gaussian);
        assert_eq!(fisher_matrix.nrows(), 2);
        assert_eq!(fisher_matrix.ncols(), 2);
    }

    #[test]
    fn test_bayesian_surprise() {
        let surprise_calc = BayesianSurprise;
        let prior = ProbabilityDistribution::new_gaussian(0.0, 1.0);
        let posterior = ProbabilityDistribution::new_gaussian(1.0, 1.0);
        let surprise = surprise_calc.calculate_surprise(&prior, &posterior);
        assert!(surprise > 0.0);
    }

    #[test]
    fn test_natural_gradient_descent() {
        let mut optimizer = NaturalGradientDescent::new(0.01, 0.9, 2);
        let mut params = Array1::from_vec(vec![0.0, 1.0]);
        let gradient = Array1::from_vec(vec![1.0, -1.0]);
        let distribution = ProbabilityDistribution::new_gaussian(0.0, 1.0);

        let result = optimizer.update(&mut params, &gradient, &distribution);
        assert!(result.is_ok());
    }

    #[test]
    fn test_qualia_space() {
        let mut qualia_space = QualiaSpace::new(3);
        let measure = InformationMeasure {
            name: "entropy".to_string(),
            value: 0.5,
            distribution: ProbabilityDistribution::new_gaussian(0.0, 1.0),
        };
        qualia_space.add_measure(measure);
        assert_eq!(qualia_space.information_measures.len(), 1);
    }
}
