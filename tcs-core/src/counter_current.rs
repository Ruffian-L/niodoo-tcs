use std::collections::VecDeque;

use anyhow::{Result, ensure};
use nalgebra::{DMatrix, DVector};

/// Represents a multivariate Gaussian belief over the latent state.
#[derive(Clone, Debug)]
pub struct ProbabilisticBelief {
    mean: DVector<f32>,
    covariance: DMatrix<f32>,
}

impl ProbabilisticBelief {
    /// Construct a belief from raw components. Covariance must be square and match the mean dimension.
    pub fn from_parts(mean: DVector<f32>, covariance: DMatrix<f32>) -> Result<Self> {
        ensure!(covariance.is_square(), "covariance matrix must be square");
        ensure!(
            covariance.nrows() == mean.len(),
            "covariance dimension must match mean length"
        );
        Ok(Self { mean, covariance })
    }

    /// Access the dimensionality of the belief.
    pub fn dimension(&self) -> usize {
        self.mean.len()
    }

    /// Immutable view of the belief mean vector.
    pub fn mean(&self) -> &DVector<f32> {
        &self.mean
    }

    /// Immutable view of the covariance matrix.
    pub fn covariance(&self) -> &DMatrix<f32> {
        &self.covariance
    }

    /// Differential entropy of the Gaussian belief (in nats).
    pub fn entropy(&self) -> f32 {
        let dim = self.dimension() as f32;
        let det = self.covariance.determinant().max(std::f32::MIN_POSITIVE);
        // 0.5 * ln((2πe)^k * det(Sigma))
        0.5 * (dim * (2.0 * std::f32::consts::PI * std::f32::consts::E).ln() + det.ln())
    }

    /// Perform a conjugate Bayesian update with Gaussian evidence.
    pub fn bayesian_update(&self, observation: &Observation) -> Self {
        let prior_precision = regularised_inverse(&self.covariance);
        let obs_precision = regularised_inverse(&observation.covariance);

        let posterior_precision = &prior_precision + &obs_precision;
        let posterior_covariance = regularised_inverse(&posterior_precision);

        let rhs = &prior_precision * &self.mean + &obs_precision * &observation.mean;
        let posterior_mean = &posterior_covariance * rhs;

        Self {
            mean: posterior_mean,
            covariance: posterior_covariance,
        }
    }
}

/// Incoming observation represented as a Gaussian estimate.
#[derive(Clone, Debug)]
pub struct Observation {
    mean: DVector<f32>,
    covariance: DMatrix<f32>,
}

impl Observation {
    pub fn from_parts(mean: DVector<f32>, covariance: DMatrix<f32>) -> Result<Self> {
        ensure!(covariance.is_square(), "covariance matrix must be square");
        ensure!(
            covariance.nrows() == mean.len(),
            "covariance dimension must match mean length"
        );
        Ok(Self { mean, covariance })
    }

    pub fn mean(&self) -> &DVector<f32> {
        &self.mean
    }

    pub fn covariance(&self) -> &DMatrix<f32> {
        &self.covariance
    }
}

/// Scheduler that pairs highest-uncertainty beliefs with freshest evidence.
#[derive(Clone, Debug)]
pub struct CounterCurrentScheduler {
    beliefs: Vec<ProbabilisticBelief>,
    data_window: VecDeque<Observation>,
}

impl CounterCurrentScheduler {
    /// Build a scheduler from a set of initial beliefs. Beliefs are sorted by entropy (descending).
    pub fn new(mut beliefs: Vec<ProbabilisticBelief>) -> Self {
        beliefs.sort_by(|a, b| {
            b.entropy()
                .partial_cmp(&a.entropy())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self {
            beliefs,
            data_window: VecDeque::new(),
        }
    }

    /// Fetch immutable view of beliefs.
    pub fn beliefs(&self) -> &[ProbabilisticBelief] {
        &self.beliefs
    }

    /// Number of cached observations currently paired with beliefs.
    pub fn window_len(&self) -> usize {
        self.data_window.len()
    }

    /// Add a new belief into the scheduler and maintain entropy ordering.
    pub fn push_belief(&mut self, belief: ProbabilisticBelief) {
        self.beliefs.push(belief);
        self.beliefs.sort_by(|a, b| {
            b.entropy()
                .partial_cmp(&a.entropy())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if self.data_window.len() > self.beliefs.len() {
            self.data_window.truncate(self.beliefs.len());
        }
    }

    /// Push new evidence into the scheduler and perform counter-current updates.
    pub fn update(&mut self, observation: Observation) {
        self.data_window.push_front(observation);
        if self.data_window.len() > self.beliefs.len() {
            self.data_window.truncate(self.beliefs.len());
        }

        for (belief, datum) in self.beliefs.iter_mut().zip(self.data_window.iter()) {
            let updated = belief.bayesian_update(datum);
            *belief = updated;
        }

        self.beliefs.sort_by(|a, b| {
            b.entropy()
                .partial_cmp(&a.entropy())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}

fn regularised_inverse(matrix: &DMatrix<f32>) -> DMatrix<f32> {
    const EPS: f32 = 1e-6;
    match matrix.clone().try_inverse() {
        Some(inv) => inv,
        None => {
            let mut regularised = matrix.clone();
            for idx in 0..regularised.nrows() {
                regularised[(idx, idx)] += EPS;
            }
            regularised
                .try_inverse()
                .expect("regularised matrix should be invertible")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gaussian(mean: f32, variance: f32) -> (DVector<f32>, DMatrix<f32>) {
        let m = DVector::from_element(1, mean);
        let cov = DMatrix::from_element(1, 1, variance);
        (m, cov)
    }

    #[test]
    fn entropy_drops_for_high_uncertainty_belief() {
        let (mean_high, cov_high) = make_gaussian(0.0, 5.0);
        let (mean_low, cov_low) = make_gaussian(0.0, 1.0);
        let belief_high = ProbabilisticBelief::from_parts(mean_high, cov_high).unwrap();
        let belief_low = ProbabilisticBelief::from_parts(mean_low, cov_low).unwrap();

        let (obs_mean, obs_cov) = make_gaussian(0.5, 0.5);
        let observation = Observation::from_parts(obs_mean, obs_cov).unwrap();

        let mut scheduler =
            CounterCurrentScheduler::new(vec![belief_high.clone(), belief_low.clone()]);

        let before_entropy = scheduler.beliefs()[0].entropy();
        scheduler.update(observation);
        let after_entropy = scheduler.beliefs()[0].entropy();

        assert!(
            after_entropy <= before_entropy + 1e-4,
            "entropy should not increase for top belief"
        );
    }

    #[test]
    fn data_window_truncates_to_belief_count() {
        let (mean_a, cov_a) = make_gaussian(0.0, 2.0);
        let (mean_b, cov_b) = make_gaussian(1.0, 1.0);
        let belief_a = ProbabilisticBelief::from_parts(mean_a, cov_a).unwrap();
        let belief_b = ProbabilisticBelief::from_parts(mean_b, cov_b).unwrap();
        let mut scheduler = CounterCurrentScheduler::new(vec![belief_a, belief_b]);

        for idx in 0..5 {
            let (obs_mean, obs_cov) = make_gaussian(idx as f32, 0.25);
            let observation = Observation::from_parts(obs_mean, obs_cov).unwrap();
            scheduler.update(observation);
        }

        assert!(scheduler.window_len() <= 2);
    }
}
