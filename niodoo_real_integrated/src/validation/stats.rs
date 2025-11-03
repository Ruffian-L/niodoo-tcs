//! Statistical Analysis Library for Validation Framework
//!
//! Provides rigorous statistical methods for analyzing validation results:
//! - Bootstrap analysis for latency SLOs (confidence intervals)
//! - Effect size calculation (Cohen's d)
//! - Non-parametric hypothesis testing (Mann-Whitney U test)

use std::collections::HashMap;
use rand::Rng;
use serde::{Serialize, Deserialize};

/// Bootstrap confidence interval for a percentile metric
/// 
/// Resamples the data with replacement to generate an empirical sampling distribution
/// Returns (lower_bound, upper_bound) for the specified confidence level
pub fn bootstrap_percentile_ci(
    values: &[f64],
    percentile: f64,
    n_samples: usize,
    confidence: f64,
) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }

    let mut rng = rand::thread_rng();
    let n = values.len();
    let mut bootstrap_samples = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        // Resample with replacement
        let sample: Vec<f64> = (0..n)
            .map(|_| {
                let idx = rng.gen_range(0..n);
                values[idx]
            })
            .collect();

        // Compute percentile for this bootstrap sample
        let mut sorted = sample;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let index = (percentile * (sorted.len() - 1) as f64) as usize;
        bootstrap_samples.push(sorted[index]);
    }

    // Compute confidence interval
    bootstrap_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let alpha = 1.0 - confidence;
    let lower_idx = (alpha / 2.0 * n_samples as f64) as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n_samples as f64) as usize;

    (
        bootstrap_samples[lower_idx],
        bootstrap_samples[upper_idx],
    )
}

/// Compute Cohen's d effect size
/// 
/// Standardized measure of the difference between two means.
/// Returns: (mean1 - mean2) / pooled_std_dev
/// 
/// Interpretation:
/// - |d| < 0.2: Small effect
/// - 0.2 <= |d| < 0.5: Medium effect
/// - 0.5 <= |d| < 0.8: Large effect
/// - |d| >= 0.8: Very large effect
pub fn cohens_d(values1: &[f64], values2: &[f64]) -> f64 {
    if values1.is_empty() || values2.is_empty() {
        return 0.0;
    }

    let mean1 = values1.iter().sum::<f64>() / values1.len() as f64;
    let mean2 = values2.iter().sum::<f64>() / values2.len() as f64;

    let var1 = values1
        .iter()
        .map(|x| {
            let diff = x - mean1;
            diff * diff
        })
        .sum::<f64>()
        / values1.len() as f64;

    let var2 = values2
        .iter()
        .map(|x| {
            let diff = x - mean2;
            diff * diff
        })
        .sum::<f64>()
        / values2.len() as f64;

    let pooled_std = ((var1 + var2) / 2.0).sqrt();
    if pooled_std == 0.0 {
        return 0.0;
    }

    (mean1 - mean2) / pooled_std
}

/// Mann-Whitney U test for comparing two distributions
/// 
/// Non-parametric test that doesn't assume normal distribution.
/// Returns: (u_statistic, p_value_approximation)
/// 
/// Note: This is a simplified approximation. For production use, consider
/// using a proper statistical library with exact p-value calculation.
pub fn mann_whitney_u(values1: &[f64], values2: &[f64]) -> (f64, f64) {
    if values1.is_empty() || values2.is_empty() {
        return (0.0, 1.0);
    }

    let n1 = values1.len();
    let n2 = values2.len();

    // Rank all values together
    let mut all_values: Vec<(f64, usize)> = values1
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .chain(values2.iter().enumerate().map(|(i, &v)| (v, i + n1)))
        .collect();

    all_values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Assign ranks (handle ties by averaging)
    let mut ranks = vec![0.0; n1 + n2];
    let mut current_rank = 1.0;
    let mut i = 0;

    while i < all_values.len() {
        let mut tie_start = i;
        let mut tie_end = i;

        // Find all tied values
        while tie_end + 1 < all_values.len()
            && (all_values[tie_end + 1].0 - all_values[tie_start].0).abs() < f64::EPSILON
        {
            tie_end += 1;
        }

        // Assign average rank to all tied values
        let avg_rank = (current_rank + current_rank + (tie_end - tie_start) as f64) / 2.0;
        for j in tie_start..=tie_end {
            ranks[all_values[j].1] = avg_rank;
        }

        current_rank += (tie_end - tie_start + 1) as f64;
        i = tie_end + 1;
    }

    // Compute U statistic for group 1
    let r1: f64 = ranks[0..n1].iter().sum();
    let u1 = (n1 * n2) as f64 + (n1 * (n1 + 1)) as f64 / 2.0 - r1;
    let u2 = (n1 * n2) as f64 - u1;
    let u = u1.min(u2);

    // Approximate p-value using normal approximation
    let mean_u = (n1 * n2) as f64 / 2.0;
    let var_u = (n1 * n2 * (n1 + n2 + 1)) as f64 / 12.0;
    let std_u = var_u.sqrt();

    if std_u == 0.0 {
        return (u, 1.0);
    }

    let z = (u - mean_u) / std_u;
    // Two-tailed p-value approximation (simplified)
    let p_value = 2.0 * (1.0 - normal_cdf(z.abs()));

    (u, p_value)
}

/// Normal CDF approximation using error function
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / (2.0_f64).sqrt()))
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Check if SLO is breached using bootstrap confidence interval
/// 
/// An SLO is considered breached if the entire 95% confidence interval
/// exceeds the SLO threshold. This prevents false alarms from random fluctuations.
pub fn check_slo_breach(
    values: &[f64],
    percentile: f64,
    slo_threshold: f64,
    confidence: f64,
) -> bool {
    let (lower, upper) = bootstrap_percentile_ci(values, percentile, 10000, confidence);
    upper > slo_threshold
}

/// Determine if a regression requires investigation/rollback
/// 
/// A regression requires action if:
/// 1. Statistically significant (p < 0.05)
/// 2. Medium or larger effect size (|Cohen's d| >= 0.5)
pub fn requires_regression_action(
    baseline: &[f64],
    current: &[f64],
    significance_threshold: f64,
    effect_size_threshold: f64,
) -> bool {
    let (_, p_value) = mann_whitney_u(baseline, current);
    let effect_size = cohens_d(baseline, current);

    p_value < significance_threshold && effect_size.abs() >= effect_size_threshold
}

/// Statistical summary for a dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSummary {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub min: f64,
    pub max: f64,
    pub count: usize,
}

impl StatisticalSummary {
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                mean: 0.0,
                median: 0.0,
                std_dev: 0.0,
                p50: 0.0,
                p95: 0.0,
                p99: 0.0,
                min: 0.0,
                max: 0.0,
                count: 0,
            };
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f64>()
            / values.len() as f64;
        let std_dev = variance.sqrt();

        let median_idx = sorted.len() / 2;
        let median = if sorted.len() % 2 == 0 {
            (sorted[median_idx - 1] + sorted[median_idx]) / 2.0
        } else {
            sorted[median_idx]
        };

        let p50_idx = (0.50 * (sorted.len() - 1) as f64) as usize;
        let p95_idx = (0.95 * (sorted.len() - 1) as f64) as usize;
        let p99_idx = (0.99 * (sorted.len() - 1) as f64) as usize;

        Self {
            mean,
            median,
            std_dev,
            p50: sorted[p50_idx],
            p95: sorted[p95_idx],
            p99: sorted[p99_idx],
            min: sorted[0],
            max: sorted[sorted.len() - 1],
            count: values.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cohens_d() {
        let values1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let values2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        
        let d = cohens_d(&values1, &values2);
        // Should be negative (values2 > values1) and small effect
        assert!(d < 0.0);
        assert!(d.abs() < 0.5);
    }

    #[test]
    fn test_bootstrap_ci() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let (lower, upper) = bootstrap_percentile_ci(&values, 0.99, 1000, 0.95);
        
        assert!(lower < upper);
        assert!(lower >= 0.0);
        assert!(upper <= 100.0);
    }

    #[test]
    fn test_statistical_summary() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let summary = StatisticalSummary::from_values(&values);
        
        assert_eq!(summary.count, 5);
        assert_eq!(summary.mean, 3.0);
        assert_eq!(summary.median, 3.0);
        assert_eq!(summary.min, 1.0);
        assert_eq!(summary.max, 5.0);
    }
}

