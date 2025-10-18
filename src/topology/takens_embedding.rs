use nalgebra::{DVector, DMatrix};
use rayon::prelude::*;
// use statistical::*;
use std::collections::HashMap;

/// Takens' Embedding Theorem implementation for time series reconstruction
#[derive(Debug, Clone)]
pub struct TakensEmbedding {
    dimension: usize,      // M - embedding dimension
    delay: usize,         // tau - time delay
    data_dim: usize,      // D - original data dimension
}

impl TakensEmbedding {
    /// Create new Takens embedding
    pub fn new(dimension: usize, delay: usize, data_dim: usize) -> Self {
        Self {
            dimension,
            delay,
            data_dim,
        }
    }

    /// Compute mutual information to find optimal tau
    pub fn optimal_delay(time_series: &[Vec<f32>]) -> usize {
        let mut mi_values = Vec::new();

        for tau in 1..50 {
            let mi = Self::mutual_information(time_series, tau);
            mi_values.push((tau, mi));
        }

        // Find first local minimum
        for i in 1..mi_values.len()-1 {
            if mi_values[i].1 < mi_values[i-1].1 &&
               mi_values[i].1 < mi_values[i+1].1 {
                return mi_values[i].0;
            }
        }

        // Default if no minimum found
        3
    }

    /// Compute false nearest neighbors to find optimal M
    pub fn optimal_dimension(
        time_series: &[Vec<f32>],
        tau: usize
    ) -> usize {
        let max_dim = 15;
        let rtol = 15.0;  // Threshold for false neighbors

        for m in 1..max_dim {
            let embedded = Self::embed_static(time_series, m, tau);
            let fnn_ratio = Self::false_nearest_neighbors(&embedded, rtol);

            if fnn_ratio < 0.01 {  // Less than 1% false neighbors
                return m;
            }
        }

        5  // Default fallback
    }

    /// Main embedding function with parallel processing
    pub fn embed(
        &self,
        time_series: &[Vec<f32>]
    ) -> Vec<DVector<f32>> {
        let n = time_series.len();
        let embed_len = n - self.dimension * self.delay;

        (0..embed_len)
            .into_par_iter()
            .map(|t| {
                let mut point = Vec::with_capacity((self.dimension + 1) * self.data_dim);

                for i in 0..=self.dimension {
                    point.extend_from_slice(&time_series[t + i * self.delay]);
                }

                DVector::from_vec(point)
            })
            .collect()
    }

    /// Static version for optimal parameter search
    fn embed_static(
        time_series: &[Vec<f32>],
        m: usize,
        tau: usize
    ) -> Vec<DVector<f32>> {
        let n = time_series.len();
        let data_dim = time_series[0].len();
        let embed_len = n - m * tau;

        (0..embed_len)
            .map(|t| {
                let mut point = Vec::with_capacity((m + 1) * data_dim);

                for i in 0..=m {
                    point.extend_from_slice(&time_series[t + i * tau]);
                }

                DVector::from_vec(point)
            })
            .collect()
    }

    /// Compute false nearest neighbors ratio
    fn false_nearest_neighbors(
        embedded: &[DVector<f32>],
        rtol: f32
    ) -> f32 {
        let mut false_neighbors = 0;
        let total_pairs = embedded.len() * (embedded.len() - 1) / 2;

        for i in 0..embedded.len() {
            for j in i+1..embedded.len() {
                let dist = (embedded[i].clone() - embedded[j].clone()).norm();
                // Simplified FNN calculation
                if dist > rtol {
                    false_neighbors += 1;
                }
            }
        }

        false_neighbors as f32 / total_pairs as f32
    }

    /// Mutual information calculation
    fn mutual_information(
        time_series: &[Vec<f32>],
        delay: usize
    ) -> f32 {
        // Simplified mutual information using correlation
        // For full implementation, would need proper entropy calculation
        let x: Vec<f32> = time_series.iter()
            .flat_map(|v| v.iter().cloned())
            .collect();

        let y: Vec<f32> = time_series[delay..]
            .iter()
            .flat_map(|v| v.iter().cloned())
            .collect();

        // Approximate MI with correlation
        let corr = Self::correlation(&x, &y);
        -corr.abs().ln()  // Simplified
    }

    fn correlation(x: &[f32], y: &[f32]) -> f32 {
        let n = x.len();
        let sum_x = x.iter().sum::<f32>();
        let sum_y = y.iter().sum::<f32>();
        let sum_xy = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum::<f32>();
        let sum_x2 = x.iter().map(|v| v * v).sum::<f32>();
        let sum_y2 = y.iter().map(|v| v * v).sum::<f32>();

        (n as f32 * sum_xy - sum_x * sum_y) /
        ((n as f32 * sum_x2 - sum_x * sum_x) *
         (n as f32 * sum_y2 - sum_y * sum_y)).sqrt()
    }
}