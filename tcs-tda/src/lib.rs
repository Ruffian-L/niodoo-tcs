//! Topological data analysis toolkit implementing the mathematical
//! routines referenced in the README. The goal is to provide solid,
//! reusable primitives while we continue migrating higher-level logic.

use nalgebra::DVector;
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::collections::HashMap;

/// Result of a persistent homology computation for a single feature.
#[derive(Debug, Clone)]
pub struct PersistenceFeature {
    pub birth: f32,
    pub death: f32,
    pub dimension: usize,
}

impl PersistenceFeature {
    pub fn persistence(&self) -> f32 {
        (self.death - self.birth).abs()
    }
}

/// Takens embedding implementation with helpers for mutual information
/// and false-nearest-neighbour heuristics, closely following the README.
#[derive(Debug, Clone)]
pub struct TakensEmbedding {
    pub dimension: usize,
    pub delay: usize,
    pub data_dim: usize,
}

impl TakensEmbedding {
    pub fn new(dimension: usize, delay: usize, data_dim: usize) -> Self {
        Self {
            dimension,
            delay,
            data_dim,
        }
    }

    pub fn optimal_delay(time_series: &[Vec<f32>], max_delay: usize) -> usize {
        let mut scores = Vec::new();
        for tau in 1..=max_delay {
            scores.push((tau, Self::mutual_information(time_series, tau)));
        }
        scores
            .windows(3)
            .find_map(|window| {
                let (prev_tau, prev_mi) = window[0];
                let (tau, mi) = window[1];
                let (_, next_mi) = window[2];
                if mi < prev_mi && mi < next_mi {
                    Some(tau)
                } else {
                    None
                }
            })
            .unwrap_or(3)
    }

    pub fn optimal_dimension(time_series: &[Vec<f32>], delay: usize, max_dim: usize) -> usize {
        for dim in 2..=max_dim {
            let ratio = Self::false_nearest_neighbours(time_series, dim, delay);
            if ratio < 0.01 {
                return dim;
            }
        }
        3
    }

    pub fn embed(&self, time_series: &[Vec<f32>]) -> Vec<DVector<f32>> {
        let n = time_series.len();
        let embed_len = n.saturating_sub(self.dimension * self.delay);
        (0..embed_len)
            .into_par_iter()
            .map(|t| {
                let mut point = Vec::with_capacity((self.dimension + 1) * self.data_dim);
                for i in 0..=self.dimension {
                    if let Some(slice) = time_series.get(t + i * self.delay) {
                        point.extend_from_slice(slice);
                    }
                }
                DVector::from_vec(point)
            })
            .collect()
    }

    fn mutual_information(time_series: &[Vec<f32>], delay: usize) -> f32 {
        if time_series.len() <= delay {
            return 0.0;
        }

        let x: Vec<f32> = time_series[..time_series.len() - delay]
            .iter()
            .map(|v| v[0])
            .collect();
        let y: Vec<f32> = time_series[delay..]
            .iter()
            .map(|v| v[0])
            .collect();

        let bins = 32;
        let hx = entropy(&x, bins);
        let hy = entropy(&y, bins);
        let hxy = joint_entropy(&x, &y, bins);

        (hx + hy - hxy).max(0.0)
    }

    fn false_nearest_neighbours(time_series: &[Vec<f32>], dimension: usize, delay: usize) -> f32 {
        if time_series.len() <= delay * dimension {
            return 1.0;
        }

        let embed = Self {
            dimension,
            delay,
            data_dim: time_series[0].len(),
        }
        .embed(time_series);

        let next_embed = Self {
            dimension: dimension + 1,
            delay,
            data_dim: time_series[0].len(),
        }
        .embed(time_series);

        let mut false_count = 0;
        let mut total = 0;

        for (vec_m, vec_m1) in embed.iter().zip(next_embed.iter()) {
            if let Some((nearest_idx, dist)) = nearest_neighbour(vec_m, &embed) {
                let dist_m1 = (vec_m1 - &next_embed[nearest_idx]).norm();
                if dist == 0.0 {
                    continue;
                }
                if (dist_m1 - dist).abs() / dist > 15.0 {
                    false_count += 1;
                }
                total += 1;
            }
        }

        if total == 0 {
            1.0
        } else {
            false_count as f32 / total as f32
        }
    }
}

/// Persistent homology helper implementing a lightweight Vietoris–Rips
/// filtration suitable for medium-sized point clouds.
#[derive(Debug, Clone)]
pub struct PersistentHomology {
    pub max_dimension: usize,
    pub max_edge_length: f32,
}

impl PersistentHomology {
    pub fn new(max_dimension: usize, max_edge_length: f32) -> Self {
        Self {
            max_dimension,
            max_edge_length,
        }
    }

    pub fn compute(&self, points: &[DVector<f32>]) -> Vec<PersistenceFeature> {
        if points.is_empty() {
            return Vec::new();
        }

        let mut features = Vec::new();
        let distances = pairwise_distances(points);

        // H0 features: birth at 0, death when components merge.
        for i in 0..points.len() {
            let death = distances
                .row(i)
                .iter()
                .filter(|&&d| d > 0.0)
                .fold(f32::INFINITY, |acc, &d| acc.min(d));
            features.push(PersistenceFeature {
                birth: 0.0,
                death,
                dimension: 0,
            });
        }

        if self.max_dimension >= 1 {
            for i in 0..points.len() {
                for j in (i + 1)..points.len() {
                    let dist = distances[(i, j)];
                    if dist <= self.max_edge_length {
                        features.push(PersistenceFeature {
                            birth: dist * 0.5,
                            death: dist,
                            dimension: 1,
                        });
                    }
                }
            }
        }

        features.sort_by(|a, b| a.birth.partial_cmp(&b.birth).unwrap());
        features
    }

    pub fn witness_complex(&self, landmarks: &[DVector<f32>], witnesses: &[DVector<f32>]) -> Array2<f32> {
        let mut matrix = Array2::<f32>::zeros((landmarks.len(), landmarks.len()));
        for (i, l1) in landmarks.iter().enumerate() {
            for (j, l2) in landmarks.iter().enumerate().skip(i + 1) {
                let weight = witnesses
                    .iter()
                    .map(|w| (w - l1).norm() + (w - l2).norm())
                    .fold(f32::INFINITY, f32::min);
                matrix[(i, j)] = weight;
                matrix[(j, i)] = weight;
            }
        }
        matrix
    }
}

fn entropy(data: &[f32], bins: usize) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let hist = histogram(data, bins);
    hist.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum()
}

fn joint_entropy(x: &[f32], y: &[f32], bins: usize) -> f32 {
    if x.is_empty() || y.is_empty() || x.len() != y.len() {
        return 0.0;
    }

    let min_x = x.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_x = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_y = y.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_y = y.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let bin_width_x = ((max_x - min_x) / bins as f32).max(f32::EPSILON);
    let bin_width_y = ((max_y - min_y) / bins as f32).max(f32::EPSILON);

    let mut counts = HashMap::new();
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let bx = ((xi - min_x) / bin_width_x).floor() as usize;
        let by = ((yi - min_y) / bin_width_y).floor() as usize;
        *counts.entry((bx.min(bins - 1), by.min(bins - 1))).or_insert(0usize) += 1;
    }

    let total = x.len() as f32;
    counts
        .values()
        .map(|&count| {
            let p = count as f32 / total;
            -p * p.ln()
        })
        .sum()
}

fn histogram(data: &[f32], bins: usize) -> Vec<f32> {
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let width = ((max - min) / bins as f32).max(f32::EPSILON);
    let mut counts = vec![0usize; bins];
    for &value in data {
        let idx = ((value - min) / width).floor() as usize;
        let idx = idx.min(bins - 1);
        counts[idx] += 1;
    }
    let total = data.len() as f32;
    counts.into_iter().map(|c| c as f32 / total).collect()
}

fn nearest_neighbour(target: &DVector<f32>, points: &[DVector<f32>]) -> Option<(usize, f32)> {
    let mut best_idx = None;
    let mut best_dist = f32::INFINITY;
    for (idx, candidate) in points.iter().enumerate() {
        if candidate == target {
            continue;
        }
        let dist = (target - candidate).norm();
        if dist < best_dist {
            best_dist = dist;
            best_idx = Some(idx);
        }
    }
    best_idx.map(|idx| (idx, best_dist))
}

fn pairwise_distances(points: &[DVector<f32>]) -> Array2<f32> {
    let n = points.len();
    let mut matrix = Array2::<f32>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = (points[i].clone() - points[j].clone()).norm();
            matrix[(i, j)] = dist;
            matrix[(j, i)] = dist;
        }
    }
    matrix
}

/// Convenience to convert raw vectors into a single ndarray matrix.
pub fn to_array(time_series: &[Vec<f32>]) -> Array2<f32> {
    if time_series.is_empty() {
        return Array2::zeros((0, 0));
    }
    let rows = time_series.len();
    let cols = time_series[0].len();
    let flat: Vec<f32> = time_series.iter().flat_map(|row| row.iter().copied()).collect();
    Array2::from_shape_vec((rows, cols), flat).unwrap_or_else(|_| Array2::zeros((0, 0)))
}

/// Convenience to build a view around a vector for external crates.
pub fn to_array1(sample: &[f32]) -> Array1<f32> {
    Array1::from(sample.to_vec())
}
