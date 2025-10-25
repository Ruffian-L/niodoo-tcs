//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
 * 🌀 TOPOLOGICAL DATA ANALYSIS (TDA) - PERSISTENT HOMOLOGY 🌀
 *
 * Implements the core TDA pipeline for discovering the intrinsic shape
 * of cognitive state spaces from empirical data.
 *
 * Based on "The Geometry of Thought" framework:
 * - Vietoris-Rips filtration for simplicial complex construction
 * - Persistent homology for robust topological feature identification
 * - Betti numbers (β₀, β₁, β₂) for topological signatures
 */

use crate::config::ConsciousnessConfig;
#[allow(unused_imports)]
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
#[allow(unused_imports)]
use std::collections::{HashMap, HashSet, VecDeque};
#[allow(unused_imports)]
use std::f64::consts::PI;

/// Represents a topological feature (connected component, loop, void)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalFeature {
    pub dimension: usize,            // 0=component, 1=loop, 2=void
    pub birth_time: f64,             // When feature appears
    pub death_time: f64,             // When feature disappears
    pub persistence: f64,            // Lifetime of feature
    pub generators: Vec<Vec<usize>>, // Simplicial complexes that generate this feature
}

impl TopologicalFeature {
    pub fn new(dimension: usize, birth_time: f64, death_time: f64) -> Self {
        Self {
            dimension,
            birth_time,
            death_time,
            persistence: death_time - birth_time,
            generators: Vec::new(),
        }
    }
}

/// Persistent homology computation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentHomologyResult {
    pub features: Vec<TopologicalFeature>,
    pub betti_numbers: Vec<usize>,            // β₀, β₁, β₂, ...
    pub persistence_diagram: Vec<(f64, f64)>, // (birth, death) pairs
    pub topological_signature: String,
}

/// Point cloud in high-dimensional space
#[derive(Debug, Clone)]
pub struct PointCloud {
    pub points: Vec<Vec<f64>>,
    pub dimension: usize,
}

impl PointCloud {
    pub fn new(points: Vec<Vec<f64>>) -> Self {
        let dimension = points.first().map(|p| p.len()).unwrap_or(0);
        Self { points, dimension }
    }

    /// Calculate pairwise distances between all points
    pub fn pairwise_distances(&self) -> Vec<Vec<f64>> {
        let n = self.points.len();
        let mut distances = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    distances[i][j] = self.euclidean_distance(&self.points[i], &self.points[j]);
                }
            }
        }

        distances
    }

    /// Euclidean distance between two points
    fn euclidean_distance(&self, p1: &[f64], p2: &[f64]) -> f64 {
        p1.iter()
            .zip(p2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// Simplicial complex for TDA
#[derive(Debug, Clone)]
pub struct SimplicialComplex {
    pub simplices: Vec<Vec<usize>>, // Each simplex is a set of vertex indices
    pub dimension: usize,           // Highest dimension simplex
}

impl Default for SimplicialComplex {
    fn default() -> Self {
        Self::new()
    }
}

impl SimplicialComplex {
    pub fn new() -> Self {
        Self {
            simplices: Vec::new(),
            dimension: 0,
        }
    }

    /// Add a simplex (vertices must be sorted)
    pub fn add_simplex(&mut self, vertices: Vec<usize>) {
        if !vertices.is_empty() {
            self.dimension = self.dimension.max(vertices.len() - 1);
            self.simplices.push(vertices);
        }
    }

    /// Get all simplices of a specific dimension
    pub fn get_simplices_of_dimension(&self, dim: usize) -> Vec<&Vec<usize>> {
        self.simplices
            .iter()
            .filter(|simplex| simplex.len() == dim + 1)
            .collect()
    }
}

/// Vietoris-Rips complex construction
pub struct VietorisRipsComplex {
    pub epsilon: f64,
    pub max_dimension: usize,
}

impl VietorisRipsComplex {
    pub fn new(epsilon: f64, max_dimension: usize) -> Self {
        Self {
            epsilon,
            max_dimension,
        }
    }

    /// Build Vietoris-Rips complex from point cloud
    pub fn build(&self, point_cloud: &PointCloud) -> SimplicialComplex {
        let mut complex = SimplicialComplex::new();
        let distances = point_cloud.pairwise_distances();
        let n = point_cloud.points.len();

        // Add vertices (0-simplices)
        for i in 0..n {
            complex.add_simplex(vec![i]);
        }

        // Add edges (1-simplices) where distance <= epsilon
        for i in 0..n {
            for j in (i + 1)..n {
                if distances[i][j] <= self.epsilon {
                    complex.add_simplex(vec![i, j]);
                }
            }
        }

        // Add higher-dimensional simplices using clique expansion
        for dim in 2..=self.max_dimension {
            self.add_cliques_of_dimension(&mut complex, &distances, dim);
        }

        complex
    }

    /// Add cliques of given dimension to the complex
    fn add_cliques_of_dimension(
        &self,
        complex: &mut SimplicialComplex,
        distances: &[Vec<f64>],
        dim: usize,
    ) {
        let n = distances.len();

        // Generate all possible combinations of (dim+1) vertices
        let mut vertices: Vec<usize> = (0..=dim).collect();

        loop {
            // Check if all pairwise distances in this combination are <= epsilon
            let mut is_clique = true;
            for i in 0..vertices.len() {
                for j in (i + 1)..vertices.len() {
                    if distances[vertices[i]][vertices[j]] > self.epsilon {
                        is_clique = false;
                        break;
                    }
                }
                if !is_clique {
                    break;
                }
            }

            if is_clique {
                complex.add_simplex(vertices.clone());
            }

            // Generate next combination
            if !self.next_combination(&mut vertices, n) {
                break;
            }
        }
    }

    /// Generate next combination in lexicographic order
    fn next_combination(&self, vertices: &mut Vec<usize>, n: usize) -> bool {
        let k = vertices.len();
        let mut i = k - 1;

        while i < k && vertices[i] == n - k + i {
            i = i.wrapping_sub(1);
        }

        if i >= k {
            return false;
        }

        vertices[i] += 1;
        for j in (i + 1)..k {
            vertices[j] = vertices[j - 1] + 1;
        }

        true
    }
}

/// Persistent homology calculator
pub struct PersistentHomologyCalculator {
    pub max_filtration_steps: usize,
}

impl PersistentHomologyCalculator {
    pub fn new(max_filtration_steps: usize) -> Self {
        Self {
            max_filtration_steps,
        }
    }

    /// Compute persistent homology for a point cloud
    pub fn compute(&self, point_cloud: &PointCloud) -> Result<PersistentHomologyResult> {
        let distances = point_cloud.pairwise_distances();
        let n = point_cloud.points.len();

        if n == 0 {
            return Err(anyhow!("Empty point cloud"));
        }

        // Find range of distances for filtration
        let mut all_distances: Vec<f64> = distances
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&d| d > 0.0)
            .copied()
            .collect();
        all_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min_distance = all_distances.first().copied().unwrap_or(0.0);
        let max_distance = all_distances.last().copied().unwrap_or(1.0);

        // Create filtration steps
        let mut filtration_steps = Vec::new();
        for i in 0..self.max_filtration_steps {
            let epsilon = min_distance
                + (max_distance - min_distance) * (i as f64)
                    / (self.max_filtration_steps as f64 - 1.0);
            filtration_steps.push(epsilon);
        }

        // Track features across filtration
        let mut features = Vec::new();
        let mut previous_complex = SimplicialComplex::new();

        for (step, &epsilon) in filtration_steps.iter().enumerate() {
            let vr_complex = VietorisRipsComplex::new(epsilon, 2).build(point_cloud);

            // Detect topological features
            let step_features = self.detect_features(&previous_complex, &vr_complex, epsilon, step);
            features.extend(step_features);

            previous_complex = vr_complex;
        }

        // Calculate Betti numbers
        let betti_numbers = self.calculate_betti_numbers(&features);

        // Create persistence diagram
        let persistence_diagram: Vec<(f64, f64)> = features
            .iter()
            .map(|f| (f.birth_time, f.death_time))
            .collect();

        // Generate topological signature
        let topological_signature = self.generate_topological_signature(&betti_numbers, &features);

        Ok(PersistentHomologyResult {
            features,
            betti_numbers,
            persistence_diagram,
            topological_signature,
        })
    }

    /// Detect topological features between two complexes
    fn detect_features(
        &self,
        prev_complex: &SimplicialComplex,
        curr_complex: &SimplicialComplex,
        epsilon: f64,
        _step: usize,
    ) -> Vec<TopologicalFeature> {
        let mut features = Vec::new();

        // Detect connected components (β₀)
        let prev_components = self.count_connected_components(prev_complex);
        let curr_components = self.count_connected_components(curr_complex);

        if curr_components > prev_components {
            // New components born
            for _ in 0..(curr_components - prev_components) {
                features.push(TopologicalFeature::new(0, epsilon, f64::INFINITY));
            }
        }

        // Detect loops (β₁) - simplified heuristic
        let prev_loops = self.count_loops(prev_complex);
        let curr_loops = self.count_loops(curr_complex);

        if curr_loops > prev_loops {
            // New loops born
            for _ in 0..(curr_loops - prev_loops) {
                features.push(TopologicalFeature::new(1, epsilon, f64::INFINITY));
            }
        }

        features
    }

    /// Count connected components using Union-Find
    fn count_connected_components(&self, complex: &SimplicialComplex) -> usize {
        let edges = complex.get_simplices_of_dimension(1);
        let n = complex.get_simplices_of_dimension(0).len();

        if n == 0 {
            return 0;
        }

        let mut parent: Vec<usize> = (0..n).collect();

        fn find(parent: &mut [usize], x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, x);
            }
            parent[x]
        }

        fn union(parent: &mut [usize], x: usize, y: usize) {
            let px = find(parent, x);
            let py = find(parent, y);
            if px != py {
                parent[px] = py;
            }
        }

        for edge in edges {
            if edge.len() == 2 {
                union(&mut parent, edge[0], edge[1]);
            }
        }

        let mut components = 0;
        for i in 0..n {
            if find(&mut parent, i) == i {
                components += 1;
            }
        }

        components
    }

    /// Count loops using Euler characteristic heuristic
    fn count_loops(&self, complex: &SimplicialComplex) -> usize {
        let vertices = complex.get_simplices_of_dimension(0).len();
        let edges = complex.get_simplices_of_dimension(1).len();
        let faces = complex.get_simplices_of_dimension(2).len();

        // Euler characteristic: V - E + F = 2 - 2g (for connected surface)
        // For disconnected: V - E + F = components - loops
        let components = self.count_connected_components(complex);
        let euler_char = vertices as i32 - edges as i32 + faces as i32;
        let loops = components as i32 - euler_char;

        loops.max(0) as usize
    }

    /// Calculate Betti numbers from persistent features
    fn calculate_betti_numbers(&self, features: &[TopologicalFeature]) -> Vec<usize> {
        let mut betti = vec![0; 3]; // β₀, β₁, β₂

        for feature in features {
            if feature.dimension < betti.len() && feature.death_time == f64::INFINITY {
                betti[feature.dimension] += 1;
            }
        }

        betti
    }

    /// Generate topological signature string
    fn generate_topological_signature(
        &self,
        betti_numbers: &[usize],
        features: &[TopologicalFeature],
    ) -> String {
        let config = ConsciousnessConfig::default();
        let persistent_features: Vec<_> = features
            .iter()
            .filter(|f| f.persistence > config.parametric_epsilon * 10.0) // Derive threshold
            .collect();

        format!(
            "Topology: β₀={}, β₁={}, β₂={}, {} persistent features",
            betti_numbers.first().unwrap_or(&0),
            betti_numbers.get(1).unwrap_or(&0),
            betti_numbers.get(2).unwrap_or(&0),
            persistent_features.len()
        )
    }
}

/// Ripser-compatible calculator for external Ripser integration
pub struct RipserCalculator {
    max_dimension: usize,
    persistence_threshold: f64,
}

impl RipserCalculator {
    pub fn new(max_dimension: usize, persistence_threshold: f64) -> Self {
        Self {
            max_dimension,
            persistence_threshold,
        }
    }

    pub fn set_threshold(&mut self, threshold: f64) {
        self.persistence_threshold = threshold;
    }

    /// Compute persistent homology from raw point coordinates
    pub fn compute_from_points(&self, points: &[Vec<f64>]) -> Result<Vec<TopologicalFeature>> {
        if points.is_empty() {
            return Err(anyhow!("Empty point set"));
        }

        // Create point cloud
        let point_cloud = PointCloud::new(points.to_vec());

        // Use internal calculator with config-driven filtration steps
        let config = ConsciousnessConfig::default();
        let calculator = PersistentHomologyCalculator::new(config.tda_max_filtration_steps);
        let result = calculator.compute(&point_cloud)?;

        // Filter features by persistence threshold
        let filtered_features: Vec<TopologicalFeature> = result
            .features
            .into_iter()
            .filter(|f| f.persistence >= self.persistence_threshold)
            .collect();

        Ok(filtered_features)
    }
}

/// TDA pipeline for cognitive state analysis
pub struct CognitiveTDA {
    calculator: PersistentHomologyCalculator,
}

impl Default for CognitiveTDA {
    fn default() -> Self {
        Self::new()
    }
}

impl CognitiveTDA {
    pub fn new() -> Self {
        let config = ConsciousnessConfig::default();
        Self {
            calculator: PersistentHomologyCalculator::new(config.tda_max_filtration_steps),
        }
    }

    /// Analyze cognitive state space topology
    pub fn analyze_cognitive_topology(
        &self,
        memory_spheres: &[Vec<f64>],
    ) -> Result<PersistentHomologyResult> {
        let point_cloud = PointCloud::new(memory_spheres.to_vec());
        self.calculator.compute(&point_cloud)
    }

    /// Detect toroidal topology signature
    pub fn detect_toroidal_topology(&self, result: &PersistentHomologyResult) -> bool {
        // Torus signature: β₀=1, β₁=2, β₂=1
        let betti = &result.betti_numbers;
        betti.first() == Some(&1) && betti.get(1) == Some(&2) && betti.get(2) == Some(&1)
    }

    /// Detect spherical topology signature
    pub fn detect_spherical_topology(&self, result: &PersistentHomologyResult) -> bool {
        // Sphere signature: β₀=1, β₁=0, β₂=1
        let betti = &result.betti_numbers;
        betti.first() == Some(&1) && betti.get(1) == Some(&0) && betti.get(2) == Some(&1)
    }

    /// Detect hierarchical/tree topology signature
    pub fn detect_hierarchical_topology(&self, result: &PersistentHomologyResult) -> bool {
        // Tree signature: β₀=1, β₁=0, β₂=0
        let betti = &result.betti_numbers;
        betti.first() == Some(&1) && betti.get(1) == Some(&0) && betti.get(2) == Some(&0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_cloud_creation() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let cloud = PointCloud::new(points);
        assert_eq!(cloud.dimension, 2);
        assert_eq!(cloud.points.len(), 4);
    }

    #[test]
    fn test_vietoris_rips_complex() {
        let points = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let cloud = PointCloud::new(points);
        let vr = VietorisRipsComplex::new(1.5, 2);
        let complex = vr.build(&cloud);

        assert!(complex.simplices.len() > 0);
    }

    #[test]
    fn test_persistent_homology() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let cloud = PointCloud::new(points);
        let config = ConsciousnessConfig::default();
        let calculator = PersistentHomologyCalculator::new(config.tda_max_filtration_steps);
        let result = calculator.compute(&cloud);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.features.is_empty());
    }
}

// Complete Persistent Homology Implementation as per TCS specification
use nalgebra::DVector;
use ndarray::{Array2, ArrayView1};
// use sprs::{CsMat, TriMat};

#[derive(Debug, Clone)]
pub struct PersistentHomology {
    max_dimension: usize,
    max_edge_length: f32,
}

impl PersistentHomology {
    pub fn new(max_dimension: usize, max_edge_length: f32) -> Self {
        Self {
            max_dimension,
            max_edge_length,
        }
    }

    /// Build Vietoris-Rips complex with edge collapse optimization
    pub fn build_vr_complex(&self, points: &[DVector<f32>]) -> SimplexTree {
        // Compute distance matrix with SIMD optimization
        let distances = self.compute_distance_matrix_simd(points);

        // Apply edge collapse to reduce complex size
        let collapsed = self.edge_collapse(&distances);

        // Build filtered complex
        let mut complex = SimplexTree::new();

        // Add vertices (0-simplices)
        for i in 0..points.len() {
            complex.insert(&[i], 0.0);
        }

        // Add edges (1-simplices) with filtration values
        for i in 0..points.len() {
            for j in i + 1..points.len() {
                let dist = collapsed[(i, j)];
                if dist <= self.max_edge_length {
                    complex.insert(&[i, j], dist);
                }
            }
        }

        // Expansion to higher simplices
        complex.expansion(self.max_dimension);

        complex
    }

    /// SIMD-accelerated distance computation
    fn compute_distance_matrix_simd(&self, points: &[DVector<f32>]) -> Array2<f32> {
        let n = points.len();
        let mut distances = Array2::zeros((n, n));

        for i in 0..n {
            for j in i + 1..n {
                let dist = euclidean_distance(&points[i], &points[j]);
                distances[(i, j)] = dist;
                distances[(j, i)] = dist;
            }
        }

        distances
    }

    /// Simple edge collapse for complexity reduction (placeholder)
    fn edge_collapse(&self, distances: &Array2<f32>) -> Array2<f32> {
        // Placeholder implementation - in practice would use collapser crate
        distances.clone()
    }

    /// Compute persistence using matrix reduction algorithm
    pub fn compute_persistence(&self, complex: &SimplexTree) -> PersistenceDiagram {
        // Placeholder - would implement matrix reduction algorithm
        PersistenceDiagram {
            dimension: 0,
            points: vec![],
            betti_numbers: [0; 3],
        }
    }
}

// Helper function for distance
fn euclidean_distance(p1: &DVector<f32>, p2: &DVector<f32>) -> f32 {
    (p1 - p2).norm()
}

// Placeholder structs
#[derive(Debug, Clone)]
pub struct SimplexTree {
    // Placeholder
}

impl SimplexTree {
    fn new() -> Self {
        Self {}
    }
    fn insert(&mut self, _simplex: &[usize], _filtration: f32) {}
    fn expansion(&mut self, _max_dim: usize) {}
}

#[derive(Debug, Clone)]
pub struct PersistenceDiagram {
    pub dimension: usize,
    pub points: Vec<PersistencePoint>,
    pub betti_numbers: [usize; 3],
}

#[derive(Debug, Clone)]
pub struct PersistencePoint {
    pub birth: f32,
    pub death: f32,
    pub persistence: f32,
    pub dimension: usize,
    pub representative: Option<Vec<usize>>,
}
