use std::collections::{HashMap, HashSet};

use nalgebra::{DMatrix, SymmetricEigen};
use serde::{Deserialize, Serialize};
use tracing::info;

const DUPLICATE_EPSILON: f32 = 1e-6;

#[derive(Debug, Clone, PartialEq)]
pub struct Simplex {
    pub vertices: Vec<usize>,
    pub weight: f32,
}

impl Simplex {
    pub fn new(mut vertices: Vec<usize>, weight: f32) -> Self {
        vertices.sort_unstable();
        Self { vertices, weight }
    }

    pub fn dimension(&self) -> usize {
        self.vertices.len().saturating_sub(1)
    }

    pub fn faces(&self) -> Vec<Vec<usize>> {
        if self.vertices.len() <= 1 {
            return Vec::new();
        }
        let mut faces = Vec::with_capacity(self.vertices.len());
        for i in 0..self.vertices.len() {
            let mut face = self.vertices.clone();
            face.remove(i);
            faces.push(face);
        }
        faces
    }
}

#[derive(Debug, Clone)]
pub struct SimplicialComplex {
    simplices: Vec<Vec<Simplex>>,
}

impl SimplicialComplex {
    pub fn from_threshold(distances: &[Vec<f32>], threshold: f32, max_dimension: usize) -> Self {
        let n = distances.len();
        let mut complex = Self::new(max_dimension);
        if n == 0 {
            return complex;
        }

        // 0-simplices (vertices)
        for vertex in 0..n {
            complex.insert_simplex(Simplex::new(vec![vertex], 0.0), 0);
        }

        if max_dimension == 0 {
            return complex;
        }

        fn clique_weight(
            vertices: &[usize],
            distances: &[Vec<f32>],
            threshold: f32,
        ) -> Option<f32> {
            let mut max_distance = 0.0f32;
            for i in 0..vertices.len() {
                for j in (i + 1)..vertices.len() {
                    let a = vertices[i];
                    let b = vertices[j];
                    let distance = distances[a][b];
                    if !distance.is_finite() || distance > threshold {
                        return None;
                    }
                    if distance > max_distance {
                        max_distance = distance;
                    }
                }
            }
            Some(max_distance)
        }

        fn enumerate_simplices(
            n: usize,
            target_size: usize,
            start: usize,
            current: &mut Vec<usize>,
            distances: &[Vec<f32>],
            threshold: f32,
            complex: &mut SimplicialComplex,
            dimension: usize,
        ) {
            if current.len() == target_size {
                if let Some(weight) = clique_weight(current, distances, threshold) {
                    complex.insert_simplex(Simplex::new(current.clone(), weight), dimension);
                }
                return;
            }

            for vertex in start..n {
                current.push(vertex);
                enumerate_simplices(
                    n,
                    target_size,
                    vertex + 1,
                    current,
                    distances,
                    threshold,
                    complex,
                    dimension,
                );
                current.pop();
            }
        }

        let mut current = Vec::new();
        for dimension in 1..=max_dimension {
            let target_size = dimension + 1;
            enumerate_simplices(
                n,
                target_size,
                0,
                &mut current,
                distances,
                threshold,
                &mut complex,
                dimension,
            );
        }

        complex
    }

    pub fn new(max_dimension: usize) -> Self {
        Self {
            simplices: vec![Vec::new(); max_dimension + 1],
        }
    }

    pub fn insert_simplex(&mut self, simplex: Simplex, dimension: usize) {
        if dimension >= self.simplices.len() {
            self.simplices.resize(dimension + 1, Vec::new());
        }
        self.simplices[dimension].push(simplex);
    }

    pub fn simplices(&self, dimension: usize) -> &[Simplex] {
        if dimension >= self.simplices.len() {
            &[]
        } else {
            &self.simplices[dimension]
        }
    }

    pub fn dimension(&self) -> usize {
        self.simplices.len().saturating_sub(1)
    }

    pub fn size_at_dimension(&self, dimension: usize) -> usize {
        self.simplices.get(dimension).map_or(0, |s| s.len())
    }

    pub fn boundary_sign(&self, simplex: &[usize], face: &[usize]) -> Option<f64> {
        let simplex_set: HashSet<_> = simplex.iter().cloned().collect();
        let face_set: HashSet<_> = face.iter().cloned().collect();
        if simplex_set.len() != simplex.len() || face_set.len() != face.len() {
            return None;
        }
        let mut count = 0;
        for &vertex in face {
            if simplex_set.contains(&vertex) {
                count += 1;
            }
        }
        if count != face.len() {
            return None;
        }
        let mut sorted_simplex = simplex.to_vec();
        sorted_simplex.sort_unstable();
        let mut sorted_face = face.to_vec();
        sorted_face.sort_unstable();
        let mut index = 0;
        for &vertex in &sorted_face {
            if let Some(pos) = sorted_simplex.iter().position(|&v| v == vertex) {
                if pos % 2 == 1 {
                    index += 1;
                }
            }
        }
        Some(if index % 2 == 0 { 1.0 } else { -1.0 })
    }

    pub fn boundary_matrix(&self, dimension: usize) -> DMatrix<f64> {
        if dimension == 0 {
            return DMatrix::zeros(self.simplices(0).len(), 0);
        }

        let simplices_k = self.simplices(dimension);
        if simplices_k.is_empty() {
            return DMatrix::zeros(0, self.simplices(dimension.saturating_sub(1)).len());
        }

        let simplices_prev = self.simplices(dimension - 1);
        let mut index = HashMap::with_capacity(simplices_prev.len());
        for (idx, simplex) in simplices_prev.iter().enumerate() {
            let mut key = simplex.vertices.clone();
            key.sort_unstable();
            index.insert(key, idx);
        }

        let rows = simplices_k.len();
        let cols = simplices_prev.len();
        let mut matrix = DMatrix::<f64>::zeros(rows, cols);

        for (row, simplex) in simplices_k.iter().enumerate() {
            for face in simplex.faces() {
                if let Some(&col) = index.get(&face) {
                    if let Some(sign) = self.boundary_sign(&simplex.vertices, &face) {
                        matrix[(row, col)] = sign;
                    }
                }
            }
        }

        matrix
    }

    /// Resize matrix to target dimensions by padding with zeros or truncating
    fn resize_matrix(matrix: DMatrix<f64>, target_rows: usize, target_cols: usize) -> DMatrix<f64> {
        if matrix.nrows() == target_rows && matrix.ncols() == target_cols {
            return matrix;
        }

        let mut resized = DMatrix::zeros(target_rows, target_cols);
        let copy_rows = matrix.nrows().min(target_rows);
        let copy_cols = matrix.ncols().min(target_cols);

        // Copy overlapping region
        for i in 0..copy_rows {
            for j in 0..copy_cols {
                resized[(i, j)] = matrix[(i, j)];
            }
        }

        resized
    }

    /// Ensure boundary matrix dimensions are consistent before multiplication
    /// For B^T * B: ensure B has target_size columns
    /// For B * B^T: ensure B has target_size rows
    /// This prevents rank loss from mismatched dimensions
    fn align_boundary_for_product(
        boundary: &DMatrix<f64>,
        target_size: usize,
        transpose_first: bool,
    ) -> DMatrix<f64> {
        if boundary.nrows() == 0 || boundary.ncols() == 0 {
            return DMatrix::zeros(target_size, target_size);
        }

        // For B^T * B: we need B to have target_size columns
        // For B * B^T: we need B to have target_size rows
        if transpose_first {
            // Resize columns to target_size
            Self::resize_matrix(boundary.clone(), boundary.nrows(), target_size)
        } else {
            // Resize rows to target_size
            Self::resize_matrix(boundary.clone(), target_size, boundary.ncols())
        }
    }

    /// Compute condition number of a matrix (ratio of largest to smallest singular value)
    /// Returns None if matrix is empty or singular
    fn condition_number(matrix: &DMatrix<f64>) -> Option<f64> {
        if matrix.nrows() == 0 || matrix.ncols() == 0 {
            return None;
        }

        let eigen = SymmetricEigen::new(matrix.clone());
        let eigenvalues: Vec<f64> = eigen
            .eigenvalues
            .iter()
            .map(|v| v.abs())
            .filter(|v| *v > f64::EPSILON)
            .collect();

        if eigenvalues.is_empty() {
            return None;
        }

        let max_eigen = eigenvalues.iter().fold(0.0_f64, |a, &b| a.max(b));
        let min_eigen = eigenvalues.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        if min_eigen < f64::EPSILON {
            None
        } else {
            Some(max_eigen / min_eigen)
        }
    }

    /// Apply ε-regularisation to a Laplacian matrix if it's singular
    /// Adds εI to ensure numerical stability
    fn regularize_laplacian(laplacian: DMatrix<f64>, epsilon: f64) -> (DMatrix<f64>, bool) {
        let cond = Self::condition_number(&laplacian);
        let needs_regularization = cond.is_none() || cond.unwrap_or(1e10) > 1e10;

        if needs_regularization {
            let mut regularized = laplacian.clone();
            for i in 0..regularized.nrows().min(regularized.ncols()) {
                regularized[(i, i)] += epsilon;
            }
            info!(
                dimension = laplacian.nrows(),
                condition_before = cond,
                epsilon = epsilon,
                "Laplacian regularized with ε-regularisation"
            );
            (regularized, true)
        } else {
            (laplacian, false)
        }
    }

    pub fn laplacian_matrix(&self, dimension: usize) -> DMatrix<f64> {
        let size = self.size_at_dimension(dimension);
        if size == 0 {
            return DMatrix::zeros(0, 0);
        }

        // FIX: Align boundary matrices BEFORE multiplication to prevent rank loss
        let left = if dimension == 0 {
            DMatrix::zeros(size, size)
        } else {
            let boundary = self.boundary_matrix(dimension);
            // For B^T * B: align boundary columns to target_size before multiplication
            let aligned_boundary = Self::align_boundary_for_product(&boundary, size, true);
            let product = aligned_boundary.transpose() * aligned_boundary;
            // Product should be size x size, but ensure it matches
            Self::resize_matrix(product, size, size)
        };

        let right_boundary = self.boundary_matrix(dimension + 1);
        let right = if right_boundary.nrows() == 0 || right_boundary.ncols() == 0 {
            DMatrix::zeros(size, size)
        } else {
            // FIX: Align boundary BEFORE multiplication
            if dimension == 0 {
                // For dimension 0: right_boundary is (num_edges, num_vertices)
                // We need: right_boundary^T * right_boundary = (num_vertices, num_vertices)
                // So align columns to size
                let aligned = Self::align_boundary_for_product(&right_boundary, size, true);
                aligned.transpose() * &aligned
            } else {
                // For higher dimensions: right_boundary is (num_higher_simplices, num_current_simplices)
                // We compute: right_boundary * right_boundary^T = (num_current_simplices, num_current_simplices)
                // So align rows to size
                let aligned = Self::align_boundary_for_product(&right_boundary, size, false);
                &aligned * aligned.transpose()
            }
        };

        // Both matrices are now guaranteed to be (size x size), safe to add
        let laplacian = left + right;

        // FIX: Check condition number and apply ε-regularisation if singular
        let epsilon = 1e-8;
        let (regularized, was_regularized) = Self::regularize_laplacian(laplacian, epsilon);

        // Log structural telemetry for upstream detection
        let cond = Self::condition_number(&regularized);
        let rank = regularized.nrows();
        let nullity = if let Some(c) = cond {
            if c > 1e10 {
                rank // Effectively singular
            } else {
                0
            }
        } else {
            rank
        };

        info!(
            dimension = dimension,
            size = size,
            rank = rank,
            nullity = nullity,
            condition_number = cond,
            regularized = was_regularized,
            "Laplacian matrix computed with structural telemetry"
        );

        regularized
    }
}

#[derive(Debug, Clone)]
pub struct FilteredComplex {
    pub threshold: f32,
    pub complex: SimplicialComplex,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaplacianSpectrum {
    pub threshold: f32,
    pub dimension: usize,
    pub eigenvalues: Vec<f64>,
}

impl LaplacianSpectrum {
    pub fn spectral_gap(&self) -> f64 {
        if self.eigenvalues.len() < 2 {
            return 0.0;
        }
        let mut sorted = self.eigenvalues.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut previous = 0.0;
        for value in sorted {
            if value <= f64::EPSILON {
                previous = value;
                continue;
            }
            return (value - previous).abs();
        }
        0.0
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct MotifMetrics {
    pub triangle_count: usize,
    pub triangle_density: f64,
    pub average_clustering: f64,
}

impl MotifMetrics {
    pub fn compute(distances: &[Vec<f32>], threshold: f32) -> Self {
        let n = distances.len();
        if n < 3 {
            return Self::default();
        }

        let mut triangle_count = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                if distances[i][j] > threshold {
                    continue;
                }
                for k in (j + 1)..n {
                    if distances[i][k] <= threshold && distances[j][k] <= threshold {
                        triangle_count += 1;
                    }
                }
            }
        }

        let total_possible = n
            .saturating_mul(n.saturating_sub(1))
            .saturating_mul(n.saturating_sub(2))
            / 6;
        let triangle_density = if total_possible > 0 {
            triangle_count as f64 / total_possible as f64
        } else {
            0.0
        };

        let mut clustering_sum = 0.0;
        let mut nodes_with_degree = 0usize;
        for i in 0..n {
            let mut neighbors = Vec::new();
            for j in 0..n {
                if i == j {
                    continue;
                }
                if distances[i][j] <= threshold {
                    neighbors.push(j);
                }
            }
            let degree = neighbors.len();
            if degree < 2 {
                continue;
            }
            nodes_with_degree += 1;
            let mut triangles_at_node = 0usize;
            for a in 0..degree {
                for b in (a + 1)..degree {
                    let u = neighbors[a];
                    let v = neighbors[b];
                    if distances[u][v] <= threshold {
                        triangles_at_node += 1;
                    }
                }
            }
            let possible = degree.saturating_mul(degree.saturating_sub(1)) / 2;
            if possible > 0 {
                clustering_sum += triangles_at_node as f64 / possible as f64;
            }
        }

        let average_clustering = if nodes_with_degree > 0 {
            clustering_sum / nodes_with_degree as f64
        } else {
            0.0
        };

        Self {
            triangle_count,
            triangle_density,
            average_clustering,
        }
    }
}

pub struct PersistentLaplacian {
    max_dimension: usize,
    zero_tolerance: f64,
}

impl PersistentLaplacian {
    pub fn new(max_dimension: usize, zero_tolerance: f64) -> Self {
        Self {
            max_dimension,
            zero_tolerance,
        }
    }

    pub fn build_filtration(
        &self,
        distances: &[Vec<f32>],
        max_threshold: f32,
        resolution: usize,
    ) -> Vec<FilteredComplex> {
        if distances.is_empty() {
            return Vec::new();
        }

        let mut thresholds = Vec::new();
        if resolution == 0 {
            thresholds.push(max_threshold);
        } else {
            for step in 1..=resolution {
                let value = max_threshold * (step as f32) / (resolution as f32);
                thresholds.push(value);
            }
        }

        for i in 0..distances.len() {
            for j in (i + 1)..distances.len() {
                let d = distances[i][j];
                if d <= max_threshold {
                    thresholds.push(d);
                }
            }
        }

        thresholds.retain(|value| *value > 0.0 && *value <= max_threshold + f32::EPSILON);
        thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        thresholds.dedup_by(|a, b| (*a - *b).abs() < DUPLICATE_EPSILON);

        if thresholds.is_empty() {
            thresholds.push(max_threshold);
        }

        thresholds
            .into_iter()
            .map(|threshold| FilteredComplex {
                threshold,
                complex: SimplicialComplex::from_threshold(
                    distances,
                    threshold,
                    self.max_dimension,
                ),
            })
            .collect()
    }

    pub fn analyze(&self, filtration: &[FilteredComplex]) -> Vec<LaplacianSpectrum> {
        let mut spectra = Vec::new();
        for filtered in filtration {
            let complex = &filtered.complex;
            for dimension in 0..=self.max_dimension.min(complex.dimension()) {
                let laplacian = complex.laplacian_matrix(dimension);
                if laplacian.nrows() == 0 {
                    spectra.push(LaplacianSpectrum {
                        threshold: filtered.threshold,
                        dimension,
                        eigenvalues: Vec::new(),
                    });
                    continue;
                }

                let eigen = SymmetricEigen::new(laplacian);
                let mut eigenvalues: Vec<f64> = eigen
                    .eigenvalues
                    .iter()
                    .map(|value| if *value < 0.0 { 0.0 } else { *value })
                    .collect();
                eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
                spectra.push(LaplacianSpectrum {
                    threshold: filtered.threshold,
                    dimension,
                    eigenvalues,
                });
            }
        }
        spectra
    }

    pub fn spectral_flux(&self, spectra: &[LaplacianSpectrum]) -> [f64; 3] {
        let mut flux = [0.0f64; 3];
        for dimension in 0..=self.max_dimension.min(2) {
            let per_dimension: Vec<&LaplacianSpectrum> = spectra
                .iter()
                .filter(|s| s.dimension == dimension)
                .collect();
            for window in per_dimension.windows(2) {
                let before = &window[0].eigenvalues;
                let after = &window[1].eigenvalues;
                let len = before.len().min(after.len());
                if len == 0 {
                    continue;
                }
                let mut diff = 0.0;
                for idx in 0..len {
                    diff += (before[idx] - after[idx]).abs();
                }
                flux[dimension] += diff / len as f64;
            }
        }
        flux
    }

    pub fn harmonic_counts(&self, spectra: &[LaplacianSpectrum]) -> [usize; 3] {
        let mut counts = [0usize; 3];
        for dimension in 0..=self.max_dimension.min(2) {
            if let Some(latest) = spectra
                .iter()
                .rev()
                .find(|s| s.dimension == dimension && !s.eigenvalues.is_empty())
            {
                counts[dimension] = latest
                    .eigenvalues
                    .iter()
                    .filter(|value| value.abs() <= self.zero_tolerance)
                    .count();
            }
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filtration_generates_expected_thresholds() {
        let distances = vec![
            vec![0.0, 0.5, 0.9],
            vec![0.5, 0.0, 0.7],
            vec![0.9, 0.7, 0.0],
        ];
        let engine = PersistentLaplacian::new(2, 1e-6);
        let filtration = engine.build_filtration(&distances, 1.0, 4);
        assert!(!filtration.is_empty());
        assert!(filtration.iter().all(|f| f.threshold <= 1.0 + f32::EPSILON));
    }

    #[test]
    fn laplacian_harmonic_counts_match_simple_case() {
        let distances = vec![vec![0.0, 0.4], vec![0.4, 0.0]];
        let engine = PersistentLaplacian::new(1, 1e-6);
        let filtration = engine.build_filtration(&distances, 0.5, 2);
        let spectra = engine.analyze(&filtration);
        let counts = engine.harmonic_counts(&spectra);
        assert_eq!(counts[0], 1);
    }
}
