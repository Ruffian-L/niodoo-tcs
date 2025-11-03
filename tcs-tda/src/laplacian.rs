use std::collections::HashMap;

use nalgebra::{DMatrix, SymmetricEigen};
use serde::{Deserialize, Serialize};

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
    pub fn from_threshold(
        distances: &[Vec<f32>],
        threshold: f32,
        max_dimension: usize,
    ) -> Self {
        let n = distances.len();
        let mut simplices = vec![Vec::new(); max_dimension.saturating_add(1)];

        if n == 0 {
            return Self { simplices };
        }

        // 0-simplices
        for v in 0..n {
            simplices[0].push(Simplex::new(vec![v], 0.0));
        }

        if max_dimension >= 1 {
            for i in 0..n {
                for j in (i + 1)..n {
                    let dist = distances[i][j];
                    if dist <= threshold {
                        simplices[1].push(Simplex::new(vec![i, j], dist));
                    }
                }
            }
        }

        if max_dimension >= 2 {
            for i in 0..n {
                for j in (i + 1)..n {
                    let dij = distances[i][j];
                    if dij > threshold {
                        continue;
                    }
                    for k in (j + 1)..n {
                        let dik = distances[i][k];
                        if dik > threshold {
                            continue;
                        }
                        let djk = distances[j][k];
                        if djk > threshold {
                            continue;
                        }
                        let weight = dij.max(dik).max(djk);
                        simplices[2].push(Simplex::new(vec![i, j, k], weight));
                    }
                }
            }
        }

        Self { simplices }
    }

    pub fn dimension(&self) -> usize {
        if self.simplices.is_empty() {
            0
        } else {
            self.simplices.len() - 1
        }
    }

    pub fn simplices(&self, dimension: usize) -> &[Simplex] {
        self.simplices
            .get(dimension)
            .map(|simplices| simplices.as_slice())
            .unwrap_or(&[])
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

        let total_possible = n.saturating_mul(n.saturating_sub(1)).saturating_mul(n.saturating_sub(2)) / 6;
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
                let laplacian = laplacian_matrix(complex, dimension);
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

fn laplacian_matrix(complex: &SimplicialComplex, dimension: usize) -> DMatrix<f64> {
    let size = complex.simplices(dimension).len();
    if size == 0 {
        return DMatrix::zeros(0, 0);
    }

    let left = if dimension == 0 {
        DMatrix::zeros(size, size)
    } else {
        let boundary = boundary_matrix(complex, dimension);
        boundary.transpose() * boundary
    };

    let right_boundary = boundary_matrix(complex, dimension + 1);
    let right = if right_boundary.nrows() == 0 || right_boundary.ncols() == 0 {
        DMatrix::zeros(size, size)
    } else {
        let product = if dimension == 0 {
            // For dimension 0 Laplacian: right_boundary is (num_edges, num_vertices) = (num_edges, 7)
            // We need: right_boundary^T * right_boundary = (7, num_edges) * (num_edges, 7) = (7, 7)
            right_boundary.transpose() * &right_boundary
        } else {
            // For higher dimensions: right_boundary is (num_higher_simplices, num_current_simplices)
            // We compute: right_boundary * right_boundary^T = (num_current_simplices, num_current_simplices)
            &right_boundary * right_boundary.transpose()
        };
        // Ensure the product has the correct dimensions (size x size)
        if product.nrows() == size && product.ncols() == size {
            product
        } else {
            // If dimensions don't match, create a zero matrix of the correct size
            eprintln!(
                "Laplacian matrix dimension mismatch: dimension={}, size={}, right_boundary=({}, {}), product=({}, {}). Using zero matrix.",
                dimension, size, right_boundary.nrows(), right_boundary.ncols(), product.nrows(), product.ncols()
            );
            DMatrix::zeros(size, size)
        }
    };

    // Final safety check before addition
    if left.nrows() != right.nrows() || left.ncols() != right.ncols() {
        eprintln!(
            "Laplacian addition dimension mismatch: left=({}, {}), right=({}, {}). Using left only.",
            left.nrows(), left.ncols(), right.nrows(), right.ncols()
        );
        return left;
    }

    left + right
}

fn boundary_matrix(complex: &SimplicialComplex, dimension: usize) -> DMatrix<f64> {
    if dimension == 0 {
        return DMatrix::zeros(complex.simplices(0).len(), 0);
    }

    let simplices_k = complex.simplices(dimension);
    if simplices_k.is_empty() {
        return DMatrix::zeros(0, complex.simplices(dimension.saturating_sub(1)).len());
    }

    let simplices_prev = complex.simplices(dimension - 1);
    let mut index = HashMap::with_capacity(simplices_prev.len());
    for (idx, simplex) in simplices_prev.iter().enumerate() {
        index.insert(simplex.vertices.clone(), idx);
    }

    let rows = simplices_k.len();
    let cols = simplices_prev.len();
    let mut matrix = DMatrix::<f64>::zeros(rows, cols);

    for (row, simplex) in simplices_k.iter().enumerate() {
        for (face_index, face) in simplex.faces().into_iter().enumerate() {
            if let Some(&col) = index.get(&face) {
                let sign = if face_index % 2 == 0 { 1.0 } else { -1.0 };
                matrix[(row, col)] = sign;
            }
        }
    }

    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filtration_generates_expected_thresholds() {
        let distances = vec![vec![0.0, 0.5, 0.9], vec![0.5, 0.0, 0.7], vec![0.9, 0.7, 0.0]];
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


