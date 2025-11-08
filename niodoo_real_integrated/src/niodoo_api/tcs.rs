//! NIODOO TCS Module - Hybrid FFI Bridge to Python TDA Libraries
//!
//! This module implements the "Rust-Orchestrated Hybrid" architecture:
//! - Rust prepares data (adjacency matrix from parser)
//! - Calls Python's giotto-tda via pyo3-async-runtimes
//! - Returns PersistenceDiagram to Rust

use crate::niodoo_api::parser::AdjacencyMatrix;
use anyhow::{Context, Result};
use ndarray::Array2;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// Persistence diagram result from TDA computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceDiagram {
    /// Betti numbers: [β₀, β₁, β₂]
    pub betti_numbers: Vec<usize>,
    /// Persistence pairs: Vec<(birth, death, dimension)>
    pub persistence_pairs: Vec<(f64, f64, usize)>,
    /// Persistence entropy
    pub persistence_entropy: f64,
}

impl PersistenceDiagram {
    pub fn new(
        betti_numbers: Vec<usize>,
        persistence_pairs: Vec<(f64, f64, usize)>,
        persistence_entropy: f64,
    ) -> Self {
        Self {
            betti_numbers,
            persistence_pairs,
            persistence_entropy,
        }
    }

    /// Get Betti-0 (connected components)
    pub fn betti_0(&self) -> usize {
        self.betti_numbers.get(0).copied().unwrap_or(0)
    }

    /// Get Betti-1 (loops)
    pub fn betti_1(&self) -> usize {
        self.betti_numbers.get(1).copied().unwrap_or(0)
    }

    /// Get Betti-2 (voids)
    pub fn betti_2(&self) -> usize {
        self.betti_numbers.get(2).copied().unwrap_or(0)
    }
}

/// TCS analyzer that bridges to Python's giotto-tda
pub struct TCSAnalyzer;

impl TCSAnalyzer {
    /// Create a new TCS analyzer
    pub fn new() -> Self {
        Self
    }

    /// Analyze adjacency matrix and return persistence diagram
    /// 
    /// This implements the hybrid FFI bridge:
    /// 1. Convert adjacency matrix to distance matrix
    /// 2. Call Python's giotto-tda.VietorisRipsPersistence
    /// 3. Extract persistence diagram and return to Rust
    /// 
    /// NOTE: Currently synchronous. For async version with pyo3-async-runtimes,
    /// wrap this in tokio::task::spawn_blocking or use pyo3-async-runtimes::tokio::run_async
    pub fn analyze(&self, matrix: &AdjacencyMatrix) -> Result<PersistenceDiagram> {
        let start = std::time::Instant::now();

        // Convert adjacency matrix to distance matrix
        let distance_matrix = self.adjacency_to_distance(&matrix.matrix)?;

        // Call Python TDA computation via FFI
        let persistence_diagram = self.compute_persistence_python(&distance_matrix)?;

        let elapsed_ms = start.elapsed().as_millis();
        info!(
            betti_0 = persistence_diagram.betti_0(),
            betti_1 = persistence_diagram.betti_1(),
            betti_2 = persistence_diagram.betti_2(),
            elapsed_ms = elapsed_ms,
            "TDA computation completed"
        );

        if elapsed_ms > 500 {
            warn!(
                elapsed_ms = elapsed_ms,
                "TDA computation exceeded 500ms target"
            );
        }

        Ok(persistence_diagram)
    }

    /// Convert adjacency matrix to distance matrix (shortest path)
    fn adjacency_to_distance(&self, adj: &Array2<f32>) -> Result<Array2<f32>> {
        let n = adj.nrows();
        let mut dist = Array2::<f32>::from_elem((n, n), f32::INFINITY);

        // Initialize: direct edges have distance 1, self-loops have distance 0
        for i in 0..n {
            dist[[i, i]] = 0.0;
            for j in 0..n {
                if adj[[i, j]] > 0.0 {
                    dist[[i, j]] = 1.0;
                }
            }
        }

        // Floyd-Warshall algorithm for shortest paths
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if dist[[i, k]] != f32::INFINITY && dist[[k, j]] != f32::INFINITY {
                        let new_dist = dist[[i, k]] + dist[[k, j]];
                        if new_dist < dist[[i, j]] {
                            dist[[i, j]] = new_dist;
                        }
                    }
                }
            }
        }

        // Replace infinity with large finite value
        let max_dist = dist
            .iter()
            .filter(|&&d| d != f32::INFINITY)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(n as f32 * 2.0);

        for i in 0..n {
            for j in 0..n {
                if dist[[i, j]] == f32::INFINITY {
                    dist[[i, j]] = max_dist * 2.0;
                }
            }
        }

        Ok(dist)
    }

    /// Call Python's giotto-tda via FFI bridge
    /// 
    /// NOTE: This is currently synchronous. For async version, use pyo3-async-runtimes
    /// to call Python's asyncio.to_thread for non-blocking execution.
    fn compute_persistence_python(&self, distance_matrix: &Array2<f32>) -> Result<PersistenceDiagram> {
        // Use Python GIL
        Python::with_gil(|py| {
            // Import required Python modules
            let giotto_tda = PyModule::import_bound(py, "giotto.diagrams")
                .or_else(|_| PyModule::import_bound(py, "giotto_tda.diagrams"))
                .map_err(|e| anyhow::anyhow!("Failed to import giotto-tda: {}. Install with: pip install giotto-tda", e))?;

            // Convert Rust Array2 to Python numpy array
            // Convert to Vec<Vec<f32>> first, then to numpy array
            let np = PyModule::import_bound(py, "numpy")?;
            let array_fn = np.getattr("array")?;
            let matrix_vec: Vec<Vec<f32>> = distance_matrix
                .outer_iter()
                .map(|row| row.to_vec())
                .collect();
            let py_array = array_fn.call1((matrix_vec,))?;

            // Create VietorisRipsPersistence transformer
            let vr_class = giotto_tda.getattr("VietorisRipsPersistence")
                .map_err(|e| anyhow::anyhow!("Failed to get VietorisRipsPersistence class: {}", e))?;

            // Create instance with metric="precomputed"
            let vr_kwargs = PyDict::new_bound(py);
            vr_kwargs.set_item("metric", "precomputed")
                .map_err(|e| anyhow::anyhow!("Failed to set metric: {}", e))?;
            vr_kwargs.set_item("homology_dimensions", vec![0, 1, 2])
                .map_err(|e| anyhow::anyhow!("Failed to set homology_dimensions: {}", e))?;

            let vr_transformer = vr_class.call((), Some(&vr_kwargs))
                .map_err(|e| anyhow::anyhow!("Failed to create VietorisRipsPersistence transformer: {}", e))?;

            // Call fit_transform (synchronous call, but we're in async context)
            // For precomputed metric, pass distance matrix as list of arrays
            let input_data = vec![py_array];
            let fit_transform = vr_transformer.getattr("fit_transform")
                .map_err(|e| anyhow::anyhow!("Failed to get fit_transform method: {}", e))?;

            let persistence_result = fit_transform.call1((input_data,))
                .map_err(|e| anyhow::anyhow!("Failed to call fit_transform: {}", e))?;

            // Extract persistence diagrams
            // giotto-tda returns a list of numpy arrays, one per homology dimension
            let diagrams_list: Vec<PyObject> = persistence_result.extract()
                .map_err(|e| anyhow::anyhow!("Failed to extract persistence diagrams: {}", e))?;

            let mut betti_numbers = vec![0, 0, 0];
            let mut persistence_pairs = Vec::new();

            // Parse each diagram (one per dimension)
            for (dim, diagram_obj) in diagrams_list.iter().enumerate() {
                if dim >= 3 {
                    break;
                }

                // Try to extract as list of lists (numpy array format)
                if let Ok(diagram_list) = diagram_obj.extract::<Vec<Vec<f64>>>(py) {
                    betti_numbers[dim] = diagram_list.len();

                    // Extract persistence pairs (birth, death)
                    for point in diagram_list {
                        if point.len() >= 2 {
                            let birth = point[0];
                            let death = point[1];
                            persistence_pairs.push((birth, death, dim));
                        }
                    }
                }
            }

            // Compute persistence entropy
            let persistence_entropy = Self::compute_persistence_entropy_static(&persistence_pairs);

            Ok(PersistenceDiagram::new(
                betti_numbers,
                persistence_pairs,
                persistence_entropy,
            ))
        })
    }

    /// Static version of compute_persistence_entropy for use in async closures
    fn compute_persistence_entropy_static(pairs: &[(f64, f64, usize)]) -> f64 {
        if pairs.is_empty() {
            return 0.0;
        }

        let total_persistence: f64 = pairs.iter().map(|(b, d, _)| (d - b).abs()).sum();
        if total_persistence == 0.0 {
            return 0.0;
        }

        let entropy: f64 = pairs
            .iter()
            .map(|(b, d, _)| {
                let persistence = (d - b).abs();
                if persistence > 0.0 {
                    let p = persistence / total_persistence;
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum();

        entropy
    }

    /// Compute persistence entropy from persistence pairs
    fn compute_persistence_entropy(&self, pairs: &[(f64, f64, usize)]) -> f64 {
        if pairs.is_empty() {
            return 0.0;
        }

        let total_persistence: f64 = pairs.iter().map(|(b, d, _)| (d - b).abs()).sum();
        if total_persistence == 0.0 {
            return 0.0;
        }

        let entropy: f64 = pairs
            .iter()
            .map(|(b, d, _)| {
                let persistence = (d - b).abs();
                if persistence > 0.0 {
                    let p = persistence / total_persistence;
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum();

        entropy
    }
}

impl Default for TCSAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn analyze(
    py: Python,
    matrix: PyObject,
) -> PyResult<PyObject> {
    // Convert Python numpy array to Rust Array2
    let np = PyModule::import_bound(py, "numpy")?;
    let array_obj = matrix.bind(py);
    
    // Extract as list of lists and convert to Array2
    let list: Vec<Vec<f32>> = array_obj.extract()?;
    let n = list.len();
    let m = list.get(0).map(|r| r.len()).unwrap_or(0);
    let mut array = Array2::<f32>::zeros((n, m));
    for (i, row) in list.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            array[[i, j]] = val;
        }
    }
    let adj_matrix = AdjacencyMatrix::new(array);

    // Create analyzer and compute persistence diagram
    let analyzer = TCSAnalyzer::new();
    
    let persistence_diagram = analyzer.analyze(&adj_matrix)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;

    // Convert to Python dict
    let result = PyDict::new_bound(py);
    result.set_item("betti_numbers", persistence_diagram.betti_numbers)?;
    result.set_item("persistence_pairs", persistence_diagram.persistence_pairs)?;
    result.set_item("persistence_entropy", persistence_diagram.persistence_entropy)?;

    Ok(result.to_object(py))
}

#[cfg(feature = "pyo3")]
#[pymodule]
pub fn tcs(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::wrap_pyfunction;
    m.add_function(wrap_pyfunction!(analyze, m)?)?;
    Ok(())
}

