//! tda.rs - Persistent Homology via Python FFI Bridge
//!
//! Implements async FFI bridge to giotto-tda for computing persistent homology.
//! Uses zero-copy ndarray → NumPy conversion for performance.
//!
//! Based on NIODOO-CODE Blueprint Section 1.5

use ndarray::Array2;
use pyo3::prelude::*;
use thiserror::Error;

/// TDA computation errors
#[derive(Error, Debug)]
pub enum TdaError {
    #[error("Python error: {0}")]
    PythonError(String),

    #[error("Failed to convert data: {0}")]
    ConversionError(String),

    #[error("TDA computation failed: {0}")]
    ComputationFailed(String),
}

impl From<PyErr> for TdaError {
    fn from(err: PyErr) -> Self {
        TdaError::PythonError(format!("{}", err))
    }
}

/// A single persistence pair (birth, death, dimension)
#[derive(Debug, Clone)]
pub struct PersistencePair {
    pub birth: f64,
    pub death: f64,
    pub dimension: i32,
}

/// Persistence diagram (collection of pairs)
pub type PersistenceDiagram = Vec<PersistencePair>;

/// Compute persistent homology from adjacency matrix
///
/// This uses giotto-tda's VietorisRipsPersistence with precomputed metric.
///
/// # Arguments
///
/// * `matrix` - Adjacency matrix from control flow graph
///
/// # Returns
///
/// * `Ok(PersistenceDiagram)` - The computed persistence pairs
/// * `Err(TdaError)` - If computation fails
pub fn compute_persistence(matrix: &Array2<f64>) -> Result<PersistenceDiagram, TdaError> {
    Python::with_gil(|py| {
        // Import giotto-tda
        let gtda_homology = py.import_bound("gtda.homology")
            .map_err(|e| TdaError::ComputationFailed(format!("Failed to import giotto-tda: {}", e)))?;

        // Convert ndarray to NumPy (zero-copy when possible)
        let numpy_matrix = numpy::PyArray2::from_array_bound(py, matrix);

        // Create VietorisRipsPersistence with precomputed metric
        let kwargs = pyo3::types::PyDict::new_bound(py);
        kwargs.set_item("metric", "precomputed")?;
        kwargs.set_item("homology_dimensions", vec![0, 1, 2])?; // β₀, β₁, β₂

        let vr_persistence = gtda_homology
            .getattr("VietorisRipsPersistence")?
            .call((), Some(&kwargs))?;

        // Reshape matrix for giotto-tda (expects 3D: [n_samples, n_points, n_points])
        // We have one sample, so add batch dimension
        let np = py.import_bound("numpy")?;
        let reshaped = np
            .getattr("expand_dims")?
            .call1((numpy_matrix, 0))?;

        // Compute persistence diagram
        let result = vr_persistence
            .call_method1("fit_transform", (reshaped,))?;

        // Extract persistence pairs
        // giotto-tda returns shape [n_samples, n_pairs, 3] where last dim is [birth, death, dimension]
        let diagrams = result.extract::<numpy::PyReadonlyArray3<f64>>()?;
        let array_view = diagrams.as_array();

        let mut persistence_diagram = Vec::new();

        // Iterate over the first sample (index 0)
        if array_view.shape()[0] > 0 {
            for i in 0..array_view.shape()[1] {
                let birth = array_view[[0, i, 0]];
                let death = array_view[[0, i, 1]];
                let dimension = array_view[[0, i, 2]] as i32;

                persistence_diagram.push(PersistencePair {
                    birth,
                    death,
                    dimension,
                });
            }
        }

        Ok(persistence_diagram)
    })
}

/// Compute Betti numbers from persistence diagram
///
/// Betti numbers count topological features:
/// - β₀: Connected components
/// - β₁: Cycles/loops
/// - β₂: Voids/cavities
pub fn compute_betti_numbers(diagram: &PersistenceDiagram) -> (usize, usize, usize) {
    let mut beta_0 = 0;
    let mut beta_1 = 0;
    let mut beta_2 = 0;

    for pair in diagram {
        // Only count features that persist (not born and die at same time)
        if (pair.death - pair.birth).abs() > 1e-10 {
            match pair.dimension {
                0 => beta_0 += 1,
                1 => beta_1 += 1,
                2 => beta_2 += 1,
                _ => {}
            }
        }
    }

    (beta_0, beta_1, beta_2)
}
