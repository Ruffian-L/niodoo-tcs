//! matrix.rs - Graph to Adjacency Matrix conversion
//!
//! Converts petgraph control flow graphs into ndarray adjacency matrices
//! for topological data analysis (TDA) via giotto-tda.
//!
//! NO STUBS. Real matrix conversion for NIODOO-CODE.

use crate::graph::ControlFlowGraph;
use ndarray::Array2;
use thiserror::Error;

/// Matrix conversion errors
#[derive(Error, Debug)]
pub enum MatrixError {
    #[error("Graph is empty - cannot create matrix")]
    EmptyGraph,

    #[error("Matrix dimension error: {0}")]
    DimensionError(String),
}

/// Converts a control flow graph to an adjacency matrix
///
/// The adjacency matrix A is defined as:
/// - A[i][j] = 1 if there is an edge from node i to node j
/// - A[i][j] = 0 otherwise
///
/// This matrix can then be used for:
/// - Persistent homology computation (β₀, β₁, β₂)
/// - Spectral analysis
/// - Topological feature extraction
///
/// # Arguments
///
/// * `graph` - The control flow graph from AST
///
/// # Returns
///
/// * `Ok(Array2<f64>)` - The adjacency matrix
/// * `Err(MatrixError)` - If conversion fails
///
/// # Example
///
/// ```no_run
/// use tcs_parser::{get_ast, graph::ast_to_graph, matrix::graph_to_matrix};
///
/// let code = "fn main() { let x = 42; }";
/// let tree = get_ast(code, "rust").unwrap();
/// let graph = ast_to_graph(&tree, code).unwrap();
/// let matrix = graph_to_matrix(&graph).unwrap();
/// println!("Adjacency matrix shape: {:?}", matrix.shape());
/// ```
pub fn graph_to_matrix(graph: &ControlFlowGraph) -> Result<Array2<f64>, MatrixError> {
    let n = graph.node_count();

    if n == 0 {
        return Err(MatrixError::EmptyGraph);
    }

    // Initialize adjacency matrix with zeros
    let mut matrix = Array2::<f64>::zeros((n, n));

    // Fill in edges
    for edge in graph.raw_edges() {
        let source_idx = edge.source().index();
        let target_idx = edge.target().index();

        // Bounds check (should never fail but being defensive)
        if source_idx >= n || target_idx >= n {
            return Err(MatrixError::DimensionError(format!(
                "Edge ({}, {}) out of bounds for matrix size {}",
                source_idx, target_idx, n
            )));
        }

        matrix[[source_idx, target_idx]] = 1.0;
    }

    Ok(matrix)
}

/// Computes the weighted adjacency matrix with edge weights
///
/// If edges have weights/attributes, this function can incorporate them.
/// For now, it returns a binary adjacency matrix (same as graph_to_matrix).
///
/// Future enhancement: Extract edge weights from EdgeData attributes.
pub fn graph_to_weighted_matrix(graph: &ControlFlowGraph) -> Result<Array2<f64>, MatrixError> {
    // For now, same as unweighted
    // TODO: Extract weights from edge.weight().edge_type or other attributes
    graph_to_matrix(graph)
}

/// Computes the Laplacian matrix L = D - A
///
/// Where:
/// - D is the degree matrix (diagonal with node degrees)
/// - A is the adjacency matrix
///
/// The Laplacian is useful for spectral graph analysis and
/// has connections to persistent homology.
pub fn graph_to_laplacian(graph: &ControlFlowGraph) -> Result<Array2<f64>, MatrixError> {
    let n = graph.node_count();

    if n == 0 {
        return Err(MatrixError::EmptyGraph);
    }

    let adj = graph_to_matrix(graph)?;

    // Compute degree matrix
    let mut degree = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        let deg: f64 = adj.row(i).sum(); // out-degree
        degree[[i, i]] = deg;
    }

    // L = D - A
    Ok(&degree - &adj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{get_ast, graph::ast_to_graph};

    #[test]
    fn test_simple_matrix_conversion() {
        let code = "fn main() { let x = 42; }";
        let tree = get_ast(code, "rust").expect("Failed to parse");
        let graph = ast_to_graph(&tree, code).expect("Failed to build graph");

        let matrix = graph_to_matrix(&graph).expect("Failed to create matrix");

        // Matrix should be n x n where n = node count
        let n = graph.node_count();
        assert_eq!(matrix.shape(), &[n, n]);
        assert!(n >= 1, "Should have at least one node");
    }

    #[test]
    fn test_complex_matrix_conversion() {
        let code = r#"
            fn factorial(n: u32) -> u32 {
                if n == 0 {
                    1
                } else {
                    n * factorial(n - 1)
                }
            }
        "#;

        let tree = get_ast(code, "rust").expect("Failed to parse");
        let graph = ast_to_graph(&tree, code).expect("Failed to build graph");
        let matrix = graph_to_matrix(&graph).expect("Failed to create matrix");

        let n = graph.node_count();
        assert_eq!(matrix.shape(), &[n, n]);
        assert!(n >= 3, "Should have multiple nodes");

        // All values should be 0 or 1
        for val in matrix.iter() {
            assert!(*val == 0.0 || *val == 1.0, "Adjacency matrix should be binary");
        }
    }

    #[test]
    fn test_laplacian_computation() {
        let code = "fn main() { let x = 42; }";
        let tree = get_ast(code, "rust").expect("Failed to parse");
        let graph = ast_to_graph(&tree, code).expect("Failed to build graph");

        let laplacian = graph_to_laplacian(&graph).expect("Failed to compute Laplacian");

        let n = graph.node_count();
        assert_eq!(laplacian.shape(), &[n, n]);
    }

    #[test]
    fn test_empty_graph_error() {
        use petgraph::graph::DiGraph;
        use crate::graph::NodeData;
        use crate::graph::EdgeData;

        let empty_graph: DiGraph<NodeData, EdgeData> = DiGraph::new();
        let result = graph_to_matrix(&empty_graph);

        assert!(matches!(result, Err(MatrixError::EmptyGraph)));
    }
}
