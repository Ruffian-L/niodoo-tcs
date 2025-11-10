//! tcs-parser: Code-to-Topology Ingestion Pipeline
//!
//! This crate implements the data ingestion layer for NIODOO-CODE,
//! converting raw source code into topological graph representations
//! compatible with the existing TCS TDA pipeline.
//!
//! ## Architecture
//!
//! Code (String) → tree-sitter::Tree → petgraph::Graph → ndarray::Array2 → FFI Bridge → giotto-tda
//!
//! ## NO HARD CODING. NO PRINTLN. NO STUBS. NO BULLSHITTING.

use tree_sitter::{Parser, Tree};
use thiserror::Error;

/// Errors that can occur during parsing
#[derive(Error, Debug)]
pub enum ParserError {
    #[error("Unsupported language: {0}")]
    UnsupportedLanguage(String),

    #[error("Failed to set language: {0}")]
    LanguageSetFailed(String),

    #[error("Failed to parse code (timeout or error)")]
    ParseFailed,
}

// Use language constants from crates
use tree_sitter_rust::LANGUAGE as RUST_LANGUAGE;
use tree_sitter_python::LANGUAGE as PYTHON_LANGUAGE;

/// Parses a string of source code into a tree-sitter AST.
///
/// # Arguments
///
/// * `code` - The source code to parse
/// * `language_name` - The language identifier ("rust" or "python")
///
/// # Returns
///
/// * `Ok(Tree)` - The parsed AST
/// * `Err(ParserError)` - If parsing fails
///
/// # Example
///
/// ```no_run
/// use tcs_parser::get_ast;
///
/// let code = "fn hello() { let a = 1; }";
/// let ast = get_ast(code, "rust").unwrap();
/// let root_node = ast.root_node();
/// println!("{}", root_node.to_sexp());
/// // Output: (source_file (function_item name: (identifier)...))
/// ```
pub fn get_ast(code: &str, language_name: &str) -> Result<Tree, ParserError> {
    let mut parser = Parser::new();

    // Select language grammar based on input
    let language = match language_name {
        "rust" => RUST_LANGUAGE.into(),
        "python" => PYTHON_LANGUAGE.into(),
        _ => return Err(ParserError::UnsupportedLanguage(language_name.to_string())),
    };

    // Set the language on the parser
    parser
        .set_language(&language)
        .map_err(|e| ParserError::LanguageSetFailed(e.to_string()))?;

    // Parse the code
    parser
        .parse(code, None)
        .ok_or(ParserError::ParseFailed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_rust_simple() {
        let code = "fn main() { let x = 42; }";
        let tree = get_ast(code, "rust").expect("Failed to parse Rust code");
        let root = tree.root_node();

        // Verify we got a valid AST
        assert_eq!(root.kind(), "source_file");
        assert!(root.child_count() > 0);
    }

    #[test]
    fn test_parse_python_simple() {
        let code = "def hello():\n    x = 42";
        let tree = get_ast(code, "python").expect("Failed to parse Python code");
        let root = tree.root_node();

        // Verify we got a valid AST
        assert_eq!(root.kind(), "module");
        assert!(root.child_count() > 0);
    }

    #[test]
    fn test_unsupported_language() {
        let code = "some code";
        let result = get_ast(code, "javascript");

        assert!(matches!(result, Err(ParserError::UnsupportedLanguage(_))));
    }

    #[test]
    fn test_parse_complex_rust() {
        let code = r#"
            fn factorial(n: u32) -> u32 {
                if n == 0 {
                    1
                } else {
                    n * factorial(n - 1)
                }
            }
        "#;

        let tree = get_ast(code, "rust").expect("Failed to parse complex Rust code");
        let root = tree.root_node();

        // Check for function definition
        let s_expr = root.to_sexp();
        assert!(s_expr.contains("function_item"));
        assert!(s_expr.contains("if_expression"));
    }
}

// Graph conversion module
pub mod graph;

// Matrix conversion module
pub mod matrix;

// Complexity metrics module
pub mod complexity;

// TDA module (Day 2: Python FFI bridge)
pub mod tda;

use graph::ast_to_graph;
use matrix::graph_to_matrix;
use ndarray::Array2;

/// Complete pipeline: Code → AST → Graph → Matrix
///
/// This is the main entry point for converting source code into
/// an adjacency matrix suitable for topological data analysis.
///
/// # Arguments
///
/// * `code` - The source code to parse
/// * `language_name` - The language identifier ("rust" or "python")
///
/// # Returns
///
/// * `Ok(Array2<f64>)` - The adjacency matrix
/// * `Err(ParserError | GraphError | MatrixError)` - If any step fails
///
/// # Example
///
/// ```no_run
/// use tcs_parser::code_to_matrix;
///
/// let code = "fn main() { let x = 42; }";
/// let matrix = code_to_matrix(code, "rust").unwrap();
/// println!("Matrix shape: {:?}", matrix.shape());
/// ```
pub fn code_to_matrix(
    code: &str,
    language_name: &str,
) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    // Step 1: Parse code to AST
    let tree = get_ast(code, language_name)?;

    // Step 2: Convert AST to control flow graph
    let graph = ast_to_graph(&tree, code)?;

    // Step 3: Convert graph to adjacency matrix
    let matrix = graph_to_matrix(&graph)?;

    Ok(matrix)
}
