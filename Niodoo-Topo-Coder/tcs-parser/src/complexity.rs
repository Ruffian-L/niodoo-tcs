//! complexity.rs - Code complexity metrics from AST traversal
//!
//! Implements cyclomatic and cognitive complexity analysis by walking
//! the tree-sitter AST. These metrics complement topological features
//! for NIODOO-CODE training data.
//!
//! NO STUBS. Real complexity calculation.

use tree_sitter::Tree;
use serde::{Deserialize, Serialize};

/// Complexity metrics for a code file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    /// Cyclomatic complexity (McCabe)
    /// Counts decision points: if, match, for, while, loop, &&, ||
    pub cyclomatic: usize,

    /// Cognitive complexity (SonarSource)
    /// Weights complexity by nesting depth
    pub cognitive: usize,

    /// Number of functions
    pub function_count: usize,

    /// Lines of code (non-comment, non-blank)
    pub loc: usize,
}

impl ComplexityMetrics {
    /// Creates default zero metrics
    pub fn new() -> Self {
        Self {
            cyclomatic: 1, // Base complexity is 1 (entry point)
            cognitive: 0,
            function_count: 0,
            loc: 0,
        }
    }
}

impl Default for ComplexityMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes complexity metrics for a parsed AST
///
/// # Arguments
///
/// * `tree` - The parsed AST from tree-sitter
/// * `source_code` - The original source code
///
/// # Returns
///
/// ComplexityMetrics containing cyclomatic, cognitive, and other metrics
///
/// # Example
///
/// ```no_run
/// use tcs_parser::{get_ast, complexity::compute_complexity};
///
/// let code = "fn main() { if true { println!(\"hi\"); } }";
/// let tree = get_ast(code, "rust").unwrap();
/// let metrics = compute_complexity(&tree, code);
/// println!("Cyclomatic complexity: {}", metrics.cyclomatic);
/// ```
pub fn compute_complexity(tree: &Tree, source_code: &str) -> ComplexityMetrics {
    let mut metrics = ComplexityMetrics::new();
    let root = tree.root_node();

    // Walk the AST and accumulate complexity
    walk_complexity(&root, source_code, &mut metrics, 0);

    // Compute LOC
    metrics.loc = count_loc(source_code);

    metrics
}

/// Recursively walk AST and accumulate complexity metrics
fn walk_complexity(
    node: &tree_sitter::Node,
    source: &str,
    metrics: &mut ComplexityMetrics,
    nesting_level: usize,
) {
    let kind = node.kind();

    match kind {
        // Function definitions
        "function_item" | "function_definition" => {
            metrics.function_count += 1;
        }

        // Cyclomatic complexity: decision points
        "if_expression" | "if_statement" => {
            metrics.cyclomatic += 1;
            metrics.cognitive += nesting_level + 1; // Cognitive weighted by nesting
        }

        "match_expression" | "match_statement" => {
            // Count each match arm as a decision point
            let arm_count = count_children_of_kind(node, "match_arm");
            metrics.cyclomatic += arm_count.max(1);
            metrics.cognitive += (nesting_level + 1) * arm_count.max(1);
        }

        "for_expression" | "for_statement" | "while_expression" | "while_statement"
        | "loop_expression" => {
            metrics.cyclomatic += 1;
            metrics.cognitive += nesting_level + 1;
        }

        // Boolean operators (each adds a path)
        "&&" | "||" | "binary_expression" => {
            if is_logical_operator(node, source) {
                metrics.cyclomatic += 1;
                // Cognitive complexity doesn't increment for boolean ops at same level
            }
        }

        // Error handling
        "?" | "try_expression" => {
            metrics.cyclomatic += 1;
        }

        _ => {}
    }

    // Recurse to children (increase nesting for blocks inside control flow)
    let should_increase_nesting = matches!(
        kind,
        "if_expression"
            | "if_statement"
            | "match_expression"
            | "match_statement"
            | "for_expression"
            | "for_statement"
            | "while_expression"
            | "while_statement"
            | "loop_expression"
    );

    let next_nesting = if should_increase_nesting {
        nesting_level + 1
    } else {
        nesting_level
    };

    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            walk_complexity(&child, source, metrics, next_nesting);
        }
    }
}

/// Count children of a specific kind
fn count_children_of_kind(node: &tree_sitter::Node, kind: &str) -> usize {
    let mut count = 0;
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            if child.kind() == kind {
                count += 1;
            }
        }
    }
    count
}

/// Check if a binary expression is a logical operator
fn is_logical_operator(node: &tree_sitter::Node, source: &str) -> bool {
    if node.kind() != "binary_expression" {
        return false;
    }

    // Check if operator child is && or ||
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            if let Ok(text) = child.utf8_text(source.as_bytes()) {
                if text == "&&" || text == "||" {
                    return true;
                }
            }
        }
    }

    false
}

/// Count lines of code (non-blank, non-comment)
fn count_loc(source: &str) -> usize {
    source
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty() && !trimmed.starts_with("//") && !trimmed.starts_with("#")
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::get_ast;

    #[test]
    fn test_simple_complexity() {
        let code = "fn main() { let x = 42; }";
        let tree = get_ast(code, "rust").expect("Failed to parse");
        let metrics = compute_complexity(&tree, code);

        assert_eq!(metrics.function_count, 1);
        assert!(metrics.cyclomatic >= 1, "Should have base complexity");
        assert!(metrics.loc > 0, "Should count lines");
    }

    #[test]
    fn test_if_complexity() {
        let code = r#"
fn test() {
    if true {
        println!("yes");
    }
}
        "#;
        let tree = get_ast(code, "rust").expect("Failed to parse");
        let metrics = compute_complexity(&tree, code);

        assert_eq!(metrics.function_count, 1);
        assert!(
            metrics.cyclomatic >= 2,
            "Should have complexity >= 2 (base + if)"
        );
        assert!(metrics.cognitive >= 1, "Should have cognitive complexity");
    }

    #[test]
    fn test_nested_complexity() {
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
        let metrics = compute_complexity(&tree, code);

        assert_eq!(metrics.function_count, 1);
        assert!(
            metrics.cyclomatic >= 2,
            "Should have if statement complexity"
        );
    }

    #[test]
    fn test_match_complexity() {
        let code = r#"
fn classify(x: i32) -> &'static str {
    match x {
        0 => "zero",
        1 => "one",
        2 => "two",
        _ => "other",
    }
}
        "#;
        let tree = get_ast(code, "rust").expect("Failed to parse");
        let metrics = compute_complexity(&tree, code);

        assert_eq!(metrics.function_count, 1);
        assert!(
            metrics.cyclomatic >= 2,
            "Match with 4 arms should add complexity"
        );
    }

    #[test]
    fn test_loop_complexity() {
        let code = r#"
fn sum_to_n(n: u32) -> u32 {
    let mut sum = 0;
    for i in 0..n {
        sum += i;
    }
    sum
}
        "#;
        let tree = get_ast(code, "rust").expect("Failed to parse");
        let metrics = compute_complexity(&tree, code);

        assert_eq!(metrics.function_count, 1);
        assert!(metrics.cyclomatic >= 2, "For loop adds complexity");
    }
}
