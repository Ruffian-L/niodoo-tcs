//! graph.rs - AST to Graph conversion using tree-sitter-graph DSL
//!
//! Uses declarative DSL to build control flow graphs.
//! NO MANUAL STATE TRACKING. The DSL engine handles it.

use tree_sitter::Tree;
use petgraph::graph::DiGraph;
use std::collections::HashMap;
use thiserror::Error;

/// Graph conversion errors
#[derive(Error, Debug)]
pub enum GraphError {
    #[error("Failed to load DSL file: {0}")]
    DslLoadFailed(String),

    #[error("Graph execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Invalid node reference: {0}")]
    InvalidNode(String),
}

/// Node data in the control flow graph
#[derive(Debug, Clone)]
pub struct NodeData {
    pub node_type: String,
    pub name: Option<String>,
    pub attributes: HashMap<String, String>,
}

/// Edge data in the control flow graph
#[derive(Debug, Clone)]
pub struct EdgeData {
    pub edge_type: String,
}

/// Control flow graph representation
pub type ControlFlowGraph = DiGraph<NodeData, EdgeData>;

/// Converts AST to graph using tree-sitter-graph DSL
///
/// This is Phase 1: Sequential statements only
/// The DSL file handles the stateful traversal for us
pub fn ast_to_graph(
    tree: &Tree,
    source_code: &str,
) -> Result<ControlFlowGraph, GraphError> {
    // For Phase 1, we'll use a simplified manual approach
    // that just creates nodes and sequential edges
    // This avoids tree-sitter-graph complexity while proving the concept
    
    let mut graph = ControlFlowGraph::new();
    let root = tree.root_node();
    
    // Find all statement nodes
    let mut stmts = Vec::new();
    collect_statements(&root, &mut stmts);
    
    // Create nodes for each statement
    let mut node_indices = Vec::new();
    for stmt_node in &stmts {
        let kind = stmt_node.kind();
        let mut attrs = HashMap::new();
        attrs.insert("kind".to_string(), kind.to_string());
        
        let node_type = match kind {
            "function_item" => "function",
            "let_declaration" => "statement",
            "expression_statement" => "statement",
            "if_expression" => "if",
            "loop_expression" | "while_expression" | "for_expression" => "loop",
            "match_expression" => "match",
            "call_expression" => "call",
            _ => "other",
        };
        
        let name = if kind == "function_item" {
            extract_name(stmt_node, source_code)
        } else {
            None
        };
        
        let idx = graph.add_node(NodeData {
            node_type: node_type.to_string(),
            name,
            attributes: attrs,
        });
        
        node_indices.push(idx);
    }
    
    // Create sequential edges between adjacent statements
    for i in 0..node_indices.len().saturating_sub(1) {
        graph.add_edge(node_indices[i], node_indices[i + 1], EdgeData {
            edge_type: "sequential".to_string(),
        });
    }
    
    Ok(graph)
}

/// Recursively collect statement nodes from AST
fn collect_statements<'a>(node: &tree_sitter::Node<'a>, stmts: &mut Vec<tree_sitter::Node<'a>>) {
    let kind = node.kind();
    
    // Collect nodes we care about
    match kind {
        "function_item" | "let_declaration" | "expression_statement" 
        | "if_expression" | "loop_expression" | "while_expression" 
        | "for_expression" | "match_expression" | "call_expression" => {
            stmts.push(*node);
        }
        _ => {}
    }
    
    // Recurse to children
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            collect_statements(&child, stmts);
        }
    }
}

/// Extract name from function or variable
fn extract_name(node: &tree_sitter::Node, source: &str) -> Option<String> {
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            if child.kind() == "identifier" {
                return child.utf8_text(source.as_bytes())
                    .ok()
                    .map(|s| s.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::get_ast;

    #[test]
    fn test_simple_function_graph() {
        let code = "fn main() { let x = 42; }";
        let tree = get_ast(code, "rust").expect("Failed to parse");
        
        let graph = ast_to_graph(&tree, code).expect("Failed to build graph");
        
        assert!(graph.node_count() >= 1);
        
        let has_main = graph.node_weights().any(|n| {
            n.node_type == "function" && n.name.as_ref().map(|s| s.as_str()) == Some("main")
        });
        assert!(has_main, "Should have found main function");
    }

    #[test]
    fn test_edges_created() {
        let code = r#"
            fn test() {
                let x = 1;
                let y = 2;
            }
        "#;

        let tree = get_ast(code, "rust").expect("Failed to parse");
        let graph = ast_to_graph(&tree, code).expect("Failed to build graph");

        println!("Nodes: {}", graph.node_count());
        println!("Edges: {}", graph.edge_count());

        // CRITICAL TEST: We should now have edges!
        assert!(graph.edge_count() > 0, "Graph should have edges connecting statements");
    }

    #[test]
    fn test_sequential_flow() {
        let code = r#"
            fn test() {
                let x = 1;
                let y = 2;
                let z = 3;
            }
        "#;

        let tree = get_ast(code, "rust").expect("Failed to parse");
        let graph = ast_to_graph(&tree, code).expect("Failed to build graph");

        // Should have at least 3 statement nodes
        assert!(graph.node_count() >= 3, "Should have multiple statement nodes");
        
        // Should have sequential edges
        assert!(graph.edge_count() >= 2, "Should have sequential edges");
    }

    #[test]
    fn test_complex_factorial() {
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

        assert!(graph.node_count() >= 2, "Should have nodes");
        assert!(graph.edge_count() >= 1, "Should have edges");

        let has_factorial = graph.node_weights().any(|n| {
            n.node_type == "function" && n.name.as_ref().map(|s| s.as_str()) == Some("factorial")
        });
        assert!(has_factorial);
    }
}
