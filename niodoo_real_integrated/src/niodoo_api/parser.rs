//! NIODOO Parser Module - Code to Graph Conversion
//!
//! Parses source code into Control Flow Graphs (CFG) and converts to adjacency matrices
//! for topological analysis.

use crate::config::CodeLanguage;
use anyhow::{Context, Result};
use ndarray::Array2;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::Undirected;
use std::collections::HashMap;
use std::path::Path;
use tree_sitter::{Language, Parser, Tree};
use tree_sitter_rust;
use tree_sitter_python;
use tree_sitter_typescript;

/// Represents an adjacency matrix for topological analysis
#[derive(Debug, Clone)]
pub struct AdjacencyMatrix {
    pub matrix: Array2<f32>,
    pub num_nodes: usize,
    pub num_edges: usize,
}

impl AdjacencyMatrix {
    pub fn new(matrix: Array2<f32>) -> Self {
        let num_nodes = matrix.nrows();
        let num_edges = (matrix.sum() as usize) / 2; // Undirected graph, divide by 2
        Self {
            matrix,
            num_nodes,
            num_edges,
        }
    }
}

/// Parser for converting code to graphs
pub struct CodeParser {
    rust_parser: Parser,
    python_parser: Parser,
    typescript_parser: Parser,
}

impl CodeParser {
    /// Create a new code parser
    pub fn new() -> Result<Self> {
        let mut rust_parser = Parser::new();
        // LanguageFn wraps a function pointer - call it via into_raw() to get Language
        let rust_lang_fn = tree_sitter_rust::LANGUAGE.into_raw();
        let rust_lang = unsafe { std::mem::transmute::<*const (), tree_sitter::Language>(rust_lang_fn()) };
        rust_parser
            .set_language(&rust_lang)
            .context("Failed to load Rust language")?;

        let mut python_parser = Parser::new();
        let python_lang_fn = tree_sitter_python::LANGUAGE.into_raw();
        let python_lang = unsafe { std::mem::transmute::<*const (), tree_sitter::Language>(python_lang_fn()) };
        python_parser
            .set_language(&python_lang)
            .context("Failed to load Python language")?;

        let mut typescript_parser = Parser::new();
        let typescript_lang_fn = tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into_raw();
        let typescript_lang = unsafe { std::mem::transmute::<*const (), tree_sitter::Language>(typescript_lang_fn()) };
        typescript_parser
            .set_language(&typescript_lang)
            .context("Failed to load TypeScript language")?;

        Ok(Self {
            rust_parser,
            python_parser,
            typescript_parser,
        })
    }

    /// Parse code string and return adjacency matrix
    pub fn get_graph_from_file(&mut self, code: &str, language: CodeLanguage) -> Result<AdjacencyMatrix> {
        let tree = match language {
            CodeLanguage::Python => self
                .python_parser
                .parse(code, None)
                .context("Failed to parse Python code")?,
            CodeLanguage::TypeScript => self
                .typescript_parser
                .parse(code, None)
                .context("Failed to parse TypeScript code")?,
        };

        // Build CFG from AST
        let cfg = self.build_cfg_from_tree(&tree, language, code)?;

        // Convert CFG to petgraph::Graph
        let graph = self.cfg_to_petgraph(&cfg)?;

        // Convert petgraph::Graph to adjacency matrix
        let adjacency_matrix = self.petgraph_to_adjacency(&graph)?;

        Ok(adjacency_matrix)
    }

    /// Parse repository and build global graph
    pub fn get_graph_from_repo(&mut self, repo_path: &Path, language: CodeLanguage) -> Result<AdjacencyMatrix> {
        let mut all_nodes = Vec::new();
        let mut all_edges = Vec::new();
        let mut node_offset = 0;

        // Traverse directory and parse all matching files
        self.traverse_and_parse(repo_path, language, &mut all_nodes, &mut all_edges, &mut node_offset)?;

        // Build combined graph
        let n = all_nodes.len();
        let mut matrix = Array2::<f32>::zeros((n, n));

        for (from, to) in all_edges {
            if from < n && to < n {
                matrix[[from, to]] = 1.0;
                matrix[[to, from]] = 1.0; // Undirected
            }
        }

        Ok(AdjacencyMatrix::new(matrix))
    }

    /// Traverse directory and parse files
    fn traverse_and_parse(
        &mut self,
        path: &Path,
        language: CodeLanguage,
        nodes: &mut Vec<String>,
        edges: &mut Vec<(usize, usize)>,
        node_offset: &mut usize,
    ) -> Result<()> {
        if path.is_file() {
            let ext = path.extension().and_then(|s| s.to_str());
            let should_parse = match language {
                CodeLanguage::Python => ext == Some("py"),
                CodeLanguage::TypeScript => ext == Some("ts") || ext == Some("tsx"),
            };

            if should_parse {
                let code = std::fs::read_to_string(path)
                    .with_context(|| format!("Failed to read file: {}", path.display()))?;
                let tree = match language {
                    CodeLanguage::Python => self
                        .python_parser
                        .parse(&code, None)
                        .context("Failed to parse Python code")?,
                    CodeLanguage::TypeScript => self
                        .typescript_parser
                        .parse(&code, None)
                        .context("Failed to parse TypeScript code")?,
                };

                let cfg = self.build_cfg_from_tree(&tree, language, &code)?;
                let start_offset = *node_offset;

                // Add nodes
                for node in &cfg.nodes {
                    nodes.push(format!("{}:{}", path.display(), node.id));
                }

                // Add edges with offset
                for (from, to) in &cfg.edges {
                    edges.push((start_offset + *from as usize, start_offset + *to as usize));
                }

                *node_offset += cfg.nodes.len();
            }
        } else if path.is_dir() {
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                self.traverse_and_parse(&entry.path(), language, nodes, edges, node_offset)?;
            }
        }

        Ok(())
    }

    /// Build Control Flow Graph from AST tree
    fn build_cfg_from_tree(&self, tree: &Tree, language: CodeLanguage, code: &str) -> Result<ControlFlowGraph> {
        let root_node = tree.root_node();
        let mut cfg = ControlFlowGraph::new();

        match language {
            CodeLanguage::Python => self.build_cfg_python(&root_node, code, &mut cfg)?,
            CodeLanguage::TypeScript => self.build_cfg_typescript(&root_node, code, &mut cfg)?,
        }

        Ok(cfg)
    }

    /// Build CFG for Python AST
    fn build_cfg_python(&self, node: &tree_sitter::Node, code: &str, cfg: &mut ControlFlowGraph) -> Result<()> {
        self.build_cfg_recursive_python(node, u32::MAX, cfg)
    }

    /// Recursive helper for Python CFG building
    fn build_cfg_recursive_python(&self, node: &tree_sitter::Node, parent_id: u32, cfg: &mut ControlFlowGraph) -> Result<()> {
        let node_type = node.kind();
        let node_id = cfg.add_node(node_type, node.start_byte(), node.end_byte());

        if parent_id != u32::MAX {
            cfg.add_edge(parent_id, node_id);
        }

        // Detect function definitions (entry points)
        if node_type == "function_definition" {
            cfg.add_entry_point(node_id);
        }

        // Process children recursively
        let mut child_count = 0;
        let mut children_to_process = Vec::new();
        
        // Collect child indices first
        while child_count < node.child_count() {
            if let Some(_child) = node.child(child_count) {
                children_to_process.push(child_count);
            }
            child_count += 1;
        }

        if matches!(node_type, "if_statement" | "for_statement" | "while_statement") {
            // Process in order
            for &idx in &children_to_process {
                if let Some(child) = node.child(idx) {
                    self.build_cfg_recursive_python(&child, node_id, cfg)?;
                }
            }
        } else {
            // Process in reverse order
            for &idx in children_to_process.iter().rev() {
                if let Some(child) = node.child(idx) {
                    self.build_cfg_recursive_python(&child, node_id, cfg)?;
                }
            }
        }

        Ok(())
    }

    /// Build CFG for TypeScript AST
    fn build_cfg_typescript(&self, node: &tree_sitter::Node, code: &str, cfg: &mut ControlFlowGraph) -> Result<()> {
        self.build_cfg_recursive_typescript(node, u32::MAX, cfg)
    }

    /// Recursive helper for TypeScript CFG building
    fn build_cfg_recursive_typescript(&self, node: &tree_sitter::Node, parent_id: u32, cfg: &mut ControlFlowGraph) -> Result<()> {
        let node_type = node.kind();
        let node_id = cfg.add_node(node_type, node.start_byte(), node.end_byte());

        if parent_id != u32::MAX {
            cfg.add_edge(parent_id, node_id);
        }

        // Detect function definitions
        if matches!(
            node_type,
            "function_declaration" | "arrow_function" | "method_definition"
        ) {
            cfg.add_entry_point(node_id);
        }

        // Process children recursively
        let mut child_count = 0;
        let mut children_to_process = Vec::new();
        
        // Collect child indices first
        while child_count < node.child_count() {
            if let Some(child) = node.child(child_count) {
                children_to_process.push(child_count);
            }
            child_count += 1;
        }

        if matches!(node_type, "if_statement" | "for_statement" | "while_statement") {
            // Process in order
            for &idx in &children_to_process {
                if let Some(child) = node.child(idx) {
                    self.build_cfg_recursive_typescript(&child, node_id, cfg)?;
                }
            }
        } else {
            // Process in reverse order
            for &idx in children_to_process.iter().rev() {
                if let Some(child) = node.child(idx) {
                    self.build_cfg_recursive_typescript(&child, node_id, cfg)?;
                }
            }
        }

        Ok(())
    }

    /// Convert CFG to petgraph::Graph
    fn cfg_to_petgraph(&self, cfg: &ControlFlowGraph) -> Result<Graph<String, f32, Undirected>> {
        let mut graph = Graph::<String, f32, Undirected>::new_undirected();
        let mut node_map = HashMap::new();

        // Add nodes
        for node in &cfg.nodes {
            let idx = graph.add_node(format!("node_{}", node.id));
            node_map.insert(node.id, idx);
        }

        // Add edges
        for (from, to) in &cfg.edges {
            if let (Some(&from_idx), Some(&to_idx)) = (node_map.get(from), node_map.get(to)) {
                graph.add_edge(from_idx, to_idx, 1.0);
            }
        }

        Ok(graph)
    }

    /// Convert petgraph::Graph to adjacency matrix
    fn petgraph_to_adjacency(&self, graph: &Graph<String, f32, Undirected>) -> Result<AdjacencyMatrix> {
        let n = graph.node_count();
        let mut matrix = Array2::<f32>::zeros((n, n));

        // Build node index mapping
        let node_indices: Vec<NodeIndex> = graph.node_indices().collect();

        // Fill adjacency matrix
        for edge in graph.edge_indices() {
            let (a, b) = graph.edge_endpoints(edge).unwrap();
            let a_idx = node_indices.iter().position(|&x| x == a).unwrap();
            let b_idx = node_indices.iter().position(|&x| x == b).unwrap();
            matrix[[a_idx, b_idx]] = 1.0;
            matrix[[b_idx, a_idx]] = 1.0; // Undirected
        }

        Ok(AdjacencyMatrix::new(matrix))
    }
}

/// Control Flow Graph structure
#[derive(Debug, Clone)]
struct CFGNode {
    id: u32,
    node_type: String,
    start_byte: usize,
    end_byte: usize,
}

#[derive(Debug, Clone)]
struct ControlFlowGraph {
    nodes: Vec<CFGNode>,
    edges: Vec<(u32, u32)>,
    entry_points: Vec<u32>,
    next_id: u32,
}

impl ControlFlowGraph {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            entry_points: Vec::new(),
            next_id: 0,
        }
    }

    fn add_node(&mut self, node_type: &str, start_byte: usize, end_byte: usize) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(CFGNode {
            id,
            node_type: node_type.to_string(),
            start_byte,
            end_byte,
        });
        id
    }

    fn add_edge(&mut self, from: u32, to: u32) {
        self.edges.push((from, to));
    }

    fn add_entry_point(&mut self, node_id: u32) {
        self.entry_points.push(node_id);
    }
}

#[cfg(feature = "pyo3")]
use pyo3::{prelude::*, wrap_pyfunction, Bound};
#[cfg(feature = "pyo3")]
use pyo3::types::{PyModule, PyString};

#[cfg(feature = "pyo3")]
#[pyfunction]
fn get_graph_from_file(
    py: Python,
    code: &str,
    language: &str,
) -> PyResult<PyObject> {
    let lang = match language {
        "python" => CodeLanguage::Python,
        "typescript" => CodeLanguage::TypeScript,
        _ => return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Unsupported language: {}", language)
        )),
    };

    let mut parser = CodeParser::new()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;

    let adj_matrix = parser.get_graph_from_file(code, lang)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;

    // Convert Array2 to numpy array
    let np = PyModule::import_bound(py, "numpy")?;
    let array = np.getattr("array")?;
    let matrix_vec: Vec<Vec<f32>> = adj_matrix.matrix
        .outer_iter()
        .map(|row| row.to_vec())
        .collect();
    let py_array = array.call1((matrix_vec,))?;

    Ok(py_array.to_object(py))
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn get_graph_from_repo(
    py: Python,
    path: &str,
    language: &str,
) -> PyResult<PyObject> {
    let repo_path = Path::new(path);
    let lang = match language {
        "python" => CodeLanguage::Python,
        "typescript" => CodeLanguage::TypeScript,
        _ => return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Unsupported language: {}", language)
        )),
    };

    let mut parser = CodeParser::new()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;

    let adj_matrix = parser.get_graph_from_repo(repo_path, lang)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;

    // Convert to numpy array
    let np = PyModule::import_bound(py, "numpy")?;
    let array = np.getattr("array")?;
    let matrix_vec: Vec<Vec<f32>> = adj_matrix.matrix
        .outer_iter()
        .map(|row| row.to_vec())
        .collect();
    let py_array = array.call1((matrix_vec,))?;

    Ok(py_array.to_object(py))
}

#[cfg(feature = "pyo3")]
#[pymodule]
pub fn parser(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::wrap_pyfunction;
    m.add_function(wrap_pyfunction!(get_graph_from_file, m)?)?;
    m.add_function(wrap_pyfunction!(get_graph_from_repo, m)?)?;
    Ok(())
}

