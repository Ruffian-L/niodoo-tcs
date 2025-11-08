//! Code topology analyzer for computing topological signatures of generated code
//!
//! This module parses generated code and computes topological metrics (Betti numbers,
//! persistence entropy) that can be used for reward shaping in the DQN learning loop.
//!
//! **RL Execution Harness Integration**: Extended to use real AST/CFG parsing and TCSAnalyzer
//! for actual topological computation instead of heuristics.

use crate::config::CodeLanguage;
use crate::tcs_analysis::{TCSAnalyzer, TopologicalSignature};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;

/// Control Flow Graph node types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CFGNodeType {
    BasicBlock,
    FunctionDef,
    Conditional,
    Loop,
}

/// Control Flow Graph node
#[derive(Debug, Clone)]
struct CFGNode {
    id: u32,
    line_start: usize,
    line_end: usize,
    node_type: CFGNodeType,
}

/// Control Flow Graph structure
#[derive(Debug, Clone)]
struct ControlFlowGraph {
    nodes: Vec<CFGNode>,
    edges: Vec<(u32, u32)>, // (from, to) node IDs
    entry_points: Vec<u32>, // Function entry points
}

/// Topological signature computed from code structure
#[derive(Debug, Clone)]
pub struct CodeTopologySignature {
    /// Cyclomatic complexity (proxy for Betti-1)
    pub cyclomatic_complexity: u32,
    /// Cognitive complexity
    pub cognitive_complexity: u32,
    /// Number of functions/methods (proxy for Betti-0)
    pub function_count: u32,
    /// Number of nested structures (proxy for Betti-2)
    pub nesting_depth: u32,
    /// Persistence entropy (computed from control flow graph)
    pub persistence_entropy: f64,
}

impl CodeTopologySignature {
    /// Convert to TopologicalSignature for use in learning loop
    pub fn to_topological_signature(&self) -> TopologicalSignature {
        // Create a simplified TopologicalSignature from code metrics
        TopologicalSignature::new(
            vec![], // persistence_features - empty for code topology
            [
                self.function_count as usize,
                self.cyclomatic_complexity as usize,
                self.nesting_depth as usize,
            ],
            self.cyclomatic_complexity as f64, // knot_complexity
            String::new(), // knot_polynomial
            0, // tqft_dimension
            None, // cobordism_type
            0.0, // computation_time_ms
            self.persistence_entropy,
            0.0, // spectral_gap
            0.0, // euler_characteristic
            0.0, // total_persistence
            0.0, // max_persistence
            0.0, // mean_persistence
            0.0, // laplacian_spectral_radius
        )
    }
}

/// Analyzer for computing topological signatures from code
pub struct CodeTopologyAnalyzer {
    // Cache for parsed ASTs (optional optimization)
    _ast_cache: HashMap<String, String>,
    // TCSAnalyzer for real topological computation
    tcs_analyzer: Option<Arc<Mutex<TCSAnalyzer>>>,
}

impl CodeTopologyAnalyzer {
    /// Create a new code topology analyzer
    pub fn new() -> Self {
        Self {
            _ast_cache: HashMap::new(),
            tcs_analyzer: None,
        }
    }

    /// Create a new code topology analyzer with TCSAnalyzer for real topology computation
    pub fn with_tcs_analyzer(tcs_analyzer: Arc<Mutex<TCSAnalyzer>>) -> Self {
        Self {
            _ast_cache: HashMap::new(),
            tcs_analyzer: Some(tcs_analyzer),
        }
    }

    /// Analyze code and compute topological signature
    /// 
    /// **RL Execution Harness**: Uses real AST/CFG parsing and TCSAnalyzer when available,
    /// falls back to heuristics if TCSAnalyzer is not provided.
    pub async fn analyze(&self, code: &str, language: CodeLanguage) -> Result<CodeTopologySignature> {
        // If TCSAnalyzer is available, use real topology computation
        if let Some(ref tcs) = self.tcs_analyzer {
            return self.analyze_with_tcs(code, language, tcs).await;
        }
        
        // Fallback to heuristics
        match language {
            CodeLanguage::Python => self.analyze_python(code).await,
            CodeLanguage::TypeScript => self.analyze_typescript(code).await,
        }
    }

    /// Analyze code using TCSAnalyzer for real topological metrics
    async fn analyze_with_tcs(
        &self,
        code: &str,
        language: CodeLanguage,
        tcs_analyzer: &Arc<Mutex<TCSAnalyzer>>,
    ) -> Result<CodeTopologySignature> {
        // Build Control Flow Graph from code
        let cfg = self.build_cfg_from_code(code, language)?;
        
        // Convert CFG to adjacency matrix (distance matrix for TCSAnalyzer)
        let adjacency_matrix = self.cfg_to_adjacency_matrix(&cfg);
        
        // Compute distance matrix from adjacency matrix
        // For CFG, we use graph distance (shortest path)
        let distances = self.adjacency_to_distances(&adjacency_matrix);
        
        // Use TCSAnalyzer to compute real topology
        let tcs = tcs_analyzer.lock().await;
        let max_filtration = 10.0; // Reasonable default for code graphs
        let snapshot = tcs.compute_topology_from_distances(&distances, max_filtration);
        
        // Extract metrics from snapshot
        let betti = snapshot.betti;
        let persistence_entropy = self.compute_persistence_entropy_from_features(&snapshot.features);
        
        // Compute cyclomatic complexity from CFG (number of edges - nodes + 1)
        let nodes = adjacency_matrix.len();
        let edges: usize = adjacency_matrix.iter()
            .map(|row| row.iter().filter(|&&w| w > 0.0).count())
            .sum();
        let cyclomatic_complexity = if nodes > 0 {
            (edges.saturating_sub(nodes) + 1) as u32
        } else {
            1
        };
        
        // Cognitive complexity: cyclomatic + nesting penalty
        let nesting_depth = self.compute_nesting_from_cfg(&cfg);
        let cognitive_complexity = (cyclomatic_complexity as f64 + nesting_depth as f64 * 0.5) as u32;
        
        // Function count from CFG entry points
        let function_count = cfg.entry_points.len() as u32;
        
        info!(
            betti = ?betti,
            cyclomatic_complexity,
            function_count,
            "Computed real code topology using TCSAnalyzer"
        );
        
        Ok(CodeTopologySignature {
            cyclomatic_complexity,
            cognitive_complexity,
            function_count,
            nesting_depth,
            persistence_entropy,
        })
    }

    /// Build Control Flow Graph from code
    fn build_cfg_from_code(&self, code: &str, language: CodeLanguage) -> Result<ControlFlowGraph> {
        match language {
            CodeLanguage::Python => self.build_cfg_python(code),
            CodeLanguage::TypeScript => self.build_cfg_typescript(code),
        }
    }

    /// Build CFG for Python code (simplified - identifies basic blocks and control flow)
    fn build_cfg_python(&self, code: &str) -> Result<ControlFlowGraph> {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut entry_points = Vec::new();
        
        let lines: Vec<&str> = code.lines().collect();
        let mut node_id = 0;
        let mut current_block_start = 0;
        
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            
            // Detect function definitions (entry points)
            if trimmed.starts_with("def ") {
                entry_points.push(node_id);
                
                // Create node for function definition
                nodes.push(CFGNode {
                    id: node_id,
                    line_start: i,
                    line_end: i,
                    node_type: CFGNodeType::FunctionDef,
                });
                node_id += 1;
                current_block_start = i + 1;
            }
            
            // Detect control flow statements
            if trimmed.starts_with("if ") || trimmed.starts_with("elif ") || trimmed.starts_with("else:") {
                // End previous block
                if current_block_start < i {
                    nodes.push(CFGNode {
                        id: node_id,
                        line_start: current_block_start,
                        line_end: i - 1,
                        node_type: CFGNodeType::BasicBlock,
                    });
                    if node_id > 0 {
                        edges.push((node_id - 1, node_id));
                    }
                    node_id += 1;
                }
                
                // Create conditional node
                nodes.push(CFGNode {
                    id: node_id,
                    line_start: i,
                    line_end: i,
                    node_type: CFGNodeType::Conditional,
                });
                if node_id > 0 {
                    edges.push((node_id - 1, node_id));
                }
                let conditional_id = node_id;
                node_id += 1;
                current_block_start = i + 1;
                
                // Edges will be added when we process the next block
            }
            
            // Detect loops
            if trimmed.starts_with("for ") || trimmed.starts_with("while ") {
                // End previous block if needed
                if current_block_start < i && node_id > 0 {
                    nodes.push(CFGNode {
                        id: node_id,
                        line_start: current_block_start,
                        line_end: i - 1,
                        node_type: CFGNodeType::BasicBlock,
                    });
                    if node_id > 0 {
                        edges.push((node_id - 1, node_id));
                    }
                    node_id += 1;
                }
                
                let loop_id = node_id;
                nodes.push(CFGNode {
                    id: loop_id,
                    line_start: i,
                    line_end: i,
                    node_type: CFGNodeType::Loop,
                });
                if loop_id > 0 {
                    edges.push((loop_id - 1, loop_id));
                }
                node_id += 1;
                current_block_start = i + 1;
                
                // Loop back edge will be added when we process the loop body
            }
        }
        
        // Add final block
        if current_block_start < lines.len() {
            nodes.push(CFGNode {
                id: node_id,
                line_start: current_block_start,
                line_end: lines.len() - 1,
                node_type: CFGNodeType::BasicBlock,
            });
            if node_id > 0 {
                edges.push((node_id - 1, node_id));
            }
        }
        
        // If no functions found, treat entire code as one entry point
        if entry_points.is_empty() {
            entry_points.push(0);
        }
        
        Ok(ControlFlowGraph {
            nodes,
            edges,
            entry_points,
        })
    }

    /// Build CFG for TypeScript code (similar to Python)
    fn build_cfg_typescript(&self, code: &str) -> Result<ControlFlowGraph> {
        // Similar implementation to Python, adapted for TypeScript syntax
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut entry_points = Vec::new();
        
        let lines: Vec<&str> = code.lines().collect();
        let mut node_id = 0;
        let mut current_block_start = 0;
        
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            
            // Detect function definitions
            if trimmed.contains("function ") || trimmed.contains("=>") || (trimmed.contains("(") && trimmed.contains("{")) {
                entry_points.push(node_id);
                nodes.push(CFGNode {
                    id: node_id,
                    line_start: i,
                    line_end: i,
                    node_type: CFGNodeType::FunctionDef,
                });
                node_id += 1;
                current_block_start = i;
            }
            
            // Detect control flow
            if trimmed.starts_with("if ") || trimmed.starts_with("else ") {
                if current_block_start < i {
                    nodes.push(CFGNode {
                        id: node_id,
                        line_start: current_block_start,
                        line_end: i - 1,
                        node_type: CFGNodeType::BasicBlock,
                    });
                    if node_id > 0 {
                        edges.push((node_id - 1, node_id));
                    }
                    node_id += 1;
                }
                
                nodes.push(CFGNode {
                    id: node_id,
                    line_start: i,
                    line_end: i,
                    node_type: CFGNodeType::Conditional,
                });
                if node_id > 0 {
                    edges.push((node_id - 1, node_id));
                }
                node_id += 1;
                current_block_start = i + 1;
            }
            
            // Detect loops
            if trimmed.starts_with("for ") || trimmed.starts_with("while ") {
                nodes.push(CFGNode {
                    id: node_id,
                    line_start: i,
                    line_end: i,
                    node_type: CFGNodeType::Loop,
                });
                if node_id > 0 {
                    edges.push((node_id - 1, node_id));
                }
                let loop_id = node_id;
                node_id += 1;
                current_block_start = i + 1;
                edges.push((node_id, loop_id)); // Loop back
            }
        }
        
        if current_block_start < lines.len() {
            nodes.push(CFGNode {
                id: node_id,
                line_start: current_block_start,
                line_end: lines.len() - 1,
                node_type: CFGNodeType::BasicBlock,
            });
            if node_id > 0 {
                edges.push((node_id - 1, node_id));
            }
        }
        
        if entry_points.is_empty() {
            entry_points.push(0);
        }
        
        Ok(ControlFlowGraph {
            nodes,
            edges,
            entry_points,
        })
    }

    /// Convert CFG to adjacency matrix
    fn cfg_to_adjacency_matrix(&self, cfg: &ControlFlowGraph) -> Vec<Vec<f32>> {
        let n = cfg.nodes.len();
        let mut matrix = vec![vec![0.0f32; n]; n];
        
        // Build adjacency matrix from edges
        for &(from, to) in &cfg.edges {
            if (from as usize) < n && (to as usize) < n {
                matrix[from as usize][to as usize] = 1.0;
                matrix[to as usize][from as usize] = 1.0; // Undirected for topology
            }
        }
        
        matrix
    }

    /// Convert adjacency matrix to distance matrix (shortest path distances)
    fn adjacency_to_distances(&self, adjacency: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n = adjacency.len();
        let mut distances = vec![vec![f32::INFINITY; n]; n];
        
        // Initialize: direct edges have distance 1, self-loops have distance 0
        for i in 0..n {
            distances[i][i] = 0.0;
            for j in 0..n {
                if adjacency[i][j] > 0.0 {
                    distances[i][j] = 1.0;
                }
            }
        }
        
        // Floyd-Warshall algorithm for shortest paths
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if distances[i][k] != f32::INFINITY && distances[k][j] != f32::INFINITY {
                        let new_dist = distances[i][k] + distances[k][j];
                        if new_dist < distances[i][j] {
                            distances[i][j] = new_dist;
                        }
                    }
                }
            }
        }
        
        // Replace INFINITY with large value for TCSAnalyzer
        for i in 0..n {
            for j in 0..n {
                if distances[i][j] == f32::INFINITY {
                    distances[i][j] = 1000.0; // Large but finite
                }
            }
        }
        
        distances
    }

    /// Compute persistence entropy from persistence features
    fn compute_persistence_entropy_from_features(&self, features: &[tcs_core::PersistentFeature]) -> f64 {
        if features.is_empty() {
            return 0.0;
        }
        
        // Compute total persistence
        let total_persistence: f64 = features.iter()
            .map(|f| (f.death - f.birth) as f64)
            .sum();
        
        if total_persistence <= 0.0 {
            return 0.0;
        }
        
        // Compute entropy: -sum(p * log(p)) where p = persistence / total
        let entropy: f64 = features.iter()
            .map(|f| {
                let persistence = (f.death - f.birth) as f64;
                if persistence > 0.0 {
                    let p = persistence / total_persistence;
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum();
        
        entropy.max(0.0).min(1.0) // Normalize to [0, 1]
    }

    /// Compute nesting depth from CFG structure
    fn compute_nesting_from_cfg(&self, cfg: &ControlFlowGraph) -> u32 {
        // Count nested structures (loops, conditionals within other structures)
        let mut max_nesting = 0u32;
        let mut current_nesting = 0u32;
        
        for node in &cfg.nodes {
            match node.node_type {
                CFGNodeType::Loop | CFGNodeType::Conditional => {
                    current_nesting += 1;
                    max_nesting = max_nesting.max(current_nesting);
                }
                _ => {}
            }
        }
        
        max_nesting
    }

    /// Analyze Python code
    async fn analyze_python(&self, code: &str) -> Result<CodeTopologySignature> {
        // Simple heuristics for Python code
        let function_count = code.matches("def ").count() as u32;
        let class_count = code.matches("class ").count() as u32;
        
        // Estimate cyclomatic complexity from control flow keywords
        let cyclomatic_complexity = self.estimate_cyclomatic_complexity(code);
        
        // Estimate nesting depth
        let nesting_depth = self.estimate_nesting_depth(code);
        
        // Estimate cognitive complexity (simplified)
        // Cognitive complexity is typically 1.2x cyclomatic complexity
        const COGNITIVE_COMPLEXITY_MULTIPLIER: f64 = 1.2;
        let cognitive_complexity = (cyclomatic_complexity as f64 * COGNITIVE_COMPLEXITY_MULTIPLIER) as u32;
        
        // Estimate persistence entropy (simplified: based on code structure)
        let persistence_entropy = self.estimate_persistence_entropy(code, cyclomatic_complexity);

        info!(
            function_count,
            cyclomatic_complexity,
            "Computed Python code topology"
        );

        Ok(CodeTopologySignature {
            cyclomatic_complexity,
            cognitive_complexity,
            function_count: function_count + class_count,
            nesting_depth,
            persistence_entropy,
        })
    }

    /// Analyze TypeScript code
    async fn analyze_typescript(&self, code: &str) -> Result<CodeTopologySignature> {
        // Simple heuristics for TypeScript code
        let function_count = code.matches("function ").count() as u32
            + code.matches("=>").count() as u32 / 2; // Approximate arrow functions
        
        let class_count = code.matches("class ").count() as u32;
        
        // Estimate cyclomatic complexity
        let cyclomatic_complexity = self.estimate_cyclomatic_complexity(code);
        
        // Estimate nesting depth
        let nesting_depth = self.estimate_nesting_depth(code);
        
        // Estimate cognitive complexity
        let cognitive_complexity = (cyclomatic_complexity as f64 * 1.2) as u32;
        
        // Estimate persistence entropy
        let persistence_entropy = self.estimate_persistence_entropy(code, cyclomatic_complexity);

        info!(
            function_count,
            cyclomatic_complexity,
            "Computed TypeScript code topology"
        );

        Ok(CodeTopologySignature {
            cyclomatic_complexity,
            cognitive_complexity,
            function_count: function_count + class_count,
            nesting_depth,
            persistence_entropy,
        })
    }

    /// Estimate cyclomatic complexity from control flow keywords
    fn estimate_cyclomatic_complexity(&self, code: &str) -> u32 {
        let mut complexity = 1; // Base complexity
        
        // Count control flow statements
        complexity += code.matches("if ").count() as u32;
        complexity += code.matches("elif ").count() as u32;
        complexity += code.matches("else:").count() as u32;
        complexity += code.matches("for ").count() as u32;
        complexity += code.matches("while ").count() as u32;
        complexity += code.matches("case ").count() as u32;
        complexity += code.matches("catch ").count() as u32;
        complexity += code.matches("&&").count() as u32;
        complexity += code.matches("||").count() as u32;
        complexity += code.matches("and ").count() as u32;
        complexity += code.matches("or ").count() as u32;
        
        complexity
    }

    /// Estimate nesting depth from indentation/brackets
    fn estimate_nesting_depth(&self, code: &str) -> u32 {
        let mut max_depth = 0;
        let mut current_depth = 0;
        
        for line in code.lines() {
            // Count opening brackets
            current_depth += line.matches('{').count() as u32;
            current_depth += line.matches('[').count() as u32;
            current_depth += line.matches('(').count() as u32;
            
            // Count closing brackets
            current_depth = current_depth.saturating_sub(line.matches('}').count() as u32);
            current_depth = current_depth.saturating_sub(line.matches(']').count() as u32);
            current_depth = current_depth.saturating_sub(line.matches(')').count() as u32);
            
            max_depth = max_depth.max(current_depth);
        }
        
        max_depth
    }

    /// Estimate persistence entropy from code structure
    fn estimate_persistence_entropy(&self, code: &str, cyclomatic_complexity: u32) -> f64 {
        // Simplified persistence entropy estimation
        // Based on code length, complexity, and structure
        let code_length = code.len() as f64;
        let complexity_factor = cyclomatic_complexity as f64;
        
        // Normalize to reasonable range [0, 1]
        let entropy = (complexity_factor / (code_length / 100.0 + 1.0)).ln_1p() / 10.0;
        entropy.min(1.0).max(0.0)
    }
}

impl Default for CodeTopologyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CodeLanguage;

    #[tokio::test]
    async fn test_code_topology_analyzer_creation() {
        let analyzer = CodeTopologyAnalyzer::new();
        // Should not panic
    }

    #[tokio::test]
    async fn test_analyze_simple_python_code() {
        let analyzer = CodeTopologyAnalyzer::new();
        let code = r#"
def add(a, b):
    return a + b
"#;

        let result = analyzer.analyze(code, CodeLanguage::Python).await;
        assert!(result.is_ok());
        
        let topology = result.unwrap();
        assert!(topology.function_count >= 1);
        assert!(topology.cyclomatic_complexity >= 1);
    }

    #[tokio::test]
    async fn test_analyze_simple_typescript_code() {
        let analyzer = CodeTopologyAnalyzer::new();
        let code = r#"
function add(a: number, b: number): number {
    return a + b;
}
"#;

        let result = analyzer.analyze(code, CodeLanguage::TypeScript).await;
        assert!(result.is_ok());
        
        let topology = result.unwrap();
        assert!(topology.function_count >= 1);
    }

    #[tokio::test]
    async fn test_topology_signature_conversion() {
        let analyzer = CodeTopologyAnalyzer::new();
        let code = r#"
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"#;

        let result = analyzer.analyze(code, CodeLanguage::Python).await;
        assert!(result.is_ok());
        
        let topology = result.unwrap();
        let sig = topology.to_topological_signature();
        
        // Should have Betti numbers
        assert!(sig.betti_numbers[0] >= 1); // At least one component
    }
}

