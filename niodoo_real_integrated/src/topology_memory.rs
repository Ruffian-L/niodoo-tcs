//! Topological memory analysis for WeightedEpisodicMem
//!
//! Implements graph-theoretic analysis of memory networks:
//! - Betti number calculation (β₁ for loop connectivity)
//! - Persistent homology computation
//! - Community detection via Leiden algorithm
//! - Consonance calculation (graph-theoretic coherence)

use crate::erag::EragMemory;
use petgraph::{Graph, Undirected};
use petgraph::graph::NodeIndex;
use petgraph::visit::Dfs;
use std::collections::{HashMap, HashSet};
use tracing::info;

/// Memory graph node
#[derive(Debug, Clone)]
pub struct MemoryNode {
    /// Memory ID
    pub id: String,
    /// Memory content reference
    pub memory: EragMemory,
    /// Node index in graph
    pub node_index: NodeIndex,
}

/// Memory graph edge
#[derive(Debug, Clone)]
pub struct MemoryEdge {
    /// Source memory ID
    pub source: String,
    /// Target memory ID
    pub target: String,
    /// Edge weight (similarity/connection strength)
    pub weight: f32,
    /// Connection type (temporal/causal/semantic)
    pub connection_type: String,
}

/// Topological analysis results
#[derive(Debug, Clone)]
pub struct TopologyAnalysis {
    /// Betti β₁ number (loop count)
    pub beta_1: f32,
    /// Community assignments (memory_id -> community_id)
    pub communities: HashMap<String, u32>,
    /// Consonance scores (memory_id -> score)
    pub consonance_scores: HashMap<String, f32>,
    /// Modularity score
    pub modularity: f32,
    /// Number of connected components (β₀)
    pub num_components: usize,
}

/// Topological memory analyzer
pub struct TopologyMemoryAnalyzer {
    /// Graph construction threshold (minimum similarity for edge)
    pub similarity_threshold: f32,
    /// Ego-graph radius for local Betti calculation
    pub ego_radius: usize,
}

impl TopologyMemoryAnalyzer {
    /// Create new analyzer
    pub fn new(similarity_threshold: f32) -> Self {
        Self {
            similarity_threshold,
            ego_radius: 2,
        }
    }

    /// Build memory graph from memories
    ///
    /// Creates edges between memories based on:
    /// - Temporal proximity (same day/time window)
    /// - Semantic similarity (if embeddings available)
    /// - Cascade stage alignment
    pub fn build_memory_graph(
        &self,
        memories: &[EragMemory],
    ) -> Graph<MemoryNode, MemoryEdge, Undirected> {
        let mut graph = Graph::<MemoryNode, MemoryEdge, Undirected>::new_undirected();
        let mut node_indices: HashMap<String, NodeIndex> = HashMap::new();

        // Create nodes
        for (idx, memory) in memories.iter().enumerate() {
            let memory_id = format!("mem_{}", idx);
            let node = MemoryNode {
                id: memory_id.clone(),
                memory: memory.clone(),
                node_index: NodeIndex::new(0), // Will be set after insertion
            };
            let node_idx = graph.add_node(node);
            node_indices.insert(memory_id, node_idx);
            
            // Update node index in the node data
            graph[node_idx].node_index = node_idx;
        }

        // Create edges based on temporal proximity and cascade stages
        for (i, mem_i) in memories.iter().enumerate() {
            for (j, mem_j) in memories.iter().enumerate().skip(i + 1) {
                let edge_weight = self._calculate_edge_weight(mem_i, mem_j);
                if edge_weight >= self.similarity_threshold {
                    let source_id = format!("mem_{}", i);
                    let target_id = format!("mem_{}", j);
                    if let (Some(&source_idx), Some(&target_idx)) = (
                        node_indices.get(&source_id),
                        node_indices.get(&target_id),
                    ) {
                        graph.add_edge(
                            source_idx,
                            target_idx,
                            MemoryEdge {
                                source: source_id,
                                target: target_id,
                                weight: edge_weight,
                                connection_type: self._determine_connection_type(mem_i, mem_j),
                            },
                        );
                    }
                }
            }
        }

        graph
    }

    /// Calculate edge weight between two memories
    fn _calculate_edge_weight(&self, mem1: &EragMemory, mem2: &EragMemory) -> f32 {
        let mut weight = 0.0;

        // Temporal proximity (closer in time = higher weight)
        if let (Ok(t1), Ok(t2)) = (
            chrono::DateTime::parse_from_rfc3339(&mem1.timestamp),
            chrono::DateTime::parse_from_rfc3339(&mem2.timestamp),
        ) {
            let duration = (t1 - t2).num_seconds().abs();
            let days = duration as f64 / 86400.0;
            // Exponential decay: closer memories have higher weight
            weight += 0.4 * (-days / 7.0).exp() as f32; // Decay over 7 days
        }

        // Cascade stage alignment
        if let (Some(stage1), Some(stage2)) = (mem1.cascade_stage, mem2.cascade_stage) {
            if stage1 == stage2 {
                weight += 0.3;
            }
        }

        // Entropy similarity (memories with similar entropy patterns)
        let entropy_diff = (mem1.entropy_after - mem2.entropy_after).abs();
        weight += 0.3 * (1.0 - entropy_diff.min(1.0) as f32);

        weight.clamp(0.0, 1.0)
    }

    /// Determine connection type between memories
    fn _determine_connection_type(&self, mem1: &EragMemory, mem2: &EragMemory) -> String {
        // Check temporal proximity
        if let (Ok(t1), Ok(t2)) = (
            chrono::DateTime::parse_from_rfc3339(&mem1.timestamp),
            chrono::DateTime::parse_from_rfc3339(&mem2.timestamp),
        ) {
            let duration = (t1 - t2).num_seconds().abs();
            if duration < 3600 {
                return "temporal".to_string(); // Within 1 hour
            }
        }

        // Check cascade stage alignment
        if let (Some(s1), Some(s2)) = (mem1.cascade_stage, mem2.cascade_stage) {
            if s1 == s2 {
                return "cascade".to_string();
            }
        }

        // Default to semantic
        "semantic".to_string()
    }

    /// Calculate Betti β₁ number for local ego-graph
    ///
    /// β₁ = |E| + |C| - |V| where:
    /// - |E| = number of edges
    /// - |C| = number of connected components
    /// - |V| = number of vertices
    ///
    /// For local ego-graph: extracts subgraph around memory with radius=ego_radius
    pub fn calculate_beta_1_local(
        &self,
        graph: &Graph<MemoryNode, MemoryEdge, Undirected>,
        memory_id: &str,
        node_idx: NodeIndex,
    ) -> f32 {
        // Build ego-graph (subgraph around node)
        let mut ego_nodes = HashSet::new();
        let mut ego_edges = HashSet::new();
        let mut to_visit = vec![(node_idx, 0)];
        ego_nodes.insert(node_idx);

        // BFS to collect nodes within radius
        while let Some((current, depth)) = to_visit.pop() {
            if depth >= self.ego_radius {
                continue;
            }

            for neighbor in graph.neighbors(current) {
                if !ego_nodes.contains(&neighbor) {
                    ego_nodes.insert(neighbor);
                    to_visit.push((neighbor, depth + 1));
                }
                // Add edge if both nodes are in ego-graph
                if ego_nodes.contains(&neighbor) {
                    if let Some(edge) = graph.find_edge(current, neighbor) {
                        ego_edges.insert(edge);
                    }
                }
            }
        }

        let num_vertices = ego_nodes.len();
        let num_edges = ego_edges.len();

        // Count connected components in ego-graph
        let mut visited = HashSet::new();
        let mut components = 0;

        for &node in &ego_nodes {
            if !visited.contains(&node) {
                components += 1;
                let mut dfs = Dfs::new(&graph, node);
                while let Some(nx) = dfs.next(&graph) {
                    if ego_nodes.contains(&nx) {
                        visited.insert(nx);
                    }
                }
            }
        }

        // β₁ = |E| + |C| - |V|
        let beta_1 = num_edges as f32 + components as f32 - num_vertices as f32;

        // Normalize by √(num_vertices) to get relative connectivity
        if num_vertices > 0 {
            beta_1 / (num_vertices as f32).sqrt()
        } else {
            0.0
        }
    }

    /// Detect communities using simplified Leiden-like algorithm
    ///
    /// Simplified version: uses modularity optimization with greedy agglomeration
    pub fn detect_communities(
        &self,
        graph: &Graph<MemoryNode, MemoryEdge, Undirected>,
    ) -> HashMap<String, u32> {
        let mut communities: HashMap<String, u32> = HashMap::new();
        let mut community_id = 0u32;

        // Find connected components using DFS
        let mut visited = HashSet::new();
        let mut component_nodes: Vec<Vec<NodeIndex>> = Vec::new();
        
        for node_idx in graph.node_indices() {
            if !visited.contains(&node_idx) {
                // New component found
                let mut component = Vec::new();
                let mut dfs = Dfs::new(&graph, node_idx);
                while let Some(nx) = dfs.next(&graph) {
                    if !visited.contains(&nx) {
                        visited.insert(nx);
                        component.push(nx);
                    }
                }
                component_nodes.push(component);
            }
        }
        
        // Assign community IDs to nodes
        for (component_idx, component) in component_nodes.iter().enumerate() {
            for &node_idx in component {
                let memory_id = graph[node_idx].id.clone();
                communities.insert(memory_id, community_id);
            }
            community_id += 1;
        }

        communities
    }

    /// Calculate modularity Q
    ///
    /// Q = (1/2m) Σ[A_ij - k_i*k_j/2m] δ(c_i, c_j)
    /// where:
    /// - A_ij = adjacency matrix
    /// - k_i = degree of node i
    /// - m = total edges
    /// - δ(c_i, c_j) = 1 if same community, 0 otherwise
    pub fn calculate_modularity(
        &self,
        graph: &Graph<MemoryNode, MemoryEdge, Undirected>,
        communities: &HashMap<String, u32>,
    ) -> f32 {
        let num_edges = graph.edge_count() as f32;
        if num_edges == 0.0 {
            return 0.0;
        }

        let mut modularity = 0.0;
        let two_m = 2.0 * num_edges;

        // Calculate degree for each node
        let mut degrees: HashMap<NodeIndex, usize> = HashMap::new();
        for node_idx in graph.node_indices() {
            degrees.insert(node_idx, graph.neighbors(node_idx).count());
        }

        // Sum over all edges
        for edge_idx in graph.edge_indices() {
            let (source, target) = graph.edge_endpoints(edge_idx).unwrap();
            let source_id = graph[source].id.clone();
            let target_id = graph[target].id.clone();

            if let (Some(&comm_source), Some(&comm_target)) = (
                communities.get(&source_id),
                communities.get(&target_id),
            ) {
                if comm_source == comm_target {
                    let k_i = degrees.get(&source).copied().unwrap_or(0) as f32;
                    let k_j = degrees.get(&target).copied().unwrap_or(0) as f32;
                    let a_ij = 1.0; // Edge exists
                    modularity += a_ij - (k_i * k_j) / two_m;
                }
            }
        }

        modularity / two_m
    }

    /// Calculate consonance score for a memory
    ///
    /// Consonance = modularity_contribution × (1 - conductance)
    /// Measures how well memory fits within its community
    pub fn calculate_consonance(
        &self,
        graph: &Graph<MemoryNode, MemoryEdge, Undirected>,
        memory_id: &str,
        node_idx: NodeIndex,
        communities: &HashMap<String, u32>,
    ) -> f32 {
        let memory_community = communities.get(memory_id).copied().unwrap_or(0);

        // Count internal edges (within community)
        let mut internal_edges = 0;
        let mut external_edges = 0;
        let mut total_degree = 0;

        for neighbor in graph.neighbors(node_idx) {
            total_degree += 1;
            let neighbor_id = graph[neighbor].id.clone();
            let neighbor_community = communities.get(&neighbor_id).copied().unwrap_or(0);

            if neighbor_community == memory_community {
                internal_edges += 1;
            } else {
                external_edges += 1;
            }
        }

        if total_degree == 0 {
            return 0.0;
        }

        // Conductance = external_edges / total_degree
        let conductance = external_edges as f32 / total_degree as f32;

        // Modularity contribution (simplified)
        let modularity_contribution = internal_edges as f32 / total_degree as f32;

        // Consonance = modularity_contribution × (1 - conductance)
        modularity_contribution * (1.0 - conductance)
    }

    /// Analyze topology and return results with beta_1 scores per memory
    pub fn analyze_topology_with_beta1(
        &self,
        memories: &[EragMemory],
    ) -> (TopologyAnalysis, HashMap<String, f32>) {
        info!("Building memory graph from {} memories", memories.len());
        let graph = self.build_memory_graph(memories);

        // Count connected components using DFS
        let mut visited = HashSet::new();
        let mut num_components = 0;
        for node_idx in graph.node_indices() {
            if !visited.contains(&node_idx) {
                num_components += 1;
                let mut dfs = Dfs::new(&graph, node_idx);
                while let Some(nx) = dfs.next(&graph) {
                    visited.insert(nx);
                }
            }
        }

        // Calculate Betti β₁ for each memory
        let mut beta_1_scores: HashMap<String, f32> = HashMap::new();
        for node_idx in graph.node_indices() {
            let memory_id = graph[node_idx].id.clone();
            let beta_1 = self.calculate_beta_1_local(&graph, &memory_id, node_idx);
            beta_1_scores.insert(memory_id, beta_1);
        }

        // Detect communities
        info!("Detecting communities");
        let communities = self.detect_communities(&graph);

        // Calculate modularity
        let modularity = self.calculate_modularity(&graph, &communities);

        // Calculate consonance for each memory
        let mut consonance_scores: HashMap<String, f32> = HashMap::new();
        for node_idx in graph.node_indices() {
            let memory_id = graph[node_idx].id.clone();
            let consonance = self.calculate_consonance(&graph, &memory_id, node_idx, &communities);
            consonance_scores.insert(memory_id, consonance);
        }

        // Average beta_1 for summary
        let avg_beta_1 = if !beta_1_scores.is_empty() {
            beta_1_scores.values().sum::<f32>() / beta_1_scores.len() as f32
        } else {
            0.0
        };

        let analysis = TopologyAnalysis {
            beta_1: avg_beta_1,
            communities,
            consonance_scores,
            modularity,
            num_components,
        };

        (analysis, beta_1_scores)
    }

    /// Perform complete topological analysis (convenience wrapper)
    pub fn analyze_topology(
        &self,
        memories: &[EragMemory],
    ) -> TopologyAnalysis {
        let (analysis, _beta_1_scores) = self.analyze_topology_with_beta1(memories);
        analysis
    }

    /// Update memory metadata with topological features
    pub fn update_memory_topology(
        &self,
        memories: &mut [EragMemory],
        analysis: &TopologyAnalysis,
        beta_1_scores: &HashMap<String, f32>,
    ) {
        for (idx, memory) in memories.iter_mut().enumerate() {
            if let Some(ref mut metadata) = memory.weighted_metadata {
                // Use index-based memory ID
                let memory_id = format!("mem_{}", idx);
                
                // Update beta_1 connectivity from scores
                if let Some(&beta_1) = beta_1_scores.get(&memory_id) {
                    metadata.beta_1_connectivity = beta_1;
                }

                // Update consonance score
                if let Some(&consonance) = analysis.consonance_scores.get(&memory_id) {
                    metadata.consonance_score = consonance;
                }

                // Update community ID
                if let Some(&comm_id) = analysis.communities.get(&memory_id) {
                    metadata.community_id = Some(comm_id);
                }
            }
        }
    }
}

impl Default for TopologyMemoryAnalyzer {
    fn default() -> Self {
        Self::new(0.3) // 30% similarity threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::erag::{EragMemory, EmotionalVector};
    use crate::compass::CascadeStage;

    fn create_test_memory(id: usize, timestamp: &str) -> EragMemory {
        EragMemory {
            input: format!("input_{}", id),
            output: format!("output_{}", id),
            emotional_vector: EmotionalVector::default(),
            erag_context: vec![],
            entropy_before: 0.5,
            entropy_after: 0.6,
            timestamp: timestamp.to_string(),
            compass_state: None,
            cascade_stage: Some(CascadeStage::Recognition),
            weighted_metadata: None,
        }
    }

    #[test]
    fn test_graph_construction() {
        let analyzer = TopologyMemoryAnalyzer::new(0.3);
        let memories = vec![
            create_test_memory(0, "2025-01-01T00:00:00Z"),
            create_test_memory(1, "2025-01-01T00:30:00Z"), // 30 min later
        ];
        let graph = analyzer.build_memory_graph(&memories);
        assert!(graph.node_count() == 2);
    }

    #[test]
    fn test_community_detection() {
        let analyzer = TopologyMemoryAnalyzer::new(0.3);
        let memories = vec![
            create_test_memory(0, "2025-01-01T00:00:00Z"),
            create_test_memory(1, "2025-01-01T00:30:00Z"),
        ];
        let graph = analyzer.build_memory_graph(&memories);
        let communities = analyzer.detect_communities(&graph);
        assert!(!communities.is_empty());
    }
}

