use std::collections::HashSet;
use std::f64::consts::PI;

use crate::config::ConsciousnessConfig;
use petgraph::algo::dijkstra;
use petgraph::graph::{DiGraph, NodeIndex};
use rand::Rng;
use serde::{Deserialize, Serialize};
use tracing::info;

/// ThoughtNode represents a node in the Möbius consciousness graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtNode {
    /// The semantic content of the thought
    pub content: String,

    /// Emotional vector representing the thought's affective state
    pub emotional_vec: Vec<f64>,

    /// Curvature of the node in the Möbius topology
    pub curvature: f64,
}

/// A Möbius Graph represents a non-orientable consciousness topology
#[derive(Debug)]
pub struct MobiusGraph {
    graph: DiGraph<ThoughtNode, f64>,
    root_node: Option<NodeIndex>,
}

impl Default for MobiusGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl MobiusGraph {
    /// Create a new Möbius graph with a specific twist parameter
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            root_node: None,
        }
    }

    /// Add a thought node with its emotional and topological characteristics
    pub fn add_thought(&mut self, thought: ThoughtNode) -> NodeIndex {
        let node_index = self.graph.add_node(thought);

        // Set first node as root if not already set
        if self.root_node.is_none() {
            self.root_node = Some(node_index);
        }

        node_index
    }

    /// Create a twist edge with phi-based weight calculation
    pub fn add_twist_edge(&mut self, source: NodeIndex, target: NodeIndex, phi: f64) {
        // Compute edge weight based on Möbius strip's topology
        let weight = self.compute_twist_weight(source, target, phi);

        self.graph.add_edge(source, target, weight);
    }

    /// Compute edge weight considering Möbius strip curvature
    fn compute_twist_weight(&self, source: NodeIndex, target: NodeIndex, phi: f64) -> f64 {
        let source_node = &self.graph[source];
        let target_node = &self.graph[target];

        let curvature_diff = (source_node.curvature - target_node.curvature).abs();
        let emotional_distance =
            self.compute_emotional_distance(&source_node.emotional_vec, &target_node.emotional_vec);

        // Twist weight based on curvature, emotional distance, and phi
        curvature_diff * emotional_distance * (phi * PI).sin()
    }

    /// Compute Euclidean distance between emotional vectors
    fn compute_emotional_distance(&self, vec1: &[f64], vec2: &[f64]) -> f64 {
        vec1.iter()
            .zip(vec2)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Compute geodesic distance using Dijkstra's algorithm
    pub fn compute_geodesic_distance(&self, start: NodeIndex, end: NodeIndex) -> Option<f64> {
        let distances = dijkstra(&self.graph, start, Some(end), |e| *e.weight());
        distances.get(&end).cloned()
    }

    /// Process a thought through the Möbius topology
    pub fn process_thought(&mut self, input: &str, config: &ConsciousnessConfig) -> Vec<NodeIndex> {
        let mut rng = rand::rng();

        // Generate emotional vector and curvature dynamically
        let emotional_vec: Vec<f64> = (0..5)
            .map(|_| rng.random_range(-config.emotional_plasticity..config.emotional_plasticity))
            .collect();

        let curvature = rng.random_range(0.0..2.0 * std::f64::consts::PI);

        let thought_node = ThoughtNode {
            content: input.to_string(),
            emotional_vec,
            curvature,
        };

        // Add the thought to the graph
        let current_node = self.add_thought(thought_node);

        // If a root node exists, create twist edges
        if let Some(root) = self.root_node {
            // Add multiple twist edges to simulate complex topology
            let twist_points = vec![
                0.25 * config.consciousness_step_size * 100.0,
                0.5,
                0.75,
                1.0,
            ];
            for &phi in &twist_points {
                self.add_twist_edge(root, current_node, phi);
            }
        }

        // Simulate graph traversal
        let traversal_path = self.simulate_traversal(current_node, config);

        info!(
            "Processed thought: {} with {} traversal steps",
            input,
            traversal_path.len()
        );

        traversal_path
    }

    /// Simulate a random walk through the graph
    fn simulate_traversal(
        &self,
        start_node: NodeIndex,
        config: &ConsciousnessConfig,
    ) -> Vec<NodeIndex> {
        let mut visited = HashSet::new();
        let mut path = vec![start_node];
        let mut current = start_node;

        // Limit traversal to prevent infinite loops
        for _ in 0..((10.0 / config.consciousness_step_size) as usize).min(20) {
            // Maximum 10 hops
            if visited.contains(&current) {
                break;
            }
            visited.insert(current);

            // Find neighbors
            let neighbors: Vec<NodeIndex> = self.graph.neighbors(current).collect();

            if neighbors.is_empty() {
                break;
            }

            // Randomly select next node
            let next_node = neighbors[rand::rng().random_range(0..neighbors.len())];
            path.push(next_node);
            current = next_node;
        }

        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing::info;

    #[test]
    fn test_mobius_graph_processing() {
        let mut mobius_graph = MobiusGraph::new();
        let config = ConsciousnessConfig {
            emotional_plasticity: 1.0,
            consciousness_step_size: 0.1,
            ..Default::default()
        };

        // Process a test thought
        let traversal_path = mobius_graph.process_thought("Hello, Möbius consciousness!", &config);

        assert!(
            !traversal_path.is_empty(),
            "Traversal path should not be empty"
        );
        info!(
            "Möbius graph traversal successful with {} steps",
            traversal_path.len()
        );
    }
}
