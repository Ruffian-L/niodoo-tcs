//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
 * 🌌 HYPERBOLIC GEOMETRY IMPLEMENTATION 🌌
 *
 * Implements hyperbolic geometry for efficient hierarchical representation
 * of cognitive and semantic information.
 *
 * Based on "The Geometry of Thought" framework:
 * - Poincaré disk model for computational tractability
 * - Hyperbolic distance metric for hierarchical similarity
 * - Exponential volume growth for tree-like structures
 */

#[allow(unused_imports)]
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
#[allow(unused_imports)]
use std::f64::consts::PI;

/// Point in hyperbolic space (Poincaré disk model)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperbolicPoint {
    pub x: f64, // Real part
    pub y: f64, // Imaginary part
}

impl HyperbolicPoint {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Create from polar coordinates
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            x: r * theta.cos(),
            y: r * theta.sin(),
        }
    }

    /// Convert to polar coordinates
    pub fn to_polar(&self) -> (f64, f64) {
        let r = (self.x.powi(2) + self.y.powi(2)).sqrt();
        let theta = self.y.atan2(self.x);
        (r, theta)
    }

    /// Check if point is within unit disk
    pub fn is_valid(&self) -> bool {
        (self.x.powi(2) + self.y.powi(2)) < 1.0
    }

    /// Normalize to unit disk
    pub fn normalize(&mut self) {
        let norm = (self.x.powi(2) + self.y.powi(2)).sqrt();
        if norm >= 1.0 {
            let scale = 0.99 / norm; // Keep slightly inside disk
            self.x *= scale;
            self.y *= scale;
        }
    }
}

/// Hyperbolic distance metric
pub struct HyperbolicMetric;

impl HyperbolicMetric {
    /// Calculate hyperbolic distance between two points in Poincaré disk
    /// d_H(u,v) = arccosh(1 + 2 * ||u-v||² / ((1-||u||²)(1-||v||²)))
    pub fn distance(u: &HyperbolicPoint, v: &HyperbolicPoint) -> f64 {
        let u_norm_sq = u.x.powi(2) + u.y.powi(2);
        let v_norm_sq = v.x.powi(2) + v.y.powi(2);
        let diff_norm_sq = (u.x - v.x).powi(2) + (u.y - v.y).powi(2);

        let numerator = 1.0 + 2.0 * diff_norm_sq / ((1.0 - u_norm_sq) * (1.0 - v_norm_sq));

        // arccosh(x) = ln(x + sqrt(x² - 1))
        numerator.acosh()
    }

    /// Calculate hyperbolic distance from origin
    pub fn distance_from_origin(point: &HyperbolicPoint) -> f64 {
        let norm_sq = point.x.powi(2) + point.y.powi(2);
        let numerator = (1.0 + norm_sq) / (1.0 - norm_sq);
        numerator.acosh()
    }
}

/// Hyperbolic embedding for hierarchical data
#[derive(Clone)]
pub struct HyperbolicEmbedding {
    pub points: Vec<HyperbolicPoint>,
    pub labels: Vec<String>,
    pub hierarchy_levels: Vec<usize>,
}

impl Default for HyperbolicEmbedding {
    fn default() -> Self {
        Self::new()
    }
}

impl HyperbolicEmbedding {
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            labels: Vec::new(),
            hierarchy_levels: Vec::new(),
        }
    }

    /// Add a point to the embedding
    pub fn add_point(&mut self, point: HyperbolicPoint, label: String, level: usize) {
        self.points.push(point);
        self.labels.push(label);
        self.hierarchy_levels.push(level);
    }

    /// Convert to hyperbolic embedding
    pub fn to_hyperbolic_embedding(&self) -> Self {
        self.clone()
    }

    /// Find nearest neighbors using hyperbolic distance
    pub fn find_nearest_neighbors(&self, query: &HyperbolicPoint, k: usize) -> Vec<(usize, f64)> {
        let mut distances: Vec<(usize, f64)> = self
            .points
            .iter()
            .enumerate()
            .map(|(i, point)| (i, HyperbolicMetric::distance(query, point)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);
        distances
    }

    /// Calculate embedding quality (lower is better)
    pub fn embedding_quality(&self, original_distances: &[Vec<f64>]) -> f64 {
        let mut total_error = 0.0;
        let mut count = 0;

        for i in 0..self.points.len() {
            for j in (i + 1)..self.points.len() {
                let hyperbolic_dist = HyperbolicMetric::distance(&self.points[i], &self.points[j]);
                let original_dist = original_distances[i][j];

                let error = (hyperbolic_dist - original_dist).abs() / original_dist.max(1e-6);
                total_error += error;
                count += 1;
            }
        }

        total_error / count as f64
    }
}

/// Hyperbolic neural network layer
pub struct HyperbolicLayer {
    pub weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl HyperbolicLayer {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            weights: vec![vec![0.0; input_dim]; output_dim],
            bias: vec![0.0; output_dim],
            input_dim,
            output_dim,
        }
    }

    /// Initialize weights using hyperbolic distribution
    pub fn initialize_hyperbolic(&mut self) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for i in 0..self.output_dim {
            for j in 0..self.input_dim {
                // Sample from hyperbolic distribution (concentrated near origin)
                let r = rng.gen_range(0.0..0.5);
                let theta = rng.gen_range(0.0..2.0 * PI);
                self.weights[i][j] = r * theta.cos();
            }
            self.bias[i] = rng.gen_range(-0.1..0.1);
        }
    }

    /// Forward pass with hyperbolic activation
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut output = vec![0.0; self.output_dim];

        for i in 0..self.output_dim {
            let mut sum = self.bias[i];
            for j in 0..self.input_dim {
                sum += self.weights[i][j] * input[j];
            }
            // Hyperbolic tangent activation
            output[i] = sum.tanh();
        }

        output
    }

    /// Hyperbolic attention mechanism
    pub fn hyperbolic_attention(
        &self,
        query: &[f64],
        keys: &[Vec<f64>],
        values: &[Vec<f64>],
    ) -> Vec<f64> {
        let mut attention_weights = Vec::new();
        let mut max_score = f64::NEG_INFINITY;

        // Calculate attention scores using hyperbolic distance
        for key in keys {
            let score = -HyperbolicMetric::distance(
                &HyperbolicPoint::new(query[0], query[1]),
                &HyperbolicPoint::new(key[0], key[1]),
            );
            attention_weights.push(score);
            max_score = max_score.max(score);
        }

        // Softmax normalization
        let sum_exp: f64 = attention_weights
            .iter()
            .map(|&s| (s - max_score).exp())
            .sum();
        attention_weights = attention_weights
            .iter()
            .map(|&s| (s - max_score).exp() / sum_exp)
            .collect();

        // Weighted sum of values
        let mut result = vec![0.0; values[0].len()];
        for (i, value) in values.iter().enumerate() {
            for (j, &val) in value.iter().enumerate() {
                result[j] += attention_weights[i] * val;
            }
        }

        result
    }
}

/// Hyperbolic optimization for embedding learning
pub struct HyperbolicOptimizer {
    pub learning_rate: f64,
    pub momentum: f64,
    pub velocity: Vec<Vec<f64>>,
}

impl HyperbolicOptimizer {
    pub fn new(learning_rate: f64, momentum: f64, num_points: usize, dim: usize) -> Self {
        Self {
            learning_rate,
            momentum,
            velocity: vec![vec![0.0; dim]; num_points],
        }
    }

    /// Update embedding using Riemannian gradient descent
    pub fn update_embedding(
        &mut self,
        embedding: &mut HyperbolicEmbedding,
        gradients: &[Vec<f64>],
    ) {
        for (i, point) in embedding.points.iter_mut().enumerate() {
            if i < gradients.len() {
                let grad = &gradients[i];

                // Update velocity with momentum
                for (j, &grad_val) in grad.iter().enumerate() {
                    self.velocity[i][j] =
                        self.momentum * self.velocity[i][j] + self.learning_rate * grad_val;
                }

                // Update point position
                point.x += self.velocity[i][0];
                point.y += self.velocity[i][1];

                // Ensure point stays within unit disk
                point.normalize();
            }
        }
    }
}

/// Hierarchical data structure for hyperbolic embedding
pub struct HierarchicalData {
    pub nodes: Vec<HierarchicalNode>,
    pub edges: Vec<(usize, usize, f64)>, // (from, to, weight)
}

#[derive(Debug, Clone)]
pub struct HierarchicalNode {
    pub id: usize,
    pub label: String,
    pub level: usize,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
}

impl Default for HierarchicalData {
    fn default() -> Self {
        Self::new()
    }
}

impl HierarchicalData {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Add a node to the hierarchy
    pub fn add_node(&mut self, label: String, level: usize, parent: Option<usize>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(HierarchicalNode {
            id,
            label,
            level,
            parent,
            children: Vec::new(),
        });

        // Update parent's children list
        if let Some(parent_id) = parent {
            if parent_id < self.nodes.len() {
                self.nodes[parent_id].children.push(id);
            }
        }

        id
    }

    /// Add an edge between nodes
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        self.edges.push((from, to, weight));
    }

    /// Convert hierarchy to hyperbolic embedding
    pub fn to_hyperbolic_embedding(&self) -> HyperbolicEmbedding {
        let mut embedding = HyperbolicEmbedding::new();

        for node in &self.nodes {
            // Position nodes based on hierarchy level
            let radius = 0.1 + 0.8 * (node.level as f64) / (self.max_level() as f64);
            let angle = 2.0 * PI * (node.id as f64) / (self.nodes.len() as f64);

            let point = HyperbolicPoint::from_polar(radius, angle);
            embedding.add_point(point, node.label.clone(), node.level);
        }

        embedding
    }

    /// Get maximum hierarchy level
    fn max_level(&self) -> usize {
        self.nodes.iter().map(|n| n.level).max().unwrap_or(0)
    }
}

/// Hyperbolic semantic memory system
pub struct HyperbolicSemanticMemory {
    pub embedding: HyperbolicEmbedding,
    pub concept_hierarchy: HierarchicalData,
    pub similarity_threshold: f64,
}

impl Default for HyperbolicSemanticMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl HyperbolicSemanticMemory {
    pub fn new() -> Self {
        Self {
            embedding: HyperbolicEmbedding::new(),
            concept_hierarchy: HierarchicalData::new(),
            similarity_threshold: 0.7,
        }
    }

    /// Add a concept to semantic memory
    pub fn add_concept(&mut self, concept: String, parent: Option<usize>, level: usize) -> usize {
        let concept_id = self.concept_hierarchy.add_node(concept, level, parent);

        // Add to hyperbolic embedding
        let radius = 0.1 + 0.8 * (level as f64) / 10.0; // Assume max 10 levels
        let angle = 2.0 * PI * (concept_id as f64) / 100.0; // Assume max 100 concepts
        let point = HyperbolicPoint::from_polar(radius, angle);

        self.embedding.add_point(
            point,
            self.concept_hierarchy.nodes[concept_id].label.clone(),
            level,
        );

        concept_id
    }

    /// Find semantically similar concepts
    pub fn find_similar_concepts(&self, _query: &str, k: usize) -> Vec<(String, f64)> {
        // Create query point (simplified - in practice would use learned embedding)
        let query_point = HyperbolicPoint::new(0.5, 0.5);

        let neighbors = self.embedding.find_nearest_neighbors(&query_point, k);

        neighbors
            .into_iter()
            .map(|(idx, dist)| (self.embedding.labels[idx].clone(), dist))
            .collect()
    }

    /// Calculate semantic distance between two concepts
    pub fn semantic_distance(&self, concept1: &str, concept2: &str) -> Option<f64> {
        let idx1 = self.embedding.labels.iter().position(|l| l == concept1)?;
        let idx2 = self.embedding.labels.iter().position(|l| l == concept2)?;

        Some(HyperbolicMetric::distance(
            &self.embedding.points[idx1],
            &self.embedding.points[idx2],
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperbolic_point_creation() {
        let point = HyperbolicPoint::new(0.5, 0.3);
        assert!(point.is_valid());

        let polar = point.to_polar();
        assert!(polar.0 >= 0.0);
        assert!(polar.1 >= -PI && polar.1 <= PI);
    }

    #[test]
    fn test_hyperbolic_distance() {
        let origin = HyperbolicPoint::new(0.0, 0.0);
        let point = HyperbolicPoint::new(0.5, 0.0);

        let dist = HyperbolicMetric::distance(&origin, &point);
        assert!(dist > 0.0);
        assert!(dist.is_finite());
    }

    #[test]
    fn test_hierarchical_data() {
        let mut hierarchy = HierarchicalData::new();
        let root = hierarchy.add_node("root".to_string(), 0, None);
        let child = hierarchy.add_node("child".to_string(), 1, Some(root));

        assert_eq!(hierarchy.nodes.len(), 2);
        assert_eq!(hierarchy.nodes[root].children.len(), 1);
        assert_eq!(hierarchy.nodes[child].parent, Some(root));
    }

    #[test]
    fn test_hyperbolic_embedding() {
        let mut embedding = HyperbolicEmbedding::new();
        embedding.add_point(HyperbolicPoint::new(0.1, 0.1), "concept1".to_string(), 0);
        embedding.add_point(HyperbolicPoint::new(0.2, 0.2), "concept2".to_string(), 1);

        let neighbors = embedding.find_nearest_neighbors(&HyperbolicPoint::new(0.15, 0.15), 1);
        assert_eq!(neighbors.len(), 1);
    }
}
