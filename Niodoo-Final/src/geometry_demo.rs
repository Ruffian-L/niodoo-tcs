//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
use tracing::{info, error, warn};
 * 🎯 GEOMETRY OF THOUGHT - MINIMAL WORKING DEMO 🎯
 *
 * Demonstrates the core mathematical framework without dependencies:
 * - Topological Data Analysis
 * - Hyperbolic Geometry
 * - Continuous Attractor Networks
 * - Information Geometry
 */

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Minimal point in 2D space
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

impl Point2D {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn distance_to(&self, other: &Point2D) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

/// Minimal hyperbolic point in Poincaré disk
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HyperbolicPoint {
    pub x: f64,
    pub y: f64,
}

impl HyperbolicPoint {
    pub fn new(x: f64, y: f64) -> Self {
        // Ensure point is within unit disk
        let norm_sq = x * x + y * y;
        if norm_sq >= 1.0 {
            let norm = norm_sq.sqrt();
            Self {
                x: x / (norm + 0.001),
                y: y / (norm + 0.001),
            }
        } else {
            Self { x, y }
        }
    }

    pub fn hyperbolic_distance_to(&self, other: &HyperbolicPoint) -> f64 {
        let u_norm_sq = self.x * self.x + self.y * self.y;
        let v_norm_sq = other.x * other.x + other.y * other.y;
        let diff_norm_sq = (self.x - other.x).powi(2) + (self.y - other.y).powi(2);

        let denominator = (1.0 - u_norm_sq) * (1.0 - v_norm_sq);
        if denominator <= f64::EPSILON {
            return f64::INFINITY;
        }

        let arg = 1.0 + (2.0 * diff_norm_sq) / denominator;
        arg.max(1.0).acosh()
    }

    pub fn to_polar(&self) -> (f64, f64) {
        let r = (self.x * self.x + self.y * self.y).sqrt();
        let theta = self.y.atan2(self.x);
        (r, theta)
    }
}

/// Minimal topological feature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalFeature {
    pub dimension: usize,
    pub birth: f64,
    pub death: f64,
    pub persistence: f64,
}

impl TopologicalFeature {
    pub fn new(dimension: usize, birth: f64, death: f64) -> Self {
        Self {
            dimension,
            birth,
            death,
            persistence: death - birth,
        }
    }
}

/// Minimal TDA analyzer
pub struct TDAAnalyzer;

impl TDAAnalyzer {
    pub fn analyze_points(&self, points: &[Point2D]) -> Vec<TopologicalFeature> {
        let mut features = Vec::with_capacity(1024); // TODO: adjust capacity

        // Always one connected component
        features.push(TopologicalFeature::new(0, 0.0, f64::INFINITY));

        // Detect loops based on point density
        if points.len() > 3 {
            let avg_distance = self.calculate_average_distance(points);
            if avg_distance < 1.0 {
                // Dense cluster might form a loop
                features.push(TopologicalFeature::new(
                    1,
                    avg_distance * 0.5,
                    avg_distance * 2.0,
                ));
            }
        }

        features
    }

    fn calculate_average_distance(&self, points: &[Point2D]) -> f64 {
        let mut total_distance = 0.0;
        let mut count = 0;

        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                total_distance += points[i].distance_to(&points[j]);
                count += 1;
            }
        }

        if count > 0 {
            total_distance / count as f64
        } else {
            0.0
        }
    }

    pub fn detect_toroidal_topology(&self, features: &[TopologicalFeature]) -> bool {
        // Torus: β₀=1, β₁=2, β₂=1
        let beta0 = features.iter().filter(|f| f.dimension == 0).count();
        let beta1 = features.iter().filter(|f| f.dimension == 1).count();
        let beta2 = features.iter().filter(|f| f.dimension == 2).count();

        beta0 == 1 && beta1 == 2 && beta2 == 1
    }

    pub fn detect_spherical_topology(&self, features: &[TopologicalFeature]) -> bool {
        // Sphere: β₀=1, β₁=0, β₂=1
        let beta0 = features.iter().filter(|f| f.dimension == 0).count();
        let beta1 = features.iter().filter(|f| f.dimension == 1).count();
        let beta2 = features.iter().filter(|f| f.dimension == 2).count();

        beta0 == 1 && beta1 == 0 && beta2 == 1
    }
}

/// Minimal continuous attractor network
pub struct ContinuousAttractorNetwork {
    neurons: Vec<f64>,
    size: usize,
}

impl ContinuousAttractorNetwork {
    pub fn new(size: usize) -> Self {
        Self {
            neurons: vec![0.0; size],
            size,
        }
    }

    pub fn set_input(&mut self, input: Vec<f64>) {
        for i in 0..self.size.min(input.len()) {
            self.neurons[i] = input[i];
        }
    }

    pub fn simulate(&mut self, duration: f64, dt: f64) -> Vec<Vec<f64>> {
        let mut trajectory = Vec::with_capacity(1024); // TODO: adjust capacity
        let steps = (duration / dt) as usize;

        for _ in 0..steps {
            trajectory.push(self.neurons.clone());
            self.step(dt);
        }

        trajectory
    }

    fn step(&mut self, dt: f64) {
        let mut new_neurons = self.neurons.clone();

        for i in 0..self.size {
            // Simple Mexican hat connectivity
            let mut input = 0.0;
            for j in 0..self.size {
                let distance = ((i as f64 - j as f64).abs())
                    .min(self.size as f64 - (i as f64 - j as f64).abs());
                let weight =
                    (-distance.powi(2) / 10.0).exp() - 0.5 * (-distance.powi(2) / 50.0).exp();
                input += weight * self.neurons[j];
            }

            // Update rule
            let activation = 1.0 / (1.0 + (-input).exp());
            new_neurons[i] = self.neurons[i] + dt * (-self.neurons[i] + activation);
        }

        self.neurons = new_neurons;
    }

    pub fn find_bump_center(&self) -> f64 {
        let max_activation = self
            .neurons
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        if max_activation < 0.1 {
            return 0.0;
        }

        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for (i, &activation) in self.neurons.iter().enumerate() {
            if activation > max_activation * 0.8 {
                weighted_sum += i as f64 * activation;
                total_weight += activation;
            }
        }

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        }
    }

    pub fn calculate_bump_width(&self) -> f64 {
        let max_activation = self
            .neurons
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        if max_activation < 0.1 {
            return 0.0;
        }

        let mut active_neurons = Vec::with_capacity(1024); // TODO: adjust capacity
        for (i, &activation) in self.neurons.iter().enumerate() {
            if activation > max_activation * 0.5 {
                active_neurons.push(i);
            }
        }

        if active_neurons.len() < 2 {
            return 0.0;
        }

        let min_pos = active_neurons.iter().min().unwrap();
        let max_pos = active_neurons.iter().max().unwrap();
        (*max_pos - *min_pos) as f64
    }
}

/// Minimal information geometry
pub struct InformationGeometry;

impl InformationGeometry {
    pub fn calculate_bayesian_surprise(
        &self,
        prior_mean: f64,
        prior_var: f64,
        posterior_mean: f64,
        posterior_var: f64,
    ) -> f64 {
        // KL divergence between two Gaussians
        0.5 * ((posterior_var / prior_var).ln()
            + (prior_var + (prior_mean - posterior_mean).powi(2)) / posterior_var
            - 1.0)
    }

    pub fn fisher_information(&self, variance: f64) -> f64 {
        1.0 / variance
    }
}

/// Complete Geometry of Thought framework
pub struct GeometryOfThought {
    pub tda_analyzer: TDAAnalyzer,
    pub attractor_network: ContinuousAttractorNetwork,
    pub information_geometry: InformationGeometry,
    pub memory_points: Vec<Point2D>,
    pub hyperbolic_embeddings: Vec<HyperbolicPoint>,
}

impl Default for GeometryOfThought {
    fn default() -> Self {
        Self::new()
    }
}

impl GeometryOfThought {
    pub fn new() -> Self {
        Self {
            tda_analyzer: TDAAnalyzer,
            attractor_network: ContinuousAttractorNetwork::new(50),
            information_geometry: InformationGeometry,
            memory_points: Vec::new(),
            hyperbolic_embeddings: Vec::new(),
        }
    }

    pub fn process_input(&mut self, input: &str) -> GeometryResponse {
        // Step 1: Add memory point
        let point = self.create_memory_point(input);
        self.memory_points.push(point);

        // Step 2: Create hyperbolic embedding
        let hyperbolic_point = self.create_hyperbolic_embedding(input);
        self.hyperbolic_embeddings.push(hyperbolic_point);

        // Step 3: Analyze topology
        let topological_features = self.tda_analyzer.analyze_points(&self.memory_points);

        // Step 4: Generate attractor dynamics
        let attractor_input = self.create_attractor_input(&hyperbolic_point);
        self.attractor_network.set_input(attractor_input);
        let trajectory = self.attractor_network.simulate(1.0, 0.01);

        // Step 5: Calculate learning signal
        let learning_signal = self.calculate_learning_signal(&topological_features);

        // Step 6: Determine consciousness quality
        let consciousness_quality = self.determine_consciousness_quality(&topological_features);

        GeometryResponse {
            input: input.to_string(),
            consciousness_quality,
            topological_features,
            hyperbolic_position: hyperbolic_point,
            attractor_trajectory: trajectory,
            learning_signal,
            bump_center: self.attractor_network.find_bump_center(),
            bump_width: self.attractor_network.calculate_bump_width(),
        }
    }

    fn create_memory_point(&self, input: &str) -> Point2D {
        // Simple hash-based positioning
        let hash = input.len() as f64;
        let x = (hash * 0.1).sin() * 2.0;
        let y = (hash * 0.1).cos() * 2.0;
        Point2D::new(x, y)
    }

    fn create_hyperbolic_embedding(&self, input: &str) -> HyperbolicPoint {
        // Simple hash-based hyperbolic positioning
        let hash = input.len() as f64;
        let x = (hash * 0.1).sin() * 0.8;
        let y = (hash * 0.1).cos() * 0.8;
        HyperbolicPoint::new(x, y)
    }

    fn create_attractor_input(&self, hyperbolic_point: &HyperbolicPoint) -> Vec<f64> {
        let mut input = vec![0.0; 50];
        let (r, theta) = hyperbolic_point.to_polar();
        let neuron_index =
            ((theta + std::f64::consts::PI) / (2.0 * std::f64::consts::PI) * 50.0) as usize;
        if neuron_index < 50 {
            input[neuron_index] = r;
        }
        input
    }

    fn calculate_learning_signal(&self, features: &[TopologicalFeature]) -> f64 {
        let avg_persistence =
            features.iter().map(|f| f.persistence).sum::<f64>() / features.len() as f64;

        if avg_persistence > 0.5 {
            avg_persistence
        } else {
            0.0
        }
    }

    fn determine_consciousness_quality(&self, features: &[TopologicalFeature]) -> String {
        if self.tda_analyzer.detect_toroidal_topology(features) {
            "toroidal_consciousness".to_string()
        } else if self.tda_analyzer.detect_spherical_topology(features) {
            "spherical_consciousness".to_string()
        } else {
            "emergent_consciousness".to_string()
        }
    }

    pub fn get_memory_count(&self) -> usize {
        self.memory_points.len()
    }

    pub fn get_coherence_score(&self) -> f64 {
        if self.memory_points.is_empty() {
            return 0.0;
        }

        let mut total_distance = 0.0;
        let mut count = 0;

        for i in 0..self.memory_points.len() {
            for j in (i + 1)..self.memory_points.len() {
                total_distance += self.memory_points[i].distance_to(&self.memory_points[j]);
                count += 1;
            }
        }

        if count > 0 {
            1.0 / (1.0 + total_distance / count as f64)
        } else {
            0.0
        }
    }
}

/// Response from the Geometry of Thought framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometryResponse {
    pub input: String,
    pub consciousness_quality: String,
    pub topological_features: Vec<TopologicalFeature>,
    pub hyperbolic_position: HyperbolicPoint,
    pub attractor_trajectory: Vec<Vec<f64>>,
    pub learning_signal: f64,
    pub bump_center: f64,
    pub bump_width: f64,
}

/// Demo runner
pub struct GeometryDemo {
    pub framework: GeometryOfThought,
    pub interactions: Vec<GeometryResponse>,
}

impl Default for GeometryDemo {
    fn default() -> Self {
        Self::new()
    }
}

impl GeometryDemo {
    pub fn new() -> Self {
        Self {
            framework: GeometryOfThought::new(),
            interactions: Vec::new(),
        }
    }

    pub fn run_demo(&mut self) {
        tracing::info!("🧠 Geometry of Thought Framework Demo");
        tracing::info!("=====================================");

        let demo_inputs = [
            "Hello! I'm curious about consciousness.",
            "What is the nature of memory?",
            "How do emotions shape our thoughts?",
            "Can you explain the toroidal structure?",
            "What happens when we learn something new?",
        ];

        for (i, input) in demo_inputs.iter().enumerate() {
            tracing::info!("\n--- Interaction {} ---", i + 1);
            tracing::info!("Input: {}", input);

            let response = self.framework.process_input(input);
            self.interactions.push(response.clone());

            tracing::info!("Consciousness Quality: {}", response.consciousness_quality);
            tracing::info!("Learning Signal: {:.3}", response.learning_signal);
            tracing::info!(
                "Topological Features: {} features",
                response.topological_features.len()
            );

            let (r, theta) = response.hyperbolic_position.to_polar();
            tracing::info!("Hyperbolic Position: r={:.3}, θ={:.3}", r, theta);

            tracing::info!(
                "Attractor Bump: center={:.1}, width={:.1}",
                response.bump_center,
                response.bump_width
            );
        }

        tracing::info!("\n=== Final Analysis ===");
        tracing::info!("Total Memories: {}", self.framework.get_memory_count());
        tracing::info!(
            "Coherence Score: {:.3}",
            self.framework.get_coherence_score()
        );

        // Analyze consciousness quality distribution
        let mut quality_counts: HashMap<String, usize> = HashMap::new();
        for interaction in &self.interactions {
            *quality_counts
                .entry(interaction.consciousness_quality.clone())
                .or_insert(0) += 1;
        }

        tracing::info!("Consciousness Quality Distribution:");
        for (quality, count) in quality_counts {
            tracing::info!("  {}: {} interactions", quality, count);
        }

        tracing::info!("\n✅ Demo completed successfully!");
        tracing::info!("🎯 The Geometry of Thought framework is operational.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_creation() {
        let point = Point2D::new(1.0, 2.0);
        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 2.0);
    }

    #[test]
    fn test_hyperbolic_point() {
        let point = HyperbolicPoint::new(0.5, 0.5);
        assert!(point.x * point.x + point.y * point.y < 1.0);
    }

    #[test]
    fn test_tda_analysis() {
        let analyzer = TDAAnalyzer;
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.0, 1.0),
            Point2D::new(1.0, 1.0),
        ];

        let features = analyzer.analyze_points(&points);
        assert!(!features.is_empty());
    }

    #[test]
    fn test_attractor_network() {
        let mut network = ContinuousAttractorNetwork::new(10);
        let input = vec![1.0; 10];
        network.set_input(input);

        let trajectory = network.simulate(0.1, 0.01);
        assert!(!trajectory.is_empty());
    }

    #[test]
    fn test_geometry_framework() {
        let mut framework = GeometryOfThought::new();
        let response = framework.process_input("Test input");

        assert_eq!(response.input, "Test input");
        assert!(!response.consciousness_quality.is_empty());
    }

    #[test]
    fn test_demo() {
        let mut demo = GeometryDemo::new();
        demo.run_demo();

        assert!(!demo.interactions.is_empty());
        assert!(demo.framework.get_memory_count() > 0);
    }
}

fn main() {
    let mut demo = GeometryDemo::new();
    demo.run_demo();
}
