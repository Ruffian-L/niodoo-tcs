/*
use tracing::{info, error, warn};
 * 🔧 TEST FIXTURES AND UTILITIES
 *
 * AGENT 9: Shared test fixtures and helper functions
 *
 * This module provides reusable test fixtures for:
 * 1. Consciousness states
 * 2. Memory fragments
 * 3. Gaussian memory spheres
 * 4. Toroidal coordinates
 * 5. Test data generators
 */

#![allow(dead_code)]

use niodoo_consciousness::*;

// ============================================================================
// CONSCIOUSNESS STATE FIXTURES
// ============================================================================

/// Create a default test consciousness state
pub fn default_consciousness_state() -> consciousness::ConsciousnessState {
    consciousness::ConsciousnessState::new()
}

/// Create a highly aroused consciousness state
pub fn aroused_consciousness_state() -> consciousness::ConsciousnessState {
    let mut state = consciousness::ConsciousnessState::new();
    state.emotional_arousal = 0.9;
    state.learning_will_activation = 0.8;
    state
}

/// Create a calm consciousness state
pub fn calm_consciousness_state() -> consciousness::ConsciousnessState {
    let mut state = consciousness::ConsciousnessState::new();
    state.emotional_arousal = 0.2;
    state.learning_will_activation = 0.3;
    state
}

/// Create consciousness states with varying emotional intensities
pub fn emotional_range_states() -> Vec<consciousness::ConsciousnessState> {
    (0..10)
        .map(|i| {
            let mut state = consciousness::ConsciousnessState::new();
            state.emotional_arousal = i as f32 * 0.1;
            state
        })
        .collect()
}

// ============================================================================
// MEMORY FRAGMENT FIXTURES
// ============================================================================

/// Create a semantic memory fragment
pub fn semantic_memory_fragment(content: &str) -> memory::mobius::MemoryFragment {
    memory::mobius::MemoryFragment {
        content: content.to_string(),
        layer: memory::mobius::MemoryLayer::Semantic,
        relevance: 0.8,
        timestamp: 1.0,
    }
}

/// Create an episodic memory fragment
pub fn episodic_memory_fragment(content: &str) -> memory::mobius::MemoryFragment {
    memory::mobius::MemoryFragment {
        content: content.to_string(),
        layer: memory::mobius::MemoryLayer::Episodic,
        relevance: 0.7,
        timestamp: 1.0,
    }
}

/// Create working memory fragment
pub fn working_memory_fragment(content: &str) -> memory::mobius::MemoryFragment {
    memory::mobius::MemoryFragment {
        content: content.to_string(),
        layer: memory::mobius::MemoryLayer::Working,
        relevance: 0.6,
        timestamp: 1.0,
    }
}

/// Create procedural memory fragment
pub fn procedural_memory_fragment(content: &str) -> memory::mobius::MemoryFragment {
    memory::mobius::MemoryFragment {
        content: content.to_string(),
        layer: memory::mobius::MemoryLayer::Procedural,
        relevance: 0.9,
        timestamp: 1.0,
    }
}

/// Create a sequence of memory fragments with increasing timestamps
pub fn temporal_memory_sequence(count: usize) -> Vec<memory::mobius::MemoryFragment> {
    (0..count)
        .map(|i| memory::mobius::MemoryFragment {
            content: format!("Memory {}", i),
            layer: memory::mobius::MemoryLayer::Episodic,
            relevance: 0.5,
            timestamp: i as f64,
        })
        .collect()
}

// ============================================================================
// GAUSSIAN MEMORY SPHERE FIXTURES
// ============================================================================

/// Create a 2D Gaussian memory sphere
pub fn gaussian_sphere_2d() -> dual_mobius_gaussian::GaussianMemorySphere {
    dual_mobius_gaussian::GaussianMemorySphere::new(
        vec![1.0, 2.0],
        vec![vec![1.0, 0.0], vec![0.0, 1.0]],
    )
}

/// Create a 3D Gaussian memory sphere
pub fn gaussian_sphere_3d() -> dual_mobius_gaussian::GaussianMemorySphere {
    dual_mobius_gaussian::GaussianMemorySphere::new(
        vec![1.0, 2.0, 3.0],
        vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ],
    )
}

/// Create a high-dimensional Gaussian memory sphere
pub fn gaussian_sphere_nd(dimensions: usize) -> dual_mobius_gaussian::GaussianMemorySphere {
    let mean = vec![1.0; dimensions];
    let mut covariance = vec![vec![0.0; dimensions]; dimensions];

    for i in 0..dimensions {
        covariance[i][i] = 1.0;
    }

    dual_mobius_gaussian::GaussianMemorySphere::new(mean, covariance)
}

/// Create a cluster of Gaussian spheres
pub fn gaussian_cluster(size: usize) -> Vec<dual_mobius_gaussian::GaussianMemorySphere> {
    (0..size)
        .map(|i| {
            dual_mobius_gaussian::GaussianMemorySphere::new(
                vec![i as f64, (i + 1) as f64],
                vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            )
        })
        .collect()
}

// ============================================================================
// TOROIDAL COORDINATE FIXTURES
// ============================================================================

/// Create origin toroidal coordinate
pub fn toroidal_origin() -> memory::toroidal::ToroidalCoordinate {
    memory::toroidal::ToroidalCoordinate::new(0.0, 0.0)
}

/// Create random toroidal coordinates
pub fn random_toroidal_coords(count: usize) -> Vec<memory::toroidal::ToroidalCoordinate> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    (0..count)
        .map(|_| {
            memory::toroidal::ToroidalCoordinate::new(
                rng.gen_range(0.0..std::f64::consts::TAU),
                rng.gen_range(0.0..std::f64::consts::TAU),
            )
        })
        .collect()
}

/// Create evenly spaced toroidal coordinates
pub fn evenly_spaced_toroidal_coords(count: usize) -> Vec<memory::toroidal::ToroidalCoordinate> {
    (0..count)
        .map(|i| {
            let theta = (i as f64 * 2.0 * std::f64::consts::PI) / count as f64;
            let phi = (i as f64 * std::f64::consts::PI) / count as f64;
            memory::toroidal::ToroidalCoordinate::new(theta, phi)
        })
        .collect()
}

// ============================================================================
// TOROIDAL MEMORY NODE FIXTURES
// ============================================================================

/// Create a basic toroidal memory node
pub fn basic_toroidal_node(id: &str) -> memory::toroidal::ToroidalMemoryNode {
    memory::toroidal::ToroidalMemoryNode {
        id: id.to_string(),
        coordinate: toroidal_origin(),
        content: format!("Content for {}", id),
        emotional_vector: vec![0.5, 0.3, 0.2],
        temporal_context: vec![1.0],
        activation_strength: 0.8,
        connections: std::collections::HashMap::new(),
    }
}

/// Create multiple connected toroidal nodes
pub fn connected_toroidal_nodes(count: usize) -> Vec<memory::toroidal::ToroidalMemoryNode> {
    let coords = evenly_spaced_toroidal_coords(count);

    coords
        .into_iter()
        .enumerate()
        .map(|(i, coord)| memory::toroidal::ToroidalMemoryNode {
            id: format!("node_{}", i),
            coordinate: coord,
            content: format!("Content {}", i),
            emotional_vector: vec![0.5],
            temporal_context: vec![i as f64],
            activation_strength: 0.7,
            connections: std::collections::HashMap::new(),
        })
        .collect()
}

// ============================================================================
// BRAIN TYPE AND PERSONALITY FIXTURES
// ============================================================================

/// Get all brain types
pub fn all_brain_types() -> Vec<brain::BrainType> {
    vec![
        brain::BrainType::Sensory,
        brain::BrainType::Motor,
        brain::BrainType::Emotional,
        brain::BrainType::Rational,
    ]
}

/// Get all personality types
pub fn all_personality_types() -> Vec<personality::PersonalityType> {
    vec![
        personality::PersonalityType::Intuitive,
        personality::PersonalityType::Analytical,
        personality::PersonalityType::Empathetic,
        personality::PersonalityType::Creative,
    ]
}

// ============================================================================
// TEST DATA GENERATORS
// ============================================================================

/// Generate random emotional input
pub fn random_emotional_input() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let emotions = vec!["happy", "sad", "angry", "calm", "excited", "anxious"];
    let emotion = emotions[rng.gen_range(0..emotions.len())];
    let intensity = rng.gen_range(0.0..1.0);

    format!("I feel {} with intensity {:.2}", emotion, intensity)
}

/// Generate test query strings
pub fn test_queries() -> Vec<String> {
    vec![
        "What is consciousness?".to_string(),
        "How do emotions work?".to_string(),
        "Tell me about memory".to_string(),
        "What is the meaning of life?".to_string(),
        "How can I improve?".to_string(),
    ]
}

// ============================================================================
// PERFORMANCE TEST UTILITIES
// ============================================================================

/// Timer for performance measurements
pub struct TestTimer {
    start: std::time::Instant,
    label: String,
}

impl TestTimer {
    pub fn new(label: &str) -> Self {
        Self {
            start: std::time::Instant::now(),
            label: label.to_string(),
        }
    }

    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    pub fn assert_under(&self, threshold_ms: f64) {
        let elapsed = self.elapsed_ms();
        assert!(
            elapsed < threshold_ms,
            "{} took {:.2}ms, exceeds {}ms threshold",
            self.label,
            elapsed,
            threshold_ms
        );
    }

    pub fn print_elapsed(&self) {
        tracing::info!("{}: {:.2}ms", self.label, self.elapsed_ms());
    }
}

// ============================================================================
// ASYNC TEST UTILITIES
// ============================================================================

/// Create a tokio runtime for async tests
pub fn create_test_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Runtime::new().unwrap()
}

/// Run async test with timeout
pub async fn run_with_timeout<F, T>(
    future: F,
    timeout_ms: u64,
) -> Result<T, tokio::time::error::Elapsed>
where
    F: std::future::Future<Output = T>,
{
    tokio::time::timeout(std::time::Duration::from_millis(timeout_ms), future).await
}

// ============================================================================
// VALIDATION UTILITIES
// ============================================================================

/// Validate consciousness state bounds
pub fn validate_consciousness_bounds(state: &consciousness::ConsciousnessState) -> bool {
    state.emotional_arousal >= 0.0
        && state.emotional_arousal <= 1.0
        && state.learning_will_activation >= 0.0
        && state.learning_will_activation <= 1.0
}

/// Validate memory fragment
pub fn validate_memory_fragment(fragment: &memory::mobius::MemoryFragment) -> bool {
    !fragment.content.is_empty()
        && fragment.relevance >= 0.0
        && fragment.relevance <= 1.0
        && fragment.timestamp >= 0.0
}

/// Validate Gaussian sphere
pub fn validate_gaussian_sphere(sphere: &dual_mobius_gaussian::GaussianMemorySphere) -> bool {
    !sphere.mean.is_empty()
        && sphere.covariance.len() == sphere.mean.len()
        && sphere
            .covariance
            .iter()
            .all(|row| row.len() == sphere.mean.len())
}

/// Validate toroidal coordinate
pub fn validate_toroidal_coordinate(coord: &memory::toroidal::ToroidalCoordinate) -> bool {
    coord.theta >= 0.0
        && coord.theta < 2.0 * std::f64::consts::PI
        && coord.phi >= 0.0
        && coord.phi < 2.0 * std::f64::consts::PI
}

// ============================================================================
// TEST CONFIGURATION
// ============================================================================

/// Standard test configuration
pub struct TestConfig {
    pub memory_capacity: usize,
    pub emotional_threshold: f32,
    pub learning_rate: f32,
    pub timeout_ms: u64,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            memory_capacity: 10000,
            emotional_threshold: 0.5,
            learning_rate: 0.01,
            timeout_ms: 5000,
        }
    }
}

impl TestConfig {
    pub fn fast() -> Self {
        Self {
            timeout_ms: 1000,
            ..Default::default()
        }
    }

    pub fn thorough() -> Self {
        Self {
            timeout_ms: 30000,
            ..Default::default()
        }
    }
}
