//! Consciousness Compass - Strategic Quadrant Mapping
//!
//! Maps PAD state + topological signature to one of four strategic quadrants:
//! - Panic: High entropy + fragmented (low β₀) → reactive, defensive
//! - Persist: Low entropy + stable (low β₁) → methodical, focused
//! - Discover: High entropy + loops (high β₁) → exploratory, creative
//! - Master: Low entropy + unified (high β₀) → confident, integrative
//!
//! This is the "brain" that decides how to interpret the current cognitive state.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::tcs_analysis::TopologicalSignature;
use crate::torus::PadState;

/// Consciousness Compass quadrant
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompassQuadrant {
    /// High entropy + fragmented → reactive, defensive
    Panic,
    /// Low entropy + stable → methodical, focused
    Persist,
    /// High entropy + loops → exploratory, creative
    Discover,
    /// Low entropy + unified → confident, integrative
    Master,
}

impl CompassQuadrant {
    pub fn as_str(&self) -> &'static str {
        match self {
            CompassQuadrant::Panic => "Panic",
            CompassQuadrant::Persist => "Persist",
            CompassQuadrant::Discover => "Discover",
            CompassQuadrant::Master => "Master",
        }
    }
}

/// Compass state with strategic decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompassState {
    /// Current quadrant
    pub quadrant: CompassQuadrant,
    
    /// Confidence in this quadrant assignment (0.0-1.0)
    pub confidence: f64,
    
    /// Entropy level (from PAD state)
    pub entropy: f64,
    
    /// Topological complexity
    pub complexity: f64,
    
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Compass configuration thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompassConfig {
    /// Entropy threshold for high/low split
    pub entropy_threshold: f64,
    
    /// β₁ (loops) threshold for topology split
    pub beta1_threshold: usize,
    
    /// β₀ (components) threshold for fragmentation
    pub beta0_threshold: usize,
}

impl Default for CompassConfig {
    fn default() -> Self {
        Self {
            entropy_threshold: 1.5,  // Mid-range for 7D distribution
            beta1_threshold: 2,       // More than 2 loops = high connectivity
            beta0_threshold: 2,       // More than 2 components = fragmented
        }
    }
}

impl CompassConfig {
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read compass config from {}", path))?;
        toml::from_str(&content).context("failed to parse compass config")
    }
}

/// Consciousness Compass engine
pub struct CompassEngine {
    config: CompassConfig,
}

impl CompassEngine {
    pub fn new(config: CompassConfig) -> Self {
        Self { config }
    }

    /// Compute compass quadrant from PAD state and topology
    ///
    /// Decision tree:
    /// 1. Check entropy: high vs low
    /// 2. Check topology: fragmented vs connected vs loopy
    /// 3. Map to quadrant based on combination
    pub fn compute_quadrant(
        &self,
        pad_state: &PadState,
        topology: &TopologicalSignature,
    ) -> CompassState {
        let entropy = pad_state.entropy;
        let beta0 = topology.betti_0();
        let beta1 = topology.betti_1();
        let complexity = topology.complexity();

        let high_entropy = entropy > self.config.entropy_threshold;
        let fragmented = beta0 > self.config.beta0_threshold;
        let loopy = beta1 > self.config.beta1_threshold;

        // Decision logic
        let (quadrant, confidence) = if high_entropy {
            if fragmented {
                // High entropy + fragmented = Panic
                (CompassQuadrant::Panic, 0.9)
            } else if loopy {
                // High entropy + loops = Discover
                (CompassQuadrant::Discover, 0.85)
            } else {
                // High entropy but stable = leaning Discover
                (CompassQuadrant::Discover, 0.6)
            }
        } else {
            // Low entropy
            if fragmented {
                // Low entropy but fragmented = Persist (trying to stabilize)
                (CompassQuadrant::Persist, 0.7)
            } else if loopy {
                // Low entropy + loops = Master (confident exploration)
                (CompassQuadrant::Master, 0.8)
            } else {
                // Low entropy + stable = Persist
                (CompassQuadrant::Persist, 0.9)
            }
        };

        CompassState {
            quadrant,
            confidence,
            entropy,
            complexity,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Get strategic advice for current quadrant
    pub fn strategic_advice(&self, quadrant: CompassQuadrant) -> &'static str {
        match quadrant {
            CompassQuadrant::Panic => {
                "Reactive mode: Focus on immediate stabilization. Retrieve defensive patterns."
            }
            CompassQuadrant::Persist => {
                "Methodical mode: Continue current approach. Retrieve proven solutions."
            }
            CompassQuadrant::Discover => {
                "Exploratory mode: Try novel approaches. Retrieve creative patterns."
            }
            CompassQuadrant::Master => {
                "Integrative mode: Synthesize knowledge. Retrieve high-level abstractions."
            }
        }
    }
}

use anyhow::Context;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compass_quadrants() {
        let config = CompassConfig::default();
        let engine = CompassEngine::new(config);

        // Mock PAD state with high entropy
        let high_entropy_pad = PadState {
            coordinates: [0.5, 0.5, 0.5, 0.3, 0.2, 0.1, 0.0],
            entropy: 2.0,  // High
            mu: [0.0; 7],
            sigma: [0.5; 7],
            surface_position: [1.0, 0.0, 0.0],
        };

        // Mock topology with loops
        let loopy_topology = TopologicalSignature {
            betti_numbers: [1, 3, 0],  // 3 loops
            persistence_pairs: vec![],
            persistence_entropy: 1.0,
            timestamp: chrono::Utc::now(),
        };

        let state = engine.compute_quadrant(&high_entropy_pad, &loopy_topology);
        assert_eq!(state.quadrant, CompassQuadrant::Discover);
        assert!(state.confidence > 0.5);
    }

    #[test]
    fn test_strategic_advice() {
        let config = CompassConfig::default();
        let engine = CompassEngine::new(config);

        let advice = engine.strategic_advice(CompassQuadrant::Master);
        assert!(advice.contains("Integrative"));
    }
}

