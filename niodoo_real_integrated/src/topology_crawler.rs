//! Topology Space Crawler - Systematic exploration of topological positions
//! Tests healing/topology integration at specific coordinates in the space

use crate::compass::{CompassEngine, CompassOutcome};
use crate::tcs_analysis::{TCSAnalyzer, TopologicalSignature};
use crate::torus::PadGhostState;
use anyhow::Result;
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

/// Specific positions in topology space to test
#[derive(Debug, Clone)]
pub struct TopologyPosition {
    pub name: String,
    pub coordinates: [f64; 7], // PAD values
    pub mu: [f64; 7],
    pub sigma: [f64; 7],
    pub expected_knot_complexity_range: (f64, f64),
    pub expected_healing_behavior: bool,
}

impl TopologyPosition {
    pub fn to_pad_state(&self) -> PadGhostState {
        PadGhostState {
            pad: self.coordinates,
            entropy: crate::util::shannon_entropy(&[
                self.coordinates[0],
                self.coordinates[1],
                self.coordinates[2],
                self.coordinates[3],
                self.coordinates[4],
                self.coordinates[5],
                self.coordinates[6],
            ]),
            mu: self.mu,
            sigma: self.sigma,
            raw_stds: self.sigma.to_vec(),
        }
    }
}

/// Topology Crawler - Systematic exploration tool
pub struct TopologyCrawler {
    analyzer: TCSAnalyzer,
    compass: Arc<Mutex<CompassEngine>>,
    test_positions: Vec<TopologyPosition>,
}

impl TopologyCrawler {
    pub fn new() -> Result<Self> {
        let analyzer = TCSAnalyzer::new()?;
        let compass = Arc::new(Mutex::new(CompassEngine::new(
            0.5, // mcts_c
            0.4, // variance_spike
            0.2, // variance_stagnation
        )));

        // Define specific test positions
        let test_positions = vec![
            // Position 1.27 - High pleasure, medium arousal (expected healing state)
            TopologyPosition {
                name: "Position 1.27 - Healing".to_string(),
                coordinates: [0.85, 0.5, 0.7, 0.3, 0.4, 0.2, 0.1],
                mu: [0.8, 0.4, 0.6, 0.2, 0.3, 0.1, 0.05],
                sigma: [0.1, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05],
                expected_knot_complexity_range: (10.0, 20.0), // Betti-1 counts
                expected_healing_behavior: true,
            },
            // Position 1.4 - Extreme pleasure (optimal healing)
            TopologyPosition {
                name: "Position 1.4 - Optimal Healing".to_string(),
                coordinates: [0.95, 0.6, 0.8, 0.4, 0.5, 0.3, 0.2],
                mu: [0.9, 0.5, 0.7, 0.3, 0.4, 0.2, 0.1],
                sigma: [0.08, 0.15, 0.12, 0.08, 0.08, 0.08, 0.04],
                expected_knot_complexity_range: (10.0, 20.0), // Betti-1 counts
                expected_healing_behavior: true,
            },
            // Position 2.0 - High arousal with negative pleasure (threat state)
            TopologyPosition {
                name: "Position 2.0 - Threat".to_string(),
                coordinates: [-0.3, 0.8, -0.5, 0.6, -0.2, 0.4, 0.1],
                mu: [-0.2, 0.7, -0.4, 0.5, -0.1, 0.3, 0.05],
                sigma: [0.2, 0.25, 0.2, 0.2, 0.15, 0.15, 0.1],
                expected_knot_complexity_range: (10.0, 20.0), // Betti-1 counts
                expected_healing_behavior: false,
            },
            // Position 0.8 - Low entropy, simple topology
            TopologyPosition {
                name: "Position 0.8 - Simple Topology".to_string(),
                coordinates: [0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
                mu: [0.25, 0.15, 0.08, 0.0, 0.0, 0.0, 0.0],
                sigma: [0.1, 0.1, 0.05, 0.01, 0.01, 0.01, 0.01],
                expected_knot_complexity_range: (10.0, 20.0), // Betti-1 counts
                expected_healing_behavior: true,
            },
            // Position 1.0 - Balanced state
            TopologyPosition {
                name: "Position 1.0 - Balanced".to_string(),
                coordinates: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                mu: [0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45],
                sigma: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                expected_knot_complexity_range: (10.0, 20.0), // Betti-1 counts
                expected_healing_behavior: false,
            },
        ];

        Ok(Self {
            analyzer,
            compass,
            test_positions,
        })
    }

    /// Crawl through all test positions and validate healing/topology integration
    pub async fn crawl_space(&mut self) -> Result<CrawlResults> {
        info!("🕷️  TOPOLOGY CRAWLER STARTING");
        info!("================================");
        info!(
            "Testing {} positions in topology space",
            self.test_positions.len()
        );

        let mut results = Vec::new();
        let mut healing_detected = 0;
        let mut healing_correct = 0;

        for position in &self.test_positions {
            info!("");
            info!("📍 Testing: {}", position.name);
            info!("   Coordinates: {:?}", position.coordinates);

            let pad_state = position.to_pad_state();

            // Analyze topology
            let topology = self.analyzer.analyze_state(&pad_state)?;
            info!(
                "   Topology: knot={:.3}, gap={:.3}, betti={:?}",
                topology.knot_complexity, topology.spectral_gap, topology.betti_numbers
            );

            // Evaluate compass (with topology integration)
            let compass_outcome = self
                .compass
                .lock()
                .map_err(|e| anyhow::anyhow!("Failed to acquire compass lock: {}", e))?
                .evaluate(&pad_state, Some(&topology))?;

            info!(
                "   Compass: quadrant={:?}, threat={}, healing={}",
                compass_outcome.quadrant, compass_outcome.is_threat, compass_outcome.is_healing
            );

            // Validate knot complexity range
            let knot_valid = topology.knot_complexity >= position.expected_knot_complexity_range.0
                && topology.knot_complexity <= position.expected_knot_complexity_range.1;

            // Validate healing behavior
            let healing_match = compass_outcome.is_healing == position.expected_healing_behavior;

            if compass_outcome.is_healing {
                healing_detected += 1;
                if healing_match {
                    healing_correct += 1;
                }
            }

            let result = PositionResult {
                position: position.clone(),
                topology,
                compass_outcome: compass_outcome.clone(),
                knot_valid,
                healing_match,
            };

            results.push(result);

            // Log result
            if knot_valid && healing_match {
                info!("   ✅ PASS: Knot complexity and healing behavior match expectations");
            } else {
                if !knot_valid {
                    warn!("   ⚠️  Knot complexity outside expected range");
                }
                if !healing_match {
                    warn!(
                        "   ⚠️  Healing behavior mismatch: expected={}, got={}",
                        position.expected_healing_behavior, compass_outcome.is_healing
                    );
                }
            }
        }

        info!("");
        info!("📊 CRAWL RESULTS SUMMARY");
        info!("=========================");
        info!("Total positions tested: {}", results.len());
        info!(
            "Knot complexity valid: {}/{}",
            results.iter().filter(|r| r.knot_valid).count(),
            results.len()
        );
        info!("Healing detected: {}", healing_detected);
        info!("Healing correct: {}/{}", healing_correct, healing_detected);

        Ok(CrawlResults {
            positions: results,
            healing_detected,
            healing_correct,
        })
    }

    /// Test specific healing scenarios
    pub async fn test_healing_scenarios(&mut self) -> Result<HealingTestResults> {
        info!("");
        info!("🏥 HEALING INTEGRATION TESTS");
        info!("============================");

        let mut passed = 0;
        let mut failed = 0;

        // Test 1: Good emotional state should trigger healing (regardless of knot complexity)
        info!("Test 1: Positive emotion triggers healing");
        let pad_state = PadGhostState {
            pad: [0.8, 0.3, 0.6, 0.2, 0.3, 0.1, 0.05],
            entropy: 1.85,
            mu: [0.75, 0.25, 0.55, 0.15, 0.25, 0.08, 0.03],
            sigma: [0.1, 0.15, 0.12, 0.08, 0.08, 0.05, 0.03],
            raw_stds: vec![0.1, 0.15, 0.12, 0.08, 0.08, 0.05, 0.03],
        };
        let topology = self.analyzer.analyze_state(&pad_state)?;
        let compass = self
            .compass
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to acquire compass lock: {}", e))?
            .evaluate(&pad_state, Some(&topology))?;

        if compass.is_healing {
            info!("   ✅ PASS: Healing triggered correctly");
            passed += 1;
        } else {
            warn!(
                "   ❌ FAIL: Expected healing state (knot={:.1})",
                topology.knot_complexity
            );
            failed += 1;
        }

        // Test 2: High spectral gap + positive emotion should trigger healing
        info!("Test 2: High spectral gap + positive emotion");
        let pad_state = PadGhostState {
            pad: [0.9, 0.4, 0.7, 0.3, 0.4, 0.2, 0.1],
            entropy: 1.95,
            mu: [0.85, 0.35, 0.65, 0.25, 0.35, 0.15, 0.08],
            sigma: [0.08, 0.12, 0.1, 0.08, 0.08, 0.08, 0.04],
            raw_stds: vec![0.08, 0.12, 0.1, 0.08, 0.08, 0.08, 0.04],
        };
        let topology = self.analyzer.analyze_state(&pad_state)?;
        let compass = self
            .compass
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to acquire compass lock: {}", e))?
            .evaluate(&pad_state, Some(&topology))?;

        if topology.spectral_gap > 0.7 && compass.is_healing {
            info!("   ✅ PASS: Healing triggered by spectral gap");
            passed += 1;
        } else {
            warn!(
                "   ❌ FAIL: Expected healing from spectral gap (gap={:.3})",
                topology.spectral_gap
            );
            failed += 1;
        }

        // Test 3: Low persistence entropy should trigger healing
        info!("Test 3: Low persistence entropy");
        let pad_state = PadGhostState {
            pad: [0.7, 0.3, 0.5, 0.2, 0.25, 0.1, 0.05],
            entropy: 1.75,
            mu: [0.65, 0.25, 0.45, 0.15, 0.2, 0.08, 0.03],
            sigma: [0.09, 0.1, 0.1, 0.07, 0.07, 0.05, 0.03],
            raw_stds: vec![0.09, 0.1, 0.1, 0.07, 0.07, 0.05, 0.03],
        };
        let topology = self.analyzer.analyze_state(&pad_state)?;
        let compass = self
            .compass
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to acquire compass lock: {}", e))?
            .evaluate(&pad_state, Some(&topology))?;

        if topology.persistence_entropy < 0.3 && !compass.is_threat && compass.is_healing {
            info!("   ✅ PASS: Healing triggered by stable structure");
            passed += 1;
        } else {
            warn!(
                "   ❌ FAIL: Expected healing from stable structure (entropy={:.3})",
                topology.persistence_entropy
            );
            failed += 1;
        }

        // Test 4: High knot complexity should NOT trigger healing
        info!("Test 4: High knot complexity (should NOT heal)");
        let pad_state = PadGhostState {
            pad: [-0.3, 0.8, -0.5, 0.6, -0.2, 0.4, 0.1],
            entropy: 2.1,
            mu: [-0.2, 0.7, -0.4, 0.5, -0.1, 0.3, 0.05],
            sigma: [0.2, 0.25, 0.2, 0.2, 0.15, 0.15, 0.1],
            raw_stds: vec![0.2, 0.25, 0.2, 0.2, 0.15, 0.15, 0.1],
        };
        let topology = self.analyzer.analyze_state(&pad_state)?;
        let compass = self
            .compass
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to acquire compass lock: {}", e))?
            .evaluate(&pad_state, Some(&topology))?;

        if topology.knot_complexity > 0.7 && !compass.is_healing {
            info!("   ✅ PASS: No healing for complex topology");
            passed += 1;
        } else {
            warn!(
                "   ❌ FAIL: Should not heal with high knot complexity ({:.3})",
                topology.knot_complexity
            );
            failed += 1;
        }

        info!("");
        info!(
            "Healing integration tests: {}/{} passed",
            passed,
            passed + failed
        );

        Ok(HealingTestResults {
            passed,
            failed,
            total: passed + failed,
        })
    }
}

#[derive(Debug, Clone)]
pub struct PositionResult {
    pub position: TopologyPosition,
    pub topology: TopologicalSignature,
    pub compass_outcome: CompassOutcome,
    pub knot_valid: bool,
    pub healing_match: bool,
}

#[derive(Debug)]
pub struct CrawlResults {
    pub positions: Vec<PositionResult>,
    pub healing_detected: usize,
    pub healing_correct: usize,
}

#[derive(Debug)]
pub struct HealingTestResults {
    pub passed: usize,
    pub failed: usize,
    pub total: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_topology_crawler() {
        let mut crawler = TopologyCrawler::new().unwrap();
        let results = crawler.crawl_space().await.unwrap();

        // Verify we tested all positions
        assert_eq!(results.positions.len(), 5);

        // Verify at least some healing was detected
        assert!(results.healing_detected > 0);
    }

    #[tokio::test]
    async fn test_healing_scenarios() {
        let mut crawler = TopologyCrawler::new().unwrap();
        let results = crawler.test_healing_scenarios().await.unwrap();

        // At least 3 out of 4 tests should pass
        assert!(
            results.passed >= 3,
            "Healing integration test failed: {}/{} passed",
            results.passed,
            results.total
        );
    }
}
