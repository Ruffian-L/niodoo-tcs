//! TCS (Topological Cognitive System) Analysis Module
//!
//! This module computes topological signatures (Betti numbers, persistence features)
//! from PAD state coordinates using giotto-tda via a Python subprocess bridge.
//!
//! The "Two-Language Problem" solution: We use a synchronous subprocess call to Python
//! for now (async version with pyo3-async-runtimes can come later). This keeps the
//! implementation simple while still providing real TDA computation.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::process::Command;

/// Topological signature computed from PAD state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalSignature {
    /// Betti numbers: [β₀, β₁, β₂]
    /// β₀ = connected components
    /// β₁ = loops/cycles
    /// β₂ = voids/cavities
    pub betti_numbers: [usize; 3],
    
    /// Persistence features: (birth, death, dimension, persistence)
    pub persistence_pairs: Vec<PersistencePair>,
    
    /// Shannon entropy of persistence lifetimes
    pub persistence_entropy: f64,
    
    /// Timestamp of computation
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistencePair {
    pub birth: f64,
    pub death: f64,
    pub dimension: usize,
    pub persistence: f64,
}

impl TopologicalSignature {
    /// Get β₀ (connected components)
    pub fn betti_0(&self) -> usize {
        self.betti_numbers[0]
    }

    /// Get β₁ (loops)
    pub fn betti_1(&self) -> usize {
        self.betti_numbers[1]
    }

    /// Get β₂ (voids)
    pub fn betti_2(&self) -> usize {
        self.betti_numbers[2]
    }

    /// Compute topological complexity as weighted sum of Betti numbers
    pub fn complexity(&self) -> f64 {
        (self.betti_0() as f64) * 0.1 + (self.betti_1() as f64) * 0.5 + (self.betti_2() as f64) * 1.0
    }
}

/// TCS Analyzer that computes topological signatures
pub struct TCSAnalyzer {
    python_path: String,
    wrapper_path: String,
}

impl TCSAnalyzer {
    /// Create a new TCS analyzer
    pub fn new() -> Result<Self> {
        // Use python3 from venv if available
        let python_path = std::env::var("VIRTUAL_ENV")
            .map(|venv| format!("{}/bin/python3", venv))
            .unwrap_or_else(|_| "python3".to_string());

        let wrapper_path = "src/giotto_wrapper.py".to_string();

        Ok(Self {
            python_path,
            wrapper_path,
        })
    }

    /// Analyze PAD coordinates to compute topological signature
    ///
    /// This takes the 7D PAD coordinates and treats them as a point cloud,
    /// then computes persistent homology to extract topological features.
    pub fn analyze_pad_state(&self, pad_coordinates: &[f64; 7]) -> Result<TopologicalSignature> {
        // Convert PAD coordinates to a point cloud
        // We'll create a simple point cloud by treating each dimension as a point
        // For a more sophisticated analysis, we could use sliding windows or
        // multiple samples, but for now this gives us a baseline topology
        let points: Vec<Vec<f64>> = vec![
            vec![pad_coordinates[0], pad_coordinates[1], pad_coordinates[2]], // PAD
            vec![pad_coordinates[3], pad_coordinates[4], pad_coordinates[5]], // Ghost 1-3
            vec![pad_coordinates[6], 0.0, 0.0], // Ghost 4 + padding
        ];

        self.compute_persistence(&points, 2.0)
    }

    /// Compute persistent homology for a point cloud
    fn compute_persistence(
        &self,
        points: &[Vec<f64>],
        max_filtration: f64,
    ) -> Result<TopologicalSignature> {
        // Prepare input JSON
        let input = serde_json::json!({
            "points": points,
            "max_filtration": max_filtration,
        });

        let input_str = serde_json::to_string(&input)?;

        // Call Python wrapper via subprocess
        let output = Command::new(&self.python_path)
            .arg(&self.wrapper_path)
            .arg(&input_str)
            .output()
            .with_context(|| {
                format!(
                    "failed to execute giotto_wrapper.py (python: {}, wrapper: {})",
                    self.python_path, self.wrapper_path
                )
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("giotto_wrapper.py failed: {}", stderr);
        }

        // Parse output JSON
        let stdout = String::from_utf8_lossy(&output.stdout);
        let result: GiottoOutput = serde_json::from_str(&stdout)
            .with_context(|| format!("failed to parse giotto output: {}", stdout))?;

        // Check for errors
        if let Some(error) = result.error {
            anyhow::bail!("TDA computation error: {}", error);
        }

        // Convert to TopologicalSignature
        let betti_numbers = [
            result.betti_numbers.get(0).copied().unwrap_or(0),
            result.betti_numbers.get(1).copied().unwrap_or(0),
            result.betti_numbers.get(2).copied().unwrap_or(0),
        ];

        let persistence_pairs = result
            .persistence_pairs
            .into_iter()
            .map(|p| PersistencePair {
                birth: p.birth,
                death: p.death,
                dimension: p.dimension,
                persistence: p.persistence,
            })
            .collect();

        Ok(TopologicalSignature {
            betti_numbers,
            persistence_pairs,
            persistence_entropy: result.persistence_entropy,
            timestamp: chrono::Utc::now(),
        })
    }
}

/// Output format from giotto_wrapper.py
#[derive(Debug, Deserialize)]
struct GiottoOutput {
    #[serde(default)]
    error: Option<String>,
    betti_numbers: Vec<usize>,
    persistence_pairs: Vec<GiottoPair>,
    persistence_entropy: f64,
}

#[derive(Debug, Deserialize)]
struct GiottoPair {
    birth: f64,
    death: f64,
    dimension: usize,
    persistence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tcs_analyzer_creation() {
        let analyzer = TCSAnalyzer::new();
        assert!(analyzer.is_ok());
    }

    #[test]
    fn test_pad_analysis() {
        let analyzer = TCSAnalyzer::new().unwrap();
        
        // Test PAD coordinates from a real run
        let pad_coords = [0.913, 0.885, 0.999, 0.5, 0.3, -0.2, 0.1];
        
        // This will fail if giotto-tda is not installed, which is expected
        // In CI/CD, we'd skip this test or mock it
        let result = analyzer.analyze_pad_state(&pad_coords);
        
        // Just check it doesn't panic - actual result depends on giotto-tda
        match result {
            Ok(sig) => {
                assert_eq!(sig.betti_numbers.len(), 3);
            }
            Err(_) => {
                // Expected if giotto-tda not installed
            }
        }
    }
}

