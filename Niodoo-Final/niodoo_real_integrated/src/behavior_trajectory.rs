//! Behavior Trajectory Analysis for Persistent Homology Trust Assessment
//!
//! Analyzes agent behavior as point clouds in 7D PAD+Ghost space:
//! - Collects PadGhostState sequences from EragMemory records
//! - Converts trajectories to point clouds for persistent homology
//! - Computes trust metrics using H1 persistence (loops) and H2 persistence (voids)
//! - Classifies behavior patterns via topological signatures

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use crate::erag::EragMemory;
use crate::torus::PadGhostState;

/// Point cloud in high-dimensional space (local definition for trajectory analysis)
#[derive(Debug, Clone)]
pub struct PointCloud {
    pub points: Vec<Vec<f64>>,
    pub dimension: usize,
}

impl PointCloud {
    pub fn new(points: Vec<Vec<f64>>) -> Self {
        let dimension = points.first().map(|p| p.len()).unwrap_or(0);
        Self { points, dimension }
    }

    /// Calculate pairwise distances between all points
    pub fn pairwise_distances(&self) -> Vec<Vec<f64>> {
        let n = self.points.len();
        let mut distances = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    distances[i][j] = self.euclidean_distance(&self.points[i], &self.points[j]);
                }
            }
        }

        distances
    }

    /// Euclidean distance between two points
    fn euclidean_distance(&self, p1: &[f64], p2: &[f64]) -> f64 {
        p1.iter()
            .zip(p2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// Behavior trajectory: sequence of PadGhostState with timestamps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorTrajectory {
    /// Sequence of PAD+Ghost states
    pub states: Vec<PadGhostState>,
    /// Timestamps corresponding to each state
    pub timestamps: Vec<DateTime<Utc>>,
    /// Memory IDs associated with each state (for tracking)
    pub memory_ids: Vec<String>,
}

impl BehaviorTrajectory {
    /// Create new empty trajectory
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            timestamps: Vec::new(),
            memory_ids: Vec::new(),
        }
    }

    /// Add a state to the trajectory
    pub fn add_state(&mut self, state: PadGhostState, timestamp: DateTime<Utc>, memory_id: String) {
        self.states.push(state);
        self.timestamps.push(timestamp);
        self.memory_ids.push(memory_id);
    }

    /// Get length of trajectory
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Check if trajectory is empty
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

impl Default for BehaviorTrajectory {
    fn default() -> Self {
        Self::new()
    }
}

/// Trust metrics computed from persistent homology analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustMetrics {
    /// H1 trust score: normalized H1 persistence (loops indicate consistency)
    /// Range: [0.0, 1.0], higher = more trustworthy
    pub h1_trust_score: f32,
    /// H2 anomaly score: normalized H2 persistence (voids indicate gaps)
    /// Range: [0.0, 1.0], higher = more anomalous
    pub h2_anomaly_score: f32,
    /// Persistence entropy for adaptive decay
    pub persistence_entropy: f64,
    /// Topological signature: pattern classification
    pub topological_signature: String,
}

impl TrustMetrics {
    /// Create default trust metrics (neutral values)
    pub fn default() -> Self {
        Self {
            h1_trust_score: 0.5,
            h2_anomaly_score: 0.0,
            persistence_entropy: 0.0,
            topological_signature: "unknown".to_string(),
        }
    }
}

/// Trajectory analyzer for collecting and analyzing behavior trajectories
pub struct TrajectoryAnalyzer {
    /// Minimum trajectory length for analysis
    pub min_trajectory_length: usize,
    /// Maximum trajectory length to keep in memory
    pub max_trajectory_length: usize,
}

impl TrajectoryAnalyzer {
    /// Create new trajectory analyzer
    pub fn new(min_length: usize, max_length: usize) -> Self {
        Self {
            min_trajectory_length: min_length,
            max_trajectory_length: max_length,
        }
    }

    /// Collect trajectory from EragMemory records
    ///
    /// Extracts PadGhostState sequences from memories, ordered by timestamp.
    /// Note: EragMemory doesn't directly contain PadGhostState, so we reconstruct
    /// it from available fields or use a placeholder approach.
    pub fn collect_trajectory(&self, memories: &[EragMemory]) -> BehaviorTrajectory {
        let mut trajectory = BehaviorTrajectory::new();

        // Sort memories by timestamp
        let mut sorted_memories: Vec<(usize, &EragMemory)> = memories
            .iter()
            .enumerate()
            .filter_map(|(idx, mem)| {
                DateTime::parse_from_rfc3339(&mem.timestamp)
                    .ok()
                    .map(|dt| (idx, (dt.with_timezone(&Utc), mem)))
            })
            .map(|(idx, (dt, mem))| (idx, mem))
            .collect();

        // Sort by timestamp
        sorted_memories.sort_by(|a, b| {
            let t1 = DateTime::parse_from_rfc3339(&a.1.timestamp).unwrap();
            let t2 = DateTime::parse_from_rfc3339(&b.1.timestamp).unwrap();
            t1.cmp(&t2)
        });

        // Reconstruct PadGhostState from EragMemory
        // Since EragMemory doesn't store PadGhostState directly, we reconstruct
        // from entropy_after and emotional_vector
        for (idx, memory) in sorted_memories {
            // Reconstruct PadGhostState approximation from available data
            // This is a simplified reconstruction - in practice, PadGhostState
            // should be stored with EragMemory or retrieved from torus projection
            let pad_state = self.reconstruct_pad_state(memory);
            let timestamp = DateTime::parse_from_rfc3339(&memory.timestamp)
                .unwrap()
                .with_timezone(&Utc);
            let memory_id = format!("mem_{}", idx);

            trajectory.add_state(pad_state, timestamp, memory_id);
        }

        // Trim to max length if needed
        if trajectory.len() > self.max_trajectory_length {
            let excess = trajectory.len() - self.max_trajectory_length;
            trajectory.states.drain(0..excess);
            trajectory.timestamps.drain(0..excess);
            trajectory.memory_ids.drain(0..excess);
        }

        trajectory
    }

    /// Reconstruct PadGhostState from EragMemory
    ///
    /// This is a simplified reconstruction. In practice, PadGhostState should
    /// be stored with EragMemory or retrieved from the torus projection.
    fn reconstruct_pad_state(&self, memory: &EragMemory) -> PadGhostState {
        // Use entropy_after as a proxy for PAD state
        // Map entropy to PAD dimensions (simplified)
        let entropy = memory.entropy_after;

        // Create a simplified PadGhostState
        // In practice, this should come from the actual torus projection
        let mut pad = [0.0; 7];
        pad[0] = entropy.sin(); // Pleasure proxy
        pad[1] = entropy.cos(); // Arousal proxy
        pad[2] = entropy; // Dominance proxy
                          // Ghost dimensions (3-6) use entropy variations
        for i in 3..7 {
            pad[i] = (entropy * (i as f64 + 1.0)).sin();
        }

        PadGhostState {
            pad,
            entropy,
            mu: pad,         // Use pad as mu approximation
            sigma: [0.1; 7], // Default sigma
        }
    }

    /// Convert trajectory to point cloud in 7D PAD+Ghost space
    pub fn to_point_cloud(&self, trajectory: &BehaviorTrajectory) -> PointCloud {
        let points: Vec<Vec<f64>> = trajectory
            .states
            .iter()
            .map(|state| {
                // Extract 7D coordinates: pad[0..7]
                state.pad.iter().map(|&x| x as f64).collect()
            })
            .collect();

        PointCloud::new(points)
    }

    /// Create sliding windows from trajectory
    ///
    /// Returns multiple trajectories of window_size, sliding by 1 step
    pub fn sliding_window(
        &self,
        trajectory: &BehaviorTrajectory,
        window_size: usize,
    ) -> Vec<BehaviorTrajectory> {
        if trajectory.len() < window_size {
            return vec![];
        }

        let mut windows = Vec::new();
        for i in 0..=(trajectory.len() - window_size) {
            let mut window = BehaviorTrajectory::new();
            for j in i..(i + window_size) {
                window.states.push(trajectory.states[j].clone());
                window.timestamps.push(trajectory.timestamps[j]);
                window.memory_ids.push(trajectory.memory_ids[j].clone());
            }
            windows.push(window);
        }

        windows
    }

    /// Classify behavior pattern from H1 and H2 persistence
    ///
    /// Patterns:
    /// - "toroidal": High H1, low H2 (balanced, consistent loops)
    /// - "balanced": Moderate H1, low H2 (normal behavior)
    /// - "suspicious": High H2 (voids indicate gaps/anomalies)
    /// - "sparse": Low H1, low H2 (insufficient data)
    pub fn classify_pattern(h1_score: f32, h2_score: f32) -> String {
        if h2_score > 0.5 {
            "suspicious".to_string()
        } else if h1_score > 0.7 && h2_score < 0.2 {
            "toroidal".to_string()
        } else if h1_score > 0.3 && h2_score < 0.3 {
            "balanced".to_string()
        } else {
            "sparse".to_string()
        }
    }

    /// Compute trust metrics from trajectory using persistent homology
    ///
    /// This method computes H1 and H2 persistence from the trajectory point cloud
    /// and calculates trust metrics including persistence entropy.
    pub fn compute_trust_metrics(
        &self,
        trajectory: &BehaviorTrajectory,
        h1_barcodes: &[(f64, f64)],
        h2_barcodes: &[(f64, f64)],
    ) -> TrustMetrics {
        // Calculate H1 trust score: average persistence of loops
        let h1_trust_score = if !h1_barcodes.is_empty() {
            let avg_persistence: f64 = h1_barcodes
                .iter()
                .map(|(birth, death)| death - birth)
                .sum::<f64>()
                / h1_barcodes.len() as f64;
            // Normalize to [0, 1] - assume max persistence around 2.0
            (avg_persistence / 2.0).min(1.0) as f32
        } else {
            0.0
        };

        // Calculate H2 anomaly score: average persistence of voids
        let h2_anomaly_score = if !h2_barcodes.is_empty() {
            let avg_persistence: f64 = h2_barcodes
                .iter()
                .map(|(birth, death)| death - birth)
                .sum::<f64>()
                / h2_barcodes.len() as f64;
            // Normalize to [0, 1] - voids indicate anomalies
            (avg_persistence / 2.0).min(1.0) as f32
        } else {
            0.0
        };

        // Calculate persistence entropy from all barcodes
        let all_barcodes: Vec<(f64, f64)> = h1_barcodes
            .iter()
            .chain(h2_barcodes.iter())
            .copied()
            .collect();
        let persistence_entropy = Self::calculate_persistence_entropy(&all_barcodes);

        // Classify pattern
        let topological_signature = Self::classify_pattern(h1_trust_score, h2_anomaly_score);

        TrustMetrics {
            h1_trust_score,
            h2_anomaly_score,
            persistence_entropy,
            topological_signature,
        }
    }

    /// Calculate persistence entropy from barcodes
    ///
    /// Entropy = -Σ(p_i * log(p_i)) where p_i = persistence_i / total_persistence
    fn calculate_persistence_entropy(barcodes: &[(f64, f64)]) -> f64 {
        if barcodes.is_empty() {
            return 0.0;
        }

        let persistences: Vec<f64> = barcodes
            .iter()
            .map(|(birth, death)| (death - birth).max(0.0))
            .collect();

        let total_persistence: f64 = persistences.iter().sum();
        if total_persistence == 0.0 {
            return 0.0;
        }

        let mut entropy = 0.0;
        for &p in &persistences {
            if p > 0.0 {
                let prob = p / total_persistence;
                entropy -= prob * prob.ln();
            }
        }

        entropy
    }
}

impl Default for TrajectoryAnalyzer {
    fn default() -> Self {
        Self::new(10, 1000) // Min 10 points, max 1000 points
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compass::CascadeStage;
    use crate::erag::{EmotionalVector, EragMemory};

    fn create_test_memory(id: usize, timestamp: &str, entropy: f64) -> EragMemory {
        EragMemory {
            input: format!("input_{}", id),
            output: format!("output_{}", id),
            emotional_vector: EmotionalVector::default(),
            erag_context: vec![],
            entropy_before: entropy - 0.1,
            entropy_after: entropy,
            timestamp: timestamp.to_string(),
            compass_state: None,
            cascade_stage: Some(CascadeStage::Recognition),
            weighted_metadata: None,
        }
    }

    #[test]
    fn test_trajectory_collection() {
        let analyzer = TrajectoryAnalyzer::default();
        let memories = vec![
            create_test_memory(0, "2025-01-01T00:00:00Z", 0.5),
            create_test_memory(1, "2025-01-01T00:30:00Z", 0.6),
            create_test_memory(2, "2025-01-01T01:00:00Z", 0.7),
        ];
        let trajectory = analyzer.collect_trajectory(&memories);
        assert_eq!(trajectory.len(), 3);
    }

    #[test]
    fn test_point_cloud_conversion() {
        let analyzer = TrajectoryAnalyzer::default();
        let mut trajectory = BehaviorTrajectory::new();
        let pad_state = PadGhostState {
            pad: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            entropy: 0.5,
            mu: [0.1; 7],
            sigma: [0.1; 7],
        };
        trajectory.add_state(pad_state, Utc::now(), "mem_0".to_string());
        let point_cloud = analyzer.to_point_cloud(&trajectory);
        assert_eq!(point_cloud.points.len(), 1);
        assert_eq!(point_cloud.dimension, 7);
    }

    #[test]
    fn test_sliding_window() {
        let analyzer = TrajectoryAnalyzer::default();
        let memories = vec![
            create_test_memory(0, "2025-01-01T00:00:00Z", 0.5),
            create_test_memory(1, "2025-01-01T00:30:00Z", 0.6),
            create_test_memory(2, "2025-01-01T01:00:00Z", 0.7),
            create_test_memory(3, "2025-01-01T01:30:00Z", 0.8),
        ];
        let trajectory = analyzer.collect_trajectory(&memories);
        let windows = analyzer.sliding_window(&trajectory, 2);
        assert_eq!(windows.len(), 3); // 4 points, window_size=2 -> 3 windows
    }

    #[test]
    fn test_pattern_classification() {
        assert_eq!(TrajectoryAnalyzer::classify_pattern(0.8, 0.1), "toroidal");
        assert_eq!(TrajectoryAnalyzer::classify_pattern(0.5, 0.2), "balanced");
        assert_eq!(TrajectoryAnalyzer::classify_pattern(0.3, 0.6), "suspicious");
        assert_eq!(TrajectoryAnalyzer::classify_pattern(0.2, 0.1), "sparse");
    }
}
