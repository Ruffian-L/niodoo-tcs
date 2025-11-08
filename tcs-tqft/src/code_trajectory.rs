//! Code Trajectory Module for TQFT Reasoning
//!
//! This module defines the "code trajectory" concept - a temporal sequence
//! representing code evolution that replaces emotional state trajectories
//! in the Thought-Knot (tce-tqft) module.
//!
//! A code trajectory can represent:
//! - CFG path: Path through a Control Flow Graph
//! - DFG path: Path through a Data Flow Graph  
//! - Commit sequence: Temporal sequence of developer commits and file changes
//! - Execution trace: Literal execution trace from a profiler

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Type of code trajectory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TrajectoryType {
    /// Control Flow Graph path
    CfgPath,
    /// Data Flow Graph path
    DfgPath,
    /// Commit sequence (temporal file changes)
    CommitSequence,
    /// Execution trace from profiler
    ExecutionTrace,
}

/// A single point in a code trajectory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryPoint {
    /// Timestamp (for temporal trajectories) or sequence index
    pub t: f64,
    /// Betti numbers at this point: [β₀, β₁, β₂]
    pub betti: [usize; 3],
    /// Optional metadata (node IDs, commit hash, etc.)
    pub metadata: HashMap<String, String>,
}

/// Code trajectory representing temporal code evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeTrajectory {
    /// Type of trajectory
    pub trajectory_type: TrajectoryType,
    /// Sequence of trajectory points
    pub points: Vec<TrajectoryPoint>,
    /// Source code file path (if applicable)
    pub source_path: Option<String>,
}

impl CodeTrajectory {
    /// Create a new code trajectory
    pub fn new(trajectory_type: TrajectoryType) -> Self {
        Self {
            trajectory_type,
            points: Vec::new(),
            source_path: None,
        }
    }

    /// Add a point to the trajectory
    pub fn add_point(&mut self, t: f64, betti: [usize; 3], metadata: HashMap<String, String>) {
        self.points.push(TrajectoryPoint { t, betti, metadata });
        // Keep points sorted by time
        self.points.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Compute Betti derivatives (dBetti/dt) for each dimension
    /// Returns vector of (dt, dβ₀/dt, dβ₁/dt, dβ₂/dt) tuples
    pub fn compute_betti_derivatives(&self) -> Vec<(f64, f64, f64, f64)> {
        if self.points.len() < 2 {
            return Vec::new();
        }

        let mut derivatives = Vec::new();
        for i in 1..self.points.len() {
            let prev = &self.points[i - 1];
            let curr = &self.points[i];

            let dt = curr.t - prev.t;
            if dt.abs() < 1e-10 {
                continue; // Skip zero-duration intervals
            }

            let d_b0 = (curr.betti[0] as f64 - prev.betti[0] as f64) / dt;
            let d_b1 = (curr.betti[1] as f64 - prev.betti[1] as f64) / dt;
            let d_b2 = (curr.betti[2] as f64 - prev.betti[2] as f64) / dt;

            derivatives.push((dt, d_b0, d_b1, d_b2));
        }

        derivatives
    }

    /// Compute the norm of Betti derivatives: ||dBetti/dt||
    /// This is the key metric for detecting "thought-knots" (persistent Betti-1 loops)
    pub fn compute_betti_derivative_norm(&self) -> f64 {
        let derivatives = self.compute_betti_derivatives();
        if derivatives.is_empty() {
            return 0.0;
        }

        // Compute L2 norm across all dimensions and time points
        let mut norm_squared = 0.0;
        for (_, d_b0, d_b1, d_b2) in &derivatives {
            norm_squared += d_b0 * d_b0 + d_b1 * d_b1 + d_b2 * d_b2;
        }

        (norm_squared / derivatives.len() as f64).sqrt()
    }

    /// Get Betti numbers at a specific time point (interpolated if needed)
    pub fn betti_at_time(&self, t: f64) -> Option<[usize; 3]> {
        if self.points.is_empty() {
            return None;
        }

        // Exact match
        for point in &self.points {
            if (point.t - t).abs() < 1e-10 {
                return Some(point.betti);
            }
        }

        // Interpolate between points
        for i in 1..self.points.len() {
            let prev = &self.points[i - 1];
            let curr = &self.points[i];

            if t >= prev.t && t <= curr.t {
                let alpha = (t - prev.t) / (curr.t - prev.t);
                let b0 = prev.betti[0] as f64 + alpha * (curr.betti[0] as f64 - prev.betti[0] as f64);
                let b1 = prev.betti[1] as f64 + alpha * (curr.betti[1] as f64 - prev.betti[1] as f64);
                let b2 = prev.betti[2] as f64 + alpha * (curr.betti[2] as f64 - prev.betti[2] as f64);
                return Some([b0.round() as usize, b1.round() as usize, b2.round() as usize]);
            }
        }

        // Extrapolate from first or last point
        if t < self.points[0].t {
            Some(self.points[0].betti)
        } else {
            Some(self.points[self.points.len() - 1].betti)
        }
    }

    /// Detect "thought-knots" - persistent Betti-1 loops
    /// Returns true if ||dBetti/dt|| → 0 (stuck state) or if β₁ is persistently high
    pub fn detect_thought_knot(&self, threshold: f64) -> bool {
        let norm = self.compute_betti_derivative_norm();
        
        // Check if derivative norm is near zero (stuck state)
        if norm < threshold {
            return true;
        }

        // Check if Betti-1 is persistently high (indicating cycles/loops)
        if let Some(avg_b1) = self.average_betti_1() {
            if avg_b1 > 5.0 {
                return true;
            }
        }

        false
    }

    /// Compute average Betti-1 across trajectory
    fn average_betti_1(&self) -> Option<f64> {
        if self.points.is_empty() {
            return None;
        }
        let sum: usize = self.points.iter().map(|p| p.betti[1]).sum();
        Some(sum as f64 / self.points.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trajectory_creation() {
        let mut traj = CodeTrajectory::new(TrajectoryType::CfgPath);
        assert_eq!(traj.points.len(), 0);
        assert_eq!(traj.trajectory_type, TrajectoryType::CfgPath);
    }

    #[test]
    fn test_add_points() {
        let mut traj = CodeTrajectory::new(TrajectoryType::CommitSequence);
        
        let mut meta1 = HashMap::new();
        meta1.insert("commit".to_string(), "abc123".to_string());
        traj.add_point(0.0, [1, 0, 0], meta1);

        let mut meta2 = HashMap::new();
        meta2.insert("commit".to_string(), "def456".to_string());
        traj.add_point(1.0, [2, 1, 0], meta2);

        assert_eq!(traj.points.len(), 2);
        assert_eq!(traj.points[0].betti, [1, 0, 0]);
        assert_eq!(traj.points[1].betti, [2, 1, 0]);
    }

    #[test]
    fn test_betti_derivatives() {
        let mut traj = CodeTrajectory::new(TrajectoryType::ExecutionTrace);
        traj.add_point(0.0, [1, 0, 0], HashMap::new());
        traj.add_point(1.0, [2, 1, 0], HashMap::new());
        traj.add_point(2.0, [2, 2, 0], HashMap::new());

        let derivatives = traj.compute_betti_derivatives();
        assert_eq!(derivatives.len(), 2);
        
        // First interval: dt=1.0, dβ₀/dt=1.0, dβ₁/dt=1.0, dβ₂/dt=0.0
        assert!((derivatives[0].0 - 1.0).abs() < 1e-5);
        assert!((derivatives[0].1 - 1.0).abs() < 1e-5);
        assert!((derivatives[0].2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_betti_derivative_norm() {
        let mut traj = CodeTrajectory::new(TrajectoryType::CfgPath);
        traj.add_point(0.0, [1, 0, 0], HashMap::new());
        traj.add_point(1.0, [2, 1, 0], HashMap::new());

        let norm = traj.compute_betti_derivative_norm();
        // ||(1, 1, 0)|| = sqrt(1² + 1² + 0²) = sqrt(2)
        assert!((norm - (2.0_f64).sqrt()).abs() < 1e-5);
    }

    #[test]
    fn test_thought_knot_detection() {
        let mut traj = CodeTrajectory::new(TrajectoryType::DfgPath);
        // Stuck state: Betti numbers don't change
        traj.add_point(0.0, [1, 1, 0], HashMap::new());
        traj.add_point(1.0, [1, 1, 0], HashMap::new());
        traj.add_point(2.0, [1, 1, 0], HashMap::new());

        assert!(traj.detect_thought_knot(0.1));
    }
}

