//! NIODOO TQFT Module - Thought-Knot Detection
//!
//! Applies knot theory (Jones polynomial) to analyze code execution trajectories
//! and detect persistent Betti-1 loops (cyclical dependencies) that span multiple files.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tcs_tqft::{CodeTrajectory, TrajectoryPoint, TrajectoryType};
use tracing::info;

/// Knot signature identifying architectural flaws
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnotSignature {
    /// Whether a thought-knot was detected
    pub has_knot: bool,
    /// Betti derivative norm: ||dBetti/dt||
    pub betti_derivative_norm: f64,
    /// Average Betti-1 (loops) across trajectory
    pub average_betti_1: f64,
    /// Persistence pairs indicating cyclical dependencies
    pub persistent_loops: Vec<(f64, f64)>, // (birth_time, death_time) for persistent loops
    /// Files/modules involved in the knot
    pub involved_modules: Vec<String>,
}

impl KnotSignature {
    pub fn new(
        has_knot: bool,
        betti_derivative_norm: f64,
        average_betti_1: f64,
        persistent_loops: Vec<(f64, f64)>,
        involved_modules: Vec<String>,
    ) -> Self {
        Self {
            has_knot,
            betti_derivative_norm,
            average_betti_1,
            persistent_loops,
            involved_modules,
        }
    }
}

/// TQFT analyzer for thought-knot detection
pub struct TQFTAnalyzer {
    /// Threshold for detecting stuck states (thought-knots)
    knot_threshold: f64,
}

impl TQFTAnalyzer {
    /// Create a new TQFT analyzer
    pub fn new(knot_threshold: f64) -> Self {
        Self { knot_threshold }
    }

    /// Analyze code trajectory and detect thought-knots
    /// 
    /// This implements the "Thought-Knot" pipeline:
    /// - Applies knot theory to code execution trajectories
    /// - Detects persistent Betti-1 loops (cyclical dependencies)
    /// - Identifies architectural flaws invisible to traditional static analysis
    pub fn analyze_trajectory(&self, trajectory: &CodeTrajectory) -> Result<KnotSignature> {
        info!(
            trajectory_type = ?trajectory.trajectory_type,
            num_points = trajectory.points.len(),
            "Analyzing code trajectory for thought-knots"
        );

        // Compute Betti derivative norm
        let betti_derivative_norm = trajectory.compute_betti_derivative_norm();

        // Compute average Betti-1
        let average_betti_1 = trajectory
            .points
            .iter()
            .map(|p| p.betti[1] as f64)
            .sum::<f64>()
            / trajectory.points.len().max(1) as f64;

        // Detect persistent loops (Betti-1 features that persist across trajectory)
        let persistent_loops = self.detect_persistent_loops(trajectory);

        // Extract involved modules from trajectory metadata
        let involved_modules = self.extract_involved_modules(trajectory);

        // Detect thought-knot using threshold
        let has_knot = trajectory.detect_thought_knot(self.knot_threshold)
            || average_betti_1 > 5.0; // High Betti-1 indicates cycles

        let signature = KnotSignature::new(
            has_knot,
            betti_derivative_norm,
            average_betti_1,
            persistent_loops,
            involved_modules,
        );

        if has_knot {
            info!(
                betti_derivative_norm = signature.betti_derivative_norm,
                average_betti_1 = signature.average_betti_1,
                persistent_loops = signature.persistent_loops.len(),
                "Thought-knot detected in code trajectory"
            );
        }

        Ok(signature)
    }

    /// Detect persistent loops (Betti-1 features that persist across trajectory)
    fn detect_persistent_loops(&self, trajectory: &CodeTrajectory) -> Vec<(f64, f64)> {
        let mut loops = Vec::new();

        // Track when Betti-1 features appear and disappear
        let mut loop_birth: Option<f64> = None;

        for point in &trajectory.points {
            if point.betti[1] > 0 {
                // Loop exists at this point
                if loop_birth.is_none() {
                    loop_birth = Some(point.t);
                }
            } else {
                // Loop disappeared
                if let Some(birth) = loop_birth {
                    let death = point.t;
                    let persistence = death - birth;
                    // Only include loops that persist for significant duration
                    if persistence > 0.1 {
                        loops.push((birth, death));
                    }
                    loop_birth = None;
                }
            }
        }

        // If loop persists to end of trajectory, mark it
        if let Some(birth) = loop_birth {
            if let Some(last_point) = trajectory.points.last() {
                let death = last_point.t;
                let persistence = death - birth;
                if persistence > 0.1 {
                    loops.push((birth, death));
                }
            }
        }

        loops
    }

    /// Extract involved modules from trajectory metadata
    fn extract_involved_modules(&self, trajectory: &CodeTrajectory) -> Vec<String> {
        let mut modules = std::collections::HashSet::new();

        for point in &trajectory.points {
            // Extract module/file information from metadata
            if let Some(file) = point.metadata.get("file") {
                modules.insert(file.clone());
            }
            if let Some(module) = point.metadata.get("module") {
                modules.insert(module.clone());
            }
            if let Some(path) = point.metadata.get("path") {
                modules.insert(path.clone());
            }
        }

        // Also check source_path
        if let Some(ref source_path) = trajectory.source_path {
            modules.insert(source_path.clone());
        }

        modules.into_iter().collect()
    }
}

impl Default for TQFTAnalyzer {
    fn default() -> Self {
        Self::new(0.1) // Default threshold
    }
}

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::types::{PyDict, PyModule};
#[cfg(feature = "pyo3")]
use pyo3::Bound;

#[cfg(feature = "pyo3")]
#[pyfunction]
fn analyze_trajectory(
    py: Python,
    trajectory_data: &Bound<PyDict>,
) -> PyResult<PyObject> {
        // Parse trajectory data from Python dict
        let trajectory_type_str: String = trajectory_data
            .get_item("trajectory_type")?
            .and_then(|v| v.extract().ok())
            .unwrap_or_else(|| "CfgPath".to_string());

        let trajectory_type = match trajectory_type_str.as_str() {
            "CfgPath" => TrajectoryType::CfgPath,
            "DfgPath" => TrajectoryType::DfgPath,
            "CommitSequence" => TrajectoryType::CommitSequence,
            "ExecutionTrace" => TrajectoryType::ExecutionTrace,
            _ => TrajectoryType::CfgPath,
        };

        let mut trajectory = CodeTrajectory::new(trajectory_type);

        // Parse points
        if let Ok(Some(points_list)) = trajectory_data.get_item("points") {
            if let Ok(points) = points_list.extract::<Vec<HashMap<String, PyObject>>>() {
                for point_dict in points {
                    let t: f64 = point_dict
                        .get("t")
                        .and_then(|v| v.extract(py).ok())
                        .unwrap_or(0.0);
                    let betti_0: usize = point_dict
                        .get("betti_0")
                        .and_then(|v| v.extract(py).ok())
                        .unwrap_or(0);
                    let betti_1: usize = point_dict
                        .get("betti_1")
                        .and_then(|v| v.extract(py).ok())
                        .unwrap_or(0);
                    let betti_2: usize = point_dict
                        .get("betti_2")
                        .and_then(|v| v.extract(py).ok())
                        .unwrap_or(0);

                    let mut metadata = HashMap::new();
                    if let Some(meta_dict) = point_dict.get("metadata") {
                        if let Ok(meta) = meta_dict.extract::<HashMap<String, String>>(py) {
                            metadata = meta;
                        }
                    }

                    trajectory.add_point(t, [betti_0, betti_1, betti_2], metadata);
                }
            }
        }

        // Analyze trajectory
        let analyzer = TQFTAnalyzer::default();
        let signature = analyzer.analyze_trajectory(&trajectory)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;

        // Convert to Python dict
        let result = PyDict::new_bound(py);
        result.set_item("has_knot", signature.has_knot)?;
        result.set_item("betti_derivative_norm", signature.betti_derivative_norm)?;
        result.set_item("average_betti_1", signature.average_betti_1)?;
        result.set_item("persistent_loops", signature.persistent_loops)?;
        result.set_item("involved_modules", signature.involved_modules)?;

        Ok(result.to_object(py))
}

#[cfg(feature = "pyo3")]
#[pymodule]
pub fn tqft(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::wrap_pyfunction;
    m.add_function(wrap_pyfunction!(analyze_trajectory, m)?)?;
    Ok(())
}

