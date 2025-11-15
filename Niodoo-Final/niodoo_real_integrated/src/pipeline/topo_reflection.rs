use nalgebra::Vector3;

use crate::config::env_value;
use crate::erag::CollapseResult;
use crate::tda::{gudhi_bridge, PDPoint};
use crate::torus::PadGhostState;
use crate::{core::topo_math::compute_linking_number, tcs_analysis::TopologicalSignature};
use std::panic::AssertUnwindSafe;

/// Reflection outputs to enrich probabilistic inference
#[derive(Debug, Clone)]
pub struct TopoReflection {
    pub thinking_depth: f64,
    pub pivot_score: f64,
}

/// Compute enriched topological reasoning metrics between Stage O₂ and O₃.
pub struct TopoReflectionStage;

const MIN_THINKING_DEPTH: f64 = 1e-3;

impl TopoReflectionStage {
    pub fn new() -> Self {
        Self
    }

    /// Run reflection with current and baseline topology plus available context.
    pub fn run(
        &self,
        current: &TopologicalSignature,
        baseline: &TopologicalSignature,
        pad_state: &PadGhostState,
        collapse: &CollapseResult,
    ) -> TopoReflection {
        let current_pd = to_pd_points(current);
        let baseline_pd = to_pd_points(baseline);

        let mut thinking_depth = std::panic::catch_unwind(AssertUnwindSafe(|| {
            gudhi_bridge::wasserstein_w2(&current_pd, &baseline_pd)
        }))
        .unwrap_or(f64::NAN);

        if !thinking_depth.is_finite() || thinking_depth <= f64::EPSILON {
            let entropy_delta = (current.persistence_entropy - baseline.persistence_entropy).abs();
            let total_persistence_delta =
                (current.total_persistence - baseline.total_persistence).abs();
            let spectral_gap_delta = (current.spectral_gap - baseline.spectral_gap).abs();
            let mean_persistence_delta =
                (current.mean_persistence - baseline.mean_persistence).abs();
            let max_persistence_delta = (current.max_persistence - baseline.max_persistence).abs();
            let knot_delta = (current.knot_complexity - baseline.knot_complexity).abs();
            let betti_delta: f64 = current
                .betti_numbers
                .iter()
                .zip(baseline.betti_numbers.iter())
                .map(|(lhs, rhs)| (*lhs as f64 - *rhs as f64).abs())
                .sum();
            let sigma_variability: f64 =
                pad_state.sigma.iter().map(|value| value.abs()).sum::<f64>()
                    / pad_state.sigma.len() as f64;

            thinking_depth = [
                entropy_delta,
                total_persistence_delta,
                spectral_gap_delta,
                mean_persistence_delta,
                max_persistence_delta,
                knot_delta,
                betti_delta,
                sigma_variability,
            ]
            .into_iter()
            .filter(|value| value.is_finite())
            .sum::<f64>()
            .max(sigma_variability.abs());
        }

        let signal_contribution =
            relative_delta_array(
                &current.takens_window_energy,
                &baseline.takens_window_energy,
            ) + relative_delta_array(
                &current.emotional_arc_lengths,
                &baseline.emotional_arc_lengths,
            ) + relative_delta_array(
                &current.gradient_energy_trend,
                &baseline.gradient_energy_trend,
            ) + relative_delta_scalar(current.emotional_flux, baseline.emotional_flux)
                + relative_delta_scalar(current.emotional_drift, baseline.emotional_drift)
                + relative_delta_scalar(current.gradient_energy, baseline.gradient_energy)
                + relative_delta_scalar(current.gradient_volatility, baseline.gradient_volatility)
                + relative_delta_scalar(current.takens_energy, baseline.takens_energy)
                + relative_delta_scalar(
                    current.takens_window_count as f64,
                    baseline.takens_window_count as f64,
                );
        if signal_contribution.is_finite() {
            thinking_depth += signal_contribution;
        }

        // Ensure we always nudge the thinking depth above zero so gating can fire.
        if thinking_depth <= f64::EPSILON {
            thinking_depth = pad_state.entropy.abs().max(MIN_THINKING_DEPTH);
        }

        // Build trajectories for pivot score
        let traj_a = closed_loop_from_pad(pad_state);
        let traj_b = memory_traj_from_erag(collapse);
        let pivot_score = if !traj_a.is_empty() && !traj_b.is_empty() {
            compute_linking_number(&traj_a, &traj_b)
        } else {
            0.0
        };

        TopoReflection {
            thinking_depth,
            pivot_score,
        }
    }

    /// Whether to trigger full TopoCoT generation based on threshold.
    pub fn should_trigger_cot(&self, reflection: &TopoReflection) -> bool {
        let threshold = env_value("TOPO_REFLECTION_DEPTH_THRESHOLD")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(MIN_THINKING_DEPTH);
        reflection.thinking_depth >= threshold
    }

    /// Build a human-readable summary for failure cases that captures topological drift.
    pub fn summarize_failure(
        current: &TopologicalSignature,
        baseline: &TopologicalSignature,
        reflection: &TopoReflection,
    ) -> String {
        let betti = current.betti_numbers;
        let baseline_betti = baseline.betti_numbers;
        let delta_b1 = betti[1] as isize - baseline_betti[1] as isize;
        let delta_gap = current.spectral_gap - baseline.spectral_gap;
        let delta_entropy = current.persistence_entropy - baseline.persistence_entropy;
        let label = if delta_b1 > 0 && delta_gap < 0.0 {
            "chaotic"
        } else if delta_b1 < 0 && delta_gap > 0.0 {
            "stabilising"
        } else {
            "adrift"
        };

        format!(
            "label={label} betti={:?} (Δβ1={:+}) gap={:.3}->{:.3} entropy={:.3}->{:.3} (Δ={:.3}) thinking_depth={:.3} pivot={:.3}; action=slow_down_structure_proof",
            betti,
            delta_b1,
            baseline.spectral_gap,
            current.spectral_gap,
            baseline.persistence_entropy,
            current.persistence_entropy,
            delta_entropy,
            reflection.thinking_depth,
            reflection.pivot_score
        )
    }
}

fn to_pd_points(sig: &TopologicalSignature) -> Vec<PDPoint> {
    sig.persistence_features
        .iter()
        .map(|f| PDPoint {
            birth: f.birth as f64,
            death: f.death as f64,
        })
        .collect()
}

fn closed_loop_from_pad(pad: &PadGhostState) -> Vec<Vector3<f64>> {
    // Build a small closed loop in 3D PAD subspace around the current point
    let center = Vector3::new(pad.pad[0], pad.pad[1], pad.pad[2]);
    let r = 0.05_f64;
    let steps = 16usize;
    let mut points = Vec::with_capacity(steps);
    for k in 0..steps {
        let t = (k as f64) / (steps as f64) * std::f64::consts::TAU;
        let offset = Vector3::new(r * t.cos(), r * t.sin(), r * 0.5 * (2.0 * t).sin());
        points.push(center + offset);
    }
    points
}

fn memory_traj_from_erag(collapse: &CollapseResult) -> Vec<Vector3<f64>> {
    if collapse.top_hits.is_empty() {
        return Vec::new();
    }
    // Reconstruct a loop from memory emotional vectors (proxy trajectory)
    let mut points = Vec::new();
    let r = 0.05_f64;
    for (k, mem) in collapse.top_hits.iter().take(16).enumerate() {
        let base = Vector3::new(
            mem.emotional_vector.joy as f64,
            mem.emotional_vector.anger as f64,
            mem.emotional_vector.surprise as f64,
        );
        let t = (k as f64) / 16.0 * std::f64::consts::TAU;
        let offset = Vector3::new(
            r * (t + 0.7).cos(),
            r * (t + 0.3).sin(),
            r * 0.5 * (t + 1.1).sin(),
        );
        points.push(base + offset);
    }
    points
}

fn relative_delta_scalar(current: f64, baseline: f64) -> f64 {
    let numerator = (current - baseline).abs();
    let denominator = current.abs() + baseline.abs() + f64::EPSILON;
    (numerator / denominator).min(1.0)
}

fn relative_delta_array(current: &[f64], baseline: &[f64]) -> f64 {
    if current.is_empty() && baseline.is_empty() {
        return 0.0;
    }
    let len = current.len().max(baseline.len());
    if len == 0 {
        return 0.0;
    }
    let mut total = 0.0;
    for index in 0..len {
        let lhs = current.get(index).copied().unwrap_or(0.0);
        let rhs = baseline.get(index).copied().unwrap_or(0.0);
        total += relative_delta_scalar(lhs, rhs);
    }
    total / len as f64
}
