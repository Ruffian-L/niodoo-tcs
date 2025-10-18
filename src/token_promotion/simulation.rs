use std::collections::HashSet;
use std::time::Instant;

use chrono::Utc;

use crate::memory::guessing_spheres::{
    EmotionalVector, GuessingMemorySystem, MemoryQuery, SphereId,
};

/// Result of a standalone promotion simulation cycle.
#[derive(Debug, Clone)]
pub struct PromotionResult {
    pub promoted_count: usize,
    pub pruned_count: usize,
    pub cycle_latency_ms: f64,
}

const PROMOTION_THRESHOLD: f32 = 0.55;
const PRUNE_THRESHOLD: f32 = 0.25;

/// Run a lightweight promotion simulation directly against the Gaussian memory system.
pub fn run_promotion_cycle(memory_system: &mut GuessingMemorySystem) -> PromotionResult {
    let cycle_start = Instant::now();
    let total_spheres = memory_system.sphere_count();

    if total_spheres == 0 {
        return PromotionResult {
            promoted_count: 0,
            pruned_count: 0,
            cycle_latency_ms: cycle_start.elapsed().as_secs_f64() * 1000.0,
        };
    }

    let mut aggregate = [0.0_f32; 5];
    for sphere in memory_system.spheres() {
        let emo = sphere.emotional_profile.as_array();
        for (acc, value) in aggregate.iter_mut().zip(emo.iter()) {
            *acc += *value;
        }
    }

    for value in &mut aggregate {
        *value /= total_spheres as f32;
    }

    if aggregate.iter().all(|value| value.abs() <= f32::EPSILON) {
        aggregate.fill(0.2);
    }

    let probe_emotion = EmotionalVector::new(
        aggregate[0],
        aggregate[1],
        aggregate[2],
        aggregate[3],
        aggregate[4],
    );

    let probe_query = MemoryQuery {
        concept: "promotion_probe".to_string(),
        emotion: probe_emotion,
        time: Utc::now().timestamp() as f64,
    };

    let recall = memory_system.collapse_recall_probability(&probe_query);

    let mut promoted: HashSet<SphereId> = HashSet::new();
    let mut pruned: HashSet<SphereId> = HashSet::new();

    for (sphere_id, score) in recall {
        if score >= PROMOTION_THRESHOLD {
            promoted.insert(sphere_id);
        } else if score <= PRUNE_THRESHOLD {
            pruned.insert(sphere_id);
        }
    }

    for sphere in memory_system.spheres_mut() {
        if promoted.contains(&sphere.id) {
            for i in 0..3 {
                sphere.covariance[i][i] *= 0.9;
            }
        } else if pruned.contains(&sphere.id) {
            for i in 0..3 {
                sphere.covariance[i][i] *= 1.1;
            }
        }
    }

    let promoted_count = promoted.len().min(total_spheres);
    let pruned_count = pruned
        .len()
        .min(total_spheres.saturating_sub(promoted_count));

    PromotionResult {
        promoted_count,
        pruned_count,
        cycle_latency_ms: cycle_start.elapsed().as_secs_f64() * 1000.0,
    }
}
