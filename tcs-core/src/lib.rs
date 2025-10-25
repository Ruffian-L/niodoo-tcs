//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham
//!
//! Core cognitive state management and embedding scaffolding for the
//! Topological Cognitive System workspace. This module centralises the
//! structures that previously lived inside `src/` so other crates can
//! depend on a stable API while we continue the migration.

use std::time::{Duration, Instant};

pub mod metrics;

pub mod events {
    //! Event types emitted by the orchestrator when notable topological
    //! structures appear in the incoming data stream.

    use super::{PersistentFeature, StateSnapshot};

    /// High-level topological events bubbling out of the pipeline.
    #[derive(Debug, Clone)]
    pub enum TopologicalEvent {
        PersistentHomologyDetected { feature: PersistentFeature },
        KnotComplexityIncrease { new_complexity: f32 },
        ConsensusReached { token_id: String },
        StateSnapshot(StateSnapshot),
    }

    /// Convenience helper for constructing snapshot events.
    pub fn snapshot_event(snapshot: StateSnapshot) -> TopologicalEvent {
        TopologicalEvent::StateSnapshot(snapshot)
    }
}

pub mod state {
    //! Cognitive state representation and update helpers.

    use super::{PersistentFeature, StateSnapshot};

    /// Aggregated cognitive state that tracks Betti numbers, active
    /// topological features, and summary metrics.
    #[derive(Debug, Clone)]
    pub struct CognitiveState {
        pub betti_numbers: [usize; 3],
        pub active_features: Vec<PersistentFeature>,
        pub resonance: f32,
        pub coherence: f32,
        pub cycles_processed: u64,
        pub last_updated_ms: u128,
    }

    impl Default for CognitiveState {
        fn default() -> Self {
            Self {
                betti_numbers: [0, 0, 0],
                active_features: Vec::new(),
                resonance: 0.0,
                coherence: 0.0,
                cycles_processed: 0,
                last_updated_ms: 0,
            }
        }
    }

    impl CognitiveState {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn update_betti_numbers(&mut self, betti: [usize; 3]) {
            self.betti_numbers = betti;
        }

        pub fn register_feature(&mut self, feature: PersistentFeature) {
            self.active_features.push(feature);
        }

        pub fn update_metrics(&mut self, resonance: f32, coherence: f32) {
            self.resonance = resonance;
            self.coherence = coherence;
        }

        pub fn increment_cycle(&mut self) {
            self.cycles_processed += 1;
            self.last_updated_ms = chrono::Utc::now().timestamp_millis() as u128;
        }

        pub fn snapshot(&self) -> StateSnapshot {
            StateSnapshot {
                betti_numbers: self.betti_numbers,
                active_features: self.active_features.clone(),
                resonance: self.resonance,
                coherence: self.coherence,
                cycles_processed: self.cycles_processed,
            }
        }
    }
}

pub mod embeddings {
    //! Embedding utilities that wrap streaming buffers before delegating
    //! to the TDA crate for Takens embedding work.

    use std::collections::VecDeque;

    /// Sliding time-series buffer that feeds the Takens embedding step.
    #[derive(Debug, Clone)]
    pub struct EmbeddingBuffer {
        capacity: usize,
        queue: VecDeque<Vec<f32>>,
    }

    impl EmbeddingBuffer {
        pub fn new(capacity: usize) -> Self {
            Self {
                capacity,
                queue: VecDeque::with_capacity(capacity),
            }
        }

        pub fn push(&mut self, sample: Vec<f32>) {
            if self.queue.len() == self.capacity {
                self.queue.pop_front();
            }
            self.queue.push_back(sample);
        }

        pub fn as_slices(&self) -> Vec<Vec<f32>> {
            self.queue.iter().cloned().collect()
        }

        pub fn len(&self) -> usize {
            self.queue.len()
        }

        pub fn is_ready(&self) -> bool {
            self.queue.len() == self.capacity
        }

        /// Clear all buffered embeddings, preserving capacity.
        pub fn clear(&mut self) {
            self.queue.clear();
        }
    }
}

/// Persistent homology feature summary used throughout the pipeline.
#[derive(Debug, Clone)]
pub struct PersistentFeature {
    pub birth: f32,
    pub death: f32,
    pub dimension: usize,
}

impl PersistentFeature {
    pub fn persistence(&self) -> f32 {
        (self.death - self.birth).abs()
    }
}

/// Lightweight snapshot used to broadcast current state to observers.
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    pub betti_numbers: [usize; 3],
    pub active_features: Vec<PersistentFeature>,
    pub resonance: f32,
    pub coherence: f32,
    pub cycles_processed: u64,
}

/// Utility timer used by the orchestrator for measuring stage latency.
#[derive(Debug, Clone)]
pub struct StageTimer {
    start: Instant,
}

impl StageTimer {
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}
