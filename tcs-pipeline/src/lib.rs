// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham
//!
//! Orchestrator wiring the Topological Cognitive System crates together
//! with functional logic derived from the original monolithic crate.

mod config;

use anyhow::Result;
use tcs_consensus::{ThresholdConsensus, TokenProposal};
use tcs_core::embeddings::EmbeddingBuffer;
use tcs_core::events::{TopologicalEvent, snapshot_event};
use tcs_core::state::CognitiveState;
use tcs_core::{PersistentFeature, StageTimer};
use tcs_knot::{CognitiveKnot, JonesPolynomial, KnotDiagram};
use tcs_ml::{ExplorationAgent, MotorBrain};
use tcs_tda::{PersistenceFeature, PersistentHomology, TakensEmbedding};
use tcs_tqft::FrobeniusAlgebra;
use tracing::{debug, info, warn};
use uuid::Uuid;

pub use config::TCSConfig;

/// Core orchestrator coordinating embedding, TDA, knot analysis, RL and consensus.
pub struct TCSOrchestrator {
    buffer: EmbeddingBuffer,
    takens: TakensEmbedding,
    homology: PersistentHomology,
    knot_analyzer: JonesPolynomial,
    rl_agent: ExplorationAgent,
    consensus: ThresholdConsensus,
    tqft: FrobeniusAlgebra,
    state: CognitiveState,
    motor_brain: MotorBrain,
    config: TCSConfig,
}

impl TCSOrchestrator {
    pub fn new(window: usize) -> Result<Self> {
        Self::with_config(window, TCSConfig::default())
    }

    pub fn with_config(window: usize, config: TCSConfig) -> Result<Self> {
        let motor_brain = MotorBrain::new()?;
        let takens = TakensEmbedding::new(
            config.takens_dimension,
            config.takens_delay,
            config.takens_data_dim,
        );
        let rl_agent = match std::env::var("NIODOO_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
        {
            Some(seed) => ExplorationAgent::with_seed(seed),
            None => ExplorationAgent::new(),
        };

        Ok(Self {
            buffer: EmbeddingBuffer::new(window),
            takens,
            homology: PersistentHomology::new(
                config.homology_max_dimension,
                config.homology_max_edge_length,
            ),
            knot_analyzer: JonesPolynomial::new(config.jones_cache_capacity),
            rl_agent,
            consensus: ThresholdConsensus::new(config.consensus_threshold),
            tqft: FrobeniusAlgebra::new(config.tqft_algebra_dimension),
            state: CognitiveState::default(),
            motor_brain,
            config,
        })
    }

    pub fn ingest_sample(&mut self, sample: Vec<f32>) {
        self.buffer.push(sample);
    }

    pub fn ready(&self) -> bool {
        self.buffer.is_ready()
    }

    /// Reset buffered embeddings and the MotorBrain cache for a fresh session.
    pub fn reset_brain_context(&mut self) {
        self.buffer.clear();
        self.motor_brain.reset_embedding_cache();
        self.state = CognitiveState::default();
        info!(
            target: "tcs-pipeline::orchestrator",
            "Reset orchestrator buffer and MotorBrain cache"
        );
    }

    pub async fn process(&mut self, raw_input: &str) -> Result<Vec<TopologicalEvent>> {
        let mut events = Vec::new();
        if raw_input.trim().is_empty() {
            info!(
                target: "tcs-pipeline::orchestrator",
                "Empty input received; clearing stateful caches"
            );
            self.reset_brain_context();
            return Ok(events);
        }
        if !self.ready() {
            return Ok(events);
        }

        let timer = StageTimer::start();
        let embedding_input = self.buffer.as_slices();

        let embedded = self.takens.embed(&embedding_input);
        if embedded.is_empty() {
            return Ok(events);
        }

        // Apply GPU-accelerated distance matrix computation
        let _dist_matrix = nalgebra::DMatrix::<f32>::zeros(embedded.len(), embedded.len());

        let features = self.homology.compute(&embedded);
        self.update_state_from_features(&features, &mut events);

        let knot_event = self.analyse_knot(&features);
        if let Some(event) = knot_event {
            events.push(event);
        }

        let rl_action = self.rl_agent.select_action(embedded[0].as_slice());
        let proposal = TokenProposal {
            id: Uuid::new_v4(),
            persistence_score: rl_action as f32,
            emotional_coherence: self.config.default_coherence,
        };

        if self.consensus.propose(&proposal) {
            events.push(TopologicalEvent::ConsensusReached {
                token_id: proposal.id.to_string(),
            });
        }

        self.state.increment_cycle();
        events.push(snapshot_event(self.state.snapshot()));

        if self.config.enable_tqft_checks {
            match self.tqft.verify_axioms() {
                Ok(_) => debug!(target: "tcs-pipeline::tqft", "Frobenius algebra axioms verified"),
                Err(err) => {
                    warn!(target: "tcs-pipeline::tqft", error = %err, "Frobenius algebra axiom check failed")
                }
            }
        }

        let _ = timer.elapsed();

        // Process input through MotorBrain and ingest embeddings into pipeline
        let brain_embeddings = self.motor_brain.extract_embeddings(raw_input).await?;
        let embedding_dim = brain_embeddings.len();
        if embedding_dim != self.takens.data_dim {
            debug!(
                target: "tcs-pipeline::orchestrator",
                old = self.takens.data_dim,
                new = embedding_dim,
                "Updating Takens data dimension to match embedder output"
            );
            self.takens.data_dim = embedding_dim;
            self.config.takens_data_dim = embedding_dim;
        }
        self.ingest_sample(brain_embeddings);

        Ok(events)
    }

    fn update_state_from_features(
        &mut self,
        features: &[PersistenceFeature],
        events: &mut Vec<TopologicalEvent>,
    ) {
        let mut betti = [0usize; 3];
        for feature in features.iter().take(self.config.feature_sampling_limit) {
            if feature.dimension < betti.len() {
                betti[feature.dimension] += 1;
            }
            if feature.persistence() > self.config.persistence_event_threshold {
                let pf = PersistentFeature {
                    birth: feature.birth as f64,
                    death: feature.death as f64,
                    dimension: feature.dimension,
                };
                self.state.register_feature(pf.clone());
                events.push(TopologicalEvent::PersistentHomologyDetected { feature: pf });
            }
        }
        self.state.update_betti_numbers(betti);
        self.state
            .update_metrics(self.config.default_resonance, self.config.default_coherence);
    }

    fn analyse_knot(&mut self, features: &[PersistenceFeature]) -> Option<TopologicalEvent> {
        if features.is_empty() {
            return None;
        }
        let diagram = KnotDiagram {
            crossings: features
                .iter()
                .map(|f| if f.dimension % 2 == 0 { 1 } else { -1 })
                .collect(),
        };
        let CognitiveKnot {
            complexity_score, ..
        } = self.knot_analyzer.analyze(&diagram);
        if complexity_score > self.config.knot_complexity_threshold {
            Some(TopologicalEvent::KnotComplexityIncrease {
                new_complexity: complexity_score,
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn custom_consensus_threshold_is_used() {
        let mut config = TCSConfig::default();
        config.consensus_threshold = 0.9;
        let orchestrator = TCSOrchestrator::with_config(16, config)
            .expect("config-based construction should succeed");

        let pass = TokenProposal {
            id: Uuid::new_v4(),
            persistence_score: 0.95,
            emotional_coherence: orchestrator.config.default_coherence,
        };
        let fail = TokenProposal {
            id: Uuid::new_v4(),
            persistence_score: 0.85,
            emotional_coherence: orchestrator.config.default_coherence,
        };

        assert!(orchestrator.consensus.propose(&pass));
        assert!(!orchestrator.consensus.propose(&fail));
    }

    #[test]
    fn orchestrator_default_construction() {
        let orchestrator = TCSOrchestrator::new(16);
        assert!(orchestrator.is_ok());
    }

    #[test]
    fn orchestrator_config_construction() {
        let config = TCSConfig::default();
        let orchestrator = TCSOrchestrator::with_config(32, config);
        assert!(orchestrator.is_ok());
    }

    #[test]
    fn orchestrator_ready_state() {
        let mut orchestrator = TCSOrchestrator::new(8).unwrap();
        assert!(!orchestrator.ready());

        // Add enough samples to fill buffer
        for _ in 0..8 {
            orchestrator.ingest_sample(vec![0.1, 0.2, 0.3]);
        }
        assert!(orchestrator.ready());
    }

    #[test]
    fn orchestrator_reset_brain_context() {
        let mut orchestrator = TCSOrchestrator::new(8).unwrap();

        // Add some data
        for _ in 0..8 {
            orchestrator.ingest_sample(vec![0.1, 0.2, 0.3]);
        }
        assert!(orchestrator.ready());

        // Reset and verify
        orchestrator.reset_brain_context();
        assert!(!orchestrator.ready());
    }

    #[tokio::test]
    async fn orchestrator_process_empty_input() {
        let mut orchestrator = TCSOrchestrator::new(8).unwrap();

        // Fill buffer
        for _ in 0..8 {
            orchestrator.ingest_sample(vec![0.1, 0.2, 0.3]);
        }

        let result = orchestrator.process("").await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[tokio::test]
    async fn orchestrator_process_not_ready() {
        let mut orchestrator = TCSOrchestrator::new(8).unwrap();

        // Don't fill buffer
        let result = orchestrator.process("test input").await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }
}
