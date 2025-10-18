//! Orchestrator wiring the Topological Cognitive System crates together
//! with functional logic derived from the original monolithic crate.

use anyhow::Result;
use tcs_consensus::{ConsensusModule, TokenProposal};
use tcs_core::embeddings::EmbeddingBuffer;
use tcs_core::events::{snapshot_event, TopologicalEvent};
use tcs_core::state::CognitiveState;
use tcs_core::{PersistentFeature, StageTimer};
use tcs_knot::{CognitiveKnot, JonesPolynomial, KnotDiagram};
use tcs_ml::{Brain, MotorBrain, UntryingAgent};
use tcs_tda::{PersistenceFeature, PersistentHomology, TakensEmbedding};
use tcs_tqft::FrobeniusAlgebra;
use uuid::Uuid;

/// Core orchestrator coordinating embedding, TDA, knot analysis, RL and consensus.
pub struct TCSOrchestrator {
    buffer: EmbeddingBuffer,
    takens: TakensEmbedding,
    homology: PersistentHomology,
    knot_analyzer: JonesPolynomial,
    rl_agent: UntryingAgent,
    consensus: ConsensusModule,
    tqft: FrobeniusAlgebra,
    state: CognitiveState,
    motor_brain: MotorBrain,
}

impl TCSOrchestrator {
    pub fn new(window: usize) -> Result<Self> {
        let motor_brain = MotorBrain::new()?;
        let takens = TakensEmbedding::new(3, 2, 3);
        Ok(Self {
            buffer: EmbeddingBuffer::new(window),
            takens,
            homology: PersistentHomology::new(2, 2.5),
            knot_analyzer: JonesPolynomial::new(256),
            rl_agent: UntryingAgent::new(),
            consensus: ConsensusModule::new(0.8),
            tqft: FrobeniusAlgebra::new(2),
            state: CognitiveState::default(),
            motor_brain,
        })
    }

    pub fn ingest_sample(&mut self, sample: Vec<f32>) {
        self.buffer.push(sample);
    }

    pub fn ready(&self) -> bool {
        self.buffer.is_ready()
    }

    pub async fn process(&mut self, raw_input: &str) -> Result<Vec<TopologicalEvent>> {
        let mut events = Vec::new();
        if !self.ready() {
            return Ok(events);
        }

        let timer = StageTimer::start();
        let embedding_input = self.buffer.as_slices();

        let embedded = self.takens.embed(&embedding_input);
        if embedded.is_empty() {
            return Ok(events);
        }
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
            emotional_coherence: 0.5,
        };

        if self.consensus.propose(&proposal) {
            events.push(TopologicalEvent::ConsensusReached {
                token_id: proposal.id.to_string(),
            });
        }

        self.state.increment_cycle();
        events.push(snapshot_event(self.state.snapshot()));

        if self.tqft.is_associative() && self.tqft.is_coassociative() {
            // Only log once the algebra checks out.
            let _ = timer.elapsed();
        }

        // Run the motor brain to maintain feature parity with the old pipeline.
        let _brain_output = self.motor_brain.process(raw_input).await?;

        Ok(events)
    }

    fn update_state_from_features(&mut self, features: &[PersistenceFeature], events: &mut Vec<TopologicalEvent>) {
        let mut betti = [0usize; 3];
        for feature in features.iter().take(3) {
            if feature.dimension < betti.len() {
                betti[feature.dimension] += 1;
            }
            if feature.persistence() > 0.1 {
                let pf = PersistentFeature {
                    birth: feature.birth,
                    death: feature.death,
                    dimension: feature.dimension,
                };
                self.state.register_feature(pf.clone());
                events.push(TopologicalEvent::PersistentHomologyDetected { feature: pf });
            }
        }
        self.state.update_betti_numbers(betti);
        self.state.update_metrics(0.6, 0.7);
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
        let CognitiveKnot { complexity_score, .. } = self.knot_analyzer.analyze(&diagram);
        if complexity_score > 1.0 {
            Some(TopologicalEvent::KnotComplexityIncrease {
                new_complexity: complexity_score,
            })
        } else {
            None
        }
    }
}
