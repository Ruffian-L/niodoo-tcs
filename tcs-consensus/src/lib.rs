//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham
//!
//! Consensus and shared vocabulary scaffolding for the Topological Cognitive System.

use uuid::Uuid;

/// Placeholder token proposal structure.
#[derive(Debug, Clone)]
pub struct TokenProposal {
    pub id: Uuid,
    pub persistence_score: f32,
    pub emotional_coherence: f32,
}

/// Single-node threshold-based acceptance helper for prototype pipelines.
pub struct ThresholdConsensus {
    threshold: f32,
}

impl ThresholdConsensus {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    pub fn propose(&self, proposal: &TokenProposal) -> bool {
        proposal.persistence_score >= self.threshold
    }
}

#[deprecated = "Use ThresholdConsensus; this alias remains during the transition away from the ConsensusModule name."]
pub type ConsensusModule = ThresholdConsensus;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn honors_threshold() {
        let module = ThresholdConsensus::new(0.8);
        let accept = TokenProposal {
            id: Uuid::new_v4(),
            persistence_score: 0.85,
            emotional_coherence: 0.5,
        };
        let reject = TokenProposal {
            id: Uuid::new_v4(),
            persistence_score: 0.65,
            emotional_coherence: 0.5,
        };

        assert!(module.propose(&accept));
        assert!(!module.propose(&reject));
    }
}
