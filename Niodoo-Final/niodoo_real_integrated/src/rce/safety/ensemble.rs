use crate::config::RceConsensusConfig;

/// Minimal consensus gate: collects boolean approvals and enforces quorum.
pub struct ConsensusGate {
    cfg: RceConsensusConfig,
}

impl ConsensusGate {
    pub fn new(cfg: RceConsensusConfig) -> Self {
        Self { cfg }
    }

    pub fn approve(&self, votes: &[bool]) -> bool {
        if !self.cfg.enabled {
            return true;
        }
        let approvals = votes.iter().filter(|&&v| v).count();
        approvals >= self.cfg.quorum.min(self.cfg.analyzers).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quorum_approves() {
        let cfg = RceConsensusConfig {
            enabled: true,
            analyzers: 3,
            quorum: 2,
        };
        let gate = ConsensusGate::new(cfg);
        assert!(gate.approve(&[true, true, false]));
        assert!(!gate.approve(&[true, false, false]));
    }

    #[test]
    fn test_disabled_always_approves() {
        let cfg = RceConsensusConfig {
            enabled: false,
            analyzers: 5,
            quorum: 3,
        };
        let gate = ConsensusGate::new(cfg);
        assert!(gate.approve(&[false, false, false, false, false]));
    }
}
