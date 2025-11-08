use crate::config::RceConsensusConfig;
use crate::constitutional::violations::{Violation, ViolationSeverity};

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

    /// Approve or reject generated code based on constitutional violations and topological metrics
    pub fn approve_code(
        &self,
        violations: &[Violation],
        topological_complexity: Option<f64>, // e.g., cyclomatic complexity or Betti numbers
    ) -> bool {
        if !self.cfg.enabled {
            return true;
        }

        let mut votes = Vec::new();

        // Vote 1: No high-severity violations
        let has_high_severity = violations
            .iter()
            .any(|v| matches!(v.severity, ViolationSeverity::High));
        votes.push(!has_high_severity);

        // Vote 2: Low topological complexity (if provided)
        if let Some(complexity) = topological_complexity {
            // Threshold: complexity < 20 is acceptable
            votes.push(complexity < 20.0);
        }

        // Vote 3: No medium-severity violations (or very few)
        let medium_severity_count = violations
            .iter()
            .filter(|v| matches!(v.severity, ViolationSeverity::Medium))
            .count();
        votes.push(medium_severity_count <= 2); // Allow up to 2 medium-severity violations

        self.approve(&votes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quorum_approves() {
        let cfg = RceConsensusConfig { enabled: true, analyzers: 3, quorum: 2 };
        let gate = ConsensusGate::new(cfg);
        assert!(gate.approve(&[true, true, false]));
        assert!(!gate.approve(&[true, false, false]));
    }

    #[test]
    fn test_disabled_always_approves() {
        let cfg = RceConsensusConfig { enabled: false, analyzers: 5, quorum: 3 };
        let gate = ConsensusGate::new(cfg);
        assert!(gate.approve(&[false, false, false, false, false]));
    }
}


