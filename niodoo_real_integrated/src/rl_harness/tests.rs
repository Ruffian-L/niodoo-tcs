//! Unit tests for RL Execution Harness

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CodeLanguage;
    use crate::rl_harness::{ExecutionHarness, RewardWeights, TrainingProblem};
    use crate::sandbox::manager::SandboxManager;
    use crate::sandbox::security::SecurityPolicy;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    #[tokio::test]
    async fn test_execution_harness_creation() {
        let security_policy = SecurityPolicy::default();
        let sandbox_manager = Arc::new(
            SandboxManager::new(security_policy)
                .expect("Failed to create sandbox manager")
        );
        let reward_weights = RewardWeights::default();

        let harness = ExecutionHarness::new(
            sandbox_manager,
            None, // No TCSAnalyzer for this test
            reward_weights,
        );

        assert!(harness.is_ok());
    }

    #[tokio::test]
    async fn test_reward_weights_default() {
        let weights = RewardWeights::default();
        assert_eq!(weights.functional, 0.5);
        assert_eq!(weights.cqs, 0.3);
        assert_eq!(weights.topological, 0.2);
        
        // Weights should sum to approximately 1.0
        let total = weights.functional + weights.cqs + weights.topological;
        assert!((total - 1.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_training_problem_creation() {
        let problem = TrainingProblem {
            id: "test_001".to_string(),
            description: "Write a function that adds two numbers".to_string(),
            language: CodeLanguage::Python,
            test_cases: Some(vec![
                "assert add(2, 3) == 5".to_string(),
                "assert add(0, 0) == 0".to_string(),
            ]),
            expected_output: None,
        };

        assert_eq!(problem.id, "test_001");
        assert_eq!(problem.language, CodeLanguage::Python);
        assert!(problem.test_cases.is_some());
        assert_eq!(problem.test_cases.as_ref().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_reward_computation_structure() {
        use crate::rl_harness::reward::ExecutionReward;

        let reward = ExecutionReward {
            functional: 1.0,
            cqs: 0.8,
            topological: 0.9,
            total: 0.9,
        };

        assert_eq!(reward.functional, 1.0);
        assert_eq!(reward.cqs, 0.8);
        assert_eq!(reward.topological, 0.9);
        assert_eq!(reward.total, 0.9);
    }

    #[tokio::test]
    async fn test_reward_zero() {
        use crate::rl_harness::reward::ExecutionReward;

        let reward = ExecutionReward::zero();
        assert_eq!(reward.functional, 0.0);
        assert_eq!(reward.cqs, 0.0);
        assert_eq!(reward.topological, 0.0);
        assert_eq!(reward.total, 0.0);
    }
}


