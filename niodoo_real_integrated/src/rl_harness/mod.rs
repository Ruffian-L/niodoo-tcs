//! RL Execution Harness for Reinforcement Learning from Execution Feedback (RLEF)
//!
//! This module implements the "hook-up" that connects code generation to real consequences:
//! - Hook 1: Functional Correctness (unit test execution)
//! - Hook 2: Static Quality (Code Quality Score from static analysis)
//! - Hook 3: Topological Quality (TCSAnalyzer on code AST/CFG)
//!
//! The harness computes composite reward: R_total = w1·R_correct + w2·R_CQS + w3·R_topo

pub mod reward;
pub mod test_generator;
#[cfg(feature = "svc")]
pub mod server;

#[cfg(test)]
mod tests;

use crate::config::CodeLanguage;
use crate::code_topology::CodeTopologyAnalyzer;
use crate::sandbox::manager::SandboxManager;
use crate::tcs_analysis::TCSAnalyzer;
use anyhow::{Context, Result};
use reward::ExecutionReward;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn};

/// Training problem definition
#[derive(Debug, Clone)]
pub struct TrainingProblem {
    pub id: String,
    pub description: String,
    pub language: CodeLanguage,
    pub test_cases: Option<Vec<String>>,
    pub expected_output: Option<String>,
}

/// Execution Harness that evaluates generated code and computes rewards
pub struct ExecutionHarness {
    sandbox_manager: Arc<SandboxManager>,
    tcs_analyzer: Option<Arc<Mutex<TCSAnalyzer>>>,
    code_topology_analyzer: CodeTopologyAnalyzer,
    test_generator: test_generator::TestGenerator,
    reward_weights: RewardWeights,
}

/// Configurable weights for composite reward computation
#[derive(Debug, Clone)]
pub struct RewardWeights {
    pub functional: f64,    // w1: weight for functional correctness
    pub cqs: f64,           // w2: weight for Code Quality Score
    pub topological: f64,   // w3: weight for topological quality
}

impl Default for RewardWeights {
    fn default() -> Self {
        Self {
            functional: 0.5,   // Functional correctness is most important
            cqs: 0.3,           // Static quality is important
            topological: 0.2,  // Topological quality is bonus
        }
    }
}

impl ExecutionHarness {
    /// Create a new execution harness
    pub fn new(
        sandbox_manager: Arc<SandboxManager>,
        tcs_analyzer: Option<Arc<Mutex<TCSAnalyzer>>>,
        reward_weights: RewardWeights,
    ) -> Result<Self> {
        let code_topology_analyzer = if let Some(ref tcs) = tcs_analyzer {
            CodeTopologyAnalyzer::with_tcs_analyzer(tcs.clone())
        } else {
            CodeTopologyAnalyzer::new()
        };

        Ok(Self {
            sandbox_manager,
            tcs_analyzer,
            code_topology_analyzer,
            test_generator: test_generator::TestGenerator::new(),
            reward_weights,
        })
    }

    /// Evaluate generated code and compute composite reward
    ///
    /// This is the main entry point for the RL harness. It:
    /// 1. Executes code with unit tests (Hook 1: Functional Correctness)
    /// 2. Computes CQS from static analysis (Hook 2: Static Quality)
    /// 3. Analyzes code topology (Hook 3: Topological Quality)
    /// 4. Combines rewards: R_total = w1·R_correct + w2·R_CQS + w3·R_topo
    pub async fn evaluate_code(
        &self,
        code: &str,
        language: CodeLanguage,
        problem: &TrainingProblem,
    ) -> Result<ExecutionReward> {
        info!(
            problem_id = %problem.id,
            language = ?language,
            "Evaluating generated code"
        );

        // Hook 1: Functional Correctness
        let functional_reward = self.evaluate_functional_correctness(
            code,
            language,
            problem,
        ).await
        .context("Failed to evaluate functional correctness")?;

        // Hook 2: Static Quality (CQS)
        let cqs_reward = self.evaluate_static_quality(code, language)
            .await
            .context("Failed to evaluate static quality")?;

        // Hook 3: Topological Quality
        let topological_reward = self.evaluate_topological_quality(code, language)
            .await
            .context("Failed to evaluate topological quality")?;

        // Compute composite reward
        let total_reward = self.reward_weights.functional * functional_reward
            + self.reward_weights.cqs * cqs_reward
            + self.reward_weights.topological * topological_reward;

        info!(
            functional = functional_reward,
            cqs = cqs_reward,
            topological = topological_reward,
            total = total_reward,
            "Computed composite reward"
        );

        Ok(ExecutionReward {
            functional: functional_reward,
            cqs: cqs_reward,
            topological: topological_reward,
            total: total_reward,
        })
    }

    /// Hook 1: Evaluate functional correctness by running unit tests
    async fn evaluate_functional_correctness(
        &self,
        code: &str,
        language: CodeLanguage,
        problem: &TrainingProblem,
    ) -> Result<f64> {
        // Generate or use provided test cases
        let test_cases = if let Some(ref tests) = problem.test_cases {
            tests.clone()
        } else {
            self.test_generator.generate_tests(&problem.description, language)
                .await
                .context("Failed to generate test cases")?
        };

        // Combine code with test cases
        let test_code = self.combine_code_with_tests(code, &test_cases, language)?;

        // Execute in sandbox
        let execution_result = self.sandbox_manager
            .execute(&test_code, language)
            .await
            .context("Failed to execute code with tests")?;

        // Binary reward: 1.0 if all tests pass, 0.0 if any fail
        let reward = if execution_result.success {
            1.0
        } else {
            0.0
        };

        info!(
            success = execution_result.success,
            reward,
            "Functional correctness evaluation"
        );

        Ok(reward)
    }

    /// Hook 2: Evaluate static quality using Code Quality Score (CQS)
    async fn evaluate_static_quality(
        &self,
        code: &str,
        language: CodeLanguage,
    ) -> Result<f64> {
        // Call Python script for CQS computation
        // TODO: Consider porting to Rust for better performance
        let cqs = self.compute_cqs_from_python(code, language)
            .await
            .context("Failed to compute CQS")?;

        // Normalize CQS to [0, 1] reward (lower CQS = higher quality = higher reward)
        // CQS is typically in [0, 1] where 0 = high quality, 1 = low quality
        // So reward = 1 - CQS
        let reward = (1.0 - cqs).max(0.0).min(1.0);

        info!(cqs, reward, "Static quality evaluation");

        Ok(reward)
    }

    /// Hook 3: Evaluate topological quality using TCSAnalyzer
    async fn evaluate_topological_quality(
        &self,
        code: &str,
        language: CodeLanguage,
    ) -> Result<f64> {
        // Analyze code topology
        let topology = self.code_topology_analyzer
            .analyze(code, language)
            .await
            .context("Failed to analyze code topology")?;

        // Convert to topological signature
        let sig = topology.to_topological_signature();

        // Compute topological reward: R_topo = -(knot_complexity · C_k) - (betti1 · C_b)
        // Lower complexity and lower Betti-1 = higher reward
        let knot_penalty = sig.knot_complexity * 0.1; // C_k = 0.1
        let betti1_penalty = sig.betti_numbers[1] as f64 * 0.05; // C_b = 0.05

        // Normalize to [0, 1] range (higher penalty = lower reward)
        // Assume max reasonable values: knot_complexity ~ 10, betti1 ~ 20
        let max_penalty = 10.0 * 0.1 + 20.0 * 0.05; // = 2.0
        let penalty = (knot_penalty + betti1_penalty).min(max_penalty);
        let reward = 1.0 - (penalty / max_penalty);

        info!(
            knot_complexity = sig.knot_complexity,
            betti1 = sig.betti_numbers[1],
            penalty,
            reward,
            "Topological quality evaluation"
        );

        Ok(reward.max(0.0).min(1.0))
    }

    /// Combine code with test cases for execution
    fn combine_code_with_tests(
        &self,
        code: &str,
        test_cases: &[String],
        language: CodeLanguage,
    ) -> Result<String> {
        match language {
            CodeLanguage::Python => {
                let mut combined = code.to_string();
                combined.push_str("\n\n# Test cases\n");
                for test in test_cases {
                    combined.push_str(test);
                    combined.push('\n');
                }
                Ok(combined)
            }
            CodeLanguage::TypeScript => {
                let mut combined = code.to_string();
                combined.push_str("\n\n// Test cases\n");
                for test in test_cases {
                    combined.push_str(test);
                    combined.push('\n');
                }
                Ok(combined)
            }
        }
    }

    /// Compute CQS by calling Python script
    async fn compute_cqs_from_python(
        &self,
        code: &str,
        language: CodeLanguage,
    ) -> Result<f64> {
        // Use Rust CQS calculator instead of Python script
        use crate::cqs_calculator::CQSCalculator;
        
        let calculator = CQSCalculator::new();
        // Git churn = 0 for generated code (can be provided if available)
        let cqs_result = calculator.compute_cqs(code, language, 0)?;
        
        Ok(cqs_result.score)
    }

    /// Simple heuristic for CQS when Python script is unavailable
    fn estimate_cqs_heuristic(&self, code: &str) -> f64 {
        // Simple heuristic: based on code length and complexity indicators
        let length = code.len() as f64;
        let complexity_indicators = code.matches("if ").count()
            + code.matches("for ").count()
            + code.matches("while ").count()
            + code.matches("def ").count();

        // Normalize to [0, 1]
        let normalized_length = (length / 1000.0).min(1.0);
        let normalized_complexity = (complexity_indicators as f64 / 20.0).min(1.0);

        (normalized_length * 0.5 + normalized_complexity * 0.5).min(1.0)
    }
}

