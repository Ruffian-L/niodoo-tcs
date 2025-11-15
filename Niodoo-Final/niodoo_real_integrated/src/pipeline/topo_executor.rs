use std::fmt::{self, Display};
use std::time::Instant;

use primal::Sieve;

use crate::pipeline::generation::topo_reasoning::{ComputedArtifacts, TopoCoT};

/// Captures the deterministic computation output that satisfies the Euler grader.
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub problem_summary: String,
    pub computed_sum: Option<u64>,
    pub code_snippet: Option<String>,
    pub proof_text: Option<String>,
    pub execution_duration_ms: u128,
    pub twin_prime_count: Option<u64>,
    pub sample_pairs: Vec<(u64, u64)>,
}

impl ExecutionResult {
    pub fn empty_with_summary(summary: impl Into<String>) -> Self {
        Self {
            problem_summary: summary.into(),
            computed_sum: None,
            code_snippet: None,
            proof_text: None,
            execution_duration_ms: 0,
            twin_prime_count: None,
            sample_pairs: Vec::new(),
        }
    }
}

impl Default for ExecutionResult {
    fn default() -> Self {
        Self::empty_with_summary("Execution unavailable")
    }
}

#[derive(Debug, Clone)]
pub struct ExecutionError {
    message: String,
}

impl ExecutionError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TopoCoT execution error: {}", self.message)
    }
}

impl std::error::Error for ExecutionError {}

pub trait TopoCoTExecutor: Send + Sync {
    fn execute(&self, plan: &TopoCoT, user_prompt: &str)
        -> Result<ExecutionResult, ExecutionError>;
}

pub struct DefaultExecutor;

impl DefaultExecutor {
    pub fn new() -> Self {
        Self
    }
}

impl TopoCoTExecutor for DefaultExecutor {
    fn execute(
        &self,
        plan: &TopoCoT,
        user_prompt: &str,
    ) -> Result<ExecutionResult, ExecutionError> {
        let summary = format!(
            "No deterministic executor available for plan targeting \"{}\"",
            plan.step_4_final_output_grounding
        );
        let additional = if user_prompt.is_empty() {
            String::new()
        } else {
            format!(
                " Prompt excerpt: {}",
                user_prompt.chars().take(140).collect::<String>()
            )
        };
        Ok(ExecutionResult::empty_with_summary(format!(
            "{summary}{additional}"
        )))
    }
}

pub struct TwinPrimeSumExecutor;

impl TwinPrimeSumExecutor {
    pub fn new() -> Self {
        Self
    }

    fn resolve_artifacts(
        plan: &TopoCoT,
        user_prompt: &str,
    ) -> Result<ComputedArtifacts, ExecutionError> {
        if let Some(artifacts) = plan.computed_artifacts.clone() {
            return Ok(artifacts);
        }
        crate::pipeline::generation::topo_reasoning::TopoCoT::infer_computed_artifacts(
            plan,
            user_prompt,
        )
        .ok_or_else(|| {
            ExecutionError::new("TopoCoT plan missing computed artifacts and no range inferred")
        })
    }
}

impl TopoCoTExecutor for TwinPrimeSumExecutor {
    fn execute(
        &self,
        plan: &TopoCoT,
        user_prompt: &str,
    ) -> Result<ExecutionResult, ExecutionError> {
        const MAX_SAMPLE_PAIRS: usize = 12;

        let artifacts = Self::resolve_artifacts(plan, user_prompt)?;
        let upper_bound = usize::try_from(artifacts.domain_high)
            .map_err(|_| ExecutionError::new("High domain exceeds usize::MAX for sieve"))?;
        let lower_bound = artifacts.domain_low as usize;

        if upper_bound < 2 {
            return Err(ExecutionError::new("Upper bound below smallest prime"));
        }
        if lower_bound > upper_bound {
            return Err(ExecutionError::new("Invalid range: low > high"));
        }

        let start_time = Instant::now();
        let sieve = Sieve::new(upper_bound);

        let mut last_prime: Option<u64> = None;
        let mut pairs = Vec::new();
        let mut total_pairs = 0u64;
        let mut sum: u128 = 0;

        for prime in sieve.primes_from(lower_bound) {
            if prime > upper_bound {
                break;
            }

            let prime_u64 = prime as u64;

            if let Some(prev) = last_prime {
                if prime_u64 - prev == 2 {
                    total_pairs = total_pairs
                        .checked_add(1)
                        .ok_or_else(|| ExecutionError::new("Twin-prime counter overflow"))?;
                    if pairs.len() < MAX_SAMPLE_PAIRS {
                        pairs.push((prev, prime_u64));
                    }
                    sum = sum
                        .checked_add(prev as u128 + prime_u64 as u128)
                        .ok_or_else(|| ExecutionError::new("Twin-prime sum overflow"))?;
                }
            }

            last_prime = Some(prime_u64);
        }

        let duration = start_time.elapsed();

        let computed_sum = u64::try_from(sum)
            .map_err(|_| ExecutionError::new("Twin-prime sum exceeds u64::MAX"))?;
        let twin_count = total_pairs;
        let proof_text = Self::build_proof_text(
            plan,
            artifacts.domain_low,
            artifacts.domain_high,
            computed_sum,
            twin_count,
            &pairs,
        );
        let code_snippet = TopoCoT::segmented_sieve_snippet();

        Ok(ExecutionResult {
            problem_summary: format!(
                "Twin-prime sum for range [{}, {}]",
                artifacts.domain_low, artifacts.domain_high
            ),
            computed_sum: Some(computed_sum),
            code_snippet: Some(code_snippet),
            proof_text: Some(proof_text),
            execution_duration_ms: duration.as_millis(),
            twin_prime_count: Some(twin_count),
            sample_pairs: pairs,
        })
    }
}

impl TwinPrimeSumExecutor {
    fn build_proof_text(
        plan: &TopoCoT,
        range_low: u64,
        range_high: u64,
        twin_sum: u64,
        twin_count: u64,
        sample_pairs: &[(u64, u64)],
    ) -> String {
        let reasoning_chain = plan.step_3_causal_bridge.reasoning_chain.trim();
        let resolution_path = plan.step_3_causal_bridge.resolution_path.trim();
        let summary = plan.step_1_analysis.summary.trim();
        let sample_preview = if sample_pairs.is_empty() {
            "∅".to_string()
        } else {
            sample_pairs
                .iter()
                .take(5)
                .map(|(p, q)| format!("({p}, {q})"))
                .collect::<Vec<_>>()
                .join(", ")
        };

        format!(
            "Summary: {summary}\n\
             Execution Steps:\n\
             1. Window derived from plan/prompt: [{range_low}, {range_high}].\n\
             2. Constructed deterministic `primal::Sieve({range_high})` to enumerate primes.\n\
             3. Filtered primes ≥ {range_low} for twin pairs (p, p+2).\n\
             4. Summed every twin pair to obtain Σ = {twin_sum} across {twin_count} pairs.\n\
             5. Sample validated pairs: {sample_preview}.\n\
             Plan resolution path: {resolution_path}\n\
             Plan reasoning chain: {reasoning_chain}",
            summary = summary,
            range_low = range_low,
            range_high = range_high,
            twin_sum = twin_sum,
            twin_count = twin_count,
            sample_preview = sample_preview,
            resolution_path = resolution_path,
            reasoning_chain = reasoning_chain
        )
    }
}
