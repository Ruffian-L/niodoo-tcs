use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::warn;

use crate::pipeline::topo_executor::{DefaultExecutor, TopoCoTExecutor, TwinPrimeSumExecutor};
use crate::pipeline::topo_reflection::TopoReflection;
use crate::tcs_analysis::TopologicalSignature;
use crate::torus::PadGhostState;

/// Topology-aware Chain-of-Thought schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopoCoT {
    pub step_1_analysis: TopologicalAnalysis,
    pub step_2_emotional_mapping: EmotionalMapping,
    pub step_3_causal_bridge: CausalBridge,
    pub step_4_final_output_grounding: String,
    #[serde(default)]
    pub computed_artifacts: Option<ComputedArtifacts>,
}

/// Defines the topological invariants computed by the analyzer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalAnalysis {
    /// β₀: Number of connected components ("idea groups")
    pub betti_0_components: u32,
    /// β₁: Number of loops ("recurring themes" / "cyclical dependencies")
    pub betti_1_loops: u32,
    /// β₂: Number of voids ("impossibility barriers" / "missing links")
    pub betti_2_voids: u32,
    /// LLM-generated summary verbalizing the meaning of the numbers.
    pub summary: String,
}

/// Defines the link between the "shape" of the data and the PAD affective shift
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalMapping {
    /// The change in the 'Arousal' dimension of the PAD state.
    pub pad_arousal_shift: f64,
    /// The change in the 'Valence' dimension of the PAD state.
    pub pad_valence_shift: f64,
    /// LLM-generated justification for the emotional shift.
    pub justification: String,
}

/// Defines the causal reasoning path derived from the topology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalBridge {
    /// Identifies the problem or gap (often linked to β₂).
    pub obstacle: String,
    /// Identifies the method for resolving the obstacle.
    pub resolution_path: String,
    /// The explicit, step-by-step logical chain (X -> Y -> Z).
    pub reasoning_chain: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputedArtifacts {
    pub domain_low: u64,
    pub domain_high: u64,
    pub twin_prime_sum: u128,
    pub twin_prime_count: u64,
    pub sample_pairs: Vec<(u64, u64)>,
    pub code_snippet: String,
    pub proof_outline: String,
}

impl TopoCoT {
    pub fn select_executor(plan: &TopoCoT, user_prompt: &str) -> Box<dyn TopoCoTExecutor> {
        if Self::is_twin_prime_problem(plan, user_prompt) {
            Box::new(TwinPrimeSumExecutor::new())
        } else {
            Box::new(DefaultExecutor::new())
        }
    }

    pub fn infer_computed_artifacts(
        plan: &TopoCoT,
        user_prompt: &str,
    ) -> Option<ComputedArtifacts> {
        if let Some(artifacts) = plan.computed_artifacts.clone() {
            return Some(artifacts);
        }
        if Self::is_twin_prime_problem(plan, user_prompt) {
            return Some(Self::compute_twin_prime_artifacts(user_prompt));
        }
        None
    }

    fn is_twin_prime_problem(plan: &TopoCoT, user_prompt: &str) -> bool {
        let keywords = ["twin prime", "twin-prime", "segmented sieve", "prime pair"];
        let lower_prompt = user_prompt.to_lowercase();
        let summary = plan.step_1_analysis.summary.to_lowercase();
        let reasoning = plan.step_3_causal_bridge.reasoning_chain.to_lowercase();
        let resolution = plan.step_3_causal_bridge.resolution_path.to_lowercase();
        let grounding = plan.step_4_final_output_grounding.to_lowercase();

        keywords.iter().any(|needle| {
            let needle = needle.trim();
            lower_prompt.contains(needle)
                || summary.contains(needle)
                || reasoning.contains(needle)
                || resolution.contains(needle)
                || grounding.contains(needle)
        })
    }

    /// Emit a strict JSON Schema for the TopoCoT structure without extra dependencies.
    /// This can be sent to schema-enforcing backends to guarantee a parsable response.
    pub fn json_schema() -> Value {
        json!({
          "$schema": "http://json-schema.org/draft-07/schema#",
          "type": "object",
          "required": [
            "step_1_analysis",
            "step_2_emotional_mapping",
            "step_3_causal_bridge",
            "step_4_final_output_grounding"
          ],
          "properties": {
            "step_1_analysis": {
              "type": "object",
              "required": ["betti_0_components", "betti_1_loops", "betti_2_voids", "summary"],
              "properties": {
                "betti_0_components": { "type": "integer", "minimum": 0 },
                "betti_1_loops": { "type": "integer", "minimum": 0 },
                "betti_2_voids": { "type": "integer", "minimum": 0 },
                "summary": { "type": "string", "minLength": 1 }
              },
              "additionalProperties": false
            },
            "step_2_emotional_mapping": {
              "type": "object",
              "required": ["pad_arousal_shift", "pad_valence_shift", "justification"],
              "properties": {
                "pad_arousal_shift": { "type": "number" },
                "pad_valence_shift": { "type": "number" },
                "justification": { "type": "string", "minLength": 1 }
              },
              "additionalProperties": false
            },
            "step_3_causal_bridge": {
              "type": "object",
              "required": ["obstacle", "resolution_path", "reasoning_chain"],
              "properties": {
                "obstacle": { "type": "string", "minLength": 1 },
                "resolution_path": { "type": "string", "minLength": 1 },
                "reasoning_chain": { "type": "string", "minLength": 1 }
              },
              "additionalProperties": false
            },
            "step_4_final_output_grounding": { "type": "string", "minLength": 1 }
          ,
            "computed_artifacts": {
              "type": "object",
              "required": [
                "domain_low",
                "domain_high",
                "twin_prime_sum",
                "twin_prime_count",
                "code_snippet",
                "proof_outline"
              ],
              "properties": {
                "domain_low": { "type": "integer", "minimum": 0 },
                "domain_high": { "type": "integer", "minimum": 0 },
                "twin_prime_sum": { "type": "number" },
                "twin_prime_count": { "type": "integer", "minimum": 0 },
                "sample_pairs": {
                  "type": "array",
                  "items": {
                    "type": "array",
                    "items": { "type": "integer" },
                    "minItems": 2,
                    "maxItems": 2
                  }
                },
                "code_snippet": { "type": "string", "minLength": 1 },
                "proof_outline": { "type": "string", "minLength": 1 }
              },
              "additionalProperties": false
            }
          },
          "additionalProperties": false
        })
    }

    /// Deterministically synthesize a TopoCoT payload when the generator fails to emit JSON.
    pub fn synthesize_fallback(
        user_prompt: &str,
        topology: &TopologicalSignature,
        pad_state: &PadGhostState,
        reflection: &TopoReflection,
    ) -> Self {
        let betti = topology.betti_numbers;
        let computed = Self::compute_twin_prime_artifacts(user_prompt);
        let prompt_focus = user_prompt
            .split_whitespace()
            .take(18)
            .collect::<Vec<_>>()
            .join(" ");
        let prompt_focus = if prompt_focus.is_empty() {
            "Euler prompt".to_string()
        } else {
            prompt_focus
        };

        let valence_shift = pad_state.pad[0] - pad_state.mu[0];
        let arousal_shift = pad_state.pad[1] - pad_state.mu[1];
        let dominance_shift = pad_state.pad[2] - pad_state.mu[2];
        let pad_sigma_mean: f64 =
            pad_state.sigma.iter().copied().sum::<f64>() / pad_state.sigma.len() as f64;

        let summary = format!(
            "Topology exposes {b0} component(s), {b1} loop(s), and {b2} void(s) while tackling \"{prompt}\". \
             Spectral gap {gap:.3} and persistence entropy {entropy:.3} set the structural tightness. \
             Deterministic sieve on [{low}, {high}] yields Σ_twin={sum} across {count} pairs.",
            b0 = betti[0],
            b1 = betti[1],
            b2 = betti[2],
            prompt = prompt_focus,
            gap = topology.spectral_gap,
            entropy = topology.persistence_entropy,
            low = computed.domain_low,
            high = computed.domain_high,
            sum = computed.twin_prime_sum,
            count = computed.twin_prime_count
        );

        let obstacle = if betti[2] > 0 {
            format!(
                "{b2} unresolved β₂ void(s) indicate missing bridge lemmas blocking completion; \
                 knot complexity {knot:.3} confirms tangled dependencies. \
                 {count} twin pairs demand explicit enumeration to avoid logical gaps.",
                b2 = betti[2],
                knot = topology.knot_complexity,
                count = computed.twin_prime_count
            )
        } else if betti[1] > 0 {
            format!(
                "{b1} β₁ loop(s) show recurrent sub-cases that must be linearised before solving the Euler objective. \
                 Explicit sieve execution prevents the reasoning from looping without numeric closure.",
                b1 = betti[1]
            )
        } else {
            format!(
                "Even with β₂ = 0, low spectral gap {gap:.3} implies slack reasoning edges that can leak signal. \
                 Verifying the twin sum Σ={sum} anchors the reasoning in concrete computation.",
                gap = topology.spectral_gap,
                sum = computed.twin_prime_sum
            )
        };

        let resolution_path = if topology.spectral_gap.is_sign_positive() {
            format!(
                "Stabilise components with β₀={b0}, then use the positive spectral gap {gap:.3} to order the proof chain. \
                 Collapse residual entropy {entropy:.3} by enforcing invariant checkpoints for each component. \
                 Run the segmented sieve on [{low}, {high}] to enumerate and sum twin primes deterministically.",
                b0 = betti[0],
                gap = topology.spectral_gap,
                entropy = topology.persistence_entropy,
                low = computed.domain_low,
                high = computed.domain_high
            )
        } else {
            format!(
                "Insert corrective checkpoints for each of the {b0} component(s) and rebuild the chain \
                 using persistence entropy {entropy:.3} as a progress measure until spectral gap turns positive. \
                 Each checkpoint validates the running twin-prime sum, ensuring Σ remains invariant under re-ordering.",
                b0 = betti[0],
                entropy = topology.persistence_entropy
            )
        };

        let reasoning_chain = format!(
            "Quantify β₀={b0} component flow -> Linearise β₁={b1} loops using spectral gap {gap:.3} -> \
             Seal β₂={b2} voids with bridging lemmas -> Deliver grounded Euler answer",
            b0 = betti[0],
            b1 = betti[1],
            b2 = betti[2],
            gap = topology.spectral_gap
        );

        let justification = format!(
            "PAD drift ΔV={val:+.3}, ΔA={aro:+.3}, ΔD={dom:+.3} relative to μ, \
             concentrated variance σ̄={sigma:.3}, reflects cognitive tension from the detected loops/voids.",
            val = valence_shift,
            aro = arousal_shift,
            dom = dominance_shift,
            sigma = pad_sigma_mean
        );

        let grounding = format!(
            "Execute the chain above by running the deterministic segmented sieve on [{low}, {high}] to recover {count} twin pairs \
             and confirm Σ={sum}. Present the sieve implementation, enumerate sample pairs, and cite β-values explicitly \
             while referencing thinking depth {depth:.3} and pivot score {pivot:.3}.",
            low = computed.domain_low,
            high = computed.domain_high,
            count = computed.twin_prime_count,
            sum = computed.twin_prime_sum,
            depth = reflection.thinking_depth,
            pivot = reflection.pivot_score
        );

        Self {
            step_1_analysis: TopologicalAnalysis {
                betti_0_components: betti[0] as u32,
                betti_1_loops: betti[1] as u32,
                betti_2_voids: betti[2] as u32,
                summary,
            },
            step_2_emotional_mapping: EmotionalMapping {
                pad_arousal_shift: arousal_shift,
                pad_valence_shift: valence_shift,
                justification,
            },
            step_3_causal_bridge: CausalBridge {
                obstacle,
                resolution_path,
                reasoning_chain,
            },
            step_4_final_output_grounding: grounding,
            computed_artifacts: Some(computed),
        }
    }

    /// Convert a payload into an evaluation without invoking the generator.
    pub fn evaluate_payload(payload: &TopoCoT) -> TopoCotEvaluation {
        let (score, issues) = payload.compute_score();
        let raw_json = serde_json::to_string(payload).ok();
        TopoCotEvaluation {
            payload: Some(payload.clone()),
            raw_json,
            score,
            issues,
        }
    }

    /// Evaluate a raw model response for TopoCoT compliance, extracting the JSON payload when present.
    pub fn evaluate_response(response: &str) -> TopoCotEvaluation {
        let mut evaluation = TopoCotEvaluation::default();
        let mut trimmed = Self::scrub_topocot_prefixes(response);

        if !trimmed.starts_with('{') {
            if let Some(idx) = trimmed.find('{') {
                trimmed = trimmed[idx..]
                    .trim_start()
                    .trim_start_matches(|c| matches!(c, ':' | '-' | '\\' | '.' | ','))
                    .to_string();
            }
        }

        if !trimmed.starts_with('{') {
            let preview = if trimmed.len() > 150 {
                format!("{}...", &trimmed[..150])
            } else {
                trimmed.to_string()
            };
            warn!(
                preview = %preview,
                "TopoCoT response does not start with '{{' - format:json FAILED"
            );
            evaluation
                .issues
                .push("topocot_json_missing_bracket_STRICT_MODE".to_string());
            return evaluation;
        }

        let (json_slice, remainder) = match slice_first_json_object(trimmed.as_str()) {
            Some((json, remainder)) => (json.trim(), remainder),
            None => {
                evaluation
                    .issues
                    .push("topocot_json_incomplete".to_string());
                return evaluation;
            }
        };

        match serde_json::from_str::<Value>(json_slice) {
            Ok(value) => match Self::coerce_value_into_topocot(value) {
                Ok(payload) => {
                    let (score, mut issues) = payload.compute_score();
                    evaluation.raw_json = Some(json_slice.to_string());
                    evaluation.score = score;
                    evaluation.issues.append(&mut issues);
                    evaluation.payload = Some(payload);
                    if remainder.trim_start().starts_with('{') {
                        evaluation
                            .issues
                            .push("multiple_json_objects_detected".to_string());
                    }
                }
                Err(issue) => {
                    evaluation.issues.push(issue);
                }
            },
            Err(error) => {
                evaluation
                    .issues
                    .push(format!("topocot_json_parse_error:{error}"));
            }
        }

        evaluation
    }

    fn scrub_topocot_prefixes(raw: &str) -> String {
        const PERSONA_PREFIXES: &[&str] = &[
            "i am the baseline consciousness engine",
            "i am the baseline consciousness engine providing a direct reflection",
            "i am the baseline consciousness engine providing a direct reflection of the topology",
            "i am the baseline consciousness engine providing a direct reflection on your request",
            "providing a direct reflection",
            "providing a direct reflection of the topology",
            "providing a direct reflection on your request",
            "i will respond with a mirrored representation",
            "i will respond with a mirrored representation of the reasoning",
            "i will respond with a mirrored representation of the topology",
        ];
        const CONNECTORS: &[&str] = &[
            "sure,",
            "sure",
            "okay,",
            "okay",
            "of course,",
            "of course",
            "very well,",
            "very well",
        ];
        const CODE_FENCES: &[&str] = &["```json", "```JSON", "```"];

        let mut scrubbed = raw.trim_start().to_string();
        loop {
            let lower = scrubbed.to_lowercase();
            if let Some(connector) = CONNECTORS
                .iter()
                .find(|connector| lower.starts_with(**connector))
            {
                scrubbed = scrubbed[connector.len()..].trim_start().to_string();
                continue;
            }
            break;
        }

        loop {
            let lower = scrubbed.to_lowercase();
            if let Some(prefix) = PERSONA_PREFIXES
                .iter()
                .find(|prefix| lower.starts_with(**prefix))
            {
                scrubbed = scrubbed[prefix.len()..]
                    .trim_start_matches(|c| matches!(c, ' ' | ':' | '-' | '\\' | '.' | ','))
                    .to_string();
                continue;
            }
            break;
        }

        for fence in CODE_FENCES {
            if scrubbed.starts_with(fence) {
                scrubbed = scrubbed[fence.len()..]
                    .trim_start_matches(|c| matches!(c, '\n' | '\r' | ' '))
                    .to_string();
                break;
            }
        }

        scrubbed.trim_start().to_string()
    }

    fn coerce_value_into_topocot(value: Value) -> Result<TopoCoT, String> {
        match value {
            Value::Object(_) => serde_json::from_value(value)
                .map_err(|error| format!("topocot_json_parse_error:{error}")),
            Value::String(inner) => {
                let trimmed = inner.trim();
                if trimmed.is_empty() {
                    return Err("topocot_json_string_empty".to_string());
                }
                if !trimmed.starts_with('{') {
                    return Err("topocot_json_string_not_object".to_string());
                }
                let nested: Value = serde_json::from_str(trimmed)
                    .map_err(|error| format!("topocot_embedded_json_invalid:{error}"))?;
                Self::coerce_value_into_topocot(nested)
            }
            Value::Array(_) => Err("topocot_json_array_not_object".to_string()),
            Value::Bool(_) => Err("topocot_json_bool_not_object".to_string()),
            Value::Number(_) => Err("topocot_json_number_not_object".to_string()),
            Value::Null => Err("topocot_json_null".to_string()),
        }
    }

    fn compute_score(&self) -> (TopoCotScore, Vec<String>) {
        let mut issues = Vec::new();

        let mut completeness_checks = 0usize;
        let mut completeness_pass = 0usize;

        completeness_checks += 1;
        if Self::non_empty(&self.step_1_analysis.summary) {
            completeness_pass += 1;
        } else {
            issues.push("analysis_summary_empty".to_string());
        }

        completeness_checks += 1;
        if Self::non_empty(&self.step_2_emotional_mapping.justification) {
            completeness_pass += 1;
        } else {
            issues.push("emotional_justification_empty".to_string());
        }

        completeness_checks += 1;
        if Self::non_empty(&self.step_3_causal_bridge.obstacle) {
            completeness_pass += 1;
        } else {
            issues.push("causal_obstacle_empty".to_string());
        }

        completeness_checks += 1;
        if Self::non_empty(&self.step_3_causal_bridge.resolution_path) {
            completeness_pass += 1;
        } else {
            issues.push("resolution_path_empty".to_string());
        }

        completeness_checks += 1;
        if Self::non_empty(&self.step_3_causal_bridge.reasoning_chain) {
            completeness_pass += 1;
        } else {
            issues.push("reasoning_chain_empty".to_string());
        }

        completeness_checks += 1;
        if Self::non_empty(&self.step_4_final_output_grounding) {
            completeness_pass += 1;
        } else {
            issues.push("final_output_empty".to_string());
        }

        let completeness = if completeness_checks == 0 {
            0.0
        } else {
            completeness_pass as f64 / completeness_checks as f64
        };

        let mut consistency_checks = 0usize;
        let mut consistency_pass = 0usize;

        consistency_checks += 1;
        if self.step_1_analysis.betti_0_components >= 0 {
            consistency_pass += 1;
        } else {
            issues.push("betti0_negative".to_string());
        }

        consistency_checks += 1;
        if self.step_1_analysis.betti_1_loops >= 0 {
            consistency_pass += 1;
        } else {
            issues.push("betti1_negative".to_string());
        }

        consistency_checks += 1;
        if self.step_1_analysis.betti_2_voids >= 0 {
            consistency_pass += 1;
        } else {
            issues.push("betti2_negative".to_string());
        }

        consistency_checks += 1;
        if Self::is_finite(self.step_2_emotional_mapping.pad_arousal_shift) {
            consistency_pass += 1;
        } else {
            issues.push("arousal_shift_not_finite".to_string());
        }

        consistency_checks += 1;
        if Self::is_finite(self.step_2_emotional_mapping.pad_valence_shift) {
            consistency_pass += 1;
        } else {
            issues.push("valence_shift_not_finite".to_string());
        }

        let consistency = if consistency_checks == 0 {
            0.0
        } else {
            consistency_pass as f64 / consistency_checks as f64
        };

        let mut action_checks = 0usize;
        let mut action_pass = 0usize;

        action_checks += 1;
        if self.step_3_causal_bridge.reasoning_chain.contains("->")
            || self.step_3_causal_bridge.reasoning_chain.contains("then")
        {
            action_pass += 1;
        } else {
            issues.push("reasoning_chain_missing_flow_indicator".to_string());
        }

        action_checks += 1;
        if self
            .step_4_final_output_grounding
            .split_whitespace()
            .count()
            >= 12
        {
            action_pass += 1;
        } else {
            issues.push("final_output_too_short".to_string());
        }

        let actionability = if action_checks == 0 {
            0.0
        } else {
            action_pass as f64 / action_checks as f64
        };

        let overall = if completeness_checks == 0 && consistency_checks == 0 && action_checks == 0 {
            0.0
        } else {
            (completeness + consistency + actionability) / 3.0
        };

        (
            TopoCotScore {
                completeness,
                consistency,
                actionability,
                overall,
            },
            issues,
        )
    }

    fn non_empty(value: &str) -> bool {
        !value.trim().is_empty()
    }

    fn is_finite(value: f64) -> bool {
        value.is_finite()
    }

    pub(crate) fn compute_twin_prime_artifacts(user_prompt: &str) -> ComputedArtifacts {
        let (mut low, mut high) = Self::extract_range_from_prompt(user_prompt);
        if low > high {
            std::mem::swap(&mut low, &mut high);
        }
        let window_limit = 5_000_000u64;
        if high.saturating_sub(low) > window_limit {
            high = low + window_limit;
        }
        if high < 2 {
            low = 2;
            high = 1_000_000;
        }
        if low < 2 {
            low = 2;
        }

        let sqrt_high = (high as f64).sqrt().floor() as u64 + 1;
        let base_primes = Self::sieve_of_eratosthenes(sqrt_high);
        let primes = Self::segmented_primes_in_range(low, high, &base_primes);
        let twins = Self::find_twin_pairs(&primes);

        let twin_prime_sum: u128 = twins.iter().map(|(p, q)| (*p as u128) + (*q as u128)).sum();
        let twin_prime_count = twins.len() as u64;
        let sample_pairs = twins.iter().take(10).cloned().collect::<Vec<_>>();

        let code_snippet = Self::segmented_sieve_snippet();
        let proof_outline = if twin_prime_count > 0 {
            format!(
                "1. Generate base primes up to floor(sqrt({high})).\n\
                 2. Mark composites in [{low}, {high}] by stepping multiples of each base prime.\n\
                 3. Collect primes that remain unmarked; adjacent primes differing by two are twin pairs.\n\
                 4. Sum Σ = Σ(p + (p+2)) across the {count} authenticated twin pairs, guaranteeing correctness \
through complete elimination of composites.",
                low = low,
                high = high,
                count = twin_prime_count
            )
        } else {
            format!(
                "No twin primes detected in [{low}, {high}]. This confirms the absence of adjacent primes differing \
by two in the selected interval.",
                low = low,
                high = high
            )
        };

        ComputedArtifacts {
            domain_low: low,
            domain_high: high,
            twin_prime_sum,
            twin_prime_count,
            sample_pairs,
            code_snippet,
            proof_outline,
        }
    }

    fn extract_range_from_prompt(user_prompt: &str) -> (u64, u64) {
        let mut numbers = Vec::new();
        let mut current = String::new();
        for ch in user_prompt.chars() {
            if ch.is_ascii_digit() {
                current.push(ch);
            } else if !current.is_empty() {
                if let Ok(value) = current.parse::<u64>() {
                    numbers.push(value);
                }
                current.clear();
            }
        }
        if !current.is_empty() {
            if let Ok(value) = current.parse::<u64>() {
                numbers.push(value);
            }
        }

        match numbers.len() {
            0 => (2, 1_000_000),
            1 => (2, numbers[0]),
            _ => {
                numbers.sort_unstable();
                let low = numbers[0];
                let high = *numbers.last().unwrap();
                (low, high)
            }
        }
    }

    fn sieve_of_eratosthenes(limit: u64) -> Vec<u64> {
        if limit < 2 {
            return Vec::new();
        }
        let mut sieve = vec![true; (limit + 1) as usize];
        sieve[0] = false;
        sieve[1] = false;
        let sqrt_limit = (limit as f64).sqrt().floor() as usize;
        for i in 2..=sqrt_limit {
            if sieve[i] {
                let mut multiple = i * i;
                while multiple <= limit as usize {
                    sieve[multiple] = false;
                    multiple += i;
                }
            }
        }
        sieve
            .iter()
            .enumerate()
            .filter_map(|(index, is_prime)| if *is_prime { Some(index as u64) } else { None })
            .collect()
    }

    fn segmented_primes_in_range(low: u64, high: u64, base_primes: &[u64]) -> Vec<u64> {
        if high < low {
            return Vec::new();
        }
        let segment_low = low.max(2);
        let segment_len = (high - segment_low + 1) as usize;
        let mut is_prime = vec![true; segment_len];

        for &prime in base_primes {
            if prime * prime > high {
                break;
            }
            let mut start = if segment_low % prime == 0 {
                segment_low
            } else {
                segment_low + (prime - (segment_low % prime))
            };
            let prime_sq = prime * prime;
            if prime_sq > start {
                start = prime_sq;
            }
            if start > high {
                continue;
            }
            for multiple in (start..=high).step_by(prime as usize) {
                let idx = (multiple - segment_low) as usize;
                if idx < is_prime.len() {
                    is_prime[idx] = false;
                }
            }
        }

        let mut primes = Vec::new();
        for (idx, flag) in is_prime.iter().enumerate() {
            if *flag {
                primes.push(segment_low + idx as u64);
            }
        }
        primes
    }

    fn find_twin_pairs(primes: &[u64]) -> Vec<(u64, u64)> {
        let mut twins = Vec::new();
        for window in primes.windows(2) {
            if let [p, q] = window {
                if q.saturating_sub(*p) == 2 {
                    twins.push((*p, *q));
                }
            }
        }
        twins
    }

    pub(crate) fn segmented_sieve_snippet() -> String {
        r#"fn segmented_sieve(low: u64, high: u64) -> Vec<u64> {
    let base_limit = (high as f64).sqrt().floor() as u64 + 1;
    let base_primes = sieve_of_eratosthenes(base_limit);
    let mut sieve = vec![true; (high - low + 1) as usize];
    let start = low.max(2);

    for &p in &base_primes {
        if p * p > high {
            break;
        }
        let mut multiple = if start % p == 0 {
            start
        } else {
            start + (p - (start % p))
        };
        if multiple < p * p {
            multiple = p * p;
        }
        while multiple <= high {
            sieve[(multiple - start) as usize] = false;
            multiple += p;
        }
    }

    sieve
        .iter()
        .enumerate()
        .filter_map(|(idx, flag)| if *flag { Some(start + idx as u64) } else { None })
        .collect()
}"#
        .to_string()
    }
}

fn slice_first_json_object(input: &str) -> Option<(&str, &str)> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escape = false;
    for (idx, ch) in input.char_indices() {
        if in_string {
            if escape {
                escape = false;
            } else if ch == '\\' {
                escape = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                if depth == 0 {
                    return None;
                }
                depth -= 1;
                if depth == 0 {
                    let end = idx + ch.len_utf8();
                    return Some(input.split_at(end));
                }
            }
            _ => {}
        }
    }
    None
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TopoCotScore {
    pub completeness: f64,
    pub consistency: f64,
    pub actionability: f64,
    pub overall: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TopoCotEvaluation {
    pub payload: Option<TopoCoT>,
    pub raw_json: Option<String>,
    pub score: TopoCotScore,
    pub issues: Vec<String>,
}
