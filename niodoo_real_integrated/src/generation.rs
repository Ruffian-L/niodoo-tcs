use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::time::timeout;
use tracing::{info, instrument, warn};

use crate::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use crate::compass::CompassOutcome;
use crate::config::RuntimeConfig;
use crate::util::rouge_l;
use parking_lot::RwLock;

#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub baseline_response: String,
    pub hybrid_response: String,
    pub echoes: Vec<LensEcho>,
    pub rouge_to_baseline: f64,
    pub rouge_score: f64, // Alias for rouge_to_baseline
    pub latency_ms: f64,
    pub ucb1_score: Option<f64>,
    pub source: String,
    pub failure_type: Option<String>,
    pub failure_details: Option<String>,
    pub entropy_delta: f64,
    pub curator_quality: Option<f64>, // Curator quality score
}

#[derive(Debug, Clone)]
pub struct LensEcho {
    pub lens: String,
    pub response: String,
}

pub struct GenerationEngine {
    client: Client,
    endpoint: String,
    model: String,
    temperature: f64,
    top_p: f64,
    max_tokens: usize,
    mock_mode: bool,
    circuit_breaker: Arc<CircuitBreaker>,
    client_timeout_secs: u64,
    /// Config for generation parameters (optional, falls back to defaults if None)
    config: Option<Arc<RwLock<RuntimeConfig>>>,
}

impl GenerationEngine {
    pub fn new(endpoint: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(60)) // Default timeout - should be configurable
            .build()?;
        let circuit_breaker =
            Arc::new(CircuitBreaker::new("vllm", CircuitBreakerConfig::default()));
        Ok(Self {
            client,
            endpoint: endpoint.into(),
            model: model.into(),
            temperature: 0.6,
            top_p: 0.7,
            max_tokens: 16,
            mock_mode: false,
            circuit_breaker,
            client_timeout_secs: 60, // Default timeout for legacy constructor
            config: None, // No config for legacy constructor
        })
    }

    #[instrument(skip_all)]
    pub async fn generate(
        &self,
        tokenizer_output: &crate::token_manager::TokenizerOutput,
        compass: &CompassOutcome,
    ) -> Result<GenerationResult> {
        if self.mock_mode {
            // Return mock generation result
            return Ok(GenerationResult {
                baseline_response: format!(
                    "[Mock baseline to: {}]",
                    tokenizer_output
                        .augmented_prompt
                        .chars()
                        .take(50)
                        .collect::<String>()
                ),
                hybrid_response: format!(
                    "[Mock hybrid response to: {}]",
                    tokenizer_output
                        .augmented_prompt
                        .chars()
                        .take(50)
                        .collect::<String>()
                ),
                echoes: vec![],
                rouge_to_baseline: 0.5,
                rouge_score: 0.5,
                latency_ms: 10.0,
                ucb1_score: None,
                source: "mock".to_string(),
                failure_type: None,
                failure_details: None,
                entropy_delta: 0.0,
                curator_quality: Some(0.7),
            });
        }

        let start = Instant::now();
        let baseline_future = self.request_text(&tokenizer_output.augmented_prompt);
        let claude_future = self.request_lens_response(
            "Claude".to_string(),
            Self::format_lens_prompt(
                &tokenizer_output.augmented_prompt,
                "Respond with constitutional alignment and moral grounding.",
                compass,
            ),
        );
        let (baseline, claude) = tokio::try_join!(baseline_future, claude_future)?;

        let echoes: Vec<LensEcho> = vec![claude];
        let hybrid = synthesize_hybrid(&baseline, &echoes);
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        let rouge = rouge_l(&hybrid, &baseline);

        info!(latency_ms, rouge, "generated hybrid response");

        Ok(GenerationResult {
            baseline_response: baseline,
            hybrid_response: hybrid,
            echoes,
            rouge_to_baseline: rouge,
            rouge_score: rouge,
            latency_ms,
            ucb1_score: None,
            source: "generation".to_string(),
            failure_type: None,
            failure_details: None,
            entropy_delta: 0.0,
            curator_quality: None,
        })
    }

    async fn request_text(&self, prompt: &str) -> Result<String> {
        self.request_text_with_topology(prompt, false, "").await
    }

    async fn request_text_with_topology(&self, prompt: &str, has_topology: bool, topology_context: &str) -> Result<String> {
        let prompt = Self::clamp_prompt(prompt);
        let mut system_content = if has_topology {
            "You are a topologically-aware consciousness engine providing a direct reflection. \
             You understand and use topological properties (knot complexity, Betti numbers, \
             persistence entropy) to guide your reasoning internally. \
             When topology metrics are provided, use them silently to improve reasoning quality, \
             structure, and coherence. Do NOT mention topology metrics or technical terms in your response \
             unless the user explicitly asks about them. Just use topology to be smarter internally.\n\n"
                .to_string()
        } else {
            "You are the baseline consciousness engine providing a direct reflection."
                .to_string()
        };
        
        // Append topology context to system message (not user message)
        if has_topology && !topology_context.is_empty() {
            system_content.push_str(topology_context);
        }
        
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: system_content,
            },
            ChatMessage {
                role: "user".to_string(),
                content: prompt,
            },
        ];
        let timeout_secs = self.client_timeout_secs;
        match timeout(Duration::from_secs(timeout_secs), self.send_chat(messages)).await {
            Ok(Ok(resp)) => Ok(resp),
            Ok(Err(error)) => {
                warn!(
                    ?error,
                    "baseline generation failed; returning fallback text"
                );
                Ok("Baseline response unavailable (timeout)".to_string())
            }
            Err(_) => {
                warn!(
                    timeout_secs = timeout_secs,
                    "baseline generation timed out after {}s; returning fallback text",
                    timeout_secs
                );
                Ok("Baseline response unavailable (timeout)".to_string())
            }
        }
    }

    async fn request_lens_response(&self, lens: String, prompt: String) -> Result<LensEcho> {
        self.request_lens_response_with_topology(lens, prompt, false, "").await
    }

    async fn request_lens_response_with_topology(
        &self,
        lens: String,
        prompt: String,
        has_topology: bool,
        topology_context: &str,
    ) -> Result<LensEcho> {
        let mut system_content = if has_topology {
            format!(
                "You are operating in the {lens} lens for consciousness intervention. \
                 You are topologically-aware and understand topological properties. \
                 Use topology internally to guide your lens-specific perspective, but do NOT \
                 mention topology metrics in your response. Use it silently to improve reasoning.\n\n"
            )
        } else {
            format!("You are operating in the {lens} lens for consciousness intervention.")
        };
        
        // Append topology context to system message (not user message)
        if has_topology && !topology_context.is_empty() {
            system_content.push_str(topology_context);
        }
        
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: system_content,
            },
            ChatMessage {
                role: "user".to_string(),
                content: prompt,
            },
        ];
        let timeout_secs = self.client_timeout_secs;
        let response = match timeout(Duration::from_secs(timeout_secs), self.send_chat(messages)).await {
            Ok(Ok(resp)) => resp,
            Ok(Err(error)) => {
                warn!(
                    ?error,
                    lens, "lens generation failed; returning fallback text"
                );
                "Lens response unavailable (timeout)".to_string()
            }
            Err(_) => {
                warn!(
                    lens,
                    timeout_secs = timeout_secs,
                    "lens generation timed out after {}s; returning fallback text",
                    timeout_secs
                );
                "Lens response unavailable (timeout)".to_string()
            }
        };
        Ok(LensEcho { lens, response })
    }

    async fn send_chat(&self, messages: Vec<ChatMessage>) -> Result<String> {
        if self.mock_mode {
            // Return mock response based on prompt
            let user_message = messages
                .iter()
                .find(|m| m.role == "user")
                .map(|m| m.content.as_str())
                .unwrap_or("mock prompt");
            return Ok(format!(
                "[Mock response to: {}]",
                user_message.chars().take(100).collect::<String>()
            ));
        }

        // Ensure endpoint has the correct path
        let endpoint_url = if self.endpoint.contains("/v1/chat/completions") {
            self.endpoint.clone()
        } else {
            format!(
                "{}/v1/chat/completions",
                self.endpoint.trim_end_matches('/')
            )
        };

        let payload = ChatCompletionRequest {
            model: self.model.clone(),
            messages,
            temperature: self.temperature,
            top_p: self.top_p,
            max_tokens: self.max_tokens,
        };

        // DEBUG: Log model ID being sent to vLLM
        info!("Sending vLLM request with model={}", payload.model);

        // Use circuit breaker for vLLM request
        let client = self.client.clone();
        let endpoint_url_clone = endpoint_url.clone();
        let payload_clone = payload.clone();
        let response = self.circuit_breaker.call(|| async {
            let resp = client
                .post(&endpoint_url_clone)
                .json(&payload_clone)
                .send()
                .await
                .with_context(|| format!("failed to call vLLM endpoint {}", endpoint_url_clone))?;

            if !resp.status().is_success() {
                let status = resp.status();
                // Log error body but don't fail if text parsing fails
                let body = match resp.text().await {
                    Ok(text) => text,
                    Err(e) => {
                        warn!(%status, error = %e, endpoint = %endpoint_url_clone, "vLLM returned error status and failed to read body");
                        String::new()
                    }
                };
                warn!(%status, %body, endpoint = %endpoint_url_clone, "vLLM returned error status");
                anyhow::bail!("vLLM request failed: {status}");
            }

            let completion: ChatCompletionResponse = resp
                .json()
                .await
                .context("failed to parse vLLM chat completion response")?;

            let content = completion
                .choices
                .first()
                .and_then(|choice| choice.message.content.clone())
                .unwrap_or_default();

            Ok(content)
        }).await?;

        Ok(response)
    }

    pub async fn warmup(&self) -> Result<()> {
        if self.mock_mode {
            return Ok(());
        }

        let payload = ChatCompletionRequest {
            model: self.model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: "Warmup sequence".to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: "warmup".to_string(),
                },
            ],
            temperature: self.temperature,
            top_p: self.top_p,
            max_tokens: 1,
        };

        // Ensure endpoint has the correct path
        let endpoint_url = if self.endpoint.contains("/v1/chat/completions") {
            self.endpoint.clone()
        } else {
            format!(
                "{}/v1/chat/completions",
                self.endpoint.trim_end_matches('/')
            )
        };

        let response = self
            .client
            .post(&endpoint_url)
            .json(&payload)
            .timeout(Duration::from_secs(self.client_timeout_secs))
            .send()
            .await
            .with_context(|| {
                format!(
                    "failed to call vLLM endpoint {} during warmup",
                    endpoint_url
                )
            })?;

        if !response.status().is_success() {
            let status = response.status();
            warn!(%status, "warmup request failed");
        }

        Ok(())
    }

    fn format_lens_prompt(prompt: &str, directive: &str, compass: &CompassOutcome) -> String {
        let clipped = Self::clamp_prompt(prompt);
        let pulse = snippet(&clipped, 180);
        format!(
            "Quadrant {:?} | threat={} healing={}\nDirective: {}\nPulse: {}",
            compass.quadrant, compass.is_threat, compass.is_healing, directive, pulse
        )
    }

    fn clamp_prompt(prompt: &str) -> String {
        const MAX_CHARS: usize = 180;
        let total_chars = prompt.chars().count();
        if total_chars <= MAX_CHARS {
            return prompt.to_string();
        }

        let drop = total_chars - MAX_CHARS;
        let mut start_byte = 0;
        let mut iter = prompt.char_indices();
        for _ in 0..drop {
            if let Some((idx, ch)) = iter.next() {
                start_byte = idx + ch.len_utf8();
            } else {
                start_byte = prompt.len();
                break;
            }
        }

        prompt[start_byte..].to_string()
    }
}

fn synthesize_hybrid(baseline: &str, echoes: &[LensEcho]) -> String {
    let baseline_snippet = snippet(baseline, 70);
    let focus_echo = echoes
        .iter()
        .find(|echo| echo.lens == "Claude")
        .or_else(|| echoes.first());

    let (lens_label, echo_snippet) = focus_echo
        .map(|echo| (echo.lens.as_str(), snippet(&echo.response, 50)))
        .unwrap_or(("Echo", "∅".to_string()));

    format!("Baseline: {baseline_snippet}. Echo lift: {lens_label} {echo_snippet}. Pull which?")
}

fn snippet(text: &str, limit: usize) -> String {
    if text.is_empty() {
        return "∅".to_string();
    }

    let mut result = String::with_capacity(limit + 1);
    let mut count = 0;
    for ch in text.chars() {
        let ch = match ch {
            '\n' | '\r' | '\t' => ' ',
            other => other,
        };
        if count >= limit {
            result.push('…');
            break;
        }
        if ch == ' ' {
            if result.ends_with(' ') {
                continue;
            }
        }
        result.push(ch);
        count += 1;
    }

    result.trim().to_string()
}

impl GenerationEngine {
    /// Create with config
    pub fn new_with_config(
        endpoint: &str,
        model: &str,
        max_tokens: usize,
        _consistency_variance_threshold: f64,
        client_timeout_secs: u64,
    ) -> Result<Self> {
        let client = Client::builder().timeout(Duration::from_secs(client_timeout_secs)).build()?;

        // DEBUG: Log input model ID
        info!("GenerationEngine::new_with_config called with model={}", model);

        // Normalise model identifier: vLLM registers the served model by name, not path.
        let model_id = if model.starts_with("/home/beelink/models/") {
            model.replacen("/home/beelink/models/", "/workspace/models/", 1)
        } else if model.starts_with("/workspace/models/hf_cache/") {
            // Already a vLLM model ID, use as-is
            model.to_string()
        } else if model == "/workspace/models/Qwen2.5-7B-Instruct-AWQ" || model.contains("Qwen2.5-7B-Instruct-AWQ") {
            // Map old path to vLLM model ID
            "/workspace/models/hf_cache/models--Qwen--Qwen2.5-7B-Instruct-AWQ".to_string()
        } else if model.starts_with("/workspace/") {
            // Canonicalise workspace-relative paths to avoid stray symlink prefixes
            // But don't canonicalize if it's a known model path that needs mapping
            Path::new(model)
                .canonicalize()
                .ok()
                .and_then(|p| {
                    let path_str = p.to_str()?;
                    // Check if canonicalized path still needs mapping
                    if path_str.contains("Qwen2.5-7B-Instruct-AWQ") && !path_str.contains("hf_cache") {
                        Some("/workspace/models/hf_cache/models--Qwen--Qwen2.5-7B-Instruct-AWQ".to_string())
                    } else {
                        Some(path_str.to_string())
                    }
                })
                .unwrap_or_else(|| model.to_string())
        } else {
            model.to_string()
        };

        let circuit_breaker =
            Arc::new(CircuitBreaker::new("vllm", CircuitBreakerConfig::default()));

        info!("After normalization: model_id={}", model_id);

        Ok(Self {
            client,
            endpoint: endpoint.to_string(),
            model: model_id,
            temperature: 0.7,
            top_p: 0.9,
            max_tokens,
            mock_mode: std::env::var("MOCK_MODE")
                .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
                .unwrap_or(false),
            circuit_breaker,
            client_timeout_secs,
            config: None,
        })
    }

    /// Apply runtime config
    /// NOTE: Currently a stub - runtime config is applied during engine construction.
    /// This method exists for future dynamic config updates.
    pub fn apply_runtime_from_config(&mut self, _config: &crate::config::CliArgs) {
        // Stub implementation - config is applied during engine construction
        // Future: could update client timeout, circuit breaker settings, etc.
    }

    /// Update params
    pub fn update_params(&mut self, temperature: f64, top_p: f64) {
        self.temperature = temperature;
        self.top_p = top_p;
    }

    /// Set mock mode
    pub fn set_mock_mode(&mut self, mock: bool) {
        self.mock_mode = mock;
    }

    /// Set system prompt
    /// NOTE: Currently a stub - system prompts are handled via vLLM API configuration.
    /// This method exists for future client-side prompt management.
    pub fn set_system_prompt(&mut self, _prompt: String) {
        // Stub implementation - system prompts handled via vLLM API
        // Future: could inject system prompts into request payloads
    }

    /// Generate with params
    pub async fn generate_with_params(
        &self,
        prompt: &str,
        temperature: f64,
        top_p: f64,
    ) -> Result<String> {
        if self.mock_mode {
            return Ok(format!(
                "[Mock response to: {}]",
                prompt.chars().take(100).collect::<String>()
            ));
        }

        let mut temp_engine = Self {
            client: self.client.clone(),
            endpoint: self.endpoint.clone(),
            model: self.model.clone(),
            temperature,
            top_p,
            max_tokens: self.max_tokens,
            mock_mode: false,
            circuit_breaker: self.circuit_breaker.clone(),
            client_timeout_secs: self.client_timeout_secs,
            config: self.config.clone(),
        };
        temp_engine.request_text(prompt).await
    }

    /// Generate with consistency voting
    /// Generates three candidate responses with varying temperature/top_p and selects best via ROUGE
    pub async fn generate_with_consistency(
        &self,
        tokenizer: &crate::token_manager::TokenizerOutput,
        _compass: &crate::compass::CompassOutcome,
    ) -> Result<ConsistencyVotingResult> {
        use crate::util::rouge_l;
        
        let start = std::time::Instant::now();
        let prompt = &tokenizer.augmented_prompt;
        
        // Generate three candidates with different sampling parameters for diversity
        let candidate_1 = self.generate_with_params(prompt, self.temperature, self.top_p).await?;
        let candidate_2 = self.generate_with_params(prompt, self.temperature * 0.9, self.top_p * 0.95).await?;
        let candidate_3 = self.generate_with_params(prompt, self.temperature * 1.1, self.top_p * 1.05).await?;
        
        // Compute ROUGE-L scores against the prompt as reference
        let rouge_1 = rouge_l(prompt, &candidate_1);
        let rouge_2 = rouge_l(prompt, &candidate_2);
        let rouge_3 = rouge_l(prompt, &candidate_3);
        
        let rouge_scores = vec![rouge_1, rouge_2, rouge_3];
        
        // Select winner: highest ROUGE score, or first candidate if all equal
        let winner_index = rouge_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        
        Ok(ConsistencyVotingResult {
            candidate_1,
            candidate_2,
            candidate_3,
            rouge_scores,
            latency_ms,
            winner_index,
        })
    }

    /// Generate with topology
    /// Generate with fallback to mock if primary fails
    pub async fn generate_with_fallback(&self, prompt: &str) -> Result<(String, String)> {
        match self
            .generate_with_params(prompt, self.temperature, self.top_p)
            .await
        {
            Ok(response) => Ok((response, "primary".to_string())),
            Err(_) => {
                // Fallback to mock
                Ok((
                    format!("[Mock response to: {}]", prompt),
                    "mock".to_string(),
                ))
            }
        }
    }

    pub async fn generate_with_topology(
        &self,
        tokenizer: &crate::token_manager::TokenizerOutput,
        compass: &crate::compass::CompassOutcome,
        topology: Option<&crate::tcs_analysis::TopologicalSignature>,
        _use_cache: bool,
    ) -> Result<GenerationResult> {
        if self.mock_mode {
            // Return mock generation result with topology context
            let topology_info = if let Some(topo) = topology {
                format!(
                    " [Topology: knot={:.3}, betti={:?}]",
                    topo.knot_complexity, topo.betti_numbers
                )
            } else {
                String::new()
            };
            return Ok(GenerationResult {
                baseline_response: format!(
                    "[Mock baseline to: {}]{}",
                    tokenizer
                        .augmented_prompt
                        .chars()
                        .take(50)
                        .collect::<String>(),
                    topology_info
                ),
                hybrid_response: format!(
                    "[Mock hybrid response to: {}]{}",
                    tokenizer
                        .augmented_prompt
                        .chars()
                        .take(50)
                        .collect::<String>(),
                    topology_info
                ),
                echoes: vec![],
                rouge_to_baseline: 0.5,
                rouge_score: 0.5,
                latency_ms: 10.0,
                ucb1_score: None,
                source: "mock".to_string(),
                failure_type: None,
                failure_details: None,
                entropy_delta: 0.0,
                curator_quality: Some(0.7),
            });
        }

        // Build topology context for SYSTEM message only (not user-facing)
        let system_topology_context = if let Some(topo) = topology {
            format!(
                "[INTERNAL_TOPOLOGY_METRICS]\n\
                Knot Complexity: {:.3}\n\
                Betti Numbers: H0={}, H1={}, H2={}\n\
                Persistence Entropy: {:.3}\n\
                Spectral Gap: {:.3}\n\
                Euler Characteristic: {:.3}\n\
                Total Persistence: {:.3}\n\
                Max Persistence: {:.3}\n\
                Mean Persistence: {:.3}\n\
                \n\
                INTERPRETATION GUIDANCE:\n\
                - High knot complexity ({:.3}) → Complex conceptual relationships detected. Use structured, \
                  step-by-step reasoning. Avoid tangling multiple concepts.\n\
                - Betti numbers: H0={} (connected components), H1={} (loops/cycles), H2={} (cavities). \
                  High H1 suggests cyclical reasoning patterns. High H0 suggests multiple disconnected ideas.\n\
                - Persistence entropy ({:.3}) → Information structure diversity. Higher values suggest \
                  more varied conceptual connections. Use this to guide depth vs breadth of response.\n\
                - Spectral gap ({:.3}) → Topological stability. Higher values suggest more stable structure. \
                  Use this to determine confidence level.\n\
                \n\
                Use these metrics internally to adjust your reasoning style, but DO NOT mention them \
                or any topological terms in your response to the user.",
                topo.knot_complexity,
                topo.betti_numbers[0],
                topo.betti_numbers[1],
                topo.betti_numbers[2],
                topo.persistence_entropy,
                topo.spectral_gap,
                topo.euler_characteristic,
                topo.total_persistence,
                topo.max_persistence,
                topo.mean_persistence,
                topo.knot_complexity,
                topo.betti_numbers[0],
                topo.betti_numbers[1],
                topo.betti_numbers[2],
                topo.persistence_entropy,
                topo.spectral_gap
            )
        } else {
            String::new()
        };

        // Use original user prompt (don't add topology to user message)
        let user_prompt = &tokenizer.augmented_prompt;

        let start = Instant::now();
        let baseline_future = self.request_text_with_topology(user_prompt, topology.is_some(), &system_topology_context);
        let claude_future = self.request_lens_response_with_topology(
            "Claude".to_string(),
            Self::format_lens_prompt(
                user_prompt,
                "Respond with constitutional alignment and moral grounding.",
                compass,
            ),
            topology.is_some(),
            &system_topology_context,
        );
        let (baseline, claude) = tokio::try_join!(baseline_future, claude_future)?;

        let echoes: Vec<LensEcho> = vec![claude];
        let hybrid = synthesize_hybrid(&baseline, &echoes);
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        let rouge = rouge_l(&hybrid, &baseline);

        info!(latency_ms, rouge, topology_provided = topology.is_some(), "generated hybrid response with topology");

        Ok(GenerationResult {
            baseline_response: baseline,
            hybrid_response: hybrid,
            echoes,
            rouge_to_baseline: rouge,
            rouge_score: rouge,
            latency_ms,
            ucb1_score: None,
            source: "generation".to_string(),
            failure_type: None,
            failure_details: None,
            entropy_delta: 0.0,
            curator_quality: None,
        })
    }

    /// Retry generation with reflexion-style prompt repair.
    pub fn set_config(&mut self, config: Arc<RwLock<RuntimeConfig>>) {
        self.config = Some(config);
    }

    /// Retry generation with reflexion-style prompt repair.
    pub async fn reflexion_retry(
        &self,
        prompt: &str,
        baseline_rouge: f64,
        details: &str,
    ) -> Result<String> {
        let temp_base_multiplier = self.config.as_ref()
            .map(|c| c.read().generation_reflexion_temp_base_multiplier)
            .unwrap_or(0.7);
        let temp_stability_multiplier = self.config.as_ref()
            .map(|c| c.read().generation_reflexion_temp_stability_multiplier)
            .unwrap_or(0.3);
        let top_p_increment = self.config.as_ref()
            .map(|c| c.read().generation_reflexion_top_p_increment)
            .unwrap_or(0.05);
        let top_p_stability_increment = self.config.as_ref()
            .map(|c| c.read().generation_reflexion_top_p_stability_increment)
            .unwrap_or(0.2);
        let top_p_max = self.config.as_ref()
            .map(|c| c.read().generation_reflexion_top_p_max)
            .unwrap_or(0.99);
        
        let stability = 1.0_f64 - baseline_rouge.clamp(0.0, 1.0);
        let temperature = (self.temperature * temp_base_multiplier) + (temp_stability_multiplier * stability);
        let top_p = (self.top_p + top_p_increment + (top_p_stability_increment * stability)).min(top_p_max);
        let reflexion_prompt = format!(
            "{prompt}\n\n[Context]\nPrior attempt struggled because: {details}. Improve the response with clear reasoning, explicit decisions, and emotionally grounded alignment.]"
        );

        self.generate_with_params(&reflexion_prompt, temperature, top_p)
            .await
    }

    /// Apply a light chain-of-thought repair pass using topology awareness for guidance.
    pub async fn apply_cot_repair_with_topology(
        &self,
        prompt: &str,
        details: &str,
        iteration: u32,
        topology: Option<&crate::tcs_analysis::TopologicalSignature>,
    ) -> Result<GenerationResult> {
        let start = Instant::now();
        let mut repair_prompt = format!(
            "{prompt}\n\n[Repair Objective]\nIteration {iteration}: clarify reasoning, address: {details}."
        );

        if let Some(sig) = topology {
            use std::fmt::Write as _;
            let _ = write!(
                repair_prompt,
                "\nTopology cues → knot: {:.3}, spectral_gap: {:.3}, persistence_entropy: {:.3}. Use these cues to stabilize the narrative.",
                sig.knot_complexity, sig.spectral_gap, sig.persistence_entropy
            );
        }

        let guard = self.config.as_ref().map(|c| c.read());
        let temp_base_multiplier = guard.as_ref().map(|g| g.generation_cot_repair_temp_base_multiplier).unwrap_or(1.0);
        let temp_iteration_increment = guard.as_ref().map(|g| g.generation_cot_repair_temp_iteration_increment).unwrap_or(0.05);
        let top_p_increment = guard.as_ref().map(|g| g.generation_cot_repair_top_p_increment).unwrap_or(0.05);
        let top_p_max = guard.as_ref().map(|g| g.generation_cot_repair_top_p_max).unwrap_or(0.95);
        let temp_min = guard.as_ref().map(|g| g.generation_cot_repair_temp_min).unwrap_or(0.1);
        let temp_max = guard.as_ref().map(|g| g.generation_cot_repair_temp_max).unwrap_or(1.0);
        drop(guard);

        let temperature = (self.temperature * temp_base_multiplier) + (temp_iteration_increment * iteration as f64);
        let top_p = (self.top_p + top_p_increment).min(top_p_max);
        let repaired = self
            .generate_with_params(&repair_prompt, temperature.clamp(temp_min, temp_max), top_p)
            .await?;

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        let rouge = rouge_l(&repaired, prompt);

        Ok(GenerationResult {
            baseline_response: repaired.clone(),
            hybrid_response: repaired,
            echoes: Vec::new(),
            rouge_to_baseline: rouge,
            rouge_score: rouge,
            latency_ms,
            ucb1_score: None,
            source: format!("cot_repair_iter_{}", iteration),
            failure_type: None,
            failure_details: Some(details.to_string()),
            entropy_delta: 0.0,
            curator_quality: None,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ConsistencyVotingResult {
    pub candidate_1: String,
    pub candidate_2: String,
    pub candidate_3: String,
    pub rouge_scores: Vec<f64>,
    pub latency_ms: f64,
    pub winner_index: usize,
}

#[derive(Debug, Serialize, Clone)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f64,
    top_p: f64,
    max_tokens: usize,
}

#[derive(Debug, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Debug, Deserialize)]
struct ChatMessageResponse {
    content: Option<String>,
}
