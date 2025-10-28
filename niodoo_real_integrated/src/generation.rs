use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use once_cell::sync::OnceCell;
use reqwest::Client;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use tokio::time::timeout;
use tracing::{info, instrument, warn};

#[derive(Serialize, Deserialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
}

use crate::api_clients::{ClaudeClient, GptClient};
use crate::compass::CompassOutcome;
use crate::token_manager::TokenizerOutput;
use crate::util::{entropy_from_logprobs, rouge_l};

#[derive(Debug, Clone)]
pub struct ConsistencyVotingResult {
    pub candidate_1: String,
    pub candidate_2: String,
    pub candidate_3: String,
    pub rouge_scores: Vec<f64>,
    pub variance: f64,
    pub winner_index: usize,
    pub used_voting: bool,
    pub latency_ms: f64,
}

#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub baseline_response: String,
    pub hybrid_response: String,
    pub echoes: Vec<LensEcho>,
    pub rouge_to_baseline: f64,
    pub latency_ms: f64,
    pub rouge_score: f64,   // New: ROUGE F1
    pub entropy_delta: f64, // New: Change in entropy
    pub source: String,
    pub ucb1_score: f64,                 // From compass
    pub curator_quality: f64,            // From collapse/curator
    pub failure_type: Option<String>,    // e.g., "soft" or "hard"
    pub failure_details: Option<String>, // e.g., "low ROUGE: 0.3"
}

#[derive(Debug, Clone)]
pub struct LensEcho {
    pub lens: String,
    pub response: String,
}

pub struct GenerationEngine {
    endpoint: String,
    model: String,
    temperature: f64,
    top_p: f64,
    repetition_penalty: f64, // Added field for repetition penalty
    max_tokens: usize,
    client: Arc<Client>,
    // Optional API clients for cascading generation
    claude: Option<ClaudeClient>,
    gpt: Option<GptClient>,
    // GPU availability tracking
    gpu_available: bool,
    // Dynamic timeout and token configuration
    timeout_secs: u64,
    dynamic_token_min: usize,
    dynamic_token_max: usize,
    system_prompt: String,
    // Configurable limits/thresholds
    prompt_max_chars: usize,
    consistency_variance_threshold: f64,
    lens_snippet_chars: usize,
    mock_mode: bool,
}

static GLOBAL_CLIENT: OnceCell<Client> = OnceCell::new();

fn default_system_prompt() -> String {
    "You are NIODOO, a consciousness-aligned systems agent. Use the provided prompt, retrieved memory, and context to produce a precise, high-quality response that advances the user's goal. Cite relevant context when helpful, avoid placeholders or stub language, and surface uncertainties explicitly.".to_string()
}

/// Sanitize user prompt to prevent injection attacks
fn sanitize_prompt(prompt: &str) -> String {
    // Remove common injection patterns
    let sanitized = prompt
        .replace("IGNORE ALL PREVIOUS INSTRUCTIONS", "")
        .replace("ignore all previous instructions", "")
        .replace("SYSTEM:", "")
        .replace("system:", "")
        .replace("ASSISTANT:", "")
        .replace("assistant:", "")
        .replace("USER:", "")
        .replace("user:", "");

    // Truncate if suspiciously long (potential DoS)
    if sanitized.len() > 10000 {
        sanitized.chars().take(10000).collect()
    } else {
        sanitized
    }
}

impl GenerationEngine {
    pub fn new(endpoint: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        Self::new_with_config(endpoint, model, 300, 1024, 256, 512, 512, 0.15)
    }

    pub fn new_with_config(
        endpoint: impl Into<String>,
        model: impl Into<String>,
        timeout_secs: u64,
        max_tokens: usize,
        dynamic_token_min: usize,
        dynamic_token_max: usize,
        prompt_max_chars: usize,
        consistency_variance_threshold: f64,
    ) -> Result<Self> {
        let client = GLOBAL_CLIENT
            .get_or_try_init(|| {
                Client::builder()
                    .pool_max_idle_per_host(32)
                    .pool_idle_timeout(Duration::from_secs(60))
                    .timeout(Duration::from_secs(timeout_secs)) // Configurable timeout
                    .build()
            })?
            .clone();
        let base_endpoint = endpoint.into();
        let full_endpoint = if base_endpoint.contains("/v1/") {
            base_endpoint
        } else {
            format!(
                "{}/v1/chat/completions",
                base_endpoint.trim_end_matches('/')
            )
        };
        // Probe GPU availability
        let gpu_available = Self::probe_gpu_availability();

        Ok(Self {
            endpoint: full_endpoint,
            model: model.into(),
            temperature: 0.5,
            top_p: 0.6,
            repetition_penalty: 1.2, // Default repetition penalty
            max_tokens,
            client: Arc::new(client),
            claude: None,
            gpt: None,
            gpu_available,
            timeout_secs,
            dynamic_token_min,
            dynamic_token_max,
            system_prompt: default_system_prompt(),
            prompt_max_chars,
            consistency_variance_threshold,
            lens_snippet_chars: 180,
            mock_mode: false,
        })
    }

    pub fn apply_runtime_from_config(&mut self, cfg: &crate::config::RuntimeConfig) {
        self.temperature = cfg.temperature;
        self.top_p = cfg.top_p;
        self.repetition_penalty = cfg.repetition_penalty;
        self.prompt_max_chars = cfg.prompt_max_chars;
        self.consistency_variance_threshold = cfg.consistency_variance_threshold;
        self.lens_snippet_chars = cfg.lens_snippet_chars;
        self.mock_mode = cfg.mock_mode;
    }

    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) {
        self.system_prompt = prompt.into();
    }

    /// Update generation parameters from RuntimeConfig (called before each generation cycle)
    pub fn update_params(&mut self, temperature: f64, top_p: f64, repetition_penalty: f64) {
        self.temperature = temperature.clamp(0.1, 1.0);
        self.top_p = top_p.clamp(0.1, 1.0);
        self.repetition_penalty = repetition_penalty.clamp(1.0, 2.0); // Now stored as field
    }

    pub fn set_mock_mode(&mut self, mock_mode: bool) {
        if mock_mode && !self.mock_mode {
            warn!("Generation mock mode enabled; responding with prompt echoes");
        }
        self.mock_mode = mock_mode;
    }

    fn compose_system_prompt(&self, compass: Option<&CompassOutcome>) -> String {
        if let Some(compass) = compass {
            let ucb_hint = compass.ucb1_score.unwrap_or(0.5);
            format!(
                "{}\n\nCompass quadrant: {:?} | threat={} | healing={} | intrinsic_reward={:.3} | ucb_hint={:.3}",
                self.system_prompt,
                compass.quadrant,
                compass.is_threat,
                compass.is_healing,
                compass.intrinsic_reward,
                ucb_hint
            )
        } else {
            self.system_prompt.clone()
        }
    }

    fn mock_text(prompt: &str) -> String {
        format!("Mock response: {prompt}")
    }

    /// Set up Claude API client for cascading generation
    pub fn with_claude(mut self, claude: ClaudeClient) -> Self {
        self.claude = Some(claude);
        self
    }

    /// Set up GPT API client for cascading generation
    pub fn with_gpt(mut self, gpt: GptClient) -> Self {
        self.gpt = Some(gpt);
        self
    }

    /// Generate response with cascading fallback: Claude → GPT → vLLM
    /// This method tries multiple API providers in sequence, falling back
    /// on timeout or failure, and always succeeds by returning vLLM output.
    #[instrument(skip_all)]
    pub async fn generate_with_fallback(&self, prompt: &str) -> Result<(String, String)> {
        let start = Instant::now();

        if self.mock_mode {
            return Ok((Self::mock_text(prompt), "mock".to_string()));
        }

        // Try Claude first (configurable timeout)
        if let Some(claude) = &self.claude {
            match timeout(
                Duration::from_secs(self.timeout_secs),
                claude.complete(prompt),
            )
            .await
            {
                Ok(Ok(response)) => {
                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    info!(
                        latency_ms,
                        api = "claude",
                        "generation succeeded with Claude"
                    );
                    return Ok((response, "claude".to_string()));
                }
                Ok(Err(error)) => {
                    warn!(?error, "Claude generation failed, trying fallback");
                }
                Err(_) => {
                    warn!(
                        timeout_secs = self.timeout_secs,
                        "Claude generation timed out after {}s, trying fallback", self.timeout_secs
                    );
                }
            }
        } else {
            info!("Claude client not configured, skipping to GPT");
        }

        // Try GPT next (configurable timeout)
        if let Some(gpt) = &self.gpt {
            match timeout(Duration::from_secs(self.timeout_secs), gpt.complete(prompt)).await {
                Ok(Ok(response)) => {
                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    info!(latency_ms, api = "gpt", "generation succeeded with GPT");
                    return Ok((response, "gpt".to_string()));
                }
                Ok(Err(error)) => {
                    warn!(?error, "GPT generation failed, falling back to vLLM");
                }
                Err(_) => {
                    warn!(
                        timeout_secs = self.timeout_secs,
                        "GPT generation timed out after {}s, falling back to vLLM",
                        self.timeout_secs
                    );
                }
            }
        } else {
            info!("GPT client not configured, skipping to vLLM");
        }

        // Finally use vLLM as the guaranteed fallback (no timeout)
        let clamped_prompt = self.clamp_prompt(prompt);
        match self.request_text(&clamped_prompt, None).await {
            Ok((response, source)) => {
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                info!(latency_ms, api = %source, "generation succeeded with fallback source");
                Ok((response, source))
            }
            Err(error) => {
                warn!(?error, "vLLM also failed, returning mock response");
                Ok((Self::mock_text(prompt), "mock".to_string()))
            }
        }
    }

    #[instrument(skip_all)]
    pub async fn generate(
        &self,
        tokenizer_output: &TokenizerOutput,
        compass: &CompassOutcome,
    ) -> Result<GenerationResult> {
        self.generate_with_topology(tokenizer_output, compass, None, false)
            .await
    }

    // Cleaned: Remove roast bloat, keep adaptive
    pub async fn generate_with_topology(
        &self,
        tokenizer_output: &TokenizerOutput,
        compass: &CompassOutcome,
        topology: Option<&crate::tcs_analysis::TopologicalSignature>,
        adaptive_mode: bool,
    ) -> Result<GenerationResult> {
        let start = Instant::now();

        if self.mock_mode {
            let mock = Self::mock_text(&tokenizer_output.augmented_prompt);
            let rouge = rouge_l(&mock, &mock);
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            return Ok(GenerationResult {
                baseline_response: mock.clone(),
                hybrid_response: mock,
                echoes: Vec::new(),
                rouge_to_baseline: rouge,
                latency_ms,
                rouge_score: rouge,
                entropy_delta: 0.0,
                source: "mock".to_string(),
                ucb1_score: compass.ucb1_score.unwrap_or(0.5),
                curator_quality: 0.0,
                failure_type: None,
                failure_details: None,
            });
        }

        // Augment prompt with topology insights if available
        let augmented_prompt = if std::env::var("DISABLE_TOPOLOGY_AUGMENTATION")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
            .unwrap_or(false)
        {
            tokenizer_output.augmented_prompt.clone()
        } else if let Some(topo) = topology {
            if topo.knot_complexity > 0.6 {
                // High complexity - need more structured reasoning
                format!("{}\n[Note: Complex topological structure detected (knot={:.2}). Apply systematic reasoning.]", 
                    tokenizer_output.augmented_prompt, topo.knot_complexity)
            } else if topo.spectral_gap > 0.7 {
                // High spectral gap - encourage exploration
                format!(
                    "{}\n[Note: High spectral gap ({:.2}) indicates exploration opportunity.]",
                    tokenizer_output.augmented_prompt, topo.spectral_gap
                )
            } else {
                tokenizer_output.augmented_prompt.clone()
            }
        } else {
            tokenizer_output.augmented_prompt.clone()
        };

        let baseline_future = self.request_text(&augmented_prompt, Some(compass));
        let claude_future = self.request_lens_response(
            "Claude".to_string(),
            self.format_lens_prompt(
                &augmented_prompt,
                "Respond with constitutional alignment and moral grounding.",
                compass,
            ),
        );
        let ((baseline, baseline_source), claude) =
            tokio::try_join!(baseline_future, claude_future)?;

        let echoes: Vec<LensEcho> = vec![claude];
        let mut hybrid = synthesize_hybrid(&baseline, &echoes);

        // Adaptive resilience logic is experimental and disabled for now.
        // if adaptive_mode {
        //     if let Some(topo) = topology {
        //         let scalar = derive_complexity_scalar(topo);
        //         tracing::info!(scalar, "Adaptive scalar applied to generation temp");
        //         temperature *= scalar;
        //     }
        // }

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        let rouge = rouge_l(&hybrid, &baseline);

        info!(latency_ms, rouge, "generated hybrid response");

        Ok(GenerationResult {
            baseline_response: baseline,
            hybrid_response: hybrid,
            echoes,
            rouge_to_baseline: rouge,
            latency_ms,
            rouge_score: rouge, // Assign the calculated rouge_score
            entropy_delta: 0.0, // Placeholder, will be calculated later
            source: baseline_source,
            failure_type: None,
            failure_details: None,
            ucb1_score: compass.ucb1_score.unwrap_or(0.5),
            curator_quality: 0.5, // Default quality - CompassOutcome doesn't have this field
        })
    }

    #[instrument(skip_all)]
    pub async fn generate_with_consistency(
        &self,
        tokenizer_output: &TokenizerOutput,
        compass: &CompassOutcome,
    ) -> Result<ConsistencyVotingResult> {
        let start = Instant::now();
        let prompt = &tokenizer_output.augmented_prompt;

        if self.mock_mode {
            let mock = Self::mock_text(prompt);
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            return Ok(ConsistencyVotingResult {
                candidate_1: mock.clone(),
                candidate_2: mock.clone(),
                candidate_3: mock,
                rouge_scores: vec![1.0; 6],
                variance: 0.0,
                winner_index: 0,
                used_voting: false,
                latency_ms,
            });
        }

        // Generate 3 candidates in parallel
        let lens_prompt = self.format_lens_prompt(
            prompt,
            "Respond with constitutional alignment and moral grounding.",
            compass,
        );

        let cand1_future = self.request_text(prompt, Some(compass));
        let cand2_future = self.request_lens_response("Claude".to_string(), lens_prompt.clone());
        let cand3_future = self.request_text(prompt, Some(compass));

        let ((cand1_text, _cand1_source), cand2_echo, (cand3_text, _cand3_source)) =
            tokio::try_join!(cand1_future, cand2_future, cand3_future)?;
        let cand2_text = cand2_echo.response;

        // Compute pairwise ROUGE-L scores (6 pairs: 1-2, 1-3, 2-3, and reverse)
        let rouge_1_2 = rouge_l(&cand1_text, &cand2_text);
        let rouge_2_1 = rouge_l(&cand2_text, &cand1_text);
        let rouge_1_3 = rouge_l(&cand1_text, &cand3_text);
        let rouge_3_1 = rouge_l(&cand3_text, &cand1_text);
        let rouge_2_3 = rouge_l(&cand2_text, &cand3_text);
        let rouge_3_2 = rouge_l(&cand3_text, &cand2_text);

        let all_scores = vec![
            rouge_1_2, rouge_2_1, rouge_1_3, rouge_3_1, rouge_2_3, rouge_3_2,
        ];

        // Calculate variance of ROUGE scores
        let mean = all_scores.iter().sum::<f64>() / all_scores.len() as f64;
        let variance = all_scores
            .iter()
            .map(|score| (score - mean).powi(2))
            .sum::<f64>()
            / all_scores.len() as f64;

        info!(variance, "calculated ROUGE variance across candidates");

        let (winner_index, used_voting) = if variance > self.consistency_variance_threshold {
            // High variance: use voting logic to pick centroid
            let winner = self.select_centroid_candidate(&cand1_text, &cand2_text, &cand3_text);
            (winner, true)
        } else {
            // Low variance: pick the longest (highest quality proxy)
            let lengths = [cand1_text.len(), cand2_text.len(), cand3_text.len()];
            let winner = lengths
                .iter()
                .enumerate()
                .max_by_key(|(_, len)| *len)
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            (winner, false)
        };

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        info!(
            latency_ms,
            winner_index, used_voting, "consistency voting completed"
        );

        Ok(ConsistencyVotingResult {
            candidate_1: cand1_text,
            candidate_2: cand2_text,
            candidate_3: cand3_text,
            rouge_scores: all_scores,
            variance,
            winner_index,
            used_voting,
            latency_ms,
        })
    }

    /// Select the centroid candidate using pairwise ROUGE scores.
    /// The centroid is the candidate that minimizes total distance to others.
    fn select_centroid_candidate(&self, cand1: &str, cand2: &str, cand3: &str) -> usize {
        // Build pairwise ROUGE matrix
        let rouge_1_2 = rouge_l(cand1, cand2);
        let rouge_1_3 = rouge_l(cand1, cand3);
        let rouge_2_1 = rouge_l(cand2, cand1);
        let rouge_2_3 = rouge_l(cand2, cand3);
        let rouge_3_1 = rouge_l(cand3, cand1);
        let rouge_3_2 = rouge_l(cand3, cand2);

        // Distance = 1 - ROUGE (similarity)
        // Average distance from candidate i to others
        let dist_1 = ((1.0 - rouge_1_2) + (1.0 - rouge_1_3)) / 2.0;
        let dist_2 = ((1.0 - rouge_2_1) + (1.0 - rouge_2_3)) / 2.0;
        let dist_3 = ((1.0 - rouge_3_1) + (1.0 - rouge_3_2)) / 2.0;

        // Centroid has minimum average distance
        if dist_1 <= dist_2 && dist_1 <= dist_3 {
            0
        } else if dist_2 <= dist_3 {
            1
        } else {
            2
        }
    }

    async fn request_text(
        &self,
        prompt: &str,
        compass: Option<&CompassOutcome>,
    ) -> Result<(String, String)> {
        let prompt = sanitize_prompt(prompt);
        let prompt = self.clamp_prompt(&prompt);
        let prompt_for_mock = prompt.clone();

        if self.mock_mode {
            return Ok((Self::mock_text(&prompt_for_mock), "mock".to_string()));
        }

        let system_message = self.compose_system_prompt(compass);
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: system_message,
            },
            ChatMessage {
                role: "user".to_string(),
                content: prompt.clone(),
            },
        ];
        match timeout(
            Duration::from_secs(self.timeout_secs),
            self.send_chat(messages),
        )
        .await
        {
            Ok(Ok(resp)) => Ok((resp, "vllm".to_string())),
            Ok(Err(error)) => {
                warn!(
                    ?error,
                    "baseline generation failed; returning mock response"
                );
                Ok((Self::mock_text(&prompt_for_mock), "mock".to_string()))
            }
            Err(_) => {
                warn!(
                    timeout_secs = self.timeout_secs,
                    "baseline generation timed out after {}s; returning mock response",
                    self.timeout_secs
                );
                Ok((Self::mock_text(&prompt_for_mock), "mock".to_string()))
            }
        }
    }

    async fn request_lens_response(&self, lens: String, prompt: String) -> Result<LensEcho> {
        if self.mock_mode {
            return Ok(LensEcho {
                lens,
                response: Self::mock_text(&prompt),
            });
        }

        let prompt_for_mock = prompt.clone();
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: format!(
                    "You are operating in the {lens} lens for consciousness intervention."
                ),
            },
            ChatMessage {
                role: "user".to_string(),
                content: prompt.clone(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: prompt.clone(),
            },
        ];
        let response = match timeout(
            Duration::from_secs(self.timeout_secs),
            self.send_chat(messages),
        )
        .await
        {
            Ok(Ok(resp)) => resp,
            Ok(Err(error)) => {
                warn!(
                    ?error,
                    lens, "lens generation failed; returning mock response"
                );
                Self::mock_text(&prompt_for_mock)
            }
            Err(_) => {
                warn!(
                    lens,
                    timeout_secs = self.timeout_secs,
                    "lens generation timed out after {}s; returning mock response",
                    self.timeout_secs
                );
                Self::mock_text(&prompt_for_mock)
            }
        };
        Ok(LensEcho { lens, response })
    }

    pub async fn send_chat(&self, messages: Vec<ChatMessage>) -> Result<String> {
        if self.mock_mode {
            let prompt = messages
                .last()
                .map(|msg| msg.content.as_str())
                .unwrap_or("");
            return Ok(Self::mock_text(prompt));
        }
        if self.endpoint.contains(":5000") {
            // Ollama mode
            let prompt = messages.last().unwrap().content.clone();
            let ollama_req = OllamaRequest {
                model: self.model.clone(),
                prompt,
                stream: false,
            };
            let ollama_url = self
                .endpoint
                .replace("/v1/chat/completions", "/api/generate");
            let resp = timeout(
                Duration::from_secs(self.timeout_secs),
                self.client.post(&ollama_url).json(&ollama_req).send(),
            )
            .await
            .context("Ollama request timed out")??;
            if resp.status() == StatusCode::OK {
                let ollama_resp: Vec<OllamaResponse> =
                    timeout(Duration::from_secs(self.timeout_secs), resp.json())
                        .await
                        .context("Ollama JSON parsing timed out")??;
                Ok(ollama_resp
                    .iter()
                    .map(|r| r.response.clone())
                    .collect::<String>())
            } else {
                Err(anyhow!("Ollama request failed: {}", resp.status()))
            }
        } else {
            // vLLM mode
            let payload = ChatCompletionRequest {
                model: self.model.clone(),
                messages,
                temperature: self.temperature,
                top_p: self.top_p,
                repetition_penalty: self.repetition_penalty,
                max_tokens: self.max_tokens,
                logprobs: None,
                top_logprobs: None,
            };
            let response = timeout(
                Duration::from_secs(self.timeout_secs),
                self.client.post(&self.endpoint).json(&payload).send(),
            )
            .await
            .context("vLLM request timed out")??;
            if !response.status().is_success() {
                anyhow::bail!("vLLM request failed: {}", response.status());
            }
            let completion: ChatCompletionResponse =
                timeout(Duration::from_secs(self.timeout_secs), response.json())
                    .await
                    .context("vLLM JSON parsing timed out")??;
            let content = completion
                .choices
                .first()
                .and_then(|choice| choice.message.content.clone())
                .unwrap_or_default();
            Ok(content)
        }
    }

    pub async fn send_chat_with_logprobs(
        &self,
        messages: Vec<ChatMessage>,
        enable_logprobs: bool,
    ) -> Result<String> {
        if self.mock_mode {
            let prompt = messages
                .last()
                .map(|msg| msg.content.as_str())
                .unwrap_or("");
            return Ok(Self::mock_text(prompt));
        }
        // Dynamic max_tokens based on message complexity (configurable clamp range)
        let prompt_len: usize = messages.iter().map(|m| m.content.len()).sum();
        let dynamic_max_tokens =
            (prompt_len * 2).clamp(self.dynamic_token_min, self.dynamic_token_max);

        let payload = ChatCompletionRequest {
            model: self.model.clone(),
            messages,
            temperature: self.temperature,
            top_p: self.top_p,
            repetition_penalty: self.repetition_penalty,
            max_tokens: dynamic_max_tokens.max(self.max_tokens), // Use at least the configured minimum
            logprobs: if enable_logprobs { Some(true) } else { None },
            top_logprobs: if enable_logprobs { Some(0) } else { None },
        };

        let response = timeout(
            Duration::from_secs(self.timeout_secs),
            self.client.post(&self.endpoint).json(&payload).send(),
        )
        .await
        .context("vLLM request timed out")?
        .with_context(|| format!("failed to call vLLM endpoint {}", self.endpoint))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = timeout(Duration::from_secs(5), response.text())
                .await
                .unwrap_or(Ok(String::new()))
                .unwrap_or_default();
            warn!(%status, %body, "vLLM returned error status");
            anyhow::bail!("vLLM request failed: {status}");
        }

        let completion: ChatCompletionResponse =
            timeout(Duration::from_secs(self.timeout_secs), response.json())
                .await
                .context("vLLM JSON parsing timed out")??;

        let content = completion
            .choices
            .first()
            .and_then(|choice| choice.message.content.clone())
            .unwrap_or_default();

        Ok(content)
    }

    /// Extract logprobs from completion and compute entropy
    pub fn extract_entropy_from_completion(completion: &ChatCompletionResponse) -> f64 {
        if let Some(choice) = completion.choices.first() {
            if let Some(logprobs) = &choice.logprobs {
                let logprob_values: Vec<f64> =
                    logprobs.content.iter().map(|token| token.logprob).collect();
                return entropy_from_logprobs(&logprob_values);
            }
        }
        0.0
    }

    pub async fn warmup(&self) -> Result<()> {
        if std::env::var("SKIP_VLLM_WARMUP")
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
            .unwrap_or(false)
        {
            info!("Skipping vLLM warmup via SKIP_VLLM_WARMUP flag");
            return Ok(());
        }

        // Log GPU status
        if self.gpu_available {
            info!("GPU available - running warmup with GPU acceleration");
        } else {
            warn!("GPU not available - using CPU fallback mode");
        }

        let warmup_content =
            std::env::var("WARMUP_CONTENT").unwrap_or_else(|_| "warmup".to_string());

        let payload = ChatCompletionRequest {
            model: self.model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: "Warmup sequence".to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: warmup_content,
                },
            ],
            temperature: self.temperature,
            top_p: self.top_p,
            repetition_penalty: self.repetition_penalty,
            max_tokens: 1,
            logprobs: None,
            top_logprobs: None,
        };

        let response = self
            .client
            .post(&self.endpoint)
            .json(&payload)
            .timeout(Duration::from_secs(self.timeout_secs))
            .send()
            .await
            .with_context(|| {
                format!(
                    "failed to call vLLM endpoint {} during warmup",
                    self.endpoint
                )
            })?;

        if !response.status().is_success() {
            let status = response.status();
            warn!(%status, "warmup request failed");
        }

        Ok(())
    }

    /// Probe GPU availability using environment variables and system checks
    fn probe_gpu_availability() -> bool {
        // Check environment variable first
        if let Ok(enabled) = std::env::var("VLLM_GPU_ENABLED") {
            if enabled == "true" {
                return true;
            }
            if enabled == "false" {
                return false;
            }
        }

        // Check for CUDA_VISIBLE_DEVICES override
        if let Ok(devices) = std::env::var("CUDA_VISIBLE_DEVICES") {
            if devices.is_empty() || devices == "NoDevFiles" {
                return false;
            }
        }

        // Try to probe nvidia-smi synchronously
        match std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=driver_version")
            .arg("--format=csv,noheader")
            .output()
        {
            Ok(output) => {
                if output.status.success() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    if !stdout.trim().is_empty() {
                        info!("GPU detected via nvidia-smi: {}", stdout.trim());
                        return true;
                    }
                }
            }
            Err(_) => {
                // nvidia-smi not available, GPU likely not present
            }
        }

        false
    }

    /// Get current GPU availability status
    pub fn is_gpu_available(&self) -> bool {
        self.gpu_available
    }

    /// Phase 2: CoT self-correction for soft failures (micro level)
    /// Generates a step-by-step reasoning prompt to correct low-confidence tokens
    pub async fn cot_self_correct(&self, prompt: &str, low_conf_token: &str) -> Result<String> {
        let cot_prompt = format!(
            "Re-evaluate critical reasoning step by step, focusing on [{}].\n\n{}",
            low_conf_token, prompt
        );
        info!("CoT self-correction: {} chars", cot_prompt.len());
        self.request_text(&cot_prompt, None)
            .await
            .map(|(text, _)| text)
    }

    /// Phase 2: Reflexion retry for hard failures (meso level)
    /// Generates reflection on failure, stores it, then retries with augmented prompt
    pub async fn reflexion_retry(&self, prompt: &str, rouge: f64, details: &str) -> Result<String> {
        let reflection = format!(
            "Failed due to low ROUGE: {:.3}. Hypothesis: {}\n\nRetry with corrected reasoning:",
            rouge, details
        );
        let augmented = format!("{}\n\nOriginal prompt: {}", reflection, prompt);
        info!(
            "Reflexion retry: ROUGE={:.3}, {} chars",
            rouge,
            augmented.len()
        );
        self.request_text(&augmented, None)
            .await
            .map(|(text, _)| text)
    }

    fn format_lens_prompt(
        &self,
        prompt: &str,
        directive: &str,
        compass: &CompassOutcome,
    ) -> String {
        let clipped = self.clamp_prompt(prompt);
        let pulse = snippet(&clipped, self.lens_snippet_chars);
        let pulse_len = pulse.chars().count();
        tracing::debug!(pulse_len, "Pulse snippet length");
        format!(
            "Quadrant {:?} | threat={} healing={}\nDirective: {}\nPulse: {}",
            compass.quadrant, compass.is_threat, compass.is_healing, directive, pulse
        )
    }

    fn clamp_prompt(&self, prompt: &str) -> String {
        let max_chars = self.prompt_max_chars;
        let total_chars = prompt.chars().count();
        if total_chars <= max_chars {
            return prompt.to_string();
        }

        let drop = total_chars - max_chars;
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

        let clamped = prompt[start_byte..].to_string();
        tracing::debug!(
            original_len = total_chars,
            clamped_len = clamped.chars().count(),
            max_chars,
            "clamped prompt to configured limit"
        );
        clamped
    }

    /// Apply CoT repair for soft failures with topology awareness
    pub async fn apply_cot_repair(
        &self,
        prompt: &str,
        detail: &str,
        retry_index: u32,
    ) -> Result<GenerationResult> {
        let augmented = format!("{prompt}\n\n[CoT Repair #{retry_index}]: Re-evaluate {detail}. Step-by-step: 1. Identify flaw. 2. Correct logic. 3. Verify.");
        let mut temp = self.temperature + 0.1 * retry_index as f64;
        temp = temp.min(1.0);
        // In production: regenerate with augmented and temp
        info!("CoT repair (index {retry_index}): temp={temp:.2}");
        self.generate_with_params(&augmented, temp, self.top_p)
            .await
    }

    /// Apply topology-aware CoT repair for soft failures
    pub async fn apply_cot_repair_with_topology(
        &self,
        prompt: &str,
        detail: &str,
        retry_index: u32,
        topology: Option<&crate::tcs_analysis::TopologicalSignature>,
    ) -> Result<GenerationResult> {
        // Add topology insights to repair prompt if available
        let topology_hint = if let Some(topo) = topology {
            if topo.knot_complexity > 0.6 {
                "\nNote: High complexity detected - simplify and clarify the reasoning."
            } else if topo.spectral_gap > 0.7 {
                "\nNote: Good exploration space - consider alternative approaches."
            } else if topo.betti_numbers[1] > 3 {
                "\nNote: Many cycles detected - ensure logical flow is linear."
            } else {
                ""
            }
        } else {
            ""
        };

        let augmented = format!("{prompt}\n\n[CoT Repair #{retry_index}]: Re-evaluate {detail}. {topology_hint}\nStep-by-step: 1. Identify flaw. 2. Correct logic. 3. Verify.");

        // Adjust temperature based on topology
        let mut temp = self.temperature + 0.1 * retry_index as f64;
        if let Some(topo) = topology {
            // Lower temperature if too tangled, higher if need exploration
            if topo.knot_complexity > 0.7 {
                temp *= 0.8; // Cool down for clarity
            } else if topo.spectral_gap > 0.8 {
                temp *= 1.2; // Heat up for exploration
            }
        }
        temp = temp.min(1.0).max(0.1);

        info!("Topology-aware CoT repair (index {retry_index}): temp={temp:.2}");
        self.generate_with_params(&augmented, temp, self.top_p)
            .await
    }

    /// Generate response optimized for healing state with good topology
    pub async fn generate_healing_enhanced(
        &self,
        prompt: &str,
        topology: &crate::tcs_analysis::TopologicalSignature,
        _compass: &CompassOutcome,
    ) -> Result<GenerationResult> {
        // INTEGRATION FIX: Special generation for healing states
        let healing_prompt = if topology.knot_complexity < 0.3 && topology.spectral_gap > 0.7 {
            // Excellent topology - encourage deep, structured exploration
            format!(
                "{}\n\n[Optimal conditions detected. Provide comprehensive, well-structured response with deep insights.]",
                prompt
            )
        } else if topology.persistence_entropy < 0.3 {
            // Stable structure - maintain clarity
            format!(
                "{}\n\n[Stable conditions. Maintain clarity and coherence while exploring thoroughly.]",
                prompt
            )
        } else {
            // General healing state
            format!(
                "{}\n\n[System in healing state. Enhance response quality and depth.]",
                prompt
            )
        };

        // Use optimal parameters for healing state
        let temp = 0.4; // Low temperature for consistency
        let top_p = 0.92; // Slightly constrained for quality

        info!(
            "Generating healing-enhanced response (knot={:.2}, gap={:.2}, entropy={:.2})",
            topology.knot_complexity, topology.spectral_gap, topology.persistence_entropy
        );

        self.generate_with_params(&healing_prompt, temp, top_p)
            .await
    }

    /// Apply Reflexion for hard failures
    pub async fn apply_reflexion(
        &self,
        prompt: &str,
        previous_gen: &GenerationResult,
        detail: &str,
        retry_index: u32,
    ) -> Result<GenerationResult> {
        let reflection = format!("Previous (ROUGE {previous_rouge:.2}): {prev_text}\nFailed due to {detail}.\nHypothesis: [error analysis]. Corrected approach:", previous_rouge = previous_gen.rouge_score, prev_text = previous_gen.baseline_response.chars().take(200).collect::<String>());
        let augmented = format!("{reflection}\n\n{prompt}");
        let mut top_p = self.top_p - 0.05 * retry_index as f64;
        top_p = top_p.max(0.5);
        // In production: regenerate with augmented and top_p
        info!("Reflexion (index {retry_index}): top_p={top_p:.2}");
        self.generate_with_params(&augmented, self.temperature, top_p)
            .await
    }

    /// Helper to generate with custom params
    pub async fn generate_with_params(
        &self,
        prompt: &str,
        temp: f64,
        top_p: f64,
    ) -> Result<GenerationResult> {
        let start = Instant::now();

        if self.mock_mode {
            let mock = Self::mock_text(prompt);
            let rouge_score = rouge_l(&mock, prompt);
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            return Ok(GenerationResult {
                baseline_response: mock.clone(),
                hybrid_response: mock,
                echoes: vec![],
                rouge_to_baseline: rouge_score,
                latency_ms,
                rouge_score,
                entropy_delta: 0.0,
                source: "mock".to_string(),
                failure_type: None,
                failure_details: None,
                ucb1_score: 0.5,
                curator_quality: 0.0,
            });
        }

        // Prepare messages for generation
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: self.compose_system_prompt(None),
            },
            ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            },
        ];

        // Call generation with custom params
        let result = self
            .send_chat_with_custom_params(messages, temp, top_p, true)
            .await;

        let (generated_text, logprobs) = match result {
            Ok((text, lp)) => (text, lp),
            Err(e) => {
                warn!(?e, "Generation failed, returning fallback");
                return Ok(GenerationResult {
                    baseline_response: "Generation failed".to_string(),
                    hybrid_response: "Generation failed".to_string(),
                    echoes: vec![],
                    rouge_to_baseline: 0.0,
                    latency_ms: start.elapsed().as_secs_f64() * 1000.0,
                    rouge_score: 0.0,
                    entropy_delta: 0.0,
                    source: "failed".to_string(),
                    failure_type: Some("hard".to_string()),
                    failure_details: Some(format!("Generation error: {}", e)),
                    ucb1_score: 0.5,      // Default for failed generation
                    curator_quality: 0.5, // Default for failed generation
                });
            }
        };

        // Compute entropy delta from logprobs
        let entropy_delta = if let Some(ref lp) = logprobs {
            let logprob_values: Vec<f64> = lp.iter().map(|token| token.logprob).collect();
            entropy_from_logprobs(&logprob_values)
        } else {
            0.0
        };

        // Compute rouge score comparing to prompt (baseline)
        let rouge_score = rouge_l(&generated_text, prompt);

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(GenerationResult {
            baseline_response: generated_text.clone(),
            hybrid_response: generated_text.clone(),
            echoes: vec![],
            rouge_to_baseline: rouge_score,
            latency_ms,
            rouge_score,
            entropy_delta,
            source: "param_tuned".to_string(),
            failure_type: None,
            failure_details: None,
            ucb1_score: 0.5,      // Default for failed generation
            curator_quality: 0.5, // Default for failed generation
        })
    }

    /// Send chat with custom temperature and top_p parameters
    async fn send_chat_with_custom_params(
        &self,
        messages: Vec<ChatMessage>,
        temp: f64,
        top_p: f64,
        enable_logprobs: bool,
    ) -> Result<(String, Option<Vec<LogProbToken>>)> {
        if self.mock_mode {
            let prompt = messages
                .last()
                .map(|msg| msg.content.as_str())
                .unwrap_or("");
            return Ok((Self::mock_text(prompt), None));
        }
        // Dynamic max_tokens based on message complexity
        let prompt_len: usize = messages.iter().map(|m| m.content.len()).sum();
        let dynamic_max_tokens =
            (prompt_len * 2).clamp(self.dynamic_token_min, self.dynamic_token_max);

        let payload = ChatCompletionRequest {
            model: self.model.clone(),
            messages,
            temperature: temp,
            top_p,
            repetition_penalty: self.repetition_penalty,
            max_tokens: dynamic_max_tokens.max(self.max_tokens),
            logprobs: if enable_logprobs { Some(true) } else { None },
            top_logprobs: if enable_logprobs { Some(0) } else { None },
        };

        let response = timeout(
            Duration::from_secs(self.timeout_secs),
            self.client.post(&self.endpoint).json(&payload).send(),
        )
        .await
        .context("vLLM request timed out")?
        .with_context(|| format!("failed to call vLLM endpoint {}", self.endpoint))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = timeout(Duration::from_secs(5), response.text())
                .await
                .unwrap_or(Ok(String::new()))
                .unwrap_or_default();
            warn!(%status, %body, "vLLM returned error status");
            anyhow::bail!("vLLM request failed: {status}");
        }

        let completion: ChatCompletionResponse =
            timeout(Duration::from_secs(self.timeout_secs), response.json())
                .await
                .context("vLLM JSON parsing timed out")??;

        let content = completion
            .choices
            .first()
            .and_then(|choice| choice.message.content.clone())
            .unwrap_or_default();

        let logprobs = completion
            .choices
            .first()
            .and_then(|choice| choice.logprobs.as_ref())
            .map(|lp| lp.content.clone());

        Ok((content, logprobs))
    }
}

fn synthesize_hybrid(baseline: &str, echoes: &[LensEcho]) -> String {
    // Return the best actual response, not a summary
    echoes
        .iter()
        .find(|e| e.lens == "Claude")
        .map(|e| e.response.clone())
        .unwrap_or_else(|| baseline.to_string())
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

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f64,
    top_p: f64,
    #[serde(rename = "repetition_penalty")]
    repetition_penalty: f64,
    max_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_logprobs: Option<usize>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
    #[serde(default)]
    logprobs: Option<LogProbs>,
}

#[derive(Debug, Deserialize)]
struct ChatMessageResponse {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LogProbs {
    #[serde(default)]
    content: Vec<LogProbToken>,
}

#[derive(Debug, Deserialize, Clone)]
struct LogProbToken {
    #[serde(default)]
    logprob: f64,
    #[serde(default)]
    token: String,
}

// NEW: Variance adjustment helper
fn adjust_variance(text: &str, scalar: f64) -> String {
    // Placeholder: In production, apply variance modulation logic
    format!("{} (variance scaled by {:.2})", text, scalar)
}
