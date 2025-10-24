use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use once_cell::sync::OnceCell;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::time::timeout;
use tracing::{info, instrument, warn};
use reqwest::StatusCode;

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
use crate::tokenizer::TokenizerOutput;
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
    pub ucb1_score: f64, // From compass
    pub curator_quality: f64, // From collapse/curator
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
}

static GLOBAL_CLIENT: OnceCell<Client> = OnceCell::new();

impl GenerationEngine {
    pub fn new(endpoint: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        Self::new_with_config(endpoint, model, 300, 1024, 256, 512)
    }

    pub fn new_with_config(
        endpoint: impl Into<String>,
        model: impl Into<String>,
        timeout_secs: u64,
        max_tokens: usize,
        dynamic_token_min: usize,
        dynamic_token_max: usize,
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
            max_tokens,
            client: Arc::new(client),
            claude: None,
            gpt: None,
            gpu_available,
            timeout_secs,
            dynamic_token_min,
            dynamic_token_max,
        })
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
                    warn!("Claude generation timed out after 5s, trying fallback");
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
                    warn!("GPT generation timed out after 5s, falling back to vLLM");
                }
            }
        } else {
            info!("GPT client not configured, skipping to vLLM");
        }

        // Finally use vLLM as the guaranteed fallback (no timeout)
        let clamped_prompt = Self::clamp_prompt(prompt);
        match self.request_text(&clamped_prompt).await {
            Ok((response, source)) => {
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                info!(latency_ms, api = %source, "generation succeeded with fallback source");
                Ok((response, source))
            }
            Err(error) => {
                warn!(?error, "vLLM also failed, returning error");
                Err(error).context("all generation APIs failed")
            }
        }
    }

    #[instrument(skip_all)]
    pub async fn generate(
        &self,
        tokenizer_output: &TokenizerOutput,
        compass: &CompassOutcome,
    ) -> Result<GenerationResult> {
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
        let ((baseline, baseline_source), claude) =
            tokio::try_join!(baseline_future, claude_future)?;

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
            latency_ms,
            rouge_score: rouge, // Assign the calculated rouge_score
            entropy_delta: 0.0, // Placeholder, will be calculated later
            source: baseline_source,
            failure_type: None,
            failure_details: None,
            ucb1_score: compass.ucb1_score.unwrap_or(0.5),
            curator_quality: compass.curator_quality.unwrap_or(0.5),
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

        // Generate 3 candidates in parallel
        let lens_prompt = Self::format_lens_prompt(
            prompt,
            "Respond with constitutional alignment and moral grounding.",
            compass,
        );

        let cand1_future = self.request_text(prompt);
        let cand2_future = self.request_lens_response("Claude".to_string(), lens_prompt.clone());
        let cand3_future = self.request_text(prompt);

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

        let (winner_index, used_voting) = if variance > 0.15 {
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

    async fn request_text(&self, prompt: &str) -> Result<(String, String)> {
        let prompt = Self::clamp_prompt(prompt);
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "Implement the EXACT 3D labyrinth solver in Python using Dijkstra with priority queue. Use the provided 7x7x7 grid, handle echo chambers with state (consec_echoes 0-3, attune_timer 0-3, multiplier starting at 1, doubles on distract, halves on attune). Ignore any metaphorical language; focus strictly on implementing the algorithm as described. Output ONLY the complete code showing Path list, Total cost (expected 46), Steps. No text or philosophy.\n\"Include full grid definition and state logic as specified.\"".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: prompt,
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
                    "baseline generation failed; returning fallback text"
                );
                Ok((
                    "Baseline response unavailable (timeout)".to_string(),
                    "fallback".to_string(),
                ))
            }
            Err(_) => {
                warn!("baseline generation timed out after 5s; returning fallback text");
                Ok((
                    "Baseline response unavailable (timeout)".to_string(),
                    "fallback".to_string(),
                ))
            }
        }
    }

    async fn request_lens_response(&self, lens: String, prompt: String) -> Result<LensEcho> {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: format!(
                    "You are operating in the {lens} lens for consciousness intervention."
                ),
            },
            ChatMessage {
                role: "user".to_string(),
                content: prompt,
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
                    lens, "lens generation failed; returning fallback text"
                );
                "Lens response unavailable (timeout)".to_string()
            }
            Err(_) => {
                warn!(
                    lens,
                    "lens generation timed out after 5s; returning fallback text"
                );
                "Lens response unavailable (timeout)".to_string()
            }
        };
        Ok(LensEcho { lens, response })
    }

    pub async fn send_chat(&self, messages: Vec<ChatMessage>) -> Result<String> {
        if self.endpoint.contains(":11434") {
            // Ollama mode
            let prompt = messages.last().unwrap().content.clone();
            let ollama_req = OllamaRequest {
                model: self.model.clone(),
                prompt,
                stream: false,
            };
            let ollama_url = self.endpoint.replace("/v1/chat/completions", "/api/generate");
            let resp = self.client.post(&ollama_url).json(&ollama_req).send().await?;
            if resp.status() == StatusCode::OK {
                let ollama_resp: Vec<OllamaResponse> = resp.json().await?;
                Ok(ollama_resp.iter().map(|r| r.response.clone()).collect::<String>())
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
                repetition_penalty: 1.2,
                max_tokens: self.max_tokens,
                logprobs: None,
                top_logprobs: None,
            };
            let response = self.client.post(&self.endpoint).json(&payload).send().await?;
            if !response.status().is_success() {
                anyhow::bail!("vLLM request failed: {}", response.status());
            }
            let completion: ChatCompletionResponse = response.json().await?;
            let content = completion.choices.first().and_then(|choice| choice.message.content.clone()).unwrap_or_default();
            Ok(content)
        }
    }

    pub async fn send_chat_with_logprobs(
        &self,
        messages: Vec<ChatMessage>,
        enable_logprobs: bool,
    ) -> Result<String> {
        // Dynamic max_tokens based on message complexity (configurable clamp range)
        let prompt_len: usize = messages.iter().map(|m| m.content.len()).sum();
        let dynamic_max_tokens =
            (prompt_len * 2).clamp(self.dynamic_token_min, self.dynamic_token_max);

        let payload = ChatCompletionRequest {
            model: self.model.clone(),
            messages,
            temperature: self.temperature,
            top_p: self.top_p,
            repetition_penalty: 1.2,
            max_tokens: dynamic_max_tokens.max(self.max_tokens), // Use at least the configured minimum
            logprobs: if enable_logprobs { Some(true) } else { None },
            top_logprobs: if enable_logprobs { Some(0) } else { None },
        };

        let response = self
            .client
            .post(&self.endpoint)
            .json(&payload)
            .send()
            .await
            .with_context(|| format!("failed to call vLLM endpoint {}", self.endpoint))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            warn!(%status, %body, "vLLM returned error status");
            anyhow::bail!("vLLM request failed: {status}");
        }

        let completion: ChatCompletionResponse = response
            .json()
            .await
            .context("failed to parse vLLM chat completion response")?;

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
            repetition_penalty: 1.2,
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
        self.request_text(&cot_prompt).await.map(|(text, _)| text)
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
        self.request_text(&augmented).await.map(|(text, _)| text)
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
        const MAX_CHARS: usize = 512;
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

        let clamped = prompt[start_byte..].to_string();
        tracing::debug!(
            original_len = total_chars,
            clamped_len = clamped.chars().count(),
            "clamped prompt to MAX_CHARS"
        );
        clamped
    }

    /// Apply CoT repair for soft failures
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
    async fn generate_with_params(
        &self,
        prompt: &str,
        temp: f64,
        top_p: f64,
    ) -> Result<GenerationResult> {
        let start = Instant::now();
        
        // Prepare messages for generation
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "Implement the EXACT 3D labyrinth solver in Python using Dijkstra with priority queue. Use the provided 7x7x7 grid, handle echo chambers with state (consec_echoes 0-3, attune_timer 0-3, multiplier starting at 1, doubles on distract, halves on attune). Ignore any metaphorical language; focus strictly on implementing the algorithm as described. Output ONLY the complete code showing Path list, Total cost (expected 46), Steps. No text or philosophy.\n\"Include full grid definition and state logic as specified.\"".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            },
        ];

        // Call generation with custom params
        let result = self.send_chat_with_custom_params(messages, temp, top_p, true).await;
        
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
                    ucb1_score: 0.5, // Default for failed generation
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
            ucb1_score: 0.5, // Default for failed generation
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
        // Dynamic max_tokens based on message complexity
        let prompt_len: usize = messages.iter().map(|m| m.content.len()).sum();
        let dynamic_max_tokens =
            (prompt_len * 2).clamp(self.dynamic_token_min, self.dynamic_token_max);

        let payload = ChatCompletionRequest {
            model: self.model.clone(),
            messages,
            temperature: temp,
            top_p,
            repetition_penalty: 1.2,
            max_tokens: dynamic_max_tokens.max(self.max_tokens),
            logprobs: if enable_logprobs { Some(true) } else { None },
            top_logprobs: if enable_logprobs { Some(0) } else { None },
        };

        let response = self
            .client
            .post(&self.endpoint)
            .json(&payload)
            .send()
            .await
            .with_context(|| format!("failed to call vLLM endpoint {}", self.endpoint))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            warn!(%status, %body, "vLLM returned error status");
            anyhow::bail!("vLLM request failed: {status}");
        }

        let completion: ChatCompletionResponse = response
            .json()
            .await
            .context("failed to parse vLLM chat completion response")?;

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
