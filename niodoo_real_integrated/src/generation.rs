use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::time::timeout;
use tracing::{info, instrument, warn};

use crate::compass::CompassOutcome;
use crate::token_manager::TokenizerOutput;
use crate::util::rouge_l;

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
}

impl GenerationEngine {
    pub fn new(endpoint: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        let client = Client::builder().timeout(Duration::from_secs(5)).build()?;
        Ok(Self {
            client,
            endpoint: endpoint.into(),
            model: model.into(),
            temperature: 0.6,
            top_p: 0.7,
            max_tokens: 16,
            mock_mode: false,
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
                baseline_response: format!("[Mock baseline to: {}]", tokenizer_output.augmented_prompt.chars().take(50).collect::<String>()),
                hybrid_response: format!("[Mock hybrid response to: {}]", tokenizer_output.augmented_prompt.chars().take(50).collect::<String>()),
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
        let prompt = Self::clamp_prompt(prompt);
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are the baseline consciousness engine providing a direct reflection."
                    .to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: prompt,
            },
        ];
        match timeout(Duration::from_secs(5), self.send_chat(messages)).await {
            Ok(Ok(resp)) => Ok(resp),
            Ok(Err(error)) => {
                warn!(
                    ?error,
                    "baseline generation failed; returning fallback text"
                );
                Ok("Baseline response unavailable (timeout)".to_string())
            }
            Err(_) => {
                warn!("baseline generation timed out after 5s; returning fallback text");
                Ok("Baseline response unavailable (timeout)".to_string())
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
        let response = match timeout(Duration::from_secs(5), self.send_chat(messages)).await {
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

    async fn send_chat(&self, messages: Vec<ChatMessage>) -> Result<String> {
        if self.mock_mode {
            // Return mock response based on prompt
            let user_message = messages.iter()
                .find(|m| m.role == "user")
                .map(|m| m.content.as_str())
                .unwrap_or("mock prompt");
            return Ok(format!("[Mock response to: {}]", user_message.chars().take(100).collect::<String>()));
        }
        
        let payload = ChatCompletionRequest {
            model: self.model.clone(),
            messages,
            temperature: self.temperature,
            top_p: self.top_p,
            max_tokens: self.max_tokens,
        };

        // Ensure endpoint has the correct path
        let endpoint_url = if self.endpoint.contains("/v1/chat/completions") {
            self.endpoint.clone()
        } else {
            format!("{}/v1/chat/completions", self.endpoint.trim_end_matches('/'))
        };

        let response = self
            .client
            .post(&endpoint_url)
            .json(&payload)
            .send()
            .await
            .with_context(|| format!("failed to call vLLM endpoint {}", endpoint_url))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            warn!(%status, %body, endpoint = %endpoint_url, "vLLM returned error status");
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
            format!("{}/v1/chat/completions", self.endpoint.trim_end_matches('/'))
        };

        let response = self
            .client
            .post(&endpoint_url)
            .json(&payload)
            .timeout(Duration::from_secs(30))
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
    ) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .build()?;
        
        // Use model ID from vLLM if model path is provided, otherwise use as-is
        let model_id = if model.starts_with("/workspace/models/") {
            // Try to get the actual model ID from vLLM
            model.to_string() // Use the path as model ID since vLLM accepts it
        } else {
            model.to_string()
        };
        
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
        })
    }

    /// Apply runtime config
    pub fn apply_runtime_from_config(&mut self, _config: &crate::config::CliArgs) {
        // Stub implementation
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
    pub fn set_system_prompt(&mut self, _prompt: String) {
        // Stub implementation
    }

    /// Generate with params
    pub async fn generate_with_params(&self, prompt: &str, temperature: f64, top_p: f64) -> Result<String> {
        if self.mock_mode {
            return Ok(format!("[Mock response to: {}]", prompt.chars().take(100).collect::<String>()));
        }
        
        let mut temp_engine = Self {
            client: self.client.clone(),
            endpoint: self.endpoint.clone(),
            model: self.model.clone(),
            temperature,
            top_p,
            max_tokens: self.max_tokens,
            mock_mode: false,
        };
        temp_engine.request_text(prompt).await
    }

    /// Generate with consistency voting
    pub async fn generate_with_consistency(
        &self,
        _tokenizer: &crate::token_manager::TokenizerOutput,
        _compass: &crate::compass::CompassOutcome,
    ) -> Result<ConsistencyVotingResult> {
        // Stub implementation
        Ok(ConsistencyVotingResult {
            candidate_1: "Stub response 1".to_string(),
            candidate_2: "Stub response 2".to_string(),
            candidate_3: "Stub response 3".to_string(),
            rouge_scores: vec![0.5, 0.5, 0.5],
            latency_ms: 0.0,
            winner_index: 0,
        })
    }

    /// Generate with topology
    /// Generate with fallback to mock if primary fails
    pub async fn generate_with_fallback(&self, prompt: &str) -> Result<(String, String)> {
        match self.generate_with_params(prompt, self.temperature, self.top_p).await {
            Ok(response) => Ok((response, "primary".to_string())),
            Err(_) => {
                // Fallback to mock
                Ok((format!("[Mock response to: {}]", prompt), "mock".to_string()))
            }
        }
    }

    pub async fn generate_with_topology(
        &self,
        tokenizer: &crate::token_manager::TokenizerOutput,
        compass: &crate::compass::CompassOutcome,
        _topology: Option<&crate::tcs_analysis::TopologicalSignature>,
        _use_cache: bool,
    ) -> Result<GenerationResult> {
        // Fallback to regular generation
        self.generate(tokenizer, compass).await
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

#[derive(Debug, Serialize)]
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
