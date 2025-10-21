use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::time::timeout;
use tracing::{info, instrument, warn};

use crate::compass::CompassOutcome;
use crate::tokenizer::TokenizerOutput;
use crate::util::rouge_l;

#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub baseline_response: String,
    pub hybrid_response: String,
    pub echoes: Vec<LensEcho>,
    pub rouge_to_baseline: f64,
    pub latency_ms: f64,
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
        })
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
            latency_ms,
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
        let payload = ChatCompletionRequest {
            model: self.model.clone(),
            messages,
            temperature: self.temperature,
            top_p: self.top_p,
            max_tokens: self.max_tokens,
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

    pub async fn warmup(&self) -> Result<()> {
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

        let response = self
            .client
            .post(&self.endpoint)
            .json(&payload)
            .timeout(Duration::from_secs(30))
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
