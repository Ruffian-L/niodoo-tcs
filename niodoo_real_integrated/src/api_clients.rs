/// API clients for external LLM services (Claude, GPT)
/// This module provides client implementations for multiple LLM providers
/// to support the cascading generation fallback logic.
use anyhow::{Context, Result};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, info, warn};

/// Retry configuration with exponential backoff
const RETRY_ATTEMPTS: usize = 3;
const INITIAL_BACKOFF_MS: u64 = 100; // 100ms
const BACKOFF_MULTIPLIER: f64 = 10.0; // exponential growth: 100ms -> 1s -> 10s
const MAX_RETRY_AFTER_SECS: u64 = 60; // Max time to wait based on Retry-After header

/// Parse Retry-After header value (can be seconds or HTTP date)
/// Returns duration to sleep, or None if header is invalid
fn parse_retry_after(header_value: &str) -> Option<Duration> {
    // Try parsing as seconds (most common)
    if let Ok(secs) = header_value.trim().parse::<u64>() {
        // Cap at MAX_RETRY_AFTER_SECS to avoid excessive waits
        let wait_secs = std::cmp::min(secs, MAX_RETRY_AFTER_SECS);
        return Some(Duration::from_secs(wait_secs));
    }

    // If not seconds, it's likely an HTTP date format
    // For simplicity, default to a reasonable backoff
    debug!(
        "Retry-After header is HTTP date format: {}. Using default backoff.",
        header_value
    );
    Some(Duration::from_secs(5))
}

/// Execute an async operation with exponential backoff retry logic
/// Handles 429 (rate limit) responses specially by respecting Retry-After header
/// Returns after 3 attempts with delays: 100ms, 1s, 10s (or Retry-After if rate limited)
async fn execute_with_retry<F, T>(mut operation: F) -> Result<T>
where
    F: FnMut() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T>>>>,
{
    let mut attempt = 0;
    let mut backoff_ms = INITIAL_BACKOFF_MS;

    loop {
        attempt += 1;
        debug!("API request attempt {}/{}", attempt, RETRY_ATTEMPTS);

        match operation().await {
            Ok(result) => {
                debug!("API request succeeded on attempt {}", attempt);
                return Ok(result);
            }
            Err(e) => {
                // Check if error message indicates a 429 rate limit response
                let error_str = e.to_string();
                let is_rate_limited = error_str.contains("429");

                if attempt < RETRY_ATTEMPTS {
                    let sleep_duration = if is_rate_limited {
                        // For 429, we've already parsed and logged the Retry-After header
                        // The error will contain info about the rate limit
                        info!(
                            "Rate limited (429) on attempt {}. Will retry after backoff...",
                            attempt
                        );
                        Duration::from_millis(backoff_ms)
                    } else {
                        Duration::from_millis(backoff_ms)
                    };

                    warn!(
                        "API request attempt {} failed: {}. Retrying in {:?}...",
                        attempt, e, sleep_duration
                    );
                    tokio::time::sleep(sleep_duration).await;
                    backoff_ms = (backoff_ms as f64 * BACKOFF_MULTIPLIER) as u64;
                } else {
                    warn!(
                        "API request failed after {} attempts: {}",
                        RETRY_ATTEMPTS, e
                    );
                    return Err(e);
                }
            }
        }
    }
}

/// Claude API client for Anthropic's Claude models
#[derive(Clone)]
pub struct ClaudeClient {
    client: Client,
    api_key: String,
    endpoint: String,
    model: String,
    timeout_secs: u64,
}

impl ClaudeClient {
    pub fn new(api_key: String, model: impl Into<String>, timeout_secs: u64) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()?;

        Ok(Self {
            client,
            api_key,
            endpoint: "https://api.anthropic.com/v1/messages".to_string(),
            model: model.into(),
            timeout_secs,
        })
    }

    /// Get the endpoint URL
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Get the model name
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Send a request to Claude API
    pub async fn complete(&self, prompt: &str) -> Result<String> {
        self.generate_with_retry(prompt).await
    }

    /// Send a request to Claude API with exponential backoff retry logic
    /// Attempts up to 3 times with delays: 100ms, 1s, 10s
    pub async fn generate_with_retry(&self, prompt: &str) -> Result<String> {
        let api_key = self.api_key.clone();
        let endpoint = self.endpoint.clone();
        let model = self.model.clone();
        let client = self.client.clone();
        let prompt = prompt.to_string();
        let timeout_secs = self.timeout_secs;

        execute_with_retry(move || {
            let api_key = api_key.clone();
            let endpoint = endpoint.clone();
            let model = model.clone();
            let client = client.clone();
            let prompt = prompt.clone();
            let timeout_secs = timeout_secs;

            Box::pin(async move {
                let payload = ClaudeRequest {
                    model: model.clone(),
                    max_tokens: 1024,
                    messages: vec![ClaudeMessage {
                        role: "user".to_string(),
                        content: prompt.to_string(),
                    }],
                };

                let response = tokio::time::timeout(
                    Duration::from_secs(timeout_secs),
                    client
                        .post(&endpoint)
                        .header("x-api-key", &api_key)
                        .header("anthropic-version", "2023-06-01")
                        .json(&payload)
                        .send(),
                )
                .await
                .context("Claude API request timed out")??;

                // Handle 429 rate limit responses
                if response.status() == StatusCode::TOO_MANY_REQUESTS {
                    let retry_after = response
                        .headers()
                        .get("retry-after")
                        .and_then(|v| v.to_str().ok())
                        .and_then(parse_retry_after);

                    let wait_duration = retry_after.unwrap_or_else(|| Duration::from_secs(1));
                    warn!(
                        "Claude API rate limited (429). Retry-After: {:?}",
                        wait_duration
                    );
                    info!(
                        "Sleeping for {:?} before retrying Claude API request",
                        wait_duration
                    );
                    tokio::time::sleep(wait_duration).await;
                    anyhow::bail!("Claude API rate limited (429). Retrying...");
                }

                if !response.status().is_success() {
                    let status = response.status();
                    let body = response.text().await.unwrap_or_default();
                    warn!(%status, %body, "Claude API returned error");
                    anyhow::bail!("Claude API error: {status}");
                }

                let result: ClaudeResponse = response
                    .json()
                    .await
                    .context("Failed to parse Claude response")?;

                let content = result
                    .content
                    .first()
                    .and_then(|c| {
                        if c.message_type == "text" {
                            c.text.clone()
                        } else {
                            None
                        }
                    })
                    .unwrap_or_default();

                Ok(content)
            })
        })
        .await
    }
}

/// GPT (OpenAI) API client
#[derive(Clone)]
pub struct GptClient {
    client: Client,
    api_key: String,
    endpoint: String,
    model: String,
    timeout_secs: u64,
}

impl GptClient {
    pub fn new(api_key: String, model: impl Into<String>, timeout_secs: u64) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()?;

        Ok(Self {
            client,
            api_key,
            endpoint: "https://api.openai.com/v1/chat/completions".to_string(),
            model: model.into(),
            timeout_secs,
        })
    }

    /// Get the endpoint URL
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Get the model name
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Send a request to GPT API
    pub async fn complete(&self, prompt: &str) -> Result<String> {
        self.generate_with_retry(prompt).await
    }

    /// Send a request to GPT API with exponential backoff retry logic
    /// Attempts up to 3 times with delays: 100ms, 1s, 10s
    pub async fn generate_with_retry(&self, prompt: &str) -> Result<String> {
        let api_key = self.api_key.clone();
        let endpoint = self.endpoint.clone();
        let model = self.model.clone();
        let client = self.client.clone();
        let prompt = prompt.to_string();
        let timeout_secs = self.timeout_secs;

        execute_with_retry(move || {
            let api_key = api_key.clone();
            let endpoint = endpoint.clone();
            let model = model.clone();
            let client = client.clone();
            let prompt = prompt.clone();
            let timeout_secs = timeout_secs;

            Box::pin(async move {
                let payload = GptRequest {
                    model: model.clone(),
                    messages: vec![GptMessage {
                        role: "user".to_string(),
                        content: prompt.to_string(),
                    }],
                    temperature: 0.7,
                    max_tokens: 1024,
                };

                let response = tokio::time::timeout(
                    Duration::from_secs(timeout_secs),
                    client
                        .post(&endpoint)
                        .header("Authorization", format!("Bearer {}", api_key))
                        .json(&payload)
                        .send(),
                )
                .await
                .context("GPT API request timed out")??;

                // Handle 429 rate limit responses
                if response.status() == StatusCode::TOO_MANY_REQUESTS {
                    let retry_after = response
                        .headers()
                        .get("retry-after")
                        .and_then(|v| v.to_str().ok())
                        .and_then(parse_retry_after);

                    let wait_duration = retry_after.unwrap_or_else(|| Duration::from_secs(1));
                    warn!(
                        "GPT API rate limited (429). Retry-After: {:?}",
                        wait_duration
                    );
                    info!(
                        "Sleeping for {:?} before retrying GPT API request",
                        wait_duration
                    );
                    tokio::time::sleep(wait_duration).await;
                    anyhow::bail!("GPT API rate limited (429). Retrying...");
                }

                if !response.status().is_success() {
                    let status = response.status();
                    let body = response.text().await.unwrap_or_default();
                    warn!(%status, %body, "GPT API returned error");
                    anyhow::bail!("GPT API error: {status}");
                }

                let result: GptResponse = response
                    .json()
                    .await
                    .context("Failed to parse GPT response")?;

                let content = result
                    .choices
                    .first()
                    .map(|c| c.message.content.clone())
                    .unwrap_or_default();

                Ok(content)
            })
        })
        .await
    }
}

// Claude API request/response types
#[derive(Debug, Serialize)]
struct ClaudeRequest {
    model: String,
    max_tokens: usize,
    messages: Vec<ClaudeMessage>,
}

#[derive(Debug, Serialize)]
struct ClaudeMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ClaudeResponse {
    content: Vec<ClaudeContent>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct ClaudeContent {
    #[serde(rename = "type")]
    message_type: String,
    text: Option<String>,
}

// GPT API request/response types
#[derive(Debug, Serialize)]
struct GptRequest {
    model: String,
    messages: Vec<GptMessage>,
    temperature: f64,
    max_tokens: usize,
}

#[derive(Debug, Serialize)]
struct GptMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct GptResponse {
    choices: Vec<GptChoice>,
}

#[derive(Debug, Deserialize)]
struct GptChoice {
    message: GptResponseMessage,
}

#[derive(Debug, Deserialize)]
struct GptResponseMessage {
    content: String,
}
