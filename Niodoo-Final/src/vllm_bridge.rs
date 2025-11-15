//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! vLLM HTTP Bridge for Qwen2.5 inference via external vLLM service
//!
//! This module communicates with a running vLLM server (typically on port 8000)
//! via OpenAI-compatible REST API. Works with any vLLM deployment.
//!
//! Example setup on Beelink:
//! ```bash
//! cd ~/vllm-service
//! # Edit config/vllm.env to set MODEL_PATH to Qwen2.5
//! ./scripts/start-vllm.sh
//! # Server now runs on http://localhost:8000
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::Duration;
use tracing::{debug, info, warn};

/// vLLM HTTP REST client
pub struct VLLMBridge {
    base_url: String,
    api_key: Option<String>,
    client: reqwest::Client,
}

/// OpenAI-compatible completion request
#[derive(Debug, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
}

/// OpenAI-compatible completion response
#[derive(Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Choice {
    pub text: String,
    pub index: usize,
    pub finish_reason: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl VLLMBridge {
    /// Connect to a running vLLM server
    pub fn connect(base_url: impl Into<String>, api_key: Option<String>) -> Result<Self> {
        let base_url = base_url.into();

        info!("🌐 Connecting to vLLM server at: {}", base_url);

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(300)) // 5 min timeout for generation
            .build()?;

        Ok(Self {
            base_url,
            api_key,
            client,
        })
    }

    /// Generate text using vLLM
    pub async fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
    ) -> Result<String> {
        info!("📤 Sending request to vLLM (max_tokens: {})", max_tokens);

        let endpoint = format!("{}/v1/completions", self.base_url);

        let request = json!({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        });

        debug!("Request payload: {}", request);

        let mut req_builder = self
            .client
            .post(&endpoint)
            .json(&request)
            .header("Content-Type", "application/json");

        // Add API key if present
        if let Some(ref key) = self.api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", key));
        }

        let response = req_builder
            .send()
            .await
            .map_err(|e| anyhow!("Failed to send request to vLLM: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow!(
                "vLLM request failed with status {}: {}",
                status,
                error_text
            ));
        }

        let completion: CompletionResponse = response
            .json()
            .await
            .map_err(|e| anyhow!("Failed to parse vLLM response: {}", e))?;

        if completion.choices.is_empty() {
            return Err(anyhow!("vLLM returned no completions"));
        }

        let text = &completion.choices[0].text;
        info!(
            "✅ Generation complete: {} tokens (prompt: {}, completion: {})",
            completion.usage.total_tokens,
            completion.usage.prompt_tokens,
            completion.usage.completion_tokens
        );

        Ok(text.clone())
    }

    /// Health check endpoint
    pub async fn health(&self) -> Result<bool> {
        let endpoint = format!("{}/health", self.base_url);

        match self.client.get(&endpoint).send().await {
            Ok(response) => {
                debug!("Health check status: {}", response.status());
                Ok(response.status().is_success())
            }
            Err(e) => {
                warn!("Health check failed: {}", e);
                Ok(false)
            }
        }
    }

    /// Get available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        let endpoint = format!("{}/v1/models", self.base_url);

        let response = self.client.get(&endpoint).send().await?;

        #[derive(Deserialize)]
        struct ModelList {
            data: Vec<Model>,
        }

        #[derive(Deserialize)]
        struct Model {
            id: String,
        }

        let model_list: ModelList = response.json().await?;
        Ok(model_list.data.iter().map(|m| m.id.clone()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires vLLM running
    async fn test_vllm_connect() {
        let bridge = VLLMBridge::connect("http://localhost:8000", None);
        assert!(bridge.is_ok());
    }

    #[tokio::test]
    #[ignore] // Requires vLLM running
    async fn test_health_check() {
        let bridge = VLLMBridge::connect("http://localhost:8000", None).unwrap();
        let healthy = bridge.health().await.unwrap();
        assert!(healthy);
    }
}
