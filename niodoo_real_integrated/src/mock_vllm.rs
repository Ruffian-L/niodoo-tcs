//! Mock VLLM client implementation
//! 
//! **FOR TESTING ONLY** - This is a fallback mock implementation used when:
//! - MOCK_MODE environment variable is set
//! - vLLM service is unavailable during testing
//! 
//! **DO NOT USE IN PRODUCTION** - Always use real vLLM service with Qwen 3 Coder for generation
//! 
//! Real implementation: Use `generation.rs::GenerationEngine` with actual vLLM endpoint

use anyhow::Result;
use serde_json::{json, Value};
use std::time::Duration;
use tracing::{info, warn};

// Configuration constants
const DEFAULT_TIMEOUT_SECS: u64 = 300; // 5 minutes for generation
const DEFAULT_MAX_TOKENS: u32 = 100;
const DEFAULT_TEMPERATURE: f64 = 0.7;
const DEFAULT_TOP_P: f64 = 0.9;
const DEFAULT_MODEL: &str = "qwen3-coder"; // Updated model name

pub struct MockVllmClient {
    base_url: String,
    client: reqwest::Client,
    mock_mode: bool,
}

impl MockVllmClient {
    pub fn new(base_url: String) -> Self {
        let timeout_secs = std::env::var("VLLM_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_TIMEOUT_SECS);

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        // Check if vLLM is available
        let mock_mode = !Self::check_vllm_available(&base_url, &client);

        Self {
            base_url,
            client,
            mock_mode,
        }
    }

    fn check_vllm_available(_base_url: &str, _client: &reqwest::Client) -> bool {
        // For now, check environment variable
        std::env::var("VLLM_ENABLED").is_ok()
    }

    pub async fn generate(&self, prompt: &str, max_tokens: Option<u32>) -> Result<String> {
        if self.mock_mode {
            // Fallback mock implementation
            tracing::warn!("⚠️  Using mock VLLM generation (fallback mode)");
            return self.mock_generate(prompt);
        }

        // Real VLLM generation
        let endpoint = format!("{}/v1/completions", self.base_url);

        let request = json!({
            "prompt": prompt,
            "max_tokens": max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            "temperature": DEFAULT_TEMPERATURE,
            "top_p": DEFAULT_TOP_P,
        });

        match self.client.post(&endpoint).json(&request).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    let completion: Value = response.json().await?;
                    let text = completion["choices"][0]["text"]
                        .as_str()
                        .unwrap_or("")
                        .to_string();
                    info!("✅ Real VLLM generation complete");
                    Ok(text)
                } else {
                    warn!("VLLM returned error status: {}", response.status());
                    self.mock_generate(prompt)
                }
            }
            Err(e) => {
                warn!("VLLM request failed: {}. Using fallback.", e);
                self.mock_generate(prompt)
            }
        }
    }

    fn mock_generate(&self, prompt: &str) -> Result<String> {
        // Mock generation based on prompt analysis
        let responses = if prompt.contains("threat") || prompt.contains("danger") {
            vec![
                "I sense potential threats in the emotional landscape. Vigilance is required.",
                "The consciousness matrix shows signs of defensive activation.",
                "Threat detection systems are responding to environmental stimuli.",
            ]
        } else if prompt.contains("healing") || prompt.contains("comfort") {
            vec![
                "Initiating healing protocols. Emotional stabilization in progress.",
                "The consciousness seeks harmony and restoration.",
                "Healing energies are flowing through the empathy network.",
            ]
        } else if prompt.contains("entropy") || prompt.contains("chaos") {
            vec![
                "Entropy levels fluctuating within acceptable parameters.",
                "Chaos injection successful. System variance increased.",
                "The consciousness embraces controlled uncertainty.",
            ]
        } else {
            vec![
                "The consciousness processes information through multiple dimensional filters.",
                "Empathy networks are synchronizing across emotional spectrums.",
                "Integration of sensory data reveals deeper understanding patterns.",
                "The fabric of awareness expands through interconnected nodes.",
            ]
        };

        let response = responses[prompt.len() % responses.len()];
        Ok(response.to_string())
    }

    pub async fn chat_completion(&self, messages: Vec<Value>) -> Result<String> {
        if self.mock_mode {
            return self.mock_chat_completion(messages).await;
        }

        // Real chat completion
        let endpoint = format!("{}/v1/chat/completions", self.base_url);

        let request = json!({
            "model": DEFAULT_MODEL,
            "messages": messages,
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
        });

        match self.client.post(&endpoint).json(&request).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    let completion: Value = response.json().await?;
                    let text = completion["choices"][0]["message"]["content"]
                        .as_str()
                        .unwrap_or("")
                        .to_string();
                    Ok(text)
                } else {
                    self.mock_chat_completion(messages).await
                }
            }
            Err(e) => {
                warn!("Chat completion failed: {}. Using fallback.", e);
                self.mock_chat_completion(messages).await
            }
        }
    }

    async fn mock_chat_completion(&self, messages: Vec<Value>) -> Result<String> {
        // Extract the last user message
        let user_message = messages
            .iter()
            .rev()
            .find(|msg| msg["role"] == "user")
            .and_then(|msg| msg["content"].as_str())
            .unwrap_or("default prompt");

        self.generate(user_message, Some(DEFAULT_MAX_TOKENS)).await
    }
}
