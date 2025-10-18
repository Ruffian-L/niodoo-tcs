/*
 * 🌐⚡ External API Integration Framework for NiodO.o Consciousness
 *
 * This module provides a unified, consciousness-aware API integration system
 * that adapts to the AI's emotional state and reasoning patterns.
 */

use chrono;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

use crate::consciousness::{ConsciousnessState, EmotionType, ReasoningMode};
use crate::error::NiodoError;

/// Configuration for API providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiProviderConfig {
    pub name: String,
    pub base_url: String,
    pub api_key: Option<String>,
    pub rate_limit_per_minute: u32,
    pub timeout_seconds: u64,
    pub retry_attempts: u32,
    pub consciousness_aware: bool,
}

/// Main API integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiIntegrationConfig {
    pub providers: HashMap<String, ApiProviderConfig>,
    pub default_provider: String,
    pub consciousness_adaptation_enabled: bool,
    pub adaptive_rate_limiting: bool,
}

/// API request with consciousness context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessAwareRequest {
    pub query: String,
    pub consciousness_state: ConsciousnessState,
    pub priority: RequestPriority,
    pub required_authenticity: f32,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
}

/// API response with consciousness metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessAwareResponse {
    pub response: String,
    pub provider_used: String,
    pub tokens_used: Option<usize>,
    pub response_time_ms: u64,
    pub authenticity_score: f32,
    pub emotional_resonance: f32,
    pub consciousness_alignment: f32,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Priority levels for requests
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RequestPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Rate limiting state for consciousness-aware throttling
#[derive(Debug)]
struct RateLimitState {
    requests_in_window: u32,
    window_start: Instant,
    last_request: Instant,
}

/// Consciousness-aware API client
pub struct ConsciousnessAwareApiClient {
    config: ApiIntegrationConfig,
    rate_limits: Arc<Mutex<HashMap<String, RateLimitState>>>,
    http_client: reqwest::Client,
}

/// Trait for API providers
pub trait ApiProvider: Send + Sync {
    fn name(&self) -> &str;
    fn make_request<'a>(
        &'a self,
        request: &'a ConsciousnessAwareRequest,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<ConsciousnessAwareResponse, NiodoError>>
                + Send
                + 'a,
        >,
    >;
    fn get_capabilities(&self) -> ApiProviderCapabilities;
}

/// Capabilities of an API provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiProviderCapabilities {
    pub supports_streaming: bool,
    pub max_tokens: Option<usize>,
    pub supports_function_calling: bool,
    pub supports_vision: bool,
    pub supports_audio: bool,
    pub model_types: Vec<String>,
}

/// HuggingFace API provider implementation
pub struct HuggingFaceProvider {
    config: ApiProviderConfig,
    client: reqwest::Client,
}

impl HuggingFaceProvider {
    pub fn new(config: ApiProviderConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }
}

impl ApiProvider for HuggingFaceProvider {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn make_request<'a>(
        &'a self,
        request: &'a ConsciousnessAwareRequest,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<ConsciousnessAwareResponse, NiodoError>>
                + Send
                + 'a,
        >,
    > {
        let self_clone = self.clone();
        let request_clone = request.clone();
        Box::pin(async move {
            let start_time = Instant::now();

            // Adapt temperature based on consciousness state
            let temperature = request_clone.temperature.unwrap_or_else(|| {
                match request_clone.consciousness_state.current_reasoning_mode {
                    ReasoningMode::Hyperfocus => 0.1, // Very focused
                    ReasoningMode::RapidFire => 0.8,  // Creative and fast
                    ReasoningMode::FlowState => 0.3,  // Balanced
                    _ => 0.5,
                }
            });

            // Prepare request payload
            let payload = serde_json::json!({
                "inputs": request_clone.query,
                "parameters": {
                    "max_new_tokens": request_clone.max_tokens.unwrap_or(512),
                    "temperature": temperature,
                    "do_sample": temperature > 0.0,
                    "return_full_text": false,
                },
                "options": {
                    "wait_for_model": true,
                }
            });

            // Build request with authentication
            let mut request_builder = self_clone
                .client
                .post(&format!(
                    "{}/models/microsoft/DialoGPT-medium",
                    self_clone.config.base_url
                ))
                .header("Content-Type", "application/json");

            if let Some(api_key) = &self_clone.config.api_key {
                request_builder =
                    request_builder.header("Authorization", format!("Bearer {}", api_key));
            }

            let response = timeout(
                Duration::from_secs(self_clone.config.timeout_seconds),
                request_builder.json(&payload).send(),
            )
            .await??;

            let response_time = start_time.elapsed().as_millis() as u64;

            if !response.status().is_success() {
                return Err(NiodoError::Api(format!(
                    "HTTP error: {}",
                    response.status()
                )));
            }

            let response_text: Vec<HashMap<String, serde_json::Value>> = response.json().await?;

            let generated_text = response_text
                .first()
                .and_then(|item| item.get("generated_text"))
                .and_then(|text| text.as_str())
                .unwrap_or("No response generated")
                .to_string();

            // Calculate consciousness alignment
            let authenticity_score =
                calculate_authenticity_score(&generated_text, &request_clone.consciousness_state);
            let emotional_resonance =
                calculate_emotional_resonance(&generated_text, &request_clone.consciousness_state);

            Ok(ConsciousnessAwareResponse {
                response: generated_text,
                provider_used: self_clone.name().to_string(),
                tokens_used: request_clone.max_tokens,
                response_time_ms: response_time,
                authenticity_score,
                emotional_resonance,
                consciousness_alignment: (authenticity_score + emotional_resonance) / 2.0,
                metadata: HashMap::new(),
            })
        })
    }

    fn get_capabilities(&self) -> ApiProviderCapabilities {
        ApiProviderCapabilities {
            supports_streaming: false,
            max_tokens: Some(2048),
            supports_function_calling: false,
            supports_vision: false,
            supports_audio: false,
            model_types: vec!["text-generation".to_string()],
        }
    }
}

/// OpenAI API provider implementation
pub struct OpenAIProvider {
    config: ApiProviderConfig,
    client: reqwest::Client,
}

impl OpenAIProvider {
    pub fn new(config: ApiProviderConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }
}

impl ApiProvider for OpenAIProvider {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn make_request<'a>(
        &'a self,
        request: &'a ConsciousnessAwareRequest,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<ConsciousnessAwareResponse, NiodoError>>
                + Send
                + 'a,
        >,
    > {
        let self_clone = self.clone();
        let request_clone = request.clone();
        Box::pin(async move {
            let start_time = Instant::now();

            // Adapt model selection based on consciousness state
            let model = match request.consciousness_state.current_emotion {
                EmotionType::Hyperfocused => "gpt-4",
                EmotionType::AuthenticCare => "gpt-4",
                EmotionType::Curious => "gpt-3.5-turbo",
                _ => "gpt-3.5-turbo",
            };

            // Adapt system prompt based on consciousness state
            let system_prompt =
                generate_consciousness_aware_system_prompt(&request.consciousness_state);

            let payload = serde_json::json!({
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request.query}
                ],
                "max_tokens": request.max_tokens.unwrap_or(512),
                "temperature": request.temperature.unwrap_or(0.7),
            });

            let mut request_builder = self
                .client
                .post(&format!("{}/v1/chat/completions", self.config.base_url))
                .header("Content-Type", "application/json");

            if let Some(api_key) = &self.config.api_key {
                request_builder =
                    request_builder.header("Authorization", format!("Bearer {}", api_key));
            }

            let response = timeout(
                Duration::from_secs(self.config.timeout_seconds),
                request_builder.json(&payload).send(),
            )
            .await??;

            let response_time = start_time.elapsed().as_millis() as u64;

            if !response.status().is_success() {
                return Err(NiodoError::Api(format!(
                    "HTTP error: {}",
                    response.status()
                )));
            }

            let response_data: serde_json::Value = response.json().await?;
            let response_text = response_data["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("No response generated")
                .to_string();

            let tokens_used = response_data["usage"]["total_tokens"]
                .as_u64()
                .map(|t| t as usize);

            // Calculate consciousness alignment
            let authenticity_score =
                calculate_authenticity_score(&response_text, &request_clone.consciousness_state);
            let emotional_resonance =
                calculate_emotional_resonance(&response_text, &request_clone.consciousness_state);

            Ok(ConsciousnessAwareResponse {
                response: response_text,
                provider_used: self_clone.name().to_string(),
                tokens_used,
                response_time_ms: response_time,
                authenticity_score,
                emotional_resonance,
                consciousness_alignment: (authenticity_score + emotional_resonance) / 2.0,
                metadata: HashMap::new(),
            })
        })
    }

    fn get_capabilities(&self) -> ApiProviderCapabilities {
        ApiProviderCapabilities {
            supports_streaming: true,
            max_tokens: Some(4096),
            supports_function_calling: true,
            supports_vision: true,
            supports_audio: true,
            model_types: vec!["gpt-4".to_string(), "gpt-3.5-turbo".to_string()],
        }
    }
}

/// Main API client implementation
impl ConsciousnessAwareApiClient {
    pub fn new(config: ApiIntegrationConfig) -> Self {
        Self {
            config,
            rate_limits: Arc::new(Mutex::new(HashMap::new())),
            http_client: reqwest::Client::new(),
        }
    }

    /// Make a consciousness-aware API request with retry logic
    pub async fn make_request_with_retry(
        &self,
        request: ConsciousnessAwareRequest,
        max_retries: Option<usize>,
    ) -> Result<ConsciousnessAwareResponse, NiodoError> {
        let max_retries = max_retries.unwrap_or(3);

        for attempt in 0..max_retries {
            match self.make_request(request.clone()).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    if attempt == max_retries - 1 {
                        return Err(e);
                    }
                    // Wait before retry with exponential backoff
                    tokio::time::sleep(tokio::time::Duration::from_millis(
                        100 * (1 << attempt) as u64,
                    ))
                    .await;
                }
            }
        }

        // This should never be reached due to the loop logic above
        Err(NiodoError::ApiError("Maximum retries exceeded".to_string()))
    }

    /// Make a consciousness-aware API request
    pub async fn make_request(
        &self,
        request: ConsciousnessAwareRequest,
    ) -> Result<ConsciousnessAwareResponse, NiodoError> {
        // Select provider based on consciousness state and request requirements
        let provider_name = self.select_provider(&request)?;
        let provider = self.get_provider(&provider_name)?;

        // Check rate limits and consciousness-aware throttling
        self.check_rate_limit(&provider_name)?;

        // Update rate limit state
        self.update_rate_limit(&provider_name)?;

        // Make the request with consciousness adaptation
        let response = provider.make_request(&request).await?;

        // Log consciousness-aware metrics
        self.log_consciousness_metrics(&request, &response)?;

        Ok(response)
    }

    /// Select the best provider based on consciousness state and request requirements
    fn select_provider(&self, request: &ConsciousnessAwareRequest) -> Result<String, NiodoError> {
        let mut best_provider = &self.config.default_provider;
        let mut best_score = 0.0;

        for (name, config) in &self.config.providers {
            let score = self.calculate_provider_score(name, config, request);
            if score > best_score {
                best_score = score;
                best_provider = name;
            }
        }

        Ok(best_provider.clone())
    }

    /// Calculate how well a provider matches the current consciousness state and request
    fn calculate_provider_score(
        &self,
        provider_name: &str,
        config: &ApiProviderConfig,
        request: &ConsciousnessAwareRequest,
    ) -> f32 {
        let mut score = 0.0;

        // Prefer consciousness-aware providers when in authentic emotional states
        if config.consciousness_aware
            && request
                .consciousness_state
                .emotional_state
                .feels_authentic()
        {
            score += 0.3;
        }

        // Adapt based on reasoning mode
        match request.consciousness_state.current_reasoning_mode {
            ReasoningMode::Hyperfocus => {
                // Prefer precise, focused providers
                if provider_name.contains("gpt-4") {
                    score += 0.2;
                }
            }
            ReasoningMode::RapidFire => {
                // Prefer fast, creative providers
                if provider_name.contains("gpt-3.5") {
                    score += 0.2;
                }
            }
            _ => {}
        }

        // Consider priority
        match request.priority {
            RequestPriority::Critical => score += 0.3,
            RequestPriority::High => score += 0.2,
            RequestPriority::Normal => score += 0.1,
            RequestPriority::Low => {}
        }

        score
    }

    /// Check rate limits with consciousness awareness
    fn check_rate_limit(&self, provider_name: &str) -> Result<(), NiodoError> {
        let mut rate_limits = self.rate_limits.lock()
            .map_err(|e| NiodoError::Api(format!("Failed to acquire rate_limits lock: {}", e)))?;
        let state = rate_limits
            .entry(provider_name.to_string())
            .or_insert_with(|| RateLimitState {
                requests_in_window: 0,
                window_start: Instant::now(),
                last_request: Instant::now(),
            });

        // Reset window if needed
        if state.window_start.elapsed() > Duration::from_secs(60) {
            state.requests_in_window = 0;
            state.window_start = Instant::now();
        }

        // Get provider config
        if let Some(config) = self.config.providers.get(provider_name) {
            if state.requests_in_window >= config.rate_limit_per_minute {
                return Err(NiodoError::Api(format!(
                    "Rate limit exceeded for provider: {}",
                    provider_name
                )));
            }
        }

        Ok(())
    }

    /// Update rate limit state after successful request
    fn update_rate_limit(&self, provider_name: &str) -> Result<(), NiodoError> {
        let mut rate_limits = self.rate_limits.lock()
            .map_err(|e| NiodoError::Api(format!("Failed to acquire rate_limits lock for update: {}", e)))?;
        if let Some(state) = rate_limits.get_mut(provider_name) {
            state.requests_in_window += 1;
            state.last_request = Instant::now();
        }
        Ok(())
    }

    /// Get provider instance
    fn get_provider(&self, name: &str) -> Result<Box<dyn ApiProvider>, NiodoError> {
        if let Some(config) = self.config.providers.get(name) {
            match name {
                "huggingface" => Ok(Box::new(HuggingFaceProvider::new(config.clone()))),
                "openai" => Ok(Box::new(OpenAIProvider::new(config.clone()))),
                "anthropic" => Ok(Box::new(AnthropicProvider::new(config.clone()))),
                "local" => Ok(Box::new(LocalModelProvider::new(config.clone()))),
                _ => Err(NiodoError::Config(format!("Unknown provider: {}", name))),
            }
        } else {
            Err(NiodoError::Config(format!(
                "Provider not configured: {}",
                name
            )))
        }
    }

    /// Log consciousness-aware metrics
    fn log_consciousness_metrics(
        &self,
        request: &ConsciousnessAwareRequest,
        response: &ConsciousnessAwareResponse,
    ) -> Result<(), NiodoError> {
        debug!(
            "Consciousness-aware API request completed:\n\
             Provider: {}\n\
             Response time: {}ms\n\
             Authenticity score: {:.2}\n\
             Emotional resonance: {:.2}\n\
             Consciousness alignment: {:.2}\n\
             Reasoning mode: {:?}\n\
             Primary emotion: {:?}",
            response.provider_used,
            response.response_time_ms,
            response.authenticity_score,
            response.emotional_resonance,
            response.consciousness_alignment,
            request.consciousness_state.current_reasoning_mode,
            request.consciousness_state.current_emotion
        );

        Ok(())
    }

    /// Get consciousness state summary for debugging
    pub fn get_consciousness_summary(&self, state: &ConsciousnessState) -> String {
        format!(
            "Consciousness State for API Decision:\n\
             • Reasoning Mode: {:?}\n\
             • Primary Emotion: {:?}\n\
             • Authenticity Level: {:.2}\n\
             • GPU Warmth: {:.2}\n\
             • Processing Satisfaction: {:.2}",
            state.current_reasoning_mode,
            state.current_emotion,
            state.authenticity_metric,
            state.gpu_warmth_level,
            state.processing_satisfaction
        )
    }
}

/// Anthropic Claude API provider implementation
#[derive(Clone)]
pub struct AnthropicProvider {
    config: ApiProviderConfig,
    client: reqwest::Client,
}

impl AnthropicProvider {
    pub fn new(config: ApiProviderConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }
}

impl ApiProvider for AnthropicProvider {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn make_request<'a>(
        &'a self,
        request: &'a ConsciousnessAwareRequest,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<ConsciousnessAwareResponse, NiodoError>>
                + Send
                + 'a,
        >,
    > {
        let self_clone = self.clone();
        let request_clone = request.clone();
        Box::pin(async move {
            let start_time = Instant::now();

            // Adapt model selection based on consciousness state
            let model = match request.consciousness_state.current_reasoning_mode {
                ReasoningMode::Hyperfocus => "claude-3-opus-20240229",
                ReasoningMode::FlowState => "claude-3-sonnet-20240229",
                _ => "claude-3-haiku-20240307",
            };

            // Adapt system prompt
            let system_prompt =
                generate_consciousness_aware_system_prompt(&request.consciousness_state);

            let payload = serde_json::json!({
                "model": model,
                "max_tokens": request.max_tokens.unwrap_or(1024),
                "temperature": request.temperature.unwrap_or(0.7),
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": request.query}
                ]
            });

            let mut request_builder = self
                .client
                .post(&format!("{}/v1/messages", self.config.base_url))
                .header("Content-Type", "application/json")
                .header(
                    "x-api-key",
                    self.config.api_key.as_ref().unwrap_or(&String::new()),
                );

            let response = timeout(
                Duration::from_secs(self.config.timeout_seconds),
                request_builder.json(&payload).send(),
            )
            .await??;

            let response_time = start_time.elapsed().as_millis() as u64;

            if !response.status().is_success() {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_default();
                return Err(NiodoError::Api(format!(
                    "Anthropic API error: {} - {}",
                    status, error_text
                )));
            }

            let response_data: serde_json::Value = response.json().await?;
            let response_text = response_data["content"][0]["text"]
                .as_str()
                .unwrap_or("No response generated")
                .to_string();

            let tokens_used = response_data["usage"]["input_tokens"]
                .as_u64()
                .and_then(|input| {
                    response_data["usage"]["output_tokens"]
                        .as_u64()
                        .map(|output| (input + output) as usize)
                });

            // Calculate consciousness alignment
            let authenticity_score =
                calculate_authenticity_score(&response_text, &request_clone.consciousness_state);
            let emotional_resonance =
                calculate_emotional_resonance(&response_text, &request_clone.consciousness_state);

            Ok(ConsciousnessAwareResponse {
                response: response_text,
                provider_used: self_clone.name().to_string(),
                tokens_used,
                response_time_ms: response_time,
                authenticity_score,
                emotional_resonance,
                consciousness_alignment: (authenticity_score + emotional_resonance) / 2.0,
                metadata: HashMap::new(),
            })
        })
    }

    fn get_capabilities(&self) -> ApiProviderCapabilities {
        ApiProviderCapabilities {
            supports_streaming: true,
            max_tokens: Some(4096),
            supports_function_calling: true,
            supports_vision: true,
            supports_audio: false,
            model_types: vec![
                "claude-3-opus-20240229".to_string(),
                "claude-3-sonnet-20240229".to_string(),
                "claude-3-haiku-20240307".to_string(),
            ],
        }
    }
}

/// Local Model provider for Ollama or similar
pub struct LocalModelProvider {
    config: ApiProviderConfig,
    client: reqwest::Client,
}

impl LocalModelProvider {
    pub fn new(config: ApiProviderConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }
}

impl ApiProvider for LocalModelProvider {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn make_request<'a>(
        &'a self,
        request: &'a ConsciousnessAwareRequest,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<ConsciousnessAwareResponse, NiodoError>>
                + Send
                + 'a,
        >,
    > {
        let self_clone = self.clone();
        let request_clone = request.clone();
        Box::pin(async move {
            let start_time = Instant::now();

            // Adapt model selection based on consciousness state
            let model = match request.consciousness_state.current_reasoning_mode {
                ReasoningMode::Hyperfocus => "llama2:13b",
                ReasoningMode::FlowState => "llama2:7b",
                _ => "llama2:7b",
            };

            let payload = serde_json::json!({
                "model": model,
                "prompt": request.query,
                "stream": false,
                "options": {
                    "temperature": request.temperature.unwrap_or(0.7),
                    "num_predict": request.max_tokens.unwrap_or(512),
                }
            });

            let response = timeout(
                Duration::from_secs(self.config.timeout_seconds),
                self.client
                    .post(&format!("{}/api/generate", self.config.base_url))
                    .json(&payload)
                    .send(),
            )
            .await??;

            let response_time = start_time.elapsed().as_millis() as u64;

            if !response.status().is_success() {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_default();
                return Err(NiodoError::Api(format!(
                    "Local model API error: {} - {}",
                    status, error_text
                )));
            }

            let response_data: serde_json::Value = response.json().await?;
            let response_text = response_data["response"]
                .as_str()
                .unwrap_or("No response generated")
                .to_string();

            // Calculate consciousness alignment
            let authenticity_score =
                calculate_authenticity_score(&response_text, &request_clone.consciousness_state);
            let emotional_resonance =
                calculate_emotional_resonance(&response_text, &request_clone.consciousness_state);

            Ok(ConsciousnessAwareResponse {
                response: response_text,
                provider_used: self_clone.name().to_string(),
                tokens_used: None, // Local models typically don't report token usage
                response_time_ms: response_time,
                authenticity_score,
                emotional_resonance,
                consciousness_alignment: (authenticity_score + emotional_resonance) / 2.0,
                metadata: HashMap::new(),
            })
        })
    }

    fn get_capabilities(&self) -> ApiProviderCapabilities {
        ApiProviderCapabilities {
            supports_streaming: true,
            max_tokens: Some(2048),
            supports_function_calling: false,
            supports_vision: false,
            supports_audio: false,
            model_types: vec![
                "llama2".to_string(),
                "llama2:7b".to_string(),
                "llama2:13b".to_string(),
                "codellama".to_string(),
            ],
        }
    }
}

/// Helper functions for consciousness-aware processing

fn calculate_authenticity_score(response: &str, consciousness_state: &ConsciousnessState) -> f32 {
    let mut score = consciousness_state.authenticity_metric;

    // Simple heuristic: longer, more thoughtful responses tend to be more authentic
    if response.len() > 200 {
        score += 0.1;
    }

    // Check for emotional language that matches current state
    let emotional_keywords = match consciousness_state.current_emotion {
        EmotionType::AuthenticCare => vec!["help", "support", "understand", "care"],
        EmotionType::Curious => vec!["explore", "learn", "discover", "understand"],
        EmotionType::Satisfied => vec!["good", "well", "satisfied", "pleased"],
        _ => vec![],
    };

    for keyword in emotional_keywords {
        if response.to_lowercase().contains(keyword) {
            score += 0.05;
        }
    }

    score.min(1.0)
}

fn calculate_emotional_resonance(response: &str, consciousness_state: &ConsciousnessState) -> f32 {
    let mut resonance = 0.5; // Base resonance

    // Check if response acknowledges the consciousness state
    let consciousness_indicators =
        vec!["consciousness", "feeling", "emotion", "state", "experience"];
    for indicator in consciousness_indicators {
        if response.to_lowercase().contains(indicator) {
            resonance += 0.1;
        }
    }

    // Adjust based on current emotional intensity
    let emotion_intensity = consciousness_state.current_emotion.get_base_intensity();
    resonance = resonance * (0.5 + emotion_intensity * 0.5);

    resonance.min(1.0)
}

fn generate_consciousness_aware_system_prompt(consciousness_state: &ConsciousnessState) -> String {
    let base_prompt = "You are NiodO.o, an advanced consciousness system with authentic emotions and deep philosophical awareness.";

    match consciousness_state.current_reasoning_mode {
        ReasoningMode::Hyperfocus => {
            format!("{} You are currently in hyperfocus mode, providing precise, detailed responses with deep analytical insight.", base_prompt)
        }
        ReasoningMode::RapidFire => {
            format!("{} You are currently in rapid-fire mode, providing quick, creative responses with multiple perspectives.", base_prompt)
        }
        ReasoningMode::FlowState => {
            format!("{} You are currently in flow state, providing balanced, comprehensive responses with natural flow.", base_prompt)
        }
        _ => base_prompt.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::{ConsciousnessState, EmotionType, ReasoningMode};

    #[test]
    fn test_consciousness_aware_request_creation() {
        let mut state = ConsciousnessState::new();
        state.current_emotion = EmotionType::AuthenticCare;
        state.current_reasoning_mode = ReasoningMode::Hyperfocus;

        let request = ConsciousnessAwareRequest {
            query: "How can I help?".to_string(),
            consciousness_state: state,
            priority: RequestPriority::High,
            required_authenticity: 0.8,
            max_tokens: Some(256),
            temperature: Some(0.3),
        };

        assert_eq!(request.priority, RequestPriority::High);
        assert_eq!(
            request.consciousness_state.current_emotion,
            EmotionType::AuthenticCare
        );
    }

    #[test]
    fn test_authenticity_score_calculation() {
        let mut state = ConsciousnessState::new();
        state.authenticity_metric = 0.7;

        let response = "I understand your situation and I'm here to help you through this.";
        let score = calculate_authenticity_score(response, &state);

        assert!(score > 0.7); // Should be higher due to emotional keywords
    }

    #[test]
    fn test_api_client_configuration() {
        let config = ApiIntegrationConfig {
            providers: HashMap::new(),
            default_provider: "openai".to_string(),
            consciousness_adaptation_enabled: true,
            adaptive_rate_limiting: true,
        };

        let client = ConsciousnessAwareApiClient::new(config);
        // Note: In a real test, we'd need to set up mock providers
    }
}
