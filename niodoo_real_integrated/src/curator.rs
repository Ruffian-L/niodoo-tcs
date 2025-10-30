//! Curator: Memory guardian and knowledge distiller
//! Adapted from curator_executor for niodoo_real_integrated integration
//! Supports both Ollama (CPU) and vLLM (GPU) backends

use anyhow::{Context, Result};
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::time::{Duration, Instant};
use tokio::time::timeout;
use tracing::{error, info, warn};

use crate::config::{CuratorBackend, CuratorConfig};
use crate::data::Experience;

#[derive(Deserialize)]
struct CuratorJson {
    learned: bool,
    refined: String,
    reason: String,
}

/// Curated response with curator judgement
#[derive(Debug, Clone)]
pub struct CuratedResponse {
    pub refined_response: String,
    pub learned: bool,
    pub reason: String,
    pub processing_time_ms: f64,
}

#[derive(Debug, Clone)]
pub struct CuratorRefineResult {
    pub refined: String,
    pub learned: bool,
    pub reason: String,
}

/// The Curator interacts with a lightweight Qwen model via Ollama or vLLM
pub struct Curator {
    client: Client,
    config: CuratorConfig,
    mock_mode: bool,
}

impl Curator {
    pub fn new(config: CuratorConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .no_proxy()
            .build()
            .context("failed to build curator HTTP client")?;

        if config.mock_mode {
            info!("Curator mock mode enabled; responses will bypass backend");
        } else {
            match config.curator_backend {
                CuratorBackend::Vllm => {
                    info!(
                        endpoint = %config.vllm_endpoint,
                        model = %config.model_name,
                        "Curator using vLLM backend (GPU-accelerated)"
                    );
                }
                CuratorBackend::Ollama => {
                    info!(
                        endpoint = %config.ollama_endpoint,
                        model = %config.model_name,
                        "Curator using Ollama backend (CPU)"
                    );
                }
            }
        }

        let mock_mode = config.mock_mode;
        Ok(Self {
            client,
            config,
            mock_mode,
        })
    }

    pub async fn curate(
        &self,
        experience: &Experience,
        knot: f64,
        entropy: f64,
    ) -> Result<CuratedResponse> {
        let start = Instant::now();

        if self.mock_mode {
            return Ok(self.mock_curated(experience, start.elapsed().as_secs_f64() * 1000.0));
        }

        let prompt = format!(
            "As code reviewer, validate/refine for quality>0.8, untangle knot {knot}, balance entropy {entropy}: Response '{response}'. Output JSON: {{\"learned\": true/false, \"refined\": \"text\", \"reason\": \"brief\"}}",
            response = experience.output
        );

        // Use backend-specific refinement
        let refine_result = match self.config.curator_backend {
            CuratorBackend::Vllm => {
                self.refine_with_vllm(&prompt, &experience.output).await?
            }
            CuratorBackend::Ollama => {
                self.refine_with_ollama(&prompt, &experience.output).await?
            }
        };

        Ok(CuratedResponse {
            refined_response: refine_result.refined,
            learned: refine_result.learned,
            reason: refine_result.reason,
            processing_time_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Refine a response using backend (Ollama or vLLM)
    pub async fn refine(
        &self,
        response: &str,
        rouge: f64,
        knot: f64,
        entropy: f64,
    ) -> Result<CuratorRefineResult> {
        let quality = rouge * 0.6 + (1.0 / (knot + 1.0)) * 0.4;

        if quality >= 0.8 {
            info!("Curator skipped: quality={} >= {}", quality, 0.8);
            return Ok(CuratorRefineResult {
                refined: response.to_string(),
                learned: false,
                reason: "quality_ok".to_string(),
            });
        }

        if self.mock_mode {
            return Ok(CuratorRefineResult {
                refined: response.to_string(),
                learned: false,
                reason: "curator mock mode".to_string(),
            });
        }

        info!(
            "Curator Qwen call: quality={}, knot={}, entropy={}",
            quality, knot, entropy
        );

        let prompt = format!(
            "As code reviewer, validate/refine for quality>0.8, untangle knot {}, balance entropy {}: Response '{}'. Output JSON: {{\"learned\": true/false, \"refined\": \"text\", \"reason\": \"brief\"}}",
            knot, entropy, response
        );

        match self.config.curator_backend {
            CuratorBackend::Vllm => {
                self.refine_with_vllm(&prompt, response).await
            }
            CuratorBackend::Ollama => {
                self.refine_with_ollama(&prompt, response).await
            }
        }
    }

    /// Refine using vLLM API (GPU-accelerated, more reliable)
    async fn refine_with_vllm(&self, prompt: &str, fallback_response: &str) -> Result<CuratorRefineResult> {
        let endpoint = if self.config.vllm_endpoint.contains("/v1/chat/completions") {
            self.config.vllm_endpoint.clone()
        } else {
            format!("{}/v1/chat/completions", self.config.vllm_endpoint.trim_end_matches('/'))
        };

        let messages = vec![
            json!({
                "role": "system",
                "content": "You are a code reviewer. Analyze responses and output JSON only: {\"learned\": true/false, \"refined\": \"text\", \"reason\": \"brief\"}"
            }),
            json!({
                "role": "user",
                "content": prompt
            }),
        ];

        let payload = json!({
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": 0.9,
            "max_tokens": self.config.max_tokens,
        });

        let resp = match timeout(
            Duration::from_secs(self.config.timeout_secs),
            self.client.post(&endpoint).json(&payload).send(),
        )
        .await
        {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                warn!(%e, "vLLM curator request failed, returning unmodified response");
                return Ok(CuratorRefineResult {
                    refined: fallback_response.to_string(),
                    learned: false,
                    reason: format!("vllm_request_failed:{e}"),
                });
            }
            Err(_) => {
                warn!("vLLM curator request timed out, returning unmodified response");
                return Ok(CuratorRefineResult {
                    refined: fallback_response.to_string(),
                    learned: false,
                    reason: "vllm_timeout".to_string(),
                });
            }
        };

        if !resp.status().is_success() {
            warn!("vLLM curator request failed with status {}, returning unmodified", resp.status());
            return Ok(CuratorRefineResult {
                refined: fallback_response.to_string(),
                learned: false,
                reason: format!("vllm_status_{}", resp.status().as_u16()),
            });
        }

        let result: serde_json::Value = timeout(
            Duration::from_secs(self.config.timeout_secs),
            resp.json(),
        )
        .await
        .context("vLLM curator JSON parsing timed out")??;

        let response_text = result["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");

        // Try to parse as JSON first, fallback to treating as plain text
        let result = if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(response_text) {
            CuratorRefineResult {
                learned: parsed["learned"].as_bool().unwrap_or(false),
                refined: parsed["refined"].as_str().unwrap_or(fallback_response).to_string(),
                reason: parsed["reason"].as_str().unwrap_or("No reason").to_string(),
            }
        } else {
            // Fallback: treat as plain text refinement
            CuratorRefineResult {
                refined: response_text.to_string(),
                learned: false,
                reason: "plain_text_response".to_string(),
            }
        };

        info!(
            "vLLM curator refined: learned={}, refined len={}, reason={}",
            result.learned,
            result.refined.len(),
            result.reason
        );
        Ok(result)
    }

    /// Refine using Ollama API (CPU, fallback)
    async fn refine_with_ollama(&self, prompt: &str, fallback_response: &str) -> Result<CuratorRefineResult> {
        let ollama_url = format!(
            "{}/api/generate",
            self.config.ollama_endpoint.trim_end_matches('/')
        );
        let body = json!({
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": false,
            "options": {
                "temperature": self.config.temperature,
                "top_p": 0.9
            }
        });

        let resp = match timeout(
            Duration::from_secs(self.config.timeout_secs),
            self.client.post(ollama_url).json(&body).send(),
        )
        .await
        {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                warn!(%e, "Ollama request failed, returning unmodified response");
                return Ok(CuratorRefineResult {
                    refined: fallback_response.to_string(),
                    learned: false,
                    reason: format!("ollama_request_failed:{e}"),
                });
            }
            Err(_) => {
                warn!("Ollama request timed out, returning unmodified response");
                return Ok(CuratorRefineResult {
                    refined: fallback_response.to_string(),
                    learned: false,
                    reason: "ollama_timeout".to_string(),
                });
            }
        };

        if !resp.status().is_success() {
            error!("Ollama down—run 'ollama serve && ollama pull qwen2:0.5b'");
            warn!("Ollama unavailable; returning unmodified response (learned=false)");
            return Ok(CuratorRefineResult {
                refined: fallback_response.to_string(),
                learned: false,
                reason: "ollama_unavailable".to_string(),
            });
        }

        let result: serde_json::Value = timeout(
            Duration::from_secs(self.config.timeout_secs),
            resp.json(),
        )
        .await
        .context("Ollama curator JSON parsing timed out")??;
        
        // Ollama returns {"response": "text here"}
        let response_text = result["response"].as_str().unwrap_or("");

        // Try to parse as JSON first, fallback to treating as plain text
        let result = if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(response_text) {
            CuratorRefineResult {
                learned: parsed["learned"].as_bool().unwrap_or(false),
                refined: parsed["refined"].as_str().unwrap_or(fallback_response).to_string(),
                reason: parsed["reason"].as_str().unwrap_or("No reason").to_string(),
            }
        } else {
            // Fallback: treat as plain text refinement
            CuratorRefineResult {
                refined: response_text.to_string(),
                learned: false,
                reason: "plain_text_response".to_string(),
            }
        };

        info!(
            "Ollama curator refined: learned={}, refined len={}, reason={}",
            result.learned,
            result.refined.len(),
            result.reason
        );
        Ok(result)
    }

    /// Refine a response (simpler version for compatibility)
    pub async fn refine_response(&self, _input: &str, output: &str) -> Result<String> {
        let result = self.refine(output, 0.5, 0.5, 0.5).await?;
        Ok(result.refined)
    }

    fn mock_curated(&self, experience: &Experience, elapsed_ms: f64) -> CuratedResponse {
        CuratedResponse {
            refined_response: experience.output.clone(),
            learned: false,
            reason: "curator mock mode".to_string(),
            processing_time_ms: elapsed_ms,
        }
    }
}
